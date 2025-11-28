# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides containers to represent and operate Delassus operators.

A Delassus operator is a symmetric semi-positive-definite matrix that
represents the apparent inertia within the space defined by the set of
active constraints imposed on a constrained rigid multi-body system.

This module thus provides building-blocks to realize Delassus operators across multiple
worlds contained in a :class:`Model`. The :class:`DelassusOperator` class provides a
high-level interface to encapsulate both the data representation as well as the
relevant operations. It provides methods to allocate the necessary data arrays, build
the Delassus matrix given the current state of the model and the active constraints,
add diagonal regularization, and solve linear systems of the form `D @ x = v` given
arrays holding the right-hand-side (rhs) vectors v. Moreover, it supports the use of
different linear solvers as a back-end for performing the aforementioned linear system
solve. Construction of the Delassus operator is realized using a set of Warp kernels
that parallelize the computation using various strategies.

Typical usage example:
    # Create a model builder and add bodies, joints, geoms, etc.
    builder = ModelBuilder()
    ...

    # Create a model from the builder and construct additional
    # containers to hold joint-limits, contacts, Jacobians
    model = builder.finalize()
    data = model.data()
    limits = Limits(builder)
    contacts = Contacts(builder)
    jacobians = DenseSystemJacobians(model, limits, contacts)

    # Define a linear solver type to use as a back-end for the
    # Delassus operator computations such as factorization and
    # solving the linear system when a rhs vector is provided
    linear_solver = LLTBlockedSolver
    ...

    # Build the Jacobians for the model and active limits and contacts
    jacobians.build(model, data, limits, contacts)
    ...

    # Create a Delassus operator and build it using the current model data
    # and active unilateral constraints (i.e. for limits and contacts).
    delassus = DelassusOperator(model, limits, contacts, linear_solver)
    delassus.build(model, data, jacobians)

    # Add diagonal regularization the Delassus matrix
    eta = ...
    delassus.regularize(eta=eta)

    # Factorize the Delassus matrix using the Cholesky factorization
    delassus.compute()

    # Solve a linear system using the Delassus operator
    rhs = ...
    solution = ...
    delassus.solve(b=rhs, x=solution)
"""

import warp as wp
from warp.context import Devicelike

from ..core.model import Model, ModelData, ModelSize
from ..core.types import float32, int32, mat33f, vec3f
from ..geometry.contacts import Contacts
from ..kinematics.constraints import max_constraints_per_world
from ..kinematics.jacobians import DenseSystemJacobiansData
from ..kinematics.limits import Limits
from ..linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, LinearSolverType

###
# Module interface
###

__all__ = [
    "DelassusOperator",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _build_delassus_elementwise(
    # Inputs:
    model_info_num_bodies: wp.array(dtype=int32),
    model_info_bodies_offset: wp.array(dtype=int32),
    model_bodies_inv_m_i: wp.array(dtype=float32),
    state_bodies_inv_I_i: wp.array(dtype=mat33f),
    jacobians_cts_offset: wp.array(dtype=int32),
    jacobians_cts_data: wp.array(dtype=float32),
    delassus_dim: wp.array(dtype=int32),
    delassus_mio: wp.array(dtype=int32),
    # Outputs:
    delassus_D: wp.array(dtype=float32),
):
    # Retrieve the thread index as the world index and Delassus element index
    wid, tid = wp.tid()

    # Retrieve the world dimensions
    nb = model_info_num_bodies[wid]
    bio = model_info_bodies_offset[wid]

    # Retrieve the problem dimensions
    ncts = delassus_dim[wid]

    # Compute i (row) and j (col) indices from the tid
    i = tid // ncts
    j = tid % ncts

    # Skip if indices exceed the problem size
    if i >= ncts or j >= ncts:
        return

    # Retrieve the world's matrix offsets
    dmio = delassus_mio[wid]
    cjmio = jacobians_cts_offset[wid]

    # Compute the number of body DoFs of the world
    nbd = 6 * nb

    # Buffers
    # tmp = vec3f(0.0)
    Jv_i = vec3f(0.0)
    Jv_j = vec3f(0.0)
    Jw_i = vec3f(0.0)
    Jw_j = vec3f(0.0)
    D_ij = float32(0.0)
    D_ji = float32(0.0)

    # Loop over rigid body blocks
    # NOTE: k is the body index w.r.t the world
    for k in range(nb):
        # Body index (bid) of body k w.r.t the model
        bid_k = bio + k
        # DoF index offset (dio) of body k in the flattened Jacobian matrix
        # NOTE: Equivalent to the column index in the matrix-form of the Jacobian matrix
        dio_k = 6 * k
        # Jacobian index offsets
        jio_ik = cjmio + nbd * i + dio_k
        jio_jk = cjmio + nbd * j + dio_k

        # Load the Jacobian blocks of body k
        for d in range(3):
            # Load the i-th row block
            Jv_i[d] = jacobians_cts_data[jio_ik + d]
            Jw_i[d] = jacobians_cts_data[jio_ik + d + 3]
            # Load the j-th row block
            Jv_j[d] = jacobians_cts_data[jio_jk + d]
            Jw_j[d] = jacobians_cts_data[jio_jk + d + 3]

        # Linear term: m_k * dot(Jv_i, Jv_j)
        lin_ij = model_bodies_inv_m_i[bid_k] * wp.dot(Jv_i, Jv_j)
        lin_ji = model_bodies_inv_m_i[bid_k] * wp.dot(Jv_j, Jv_i)

        # Angular term: dot(Jw_i.T * I_k, Jw_j)
        I_k = state_bodies_inv_I_i[bid_k]
        ang_ij = float32(0.0)
        ang_ji = float32(0.0)
        for r in range(3):  # Loop over rows of A (and elements of v)
            for c in range(r, 3):  # Loop over upper triangular part of A (including diagonal)
                ang_ij += Jw_i[r] * I_k[r, c] * Jw_j[c]
                ang_ji += Jw_j[r] * I_k[r, c] * Jw_i[c]
                if r != c:
                    ang_ij += Jw_i[c] * I_k[r, c] * Jw_j[r]
                    ang_ji += Jw_j[c] * I_k[r, c] * Jw_i[r]

        # Accumulate
        D_ij += lin_ij + ang_ij
        D_ji += lin_ji + ang_ji

    # Store the result in the Delassus matrix
    delassus_D[dmio + ncts * i + j] = 0.5 * (D_ij + D_ji)


@wp.kernel
def _regularize_delassus_diagonal(
    # Inputs:
    delassus_dim: wp.array(dtype=int32),
    delassus_vio: wp.array(dtype=int32),
    delassus_mio: wp.array(dtype=int32),
    eta: wp.array(dtype=float32),
    # Outputs:
    delassus_D: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the problem dimensions and matrix block index offset
    dim = delassus_dim[wid]
    vio = delassus_vio[wid]
    mio = delassus_mio[wid]

    # Skip if row index exceed the problem size
    if tid >= dim:
        return

    # Regularize the diagonal element
    delassus_D[mio + dim * tid + tid] += eta[vio + tid]


###
# Interfaces
###


class DelassusOperator:
    """
    A container to represent the Delassus matrix operator.
    """

    def __init__(
        self,
        model: Model | None = None,
        data: ModelData | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        solver: LinearSolverType = None,
        device: Devicelike = None,
    ):
        """
        Creates a Delassus operator for the given model, limits and contacts containers.

        This class also supports deferred allocation, i.e. it can be initialized without
        a model, limits, or contacts, and the allocation can be performed later using the
        `allocate` method. This is useful for scenarios where the model or constraints are
        not known at the time of Delassus operator creation, but will be available later.

        The dimension of a Delassus matrix is defined as the sum over active
        joint, limit, and contact constraints, and the maximum dimension is
        the maximum number of constraints that can be active in each world.

        Args:
            model (Model): The model container for which the Delassus operator is built.
            data (ModelData, optional): The model data container holding the state info and data.
            limits (Limits, optional): The container holding the allocated joint-limit data.
            contacts (Contacts, optional): The container holding the allocated contacts data.
            device (Devicelike, optional): The device identifier for the Delassus operator. Defaults to None.
            factorizer (CholeskyFactorizer, optional): An optional Cholesky factorization object. Defaults to None.
        """
        # Declare and initialize the host-side cache of the necessary memory allocations
        self._num_worlds: int = 0
        self._model_maxdims: int = 0
        self._model_maxsize: int = 0
        self._world_maxdims: list[int] = []
        self._world_maxsize: list[int] = []
        self._max_of_max_total_D_size: int = 0

        # Cache the requested device
        self._device: Devicelike = device

        # Declare the model size cache
        self._size: ModelSize | None = None

        # Initialize the Delassus data container
        self._operator: DenseLinearOperatorData | None = None

        # Declare the optional Cholesky factorization
        self._solver: LinearSolverType | None = None

        # Allocate the Delassus operator data if at least the model is provided
        if model is not None:
            self.allocate(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                solver=solver,
                device=device,
            )

    @property
    def num_worlds(self) -> int:
        """
        Returns the number of worlds represented by the Delassus operator.
        This is equal to the number of matrix blocks contained in the flat array.
        """
        return self._num_worlds

    @property
    def num_maxdims(self) -> int:
        """
        Returns the maximum dimension of the Delassus matrix across all worlds.
        This is the sum of per matrix block maximum dimensions.
        """
        return self._model_maxdims

    @property
    def num_maxsize(self) -> int:
        """
        Returns the maximum size of the Delassus matrix across all worlds.
        This is the sum over the sizes of all matrix blocks.
        """
        return self._model_maxsize

    @property
    def operator(self) -> DenseLinearOperatorData:
        """
        Returns a reference to the flat Delassus matrix array.
        """
        return self._operator

    @property
    def solver(self) -> LinearSolverType:
        """
        The linear solver object for the Delassus operator.
        This is used to perform the factorization of the Delassus matrix.
        """
        return self._solver

    @property
    def info(self) -> DenseSquareMultiLinearInfo:
        """
        Returns a reference to the flat Delassus matrix array.
        """
        return self._operator.info

    @property
    def D(self) -> wp.array:
        """
        Returns a reference to the flat Delassus matrix array.
        """
        return self._operator.mat

    def allocate(
        self,
        model: Model,
        data: ModelData,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        solver: LinearSolverType = None,
        device: Devicelike = None,
    ):
        """
        Allocates the Delassus operator with the specified dimensions and device.

        Args
        ----
            dims (List[int]): The dimensions of the Delassus matrix for each world.
            device (Devicelike, optional): The device identifier for the Delassus operator. Defaults to None.
            factorizer (CholeskyFactorizer, optional): An optional Cholesky factorization object. Defaults to None.
        """

        # Ensure the model container is valid
        if model is None:
            raise ValueError("A model of type `Model` must be provided to allocate the Delassus operator.")
        elif not isinstance(model, Model):
            raise ValueError("Invalid model provided. Must be an instance of `Model`.")

        # Ensure the data container is valid if provided
        if data is not None:
            if not isinstance(data, ModelData):
                raise ValueError("Invalid data container provided. Must be an instance of `ModelData`.")

        # Ensure the limits container is valid if provided
        if limits is not None:
            if not isinstance(limits, Limits):
                raise ValueError("Invalid limits container provided. Must be an instance of `Limits`.")

        # Ensure the contacts container is valid if provided
        if contacts is not None:
            if not isinstance(contacts, Contacts):
                raise ValueError("Invalid contacts container provided. Must be an instance of `Contacts`.")

        # Capture reference to the model size
        self._size = model.size

        # Extract required maximum number of constraints for each world
        maxdims = max_constraints_per_world(model, limits, contacts)

        # Update the allocation meta-data the specified constraint dimensions
        self._num_worlds = model.size.num_worlds
        self._world_dims = maxdims
        self._world_size = [maxdims[i] * maxdims[i] for i in range(self._num_worlds)]
        self._model_maxdims = sum(self._world_dims)
        self._model_maxsize = sum(self._world_size)
        self._max_of_max_total_D_size = max(self._world_size)

        # Override the device identifier if specified, otherwise use the current device
        if device is not None:
            self._device = device

        # Construct the Delassus operator data structure
        self._operator = DenseLinearOperatorData()
        self._operator.info = DenseSquareMultiLinearInfo()
        self._operator.mat = wp.zeros(shape=(self._model_maxsize,), dtype=float32, device=self._device)
        if (model.info is not None) and (data.info is not None):
            mat_offsets = [0] + [sum(self._world_size[:i]) for i in range(1, self._num_worlds + 1)]
            self._operator.info.assign(
                maxdim=model.info.max_total_cts,
                dim=data.info.num_total_cts,
                vio=model.info.total_cts_offset,
                mio=wp.array(mat_offsets[: self._num_worlds], dtype=int32, device=self._device),
                dtype=float32,
                device=self._device,
            )
        else:
            self._operator.info.allocate(dimensions=maxdims, dtype=float32, itype=int32, device=self._device)

        # Optionally initialize the linear system solver if one is specified
        if solver is not None:
            if not issubclass(solver, LinearSolverType):
                raise ValueError("Invalid solver provided. Must be a subclass of `LinearSolverType`.")
            self._solver = solver(operator=self._operator, device=self._device)

    def zero(self):
        """
        Sets all values of the Delassus matrix to zero.
        This is useful for resetting the operator before recomputing it.
        """
        self._operator.mat.zero_()

    def build(self, model: Model, data: ModelData, jacobians: DenseSystemJacobiansData, reset_to_zero: bool = True):
        """
        Builds the Delassus matrix using the provided Model, ModelData, and constraint Jacobians.

        Args:
            model (Model): The model for which the Delassus operator is built.
            data (ModelData): The current data of the model.
            reset_to_zero (bool, optional): If True, resets the Delassus matrix to zero before building. Defaults to True.

        Raises:
            ValueError: If the model, data, or Jacobians are not valid.
            ValueError: If the Delassus matrix is not allocated.
        """
        # Ensure the model is valid
        if model is None or not isinstance(model, Model):
            raise ValueError("A valid model of type `Model` must be provided to build the Delassus operator.")

        # Ensure the data is valid
        if data is None or not isinstance(data, ModelData):
            raise ValueError("A valid model data of type `ModelData` must be provided to build the Delassus operator.")

        # Ensure the Jacobians are valid
        if jacobians is None or not isinstance(jacobians, DenseSystemJacobiansData):
            raise ValueError(
                "A valid Jacobians data of type `DenseSystemJacobiansData` must be provided to build the Delassus operator."
            )

        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call allocate() first.")

        # Initialize the Delassus matrix to zero
        if reset_to_zero:
            self.zero()

        # Build the Delassus matrix parallelized element-wise
        wp.launch(
            kernel=_build_delassus_elementwise,
            dim=(self._size.num_worlds, self._max_of_max_total_D_size),
            inputs=[
                # Inputs:
                model.info.num_bodies,
                model.info.bodies_offset,
                model.bodies.inv_m_i,
                data.bodies.inv_I_i,
                jacobians.J_cts_offsets,
                jacobians.J_cts_data,
                self._operator.info.dim,
                self._operator.info.mio,
                # Outputs:
                self._operator.mat,
            ],
        )

    def regularize(self, eta: wp.array):
        """
        Adds diagonal regularization to each matrix block of the Delassus operator.

        Args:
            eta (wp.array): The regularization values to add to the diagonal of each matrix block.
            This should be an array of shape `(maxdims,)` and type :class:`float32`.
            Each value in `eta` corresponds to the regularization along each constraint.
        """
        wp.launch(
            kernel=_regularize_delassus_diagonal,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[self._operator.info.dim, self._operator.info.vio, self._operator.info.mio, eta, self._operator.mat],
        )

    def compute(self, reset_to_zero: bool = True):
        """
        Factorizes the Delassus matrix using the Cholesky factorization.\n
        Returns True if the factorization was successful, False otherwise.

        Args:
            reset_to_zero (bool): If True, resets the Delassus matrix to zero before factorizing.
            This is useful for ensuring that the matrix is in a clean state before factorization.
        """
        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call allocate() first.")

        # Ensure the solver is available if pre-computation is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Optionally initialize the factorization matrix before factorizing
        if reset_to_zero:
            self._solver.reset()

        # Perform the Cholesky factorization
        self._solver.compute(A=self._operator.mat)

    def solve(self, v: wp.array, x: wp.array):
        """
        Solves the linear system D * x = v using the Cholesky factorization.

        Args:
            v (wp.array): The right-hand side vector of the linear system.
            x (wp.array): The array to hold the solution.

        Raises:
            ValueError: If the Delassus matrix is not allocated or the factorizer is not available.
            ValueError: If a factorizer has not been configured set.
        """
        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call allocate() first.")

        # Ensure the solver is available if solving is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Solve the linear system using the factorized matrix
        return self._solver.solve(b=v, x=x)

    def solve_inplace(self, x: wp.array):
        """
        Solves the linear system D * x = v in-place.\n
        This modifies the input array x to contain the solution assuming it is initialized as x=v.

        Args:
            x (wp.array): The array to hold the solution. It should be initialized with the right-hand side vector v.

        Raises:
            ValueError: If the Delassus matrix is not allocated or the factorizer is not available.
            ValueError: If a factorizer has not been configured set.
        """
        # Ensure the Delassus matrix is allocated
        if self._operator.mat is None:
            raise ValueError("Delassus matrix is not allocated. Call allocate() first.")

        # Ensure the solvers is available if solving in-place is requested
        if self._solver is None:
            raise ValueError("A linear system solver is not available. Allocate with solver=LINEAR_SOLVER_TYPE.")

        # Solve the linear system in-place
        return self._solver.solve_inplace(x=x)
