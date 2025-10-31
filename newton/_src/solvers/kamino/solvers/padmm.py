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
Proximal-ADMM Solver

Provides a forward dynamics solver for constrained rigid multi-body systems using
the Alternating Direction Method of Multipliers (ADMM). This implementation realizes
the Proximal-ADMM algorithm described in [1] and is based on the work of J. Carpentier
et al in [2]. It solves the Lagrange dual of the constrained forward dynamics problem
in constraint reactions (i.e. impulses) and post-event constraint-space velocities. The
diagonal preconditioner strategy described in [3] is also implemented to improve
numerical conditioning. This version also incorporates Nesterov-style gradient
acceleration with adaptive restarts based on the work of O'Donoghue and Candes in [4].

Notes
----
- ADMM is based on the Augmented Lagrangian Method (ALM) for dealing with set-inclusion
  (i.e. inequality) constraints, but introduces an alternating primal-dual descent/ascent scheme.
- Proximal-ADMM introduces an additional proximal regularization term to the optimization objective.
- Uses (optional) Nesterov-style gradient acceleration with adaptive restarts.
- Uses (optional) adaptive penalty updates based on primal-dual residual balancing.
- Uses (optional) over-relaxation updates (not applicable to accelerated version).

References
----
- [1] https://arxiv.org/abs/2504.19771
- [2] https://arxiv.org/pdf/2405.17020
- [3] https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6693
- [4] https://epubs.siam.org/doi/abs/10.1137/120896219

Usage
----
A typical example for using this module is:

    # Import all relevant types from Kamino
    from newton._src.solvers.kamino.core import ModelBuilder
    from newton._src.solvers.kamino.geometry import Contacts
    from newton._src.solvers.kamino.kinematics import Limits
    from newton._src.solvers.kamino.kinematics import DenseSystemJacobians
    from newton._src.solvers.kamino.dynamics import DualProblem
    from newton._src.solvers.kamino.solvers import PADMMSolver

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

    # Build the Jacobians for the model and active limits and contacts
    jacobians.build(model, data, limits, contacts)
    ...

    # Create a forward-dynamics DualProblem to be solved
    dual = DualProblem(model, limits, contacts)
    dual.build(model, data, limits, contacts, jacobians)

    # Create a forward-dynamics PADMM solver
    solver = PADMMSolver(model, limits, contacts)

    # Solve the dual forward dynamics problem
    solver.solve(problem=dual)

    # Extract the resulting constraint reactions
    multipliers = solver.solution.lambdas
"""

from dataclasses import dataclass
from enum import IntEnum

import warp as wp
from warp.context import Devicelike

from ..core.math import FLOAT32_EPS, FLOAT32_MAX
from ..core.model import Model, ModelSize
from ..core.types import float32, int32, vec3f
from ..dynamics.dual import DualProblem
from ..geometry.contacts import Contacts
from ..kinematics.limits import Limits

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class PADMMPenaltyUpdate(IntEnum):
    """
    An enumeration of the penalty update methods used in PADMM.
    """

    FIXED = 0
    """
    Fixed penalty:
    `rho` is initialized to `settings.rho_0`, remaining constant over the solve.
    """

    # TODO: Implement adaptive penalty updates
    # LINEAR = 1
    # """
    # Linear penalty update:
    # `rho` is increased by a fixed factor.
    # """
    # BALANCED = 1
    # """
    # Balanced-residuals penalty update:
    # `rho` is increased in order for the ratio of primal/dual residuals to be close to unity.
    # """
    # SPECTRAL = 2
    # """
    # Spectral penalty update:
    # `rho` is increased by the spectral radius of the Delassus matrix.
    # """


@wp.struct
class PADMMConfig:
    """
    A warp struct to hold PADMM per-world solver configurations on the target device.

    Intended to be used as ``dtype`` for warp arrays.

    Attributes:
        primal_tolerance (float32): The target tolerance on the total primal residual `r_primal`.\n
            Must be greater than zero. Defaults to `1e-6`.
        dual_tolerance (float32): The target tolerance on the total dual residual `r_dual`.\n
            Must be greater than zero. Defaults to `1e-6`.
        compl_tolerance (float32): The target tolerance on the total complementarity residual `r_compl`.\n
            Must be greater than zero. Defaults to `1e-6`.
        restart_tolerance (float32): The tolerance on the total combined primal-dual
            residual `r_comb`, for determining when gradient acceleration should be restarted.\n
            Must be greater than zero. Defaults to `0.999`.
        eta (float32): The proximal regularization parameter.\n
            Must be greater than zero. Defaults to `1e-5`.
        rho_0 (float32): The initial value of the penalty parameter.\n
            Must be greater than zero. Defaults to `1.0`.
        a_0 (float32): The initial value of the acceleration parameter.\n
            Must be greater than zero. Defaults to `1.0`.
        omega (float32): The over-relaxation factor.\n
            Must be in the range `[0.0, 2.0]`. Defaults to `1.0`.
        alpha (float32): The threshold on primal-dual residual ratios,
            used to determine when penalty updates should occur.\n
            Must be greater than `1.0`. Defaults to `10.0`.
        tau (float32): The factor by which the penalty is increased/decreased
            when the primal-dual residual ratios exceed the threshold `alpha`.\n
            Must be greater than `1.0`. Defaults to `1.5`.
        max_iterations (int32): The maximum number of solver iterations.\n
            Must be greater than zero. Defaults to `200`.
        penalty_update_freq (int32): The permitted frequency of penalty updates.\n
            If zero, no updates are performed. Otherwise, updates are performed every
            `penalty_update_freq` iterations. Defaults to `10`.
        penalty_update_method (int32): The penalty update method used to adapt the penalty parameter.\n
            Defaults to `PADMMPenaltyUpdate.FIXED`.\n
            See :class:`PADMMPenaltyUpdate` for details.
    """

    primal_tolerance: float32
    """
    The target tolerance on the total primal residual `r_primal`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    dual_tolerance: float32
    """
    The target tolerance on the total dual residual `r_dual`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    compl_tolerance: float32
    """
    The target tolerance on the total complementarity residual `r_compl`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    restart_tolerance: float32
    """
    The tolerance applied on the total combined primal-dual residual `r_comb`,
    for determining when gradient acceleration should be restarted.\n
    Must be greater than zero. Defaults to `0.999`.
    """

    eta: float32
    """
    The proximal regularization parameter.\n
    Must be greater than zero. Defaults to `1e-5`.
    """

    rho_0: float32
    """
    The initial value of the ALM penalty parameter.\n
    Must be greater than zero. Defaults to `1.0`.
    """

    rho_min: float32
    """
    The lower-bound applied to the ALM penalty parameter.\n
    Must be greater than zero. Defaults to `1e-5`.
    """

    a_0: float32
    """
    The initial value of the acceleration parameter.\n
    Must be greater than zero. Defaults to `1.0`.
    """

    omega: float32
    """
    The over-relaxation factor.\n
    Must be in the range `[0.0, 2.0]`. Defaults to `1.0`.
    """

    alpha: float32
    """
    The threshold on primal-dual residual ratios,
    used to determine when penalty updates should occur.\n
    Must be greater than `1.0`. Defaults to `10.0`.
    """

    tau: float32
    """
    The factor by which the penalty is increased/decreased
    when the primal-dual residual ratios exceed the threshold `alpha`.\n
    Must be greater than `1.0`. Defaults to `1.5`.
    """

    max_iterations: int32
    """
    The maximum number of solver iterations.\n
    Must be greater than zero. Defaults to `200`.
    """

    penalty_update_freq: int32
    """
    The permitted frequency of penalty updates.\n
    If zero, no updates are performed. Otherwise, updates are performed every
    `penalty_update_freq` iterations. Defaults to `10`.
    """

    penalty_update_method: int32
    """
    The penalty update method used to adapt the penalty parameter.\n
    Defaults to `PADMMPenaltyUpdate.FIXED`.\n
    See :class:`PADMMPenaltyUpdate` for details.
    """


@wp.struct
class PADMMStatus:
    """
    A warp struct to hold the PADMM per-world solver status on the target device.

    Intended to be used as ``dtype`` for warp arrays.

    Attributes:
        converged (int32): A flag indicating whether the solver has converged (`1`) or not (`0`).\n
            Used internally to keep track of per-world convergence status,
            with `1` being set only when all total residuals have satisfied their
            respective tolerances. If by the end of the solve the flag is still `0`,
            it indicates that the solve reached the maximum number of iterations.
        iterations (int32): The number of iterations performed by the solver.\n
            Used internally to keep track of per-world iteration counts.
        r_p (float32): The total primal residual.\n
            Computed using the L-inf norm as `r_primal := || x - y ||_inf`.\n
        r_d (float32): The total dual residual.\n
            Computed using the L-inf norm as `r_dual := || eta * (x - x_p) + rho * (y - y_p) ||_inf`.\n
        r_c (float32): The total complementarity residual.
            Computed using the L-inf norm as `r_compl := || [x_k.T @ z_k] ||_inf`,
            with `k` indexing each unilateral constraint set, i.e. 1D limits and 3D contacts.
        r_dx (float32): The total primal iterate residual.\n
            Computed as the L2-norm `r_dx := || x - x_p ||_2`.
        r_dy (float32): The total slack iterate residual.\n
            Computed as the L2-norm `r_dy := || y - y_p ||_2`.
        r_dz (float32): The total dual iterate residual.\n
            Computed as the L2-norm `r_dz := || z - z_p ||_2`.
        r_a (float32): The total combined primal-dual residual used for acceleration restart checks.
            Computed as `r_a := rho * r_dy + (1.0 / rho) * r_dz`.
        r_a_p (float32): The previous total combined primal-dual residual.
        r_a_pp (float32): An auxiliary cache of the previous total combined primal-dual residual.
        restart (int32): A flag indicating whether gradient acceleration requires a restart (`1`) or not (`0`).\n
            Used internally to keep track of per-world acceleration restarts.
        num_restarts (int32): The number of acceleration restarts performed during the solve.
    """

    converged: int32
    """
    A flag indicating whether the solver has converged (`1`) or not (`0`).\n
    Used internally to keep track of per-world convergence status,
    with `1` being set only when all total residuals have satisfied their
    respective tolerances. If by the end of the solve the flag is still `0`,
    it indicates that the solve reached the maximum number of iterations.
    """

    iterations: int32
    """
    The number of iterations performed by the solver.\n
    Used internally to keep track of per-world iteration counts.
    """

    r_p: float32
    """
    The total primal residual.\n
    Computed using the L-inf norm as `r_primal := || x - y ||_inf`.\n
    """

    r_d: float32
    """
    The total dual residual.\n
    Computed using the L-inf norm as `r_dual := || eta * (x - x_p) + rho * (y - y_p) ||_inf`.\n
    """

    r_c: float32
    """
    The total complementarity residual.
    Computed using the L-inf norm as `r_compl := || [x_k.T @ z_k] ||_inf`,
    with `k` indexing each unilateral constraint set, i.e. 1D limits and 3D contacts.
    """

    r_dx: float32
    """
    The total primal iterate residual.\n
    Computed as the L2-norm `r_dx := || x - x_p ||_2`.
    """

    r_dy: float32
    """
    The total slack iterate residual.\n
    Computed as the L2-norm `r_dy := || y - y_p ||_2`.
    """

    r_dz: float32
    """
    The total dual iterate residual.\n
    Computed as the L2-norm `r_dz := || z - z_p ||_2`.
    """

    r_a: float32
    """
    The total combined primal-dual residual used for acceleration restart checks.\n
    Computed as `r_a := rho * r_dy + (1.0 / rho) * r_dz`.
    """

    r_a_p: float32
    """The previous total combined primal-dual residual."""

    r_a_pp: float32
    """An auxiliary cache of the previous total combined primal-dual residual."""

    restart: int32
    """
    A flag indicating whether gradient acceleration requires a restart (`1`) or not (`0`).\n
    Used internally to keep track of per-world acceleration restarts.
    """

    num_restarts: int32
    """The number of acceleration restarts performed during the solve."""


@wp.struct
class PADMMPenalty:
    """
    A warp struct to hold the on-device PADMM solver penalty state.

    Intended to be used as ``dtype`` for warp arrays.

    Attributes:
        num_updates (int32): The number of penalty updates performed during a solve.\n
            If a direct linear-system solver is used, this also
            equals the number of matrix factorizations performed.
        rho (float32): The current value of the ALM penalty parameter.\n
            If adaptive penalty scheme is used, this value may change during
            solve operations, while being lower-bounded by `settings.rho_min`
            to ensure numerical stability.
        rho_p (float32): The previous value of the ALM penalty parameter.\n
            As diagonal regularization of the lhs matrix (e.g. Delassus
            operator) is performed in-place, we must keep track of the
            previous penalty value  to remove the previous regularization
            before applying the current penalty value.
    """

    num_updates: int32
    """
    The number of penalty updates performed during a solve.\n
    If a direct linear-system solver is used, this also
    equals the number of matrix factorizations performed.
    """

    rho: float32
    """
    The current value of the ALM penalty parameter.\n
    If adaptive penalty scheme is used, this value may change during
    solve operations, while being lower-bounded by `settings.rho_min`
    to ensure numerical stability.
    """

    rho_p: float32
    """
    The previous value of the ALM penalty parameter.\n
    As diagonal regularization of the lhs matrix (e.g. Delassus
    operator) is performed in-place, we must keep track of the
    previous penalty value  to remove the previous regularization
    before applying the current penalty value.
    """


###
# Containers
###


@dataclass
class PADMMSettings:
    """
    A container to bundle PADMM solver settings.

    Attributes:
        primal_tolerance (float): The target tolerance on the total primal residual `r_primal`.\n
            Must be greater than zero. Defaults to `1e-6`.
        dual_tolerance (float): The target tolerance on the total dual residual `r_dual`.\n
            Must be greater than zero. Defaults to `1e-6`.
        compl_tolerance (float): The target tolerance on the total complementarity residual `r_compl`.\n
            Must be greater than zero. Defaults to `1e-6`.
        restart_tolerance (float): The tolerance on the total combined primal-dual
            residual `r_comb`, for determining when gradient acceleration should be restarted.\n
            Must be greater than zero. Defaults to `0.999`.
        eta (float): The proximal regularization parameter.\n
            Must be greater than zero. Defaults to `1e-5`.
        rho_0 (float): The initial value of the ALM penalty parameter.\n
            Must be greater than zero. Defaults to `1.0`.
        rho_min (float): The lower-bound applied to the ALM penalty parameter.\n
            Used to ensure numerical stability when adaptive penalty updates are used.\n
            Must be greater than zero. Defaults to `1e-5`.
        a_0 (float): The initial value of the acceleration parameter.\n
            Must be greater than zero. Defaults to `1.0`.
        omega (float): The over-relaxation factor.\n
            Must be in the range `[0.0, 2.0]`. Defaults to `1.0`.
        alpha (float): The threshold on primal-dual residual ratios,
            used to determine when penalty updates should occur.\n
            Must be greater than `1.0`. Defaults to `10.0`.
        tau (float): The factor by which the penalty is increased/decreased
            when the primal-dual residual ratios exceed the threshold `alpha`.\n
            Must be greater than `1.0`. Defaults to `1.5`.
        max_iterations (int): The maximum number of solver iterations.\n
            Must be greater than zero. Defaults to `200`.
        penalty_update_freq (int): The permitted frequency of penalty updates.\n
            If zero, no updates are performed. Otherwise, updates are performed every
            `penalty_update_freq` iterations. Defaults to `10`.
        penalty_update_method (PADMMPenaltyUpdate): The penalty update method used to adapt the penalty parameter.\n
            Defaults to `PADMMPenaltyUpdate.FIXED`. See :class:`PADMMPenaltyUpdate` for details.

    """

    primal_tolerance: float = 1e-6
    """
    The target tolerance on the total primal residual `r_primal`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    dual_tolerance: float = 1e-6
    """
    The target tolerance on the total dual residual `r_dual`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    compl_tolerance: float = 1e-6
    """
    The target tolerance on the total complementarity residual `r_compl`.\n
    Must be greater than zero. Defaults to `1e-6`.
    """

    restart_tolerance: float = 0.999
    """
    The tolerance on the total combined primal-dual residual `r_comb`,
    for determining when gradient acceleration should be restarted.\n
    Must be greater than zero. Defaults to `0.999`.
    """

    eta: float = 1e-5
    """
    The proximal regularization parameter.\n
    Must be greater than zero. Defaults to `1e-5`.
    """

    rho_0: float = 1.0
    """
    The initial value of the ALM penalty parameter.\n
    Must be greater than zero. Defaults to `1.0`.
    """

    rho_min: float = 1e-5
    """
    The lower-bound applied to the ALM penalty parameter.\n
    Used to ensure numerical stability when adaptive penalty updates are used.\n
    Must be greater than zero. Defaults to `1e-5`.
    """

    a_0: float = 1.0
    """
    The initial value of the acceleration parameter.\n
    Must be greater than zero. Defaults to `1.0`.
    """

    omega: float = 1.0
    """
    The over-relaxation factor.\n
    Must be in the range [0.0, 2.0]. Defaults to `1.0`.
    """

    alpha: float = 10.0
    """
    The primal-dual residual threshold used to determine when penalty updates are needed.
    Must be greater than one. Defaults to `10.0`.
    """

    tau: float = 1.5
    """
    The factor by which the ALM penalty is increased/decreased when
    the primal-dual residual ratios exceed the threshold `alpha`.\n
    Must be greater than `1.0`. Defaults to `1.5`.
    """

    max_iterations: int = 200
    """
    The maximum number of solver iterations.\n
    Must be greater than zero. Defaults to `200`.
    """

    penalty_update_freq: int = 1
    """
    The permitted frequency of penalty updates.\n
    If zero, no updates are performed. Otherwise, updates are performed every
    `penalty_update_freq` iterations. Defaults to `10`.
    """

    penalty_update_method: PADMMPenaltyUpdate = PADMMPenaltyUpdate.FIXED
    """
    The penalty update method used to adapt the penalty parameter.\n
    Defaults to `PADMMPenaltyUpdate.FIXED`. See :class:`PADMMPenaltyUpdate` for details.
    """

    def check(self):
        """
        Checks the validity of PADMM solver settings.
        """
        if self.primal_tolerance < 0.0:
            raise ValueError(f"Invalid primal tolerance: {self.primal_tolerance}. Must be non-negative.")
        if self.dual_tolerance < 0.0:
            raise ValueError(f"Invalid dual tolerance: {self.dual_tolerance}. Must be non-negative.")
        if self.compl_tolerance < 0.0:
            raise ValueError(f"Invalid complementarity tolerance: {self.compl_tolerance}. Must be non-negative.")
        if not (0.0 <= self.restart_tolerance < 1.0):
            raise ValueError(f"Invalid restart tolerance: {self.restart_tolerance}. Must be in the range [0.0, 1.0).")
        if self.eta <= 0.0:
            raise ValueError(f"Invalid proximal parameter: {self.eta}. Must be greater than zero.")
        if self.rho_0 <= 0.0:
            raise ValueError(f"Invalid initial ALM penalty: {self.rho_0}. Must be greater than zero.")
        if self.rho_min <= 0.0:
            raise ValueError(f"Invalid minimum ALM penalty: {self.rho_min}. Must be greater than zero.")
        if self.a_0 <= 0.0:
            raise ValueError(f"Invalid initial acceleration parameter: {self.a_0}. Must be greater than zero.")
        if not (0.0 <= self.omega <= 2.0):
            raise ValueError(f"Invalid over-relaxation factor: {self.omega}. Must be in the range [0.0, 2.0].")
        if self.alpha <= 1.0:
            raise ValueError(f"Invalid penalty threshold: {self.alpha}. Must be greater than one.")
        if self.tau <= 1.0:
            raise ValueError(f"Invalid penalty increment factor: {self.tau}. Must be greater than one.")
        if self.max_iterations <= 0:
            raise ValueError(f"Invalid maximum iterations: {self.max_iterations}. Must be a positive integer.")
        if self.penalty_update_freq < 0:
            raise ValueError(f"Invalid penalty update frequency: {self.penalty_update_freq}. Must be non-negative.")
        if not isinstance(self.penalty_update_method, PADMMPenaltyUpdate):
            raise TypeError(
                f"Invalid penalty update method: {self.penalty_update_method}. "
                "Must be an instance of PADMMPenaltyUpdate."
            )

    def to_config(self) -> PADMMConfig:
        """
        Converts the host-side settings to the corresponding device-side object.

        Returns:
            PADMMConfig: The solver settings as a warp struct.
        """
        config = PADMMConfig()
        config.primal_tolerance = self.primal_tolerance
        config.dual_tolerance = self.dual_tolerance
        config.compl_tolerance = self.compl_tolerance
        config.restart_tolerance = self.restart_tolerance
        config.eta = self.eta
        config.rho_0 = self.rho_0
        config.rho_min = self.rho_min
        config.a_0 = self.a_0
        config.omega = self.omega
        config.alpha = self.alpha
        config.tau = self.tau
        config.max_iterations = self.max_iterations
        config.penalty_update_freq = self.penalty_update_freq
        config.penalty_update_method = self.penalty_update_method
        return config


class PADMMState:
    """
    A data container to bundle the internal PADMM state arrays.

    Attributes:
        done (wp.array): A single-element array containing the global
            flag that indicates whether the solver should terminate.\n
            Its value is initialized to ``num_worlds`` at the beginning of each
            solve, and decremented by one for each world that has converged.\n
            Shape of ``(1,)`` and type :class:`int`.
        s (wp.array): The De Saxce correction velocities `s = Gamma(v_plus)`.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        v (wp.array): The total bias velocity vector serving as the right-hand-side of the PADMM linear system.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        x (wp.array): The current PADMM primal variables.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        x_p (wp.array): The previous PADMM primal variables.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        y (wp.array): The current PADMM slack variables.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        y_p (wp.array): The previous PADMM slack variables.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        z (wp.array): The current PADMM dual variables.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        z_p (wp.array): The previous PADMM dual variables.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        y_hat (wp.array): The auxiliary PADMM slack variables used with gradient acceleration.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        z_hat (wp.array): The auxiliary PADMM dual variables used with gradient acceleration.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        a (wp.array): The current PADMM acceleration variables.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        a_p (wp.array): The previous PADMM acceleration variables.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
    """

    def __init__(self, size: ModelSize | None = None):
        """
        Initializes the PADMM solver state container.

        If a model size is provided, allocates the state arrays accordingly.

        Args:
            size (ModelSize | None): The model-size utility container holding the dimensionality of the model.
        """

        self.done: wp.array(dtype=int32) | None = None
        """
        A single-element array containing the global flag that indicates whether the solver should terminate.\n
        Its value is initialized to ``num_worlds`` at the beginning of each
        solve, and decremented by one for each world that has converged.\n
        Shape of ``(1,)`` and type :class:`int`.
        """

        self.s: wp.array(dtype=float32) | None = None
        """
        The De Saxce correction velocities `s = Gamma(v_plus)`.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.v: wp.array(dtype=float32) | None = None
        """
        The total bias velocity vector serving as the right-hand-side of the PADMM linear system.\n
        It is computed from the PADMM state and proximal parameters `eta` and `rho`.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.x: wp.array(dtype=float32) | None = None
        """
        The current PADMM primal variables.
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.x_p: wp.array(dtype=float32) | None = None
        """
        The previous PADMM primal variables.
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.y: wp.array(dtype=float32) | None = None
        """
        The current PADMM slack variables.
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.y_p: wp.array(dtype=float32) | None = None
        """
        The previous PADMM slack variables.
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.z: wp.array(dtype=float32) | None = None
        """
        The current PADMM dual variables.
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.z_p: wp.array(dtype=float32) | None = None
        """
        The previous PADMM dual variables.
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.y_hat: wp.array(dtype=float32) | None = None
        """
        The auxiliary PADMM slack variables used with gradient acceleration.
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.z_hat: wp.array(dtype=float32) | None = None
        """
        The auxiliary PADMM dual variables used with gradient acceleration.
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.a: wp.array(dtype=float32) | None = None
        """
        The current PADMM acceleration variables.
        Shape of ``(num_worlds,)`` and type :class:`float32`.
        """

        self.a_p: wp.array(dtype=float32) | None = None
        """
        The previous PADMM acceleration variables.
        Shape of ``(num_worlds,)`` and type :class:`float32`.
        """

        # Perform memory allocations if model size is specified
        if size is not None:
            self.allocate(size)

    def allocate(self, size: ModelSize):
        """
        Allocates the PADMM solver state arrays based on the model size.

        Args:
            size (ModelSize): The model-size utility container holding the dimensionality of the model.
        """
        # Allocate per-world solver done flags
        self.done = wp.zeros(1, dtype=int32)

        # Allocate primary state variables
        self.s = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.v = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.x = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.x_p = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.y = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.y_p = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.z = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.z_p = wp.zeros(size.sum_of_max_total_cts, dtype=float32)

        # Allocate auxiliary state variables used with acceleration
        self.y_hat = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.z_hat = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.a = wp.zeros(size.num_worlds, dtype=float32)
        self.a_p = wp.zeros(size.num_worlds, dtype=float32)

    def reset(self):
        """
        Resets all PADMM state arrays.

        Specifically:
        - PADMM state arrays (primal, slack, dual, rhs, De Saxce correction) are set to zeros.
        - If acceleration is enabled, the momentum arrays `a_p` and `a` are set to ones.
        """
        # Reset primary state variables
        self.s.zero_()
        self.v.zero_()
        self.x.zero_()
        self.x_p.zero_()
        self.y.zero_()
        self.y_p.zero_()
        self.z.zero_()
        self.z_p.zero_()

        # Reset auxiliary state variables
        self.y_hat.zero_()
        self.z_hat.zero_()

        # Reset acceleration scale variables
        self.a.fill_(1.0)
        self.a_p.fill_(1.0)


class PADMMResiduals:
    """
    A data container to bundle the internal PADMM residual arrays.

    Attributes:
        r_primal (wp.array): The PADMM primal residual vector, computed as `r_primal := x - y`.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        r_dual (wp.array): The PADMM dual residual vector, computed as `r_dual := eta * (x - x_p) + rho * (y - y_p)`.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        r_compl (wp.array): The PADMM complementarity residual vector, computed as `r_compl := [x_j.dot(z_j)]`,\n
            where `j` indexes each unilateral constraint set (i.e. 1D limits and 3D contacts).\n
            Shape of ``(sum_of_num_unilateral_cts,)`` and type :class:`float32`.
        r_dx (wp.array): The PADMM primal iterate residual vector, computed as `r_dx := x - x_p`.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        r_dy (wp.array): The PADMM slack iterate residual vector, computed as `r_dy := y - y_p`.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        r_dz (wp.array): The PADMM dual iterate residual vector, computed as `r_dz := z - z_p`.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
    """

    def __init__(self, size: ModelSize | None = None):
        """
        Initializes the PADMM residuals container.

        If a model size is provided, allocates the residuals arrays accordingly.

        Args:
            size (ModelSize | None): The model-size utility container holding the dimensionality of the model.
        """

        self.r_primal: wp.array(dtype=float32) | None = None
        """
        The PADMM primal residual vector, computed as `r_primal := x - y`.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.r_dual: wp.array(dtype=float32) | None = None
        """
        The PADMM dual residual vector, computed as `r_dual := eta * (x - x_p) + rho * (y - y_p)`.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.r_compl: wp.array(dtype=float32) | None = None
        """
        The PADMM complementarity residual vector, computed as `r_compl := [x_j.dot(z_j)]`,\n
        where `j` indexes each unilateral constraint set (i.e. 1D limits and 3D contacts).\n
        Shape of ``(sum_of_num_unilateral_cts,)`` and type :class:`float32`.
        """

        self.r_dx: wp.array(dtype=float32) | None = None
        """
        The PADMM primal iterate residual vector, computed as `r_dx := x - x_p`.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.r_dy: wp.array(dtype=float32) | None = None
        """
        The PADMM slack iterate residual vector, computed as `r_dy := y - y_p`.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.r_dz: wp.array(dtype=float32) | None = None
        """
        The PADMM dual iterate residual vector, computed as `r_dz := z - z_p`.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        # Perform memory allocations if model size is specified
        if size is not None:
            self.allocate(size)

    def allocate(self, size: ModelSize):
        """
        Allocates the residuals arrays based on the model size.

        Args:
            size (ModelSize): The model-size utility container holding the dimensionality of the model.
        """
        self.r_primal = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.r_dual = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.r_compl = wp.zeros(size.sum_of_max_unilaterals, dtype=float32)
        self.r_dx = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.r_dy = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.r_dz = wp.zeros(size.sum_of_max_total_cts, dtype=float32)

    def zero(self):
        """
        Resets all PADMM residual arrays to zeros.
        """
        self.r_primal.zero_()
        self.r_dual.zero_()
        self.r_compl.zero_()
        self.r_dx.zero_()
        self.r_dy.zero_()
        self.r_dz.zero_()


class PADMMSolution:
    """
    An interface container to the PADMM solver solution arrays.

    Attributes:
        lambdas (wp.array): The constraint reactions (i.e. impulses) solution array.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        v_plus (wp.array): The post-event constraint-space velocities solution array.
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
    """

    def __init__(self, size: ModelSize | None = None):
        """
        Initializes the PADMM solution container.

        If a model size is provided, allocates the solution arrays accordingly.

        Args:
            size (ModelSize | None): The model-size utility container holding the dimensionality of the model.
        """

        self.lambdas: wp.array(dtype=float32) | None = None
        """
        The constraint reactions (i.e. impulses) solution array.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.v_plus: wp.array(dtype=float32) | None = None
        """
        The post-event constraint-space velocities solution array.
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        # Perform memory allocations if model size is specified
        if size is not None:
            self.allocate(size)

    def allocate(self, size: ModelSize):
        """
        Allocates the PADMM solution arrays based on the model size.

        Args:
            size (ModelSize): The model-size utility container holding the dimensionality of the model.
        """
        self.lambdas = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.v_plus = wp.zeros(size.sum_of_max_total_cts, dtype=float32)

    def zero(self):
        """
        Resets all PADMM solution arrays to zeros.
        """
        self.lambdas.zero_()
        self.v_plus.zero_()


class PADMMInfo:
    """
    An interface container to hold the PADMM solver convergence info arrays.

    Attributes:
        lambdas (wp.array): The constraint reactions (i.e. impulses) of each world.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        v_plus (wp.array): The post-event constraint-space velocities of each world.\n
            This is computed using the current solution as: `v_plus := v_f + D @ lambdas`.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        v_aug (wp.array): The post-event augmented constraint-space velocities of each world.\n
            This is computed using the current solution as: `v_aug := v_plus + s`.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        s (wp.array): The De Saxce correction velocities of each world.\n
            This is computed using the current solution as: `s := Gamma(v_plus)`.\n
            Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        offsets (wp.array): The residuals index offset of each world.\n
            Shape of ``(num_worlds,)`` and type :class:`int32`.
        num_restarts (wp.array): History of the number of acceleration restarts performed for each world.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`int32`.
        num_rho_updates (wp.array): History of the number of penalty updates performed for each world.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`int32`.
        a (wp.array): History of PADMM acceleration variables.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        norm_s (wp.array): History of the L2 norm of De Saxce correction velocities.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        norm_x (wp.array): History of the L2 norm of primal variables.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        norm_y (wp.array): History of the L2 norm of slack variables.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        norm_z (wp.array): History of the L2 norm of dual variables.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        f_ccp (wp.array): History of CCP optimization objectives.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        f_ncp (wp.array): History of the NCP optimization objectives.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_dx (wp.array): History of the total primal iterate residual.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_dy (wp.array): History of the total slack iterate residual.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_dz (wp.array): History of the total dual iterate residual.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_primal (wp.array): History of the total primal residual.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_dual (wp.array): History of the total dual residual.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_compl (wp.array): History of the total complementarity residual.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_comb (wp.array): History of the total combined primal-dual residual.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_comb_ratio (wp.array): History of the combined primal-dual residual ratio.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_ncp_primal (wp.array): History of NCP primal residuals.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_ncp_dual (wp.array): History of NCP dual residuals.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_ncp_compl (wp.array): History of NCP complementarity residuals.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        r_ncp_natmap (wp.array): History of NCP natural-map residuals.\n
            Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.

    Notes:
    - The length of the arrays is determined by the maximum number of iterations
    and is filled up to the number of iterations performed by the solver on each
    solve. This allows for post-solve analysis of the convergence behavior.
    - This has a significant impact on solver performance and memory usage, so it
    is recommended to only enable this for testing and debugging purposes.
    """

    def __init__(
        self,
        size: ModelSize | None = None,
        max_iters: int | None = None,
    ):
        """
        Initializes the PADMM solver info container.

        If a model size is provided, allocates the solution arrays accordingly.

        Args:
            size (ModelSize | None): The model-size utility container holding the dimensionality of the model.
            max_iters (int | None): The maximum number of iterations for which to allocate convergence data.
        """

        self.lambdas: wp.array(dtype=float32) | None = None
        """
        The constraint reactions (i.e. impulses) of each world.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.v_plus: wp.array(dtype=float32) | None = None
        """
        The post-event constraint-space velocities of each world.\n
        This is computed using the current solution as: `v_plus := v_f + D @ lambdas`.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.v_aug: wp.array(dtype=float32) | None = None
        """
        The post-event augmented constraint-space velocities of each world.\n
        This is computed using the current solution as: `v_aug := v_plus + s`.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.s: wp.array(dtype=float32) | None = None
        """
        The De Saxce correction velocities of each world.\n
        This is computed using the current solution as: `s := Gamma(v_plus)`.\n
        Shape of ``(sum_of_max_total_cts,)`` and type :class:`float32`.
        """

        self.offsets: wp.array(dtype=int32) | None = None
        """
        The residuals index offset of each world.\n
        Shape of ``(num_worlds,)`` and type :class:`int32`.
        """

        self.num_restarts: wp.array(dtype=int32) | None = None
        """
        History of the number of acceleration restarts performed for each world.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`int32`.
        """

        self.num_rho_updates: wp.array(dtype=int32) | None = None
        """
        History of the number of penalty updates performed for each world.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`int32`.
        """

        self.a: wp.array(dtype=float32) | None = None
        """
        History of PADMM acceleration variables.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.norm_s: wp.array(dtype=float32) | None = None
        """
        History of the L2 norm of De Saxce correction velocities.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.norm_x: wp.array(dtype=float32) | None = None
        """
        History of the L2 norm of primal variables.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.norm_y: wp.array(dtype=float32) | None = None
        """
        History of the L2 norm of slack variables.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.norm_z: wp.array(dtype=float32) | None = None
        """
        History of the L2 norm of dual variables.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.f_ccp: wp.array(dtype=float32) | None = None
        """
        History of CCP optimization objectives.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.f_ncp: wp.array(dtype=float32) | None = None
        """
        History of the NCP optimization objectives.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_dx: wp.array(dtype=float32) | None = None
        """
        History of the total primal iterate residual.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_dy: wp.array(dtype=float32) | None = None
        """
        History of the total slack iterate residual.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_dz: wp.array(dtype=float32) | None = None
        """
        History of the total dual iterate residual.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_primal: wp.array(dtype=float32) | None = None
        """
        History of PADMM primal residuals.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_dual: wp.array(dtype=float32) | None = None
        """
        History of PADMM dual residuals.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_compl: wp.array(dtype=float32) | None = None
        """
        History of PADMM complementarity residuals.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_pd: wp.array(dtype=float32) | None = None
        """
        History of PADMM primal-dual residual ratio.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_dp: wp.array(dtype=float32) | None = None
        """
        History of PADMM dual-primal residual ratio.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_comb: wp.array(dtype=float32) | None = None
        """
        History of PADMM combined residuals.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_comb_ratio: wp.array(dtype=float32) | None = None
        """
        History of PADMM combined residuals ratio.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_ncp_primal: wp.array(dtype=float32) | None = None
        """
        History of NCP primal residuals.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_ncp_dual: wp.array(dtype=float32) | None = None
        """
        History of NCP dual residuals.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_ncp_compl: wp.array(dtype=float32) | None = None
        """
        History of NCP complementarity residuals.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        self.r_ncp_natmap: wp.array(dtype=float32) | None = None
        """
        History of NCP natural-map residuals.\n
        Shape of ``(num_worlds * max_iters,)`` and type :class:`float32`.
        """

        # Perform memory allocations if model size is specified
        if size is not None:
            self.allocate(size=size, max_iters=max_iters)

    def allocate(self, size: ModelSize, max_iters: int):
        """
        Allocates the PADMM solver info arrays based on the model size and maximum number of iterations.

        Args:
            size (ModelSize): The model-size utility container holding the dimensionality of the model.
            max_iters (int): The maximum number of iterations for which to allocate convergence data.

        Raises:
            ValueError: If either ``size.num_worlds`` or `max_iters`` are not a positive integers.
        """

        # Ensure num_worlds is valid
        if not isinstance(size.num_worlds, int) or size.num_worlds <= 0:
            raise ValueError("num_worlds must be a positive integer specifying the number of worlds.")

        # Ensure max_iters is valid
        if not isinstance(max_iters, int) or max_iters <= 0:
            raise ValueError("max_iters must be a positive integer specifying the maximum number of iterations.")

        # Allocate intermediate arrays
        self.lambdas = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.v_plus = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.v_aug = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.s = wp.zeros(size.sum_of_max_total_cts, dtype=float32)

        # Compute the index offsets for the info of each world
        maxsize = max_iters * size.num_worlds
        offsets = [max_iters * i for i in range(size.num_worlds)]

        # Allocate the in-device solver info data arrays
        self.offsets = wp.array(offsets, dtype=int32)
        self.num_restarts = wp.zeros(maxsize, dtype=int32)
        self.num_rho_updates = wp.zeros(maxsize, dtype=int32)
        self.a = wp.zeros(maxsize, dtype=float32)
        self.norm_s = wp.zeros(maxsize, dtype=float32)
        self.norm_x = wp.zeros(maxsize, dtype=float32)
        self.norm_y = wp.zeros(maxsize, dtype=float32)
        self.norm_z = wp.zeros(maxsize, dtype=float32)
        self.r_dx = wp.zeros(maxsize, dtype=float32)
        self.r_dy = wp.zeros(maxsize, dtype=float32)
        self.r_dz = wp.zeros(maxsize, dtype=float32)
        self.f_ccp = wp.zeros(maxsize, dtype=float32)
        self.f_ncp = wp.zeros(maxsize, dtype=float32)
        self.r_primal = wp.zeros(maxsize, dtype=float32)
        self.r_dual = wp.zeros(maxsize, dtype=float32)
        self.r_compl = wp.zeros(maxsize, dtype=float32)
        self.r_pd = wp.zeros(maxsize, dtype=float32)
        self.r_dp = wp.zeros(maxsize, dtype=float32)
        self.r_comb = wp.zeros(maxsize, dtype=float32)
        self.r_comb_ratio = wp.zeros(maxsize, dtype=float32)
        self.r_ncp_primal = wp.zeros(maxsize, dtype=float32)
        self.r_ncp_dual = wp.zeros(maxsize, dtype=float32)
        self.r_ncp_compl = wp.zeros(maxsize, dtype=float32)
        self.r_ncp_natmap = wp.zeros(maxsize, dtype=float32)

    def zero(self):
        """
        Resets all PADMM solver info arrays to zeros.
        """
        self.lambdas.zero_()
        self.v_plus.zero_()
        self.v_aug.zero_()
        self.s.zero_()
        self.a.zero_()
        self.num_restarts.zero_()
        self.num_rho_updates.zero_()
        self.norm_s.zero_()
        self.norm_x.zero_()
        self.norm_y.zero_()
        self.norm_z.zero_()
        self.f_ccp.zero_()
        self.f_ncp.zero_()
        self.r_dx.zero_()
        self.r_dy.zero_()
        self.r_dz.zero_()
        self.r_primal.zero_()
        self.r_dual.zero_()
        self.r_compl.zero_()
        self.r_pd.zero_()
        self.r_dp.zero_()
        self.r_comb.zero_()
        self.r_comb_ratio.zero_()
        self.r_ncp_primal.zero_()
        self.r_ncp_dual.zero_()
        self.r_ncp_compl.zero_()
        self.r_ncp_natmap.zero_()


class PADMMData:
    """
    A high-level container to manage all internal PADMM solver data.

    Attributes:
        config (wp.array): Array of per-world solver configurations,
            of type :class:`PADMMConfig` and shape ``(num_worlds,)``.\n
            Each element is the on-device version of :class:`PADMMSettings`.
        status (wp.array): Array of per-world solver status,
            of type :class:`PADMMStatus` and shape ``(num_worlds,)``.\n
            Each element holds the status of the solver on
            solving the dynamics of the corresponding world.
        penalty (wp.array): Array of per-world ALM penalty states,
            of type :class:`PADMMPenalty` and shape ``(num_worlds,)``.\n
            Each element holds the current and previous ALM penalty `rho`,
            as well as additional meta-data regarding it's adaptation.
        state (PADMMState): A container holding the PADMM state variable arrays.
        residuals (PADMMResiduals): A container holding the PADMM residuals arrays.
        solution (PADMMSolution): A container holding the PADMM solution arrays.
        info (PADMMInfo): A container holding the PADMM solver convergence info arrays.
    """

    def __init__(
        self, size: ModelSize | None = None, max_iters: int = 0, collect_info: bool = False, device: Devicelike = None
    ):
        """
        Initializes a PADMM solver data object.

        Args:
            size (ModelSize): The model-size utility container holding the dimensionality of the model.
            max_iters (int): The maximum number of iterations for which to allocate convergence data.
            collect_info (bool): Set to `True` to allocate data for reporting solver convergence info.
            device (Devicelike): The target Warp device on which all data will be allocated.

        Raises:
            ValueError: If either ``size.num_worlds`` or `max_iters`` are not a positive integers.
        """

        self.config: wp.array(dtype=PADMMConfig) | None = None
        """
        Array of on-device PADMM solver settings.\n
        Shape is (num_worlds,) and type :class:`PADMMConfig`.
        """

        self.status: wp.array(dtype=PADMMStatus) | None = None
        """
        Array of PADMM solver status.\n
        Shape is (num_worlds,) and type :class:`PADMMStatus`.
        """

        self.penalty: wp.array(dtype=PADMMPenalty) | None = None
        """
        Array of PADMM solver penalty parameters.\n
        Shape is (num_worlds,) and type :class:`PADMMPenalty`.
        """

        self.state: PADMMState | None = None
        """The PADMM internal solver state container."""

        self.residuals: PADMMResiduals | None = None
        """The PADMM residuals container."""

        self.solution: PADMMSolution | None = None
        """The PADMM solution container."""

        self.info: PADMMInfo | None = None
        """The (optional) PADMM solver info container."""

        # Perform memory allocations if model size is specified
        if size is not None:
            self.allocate(size=size, max_iters=max_iters, collect_info=collect_info, device=device)

    def allocate(self, size: ModelSize, max_iters: int = 0, collect_info: bool = False, device: Devicelike = None):
        """
        Allocates the PADMM solver data based on the model size and maximum number of iterations.

        Args:
            size (ModelSize): The model-size utility container holding the dimensionality of the model.
            max_iters (int): The maximum number of iterations for which to allocate convergence data.
            collect_info (bool): Set to `True` to allocate data for reporting solver convergence info.
            device (Devicelike): The target Warp device on which all data will be allocated.

        Raises:
            ValueError: If either ``size.num_worlds`` or `max_iters`` are not a positive integers.
        """
        with wp.ScopedDevice(device):
            self.config = wp.zeros(shape=(size.num_worlds,), dtype=PADMMConfig)
            self.status = wp.zeros(shape=(size.num_worlds,), dtype=PADMMStatus)
            self.penalty = wp.zeros(shape=(size.num_worlds,), dtype=PADMMPenalty)
            self.state = PADMMState(size)
            self.residuals = PADMMResiduals(size)
            self.solution = PADMMSolution(size)
            if collect_info and max_iters > 0:
                self.info = PADMMInfo(size=size, max_iters=max_iters)


###
# Functions
###


@wp.func
def project_to_coulomb_cone(x: vec3f, mu: float32, epsilon: float32 = 0.0) -> vec3f:
    """
    Projects a 3D vector `x` onto an isotropic Coulomb friction cone defined by the friction coefficient `mu`.

    Args:
        x (vec3f): The input vector to be projected.
        mu (float32): The friction coefficient defining the aperture of the cone.
        epsilon (float32, optional): A numerical tolerance applied to the cone boundary. Defaults to 0.0.

    Returns:
        vec3f: The vector projected onto the Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    if mu * xt_norm > -xn + epsilon:
        if xt_norm <= mu * xn + epsilon:
            y = x
        else:
            ys = (mu * xt_norm + xn) / (mu * mu + 1.0)
            yts = mu * ys / xt_norm
            y[0] = yts * x[0]
            y[1] = yts * x[1]
            y[2] = ys
    return y


@wp.func
def project_to_coulomb_dual_cone(x: vec3f, mu: float32, epsilon: float32 = 0.0) -> vec3f:
    """
    Projects a 3D vector `x` onto the dual of an isotropic Coulomb
    friction cone defined by the friction coefficient `mu`.

    Args:
        x (vec3f): The input vector to be projected.
        mu (float32): The friction coefficient defining the aperture of the cone.
        epsilon (float32, optional): A numerical tolerance applied to the cone boundary. Defaults to 0.0.

    Returns:
        vec3f: The vector projected onto the dual Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    if xt_norm > -mu * xn + epsilon:
        if mu * xt_norm <= xn + epsilon:
            y = x
        else:
            ys = (xt_norm + mu * xn) / (mu * mu + 1.0)
            yts = ys / xt_norm
            y[0] = yts * x[0]
            y[1] = yts * x[1]
            y[2] = mu * ys
    return y


@wp.func
def compute_inf_norm(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
) -> float32:
    norm = float(0.0)
    for i in range(dim):
        norm = wp.max(norm, wp.abs(x[vio + i]))
    return norm


@wp.func
def compute_l1_norm(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
) -> float32:
    sum = float(0.0)
    for i in range(dim):
        sum += wp.abs(x[vio + i])
    return sum


@wp.func
def compute_l2_norm(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
) -> float32:
    sum = float(0.0)
    for i in range(dim):
        x_i = x[vio + i]
        sum += x_i * x_i
    return wp.sqrt(sum)


@wp.func
def compute_dot_product(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
) -> float32:
    """
    Computes the dot (i.e. inner) product between two vectors `x` and `y` stored in flat arrays.\n
    Both vectors are of dimension `dim`, starting from the vector index offset `vio`.

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        x (wp.array(dtype=float32)): The first vector.
        y (wp.array(dtype=float32)): The second vector.

    Returns:
        float32: The dot product of the two vectors.
    """
    product = float(0.0)
    for i in range(dim):
        v_i = vio + i
        product += x[v_i] * y[v_i]
    return product


@wp.func
def compute_double_dot_product(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    z: wp.array(dtype=float32),
) -> float32:
    """
    Computes the the inner product `x.T @ (y + z)` between a vector `x` and the sum of two vectors `y` and `z`.\n
    All vectors are stored in flat arrays, with dimension `dim` and starting from the vector index offset `vio`.

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        x (wp.array(dtype=float32)): The first vector.
        y (wp.array(dtype=float32)): The second vector.
        z (wp.array(dtype=float32)): The third vector.

    Returns:
        float32: The inner product of `x` with the sum of `y` and `z`.
    """
    product = float(0.0)
    for i in range(dim):
        v_i = vio + i
        product += x[v_i] * (y[v_i] + z[v_i])
    return product


@wp.func
def compute_vector_sum(
    dim: int32, vio: int32, x: wp.array(dtype=float32), y: wp.array(dtype=float32), z: wp.array(dtype=float32)
):
    """
    Computes the sum of two vectors `x` and `y` and stores the result in vector `z`.\n
    All vectors are stored in flat arrays, with dimension `dim` and starting from the vector index offset `vio`.

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        x (wp.array(dtype=float32)): The first vector.
        y (wp.array(dtype=float32)): The second vector.
        z (wp.array(dtype=float32)): The output vector where the sum is stored.

    Returns:
        None: The result is stored in the output vector `z`.
    """
    for i in range(dim):
        v_i = vio + i
        z[v_i] = x[v_i] + y[v_i]


@wp.func
def compute_cwise_vec_mul(
    dim: int32,
    vio: int32,
    a: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
):
    """
    Computes the coefficient-wise vector-vector product `y =  a * x`.\n

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        a (wp.array(dtype=float32)): Input array containing the first set of vectors.
        x (wp.array(dtype=float32)): Input array containing the second set of vectors.
        y (wp.array(dtype=float32)): Output array where the result is stored.
    """
    for i in range(dim):
        v_i = vio + i
        y[v_i] = a[v_i] * x[v_i]


@wp.func
def compute_gemv(
    dim: int32,
    vio: int32,
    mio: int32,
    sigma: float32,
    P: wp.array(dtype=float32),
    A: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    b: wp.array(dtype=float32),
    c: wp.array(dtype=float32),
):
    """
    Computes the generalized matrix-vector product `c =  b + (A - sigma * I_n)@ x`.\n

    The matrix `A` is stored using row-major order in flat array with allocation size `maxdim x maxdim`,
    starting from the matrix index offset `mio`. The active dimensions of the matrix are `dim x dim`,
    where `dim` is the number of rows and columns. The vectors `x, b, c` are stored in flat arrays with
    dimensions `dim`, starting from the vector index offset `vio`.

    Args:
        maxdim (int32): The maximum dimension of the matrix `A`.
        dim (int32): The active dimension of the matrix `A` and the vectors `x, b, c`.
        vio (int32): The vector index offset (i.e. start index) for the vectors `x, b, c`.
        mio (int32): The matrix index offset (i.e. start index) for the matrix `A`.
        A (wp.array(dtype=float32)):
            Input matrix `A` stored in row-major order.
        x (wp.array(dtype=float32)):
            Input array `x` to be multiplied with the matrix `A`.
        b (wp.array(dtype=float32)):
            Input array `b` to be added to the product `A @ x`.
        c (wp.array(dtype=float32)):
            Output array `c` where the result of the operation is stored.
    """
    b_i = float(0.0)
    x_j = float(0.0)
    for i in range(dim):
        v_i = vio + i
        m_i = mio + dim * i
        b_i = b[v_i]
        for j in range(dim):
            x_j = x[vio + j]
            b_i += A[m_i + j] * x_j
        b_i -= sigma * x[v_i]
        c[v_i] = (1.0 / P[v_i]) * b_i


@wp.func
def compute_desaxce_corrections(
    nc: int32,
    cio: int32,
    vio: int32,
    ccgo: int32,
    mu: wp.array(dtype=float32),
    v_plus: wp.array(dtype=float32),
    s: wp.array(dtype=float32),
):
    """
    Computes the De Saxce correction for each active contact.

    `s = G(v) := [ 0, 0 , mu * || vt ||_2,]^T, where v := [vtx, vty, vn]^T, vt := [vtx, vty]^T`

    Args:
        nc (int32): The number of active contact constraints.
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        vio (int32): The vector index offset (i.e. start index)
        ccgo (int32): The contact constraint group offset (i.e. start index)
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact constraint.
        v_plus (wp.array(dtype=float32)):
            The post-event constraint-space velocities array, which contains the tangential velocities `vtx`
            and `vty` for each contact constraint.
        s (wp.array(dtype=float32)):
            The output array where the De Saxce corrections are stored.\n
            The size of this array should be at least `vio + ccgo + 3 * nc`, where `vio` is the vector index offset,
            `ccgo` is the contact constraint group offset, and `nc` is the number of active contact constraints.

    Returns:
        None: The De Saxce corrections are stored in the output array `s`.
    """
    # Iterate over each active contact
    for k in range(nc):
        # Compute the contact index
        c_k = cio + k

        # Compute the constraint vector index
        v_k = vio + ccgo + 3 * k

        # Retrieve the friction coefficient for this contact
        mu_k = mu[c_k]

        # Compute the 2D norm of the tangential velocity
        vtx_k = v_plus[v_k]
        vty_k = v_plus[v_k + 1]
        vt_norm_k = wp.sqrt(vtx_k * vtx_k + vty_k * vty_k)

        # Store De Saxce correction for this block
        s[v_k] = 0.0
        s[v_k + 1] = 0.0
        s[v_k + 2] = mu_k * vt_norm_k


@wp.func
def compute_ncp_primal_residual(
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    cio: int32,
    mu: wp.array(dtype=float32),
    lambdas: wp.array(dtype=float32),
) -> float32:
    """
    Computes the NCP primal residual as: `r_p := || lambda - proj_K(lambda) ||_inf`, where:
    - `lambda` is the vector of constraint reactions (i.e. impulses)
    - `proj_K()` is the projection operator onto the cone `K`
    - `K` is the total cone defined by the unilateral constraints such as limits and contacts
    - `|| . ||_inf` is the infinity norm (i.e. maximum absolute value of the vector components)

    Notes:
    - The cone for joint constraints is all of `R^njc`, so projection is a no-op.
    - For limit constraints, the cone is defined as `K_l := { lambda | lambda >= 0 }`
    - For contact constraints, the cone is defined as `K_c := { lambda | || lambda ||_2 <= mu * || vn ||_2 }`

    Args:
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact.
        lambdas (wp.array(dtype=float32)):
            The array of constraint reactions (i.e. impulses).

    Returns:
        float32: The maximum primal residual across all constraints, computed as the infinity norm.
    """
    # Initialize the primal residual
    r_ncp_p = float(0.0)

    # NOTE: We skip the joint constraint reactions are not bounded, the cone is all of R^njc

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio = vio + lcgo + lid
        # Compute the primal residual for the limit constraints
        lambda_l = lambdas[lcio]
        lambda_l -= wp.max(0.0, lambda_l)
        r_ncp_p = wp.max(r_ncp_p, wp.abs(lambda_l))

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio = vio + ccgo + 3 * cid
        # Retrieve the friction coefficient for this contact
        mu_c = mu[cio + cid]
        # Compute the primal residual for the contact constraints
        lambda_c = vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        lambda_c -= project_to_coulomb_cone(lambda_c, mu_c)
        r_ncp_p = wp.max(r_ncp_p, wp.max(wp.abs(lambda_c)))

    # Return the maximum primal residual
    return r_ncp_p


@wp.func
def compute_ncp_dual_residual(
    njc: int32,
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    cio: int32,
    mu: wp.array(dtype=float32),
    v_aug: wp.array(dtype=float32),
) -> float32:
    """
    Computes the NCP dual residual as: `r_d := || v_aug - proj_K^*(v_aug) ||_inf`, where:
    - `v_aug` is the vector of augmented constraint velocities: v_aug := v_plus + s
    - `v_plus` is the post-event constraint-space velocities
    - `s` is the De Saxce correction vector
    - `proj_K^*()` is the projection operator onto the dual cone `K^*`
    - `K^*` is the dual of the total cone defined by the unilateral constraints such as limits and contacts
    - `|| . ||_inf` is the infinity norm (i.e. maximum absolute value of the vector components)

    Notes:
    - The dual cone for joint constraints is the origin point x=0.
    - For limit constraints, the cone is defined as `K_l := { lambda | lambda >= 0 }`
    - For contact constraints, the cone is defined as `K_c := { lambda | || lambda ||_2 <= mu * || vn ||_2 }`

    Args:
        njc (int32): The number of joint constraints.
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact constraint.
        v_aug (wp.array(dtype=float32)):
            The array of augmented constraint velocities.

    Returns:
        float32: The maximum dual residual across all constraints, computed as the infinity norm.
    """
    # Initialize the dual residual
    r_ncp_d = float(0.0)

    for jid in range(njc):
        # Compute the joint constraint index offset
        jcio_j = vio + jid
        # Compute the dual residual for the joint constraints
        # NOTE #1: Each constraint-space velocity for joint should be zero
        # NOTE #2: the dual of R^njc is the origin zero vector
        v_j = v_aug[jcio_j]
        r_ncp_d = wp.max(r_ncp_d, wp.abs(v_j))

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio_l = vio + lcgo + lid
        # Compute the dual residual for the limit constraints
        # NOTE: Each constraint-space velocity should be non-negative
        v_l = v_aug[lcio_l]
        v_l -= wp.max(0.0, v_l)
        r_ncp_d = wp.max(r_ncp_d, wp.abs(v_l))

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio_c = vio + ccgo + 3 * cid
        # Retrieve the friction coefficient for this contact
        mu_c = mu[cio + cid]
        # Compute the dual residual for the contact constraints
        # NOTE: Each constraint-space velocity should be lie in the dual of the Coulomb friction cone
        v_c = vec3f(v_aug[ccio_c], v_aug[ccio_c + 1], v_aug[ccio_c + 2])
        v_c -= project_to_coulomb_dual_cone(v_c, mu_c)
        r_ncp_d = wp.max(r_ncp_d, wp.max(wp.abs(v_c)))

    # Return the maximum dual residual
    return r_ncp_d


@wp.func
def compute_ncp_complementarity_residual(
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    v_aug: wp.array(dtype=float32),
    lambdas: wp.array(dtype=float32),
) -> float32:
    """
    Computes the NCP complementarity residual as `r_c := || lambda.dot(v_plus + s) ||_inf`

    Satisfaction of the complementarity condition `lambda _|_ (v_plus + s))` is measured
    using the per-constraint entity inner product, i.e. per limit and per contact. Thus,
    for each limit constraint `k`, we compute `v_k * lambda_k` and for each contact
    constraint `k`, we compute `v_k.dot(lambda_k)`.

    Args:
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        v_aug (wp.array(dtype=float32)):
            The array of augmented constraint velocities.
        lambdas (wp.array(dtype=float32)):
            The array of constraint reactions (i.e. impulses).

    Returns:
        float32: The maximum complementarity residual across all constraints, computed as the infinity norm.
    """
    # Initialize the complementarity residual
    r_ncp_c = float(0.0)

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio = vio + lcgo + lid
        # Compute the complementarity residual for the limit constraints
        v_l = v_aug[lcio]
        lambda_l = lambdas[lcio]
        r_ncp_c = wp.max(r_ncp_c, wp.abs(v_l * lambda_l))

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio = vio + ccgo + 3 * cid
        # Compute the complementarity residual for the contact constraints
        v_c = vec3f(v_aug[ccio], v_aug[ccio + 1], v_aug[ccio + 2])
        lambda_c = vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        r_ncp_c = wp.max(r_ncp_c, wp.abs(wp.dot(v_c, lambda_c)))

    # Return the maximum complementarity residual
    return r_ncp_c


@wp.func
def compute_ncp_natural_map_residual(
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    cio: int32,
    mu: wp.array(dtype=float32),
    v_aug: wp.array(dtype=float32),
    lambdas: wp.array(dtype=float32),
) -> float32:
    """
    Computes the natural-map residuals as: `r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf`

    Args:
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact.
        v_aug (wp.array(dtype=float32)):
            The array of augmented constraint velocities.
        lambdas (wp.array(dtype=float32)):
            The array of constraint reactions (i.e. impulses).
    """

    # Initialize the natural-map residual
    r_ncp_natmap = float(0.0)

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio = vio + lcgo + lid
        # Compute the natural-map residual for the limit constraints
        v_l = v_aug[lcio]
        lambda_l = lambdas[lcio]
        lambda_l -= wp.max(0.0, lambda_l - v_l)
        r_ncp_natmap = wp.max(r_ncp_natmap, wp.abs(lambda_l))

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio = vio + ccgo + 3 * cid
        # Retrieve the friction coefficient for this contact
        mu_c = mu[cio + cid]
        # Compute the natural-map residual for the contact constraints
        v_c = vec3f(v_aug[ccio], v_aug[ccio + 1], v_aug[ccio + 2])
        lambda_c = vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        lambda_c -= project_to_coulomb_cone(lambda_c - v_c, mu_c)
        r_ncp_natmap = wp.max(r_ncp_natmap, wp.max(wp.abs(lambda_c)))

    # Return the maximum natural-map residual
    return r_ncp_natmap


@wp.func
def compute_preconditioned_iterate_residual(
    ncts: int32, vio: int32, P: wp.array(dtype=float32), x: wp.array(dtype=float32), x_p: wp.array(dtype=float32)
) -> float32:
    """
    Computes the iterate residual as: `r_dx := || P @ (x - x_p) ||_inf`

    Args:
        ncts (int32): The number of active constraints in the world.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        x (wp.array(dtype=float32)):
            The current solution vector.
        x_p (wp.array(dtype=float32)):
            The previous solution vector.

    Returns:
        float32: The maximum iterate residual across all active constraints, computed as the infinity norm.
    """
    # Initialize the iterate residual
    r_dx = float(0.0)
    for i in range(ncts):
        # Compute the index offset of the vector block of the world
        v_i = vio + i
        # Update the iterate and proximal-point residuals
        r_dx = wp.max(r_dx, P[v_i] * wp.abs(x[v_i] - x_p[v_i]))
    # Return the maximum iterate residual
    return r_dx


@wp.func
def compute_inverse_preconditioned_iterate_residual(
    ncts: int32, vio: int32, P: wp.array(dtype=float32), x: wp.array(dtype=float32), x_p: wp.array(dtype=float32)
) -> float32:
    """
    Computes the iterate residual as: `r_dx := || P^{-1} @ (x - x_p) ||_inf`

    Args:
        ncts (int32): The number of active constraints in the world.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        x (wp.array(dtype=float32)):
            The current solution vector.
        x_p (wp.array(dtype=float32)):
            The previous solution vector.

    Returns:
        float32: The maximum iterate residual across all active constraints, computed as the infinity norm.
    """
    # Initialize the iterate residual
    r_dx = float(0.0)
    for i in range(ncts):
        # Compute the index offset of the vector block of the world
        v_i = vio + i
        # Update the iterate and proximal-point residuals
        r_dx = wp.max(r_dx, (1.0 / P[v_i]) * wp.abs(x[v_i] - x_p[v_i]))
    # Return the maximum iterate residual
    return r_dx


###
# Kernels
###


# TODO: Make multiple initialization kernels for the different penalty update methods
@wp.kernel
def _initialize_solver(
    # Inputs:
    solver_config: wp.array(dtype=PADMMConfig),
    # Outputs:
    solver_status: wp.array(dtype=PADMMStatus),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_state_a_p: wp.array(dtype=float32),
):
    # Retrieve the world index as thread index
    wid = wp.tid()

    # Retrieve the solver configuration
    config = solver_config[wid]

    # Initialize status
    s = solver_status[wid]
    s.iterations = int(0)
    s.converged = int(0)
    s.r_p = float(0.0)
    s.r_d = float(0.0)
    s.r_c = float(0.0)
    s.r_dx = float(0.0)
    s.r_dy = float(0.0)
    s.r_dz = float(0.0)
    s.r_a = FLOAT32_MAX
    s.r_a_p = FLOAT32_MAX
    s.r_a_pp = FLOAT32_MAX
    s.restart = int(0)
    s.num_restarts = int(0)
    solver_status[wid] = s

    # Initialize penalty
    # NOTE: Currently only fixed penalty is used
    p = solver_penalty[wid]
    p.rho = config.rho_0
    p.rho_p = float(0.0)
    p.num_updates = int(0)
    solver_penalty[wid] = p

    # Initialize the previous acclereration variable
    solver_state_a_p[wid] = config.a_0


@wp.kernel
def _update_delassus_proximal_regularization(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_mio: wp.array(dtype=int32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    # Outputs:
    D: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the matrix index offset of the world
    mio = problem_mio[wid]

    # Retrieve the solver parameters
    cfg = solver_config[wid]
    pen = solver_penalty[wid]

    # Extract the regularization parameters
    rho_p = pen.rho_p
    rho = pen.rho
    eta = cfg.eta

    # Add the proximal regularization to the diagonal of the Delassus matrix
    D[mio + ncts * tid + tid] += eta + (rho - rho_p)


@wp.kernel
def _compute_desaxce_correction(
    # Inputs:
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_z_p: wp.array(dtype=float32),
    # Outputs:
    solver_s: wp.array(dtype=float32),
):
    # Retrieve the thread index as the contact index
    wid, cid = wp.tid()

    # Retrieve the number of contact active in the world
    nc = problem_nc[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if cid >= nc or status.converged > 0:
        return

    # Retrieve the contacts index offset of the world
    cio = problem_cio[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Retrieve the contact constraints group offset of the world
    ccgo = problem_ccgo[wid]

    # Compute the index offset of the corresponding contact constraint
    ccio_k = vio + ccgo + 3 * cid

    # Retrieve the contact index w.r.t the model
    cio_k = cio + cid

    # Compute the norm of the tangential components
    vtx = solver_z_p[ccio_k]
    vty = solver_z_p[ccio_k + 1]
    vt_norm = wp.sqrt(vtx * vtx + vty * vty)

    # Store De Saxce correction for this block
    solver_s[ccio_k] = 0.0
    solver_s[ccio_k + 1] = 0.0
    solver_s[ccio_k + 2] = problem_mu[cio_k] * vt_norm


@wp.kernel
def _compute_velocity_bias(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_v_f: wp.array(dtype=float32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_s: wp.array(dtype=float32),
    solver_x_p: wp.array(dtype=float32),
    solver_y_p: wp.array(dtype=float32),
    solver_z_p: wp.array(dtype=float32),
    # Outputs:
    solver_v: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Retrieve solver parameters
    eta = solver_config[wid].eta
    rho = solver_penalty[wid].rho

    # Compute the index offset of the vector block of the world
    thread_offset = vio + tid

    # Retrieve the solver state
    v_f = problem_v_f[thread_offset]
    s = solver_s[thread_offset]
    x_p = solver_x_p[thread_offset]
    y_p = solver_y_p[thread_offset]
    z_p = solver_z_p[thread_offset]

    # Compute the total velocity bias for the thread_offset-th constraint
    solver_v[thread_offset] = -v_f - s + eta * x_p + rho * y_p + z_p


@wp.kernel
def _compute_projection_argument(
    # Inputs
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_z_hat: wp.array(dtype=float32),
    solver_x: wp.array(dtype=float32),
    # Outputs
    solver_y: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Capture the ALM penalty
    rho = solver_penalty[wid].rho

    # Compute the index offset of the vector block of the world
    thread_offset = vio + tid

    # Retrieve the solver state variables
    z_hat = solver_z_hat[thread_offset]
    x = solver_x[thread_offset]

    # Compute and store the updated values back to the solver state
    solver_y[thread_offset] = x - (1.0 / rho) * z_hat


@wp.kernel
def _apply_overrelaxation_and_compute_projection_argument(
    # Inputs
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_y_p: wp.array(dtype=float32),
    solver_z_p: wp.array(dtype=float32),
    # Outputs
    solver_x: wp.array(dtype=float32),
    solver_y: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Retrieve the relaxation factor
    omega = solver_config[wid].omega

    # Capture the ALM penalty
    rho = solver_penalty[wid].rho

    # Compute the index offset of the vector block of the world
    thread_offset = vio + tid

    # Retrieve the solver state variables
    y_p = solver_y_p[thread_offset]
    z_p = solver_z_p[thread_offset]
    x = solver_x[thread_offset]
    y = solver_y[thread_offset]

    # Compute the over-relaxation update
    x = omega * x + (1.0 - omega) * y_p

    # Compute argument for the projection operator
    y = x - (1.0 / rho) * z_p

    # Store the updated values back to the solver state
    solver_x[thread_offset] = x
    solver_y[thread_offset] = y


# TODO: Break this up to two kernels launched simultaneously (1x for limits, 1x for contacts)
@wp.kernel
def _project_to_feasible_cone(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_lcgo: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    solver_status: wp.array(dtype=PADMMStatus),
    # Outputs:
    solver_y: wp.array(dtype=float32),
):
    # Retrieve the thread index as the unilateral entity index
    wid, uid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Retrieve the number of active limits and contacts in the world
    nl = problem_nl[wid]
    nc = problem_nc[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if uid >= (nl + nc) or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Check if the thread should handle a limit
    if nl > 0 and uid < nl:
        # Retrieve the limit constraint group offset of the world
        lcgo = problem_lcgo[wid]
        # Compute the constraint index offset of the limit element
        lcio_j = vio + lcgo + uid
        # Project to the non-negative orthant
        solver_y[lcio_j] = wp.max(solver_y[lcio_j], 0.0)

    # Check if the thread should handle a contact
    elif nc > 0 and uid >= nl:
        # Retrieve the contact index offset of the world
        cio = problem_cio[wid]
        # Retrieve the limit constraint group offset of the world
        ccgo = problem_ccgo[wid]
        # Compute the index of the contact element in the unilaterals array
        # NOTE: We need to subtract the number of active limits
        cid = uid - nl
        # Compute the index offset of the contact constraint
        ccio_j = vio + ccgo + 3 * cid
        # Capture a 3D vector
        x = vec3f(solver_y[ccio_j], solver_y[ccio_j + 1], solver_y[ccio_j + 2])
        # Project to the coulomb friction cone
        y_proj = project_to_coulomb_cone(x, problem_mu[cio + cid])
        # Copy vec3 projection into the slack variable array
        solver_y[ccio_j] = y_proj[0]
        solver_y[ccio_j + 1] = y_proj[1]
        solver_y[ccio_j + 2] = y_proj[2]


@wp.kernel
def _update_dual_variables_and_compute_primal_dual_residuals(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_P: wp.array(dtype=float32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_x: wp.array(dtype=float32),
    solver_y: wp.array(dtype=float32),
    solver_x_p: wp.array(dtype=float32),
    solver_y_hat: wp.array(dtype=float32),
    solver_z_hat: wp.array(dtype=float32),
    # Outputs:
    solver_z: wp.array(dtype=float32),
    solver_r_prim: wp.array(dtype=float32),
    solver_r_dual: wp.array(dtype=float32),
    solver_r_dx: wp.array(dtype=float32),
    solver_r_dy: wp.array(dtype=float32),
    solver_r_dz: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Capture proximal parameter and the ALM penalty
    eta = solver_config[wid].eta
    rho = solver_penalty[wid].rho

    # Compute the index offset of the vector block of the world
    thread_offset = vio + tid

    # Retrieve
    P_i = problem_P[thread_offset]

    # Retrieve the solver state inputs
    x = solver_x[thread_offset]
    y = solver_y[thread_offset]
    x_p = solver_x_p[thread_offset]
    y_hat = solver_y_hat[thread_offset]
    z_hat = solver_z_hat[thread_offset]

    # Compute and store the dual variable update
    z = z_hat + rho * (y - x)
    solver_z[thread_offset] = z

    # Compute the primal residual as the consensus of the primal and slack variable
    solver_r_prim[thread_offset] = P_i * (x - y)

    # Compute the dual residual using the ADMM-specific shortcut
    solver_r_dual[thread_offset] = (1.0 / P_i) * (eta * (x - x_p) + rho * (y - y_hat))

    # Compute the combined residual terms
    solver_r_dx[thread_offset] = P_i * (x - x_p)
    solver_r_dy[thread_offset] = P_i * (y - y_hat)
    solver_r_dz[thread_offset] = (1.0 / P_i) * (z - z_hat)


# TODO: Break this up to two kernels launched simultaneously (1x for limits, 1x for contacts)?
@wp.kernel
def _compute_complementarity_residuals(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_uio: wp.array(dtype=int32),
    problem_lcgo: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_x: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
    # Outputs:
    solver_r_c: wp.array(dtype=float32),
):
    # Retrieve the thread index as the unilateral entity index
    wid, uid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Retrieve the number of active limits and contacts in the world
    nl = problem_nl[wid]
    nc = problem_nc[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if uid >= (nl + nc) or status.converged > 0:
        return

    # Retrieve the index offsets of the unilateral elements
    uio = problem_uio[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Compute the index offset of the vector block of the world
    uio_u = uio + uid

    # Check if the thread should handle a limit
    if nl > 0 and uid < nl:
        # Retrieve the limit constraint group offset of the world
        lcgo = problem_lcgo[wid]
        # Compute the constraint index offset of the limit element
        lcio_j = vio + lcgo + uid
        # Compute the scalar product of the primal and dual variables
        solver_r_c[uio_u] = solver_x[lcio_j] * solver_z[lcio_j]

    # Check if the thread should handle a contact
    elif nc > 0 and uid >= nl:
        # Retrieve the limit constraint group offset of the world
        ccgo = problem_ccgo[wid]
        # Compute the index of the contact element in the unilaterals array
        # NOTE: We need to subtract the number of active limits
        cid = uid - nl
        # Compute the index offset of the contact constraint
        ccio_j = vio + ccgo + 3 * cid
        # Capture 3D vectors
        x_c = vec3f(solver_x[ccio_j], solver_x[ccio_j + 1], solver_x[ccio_j + 2])
        z_c = vec3f(solver_z[ccio_j], solver_z[ccio_j + 1], solver_z[ccio_j + 2])
        # Compute the inner product of the primal and dual variables
        solver_r_c[uio_u] = wp.dot(x_c, z_c)


@wp.kernel
def _compute_infnorm_residuals_serially(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_uio: wp.array(dtype=int32),
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_r_p: wp.array(dtype=float32),
    solver_r_d: wp.array(dtype=float32),
    solver_r_c: wp.array(dtype=float32),
    # Outputs:
    solver_state_done: wp.array(dtype=int32),
    solver_status: wp.array(dtype=PADMMStatus),
):
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip this step if already converged
    if status.converged:
        return

    # Update iteration counter
    status.iterations += 1

    # Capture the size of the residuals arrays
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    ncts = problem_dim[wid]

    # Retrieve the solver configurations
    config = solver_config[wid]

    # Retrieve the index offsets of the vector block and unilateral elements
    vio = problem_vio[wid]
    uio = problem_uio[wid]

    # Extract the solver tolerances
    eps_p = config.primal_tolerance
    eps_d = config.dual_tolerance
    eps_c = config.compl_tolerance

    # Extract the maximum number of iterations
    maxiters = config.max_iterations

    # Compute element-wise max over each residual vector to compute the infinity-norm
    r_p_max = float(0.0)
    r_d_max = float(0.0)
    for j in range(ncts):
        rio_j = vio + j
        r_p_max = wp.max(r_p_max, wp.abs(solver_r_p[rio_j]))
        r_d_max = wp.max(r_d_max, wp.abs(solver_r_d[rio_j]))

    # Compute the infinity-norm of the complementarity residuals
    nu = nl + nc
    r_c_max = float(0.0)
    for j in range(nu):
        r_c_max = wp.max(r_c_max, wp.abs(solver_r_c[uio + j]))

    # Store the scalar metric residuals in the solver status
    status.r_p = r_p_max
    status.r_d = r_d_max
    status.r_c = r_c_max

    # Check and store convergence state
    if status.iterations > 1 and r_p_max <= eps_p and r_d_max <= eps_d and r_c_max <= eps_c:
        status.converged = 1

    # If converged or reached max iterations, decrement the number of active worlds
    if status.converged or status.iterations >= maxiters:
        solver_state_done[0] -= 1

    # Store the updated status
    solver_status[wid] = status


@wp.kernel
def _compute_infnorm_residuals_serially_accel(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_uio: wp.array(dtype=int32),
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_params: wp.array(dtype=PADMMPenalty),
    solver_r_p: wp.array(dtype=float32),
    solver_r_d: wp.array(dtype=float32),
    solver_r_c: wp.array(dtype=float32),
    solver_r_dx: wp.array(dtype=float32),
    solver_r_dy: wp.array(dtype=float32),
    solver_r_dz: wp.array(dtype=float32),
    solver_state_a_p: wp.array(dtype=float32),
    # Outputs:
    solver_state_done: wp.array(dtype=int32),
    solver_state_a: wp.array(dtype=float32),
    solver_status: wp.array(dtype=PADMMStatus),
):
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip this step if already converged
    if status.converged:
        return

    # Update iteration counter
    status.iterations += 1

    # Capture the size of the residuals arrays
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    ncts = problem_dim[wid]

    # Retrieve the solver configurations
    config = solver_config[wid]

    # Retrieve the index offsets of the vector block and unilateral elements
    vio = problem_vio[wid]
    uio = problem_uio[wid]

    # Extract the solver tolerances
    eps_p = config.primal_tolerance
    eps_d = config.dual_tolerance
    eps_c = config.compl_tolerance

    # Extract the maximum number of iterations
    maxiters = config.max_iterations

    # Extract the penalty parameters
    params = solver_params[wid]
    rho = params.rho

    # Compute element-wise max over each residual vector to compute the infinity-norm
    r_p_max = float(0.0)
    r_d_max = float(0.0)
    r_dx_l2_sum = float(0.0)
    r_dy_l2_sum = float(0.0)
    r_dz_l2_sum = float(0.0)
    for j in range(ncts):
        rio_j = vio + j
        r_p_max = wp.max(r_p_max, wp.abs(solver_r_p[rio_j]))
        r_d_max = wp.max(r_d_max, wp.abs(solver_r_d[rio_j]))
        r_dx = solver_r_dx[rio_j]
        r_dy = solver_r_dy[rio_j]
        r_dz = solver_r_dz[rio_j]
        r_dx_l2_sum += r_dx * r_dx
        r_dy_l2_sum += r_dy * r_dy
        r_dz_l2_sum += r_dz * r_dz

    # Compute the infinity-norm of the complementarity residuals
    nu = nl + nc
    r_c_max = float(0.0)
    for j in range(nu):
        r_c_max = wp.max(r_c_max, wp.abs(solver_r_c[uio + j]))

    # Store the scalar metric residuals in the solver status
    status.r_p = r_p_max
    status.r_d = r_d_max
    status.r_c = r_c_max
    status.r_dx = wp.sqrt(r_dx_l2_sum)
    status.r_dy = wp.sqrt(r_dy_l2_sum)
    status.r_dz = wp.sqrt(r_dz_l2_sum)
    status.r_a = rho * status.r_dy + (1.0 / rho) * status.r_dz

    # Check and store convergence state
    if status.iterations > 1 and r_p_max <= eps_p and r_d_max <= eps_d and r_c_max <= eps_c:
        status.converged = 1

    # If converged or reached max iterations, decrement the number of active worlds
    if status.converged or status.iterations >= maxiters:
        solver_state_done[0] -= 1

    # Restart acceleration if the residuals are not decreasing sufficiently
    # TODO: Use a warp function that is wrapped with wp.static for conditionally compiling this
    if status.r_a < config.restart_tolerance * status.r_a_p:
        status.restart = 0
        a_p = solver_state_a_p[wid]
        solver_state_a[wid] = (1.0 + wp.sqrt(1.0 + 4.0 * a_p * a_p)) / 2.0
    else:
        status.restart = 1
        status.num_restarts += 1
        status.r_a = status.r_a_p / config.restart_tolerance
        solver_state_a[wid] = float(config.a_0)
    status.r_a_pp = status.r_a_p
    status.r_a_p = status.r_a

    # Store the updated status
    solver_status[wid] = status


@wp.kernel
def _update_state_with_acceleration(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_status: wp.array(dtype=PADMMStatus),
    solver_state_a: wp.array(dtype=float32),
    solver_state_y: wp.array(dtype=float32),
    solver_state_z: wp.array(dtype=float32),
    solver_state_a_p: wp.array(dtype=float32),
    solver_state_y_p: wp.array(dtype=float32),
    solver_state_z_p: wp.array(dtype=float32),
    # Outputs:
    solver_state_y_hat: wp.array(dtype=float32),
    solver_state_z_hat: wp.array(dtype=float32),
):
    # Retrieve the thread indices as the world and constraint index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Retrieve the solver status
    status = solver_status[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts or status.converged > 0:
        return

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Compute the index offset of the vector block of the world
    vid = vio + tid

    # Only apply acceleration if not restarting
    if status.restart == 0:
        # Retrieve the previous and current states
        a = solver_state_a[wid]
        y = solver_state_y[vid]
        z = solver_state_z[vid]
        a_p = solver_state_a_p[wid]
        y_p = solver_state_y_p[vid]
        z_p = solver_state_z_p[vid]

        # Compute the current acceleration factor
        factor = (a_p - 1.0) / a

        # Update the primal and dual variables with Nesterov acceleration
        y_hat_new = y + factor * (y - y_p)
        z_hat_new = z + factor * (z - z_p)

        # Store the updated states
        solver_state_y_hat[vid] = y_hat_new
        solver_state_z_hat[vid] = z_hat_new
    else:
        # If restarting, reset the auxiliary primal-dual state to the previous-step values
        solver_state_y_hat[vid] = solver_state_y_p[vid]
        solver_state_z_hat[vid] = solver_state_z_p[vid]


@wp.kernel
def _collect_solver_convergence_info(
    # Inputs:
    problem_nl: wp.array(dtype=int32),
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_lcgo: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    problem_v_f: wp.array(dtype=float32),
    problem_D: wp.array(dtype=float32),
    problem_P: wp.array(dtype=float32),
    solver_state_a: wp.array(dtype=float32),
    solver_state_s: wp.array(dtype=float32),
    solver_state_x: wp.array(dtype=float32),
    solver_state_x_p: wp.array(dtype=float32),
    solver_state_y: wp.array(dtype=float32),
    solver_state_y_p: wp.array(dtype=float32),
    solver_state_z: wp.array(dtype=float32),
    solver_state_z_p: wp.array(dtype=float32),
    solver_config: wp.array(dtype=PADMMConfig),
    solver_penalty: wp.array(dtype=PADMMPenalty),
    solver_status: wp.array(dtype=PADMMStatus),
    # Outputs:
    solver_info_lambdas: wp.array(dtype=float32),
    solver_info_v_plus: wp.array(dtype=float32),
    solver_info_v_aug: wp.array(dtype=float32),
    solver_info_s: wp.array(dtype=float32),
    solver_info_offset: wp.array(dtype=int32),
    solver_info_num_restarts: wp.array(dtype=int32),
    solver_info_num_rho_updates: wp.array(dtype=int32),
    solver_info_a: wp.array(dtype=float32),
    solver_info_norm_s: wp.array(dtype=float32),
    solver_info_norm_x: wp.array(dtype=float32),
    solver_info_norm_y: wp.array(dtype=float32),
    solver_info_norm_z: wp.array(dtype=float32),
    solver_info_f_ccp: wp.array(dtype=float32),
    solver_info_f_ncp: wp.array(dtype=float32),
    solver_info_r_dx: wp.array(dtype=float32),
    solver_info_r_dy: wp.array(dtype=float32),
    solver_info_r_dz: wp.array(dtype=float32),
    solver_info_r_primal: wp.array(dtype=float32),
    solver_info_r_dual: wp.array(dtype=float32),
    solver_info_r_compl: wp.array(dtype=float32),
    solver_info_r_pd: wp.array(dtype=float32),
    solver_info_r_dp: wp.array(dtype=float32),
    solver_info_r_comb: wp.array(dtype=float32),
    solver_info_r_comb_ratio: wp.array(dtype=float32),
    solver_info_r_ncp_primal: wp.array(dtype=float32),
    solver_info_r_ncp_dual: wp.array(dtype=float32),
    solver_info_r_ncp_compl: wp.array(dtype=float32),
    solver_info_r_ncp_natmap: wp.array(dtype=float32),
):
    # Retrieve the thread index as the world index
    wid = wp.tid()

    # Retrieve the world-specific data
    nl = problem_nl[wid]
    nc = problem_nc[wid]
    ncts = problem_dim[wid]
    cio = problem_cio[wid]
    lcgo = problem_lcgo[wid]
    ccgo = problem_ccgo[wid]
    vio = problem_vio[wid]
    mio = problem_mio[wid]
    rio = solver_info_offset[wid]
    config = solver_config[wid]
    penalty = solver_penalty[wid]
    status = solver_status[wid]

    # Retrieve parameters
    iter = status.iterations - 1

    # Compute additional info
    njc = ncts - (nl + 3 * nc)

    # Compute total diagonal regularization applied to the Delassus matrix
    sigma = config.eta + penalty.rho

    # Compute and store the norms of the current solution state
    norm_s = compute_l2_norm(ncts, vio, solver_state_s)
    norm_x = compute_l2_norm(ncts, vio, solver_state_x)
    norm_y = compute_l2_norm(ncts, vio, solver_state_y)
    norm_z = compute_l2_norm(ncts, vio, solver_state_z)

    # Compute (division safe) residual ratios
    r_pd = status.r_p / (status.r_d + FLOAT32_EPS)
    r_dp = status.r_d / (status.r_p + FLOAT32_EPS)

    # Compute the post-event constraint-space velocity from the current solution: v_plus = v_f + D @ lambda
    compute_cwise_vec_mul(ncts, vio, problem_P, solver_state_y, solver_info_lambdas)

    # Compute the post-event constraint-space velocity from the current solution: v_plus = v_f + D @ lambda
    compute_gemv(ncts, vio, mio, sigma, problem_P, problem_D, solver_state_y, problem_v_f, solver_info_v_plus)

    # Compute the De Saxce correction for each contact as: s = G(v_plus)
    compute_desaxce_corrections(nc, cio, vio, ccgo, problem_mu, solver_info_v_plus, solver_info_s)

    # Compute the CCP optimization objective as: f_ccp = 0.5 * lambda.dot(v_plus + v_f)
    f_ccp = 0.5 * compute_double_dot_product(ncts, vio, solver_info_lambdas, solver_info_v_plus, problem_v_f)

    # Compute the NCP optimization objective as:  f_ncp = f_ccp + lambda.dot(s)
    f_ncp = compute_dot_product(ncts, vio, solver_info_lambdas, solver_info_s)
    f_ncp += f_ccp

    # Compute the augmented post-event constraint-space velocity as: v_aug = v_plus + s
    compute_vector_sum(ncts, vio, solver_info_v_plus, solver_info_s, solver_info_v_aug)

    # Compute the NCP primal residual as: r_p := || lambda - proj_K(lambda) ||_inf
    r_ncp_p = compute_ncp_primal_residual(nl, nc, vio, lcgo, ccgo, cio, problem_mu, solver_info_lambdas)

    # Compute the NCP dual residual as: r_d := || v_plus + s - proj_dual_K(v_plus + s)  ||_inf
    r_ncp_d = compute_ncp_dual_residual(njc, nl, nc, vio, lcgo, ccgo, cio, problem_mu, solver_info_v_aug)

    # Compute the NCP complementarity (lambda _|_ (v_plus + s)) residual as r_c := || lambda.dot(v_plus + s) ||_inf
    r_ncp_c = compute_ncp_complementarity_residual(nl, nc, vio, lcgo, ccgo, solver_info_v_aug, solver_info_lambdas)

    # Compute the natural-map residuals as: r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf
    r_ncp_natmap = compute_ncp_natural_map_residual(
        nl, nc, vio, lcgo, ccgo, cio, problem_mu, solver_info_v_aug, solver_info_lambdas
    )

    # Compute the iterate residual as: r_iter := || y - y_p ||_inf
    r_dx = compute_preconditioned_iterate_residual(ncts, vio, problem_P, solver_state_x, solver_state_x_p)
    r_dy = compute_preconditioned_iterate_residual(ncts, vio, problem_P, solver_state_y, solver_state_y_p)
    r_dz = compute_inverse_preconditioned_iterate_residual(ncts, vio, problem_P, solver_state_z, solver_state_z_p)

    # Compute index offset for the info of the current iteration
    iio = rio + iter

    # Store the convergence information in the solver info arrays
    solver_info_num_restarts[iio] = status.num_restarts
    solver_info_num_rho_updates[iio] = penalty.num_updates
    solver_info_a[iio] = solver_state_a[wid]
    solver_info_norm_s[iio] = norm_s
    solver_info_norm_x[iio] = norm_x
    solver_info_norm_y[iio] = norm_y
    solver_info_norm_z[iio] = norm_z
    solver_info_r_dx[iio] = r_dx
    solver_info_r_dy[iio] = r_dy
    solver_info_r_dz[iio] = r_dz
    solver_info_r_primal[iio] = status.r_p
    solver_info_r_dual[iio] = status.r_d
    solver_info_r_compl[iio] = status.r_c
    solver_info_r_pd[iio] = r_pd
    solver_info_r_dp[iio] = r_dp
    solver_info_r_comb[iio] = status.r_a
    solver_info_r_comb_ratio[iio] = status.r_a / (status.r_a_pp)
    solver_info_r_ncp_primal[iio] = r_ncp_p
    solver_info_r_ncp_dual[iio] = r_ncp_d
    solver_info_r_ncp_compl[iio] = r_ncp_c
    solver_info_r_ncp_natmap[iio] = r_ncp_natmap
    solver_info_f_ccp[iio] = f_ccp
    solver_info_f_ncp[iio] = f_ncp


@wp.kernel
def _apply_dual_preconditioner_to_state(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_P: wp.array(dtype=float32),
    # Outputs:
    solver_x: wp.array(dtype=float32),
    solver_y: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts:
        return

    # Retrieve the vector index offset of the world
    vio = problem_vio[wid]

    # Compute the global index of the vector entry
    v_i = vio + tid

    # Retrieve the i-th entries of the target vectors
    x_i = solver_x[v_i]
    y_i = solver_y[v_i]
    z_i = solver_z[v_i]

    # Retrieve the i-th entry of the diagonal preconditioner
    P_i = problem_P[v_i]

    # Store the preconditioned i-th entry of the vector
    solver_x[v_i] = P_i * x_i
    solver_y[v_i] = P_i * y_i
    solver_z[v_i] = (1.0 / P_i) * z_i


@wp.kernel
def _compute_final_desaxce_correction(
    problem_nc: wp.array(dtype=int32),
    problem_cio: wp.array(dtype=int32),
    problem_ccgo: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    problem_mu: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
    # Outputs:
    solver_s: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, cid = wp.tid()

    # Retrieve the number of contact active in the world
    nc = problem_nc[wid]

    # Retrieve the limit constraint group offset of the world
    ccgo = problem_ccgo[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if cid >= nc:
        return

    # Retrieve the index offset of the vector block of the world
    cio = problem_cio[wid]

    # Retrieve the index offset of the vector block of the world
    vio = problem_vio[wid]

    # Compute the vector index offset of the corresponding contact constraint
    ccio_k = vio + ccgo + 3 * cid

    # Compute the norm of the tangential components
    vtx = solver_z[ccio_k]
    vty = solver_z[ccio_k + 1]
    vt_norm = wp.sqrt(vtx * vtx + vty * vty)

    # Store De Saxce correction for this block
    solver_s[ccio_k] = 0.0
    solver_s[ccio_k + 1] = 0.0
    solver_s[ccio_k + 2] = problem_mu[cio + cid] * vt_norm


# Update solution vectors
@wp.kernel
def _compute_solution_vectors(
    # Inputs:
    problem_dim: wp.array(dtype=int32),
    problem_vio: wp.array(dtype=int32),
    solver_s: wp.array(dtype=float32),
    solver_y: wp.array(dtype=float32),
    solver_z: wp.array(dtype=float32),
    # Outputs:
    solver_v_plus: wp.array(dtype=float32),
    solver_lambdas: wp.array(dtype=float32),
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the total number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size or if the solver has already converged
    if tid >= ncts:
        return

    # Retrieve the index offset of the vector block of the world
    vector_offset = problem_vio[wid]

    # Compute the index offset of the vector block of the world
    thread_offset = vector_offset + tid

    # Retrieve the solver state
    z = solver_z[thread_offset]
    s = solver_s[thread_offset]
    y = solver_y[thread_offset]

    # Update constraint velocity: v_plus = z - s;
    solver_v_plus[thread_offset] = z - s

    # Update constraint reactions: lambda = y
    solver_lambdas[thread_offset] = y


###
# Interfaces
###


class PADMMSolver:
    """
    The Proximal-ADMM (PADMM) forward dynamics solver for constrained rigid multi-body systems.

    This solver implements the Proximal-ADMM algorithm to solve the Lagrange dual of the
    constrained forward dynamics problem in constraint reactions (i.e. impulses) and
    post-event constraint-space velocities.

    Notes:
    - The solver is designed to work with the DualProblem formulation.
    - The solver operates on the Lagrange dual of the constrained forward dynamics problem.
    - The solver is based on the Proximal-ADMM algorithm, which introduces a proximal regularization term
    - The solver supports multiple penalty update methods, including fixed, linear, and spectral updates.
    - The solver can be configured with various tolerances, penalty parameters, and other settings.
    """

    @staticmethod
    def _check_settings(
        model: Model | None = None, settings: list[PADMMSettings] | PADMMSettings | None = None
    ) -> list[PADMMSettings]:
        # If no settings are provided, use defaults
        if settings is None:
            # If no model is provided, use a single default settings object
            if model is None:
                settings = [PADMMSettings()]

            # If a model is provided, create a list of default settings
            # objects based on the number of worlds in the model
            else:
                num_worlds = model.info.num_worlds
                settings = [PADMMSettings()] * num_worlds

        # If a single settings object is provided, convert it to a list
        elif isinstance(settings, PADMMSettings):
            settings = [settings] * (model.info.num_worlds if model else 1)

        # If a list of settings is provided, ensure it matches the number
        # of worlds and that all settings are instances of PADMMSettings
        elif isinstance(settings, list):
            if model is not None and len(settings) != model.info.num_worlds:
                raise ValueError(f"Expected {model.info.num_worlds} settings, got {len(settings)}")
            if not all(isinstance(s, PADMMSettings) for s in settings):
                raise TypeError("All settings must be instances of PADMMSettings")

        # Return the validated settings
        return settings

    def __init__(
        self,
        model: Model | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        settings: list[PADMMSettings] | PADMMSettings | None = None,
        use_acceleration: bool = True,
        collect_info: bool = False,
        device: Devicelike = None,
    ):
        """
        Initializes a PADMM solver.

        If a model is provided, it will perform all necessary memory allocations on the
        target device, otherwise the user must call `allocate()` before using the solver.

        Args:
            model (Model | None): The model for which to allocate the solver data.
            limits (Limits | None): The limits container associated with the model.
            contacts (Contacts | None): The contacts container associated with the model.
            settings (list[PADMMSettings] | PADMMSettings | None): The solver settings to use.
            use_acceleration (bool): Set to `True` to enable Nesterov acceleration.
            collect_info (bool): Set to `True` to enable collection of solver convergence info.\n
                This setting is intended only for analysis and debugging purposes, as it
                will increase memory consumption and reduce wall-clock time.
            device (Devicelike | None): The target device on which to allocate the solver data.
        """

        # Declare the internal solver settings cache
        self._settings: list[PADMMSettings] = []
        self._use_acceleration: bool = False
        self._collect_info: bool = False

        # Declare the model size cache
        self._size: ModelSize | None = None

        # Declare the solver data container
        self._data: PADMMData | None = None

        # Cache the requested device
        self._device = device

        # Perform memory allocations if a model is provided
        if model is not None:
            self.allocate(
                model=model,
                limits=limits,
                contacts=contacts,
                settings=settings,
                use_acceleration=use_acceleration,
                collect_info=collect_info,
                device=device,
            )

    @property
    def settings(self) -> list[PADMMSettings]:
        """
        Returns the host-side cache of the solver settings.\n
        They are used to construct the warp array of type `PADMMConfig` on the target device.
        """
        return self._settings

    @property
    def size(self) -> ModelSize:
        """
        Returns the host-side cache of the solver allocation sizes.
        """
        return self._size

    @property
    def data(self) -> PADMMData:
        """
        Returns a reference to the high-level solver data container.
        """
        if self._data is None:
            raise RuntimeError("Solver data has not been allocated yet. Call `allocate()` first.")
        return self._data

    @property
    def device(self) -> Devicelike:
        """
        Returns the device on which the solver data is allocated.
        """
        return self._device

    def allocate(
        self,
        model: Model | None = None,
        limits: Limits | None = None,
        contacts: Contacts | None = None,
        settings: list[PADMMSettings] | PADMMSettings | None = None,
        use_acceleration: bool = True,
        collect_info: bool = False,
        device: Devicelike = None,
    ):
        """
        Allocates the solver data structures on the specified device.

        Args:
            model (Model | None): The model for which to allocate the solver data.
            limits (Limits | None): The limits container associated with the model.
            contacts (Contacts | None): The contacts container associated with the model.
            settings (list[PADMMSettings] | PADMMSettings | None): The solver settings to use.
            use_acceleration (bool): Set to `True` to enable Nesterov acceleration.
            collect_info (bool): Set to `True` to enable collection of solver convergence info.\n
                This setting is intended only for analysis and debugging purposes, as it
                will increase memory consumption and reduce wall-clock time.
            device (Devicelike | None): The target device on which to allocate the solver data.
        """

        # Ensure the model is valid
        if model is None:
            raise ValueError("A model of type `Model` must be provided to allocate the Delassus operator.")
        elif not isinstance(model, Model):
            raise ValueError("Invalid model provided. Must be an instance of `Model`.")

        # Ensure the limits container is valid if provided
        if limits is not None:
            if not isinstance(limits, Limits):
                raise ValueError("Invalid limits container provided. Must be an instance of `Limits`.")

        # Ensure the contacts container is valid if provided
        if contacts is not None:
            if not isinstance(contacts, Contacts):
                raise ValueError("Invalid contacts container provided. Must be an instance of `Contacts`.")

        # Cache a reference to the model size meta-data container
        self._size = model.size

        # Cache high-level solver options
        self._use_acceleration = use_acceleration
        self._collect_info = collect_info
        self._device = device

        # Cache the solver settings
        if settings is not None:
            self._settings = self._check_settings(model, settings)
        elif len(self._settings) == 0:
            self._settings = self._check_settings(model, None)

        # Compute the largest max iterations across all worlds
        # NOTE: This is needed to allocate the solver
        # info arrays if `collect_info` is enabled
        max_of_max_iters = max([s.max_iterations for s in self._settings])

        # Allocate memory in device global memory
        self._data = PADMMData(
            size=self._size, max_iters=max_of_max_iters, collect_info=self._collect_info, device=self._device
        )

        # Write algorithm configs into device memory
        configs = [s.to_config() for s in self._settings]
        with wp.ScopedDevice(self._device):
            self._data.config = wp.array(configs, dtype=PADMMConfig)

    # TODO(team): We should think about how to introduce an optional systematic warm-starting mechanism here

    def coldstart(self):
        """
        Resets the internal solver state to perform a cold-start solve.\n
        This method sets all solver state variables to zeros.
        """

        # Initialize state arrays to zero
        self._data.state.reset()

    def initialize(self):
        """
        Launches a kernel to initialize the internal solver state before starting a new solve.\n
        The kernel is parallelized over the number of worlds.
        """

        # Initialize solver status and penalty parameters from the set configurations
        wp.launch(
            kernel=_initialize_solver,
            dim=self._size.num_worlds,
            inputs=[self._data.config, self._data.status, self._data.penalty, self._data.state.a_p],
        )

        # Initialize the global while condition flag
        # NOTE: We use a single-element array that is initialized
        # to number of worlds and decremented by each world that
        # converges or reaches the maximum number of iterations
        self._data.state.done.fill_(self._size.num_worlds)

    def update_regularization(self, problem: DualProblem):
        """
        Updates the diagonal regularization of the lhs matrix with the proximal regularization terms.\n
        For `DualProblem` solves, the lhs matrix corresponds to the Delassus matrix.\n
        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Update the proximal regularization term in the Delassus matrix
        wp.launch(
            kernel=_update_delassus_proximal_regularization,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.mio,
                self._data.config,
                self._data.penalty,
                self._data.status,
                # Outputs:
                problem.data.D,
            ],
        )

        # Compute Choleky/LDLT factorization of the Delassus matrix
        problem._delassus.compute(reset_to_zero=True)

    def update_desaxce_correction(self, problem: DualProblem):
        """
        Launches a kernel to compute the De Saxce correction velocity using the previous dual variables.\n
        The kernel is parallelized over the number of worlds and the maximum number of contacts.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        wp.launch(
            kernel=_compute_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.status,
                self._data.state.z_p,
                # Outputs:
                self._data.state.s,
            ],
        )

    def update_desaxce_correction_accel(self, problem: DualProblem):
        """
        Launches a kernel to compute the De Saxce correction velocity using the accelerated dual variables.\n
        The kernel is parallelized over the number of worlds and the maximum number of contacts.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        wp.launch(
            kernel=_compute_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.status,
                self._data.state.z_hat,
                # Outputs:
                self._data.state.s,
            ],
        )

    def update_velocity_bias(self, problem: DualProblem):
        """
        Launches a kernel to compute the total bias velocity vector using the previous state variables.\n
        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        wp.launch(
            kernel=_compute_velocity_bias,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.v_f,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.s,
                self._data.state.x_p,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.v,
            ],
        )

    def update_velocity_bias_accel(self, problem: DualProblem):
        """
        Launches a kernel to compute the total bias velocity vector using the accelerated state variables.\n
        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        wp.launch(
            kernel=_compute_velocity_bias,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.v_f,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.s,
                self._data.state.x_p,
                self._data.state.y_hat,
                self._data.state.z_hat,
                # Outputs:
                self._data.state.v,
            ],
        )

    def update_unconstrained_solution(self, problem: DualProblem):
        """
        Launches a kernel to solve the unconstrained sub-problem for the primal variables.\n
        For `DualProblem` solves, this corresponds to solving a linear system with the Delassus matrix.\n
        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # TODO: We should do this in-place
        # wp.copy(self._data.state.x, self._data.state.v)
        # problem._delassus.solve_inplace(x=self._data.state.x)
        problem._delassus.solve(v=self._data.state.v, x=self._data.state.x)

    def update_projection_argument(self, problem: DualProblem):
        """
        Launches a kernel to compute the argument for the projection operator onto the
        feasible set using the previous state variables and the unconstrained solution.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Apply over-relaxation and compute the argument to the projection operator
        wp.launch(
            kernel=_apply_overrelaxation_and_compute_projection_argument,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.x,
                self._data.state.y,
            ],
        )

    def update_projection_argument_accel(self, problem: DualProblem):
        """
        Launches a kernel to compute the argument for the projection operator onto the
        feasible set using the accelerated state variables and the unconstrained solution.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Apply over-relaxation and compute the argument to the projection operator
        wp.launch(
            kernel=_compute_projection_argument,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.penalty,
                self._data.status,
                self._data.state.z_hat,
                self._data.state.x,
                # Outputs:
                self._data.state.y,
            ],
        )

    def update_projection_to_feasible_set(self, problem: DualProblem):
        """
        Launches a kernel to project the current primal variables
        onto the feasible set defined by the constraint cone K.

        The kernel is parallelized over the number of worlds and the maximum
        number of unilateral constraints, i.e. 1D limits and 3D contacts.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Project to the feasible set defined by the cone K := R^{njd} x R_+^{nld} x K_{mu}^{nc}
        wp.launch(
            kernel=_project_to_feasible_cone,
            dim=(self._size.num_worlds, self._size.max_of_max_unilaterals),
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.cio,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.status,
                # Outputs:
                self._data.state.y,
            ],
        )

    def update_complementarity_residuals(self, problem: DualProblem):
        """
        Launches a kernel to compute the complementarity residuals from the current state variables.
        The kernel is parallelized over the number of worlds and the maximum number of unilateral constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Compute complementarity residual from the current state
        wp.launch(
            kernel=_compute_complementarity_residuals,
            dim=(self._size.num_worlds, self._size.max_of_max_unilaterals),
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.cio,
                problem.data.uio,
                problem.data.lcgo,
                problem.data.ccgo,
                self._data.status,
                self._data.state.x,
                self._data.state.z,
                # Outputs:
                self._data.residuals.r_compl,
            ],
        )

    def update_dual_variables_and_residuals(self, problem: DualProblem):
        """
        Launches a kernel to update the dual variables and compute the
        PADMM residuals from the current and previous state variables.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Update the dual variables and compute primal-dual residuals from the current state
        # NOTE: These are combined into a single kernel to reduce kernel launch overhead
        wp.launch(
            kernel=_update_dual_variables_and_compute_primal_dual_residuals,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.x,
                self._data.state.y,
                self._data.state.x_p,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.z,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_dx,
                self._data.residuals.r_dy,
                self._data.residuals.r_dz,
            ],
        )

        # Compute complementarity residual from the current state
        self.update_complementarity_residuals(problem)

    def update_dual_variables_and_residuals_accel(self, problem: DualProblem):
        """
        Launches a kernel to update the dual variables and compute the
        PADMM residuals from the current and accelerated state variables.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Update the dual variables and compute primal-dual residuals from the current state
        # NOTE: These are combined into a single kernel to reduce kernel launch overhead
        wp.launch(
            kernel=_update_dual_variables_and_compute_primal_dual_residuals,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.x,
                self._data.state.y,
                self._data.state.x_p,
                self._data.state.y_hat,
                self._data.state.z_hat,
                # Outputs:
                self._data.state.z,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_dx,
                self._data.residuals.r_dy,
                self._data.residuals.r_dz,
            ],
        )

        # Compute complementarity residual from the current state
        self.update_complementarity_residuals(problem)

    def update_convergence_check(self, problem: DualProblem):
        """
        Launches a kernel to compute the infinity-norm of the PADMM residuals
        using the current and previous state variables and check for convergence.

        The kernel is parallelized over the number of worlds.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Compute infinity-norm of all residuals and check for convergence
        wp.launch(
            kernel=_compute_infnorm_residuals_serially,
            dim=self._size.num_worlds,
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.uio,
                problem.data.dim,
                problem.data.vio,
                self._data.config,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_compl,
                # Outputs:
                self._data.state.done,
                self._data.status,
            ],
        )

    def update_convergence_check_accel(self, problem: DualProblem):
        """
        Launches a kernel to compute the infinity-norm of the PADMM residuals
        using the current and accelerated state variables and check for convergence.

        The kernel is parallelized over the number of worlds.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Compute infinity-norm of all residuals and check for convergence
        wp.launch(
            kernel=_compute_infnorm_residuals_serially_accel,
            dim=self._size.num_worlds,
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.uio,
                problem.data.dim,
                problem.data.vio,
                self._data.config,
                self._data.penalty,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_compl,
                self._data.residuals.r_dx,
                self._data.residuals.r_dy,
                self._data.residuals.r_dz,
                self._data.state.a_p,
                # Outputs:
                self._data.state.done,
                self._data.state.a,
                self._data.status,
            ],
        )

    def update_acceleration(self, problem: DualProblem):
        """
        Launches a kernel to update gradient acceleration and the accelerated state variables.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        wp.launch(
            kernel=_update_state_with_acceleration,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.status,
                self._data.state.a,
                self._data.state.y,
                self._data.state.z,
                self._data.state.a_p,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.y_hat,
                self._data.state.z_hat,
            ],
        )

    def update_solver_info(self, problem: DualProblem):
        """
        Launches a kernel to update the solver info history from the current solver data.

        The kernel is parallelized over the number of worlds.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # First reset the internal buffer arrays to zero
        # to ensure we do not accumulate values across iterations
        self.data.info.v_plus.zero_()
        self.data.info.v_aug.zero_()
        self._data.info.s.zero_()

        # Collect convergence information from the current state
        wp.launch(
            kernel=_collect_solver_convergence_info,
            dim=self._size.num_worlds,
            inputs=[
                # Inputs:
                problem.data.nl,
                problem.data.nc,
                problem.data.cio,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.dim,
                problem.data.vio,
                problem.data.mio,
                problem.data.mu,
                problem.data.v_f,
                problem.data.D,
                problem.data.P,
                self._data.state.a,
                self._data.state.s,
                self._data.state.x,
                self._data.state.x_p,
                self._data.state.y,
                self._data.state.y_p,
                self._data.state.z,
                self._data.state.z_p,
                self._data.config,
                self._data.penalty,
                self._data.status,
                # Outputs:
                self._data.info.lambdas,
                self._data.info.v_plus,
                self._data.info.v_aug,
                self._data.info.s,
                self._data.info.offsets,
                self._data.info.num_restarts,
                self._data.info.num_rho_updates,
                self._data.info.a,
                self._data.info.norm_s,
                self._data.info.norm_x,
                self._data.info.norm_y,
                self._data.info.norm_z,
                self._data.info.f_ccp,
                self._data.info.f_ncp,
                self._data.info.r_dx,
                self._data.info.r_dy,
                self._data.info.r_dz,
                self._data.info.r_primal,
                self._data.info.r_dual,
                self._data.info.r_compl,
                self._data.info.r_pd,
                self._data.info.r_dp,
                self._data.info.r_comb,
                self._data.info.r_comb_ratio,
                self._data.info.r_ncp_primal,
                self._data.info.r_ncp_dual,
                self._data.info.r_ncp_compl,
                self._data.info.r_ncp_natmap,
            ],
        )

    def update_previous_state(self):
        """
        Updates the cached previous state variables with the current.
        This function uses on-device memory copy operations.
        """
        wp.copy(self._data.state.x_p, self._data.state.x)
        wp.copy(self._data.state.y_p, self._data.state.y)
        wp.copy(self._data.state.z_p, self._data.state.z)

    def update_previous_state_accel(self):
        """
        Updates the cached previous acceleration and state variable with the current.
        This function uses on-device memory copy operations.
        """
        wp.copy(self._data.state.a_p, self._data.state.a)

        # Cache previous state variables
        self.update_previous_state()

    def update_solution(self, problem: DualProblem):
        """
        Launches a set of kernels to extract and post-process
        the final solution from the internal PADMM state data.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Apply the dual preconditioner to recover the final PADMM state
        wp.launch(
            kernel=_apply_dual_preconditioner_to_state,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                # Outputs:
                self._data.state.x,
                self._data.state.y,
                self._data.state.z,
            ],
        )

        # Update the De Saxce correction from terminal PADMM dual variables
        wp.launch(
            kernel=_compute_final_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.state.z,
                # Outputs:
                self._data.state.s,
            ],
        )

        # Update solution vectors from the terminal PADMM state
        wp.launch(
            kernel=_compute_solution_vectors,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.state.s,
                self._data.state.y,
                self._data.state.z,
                # Outputs:
                self._data.solution.v_plus,
                self._data.solution.lambdas,
            ],
        )

    def step(self, problem: DualProblem):
        """
        Performs a single PADMM solver iteration.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Compute De Saxce correction from the previous dual variables
        self.update_desaxce_correction(problem)

        # Compute the total velocity bias, i.e. rhs vector of the unconstrained linear system
        self.update_velocity_bias(problem)

        # Compute the unconstrained solution and store in the primal variables
        self.update_unconstrained_solution(problem)

        # Compute the argument to the projection operator with over-relaxation
        self.update_projection_argument(problem)

        # Project the over-relaxed primal variables to the feasible set
        self.update_projection_to_feasible_set(problem)

        # Update the dual variables and compute residuals from the current state
        self.update_dual_variables_and_residuals(problem)

        # Compute infinity-norm of all residuals and check for convergence
        self.update_convergence_check(problem)

        # Optionally record internal solver info
        if self._collect_info:
            self.update_solver_info(problem)

        # Update caches of previous state variables
        self.update_previous_state()

    def step_accel(self, problem: DualProblem):
        """
        Performs a single PADMM solver iteration with Nesterov acceleration.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Compute De Saxce correction from the previous dual variables
        self.update_desaxce_correction_accel(problem)

        # Compute the total velocity bias, i.e. rhs vector of the unconstrained linear system
        self.update_velocity_bias_accel(problem)

        # Compute the unconstrained solution and store in the primal variables
        self.update_unconstrained_solution(problem)

        # Project the over-relaxed primal variables to the feasible set
        self.update_projection_argument_accel(problem)

        # Project the over-relaxed primal variables to the feasible set
        self.update_projection_to_feasible_set(problem)

        # Update the dual variables and compute residuals from the current state
        self.update_dual_variables_and_residuals_accel(problem)

        # Compute infinity-norm of all residuals and check for convergence
        self.update_convergence_check_accel(problem)

        # Optionally update Nesterov acceleration states from the current iteration
        self.update_acceleration(problem)

        # Optionally record internal solver info
        if self._collect_info:
            self.update_solver_info(problem)

        # Update caches of previous state variables
        self.update_previous_state_accel()

    def solve(self, problem: DualProblem):
        """
        Solves the given dual problem using PADMM.

        Args:
            problem (DualProblem): The dual forward dynamics problem to be solved.
        """
        # Cold-start the solver state
        self.coldstart()

        # Initialize the solver status and ALM penalty parameter
        self.initialize()

        # Add the diagonal proximal regularization to the Delassus matrix
        # D_{eta,rho} := D + (eta + rho) * I_{ncts}
        self.update_regularization(problem)

        # Reset the solver info to zero if collection is enabled
        if self._collect_info:
            self._data.info.zero()

        # Iterate until convergence or maximum number of iterations is reached
        if self._use_acceleration:
            wp.capture_while(self._data.state.done, while_body=self.step_accel, problem=problem)
        else:
            wp.capture_while(self._data.state.done, while_body=self.step, problem=problem)

        # Update the final solution from the terminal PADMM state
        self.update_solution(problem)
