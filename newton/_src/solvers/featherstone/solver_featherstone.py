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

import warp as wp

from ...core.types import Devicelike, override
from ...sim import Contacts, Control, Model, State, eval_fk
from ..semi_implicit.kernels_contact import (
    eval_body_contact,
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
    eval_triangle_contact_forces,
)
from ..semi_implicit.kernels_muscle import (
    eval_muscle_forces,
)
from ..semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)
from ..solver import SolverBase
from .kernels_body import (
    compute_com_transforms,
    compute_spatial_inertia,
    create_inertia_matrix_cholesky_kernel,
    create_inertia_matrix_kernel,
    eval_dense_cholesky_batched,
    eval_dense_gemm_batched,
    eval_dense_solve_batched,
    eval_rigid_fk,
    eval_rigid_id,
    eval_rigid_jacobian,
    eval_rigid_mass,
    eval_rigid_tau,
    integrate_generalized_joints,
)


class StateFeatherstone:
    def __init__(self):
        self.M: wp.array | None = None
        self.J: wp.array | None = None
        self.P: wp.array | None = None
        self.H: wp.array | None = None
        self.L: wp.array | None = None

        self.joint_qdd: wp.array | None = None
        self.joint_tau: wp.array | None = None

        self.joint_solve_tmp: wp.array | None = None
        self.joint_S_s: wp.array | None = None

        self.body_q_com: wp.array | None = None
        self.body_I_s: wp.array | None = None
        self.body_v_s: wp.array | None = None
        self.body_a_s: wp.array | None = None
        self.body_f_s: wp.array | None = None
        self.body_ft_s: wp.array | None = None


class ModelFeatherstone:
    def __init__(self, requires_grad: bool = False):
        self.requires_grad = requires_grad
        self.device = wp.get_device()

        self.J_size: int = 0
        self.M_size: int = 0
        self.H_size: int = 0

        self.tile_joint_count: int = 0
        self.tile_dof_count: int = 0

        self.articulation_J_start: wp.array | None = None
        self.articulation_M_start: wp.array | None = None
        self.articulation_H_start: wp.array | None = None

        self.articulation_M_rows: wp.array | None = None
        self.articulation_H_rows: wp.array | None = None
        self.articulation_J_rows: wp.array | None = None
        self.articulation_J_cols: wp.array | None = None

        self.articulation_dof_start: wp.array | None = None
        self.articulation_coord_start: wp.array | None = None

        self.M: wp.array | None = None
        self.J: wp.array | None = None
        self.P: wp.array | None = None
        self.H: wp.array | None = None
        self.L: wp.array | None = None

        self.joint_qdd: wp.array | None = None
        self.joint_tau: wp.array | None = None

        self.joint_solve_tmp: wp.array | None = None
        self.joint_S_s: wp.array | None = None

        self.body_q_com: wp.array | None = None
        self.body_I_s: wp.array | None = None
        self.body_v_s: wp.array | None = None
        self.body_a_s: wp.array | None = None
        self.body_f_s: wp.array | None = None
        self.body_ft_s: wp.array | None = None

    def state_custom(self, model: Model, requires_grad: bool | None = None) -> StateFeatherstone:
        _s = StateFeatherstone()
        if requires_grad is None:
            requires_grad = self.requires_grad

        if model.joint_count:
            _s.M = wp.zeros_like(self.M, requires_grad=requires_grad)
            _s.J = wp.zeros_like(self.J, requires_grad=requires_grad)
            _s.P = wp.zeros_like(self.P, requires_grad=requires_grad)
            _s.H = wp.zeros_like(self.H, requires_grad=requires_grad)
            _s.L = wp.zeros_like(self.L)

        if model.body_count:
            _s.joint_qdd = wp.zeros_like(self.joint_qdd, requires_grad=requires_grad)
            _s.joint_tau = wp.zeros_like(self.joint_tau, requires_grad=requires_grad)
            if requires_grad:
                _s.joint_solve_tmp = wp.zeros_like(self.joint_solve_tmp, requires_grad=requires_grad)
            else:
                _s.joint_solve_tmp = None
            _s.joint_S_s = wp.zeros_like(self.joint_S_s, requires_grad=requires_grad)

            _s.body_q_com = wp.zeros_like(self.body_q_com, requires_grad=requires_grad)
            _s.body_I_s = wp.zeros_like(self.body_I_s, requires_grad=requires_grad)
            _s.body_v_s = wp.zeros_like(self.body_v_s, requires_grad=requires_grad)
            _s.body_a_s = wp.zeros_like(self.body_a_s, requires_grad=requires_grad)
            _s.body_f_s = wp.zeros_like(self.body_f_s, requires_grad=requires_grad)
            _s.body_ft_s = wp.zeros_like(self.body_ft_s, requires_grad=requires_grad)

        return _s


class ModelBuilderFeatherstone:
    def __init__(self, use_tile_gemm: bool = False):
        self.use_tile_gemm = use_tile_gemm

    def _compute_articulation_indices(self, model: Model, _model: ModelFeatherstone, use_tile_gemm: bool = False):
        device = model.device
        with wp.ScopedDevice(device):
            # calculate total size and offsets of Jacobian and mass matrices for entire system
            if model.joint_count:
                _model.J_size = 0
                _model.M_size = 0
                _model.H_size = 0

                articulation_J_start = []
                articulation_M_start = []
                articulation_H_start = []

                articulation_M_rows = []
                articulation_H_rows = []
                articulation_J_rows = []
                articulation_J_cols = []

                articulation_dof_start = []
                articulation_coord_start = []

                articulation_start = model.articulation_start.numpy()
                joint_q_start = model.joint_q_start.numpy()
                joint_qd_start = model.joint_qd_start.numpy()

                for i in range(model.articulation_count):
                    first_joint = articulation_start[i]
                    last_joint = articulation_start[i + 1]

                    first_coord = joint_q_start[first_joint]

                    first_dof = joint_qd_start[first_joint]
                    last_dof = joint_qd_start[last_joint]

                    joint_count = last_joint - first_joint
                    dof_count = last_dof - first_dof

                    articulation_J_start.append(_model.J_size)
                    articulation_M_start.append(_model.M_size)
                    articulation_H_start.append(_model.H_size)
                    articulation_dof_start.append(first_dof)
                    articulation_coord_start.append(first_coord)

                    # bit of data duplication here, but will leave it as such for clarity
                    articulation_M_rows.append(joint_count * 6)
                    articulation_H_rows.append(dof_count)
                    articulation_J_rows.append(joint_count * 6)
                    articulation_J_cols.append(dof_count)

                    if use_tile_gemm:
                        # store the joint and dof count assuming all
                        # articulations have the same structure
                        _model.tile_joint_count = int(joint_count)
                        _model.tile_dof_count = int(dof_count)

                    _model.J_size += 6 * joint_count * dof_count
                    _model.M_size += 6 * joint_count * 6 * joint_count
                    _model.H_size += dof_count * dof_count

                # matrix offsets for batched gemm
                _model.articulation_J_start = wp.array(articulation_J_start, dtype=wp.int32)
                _model.articulation_M_start = wp.array(articulation_M_start, dtype=wp.int32)
                _model.articulation_H_start = wp.array(articulation_H_start, dtype=wp.int32)

                _model.articulation_M_rows = wp.array(articulation_M_rows, dtype=wp.int32)
                _model.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32)
                _model.articulation_J_rows = wp.array(articulation_J_rows, dtype=wp.int32)
                _model.articulation_J_cols = wp.array(articulation_J_cols, dtype=wp.int32)

                _model.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32)
                _model.articulation_coord_start = wp.array(articulation_coord_start, dtype=wp.int32)

    def finalize_custom(self, model: Model, device: Devicelike | None = None, requires_grad: bool = False):
        with wp.ScopedDevice(device):
            _model = ModelFeatherstone(requires_grad=model.requires_grad)

            self._compute_articulation_indices(model, _model, use_tile_gemm=self.use_tile_gemm)

            if model.body_count:
                _model.body_I_m = wp.empty((model.body_count,), dtype=wp.spatial_matrix, requires_grad=requires_grad)
                wp.launch(
                    compute_spatial_inertia,
                    model.body_count,
                    inputs=[model.body_inertia, model.body_mass],
                    outputs=[_model.body_I_m],
                    device=model.device,
                )
                _model.body_X_com = wp.empty((model.body_count,), dtype=wp.transform, requires_grad=requires_grad)
                wp.launch(
                    compute_com_transforms,
                    model.body_count,
                    inputs=[model.body_com],
                    outputs=[_model.body_X_com],
                    device=model.device,
                )

            # allocate mass, Jacobian matrices
            if model.joint_count:
                # system matrices
                _model.M = wp.zeros((_model.M_size,), dtype=wp.float32, requires_grad=requires_grad)
                _model.J = wp.zeros((_model.J_size,), dtype=wp.float32, requires_grad=requires_grad)
                _model.P = wp.empty_like(_model.J, requires_grad=requires_grad)
                _model.H = wp.empty((_model.H_size,), dtype=wp.float32, requires_grad=requires_grad)
                # zero since only upper triangle is set which can trigger NaN detection
                _model.L = wp.zeros_like(_model.H)

            # allocate other auxiliary variables that vary with state
            if model.body_count:
                # joints
                _model.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
                _model.joint_tau = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
                if requires_grad:
                    # used in the custom grad implementation of eval_dense_solve_batched
                    _model.joint_solve_tmp = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
                else:
                    _model.joint_solve_tmp = None
                _model.joint_S_s = wp.empty(
                    (model.joint_dof_count,), dtype=wp.spatial_vector, requires_grad=requires_grad
                )

                # derived rigid body data (maximal coordinates)
                _model.body_q_com = wp.empty_like(model.body_q, requires_grad=requires_grad)
                _model.body_I_s = wp.empty((model.body_count,), dtype=wp.spatial_matrix, requires_grad=requires_grad)
                _model.body_v_s = wp.empty((model.body_count,), dtype=wp.spatial_vector, requires_grad=requires_grad)
                _model.body_a_s = wp.empty((model.body_count,), dtype=wp.spatial_vector, requires_grad=requires_grad)
                _model.body_f_s = wp.zeros((model.body_count,), dtype=wp.spatial_vector, requires_grad=requires_grad)
                _model.body_ft_s = wp.zeros((model.body_count,), dtype=wp.spatial_vector, requires_grad=requires_grad)

            return _model


class SolverFeatherstone(SolverBase):
    """A semi-implicit integrator using symplectic Euler that operates
    on reduced (also called generalized) coordinates to simulate articulated rigid body dynamics
    based on Featherstone's composite rigid body algorithm (CRBA).

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Instead of maximal coordinates :attr:`~newton.State.body_q` (rigid body positions) and :attr:`~newton.State.body_qd`
    (rigid body velocities) as is the case in :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`,
    :class:`~newton.solvers.SolverFeatherstone` uses :attr:`~newton.State.joint_q` and :attr:`~newton.State.joint_qd` to represent
    the positions and velocities of joints without allowing any redundant degrees of freedom.

    After constructing :class:`~newton.Model` and :class:`~newton.State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Note:
        Unlike :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverFeatherstone`
        does not simulate rigid bodies with nonzero mass as floating bodies if they are not connected through any joints.
        Floating-base systems require an explicit free joint with which the body is connected to the world,
        see :meth:`newton.ModelBuilder.add_joint_free`.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    This solver uses the routines from :class:`~newton.solvers.SolverSemiImplicit` to simulate particles, cloth, and soft bodies.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverFeatherstone(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    def __init__(
        self,
        model: Model,
        angular_damping: float = 0.05,
        update_mass_matrix_interval: int = 1,
        friction_smoothing: float = 1.0,
        use_tile_gemm: bool = False,
        fuse_cholesky: bool = True,
        enable_tri_contact: bool = True,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            angular_damping (float, optional): Angular damping factor. Defaults to 0.05.
            update_mass_matrix_interval (int, optional): How often to update the mass matrix (every n-th time the :meth:`step` function gets called). Defaults to 1.
            friction_smoothing (float, optional): The delta value for the Huber norm (see :func:`warp.math.norm_huber`) used for the friction velocity normalization. Defaults to 1.0.
            use_tile_gemm (bool, optional): Whether to use operators from Warp's Tile API to solve for joint accelerations. Defaults to False.
            fuse_cholesky (bool, optional): Whether to fuse the Cholesky decomposition into the inertia matrix evaluation kernel when using the Tile API. Only used if `use_tile_gemm` is true. Defaults to True.
            enable_tri_contact (bool, optional): Enable triangle contact. Defaults to True.
        """
        super().__init__(model)

        self.angular_damping = angular_damping
        self.update_mass_matrix_interval = update_mass_matrix_interval
        self.friction_smoothing = friction_smoothing
        self.use_tile_gemm = use_tile_gemm
        self.fuse_cholesky = fuse_cholesky
        self.enable_tri_contact = enable_tri_contact

        self._step = 0

        # custom model attributes for Featherstone
        _builder = ModelBuilderFeatherstone(use_tile_gemm=self.use_tile_gemm)
        _model = _builder.finalize_custom(model, device=model.device, requires_grad=model.requires_grad)
        model.featherstone = _model

        if self.use_tile_gemm:
            # create a custom kernel to evaluate the system matrix for this type
            if self.fuse_cholesky:
                self.eval_inertia_matrix_cholesky_kernel = create_inertia_matrix_cholesky_kernel(
                    _model.tile_joint_count, _model.tile_dof_count
                )
            else:
                self.eval_inertia_matrix_kernel = create_inertia_matrix_kernel(
                    _model.tile_joint_count, _model.tile_dof_count
                )

            # ensure matrix is reloaded since otherwise an unload can happen during graph capture
            # todo: should not be necessary?
            wp.load_module(device=wp.get_device())

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ):
        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            model = self.model
            _model = model.featherstone

            # optionally create dynamical auxiliary variables
            requires_grad = state_in.requires_grad
            if requires_grad:
                if not hasattr(state_in, "featherstone"):
                    state_in.featherstone = _model.state_custom(model, requires_grad)
                if not hasattr(state_out, "featherstone"):
                    state_out.featherstone = _model.state_custom(model, requires_grad)
                _state_in = state_in.featherstone
                _state_out = state_out.featherstone
            else:
                _state_in = _model
                _state_out = _model

            if control is None:
                control = model.control(clone_variables=False)

            # damped springs
            eval_spring_forces(model, state_in, particle_f)

            # triangle elastic and lift/drag forces
            eval_triangle_forces(model, state_in, control, particle_f)

            # triangle bending
            eval_bending_forces(model, state_in, particle_f)

            # tetrahedral FEM
            eval_tetrahedra_forces(model, state_in, control, particle_f)

            # muscles
            if False:
                eval_muscle_forces(model, state_in, control, body_f)

            # particle-particle interactions
            eval_particle_contact_forces(model, state_in, particle_f)

            # triangle/triangle contacts
            if self.enable_tri_contact:
                eval_triangle_contact_forces(model, state_in, particle_f)

            # particle shape contact
            eval_particle_body_contact_forces(model, state_in, contacts, particle_f, body_f, body_f_in_world_frame=True)

            # ----------------------------
            # articulations

            if model.joint_count:
                # evaluate body transforms
                wp.launch(
                    eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        model.joint_X_p,
                        model.joint_X_c,
                        _model.body_X_com,
                        model.joint_axis,
                        model.joint_dof_dim,
                    ],
                    outputs=[state_in.body_q, _state_in.body_q_com],
                    device=model.device,
                )

                # evaluate joint inertias, motion vectors, and forces
                _state_in.body_f_s.zero_()

                wp.launch(
                    eval_rigid_id,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_qd_start,
                        state_in.joint_qd,
                        model.joint_axis,
                        model.joint_dof_dim,
                        _model.body_I_m,
                        state_in.body_q,
                        _state_in.body_q_com,
                        model.joint_X_p,
                        model.gravity,
                    ],
                    outputs=[
                        _state_in.joint_S_s,
                        _state_in.body_I_s,
                        _state_in.body_v_s,
                        _state_in.body_f_s,
                        _state_in.body_a_s,
                    ],
                    device=model.device,
                )

                if contacts is not None and contacts.rigid_contact_max:
                    wp.launch(
                        kernel=eval_body_contact,
                        dim=contacts.rigid_contact_max,
                        inputs=[
                            state_in.body_q,
                            _state_in.body_v_s,
                            model.body_com,
                            model.shape_material_ke,
                            model.shape_material_kd,
                            model.shape_material_kf,
                            model.shape_material_ka,
                            model.shape_material_mu,
                            model.shape_body,
                            contacts.rigid_contact_count,
                            contacts.rigid_contact_point0,
                            contacts.rigid_contact_point1,
                            contacts.rigid_contact_normal,
                            contacts.rigid_contact_shape0,
                            contacts.rigid_contact_shape1,
                            contacts.rigid_contact_thickness0,
                            contacts.rigid_contact_thickness1,
                            True,
                            self.friction_smoothing,
                        ],
                        outputs=[body_f],
                        device=model.device,
                    )

                if model.articulation_count:
                    # evaluate joint torques
                    _state_in.body_ft_s.zero_()
                    _state_in.joint_tau.zero_()
                    wp.launch(
                        eval_rigid_tau,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_q_start,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_dof_mode,
                            state_in.joint_q,
                            state_in.joint_qd,
                            control.joint_f,
                            control.joint_target,
                            model.joint_target_ke,
                            model.joint_target_kd,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            model.joint_limit_ke,
                            model.joint_limit_kd,
                            _state_in.joint_S_s,
                            _state_in.body_f_s,
                            body_f,
                        ],
                        outputs=[
                            _state_in.body_ft_s,
                            _state_in.joint_tau,
                        ],
                        device=model.device,
                    )

                    if self._step % self.update_mass_matrix_interval == 0:
                        # build J
                        wp.launch(
                            eval_rigid_jacobian,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                _model.articulation_J_start,
                                model.joint_ancestor,
                                model.joint_qd_start,
                                _state_in.joint_S_s,
                            ],
                            outputs=[_state_out.J],
                            device=model.device,
                        )

                        # build M
                        wp.launch(
                            eval_rigid_mass,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                _model.articulation_M_start,
                                _state_in.body_I_s,
                            ],
                            outputs=[_state_out.M],
                            device=model.device,
                        )

                        if self.use_tile_gemm:
                            # reshape arrays
                            M_tiled = _state_out.M.reshape(
                                (-1, 6 * _model.tile_joint_count, 6 * _model.tile_joint_count)
                            )
                            J_tiled = _state_out.J.reshape((-1, 6 * _model.tile_joint_count, _model.tile_dof_count))
                            H_tiled = _state_out.H.reshape((-1, _model.tile_dof_count, _model.tile_dof_count))
                            L_tiled = _state_out.L.reshape((-1, _model.tile_dof_count, _model.tile_dof_count))
                            R_tiled = model.joint_armature.reshape((-1, _model.tile_dof_count))
                            assert H_tiled.shape == (model.articulation_count, 18, 18)
                            assert L_tiled.shape == (model.articulation_count, 18, 18)
                            assert R_tiled.shape == (model.articulation_count, 18)

                            if self.fuse_cholesky:
                                wp.launch_tiled(
                                    self.eval_inertia_matrix_cholesky_kernel,
                                    dim=model.articulation_count,
                                    inputs=[J_tiled, M_tiled, R_tiled],
                                    outputs=[H_tiled, L_tiled],
                                    device=model.device,
                                    block_dim=64,
                                )

                            else:
                                wp.launch_tiled(
                                    self.eval_inertia_matrix_kernel,
                                    dim=model.articulation_count,
                                    inputs=[J_tiled, M_tiled],
                                    outputs=[H_tiled],
                                    device=model.device,
                                    block_dim=256,
                                )

                                wp.launch(
                                    eval_dense_cholesky_batched,
                                    dim=model.articulation_count,
                                    inputs=[
                                        _model.articulation_H_start,
                                        _model.articulation_H_rows,
                                        _state_out.H,
                                        model.joint_armature,
                                    ],
                                    outputs=[_state_out.L],
                                    device=model.device,
                                )

                            # import numpy as np
                            # J = J_tiled.numpy()
                            # M = M_tiled.numpy()
                            # R = R_tiled.numpy()
                            # for i in range(model.articulation_count):
                            #     r = R[i,:,0]
                            #     H = J[i].T @ M[i] @ J[i]
                            #     L = np.linalg.cholesky(H + np.diag(r))
                            #     np.testing.assert_allclose(H, H_tiled.numpy()[i], rtol=1e-2, atol=1e-2)
                            #     np.testing.assert_allclose(L, L_tiled.numpy()[i], rtol=1e-1, atol=1e-1)

                        else:
                            # form P = M*J
                            wp.launch(
                                eval_dense_gemm_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    _model.articulation_M_rows,
                                    _model.articulation_J_cols,
                                    _model.articulation_J_rows,
                                    False,
                                    False,
                                    _model.articulation_M_start,
                                    _model.articulation_J_start,
                                    # P start is the same as J start since it has the same dims as J
                                    _model.articulation_J_start,
                                    _state_out.M,
                                    _state_out.J,
                                ],
                                outputs=[_state_out.P],
                                device=model.device,
                            )

                            # form H = J^T*P
                            wp.launch(
                                eval_dense_gemm_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    _model.articulation_J_cols,
                                    _model.articulation_J_cols,
                                    # P rows is the same as J rows
                                    _model.articulation_J_rows,
                                    True,
                                    False,
                                    _model.articulation_J_start,
                                    # P start is the same as J start since it has the same dims as J
                                    _model.articulation_J_start,
                                    _model.articulation_H_start,
                                    _model.J,
                                    _model.P,
                                ],
                                outputs=[_model.H],
                                device=model.device,
                            )

                            # compute decomposition
                            wp.launch(
                                eval_dense_cholesky_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    _model.articulation_H_start,
                                    _model.articulation_H_rows,
                                    _state_out.H,
                                    model.joint_armature,
                                ],
                                outputs=[_state_out.L],
                                device=model.device,
                            )
                    else:
                        if requires_grad:
                            wp.copy(_state_out.H, _state_in.H)
                            wp.copy(_state_out.L, _state_in.L)

                    # solve for qdd
                    _state_in.joint_qdd.zero_()
                    wp.launch(
                        eval_dense_solve_batched,
                        dim=model.articulation_count,
                        inputs=[
                            _model.articulation_H_start,
                            _model.articulation_H_rows,
                            _model.articulation_dof_start,
                            _model.H,
                            _model.L,
                            _state_in.joint_tau,
                        ],
                        outputs=[
                            _state_in.joint_qdd,
                            _state_in.joint_solve_tmp,
                        ],
                        device=model.device,
                    )

            # -------------------------------------
            # integrate bodies

            if model.joint_count:
                wp.launch(
                    kernel=integrate_generalized_joints,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        state_in.joint_q,
                        state_in.joint_qd,
                        _state_in.joint_qdd,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    device=model.device,
                )

                # update maximal coordinates
                eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)

            self.integrate_particles(model, state_in, state_out, dt)

            self._step += 1

            return state_out
