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

from contextlib import contextmanager
from typing import ClassVar

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, JointType, Model, State
from ..featherstone.kernels import eval_fk_with_velocity_conversion
from ..semi_implicit.kernels_contact import (
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
)
from ..semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)
from ..solver import SolverBase
from .kernels import (
    TILE_THREADS,
    add_contact_compliance_to_diag,
    allocate_joint_limit_slots,
    allocate_world_contact_slots,
    apply_augmented_joint_tau,
    apply_augmented_mass_diagonal_grouped,
    apply_impulses_world_par_dof,
    build_augmented_joint_rows,
    build_mass_update_mask,
    build_mf_contact_rows,
    cholesky_loop,
    clamp_joint_tau,
    compute_com_transforms,
    compute_composite_inertia,
    compute_contact_linear_force_from_impulses,
    compute_mf_body_Hinv,
    compute_mf_effective_mass_and_rhs,
    compute_mf_world_dof_offsets,
    compute_spatial_inertia,
    compute_velocity_predictor,
    compute_world_contact_bias,
    convert_root_free_qd_local_to_world,
    convert_root_free_qd_world_to_local,
    copy_int_array_masked,
    crba_fill_par_dof,
    delassus_par_row_col,
    detect_limit_count_changes,
    eval_rigid_fk,
    eval_rigid_id,
    eval_rigid_mass,
    eval_rigid_tau,
    extract_diag_from_JY_par_art,
    finalize_mf_constraint_counts,
    finalize_world_constraint_counts,
    finalize_world_diag_cfm,
    gather_JY_to_world,
    gather_tau_to_groups,
    hinv_jt_par_row,
    integrate_generalized_joints,
    pack_contact_linear_force_as_spatial,
    pgs_convergence_diagnostic_velocity,
    populate_joint_limit_J_for_size,
    populate_world_J_for_size,
    prepare_world_impulses,
    rhs_accum_world_par_art,
    scatter_qdd_from_groups,
    trisolve_loop,
    update_articulation_origins,
    update_articulation_root_com_offsets,
    update_body_qd_from_featherstone,
    update_qdd_from_velocity,
)


@wp.kernel
def localize_parent_indices(
    counts: wp.array[int],
    max_constraints: int,
    parent_arr: wp.array[int],
    parent_local_arr: wp.array[int],
):
    art = wp.tid()
    m = counts[art]
    base = art * max_constraints

    for i in range(m):
        idx = base + i
        p = parent_arr[idx]
        if p >= 0:
            parent_local_arr[idx] = p - base
        else:
            parent_local_arr[idx] = -1


class SolverFeatherPGS(SolverBase):
    """A semi-implicit integrator using symplectic Euler that operates
    on reduced (also called generalized) coordinates to simulate articulated rigid body dynamics
    based on Featherstone's composite rigid body algorithm (CRBA).

    This private solver branch keeps only the matrix-free contact solve path and
    the current winner kernel strategy.

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Instead of maximal coordinates :attr:`~newton.State.body_q` (rigid body positions) and :attr:`~newton.State.body_qd`
    (rigid body velocities) as is the case in :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`,
    :class:`~newton._src.solvers.feather_pgs.SolverFeatherPGS` uses :attr:`~newton.State.joint_q` and :attr:`~newton.State.joint_qd` to represent
    the positions and velocities of joints without allowing any redundant degrees of freedom.

    After constructing :class:`~newton.Model` and :class:`~newton.State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Note:
        Unlike :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`, :class:`~newton._src.solvers.feather_pgs.SolverFeatherPGS`
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

        from newton._src.solvers.feather_pgs import SolverFeatherPGS

        solver = SolverFeatherPGS(model)

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
        enable_contact_friction: bool = True,
        enable_joint_limits: bool = False,
        pgs_iterations: int = 12,
        pgs_beta: float = 0.2,
        pgs_cfm: float = 1.0e-6,
        contact_compliance: float = 0.0,
        pgs_omega: float = 1.0,
        max_constraints: int = 32,
        pgs_warmstart: bool = False,
        mf_max_constraints: int = 512,
        # Parallelism options
        use_parallel_streams: bool = True,
        double_buffer: bool = True,
        nvtx: bool = False,
        pgs_debug: bool = False,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            angular_damping (float, optional): Angular damping factor. Defaults to 0.05.
            update_mass_matrix_interval (int, optional): How often to update the mass matrix (every n-th time the :meth:`step` function gets called). Defaults to 1.
            friction_smoothing (float, optional): The delta value for the Huber norm (see :func:`warp.math.norm_huber`) used for the friction velocity normalization. Defaults to 1.0.
            enable_contact_friction (bool, optional): Enables Coulomb friction contacts inside the PGS solve. Defaults to True.
            enable_joint_limits (bool, optional): Enforce joint position limits as unilateral PGS
                constraints. Each violated limit adds one constraint row. Defaults to False.
            pgs_iterations (int, optional): Number of Gauss-Seidel iterations to apply per frame. Defaults to 12.
            pgs_beta (float, optional): ERP style position correction factor. Defaults to 0.2.
            pgs_cfm (float, optional): Compliance/regularization added to the Delassus diagonal. Defaults to 1.0e-6.
            contact_compliance (float, optional): Normal contact compliance [m/N] applied
                to articulated contact rows. Converted to an impulse-space diagonal term using
                ``compliance / dt^2``. Defaults to 0.0.
            pgs_omega (float, optional): Successive over-relaxation factor for the PGS sweep. Defaults to 1.0.
            max_constraints (int, optional): Maximum number of articulated contact constraint
                rows stored per world. Free rigid body contacts are stored separately, bounded by
                mf_max_constraints. Defaults to 32.
            pgs_warmstart (bool, optional): Re-use impulses from the previous frame when contacts persist. Defaults to False.
            mf_max_constraints (int, optional): Maximum number of matrix-free constraints per world. Defaults to 512.
            use_parallel_streams (bool, optional): Dispatch size groups on separate CUDA streams.
                Defaults to True.
        """
        super().__init__(model)

        self.angular_damping = angular_damping
        self.update_mass_matrix_interval = update_mass_matrix_interval
        self.friction_smoothing = friction_smoothing
        self.enable_contact_friction = enable_contact_friction
        self.enable_joint_limits = enable_joint_limits
        self.pgs_iterations = pgs_iterations
        self.pgs_beta = pgs_beta
        self.pgs_cfm = pgs_cfm
        self.contact_compliance = contact_compliance
        self.pgs_omega = pgs_omega
        self.max_constraints = max_constraints
        self.pgs_warmstart = pgs_warmstart
        self.mf_max_constraints = mf_max_constraints
        self._double_buffer = double_buffer
        self._nvtx = nvtx
        self.pgs_debug = pgs_debug
        self._pgs_convergence_log: list[np.ndarray] = []
        self.small_dof_threshold = 12
        self.delassus_chunk_size = None
        self.use_parallel_streams = use_parallel_streams

        self._step = 0
        self._force_mass_update = False
        self._last_step_dt = None

        self._compute_articulation_metadata(model)

        self._allocate_common_buffers(model)
        self._allocate_buffers(model)
        self._allocate_world_buffers(model)
        self._allocate_mf_buffers(model)
        self._allocate_debug_buffers(model)
        self._scatter_armature_to_groups(model)
        self._init_size_group_streams(model)
        self._dummy_contact_impulses = wp.zeros((1, 1), dtype=wp.float32, device=model.device)

        if model.shape_material_mu is not None:
            self.shape_material_mu = model.shape_material_mu
        else:
            self.shape_material_mu = wp.zeros(
                (1,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad
            )

        self._init_double_buffer_stream()

    def _compute_articulation_metadata(self, model):
        self._compute_articulation_indices(model)
        self._compute_root_free_metadata(model)
        self._setup_size_grouping(model)
        self._setup_world_mapping(model)
        self._is_one_art_per_world = self.world_count == model.articulation_count
        self._is_homogeneous = (len(self.size_groups) == 1) if self.size_groups else True
        self._build_body_maps(model)
        self._classify_free_rigid_bodies(model)

    def _compute_articulation_indices(self, model):
        # calculate total size and offsets of Jacobian and mass matrices for entire system
        if model.joint_count:
            self.J_size = 0
            self.M_size = wp.int64(0)
            self.H_size = 0

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

                articulation_J_start.append(self.J_size)
                articulation_M_start.append(int(self.M_size))
                articulation_H_start.append(self.H_size)
                articulation_dof_start.append(first_dof)
                articulation_coord_start.append(first_coord)

                # bit of data duplication here, but will leave it as such for clarity
                articulation_M_rows.append(joint_count * 6)
                articulation_H_rows.append(dof_count)
                articulation_J_rows.append(joint_count * 6)
                articulation_J_cols.append(dof_count)

                self.J_size += 6 * joint_count * dof_count
                self.M_size = wp.int64(self.M_size + wp.int64(joint_count * 36))
                self.H_size += dof_count * dof_count

            # matrix offsets for grouped gemm
            self.articulation_J_start = wp.array(articulation_J_start, dtype=wp.int32, device=model.device)
            self.articulation_M_start = wp.array(articulation_M_start, dtype=wp.int32, device=model.device)
            self.articulation_H_start = wp.array(articulation_H_start, dtype=wp.int32, device=model.device)

            self.articulation_M_rows = wp.array(articulation_M_rows, dtype=wp.int32, device=model.device)
            self.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_rows = wp.array(articulation_J_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_cols = wp.array(articulation_J_cols, dtype=wp.int32, device=model.device)

            self.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32, device=model.device)
            self.articulation_coord_start = wp.array(articulation_coord_start, dtype=wp.int32, device=model.device)

            self.articulation_max_dofs = int(max(articulation_H_rows)) if articulation_H_rows else 0
            self.M_size = int(self.M_size)
        else:
            self.M_size = 0
            self.articulation_max_dofs = 0

    def _compute_root_free_metadata(self, model):
        if not model.articulation_count or not model.joint_count:
            self.articulation_root_is_free = None
            self.articulation_root_dof_start = None
            self._has_root_free = False
            return

        articulation_start = model.articulation_start.numpy()
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        root_is_free = np.zeros(model.articulation_count, dtype=np.int32)
        root_dof_start = np.zeros(model.articulation_count, dtype=np.int32)

        for art in range(model.articulation_count):
            root_joint = articulation_start[art]
            root_dof_start[art] = int(joint_qd_start[root_joint])
            jt = int(joint_type[root_joint])
            jp = int(joint_parent[root_joint])
            if jp == -1 and (jt == int(JointType.FREE) or jt == int(JointType.DISTANCE)):
                root_is_free[art] = 1

        self.articulation_root_is_free = wp.array(root_is_free, dtype=wp.int32, device=model.device)
        self.articulation_root_dof_start = wp.array(root_dof_start, dtype=wp.int32, device=model.device)
        self._has_root_free = bool(np.any(root_is_free != 0))

    def _setup_size_grouping(self, model):
        """Set up size-grouped storage and indirection arrays for multi-articulation support.

        This enables efficient handling of articulations with different DOF counts by grouping
        them by size, allowing optimized tiled kernel launches for each size group.
        """
        if not model.articulation_count or not model.joint_count:
            self.size_groups = []
            self.n_arts_by_size = {}
            return

        device = model.device

        # Get DOF counts per articulation
        articulation_start = model.articulation_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        articulation_dof_counts = np.zeros(model.articulation_count, dtype=np.int32)
        for art_idx in range(model.articulation_count):
            first_joint = articulation_start[art_idx]
            last_joint = articulation_start[art_idx + 1]
            first_dof = joint_qd_start[first_joint]
            last_dof = joint_qd_start[last_joint]
            articulation_dof_counts[art_idx] = last_dof - first_dof

        # Determine unique sizes (sorted descending for largest first)
        unique_sizes = sorted(set(articulation_dof_counts), reverse=True)
        self.size_groups = unique_sizes
        self.n_arts_by_size = {size: int(np.sum(articulation_dof_counts == size)) for size in unique_sizes}

        # Build indirection arrays
        art_size_np = articulation_dof_counts.copy()
        art_group_idx_np = np.zeros(model.articulation_count, dtype=np.int32)
        group_to_art_np = {size: np.zeros(self.n_arts_by_size[size], dtype=np.int32) for size in unique_sizes}

        # Track current index within each size group
        size_counters = dict.fromkeys(unique_sizes, 0)

        for art_idx in range(model.articulation_count):
            size = articulation_dof_counts[art_idx]
            group_idx = size_counters[size]

            art_group_idx_np[art_idx] = group_idx
            group_to_art_np[size][group_idx] = art_idx

            size_counters[size] += 1

        # Copy to GPU
        self.art_size = wp.array(art_size_np, dtype=wp.int32, device=device)
        self.art_group_idx = wp.array(art_group_idx_np, dtype=wp.int32, device=device)
        self.group_to_art = {
            size: wp.array(group_to_art_np[size], dtype=wp.int32, device=device) for size in unique_sizes
        }

    def _setup_world_mapping(self, model):
        """Set up world-level mapping for multi-articulation support.

        Maps articulations to worlds and computes per-world articulation ranges.
        """
        if not model.articulation_count:
            self.world_count = 0
            self.art_to_world = None
            self.world_art_start = None
            self._is_multi_articulation = False
            self._max_arts_per_world = 0
            return

        device = model.device

        # Get articulation-to-world mapping from model
        if model.articulation_world is not None:
            art_to_world_np = model.articulation_world.numpy().astype(np.int32)
            # Handle -1 (global) by mapping to world 0
            art_to_world_np = np.where(art_to_world_np < 0, 0, art_to_world_np)
            self.world_count = int(np.max(art_to_world_np)) + 1
        else:
            # Default: one articulation per world (current behavior)
            art_to_world_np = np.arange(model.articulation_count, dtype=np.int32)
            self.world_count = model.articulation_count

        self.art_to_world = wp.array(art_to_world_np, dtype=wp.int32, device=device)

        # Compute per-world articulation ranges
        # Count articulations per world
        world_art_counts = np.zeros(self.world_count, dtype=np.int32)
        for world_idx in art_to_world_np:
            world_art_counts[world_idx] += 1

        # Compute start indices (exclusive prefix sum)
        world_art_start_np = np.zeros(self.world_count + 1, dtype=np.int32)
        world_art_start_np[1:] = np.cumsum(world_art_counts)

        self.world_art_start = wp.array(world_art_start_np, dtype=wp.int32, device=device)

        # Detect if we have multiple articulations per world
        self._max_arts_per_world = int(np.max(world_art_counts)) if len(world_art_counts) > 0 else 0
        self._is_multi_articulation = self._max_arts_per_world > 1

    def _build_body_maps(self, model):
        if not model.body_count or not model.articulation_count:
            self.body_to_joint = None
            self.body_to_articulation = None
            return

        joint_child = model.joint_child.numpy()
        articulation_start = model.articulation_start.numpy()

        body_to_joint = [-1] * model.body_count
        body_to_articulation = [-1] * model.body_count

        for articulation in range(model.articulation_count):
            joint_start = articulation_start[articulation]
            joint_end = articulation_start[articulation + 1]

            for joint_index in range(joint_start, joint_end):
                child = joint_child[joint_index]
                if child < 0:
                    continue

                body_to_joint[child] = joint_index
                body_to_articulation[child] = articulation

        device = model.device
        self.body_to_joint = wp.array(body_to_joint, dtype=wp.int32, device=device)
        self.body_to_articulation = wp.array(body_to_articulation, dtype=wp.int32, device=device)

    def _classify_free_rigid_bodies(self, model):
        """Identify articulations that are single free rigid bodies.

        An articulation is "free rigid" if it has exactly 1 joint, that joint
        is FREE type, and the joint parent is -1 (world). These can be solved
        with a cheaper matrix-free PGS path.
        """
        if not model.articulation_count or not model.joint_count:
            self._has_free_rigid_bodies = False
            self._has_mixed_contacts = False
            self._n_free_rigid = 0
            self.is_free_rigid = None
            return

        joint_type_np = model.joint_type.numpy()
        joint_parent_np = model.joint_parent.numpy()
        articulation_start_np = model.articulation_start.numpy()

        is_free_rigid_np = np.zeros(model.articulation_count, dtype=np.int32)
        for art_idx in range(model.articulation_count):
            first_joint = articulation_start_np[art_idx]
            last_joint = articulation_start_np[art_idx + 1]
            if last_joint - first_joint == 1:
                if int(joint_type_np[first_joint]) == int(JointType.FREE) and int(joint_parent_np[first_joint]) == -1:
                    is_free_rigid_np[art_idx] = 1

        n_free = int(np.sum(is_free_rigid_np))
        self._has_free_rigid_bodies = n_free > 0
        self._n_free_rigid = n_free
        self._has_mixed_contacts = self._has_free_rigid_bodies and self._n_free_rigid < model.articulation_count
        self.is_free_rigid = wp.array(is_free_rigid_np, dtype=wp.int32, device=model.device)

    def _compute_world_dof_mapping(self, model):
        """Compute per-world DOF start and max DOF count for consolidated J/Y arrays."""
        art_to_world_np = self.art_to_world.numpy()
        art_dof_start_np = self.articulation_dof_start.numpy()
        art_H_rows_np = self.articulation_H_rows.numpy()

        world_dof_start_np = np.full(self.world_count, np.iinfo(np.int32).max, dtype=np.int32)
        world_dof_end_np = np.zeros(self.world_count, dtype=np.int32)

        for art_idx in range(model.articulation_count):
            w = art_to_world_np[art_idx]
            ds = art_dof_start_np[art_idx]
            de = ds + art_H_rows_np[art_idx]
            world_dof_start_np[w] = min(world_dof_start_np[w], ds)
            world_dof_end_np[w] = max(world_dof_end_np[w], de)

        # For worlds with no articulations, set start to 0
        world_dof_start_np = np.where(world_dof_start_np == np.iinfo(np.int32).max, 0, world_dof_start_np)

        world_dof_counts = world_dof_end_np - world_dof_start_np
        self.max_world_dofs = int(np.max(world_dof_counts)) if len(world_dof_counts) > 0 else 0
        self.world_dof_start = wp.array(world_dof_start_np, dtype=wp.int32, device=model.device)

    def _allocate_common_buffers(self, model):
        if model.joint_count:
            self.M_blocks = wp.zeros(
                (self.M_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad
            )
            self.mass_update_mask = wp.zeros(
                (model.articulation_count,), dtype=wp.int32, device=model.device, requires_grad=model.requires_grad
            )
            self.v_hat = wp.zeros_like(model.joint_qd, requires_grad=model.requires_grad)
            self.v_out = wp.zeros_like(model.joint_qd, requires_grad=model.requires_grad)
            self.qd_work = wp.zeros_like(model.joint_qd, requires_grad=model.requires_grad)
            self.v_mf_accum = wp.zeros_like(model.joint_qd, requires_grad=model.requires_grad)
            self.v_out_snap = wp.zeros_like(model.joint_qd, requires_grad=model.requires_grad)
        else:
            self.M_blocks = None
            self.mass_update_mask = None
            self.v_hat = None
            self.v_out = None
            self.qd_work = None
            self.v_mf_accum = None
            self.v_out_snap = None

        if model.body_count:
            self.body_I_m = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=model.requires_grad
            )
            wp.launch(
                compute_spatial_inertia,
                model.body_count,
                inputs=[model.body_inertia, model.body_mass],
                outputs=[self.body_I_m],
                device=model.device,
            )
            self.body_X_com = wp.empty(
                (model.body_count,), dtype=wp.transform, device=model.device, requires_grad=model.requires_grad
            )
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=model.device,
            )
            self.body_I_c = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=model.requires_grad
            )
        else:
            self.body_I_m = None
            self.body_X_com = None
            self.body_I_c = None

        if not model.articulation_count or not model.joint_count:
            self.articulation_origin = None
            self.articulation_root_com_offset = None
            return

        self.articulation_origin = wp.zeros(
            (model.articulation_count,), dtype=wp.vec3, device=model.device, requires_grad=model.requires_grad
        )
        self.articulation_root_com_offset = wp.zeros(
            (model.articulation_count,), dtype=wp.vec3, device=model.device, requires_grad=model.requires_grad
        )

        max_dofs = self.articulation_max_dofs
        if max_dofs == 0:
            return

        device = model.device
        requires_grad = model.requires_grad
        articulation_count = model.articulation_count
        total_rows = articulation_count * max_dofs

        self.aug_row_counts = wp.zeros(
            (articulation_count,), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.aug_limit_counts = wp.zeros(
            (articulation_count,), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.aug_prev_limit_counts = wp.zeros_like(self.aug_limit_counts)
        self.limit_change_mask = wp.zeros_like(self.aug_limit_counts)
        self.aug_row_dof_index = wp.zeros((total_rows,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.aug_row_K = wp.zeros((total_rows,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.aug_row_u0 = wp.zeros((total_rows,), dtype=wp.float32, device=device, requires_grad=requires_grad)

    def _allocate_buffers(self, model):
        if not self.size_groups:
            self.H_by_size = {}
            self.L_by_size = {}
            self.J_by_size = {}
            self.Y_by_size = {}
            self.R_by_size = {}
            self.tau_by_size = {}
            self.qdd_by_size = {}
            self._H_bufs = None
            self._J_bufs = None
            return

        device = model.device
        requires_grad = model.requires_grad
        max_constraints = self.max_constraints

        self.L_by_size = {}
        self.Y_by_size = {}
        self.R_by_size = {}
        self.tau_by_size = {}
        self.qdd_by_size = {}

        if self._double_buffer and device.is_cuda:
            self._H_bufs = [{}, {}]
            self._J_bufs = [{}, {}]
        else:
            self._H_bufs = None
            self._J_bufs = None

        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]

            h_dim = size
            j_rows = max_constraints

            if self._H_bufs is not None:
                for buf_idx in range(2):
                    self._H_bufs[buf_idx][size] = wp.zeros(
                        (n_arts, h_dim, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
                    )
                    self._J_bufs[buf_idx][size] = wp.zeros(
                        (n_arts, j_rows, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
                    )
            else:
                pass  # allocated below after the if/else

            self.L_by_size[size] = wp.zeros(
                (n_arts, h_dim, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
            )

            self.Y_by_size[size] = wp.zeros(
                (n_arts, j_rows, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
            )

            # Armature (regularization) [n_arts, h_dim] - needs to match H dimension for tile_diag_add
            self.R_by_size[size] = wp.zeros(
                (n_arts, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
            )

            # Tau and qdd grouped buffers for tiled triangular solve [n_arts, h_dim, 1]
            self.tau_by_size[size] = wp.zeros((n_arts, h_dim, 1), dtype=wp.float32, device=device)
            self.qdd_by_size[size] = wp.zeros((n_arts, h_dim, 1), dtype=wp.float32, device=device)

        if self._H_bufs is not None:
            self.H_by_size = self._H_bufs[0]
            self.J_by_size = self._J_bufs[0]
            self._buf_idx = 0
        else:
            self.H_by_size = {}
            self.J_by_size = {}
            for size in self.size_groups:
                n_arts = self.n_arts_by_size[size]
                h_dim = size
                j_rows = max_constraints
                self.H_by_size[size] = wp.zeros(
                    (n_arts, h_dim, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
                )
                self.J_by_size[size] = wp.zeros(
                    (n_arts, j_rows, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
                )

        max_contacts = int(model.rigid_contact_max)
        if max_contacts <= 0:
            # Unified collision pipeline manages its own contact capacity and may
            # leave model.rigid_contact_max unset. Use the same estimator as collide.py.
            from ...sim.collide import _estimate_rigid_contact_max  # noqa: PLC0415

            max_contacts = int(_estimate_rigid_contact_max(model))
        max_contacts = max(max_contacts, 1)
        self.contact_world = wp.zeros((max_contacts,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.contact_slot = wp.zeros((max_contacts,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.contact_art_a = wp.zeros((max_contacts,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.contact_art_b = wp.zeros((max_contacts,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.slot_counter = wp.zeros((self.world_count,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.contact_path = wp.zeros((max_contacts,), dtype=wp.int32, device=device, requires_grad=requires_grad)

        # Joint limit buffers (per-DOF tracking)
        if self.enable_joint_limits and model.joint_dof_count > 0:
            dof_count = model.joint_dof_count
            self.limit_slot = wp.full((dof_count,), -1, dtype=wp.int32, device=device, requires_grad=requires_grad)
            self.limit_sign = wp.zeros((dof_count,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        else:
            self.limit_slot = None
            self.limit_sign = None

    def _allocate_world_buffers(self, model):
        """Allocate world-level constraint system buffers for multi-articulation support."""
        if self.world_count == 0:
            return

        device = model.device
        requires_grad = model.requires_grad
        max_constraints = self.max_constraints

        self.C = None
        self._compute_world_dof_mapping(model)
        self.J_world = wp.zeros(
            (self.world_count, max_constraints, self.max_world_dofs),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        self.Y_world = wp.zeros(
            (self.world_count, max_constraints, self.max_world_dofs),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )

        self.rhs = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.impulses = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.diag = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )

        # Constraint metadata (per world x constraint)
        self.row_type = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.row_parent = wp.full(
            (self.world_count, max_constraints), -1, dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.row_mu = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.row_beta = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.row_cfm = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.phi = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.target_velocity = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )

        # Per-world constraint counts
        self.constraint_count = wp.zeros(
            (self.world_count,), dtype=wp.int32, device=device, requires_grad=requires_grad
        )

    def _allocate_mf_buffers(self, model):
        """Allocate buffers for matrix-free PGS path for free rigid body contacts."""
        if not self._has_free_rigid_bodies:
            return

        device = model.device
        requires_grad = model.requires_grad
        worlds = self.world_count
        mf_max_c = self.mf_max_constraints
        body_count = model.body_count

        self.mf_constraint_count = wp.zeros((worlds,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.mf_slot_counter = wp.zeros((worlds,), dtype=wp.int32, device=device, requires_grad=requires_grad)

        self.mf_body_a = wp.zeros((worlds, mf_max_c), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.mf_body_b = wp.zeros((worlds, mf_max_c), dtype=wp.int32, device=device, requires_grad=requires_grad)

        self.mf_J_a = wp.zeros((worlds, mf_max_c, 6), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.mf_J_b = wp.zeros((worlds, mf_max_c, 6), dtype=wp.float32, device=device, requires_grad=requires_grad)

        self.mf_MiJt_a = wp.zeros((worlds, mf_max_c, 6), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.mf_MiJt_b = wp.zeros((worlds, mf_max_c, 6), dtype=wp.float32, device=device, requires_grad=requires_grad)

        self.mf_rhs = wp.zeros((worlds, mf_max_c), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.mf_impulses = wp.zeros((worlds, mf_max_c), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.mf_eff_mass_inv = wp.zeros(
            (worlds, mf_max_c), dtype=wp.float32, device=device, requires_grad=requires_grad
        )

        self.mf_row_type = wp.zeros((worlds, mf_max_c), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.mf_row_parent = wp.full((worlds, mf_max_c), -1, dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.mf_row_mu = wp.zeros((worlds, mf_max_c), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.mf_phi = wp.zeros((worlds, mf_max_c), dtype=wp.float32, device=device, requires_grad=requires_grad)

        self.mf_body_Hinv = wp.zeros((body_count,), dtype=wp.spatial_matrix, device=device, requires_grad=requires_grad)

        # World-relative DOF offsets for the unified articulated + free-rigid GS kernel.
        self.mf_dof_a = wp.zeros((worlds, mf_max_c), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.mf_dof_b = wp.zeros((worlds, mf_max_c), dtype=wp.int32, device=device, requires_grad=requires_grad)

        # Packed MF metadata for the unified GS kernel (int4 per constraint):
        #   .x = (dof_a << 16) | (dof_b & 0xFFFF)
        #   .y = __float_as_int(eff_mass_inv)
        #   .z = __float_as_int(rhs)
        #   .w = row_type | (row_parent << 16)
        self.mf_meta_packed = wp.zeros((worlds, mf_max_c * 4), dtype=wp.int32, device=device)

        # Body map buffers for tiled MF PGS kernel
        self.max_mf_bodies = 64
        self.mf_body_list = wp.zeros(
            (worlds, self.max_mf_bodies), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.mf_body_dof_start = wp.zeros(
            (worlds, self.max_mf_bodies), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.mf_body_count = wp.zeros((worlds,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.mf_local_body_a = wp.zeros((worlds, mf_max_c), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.mf_local_body_b = wp.zeros((worlds, mf_max_c), dtype=wp.int32, device=device, requires_grad=requires_grad)

    def _allocate_debug_buffers(self, model):
        """Allocate buffers for PGS convergence diagnostics."""
        if not self.pgs_debug:
            return
        device = model.device
        worlds = self.world_count
        max_c = self.max_constraints
        mf_max_c = self.mf_max_constraints

        self._diag_metrics = wp.zeros((worlds, 4), dtype=wp.float32, device=device)
        self._diag_prev_impulses = wp.zeros((worlds, max_c), dtype=wp.float32, device=device)
        if hasattr(self, "mf_impulses"):
            self._diag_prev_mf_impulses = wp.zeros((worlds, mf_max_c), dtype=wp.float32, device=device)
        else:
            self._diag_prev_mf_impulses = None

    def _scatter_armature_to_groups(self, model):
        """Copy armature from model (DOF-ordered) to size-grouped storage."""
        if not self.size_groups:
            return

        armature_np = model.joint_armature.numpy()
        art_dof_start_np = self.articulation_dof_start.numpy()
        art_H_rows_np = self.articulation_H_rows.numpy()

        # R_by_size is sized to actual DOF count (matches H_by_size allocation)
        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            R_np = np.zeros((n_arts, size), dtype=np.float32)

            group_to_art_np = self.group_to_art[size].numpy()
            for group_idx in range(n_arts):
                art_idx = group_to_art_np[group_idx]
                dof_start = art_dof_start_np[art_idx]
                dof_count = art_H_rows_np[art_idx]
                R_np[group_idx, :dof_count] = armature_np[dof_start : dof_start + dof_count]

            self.R_by_size[size] = wp.array(R_np, dtype=wp.float32, device=model.device)

    def _init_size_group_streams(self, model):
        """Initialize CUDA streams for parallel kernel launches across size groups.

        When multiple DOF sizes exist (heterogeneous articulations), we can launch
        tiled kernels for different sizes in parallel using separate CUDA streams.
        """
        self._size_streams: dict[int, wp.Stream | None] = {}
        self._size_events: dict[int, wp.Event | None] = {}

        if self.use_parallel_streams and model.device.is_cuda and len(self.size_groups) > 1:
            for size in self.size_groups:
                self._size_streams[size] = wp.Stream(model.device)
                self._size_events[size] = wp.Event(model.device)
        else:
            # No streams needed for CPU or single size group
            for size in self.size_groups:
                self._size_streams[size] = None
                self._size_events[size] = None

    def _init_double_buffer_stream(self):
        """Create a dedicated CUDA stream for async memset of H/J buffers."""
        if self._H_bufs is None or not self.model.device.is_cuda:
            self._memset_stream = None
            return
        self._memset_stream = wp.Stream(self.model.device)
        # Track the last memset-done event per buffer slot so the main stream
        # can wait only for the specific buffer it needs.
        self._memset_done_event: list[wp.Event | None] = [None, None]

    def seed_double_buffer_events(self):
        """Record initial memset_done events on the main stream.

        Must be called inside CUDA graph capture, before the first ``step()`` call.
        Since buffers are allocated with ``wp.zeros()``, they are already zeroed;
        recording here provides trivially-satisfied wait targets for the first two
        substeps.
        """
        if self._memset_stream is None:
            return
        main_stream = wp.get_stream(self.model.device)
        self._memset_done_event[0] = main_stream.record_event()
        self._memset_done_event[1] = main_stream.record_event()

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
        collide_done_event=None,
    ):
        if self._last_step_dt is None:
            self._last_step_dt = dt
        elif abs(self._last_step_dt - dt) > 1.0e-8:
            self._force_mass_update = True
            self._last_step_dt = dt
        else:
            self._last_step_dt = dt

        model = self.model

        if control is None:
            control = model.control(clone_variables=False)
        state_aug = self._prepare_augmented_state(state_in, state_out, control)

        if collide_done_event is not None and state_in.particle_count > 0:
            wp.get_stream(self.model.device).wait_event(collide_done_event)
            collide_done_event = None  # consumed

        self._eval_particle_forces(state_in, control, contacts)

        if not model.joint_count:
            self.integrate_particles(model, state_in, state_out, dt)
            self._step += 1
            return state_out

        # Double-buffer: select buffer set and wait for its memset to finish
        if self._memset_stream is not None:
            self.H_by_size = self._H_bufs[self._buf_idx]
            self.J_by_size = self._J_bufs[self._buf_idx]
            evt = self._memset_done_event[self._buf_idx]
            if evt is not None:
                wp.get_stream(model.device).wait_event(evt)

        # ══════════════════════════════════════════════════════════════
        # STAGE 1: FK/ID + drives + CRBA
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S1_FK_ID_CRBA", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            self._stage1_fk_id(state_in, state_aug, state_out)

            if model.articulation_count:
                self._stage1_drives(state_in, state_aug, control, dt)

            self._stage1_crba(state_aug)
        # ══════════════════════════════════════════════════════════════
        # STAGE 2: Cholesky
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S2_Cholesky", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                with ctx:
                    use_tiled = size > self.small_dof_threshold
                    if use_tiled:
                        self._stage2_cholesky_tiled(size)
                    else:
                        self._stage2_cholesky_loop(size)
        # ══════════════════════════════════════════════════════════════
        # STAGE 3: Triangular solve + v_hat
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S3_Trisolve_Vhat", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            self._stage3_zero_qdd(state_aug)
            for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                with ctx:
                    use_tiled = size > self.small_dof_threshold
                    if use_tiled:
                        self._stage3_trisolve_tiled(size, state_aug)
                    else:
                        self._stage3_trisolve_loop(size, state_aug)
            self._stage3_compute_v_hat(state_in, state_aug, dt)

        # Wait for pipelined collide (if running on separate stream)
        if collide_done_event is not None:
            wp.get_stream(model.device).wait_event(collide_done_event)

        # ══════════════════════════════════════════════════════════════
        # STAGE 4: Build contact problem
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S4_ContactBuild", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            self._stage4_build_rows(state_in, state_aug, contacts)

        with wp.ScopedTimer("S4_HinvJt_Diag_RHS", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                with ctx:
                    use_tiled = size > self.small_dof_threshold
                    if use_tiled:
                        self._stage4_hinv_jt_tiled(size)
                    else:
                        self._stage4_hinv_jt_par_row(size)

            # Extract only the world diagonal from J*Y; do not assemble the full Delassus matrix.
            self.diag.zero_()
            for size in self.size_groups:
                self._stage4_extract_diag_from_JY(size)
            self._stage4_finalize_world_diag_cfm()
            self._stage4_add_contact_compliance(dt)

            # RHS = bias only (J*v recomputed per iteration)
            self._stage4_compute_rhs_world(dt)
            # NOTE: skip _stage4_accumulate_rhs_world — J*v_hat not baked into rhs

        if self._has_free_rigid_bodies:
            with wp.ScopedTimer("S4_MF_Setup", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                self._mf_pgs_setup(state_aug, dt)

                wp.launch(
                    compute_mf_world_dof_offsets,
                    dim=self.world_count * self.mf_max_constraints,
                    inputs=[
                        self.mf_constraint_count,
                        self.mf_body_a,
                        self.mf_body_b,
                        self.body_to_articulation,
                        self.articulation_dof_start,
                        self.world_dof_start,
                        self.mf_max_constraints,
                    ],
                    outputs=[self.mf_dof_a, self.mf_dof_b],
                    device=self.model.device,
                )

        # ══════════════════════════════════════════════════════════════
        # STAGE 5+6: PGS solve
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S5_PGS_Prep", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            self._stage5_prepare_impulses_world()

        with wp.ScopedTimer("S5_GatherJY", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            for size in self.size_groups:
                n_arts = self.n_arts_by_size[size]
                wp.launch(
                    gather_JY_to_world,
                    dim=int(n_arts * self.max_constraints * size),
                    inputs=[
                        self.group_to_art[size],
                        self.art_to_world,
                        self.articulation_dof_start,
                        self.constraint_count,
                        self.world_dof_start,
                        self.J_by_size[size],
                        self.Y_by_size[size],
                        size,
                        self.max_constraints,
                        n_arts,
                    ],
                    outputs=[self.J_world, self.Y_world],
                    device=self.model.device,
                )

            self._stage6_prepare_world_velocity()

            pack_kernel = TiledKernelFactory.get_pack_mf_meta_kernel(self.mf_max_constraints, self.model.device)
            wp.launch_tiled(
                pack_kernel,
                dim=[self.world_count],
                inputs=[
                    self.mf_constraint_count,
                    self.mf_dof_a,
                    self.mf_dof_b,
                    self.mf_eff_mass_inv,
                    self.mf_rhs,
                    self.mf_row_type,
                    self.mf_row_parent,
                ],
                outputs=[self.mf_meta_packed],
                block_dim=32,
                device=self.model.device,
            )

        with wp.ScopedTimer("S6_PGS_Solve", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            mf_gs_kernel = TiledKernelFactory.get_pgs_solve_mf_gs_kernel(
                self.max_constraints,
                self.mf_max_constraints,
                self.max_world_dofs,
                self.model.device,
            )

            if self.pgs_debug:
                self._pgs_convergence_log.append([])
                for _pgs_dbg_iter in range(self.pgs_iterations):
                    wp.copy(self._diag_prev_impulses, self.impulses)
                    if self._diag_prev_mf_impulses is not None:
                        wp.copy(self._diag_prev_mf_impulses, self.mf_impulses)

                    wp.launch_tiled(
                        mf_gs_kernel,
                        dim=[self.world_count],
                        inputs=[
                            self.constraint_count,
                            self.world_dof_start,
                            self.rhs,
                            self.diag,
                            self.impulses,
                            self.J_world,
                            self.Y_world,
                            self.row_type,
                            self.row_parent,
                            self.row_mu,
                            self.mf_constraint_count,
                            self.mf_meta_packed,
                            self.mf_impulses,
                            self.mf_J_a,
                            self.mf_J_b,
                            self.mf_MiJt_a,
                            self.mf_MiJt_b,
                            self.mf_row_mu,
                            1,
                            self.pgs_omega,
                        ],
                        outputs=[self.v_out],
                        block_dim=32,
                        device=self.model.device,
                    )

                    wp.launch(
                        pgs_convergence_diagnostic_velocity,
                        dim=self.world_count,
                        inputs=[
                            self.constraint_count,
                            self.world_dof_start,
                            self.rhs,
                            self.impulses,
                            self._diag_prev_impulses,
                            self.row_type,
                            self.row_parent,
                            self.row_mu,
                            self.J_world,
                            self.max_constraints,
                            self.max_world_dofs,
                            self.mf_constraint_count,
                            self.mf_rhs,
                            self.mf_impulses,
                            self._diag_prev_mf_impulses,
                            self.mf_row_type,
                            self.mf_row_parent,
                            self.mf_row_mu,
                            self.mf_J_a,
                            self.mf_J_b,
                            self.mf_dof_a,
                            self.mf_dof_b,
                            self.mf_max_constraints,
                            self.v_out,
                        ],
                        outputs=[self._diag_metrics],
                        device=self.model.device,
                    )

                    metrics_np = self._diag_metrics.numpy()
                    row = np.array(
                        [
                            np.max(metrics_np[:, 0]),
                            np.sum(metrics_np[:, 1]),
                            np.sum(metrics_np[:, 2]),
                            np.sum(metrics_np[:, 3]),
                        ]
                    )
                    self._pgs_convergence_log[-1].append(row)

                self._pgs_convergence_log[-1] = np.array(self._pgs_convergence_log[-1])

            else:
                wp.launch_tiled(
                    mf_gs_kernel,
                    dim=[self.world_count],
                    inputs=[
                        self.constraint_count,
                        self.world_dof_start,
                        self.rhs,
                        self.diag,
                        self.impulses,
                        self.J_world,
                        self.Y_world,
                        self.row_type,
                        self.row_parent,
                        self.row_mu,
                        self.mf_constraint_count,
                        self.mf_meta_packed,
                        self.mf_impulses,
                        self.mf_J_a,
                        self.mf_J_b,
                        self.mf_MiJt_a,
                        self.mf_MiJt_b,
                        self.mf_row_mu,
                        self.pgs_iterations,
                        self.pgs_omega,
                    ],
                    outputs=[self.v_out],
                    block_dim=32,
                    device=self.model.device,
                )

        # ══════════════════════════════════════════════════════════════
        # STAGE 7: Update qdd + integrate
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S7_Integrate", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            self._stage6_update_qdd(state_in, state_aug, dt)
            self._stage6_integrate(state_in, state_aug, state_out, dt)

        # Double-buffer: fork memset stream to zero current buffer for reuse.
        # ScopedStream(sync_enter=True) records an event on the main stream and
        # makes the memset stream wait — this is what forks it into graph capture.
        if self._memset_stream is not None:
            with wp.ScopedTimer("DB_Memset", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                with wp.ScopedStream(self._memset_stream):
                    for size in self.size_groups:
                        self._H_bufs[self._buf_idx][size].zero_()
                        self._J_bufs[self._buf_idx][size].zero_()
                self._memset_done_event[self._buf_idx] = self._memset_stream.record_event()
                self._buf_idx = 1 - self._buf_idx

        self._step += 1
        return state_out

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        """Populate Newton contact-force buffers from the last FeatherPGS solve."""
        if contacts is None or contacts.rigid_contact_count is None:
            return

        dt = self._last_step_dt
        inv_dt = 0.0 if dt is None or dt <= 0.0 else 1.0 / dt
        enable_friction_flag = 1 if self.enable_contact_friction else 0
        mf_impulses = getattr(self, "mf_impulses", None)
        if mf_impulses is None:
            mf_impulses = self._dummy_contact_impulses

        wp.launch(
            compute_contact_linear_force_from_impulses,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_normal,
                self.contact_world,
                self.contact_slot,
                self.contact_path,
                self.impulses,
                mf_impulses,
                enable_friction_flag,
                inv_dt,
            ],
            outputs=[contacts.rigid_contact_force],
            device=self.model.device,
        )

        if contacts.force is not None:
            wp.launch(
                pack_contact_linear_force_as_spatial,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_force,
                ],
                outputs=[contacts.force],
                device=self.model.device,
            )

    def _prepare_augmented_state(
        self,
        state_in: State,
        state_out: State,
        control: Control,
    ) -> State:
        requires_grad = state_in.requires_grad
        state_aug = state_out if requires_grad else self
        model = self.model

        if not getattr(state_aug, "_featherstone_augmented", False):
            self._allocate_state_aux_vars(model, state_aug, requires_grad)

        return state_aug

    def _allocate_state_aux_vars(self, model, target, requires_grad):
        # allocate auxiliary variables that vary with state
        if model.body_count:
            # joints
            target.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_tau = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            if requires_grad:
                # used in the custom grad implementation of trisolve_loop
                target.joint_solve_tmp = wp.zeros_like(model.joint_qd, requires_grad=True)
            else:
                target.joint_solve_tmp = None
            target.joint_S_s = wp.empty(
                (model.joint_dof_count,),
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=requires_grad,
            )

            # derived rigid body data (maximal coordinates)
            target.body_q_com = wp.empty_like(model.body_q, requires_grad=requires_grad)
            target.body_I_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=requires_grad
            )
            target.body_v_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_a_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_f_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_ft_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )

            target._featherstone_augmented = True

    def _eval_particle_forces(self, state_in: State, control: Control, contacts: Contacts):
        model = self.model

        particle_f = state_in.particle_f if state_in.particle_count else None
        body_f = state_in.body_f if state_in.body_count else None

        # damped springs
        eval_spring_forces(model, state_in, particle_f)

        # triangle elastic and lift/drag forces
        eval_triangle_forces(model, state_in, control, particle_f)

        # triangle bending
        eval_bending_forces(model, state_in, particle_f)

        # tetrahedral FEM
        eval_tetrahedra_forces(model, state_in, control, particle_f)

        # particle-particle interactions
        eval_particle_contact_forces(model, state_in, particle_f)

        # particle shape contact
        eval_particle_body_contact_forces(model, state_in, contacts, particle_f, body_f, body_f_in_world_frame=True)

    @contextmanager
    def _parallel_size_region(self, enabled: bool = True):
        """Context for parallel dispatch across size groups."""
        if not enabled or not self.use_parallel_streams or not self.model.device.is_cuda or len(self.size_groups) <= 1:
            yield
            return

        main_stream = wp.get_stream(self.model.device)
        self._main_stream = main_stream
        self._init_event = main_stream.record_event()
        try:
            yield
        finally:
            for size in self.size_groups:
                stream = self._size_streams.get(size)
                if stream is not None:
                    main_stream.wait_event(stream.record_event())
            self._main_stream = None
            self._init_event = None

    @contextmanager
    def _on_size_stream(self, size: int):
        """Execute block on this size's CUDA stream."""
        stream = self._size_streams.get(size)
        init_event = getattr(self, "_init_event", None)
        if stream is not None and init_event is not None:
            stream.wait_event(init_event)
            with wp.ScopedStream(stream):
                yield
        else:
            yield

    @contextmanager
    def _size_dispatch(self, enabled: bool):
        with self._parallel_size_region(enabled=enabled):
            yield

    @contextmanager
    def _size_ctx(self, size: int):
        with self._on_size_stream(size):
            yield

    def _for_sizes(self, enabled: bool):
        # convenience generator; keeps step code tight
        with self._size_dispatch(enabled):
            for size in self.size_groups:
                yield size, self._size_ctx(size)

    def _stage1_fk_id(self, state_in: State, state_aug: State, state_out: State):
        model = self.model

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
                self.body_X_com,
                model.joint_axis,
                model.joint_dof_dim,
            ],
            outputs=[state_in.body_q, state_aug.body_q_com],
            device=model.device,
        )

        wp.launch(
            update_articulation_origins,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                model.joint_child,
                state_in.body_q,
                model.body_com,
            ],
            outputs=[self.articulation_origin],
            device=model.device,
        )
        wp.launch(
            update_articulation_root_com_offsets,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                model.joint_child,
                state_in.body_q,
                model.body_com,
            ],
            outputs=[self.articulation_root_com_offset],
            device=model.device,
        )
        # evaluate joint inertias, motion vectors, and forces
        state_aug.body_f_s.zero_()
        wp.copy(self.qd_work, state_in.joint_qd)
        if self._has_root_free:
            wp.launch(
                convert_root_free_qd_world_to_local,
                dim=model.articulation_count,
                inputs=[
                    self.articulation_root_is_free,
                    self.articulation_root_dof_start,
                    self.articulation_root_com_offset,
                ],
                outputs=[self.qd_work],
                device=model.device,
            )

        wp.launch(
            eval_rigid_id,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_articulation,
                model.joint_qd_start,
                self.qd_work,
                model.joint_axis,
                model.joint_dof_dim,
                self.body_I_m,
                state_in.body_q,
                state_aug.body_q_com,
                model.joint_X_p,
                self.articulation_origin,
                model.gravity,
            ],
            outputs=[
                state_aug.joint_S_s,
                state_aug.body_I_s,
                state_aug.body_v_s,
                state_aug.body_f_s,
                state_aug.body_a_s,
            ],
            device=model.device,
        )
        if model.body_count:
            wp.launch(
                update_body_qd_from_featherstone,
                dim=model.body_count,
                inputs=[
                    state_aug.body_v_s,
                    state_in.body_q,
                    model.body_com,
                    self.body_to_articulation,
                    self.articulation_origin,
                ],
                outputs=[state_out.body_qd],
                device=model.device,
            )

    def _stage1_drives(self, state_in: State, state_aug: State, control: Control, dt: float):
        model = self.model

        if model.articulation_count:
            body_f = state_in.body_f if state_in.body_count else None
            # evaluate joint torques
            state_aug.body_ft_s.zero_()
            wp.launch(
                eval_rigid_tau,
                dim=model.articulation_count,
                inputs=[
                    model.articulation_start,
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_articulation,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    control.joint_f,
                    state_aug.joint_S_s,
                    state_aug.body_f_s,
                    body_f,
                    state_in.body_q,
                    model.body_com,
                    self.articulation_origin,
                ],
                outputs=[
                    state_aug.body_ft_s,
                    state_aug.joint_tau,
                ],
                device=model.device,
            )

            self.build_augmented_joint_targets(state_in, control, dt)
            self.apply_augmented_joint_tau(state_in, state_aug, dt)

            wp.launch(
                clamp_joint_tau,
                dim=model.joint_dof_count,
                inputs=[state_aug.joint_tau, model.joint_effort_limit],
                device=model.device,
            )

    def build_augmented_joint_targets(self, state_in: State, control: Control, dt: float):
        model = self.model
        if model.articulation_count == 0 or self.articulation_max_dofs == 0:
            return
        device = model.device

        self.aug_row_counts.zero_()
        self.aug_limit_counts.zero_()

        wp.launch(
            build_augmented_joint_rows,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.articulation_dof_start,
                self.articulation_H_rows,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_target_ke,
                model.joint_target_kd,
                state_in.joint_q,
                state_in.joint_qd,
                control.joint_target_pos,
                control.joint_target_vel,
                self.articulation_max_dofs,
                dt,
            ],
            outputs=[
                self.aug_row_counts,
                self.aug_row_dof_index,
                self.aug_row_K,
                self.aug_row_u0,
                self.aug_limit_counts,
            ],
            device=device,
        )

        wp.launch(
            detect_limit_count_changes,
            dim=model.articulation_count,
            inputs=[
                self.aug_limit_counts,
                self.aug_prev_limit_counts,
            ],
            outputs=[
                self.limit_change_mask,
            ],
            device=device,
        )

    def apply_augmented_joint_tau(self, state_in: State, state_aug: State, dt: float):
        model = self.model
        if model.articulation_count == 0 or self.articulation_max_dofs == 0:
            return

        wp.launch(
            apply_augmented_joint_tau,
            dim=model.articulation_count,
            inputs=[
                self.articulation_max_dofs,
                self.aug_row_counts,
                self.aug_row_dof_index,
                self.aug_row_u0,
            ],
            outputs=[state_aug.joint_tau],
            device=model.device,
        )

    def _stage1_crba(self, state_aug: State):
        model = self.model
        global_flag = 1 if ((self._step % self.update_mass_matrix_interval) == 0 or self._force_mass_update) else 0

        wp.launch(
            build_mass_update_mask,
            dim=model.articulation_count,
            inputs=[
                global_flag,
                self.limit_change_mask,
            ],
            outputs=[self.mass_update_mask],
            device=model.device,
        )

        wp.launch(
            eval_rigid_mass,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.articulation_M_start,
                self.mass_update_mask,
                state_aug.body_I_s,
            ],
            outputs=[self.M_blocks],
            device=model.device,
        )

        wp.launch(
            compute_composite_inertia,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.mass_update_mask,
                model.joint_ancestor,
                state_aug.body_I_s,
            ],
            outputs=[self.body_I_c],
            device=model.device,
            block_dim=128,
        )

        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            if self._H_bufs is None:  # not double-buffered
                self.H_by_size[size].zero_()
            wp.launch(
                crba_fill_par_dof,
                dim=int(n_arts * size),
                inputs=[
                    model.articulation_start,
                    self.articulation_dof_start,
                    self.mass_update_mask,
                    model.joint_ancestor,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    state_aug.joint_S_s,
                    self.body_I_c,
                    self.group_to_art[size],
                    size,
                ],
                outputs=[self.H_by_size[size]],
                device=model.device,
                block_dim=128,
            )

        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            wp.launch(
                apply_augmented_mass_diagonal_grouped,
                dim=n_arts,
                inputs=[
                    self.group_to_art[size],
                    self.articulation_dof_start,
                    size,
                    self.articulation_max_dofs,
                    self.mass_update_mask,
                    self.aug_row_counts,
                    self.aug_row_dof_index,
                    self.aug_row_K,
                ],
                outputs=[self.H_by_size[size]],
                device=model.device,
            )

        wp.launch(
            copy_int_array_masked,
            dim=model.articulation_count,
            inputs=[self.aug_limit_counts, self.mass_update_mask],
            outputs=[self.aug_prev_limit_counts],
            device=model.device,
        )

        self._force_mass_update = False

    def _stage2_cholesky_tiled(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        cholesky_kernel = TiledKernelFactory.get_cholesky_kernel(size, model.device)
        wp.launch_tiled(
            cholesky_kernel,
            dim=[n_arts],
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.group_to_art[size],
                self.mass_update_mask,
            ],
            outputs=[self.L_by_size[size]],
            block_dim=TILE_THREADS,
            device=model.device,
        )

    def _stage2_cholesky_loop(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            cholesky_loop,
            dim=n_arts,
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.group_to_art[size],
                self.mass_update_mask,
                size,
            ],
            outputs=[self.L_by_size[size]],
            device=model.device,
        )

    def _stage3_zero_qdd(self, state_aug: State):
        state_aug.joint_qdd.zero_()

    def _stage3_trisolve_tiled(self, size: int, state_aug: State):
        model = self.model
        n_arts = self.n_arts_by_size[size]

        wp.launch(
            gather_tau_to_groups,
            dim=n_arts,
            inputs=[
                state_aug.joint_tau,
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
            ],
            outputs=[self.tau_by_size[size]],
            device=model.device,
        )
        solve_kernel = TiledKernelFactory.get_triangular_solve_kernel(size, model.device)
        wp.launch_tiled(
            solve_kernel,
            dim=[n_arts],
            inputs=[
                self.L_by_size[size],
                self.tau_by_size[size],
            ],
            outputs=[self.qdd_by_size[size]],
            block_dim=TILE_THREADS,
            device=model.device,
        )
        wp.launch(
            scatter_qdd_from_groups,
            dim=n_arts,
            inputs=[
                self.qdd_by_size[size],
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
            ],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage3_trisolve_loop(self, size: int, state_aug: State):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            trisolve_loop,
            dim=n_arts,
            inputs=[
                self.L_by_size[size],
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
                state_aug.joint_tau,
            ],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage3_compute_v_hat(self, state_in: State, state_aug: State, dt: float):
        model = self.model
        if not model.joint_count:
            return
        wp.launch(
            compute_velocity_predictor,
            dim=model.joint_dof_count,
            inputs=[
                self.qd_work,
                state_aug.joint_qdd,
                dt,
            ],
            outputs=[self.v_hat],
            device=model.device,
        )

    def _stage4_build_rows(self, state_in: State, state_aug: State, contacts: Contacts):
        model = self.model
        max_constraints = self.max_constraints
        mf_active = self._has_free_rigid_bodies

        # Zero world-level buffers (only arrays that require it)
        self.slot_counter.zero_()  # atomic-add counter

        if mf_active:
            self.mf_slot_counter.zero_()  # atomic-add counter
            self.mf_constraint_count.zero_()  # finalize only runs when contacts exist
            self.mf_impulses.zero_()  # PGS reads before first write
            # mf_J_a/b, mf_MiJt_a/b: writers cover all used slots, readers gated by body >= 0
            # mf_body_a/b, mf_row_type, mf_row_parent, mf_row_mu, mf_phi: unconditionally overwritten
            # constraint_count: fully overwritten by finalize_world_constraint_counts

        has_free_rigid_flag = 1 if mf_active else 0
        # Dummy arrays when MF is not active (kernel still needs valid pointers)
        is_free_rigid = (
            self.is_free_rigid
            if self.is_free_rigid is not None
            else wp.zeros((1,), dtype=wp.int32, device=model.device)
        )
        mf_slot_counter = (
            self.mf_slot_counter if mf_active else wp.zeros((self.world_count,), dtype=wp.int32, device=model.device)
        )

        if (
            contacts is not None
            and getattr(contacts, "rigid_contact_count", None) is not None
            and contacts.rigid_contact_max > 0
        ):
            enable_friction_flag = 1 if self.enable_contact_friction else 0

            wp.launch(
                allocate_world_contact_slots,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    state_in.body_q,
                    model.shape_transform,
                    model.shape_body,
                    self.body_to_articulation,
                    self.art_to_world,
                    is_free_rigid,
                    has_free_rigid_flag,
                    max_constraints,
                    self.mf_max_constraints,
                    enable_friction_flag,
                ],
                outputs=[
                    self.contact_world,
                    self.contact_slot,
                    self.contact_art_a,
                    self.contact_art_b,
                    self.slot_counter,
                    self.contact_path,
                    mf_slot_counter,
                ],
                device=model.device,
            )

            # Allocate joint limit constraint slots (same counter as contacts)
            if self.enable_joint_limits and self.limit_slot is not None:
                wp.launch(
                    allocate_joint_limit_slots,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        self.articulation_dof_start,
                        self.articulation_H_rows,
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        state_in.joint_q,
                        self.art_to_world,
                        max_constraints,
                    ],
                    outputs=[
                        self.limit_slot,
                        self.limit_sign,
                        self.slot_counter,
                    ],
                    device=model.device,
                )

            if self._H_bufs is None:  # not double-buffered
                for size in self.size_groups:
                    self.J_by_size[size].zero_()

            for size in self.size_groups:
                wp.launch(
                    populate_world_J_for_size,
                    dim=contacts.rigid_contact_max,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        self.contact_world,
                        self.contact_slot,
                        self.contact_art_a,
                        self.contact_art_b,
                        self.contact_path,
                        size,  # target_size
                        self.art_size,
                        self.art_group_idx,
                        self.articulation_dof_start,
                        self.articulation_origin,
                        self.body_to_joint,
                        model.joint_ancestor,
                        model.joint_qd_start,
                        state_aug.joint_S_s,
                        model.shape_body,
                        state_in.body_q,
                        model.shape_transform,
                        self.shape_material_mu,
                        enable_friction_flag,
                        self.pgs_beta,
                        self.pgs_cfm,
                    ],
                    outputs=[
                        self.J_by_size[size],
                        self.row_type,
                        self.row_parent,
                        self.row_mu,
                        self.row_beta,
                        self.row_cfm,
                        self.phi,
                        self.target_velocity,
                    ],
                    device=model.device,
                )

            # Populate joint limit Jacobian rows (per size group)
            if self.enable_joint_limits and self.limit_slot is not None:
                for size in self.size_groups:
                    n_arts = self.n_arts_by_size[size]
                    wp.launch(
                        populate_joint_limit_J_for_size,
                        dim=n_arts,
                        inputs=[
                            model.articulation_start,
                            self.articulation_dof_start,
                            model.joint_type,
                            model.joint_q_start,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            state_in.joint_q,
                            self.art_to_world,
                            self.limit_slot,
                            self.limit_sign,
                            self.group_to_art[size],
                            self.pgs_beta,
                            self.pgs_cfm,
                        ],
                        outputs=[
                            self.J_by_size[size],
                            self.row_type,
                            self.row_parent,
                            self.row_mu,
                            self.row_beta,
                            self.row_cfm,
                            self.phi,
                            self.target_velocity,
                        ],
                        device=model.device,
                    )

            # Build MF contact rows
            if mf_active:
                wp.launch(
                    build_mf_contact_rows,
                    dim=contacts.rigid_contact_max,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        self.contact_world,
                        self.contact_slot,
                        self.contact_path,
                        self.contact_art_a,
                        self.contact_art_b,
                        self.articulation_origin,
                        model.shape_body,
                        state_in.body_q,
                        self.shape_material_mu,
                        enable_friction_flag,
                        self.pgs_beta,
                    ],
                    outputs=[
                        self.mf_body_a,
                        self.mf_body_b,
                        self.mf_J_a,
                        self.mf_J_b,
                        self.mf_row_type,
                        self.mf_row_parent,
                        self.mf_row_mu,
                        self.mf_phi,
                    ],
                    device=model.device,
                )

                slots_per_contact = 3 if self.enable_contact_friction else 1
                wp.launch(
                    finalize_mf_constraint_counts,
                    dim=self.world_count,
                    inputs=[self.mf_slot_counter, self.mf_max_constraints, slots_per_contact],
                    outputs=[self.mf_constraint_count],
                    device=model.device,
                )

        # Joint limit constraints (outside contact block — limits work with or without contacts)
        if self.enable_joint_limits and self.limit_slot is not None:
            has_contacts = (
                contacts is not None
                and getattr(contacts, "rigid_contact_count", None) is not None
                and contacts.rigid_contact_max > 0
            )
            if not has_contacts:
                # Contacts block was skipped — allocate limits and J from scratch
                wp.launch(
                    allocate_joint_limit_slots,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        self.articulation_dof_start,
                        self.articulation_H_rows,
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        state_in.joint_q,
                        self.art_to_world,
                        max_constraints,
                    ],
                    outputs=[
                        self.limit_slot,
                        self.limit_sign,
                        self.slot_counter,
                    ],
                    device=model.device,
                )
                if self._H_bufs is None:
                    for size in self.size_groups:
                        self.J_by_size[size].zero_()
                for size in self.size_groups:
                    n_arts = self.n_arts_by_size[size]
                    wp.launch(
                        populate_joint_limit_J_for_size,
                        dim=n_arts,
                        inputs=[
                            model.articulation_start,
                            self.articulation_dof_start,
                            model.joint_type,
                            model.joint_q_start,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            state_in.joint_q,
                            self.art_to_world,
                            self.limit_slot,
                            self.limit_sign,
                            self.group_to_art[size],
                            self.pgs_beta,
                            self.pgs_cfm,
                        ],
                        outputs=[
                            self.J_by_size[size],
                            self.row_type,
                            self.row_parent,
                            self.row_mu,
                            self.row_beta,
                            self.row_cfm,
                            self.phi,
                            self.target_velocity,
                        ],
                        device=model.device,
                    )

        slots_per_contact_dense = 3 if self.enable_contact_friction else 1
        wp.launch(
            finalize_world_constraint_counts,
            dim=self.world_count,
            inputs=[self.slot_counter, max_constraints, slots_per_contact_dense],
            outputs=[self.constraint_count],
            device=model.device,
        )

    def _stage4_zero_world_C(self):
        self.C.zero_()
        self.diag.zero_()

    def _stage4_hinv_jt_tiled(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        hinv_jt_kernel = TiledKernelFactory.get_hinv_jt_kernel(size, self.max_constraints, model.device)
        wp.launch_tiled(
            hinv_jt_kernel,
            dim=[n_arts],
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
            ],
            outputs=[self.Y_by_size[size]],
            block_dim=TILE_THREADS,
            device=model.device,
        )

    def _stage4_hinv_jt_tiled_fused(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        hinv_jt_kernel = TiledKernelFactory.get_hinv_jt_fused_kernel(size, self.max_constraints, model.device)
        wp.launch_tiled(
            hinv_jt_kernel,
            dim=[n_arts],
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                self.row_cfm,
            ],
            outputs=[self.C, self.diag, self.Y_by_size[size]],
            block_dim=TILE_THREADS,
            device=model.device,
        )

    def _stage4_hinv_jt_par_row(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            hinv_jt_par_row,
            dim=n_arts * self.max_constraints,
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.max_constraints,
                n_arts,
            ],
            outputs=[self.Y_by_size[size]],
            device=model.device,
        )

    def _stage4_delassus_par_row_col(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            delassus_par_row_col,
            dim=n_arts * self.max_constraints * self.max_constraints,
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.max_constraints,
                n_arts,
            ],
            outputs=[self.C, self.diag],
            device=model.device,
        )

    def _stage4_delassus_tiled(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        delassus_kernel = TiledKernelFactory.get_delassus_kernel(
            size, self.max_constraints, model.device, chunk_size=self.delassus_chunk_size
        )
        wp.launch_tiled(
            delassus_kernel,
            dim=[n_arts],
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                n_arts,
            ],
            outputs=[self.C, self.diag],
            block_dim=128,
            device=model.device,
        )

    def _stage4_finalize_world_diag_cfm(self):
        model = self.model
        wp.launch(
            finalize_world_diag_cfm,
            dim=self.world_count,
            inputs=[self.constraint_count, self.row_cfm],
            outputs=[self.diag],
            device=model.device,
        )

    def _stage4_add_contact_compliance(self, dt: float):
        if self.contact_compliance <= 0.0:
            return

        contact_alpha = float(self.contact_compliance / (dt * dt))
        wp.launch(
            add_contact_compliance_to_diag,
            dim=self.world_count,
            inputs=[self.constraint_count, self.row_type, contact_alpha],
            outputs=[self.diag],
            device=self.model.device,
        )

    def _stage4_extract_diag_from_JY(self, size: int):
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            extract_diag_from_JY_par_art,
            dim=n_arts * self.max_constraints,
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.max_constraints,
                n_arts,
            ],
            outputs=[self.diag],
            device=self.model.device,
        )

    def _stage4_compute_rhs_world(self, dt: float):
        model = self.model
        wp.launch(
            compute_world_contact_bias,
            dim=self.world_count,
            inputs=[
                self.constraint_count,
                self.max_constraints,
                self.phi,
                self.row_beta,
                self.row_type,
                self.target_velocity,
                dt,
            ],
            outputs=[self.rhs],
            device=model.device,
        )

    def _stage4_accumulate_rhs_world(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            rhs_accum_world_par_art,
            dim=n_arts,
            inputs=[
                self.constraint_count,
                self.max_constraints,
                self.art_to_world,
                self.art_size,
                self.art_group_idx,
                self.articulation_dof_start,
                self.v_hat,
                self.group_to_art[size],
                self.J_by_size[size],
                size,
            ],
            outputs=[self.rhs],
            device=model.device,
        )

    def _stage5_prepare_impulses_world(self):
        warmstart_flag = 1 if self.pgs_warmstart else 0
        wp.launch(
            prepare_world_impulses,
            dim=self.world_count,
            inputs=[self.constraint_count, self.max_constraints, warmstart_flag],
            outputs=[self.impulses],
            device=self.model.device,
        )

    def _stage6_prepare_world_velocity(self):
        wp.copy(self.v_out, self.v_hat)

    def _stage6_apply_impulses_world(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            apply_impulses_world_par_dof,
            dim=int(n_arts * size),
            inputs=[
                self.group_to_art[size],
                self.art_to_world,
                self.articulation_dof_start,
                size,
                n_arts,
                self.constraint_count,
                self.max_constraints,
                self.Y_by_size[size],
                self.impulses,
                self.v_hat,
            ],
            outputs=[self.v_out],
            device=model.device,
        )

    def _mf_pgs_setup(self, state_aug: State, dt: float):
        """MF PGS setup: compute Hinv, compute effective mass and RHS."""
        model = self.model

        # Compute H^-1 = inverse(body_I_s) for free rigid bodies
        wp.launch(
            compute_mf_body_Hinv,
            dim=model.body_count,
            inputs=[
                state_aug.body_I_s,
                self.is_free_rigid,
                self.body_to_articulation,
            ],
            outputs=[self.mf_body_Hinv],
            device=model.device,
        )

        # Compute effective mass and RHS
        self.mf_rhs.zero_()
        self.mf_eff_mass_inv.zero_()
        wp.launch(
            compute_mf_effective_mass_and_rhs,
            dim=self.world_count * self.mf_max_constraints,
            inputs=[
                self.mf_constraint_count,
                self.mf_body_a,
                self.mf_body_b,
                self.mf_J_a,
                self.mf_J_b,
                self.mf_body_Hinv,
                self.mf_phi,
                self.mf_row_type,
                self.pgs_cfm,
                self.pgs_beta,
                dt,
                self.mf_max_constraints,
            ],
            outputs=[
                self.mf_eff_mass_inv,
                self.mf_MiJt_a,
                self.mf_MiJt_b,
                self.mf_rhs,
            ],
            device=model.device,
        )

    def _stage6_update_qdd(self, state_in: State, state_aug: State, dt: float):
        model = self.model
        if self._has_root_free:
            wp.launch(
                convert_root_free_qd_local_to_world,
                dim=model.articulation_count,
                inputs=[
                    self.articulation_root_is_free,
                    self.articulation_root_dof_start,
                    self.articulation_root_com_offset,
                ],
                outputs=[self.v_out],
                device=model.device,
            )
        wp.launch(
            update_qdd_from_velocity,
            dim=model.joint_dof_count,
            inputs=[state_in.joint_qd, self.v_out, 1.0 / dt],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage6_integrate(self, state_in: State, state_aug: State, state_out: State, dt: float):
        model = self.model

        if model.joint_count:
            wp.launch(
                kernel=integrate_generalized_joints,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.body_com,
                    state_in.joint_q,
                    state_in.joint_qd,
                    state_aug.joint_qdd,
                    dt,
                ],
                outputs=[state_out.joint_q, state_out.joint_qd],
                device=model.device,
            )

            # Match Featherstone FK writeback so FREE/DISTANCE body_qd stores COM velocity.
            eval_fk_with_velocity_conversion(model, state_out.joint_q, state_out.joint_qd, state_out)

        self.integrate_particles(model, state_in, state_out, dt)


class TiledKernelFactory:
    """Factory for generating size-specialized tiled kernels for heterogeneous multi-articulation.

    This factory generates and caches tiled kernels specialized for specific DOF counts,
    enabling optimal tiled operations (Cholesky, triangular solves) for articulations
    with different numbers of degrees of freedom.

    The pattern follows ik_lbfgs_optimizer.py: kernels are generated on-demand with
    wp.constant() captured via closure, then cached by (dimensions, device.arch).
    """

    # Class-level caches: key -> compiled kernel
    _hinv_jt_cache: ClassVar[dict[tuple[int, int, str], "wp.Kernel"]] = {}
    _hinv_jt_fused_cache: ClassVar[dict[tuple[int, int, str], "wp.Kernel"]] = {}
    _cholesky_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _pgs_solve_mf_gs_cache: ClassVar[dict[tuple[int, int, str], "wp.Kernel"]] = {}
    _pack_mf_meta_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _triangular_solve_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _delassus_cache: ClassVar[dict[tuple[int, int, str], "wp.Kernel"]] = {}

    @classmethod
    def get_hinv_jt_kernel(cls, n_dofs: int, max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled H^-1*J^T kernel for the given dimensions."""
        key = (n_dofs, max_constraints, device.arch)
        if key not in cls._hinv_jt_cache:
            cls._hinv_jt_cache[key] = cls._build_hinv_jt_kernel(n_dofs, max_constraints)
        return cls._hinv_jt_cache[key]

    @classmethod
    def get_hinv_jt_fused_kernel(cls, n_dofs: int, max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled fused H^-1*J^T + Delassus kernel for the given dimensions."""
        key = (n_dofs, max_constraints, device.arch)
        if key not in cls._hinv_jt_fused_cache:
            cls._hinv_jt_fused_cache[key] = cls._build_hinv_jt_fused_kernel(n_dofs, max_constraints)
        return cls._hinv_jt_fused_cache[key]

    @classmethod
    def _build_hinv_jt_kernel(cls, n_dofs: int, max_constraints: int) -> "wp.Kernel":
        """Build specialized H^-1*J^T kernel for given dimensions.

        Solves Y = H^-1 * J^T using tiled Cholesky solve:
          L * L^T * Y = J^T
          => L * Z = J^T (forward solve)
          => L^T * Y = Z (backward solve)
        """
        # Create compile-time constants via closure
        # Convert to Python int to ensure wp.constant() accepts them
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))
        TILE_CONSTRAINTS_LOCAL = wp.constant(int(max_constraints))

        def hinv_jt_tiled_template(
            L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
            J_group: wp.array3d[float],  # [n_arts, max_c, n_dofs]
            group_to_art: wp.array[int],
            art_to_world: wp.array[int],
            world_constraint_count: wp.array[int],
            # output
            Y_group: wp.array3d[float],  # [n_arts, max_c, n_dofs]
        ):
            idx = wp.tid()
            art = group_to_art[idx]
            world = art_to_world[art]
            n_constraints = world_constraint_count[world]

            if n_constraints == 0:
                return

            # Load L (Cholesky factor) and J (Jacobian rows)
            L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            J_tile = wp.tile_load(J_group[idx], shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL), bounds_check=False)

            # Solve L * Z = J^T (forward substitution)
            # J_tile is (max_c x n_dofs), J^T is (n_dofs x max_c)
            Jt_tile = wp.tile_transpose(J_tile)
            Z_tile = wp.tile_lower_solve(L_tile, Jt_tile)

            # Solve L^T * Y = Z (backward substitution)
            Lt_tile = wp.tile_transpose(L_tile)
            X_tile = wp.tile_upper_solve(Lt_tile, Z_tile)

            # Store Y = H^-1 * J^T (transpose back to row layout)
            Y_out_tile = wp.tile_transpose(X_tile)
            wp.tile_store(Y_group[idx], Y_out_tile)

        hinv_jt_tiled_template.__name__ = f"hinv_jt_tiled_{n_dofs}_{max_constraints}"
        hinv_jt_tiled_template.__qualname__ = f"hinv_jt_tiled_{n_dofs}_{max_constraints}"
        return wp.kernel(enable_backward=False, module="unique")(hinv_jt_tiled_template)

    @classmethod
    def _build_hinv_jt_fused_kernel(cls, n_dofs: int, max_constraints: int) -> "wp.Kernel":
        """Build specialized fused H^-1*J^T + Delassus kernel for given dimensions."""
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))
        TILE_CONSTRAINTS_LOCAL = wp.constant(int(max_constraints))

        def hinv_jt_tiled_fused_template(
            L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
            J_group: wp.array3d[float],  # [n_arts, max_c, n_dofs]
            group_to_art: wp.array[int],
            art_to_world: wp.array[int],
            world_constraint_count: wp.array[int],
            row_cfm: wp.array2d[float],
            # outputs
            world_C: wp.array3d[float],  # [world_count, max_c, max_c]
            world_diag: wp.array2d[float],  # [world_count, max_c]
            Y_group: wp.array3d[float],  # [n_arts, max_c, n_dofs]
        ):
            idx, thread = wp.tid()
            art = group_to_art[idx]
            world = art_to_world[art]
            n_constraints = world_constraint_count[world]

            if n_constraints == 0:
                return

            # Load L (Cholesky factor) and J (Jacobian rows)
            L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            J_tile = wp.tile_load(J_group[idx], shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL), bounds_check=False)

            # Solve L * Z = J^T (forward substitution)
            Jt_tile = wp.tile_transpose(J_tile)
            Z_tile = wp.tile_lower_solve(L_tile, Jt_tile)

            # Solve L^T * Y = Z (backward substitution)
            Lt_tile = wp.tile_transpose(L_tile)
            X_tile = wp.tile_upper_solve(Lt_tile, Z_tile)

            # Store Y = H^-1 * J^T (transpose back to row layout)
            Y_out_tile = wp.tile_transpose(X_tile)
            wp.tile_store(Y_group[idx], Y_out_tile)

            # Form C = J * H^-1 * J^T
            C_tile = wp.tile_zeros(shape=(TILE_CONSTRAINTS_LOCAL, TILE_CONSTRAINTS_LOCAL), dtype=wp.float32)
            wp.tile_matmul(J_tile, X_tile, C_tile)
            wp.tile_store(world_C[world], C_tile)

            if thread == 0:
                for i in range(n_constraints):
                    world_diag[world, i] = C_tile[i, i] + row_cfm[world, i]

        hinv_jt_tiled_fused_template.__name__ = f"hinv_jt_tiled_fused_{n_dofs}_{max_constraints}"
        hinv_jt_tiled_fused_template.__qualname__ = f"hinv_jt_tiled_fused_{n_dofs}_{max_constraints}"
        return wp.kernel(enable_backward=False, module="unique")(hinv_jt_tiled_fused_template)

    @classmethod
    def get_delassus_kernel(
        cls, n_dofs: int, max_constraints: int, device: "wp.Device", chunk_size: int | None = None
    ) -> "wp.Kernel":
        """Get or create a streaming Delassus kernel for the given dimensions."""
        key = (n_dofs, max_constraints, device.arch, chunk_size)
        if key not in cls._delassus_cache:
            cls._delassus_cache[key] = cls._build_delassus_kernel(n_dofs, max_constraints, chunk_size)
        return cls._delassus_cache[key]

    @classmethod
    def _build_delassus_kernel(cls, n_dofs: int, max_constraints: int, chunk_size: int | None = None) -> "wp.Kernel":
        """Streaming Delassus: C += J * Y^T with shared memory."""
        TILE_D = n_dofs
        TILE_M = max_constraints
        if chunk_size is not None:
            CHUNK = chunk_size
        else:
            CHUNK = 64 if (2 * TILE_M * TILE_D * 4 > 45000) else TILE_M

        snippet = f"""
#if defined(__CUDA_ARCH__)
    const int TILE_D = {TILE_D};
    const int TILE_M = {TILE_M};
    const int CHUNK = {CHUNK};

    int lane = threadIdx.x;
    int art = group_to_art.data[idx];
    int world = art_to_world.data[art];
    int m = world_constraint_count.data[world];
    if (m == 0) return;

    __shared__ float s_J[CHUNK * TILE_D];
    __shared__ float s_Y[CHUNK * TILE_D];

    int num_chunks = (m + CHUNK - 1) / CHUNK;

    for (int ci = 0; ci < num_chunks; ci++) {{
        int i0 = ci * CHUNK, i1 = min(i0 + CHUNK, m);

        for (int t = lane; t < (i1 - i0) * TILE_D; t += blockDim.x)
            s_J[t] = J_group.data[idx * TILE_M * TILE_D + i0 * TILE_D + t];
        __syncthreads();

        for (int cj = 0; cj < num_chunks; cj++) {{
            int j0 = cj * CHUNK, j1 = min(j0 + CHUNK, m);

            for (int t = lane; t < (j1 - j0) * TILE_D; t += blockDim.x)
                s_Y[t] = Y_group.data[idx * TILE_M * TILE_D + j0 * TILE_D + t];
            __syncthreads();

            // Each thread computes multiple C elements
            for (int e = lane; e < (i1 - i0) * (j1 - j0); e += blockDim.x) {{
                int il = e / (j1 - j0), jl = e % (j1 - j0);
                float sum = 0.0f;
                for (int k = 0; k < TILE_D; k++)
                    sum += s_J[il * TILE_D + k] * s_Y[jl * TILE_D + k];
                if (sum != 0.0f) {{
                    int ig = i0 + il, jg = j0 + jl;
                    atomicAdd(&world_C.data[world * TILE_M * TILE_M + ig * TILE_M + jg], sum);
                    if (ig == jg) atomicAdd(&world_diag.data[world * TILE_M + ig], sum);
                }}
            }}
            __syncthreads();
        }}
    }}
#endif
"""

        @wp.func_native(snippet)
        def delassus_native(
            idx: int,
            J_group: wp.array3d[float],
            Y_group: wp.array3d[float],
            group_to_art: wp.array[int],
            art_to_world: wp.array[int],
            world_constraint_count: wp.array[int],
            world_C: wp.array3d[float],
            world_diag: wp.array2d[float],
        ): ...

        def delassus_template(
            J_group: wp.array3d[float],
            Y_group: wp.array3d[float],
            group_to_art: wp.array[int],
            art_to_world: wp.array[int],
            world_constraint_count: wp.array[int],
            n_arts: int,
            world_C: wp.array3d[float],
            world_diag: wp.array2d[float],
        ):
            idx, _lane = wp.tid()
            if idx < n_arts:
                delassus_native(
                    idx, J_group, Y_group, group_to_art, art_to_world, world_constraint_count, world_C, world_diag
                )

        delassus_template.__name__ = f"delassus_streaming_{n_dofs}_{max_constraints}_chunk{CHUNK}"
        delassus_template.__qualname__ = f"delassus_streaming_{n_dofs}_{max_constraints}_chunk{CHUNK}"
        return wp.kernel(enable_backward=False, module="unique")(delassus_template)

    @classmethod
    def get_cholesky_kernel(cls, n_dofs: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled Cholesky kernel for the given DOF count."""
        key = (n_dofs, device.arch)
        if key not in cls._cholesky_cache:
            cls._cholesky_cache[key] = cls._build_cholesky_kernel(n_dofs)
        return cls._cholesky_cache[key]

    @classmethod
    def _build_cholesky_kernel(cls, n_dofs: int) -> "wp.Kernel":
        """Build specialized Cholesky kernel for given DOF count.

        Computes L such that H + diag(armature) = L * L^T.
        """
        # Convert to Python int to ensure wp.constant() accepts them
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))

        def cholesky_tiled_template(
            H_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
            R_group: wp.array2d[float],  # [n_arts, n_dofs] armature
            group_to_art: wp.array[int],
            mass_update_mask: wp.array[int],
            # output
            L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
        ):
            idx = wp.tid()
            art = group_to_art[idx]

            if mass_update_mask[art] == 0:
                return

            # Load H and armature
            H_tile = wp.tile_load(H_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            armature = wp.tile_load(R_group[idx], shape=(TILE_DOF_LOCAL,), bounds_check=False)

            # Add armature to diagonal
            H_tile = wp.tile_diag_add(H_tile, armature)

            # Compute Cholesky factorization
            L_tile = wp.tile_cholesky(H_tile)

            # Store result
            wp.tile_store(L_group[idx], L_tile)

        cholesky_tiled_template.__name__ = f"cholesky_tiled_{n_dofs}"
        cholesky_tiled_template.__qualname__ = f"cholesky_tiled_{n_dofs}"
        return wp.kernel(enable_backward=False, module="unique")(cholesky_tiled_template)

    @classmethod
    def get_triangular_solve_kernel(cls, n_dofs: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled triangular solve kernel for the given DOF count."""
        key = (n_dofs, device.arch)
        if key not in cls._triangular_solve_cache:
            cls._triangular_solve_cache[key] = cls._build_triangular_solve_kernel(n_dofs)
        return cls._triangular_solve_cache[key]

    @classmethod
    def _build_triangular_solve_kernel(cls, n_dofs: int) -> "wp.Kernel":
        """Build specialized triangular solve kernel for given DOF count.

        Solves L * L^T * x = b for x using tiled forward and backward substitution.
        """
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))

        def trisolve_tiled_template(
            L_group: wp.array3d[float],  # [n_arts, n_dofs, n_dofs]
            tau_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
            qdd_group: wp.array3d[float],  # [n_arts, n_dofs, 1]
        ):
            idx = wp.tid()
            L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            tau_tile = wp.tile_load(tau_group[idx], shape=(TILE_DOF_LOCAL, 1), bounds_check=False)

            # Forward substitution: L * z = tau
            z_tile = wp.tile_lower_solve(L_tile, tau_tile)

            # Backward substitution: L^T * qdd = z
            Lt_tile = wp.tile_transpose(L_tile)
            qdd_tile = wp.tile_upper_solve(Lt_tile, z_tile)

            wp.tile_store(qdd_group[idx], qdd_tile)

        trisolve_tiled_template.__name__ = f"trisolve_tiled_{n_dofs}"
        trisolve_tiled_template.__qualname__ = f"trisolve_tiled_{n_dofs}"
        return wp.kernel(enable_backward=False, module="unique")(trisolve_tiled_template)

    @classmethod
    def get_pack_mf_meta_kernel(cls, mf_max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a kernel to pack MF metadata into int4 format."""
        key = (mf_max_constraints, device.arch)
        if key not in cls._pack_mf_meta_cache:
            cls._pack_mf_meta_cache[key] = cls._build_pack_mf_meta_kernel(mf_max_constraints)
        return cls._pack_mf_meta_cache[key]

    @classmethod
    def _build_pack_mf_meta_kernel(cls, mf_max_constraints: int) -> "wp.Kernel":
        """Build a kernel to pack MF constraint metadata into int4 structs.

        Packs dof_a, dof_b, eff_mass_inv, rhs, row_type, row_parent into
        4 contiguous int32s per constraint for 128-bit coalesced loads.
        """
        M_MF = mf_max_constraints

        snippet = f"""
    #if defined(__CUDA_ARCH__)
        int lane = threadIdx.x;
        int m_mf = mf_constraint_count.data[world];
        int off_mf = world * {M_MF};
        int off_meta = off_mf * 4;

        for (int i = lane; i < m_mf; i += 32) {{
            int da = mf_dof_a.data[off_mf + i];
            int db = mf_dof_b.data[off_mf + i];
            float diag = mf_eff_mass_inv.data[off_mf + i];
            float rhs_val = mf_rhs.data[off_mf + i];
            int rt = mf_row_type.data[off_mf + i];
            int par = mf_row_parent.data[off_mf + i];

            int4 packed;
            packed.x = (da << 16) | (db & 0xFFFF);
            packed.y = __float_as_int(diag);
            packed.z = __float_as_int(rhs_val);
            packed.w = rt | (par << 16);
            *reinterpret_cast<int4*>(&mf_meta.data[off_meta + i * 4]) = packed;
        }}
    #endif
    """

        @wp.func_native(snippet)
        def pack_mf_meta_native(
            world: int,
            mf_constraint_count: wp.array[int],
            mf_dof_a: wp.array2d[int],
            mf_dof_b: wp.array2d[int],
            mf_eff_mass_inv: wp.array2d[float],
            mf_rhs: wp.array2d[float],
            mf_row_type: wp.array2d[int],
            mf_row_parent: wp.array2d[int],
            mf_meta: wp.array2d[int],
        ): ...

        def pack_mf_meta_template(
            mf_constraint_count: wp.array[int],
            mf_dof_a: wp.array2d[int],
            mf_dof_b: wp.array2d[int],
            mf_eff_mass_inv: wp.array2d[float],
            mf_rhs: wp.array2d[float],
            mf_row_type: wp.array2d[int],
            mf_row_parent: wp.array2d[int],
            mf_meta: wp.array2d[int],
        ):
            world, _lane = wp.tid()
            pack_mf_meta_native(
                world,
                mf_constraint_count,
                mf_dof_a,
                mf_dof_b,
                mf_eff_mass_inv,
                mf_rhs,
                mf_row_type,
                mf_row_parent,
                mf_meta,
            )

        name = f"pack_mf_meta_{mf_max_constraints}"
        pack_mf_meta_template.__name__ = name
        pack_mf_meta_template.__qualname__ = name
        return wp.kernel(enable_backward=False, module="unique")(pack_mf_meta_template)

    @classmethod
    def get_pgs_solve_mf_gs_kernel(
        cls, max_constraints: int, mf_max_constraints: int, max_world_dofs: int, device: "wp.Device"
    ) -> "wp.Kernel":
        """Get or create the unified GS kernel for the matrix-free winner path.

        Articulated rows use the gathered world-space `J_world` and `Y_world`
        inputs, while free-rigid rows use the packed matrix-free metadata. Both
        row sets share one world-velocity tile in shared memory.
        """
        key = (max_constraints, mf_max_constraints, max_world_dofs, device.arch)
        if key not in cls._pgs_solve_mf_gs_cache:
            cls._pgs_solve_mf_gs_cache[key] = cls._build_pgs_solve_mf_gs_kernel(
                max_constraints, mf_max_constraints, max_world_dofs
            )
        return cls._pgs_solve_mf_gs_cache[key]

    @classmethod
    def _build_pgs_solve_mf_gs_kernel(
        cls, max_constraints: int, mf_max_constraints: int, max_world_dofs: int
    ) -> "wp.Kernel":
        """Build the unified GS kernel for articulated and free-rigid rows.

        Uses one warp (32 threads) per world.

        The articulated-contact rows use warp-parallel dot/update work over the
        world DOF tile. The free-rigid rows use 6-DOF body slices for body A and
        body B while reusing the same shared velocity state.

        Shared memory layout:
          s_v[D] — world velocity
          s_lam_dense[M_D] + metadata — articulated-row impulses and row info
          s_lam_mf[M_MF] — free-rigid impulses (metadata read from global)
        """
        M_D = max_constraints
        M_MF = mf_max_constraints
        D = max_world_dofs

        # How many DOF elements each lane handles (ceil(D/32))
        ELEMS_PER_LANE = (D + 31) // 32

        # --- Code generation for articulated rows (D-wide dot/update, software-pipelined) ---

        # Pipeline register declarations
        dense_pipe_decl = "\n".join(
            [f"        float pre_dJ_{k} = 0.0f, pre_dY_{k} = 0.0f;" for k in range(ELEMS_PER_LANE)]
        )

        # Initial prefetch (constraint 0)
        dense_prefetch_init_parts = []
        for k in range(ELEMS_PER_LANE):
            d_expr = f"lane + {k * 32}" if k > 0 else "lane"
            dense_prefetch_init_parts.append(f"""
                if ({d_expr} < {D}) {{
                    pre_dJ_{k} = J_world.data[jy_world_base + {d_expr}];
                    pre_dY_{k} = Y_world.data[jy_world_base + {d_expr}];
                }}""")
        dense_prefetch_init_code = "\n".join(dense_prefetch_init_parts)

        # Consume prefetched values into cur_ variables
        dense_consume_code = "\n".join(
            [f"                float cur_dJ_{k} = pre_dJ_{k}, cur_dY_{k} = pre_dY_{k};" for k in range(ELEMS_PER_LANE)]
        )

        # Prefetch next constraint (i+1)
        dense_prefetch_next_parts = []
        for k in range(ELEMS_PER_LANE):
            d_expr = f"lane + {k * 32}" if k > 0 else "lane"
            dense_prefetch_next_parts.append(f"""
                    if ({d_expr} < {D}) {{
                        pre_dJ_{k} = J_world.data[next_jy_base + {d_expr}];
                        pre_dY_{k} = Y_world.data[next_jy_base + {d_expr}];
                    }}""")
        dense_prefetch_next_code = "\n".join(dense_prefetch_next_parts)

        # Dense dot product using prefetched J: cur_dJ_k * s_v[d]
        dense_dot_parts = []
        for k in range(ELEMS_PER_LANE):
            d_expr = f"lane + {k * 32}" if k > 0 else "lane"
            dense_dot_parts.append(f"""
                if ({d_expr} < {D}) {{
                    my_sum += cur_dJ_{k} * s_v[{d_expr}];
                }}""")
        dense_dot_code = "\n".join(["float my_sum = 0.0f;", *dense_dot_parts])

        # Dense v update using prefetched Y: s_v[d] += cur_dY_k * delta
        dense_v_update_parts = []
        for k in range(ELEMS_PER_LANE):
            d_expr = f"lane + {k * 32}" if k > 0 else "lane"
            dense_v_update_parts.append(f"""
                if ({d_expr} < {D}) {{
                    s_v[{d_expr}] += cur_dY_{k} * delta_impulse;
                }}""")
        dense_v_update_code = "\n".join(dense_v_update_parts)

        # Dense sibling v update — NOT pipelined (random sib index)
        dense_sib_v_parts = []
        for k in range(ELEMS_PER_LANE):
            d_expr = f"lane + {k * 32}" if k > 0 else "lane"
            dense_sib_v_parts.append(f"""
                    if ({d_expr} < {D}) {{
                        s_v[{d_expr}] += Y_world.data[sib_row_base + {d_expr}] * sib_delta;
                    }}""")
        dense_sib_v_code = "\n".join(dense_sib_v_parts)

        snippet = f"""
    #if defined(__CUDA_ARCH__)
        const unsigned MASK = 0xFFFFFFFF;
        int lane = threadIdx.x;

        int m_dense = world_constraint_count.data[world];
        int m_mf = mf_constraint_count.data[world];
        if (m_dense == 0 && m_mf == 0) return;
        if (m_dense > {M_D}) m_dense = {M_D};
        if (m_mf > {M_MF}) m_mf = {M_MF};

        int w_dof_start = world_dof_start.data[world];
        int off_dense = world * {M_D};
        int off_mf = world * {M_MF};
        int off_meta = off_mf * 4;
        int jy_world_base = world * {M_D} * {D};
        int mf6_base = world * {M_MF} * 6;

        // ═══════════════════════════════════════════════════════
        // SHARED MEMORY
        // ═══════════════════════════════════════════════════════
        __shared__ float s_v[{D}];
        __shared__ float s_lam_dense[{M_D}];
        __shared__ float s_rhs_dense[{M_D}];
        __shared__ float s_diag_dense[{M_D}];
        __shared__ int   s_rtype_dense[{M_D}];
        __shared__ int   s_parent_dense[{M_D}];
        __shared__ float s_mu_dense[{M_D}];
        __shared__ float s_lam_mf[{M_MF}];

        // ═══════════════════════════════════════════════════════
        // LOAD PHASE
        // ═══════════════════════════════════════════════════════
        for (int i = lane; i < m_dense; i += 32) {{
            s_lam_dense[i] = world_impulses.data[off_dense + i];
            s_rhs_dense[i] = rhs_bias.data[off_dense + i];
            s_diag_dense[i] = world_diag.data[off_dense + i];
            s_rtype_dense[i] = world_row_type.data[off_dense + i];
            s_parent_dense[i] = world_row_parent.data[off_dense + i];
            s_mu_dense[i] = world_row_mu.data[off_dense + i];
        }}
        for (int i = lane; i < m_mf; i += 32) {{
            s_lam_mf[i] = mf_impulses.data[off_mf + i];
        }}
        for (int d = lane; d < {D}; d += 32) {{
            s_v[d] = v_out.data[w_dof_start + d];
        }}
        __syncwarp();

        // ═══════════════════════════════════════════════════════
        // SOLVE PHASE
        // ═══════════════════════════════════════════════════════
        // Dense pipeline registers
{dense_pipe_decl}

        for (int iter = 0; iter < iterations; iter++) {{

            // ── Articulated rows (D-DOF warp-parallel, software-pipelined) ──

            // Prefetch constraint 0
            if (m_dense > 0) {{
                {dense_prefetch_init_code}
            }}

            for (int i = 0; i < m_dense; i++) {{
                // Consume prefetched J/Y for constraint i
                {dense_consume_code}

                // Prefetch constraint i+1
                if (i + 1 < m_dense) {{
                    int next_jy_base = jy_world_base + (i + 1) * {D};
                    {dense_prefetch_next_code}
                }}

                float denom = s_diag_dense[i];
                if (denom <= 0.0f) continue;

                // J_i · v (using prefetched J)
                {dense_dot_code}

                // Warp reduce
                my_sum += __shfl_down_sync(MASK, my_sum, 16);
                my_sum += __shfl_down_sync(MASK, my_sum, 8);
                my_sum += __shfl_down_sync(MASK, my_sum, 4);
                my_sum += __shfl_down_sync(MASK, my_sum, 2);
                my_sum += __shfl_down_sync(MASK, my_sum, 1);
                float jv = __shfl_sync(MASK, my_sum, 0);

                float residual = jv + s_rhs_dense[i];
                float delta = -residual / denom;
                float old_impulse = s_lam_dense[i];
                float new_impulse = old_impulse + omega * delta;
                int row_type = s_rtype_dense[i];

                if (row_type == 0 || row_type == 3) {{
                    if (new_impulse < 0.0f) new_impulse = 0.0f;
                }} else if (row_type == 2) {{
                    int parent_idx = s_parent_dense[i];
                    float lambda_n = s_lam_dense[parent_idx];
                    float mu = s_mu_dense[i];
                    float radius = fmaxf(mu * lambda_n, 0.0f);

                    if (radius <= 0.0f) {{
                        new_impulse = 0.0f;
                    }} else {{
                        int sib = (i == parent_idx + 1) ? parent_idx + 2 : parent_idx + 1;
                        s_lam_dense[i] = new_impulse;
                        float a_val = new_impulse;
                        float b_val = s_lam_dense[sib];
                        float mag = sqrtf(a_val * a_val + b_val * b_val);
                        if (mag > radius) {{
                            float scale = radius / mag;
                            new_impulse = a_val * scale;
                            float sib_new = b_val * scale;
                            float sib_delta = sib_new - b_val;
                            s_lam_dense[sib] = sib_new;

                            int sib_row_base = jy_world_base + sib * {D};
                            {dense_sib_v_code}
                        }}
                    }}
                }}

                float delta_impulse = new_impulse - old_impulse;
                s_lam_dense[i] = new_impulse;

                // V update using prefetched Y
                if (delta_impulse != 0.0f) {{
                    {dense_v_update_code}
                }}
                __syncwarp();
            }}

            // ── Free-rigid rows (6-DOF per body, software-pipelined) ──

            // Pipeline registers: prefetch next constraint's global data
            int4 pre_meta;
            float pre_Ja = 0.0f, pre_Jb = 0.0f;
            float pre_MiJta = 0.0f, pre_MiJtb = 0.0f;

            // Prefetch constraint 0
            if (m_mf > 0) {{
                pre_meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta]);
                if (lane < 6) {{
                    pre_Ja = mf_J_a.data[mf6_base + lane];
                    pre_MiJta = mf_MiJt_a.data[mf6_base + lane];
                }}
                if (lane >= 6 && lane < 12) {{
                    pre_Jb = mf_J_b.data[mf6_base + lane - 6];
                    pre_MiJtb = mf_MiJt_b.data[mf6_base + lane - 6];
                }}
            }}

            for (int i = 0; i < m_mf; i++) {{
                // Consume prefetched data for constraint i
                int4 meta = pre_meta;
                float cur_Ja = pre_Ja;
                float cur_Jb = pre_Jb;
                float cur_MiJta = pre_MiJta;
                float cur_MiJtb = pre_MiJtb;

                // Prefetch constraint i+1 (loads issued now, complete during compute)
                if (i + 1 < m_mf) {{
                    int next_mf6 = mf6_base + (i + 1) * 6;
                    pre_meta = *reinterpret_cast<const int4*>(&mf_meta.data[off_meta + (i + 1) * 4]);
                    if (lane < 6) {{
                        pre_Ja = mf_J_a.data[next_mf6 + lane];
                        pre_MiJta = mf_MiJt_a.data[next_mf6 + lane];
                    }}
                    if (lane >= 6 && lane < 12) {{
                        pre_Jb = mf_J_b.data[next_mf6 + lane - 6];
                        pre_MiJtb = mf_MiJt_b.data[next_mf6 + lane - 6];
                    }}
                }}

                // Process constraint i
                int packed_dofs = meta.x;
                int dof_a = packed_dofs >> 16;
                int dof_b = (packed_dofs << 16) >> 16;
                float mf_diag = __int_as_float(meta.y);

                if (mf_diag <= 0.0f) continue;

                // J · v using prefetched J values
                float my_sum = 0.0f;
                if (lane < 6 && dof_a >= 0) {{
                    my_sum = cur_Ja * s_v[dof_a + lane];
                }}
                if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                    my_sum = cur_Jb * s_v[dof_b + lane - 6];
                }}
                my_sum += __shfl_down_sync(MASK, my_sum, 16);
                my_sum += __shfl_down_sync(MASK, my_sum, 8);
                my_sum += __shfl_down_sync(MASK, my_sum, 4);
                my_sum += __shfl_down_sync(MASK, my_sum, 2);
                my_sum += __shfl_down_sync(MASK, my_sum, 1);
                float jv = __shfl_sync(MASK, my_sum, 0);

                float residual = jv + __int_as_float(meta.z);
                float delta = -residual * mf_diag;
                float old_impulse = s_lam_mf[i];
                float new_impulse = old_impulse + omega * delta;
                int packed_tp = meta.w;
                int mf_rt = packed_tp & 0xFFFF;

                if (mf_rt == 0) {{
                    if (new_impulse < 0.0f) new_impulse = 0.0f;
                }} else if (mf_rt == 2) {{
                    int mf_par = packed_tp >> 16;
                    float lambda_n = s_lam_mf[mf_par];
                    float mu = mf_row_mu.data[off_mf + i];
                    float radius = fmaxf(mu * lambda_n, 0.0f);

                    if (radius <= 0.0f) {{
                        new_impulse = 0.0f;
                    }} else {{
                        int sib = (i == mf_par + 1) ? mf_par + 2 : mf_par + 1;
                        s_lam_mf[i] = new_impulse;
                        float a_val = new_impulse;
                        float b_val = s_lam_mf[sib];
                        float mag = sqrtf(a_val * a_val + b_val * b_val);
                        if (mag > radius) {{
                            float scale = radius / mag;
                            new_impulse = a_val * scale;
                            float sib_new = b_val * scale;
                            float sib_delta = sib_new - b_val;
                            s_lam_mf[sib] = sib_new;

                            // Sibling v update (can't prefetch — random sib index)
                            int sib_packed_dofs = mf_meta.data[off_meta + sib * 4];
                            int sib_dof_a = sib_packed_dofs >> 16;
                            int sib_dof_b = (sib_packed_dofs << 16) >> 16;
                            int sib_mf6 = mf6_base + sib * 6;
                            if (lane < 6 && sib_dof_a >= 0) {{
                                s_v[sib_dof_a + lane] += mf_MiJt_a.data[sib_mf6 + lane] * sib_delta;
                            }}
                            if (lane >= 6 && lane < 12 && sib_dof_b >= 0) {{
                                s_v[sib_dof_b + lane - 6] += mf_MiJt_b.data[sib_mf6 + lane - 6] * sib_delta;
                            }}
                        }}
                    }}
                }}

                float delta_impulse = new_impulse - old_impulse;
                s_lam_mf[i] = new_impulse;

                // V update using prefetched MiJt values
                if (delta_impulse != 0.0f) {{
                    if (lane < 6 && dof_a >= 0) {{
                        s_v[dof_a + lane] += cur_MiJta * delta_impulse;
                    }}
                    if (lane >= 6 && lane < 12 && dof_b >= 0) {{
                        s_v[dof_b + lane - 6] += cur_MiJtb * delta_impulse;
                    }}
                }}
                __syncwarp();
            }}
        }}

        // ═══════════════════════════════════════════════════════
        // STORE PHASE
        // ═══════════════════════════════════════════════════════
        for (int d = lane; d < {D}; d += 32) {{
            v_out.data[w_dof_start + d] = s_v[d];
        }}
        for (int i = lane; i < m_dense; i += 32) {{
            world_impulses.data[off_dense + i] = s_lam_dense[i];
        }}
        for (int i = lane; i < m_mf; i += 32) {{
            mf_impulses.data[off_mf + i] = s_lam_mf[i];
        }}
    #endif
    """

        @wp.func_native(snippet)
        def pgs_solve_mf_gs_native(
            world: int,
            # Dense
            world_constraint_count: wp.array[int],
            world_dof_start: wp.array[int],
            rhs_bias: wp.array2d[float],
            world_diag: wp.array2d[float],
            world_impulses: wp.array2d[float],
            J_world: wp.array3d[float],
            Y_world: wp.array3d[float],
            world_row_type: wp.array2d[int],
            world_row_parent: wp.array2d[int],
            world_row_mu: wp.array2d[float],
            # MF
            mf_constraint_count: wp.array[int],
            mf_meta: wp.array2d[int],
            mf_impulses: wp.array2d[float],
            mf_J_a: wp.array3d[float],
            mf_J_b: wp.array3d[float],
            mf_MiJt_a: wp.array3d[float],
            mf_MiJt_b: wp.array3d[float],
            mf_row_mu: wp.array2d[float],
            # Shared
            iterations: int,
            omega: float,
            # Output
            v_out: wp.array[float],
        ): ...

        def pgs_solve_mf_gs_template(
            # Dense
            world_constraint_count: wp.array[int],
            world_dof_start: wp.array[int],
            rhs_bias: wp.array2d[float],
            world_diag: wp.array2d[float],
            world_impulses: wp.array2d[float],
            J_world: wp.array3d[float],
            Y_world: wp.array3d[float],
            world_row_type: wp.array2d[int],
            world_row_parent: wp.array2d[int],
            world_row_mu: wp.array2d[float],
            # MF
            mf_constraint_count: wp.array[int],
            mf_meta: wp.array2d[int],
            mf_impulses: wp.array2d[float],
            mf_J_a: wp.array3d[float],
            mf_J_b: wp.array3d[float],
            mf_MiJt_a: wp.array3d[float],
            mf_MiJt_b: wp.array3d[float],
            mf_row_mu: wp.array2d[float],
            # Shared
            iterations: int,
            omega: float,
            # Output
            v_out: wp.array[float],
        ):
            world, _lane = wp.tid()
            pgs_solve_mf_gs_native(
                world,
                world_constraint_count,
                world_dof_start,
                rhs_bias,
                world_diag,
                world_impulses,
                J_world,
                Y_world,
                world_row_type,
                world_row_parent,
                world_row_mu,
                mf_constraint_count,
                mf_meta,
                mf_impulses,
                mf_J_a,
                mf_J_b,
                mf_MiJt_a,
                mf_MiJt_b,
                mf_row_mu,
                iterations,
                omega,
                v_out,
            )

        name = f"pgs_solve_mf_gs_{max_constraints}_{mf_max_constraints}_{max_world_dofs}"
        pgs_solve_mf_gs_template.__name__ = name
        pgs_solve_mf_gs_template.__qualname__ = name
        return wp.kernel(enable_backward=False, module="unique")(pgs_solve_mf_gs_template)
