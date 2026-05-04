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
from ...sim import Contacts, Control, Model, State
from ...sim.enums import JointType
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
    FRICTION_MODE_CURRENT,
    TILE_THREADS,
    add_dense_contact_compliance_to_diag,
    allocate_joint_limit_slots,
    allocate_joint_velocity_limit_slots,
    allocate_world_contact_slots,
    apply_augmented_joint_tau,
    apply_augmented_mass_diagonal_grouped,
    apply_impulses_world_par_dof,
    build_augmented_joint_rows,
    build_mass_update_mask,
    build_mf_body_map,
    build_mf_contact_rows,
    cholesky_loop,
    clamp_augmented_joint_u0,
    compute_com_transforms,
    compute_composite_inertia,
    compute_contact_linear_force_from_impulses,
    compute_delta_and_accumulate,
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
    diag_from_JY_par_art,
    eval_rigid_fk,
    eval_rigid_id,
    eval_rigid_mass,
    eval_rigid_tau,
    finalize_mf_constraint_counts,
    finalize_world_constraint_counts,
    finalize_world_diag_cfm,
    gather_JY_to_world,
    gather_tau_to_groups,
    hinv_jt_par_row,
    integrate_generalized_joints,
    pack_contact_linear_force_as_spatial,
    pack_contact_triplets_vec3,
    pack_mf_meta,
    pgs_convergence_diagnostic_velocity,
    pgs_ncp_residuals_diagnostic_velocity,
    pgs_solve_loop,
    pgs_solve_mf_loop,
    populate_joint_limit_J_for_size,
    populate_joint_velocity_limit_J_for_size,
    populate_world_J_for_size,
    prepare_world_impulses,
    rhs_accum_world_par_art,
    scatter_qdd_from_groups,
    trisolve_loop,
    update_articulation_origins,
    update_articulation_root_com_offsets,
    update_body_qd_from_featherstone,
    update_qdd_from_velocity,
    vector_add_inplace,
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

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Instead of maximal coordinates :attr:`~newton.State.body_q` (rigid body positions) and :attr:`~newton.State.body_qd`
    (rigid body velocities) as is the case in :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`,
    :class:`~newton.solvers.SolverFeatherPGS` uses :attr:`~newton.State.joint_q` and :attr:`~newton.State.joint_qd` to represent
    the positions and velocities of joints without allowing any redundant degrees of freedom.

    After constructing :class:`~newton.Model` and :class:`~newton.State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Note:
        Unlike :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverFeatherPGS`
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
        enable_joint_velocity_limits: bool = False,
        pgs_iterations: int = 12,
        pgs_beta: float = 0.2,
        pgs_cfm: float = 1.0e-6,
        contact_compliance: float = 0.0,
        pgs_omega: float = 1.0,
        max_constraints: int = 33,
        pgs_warmstart: bool = False,
        mf_max_constraints: int = 512,
        use_parallel_streams: bool = True,
        double_buffer: bool = True,
        nvtx: bool = False,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            angular_damping (float, optional): Angular damping factor. Defaults to 0.05.
            update_mass_matrix_interval (int, optional): How often to update the mass matrix (every n-th time the :meth:`step` function gets called). Defaults to 1.
            friction_smoothing (float, optional): The delta value for the Huber norm (see :func:`warp.math.norm_huber`) used for the friction velocity normalization. Defaults to 1.0.
            enable_contact_friction (bool, optional): Enables Coulomb friction contacts inside the PGS solve. Defaults to True.
            enable_joint_limits (bool, optional): Enforce joint position limits as unilateral PGS
                constraints.  Each violated limit adds one constraint row. Defaults to False.
            enable_joint_velocity_limits (bool, optional): Enforce joint velocity limits
                (``model.joint_velocity_limit``) as per-DOF PGS constraints. Mirrors the
                PhysX velocity-limit row: when ``|qdot_i| > qdot_max_i``, a single
                signed-Jacobian row is added that projects ``qdot_i`` back onto the
                bilateral box ``[-qdot_max_i, +qdot_max_i]`` through the articulated-body
                response ``J M^-1 J^T + cfm``, so momentum redistributes correctly across
                the articulation (rather than a naive joint-space clip which breaks
                multi-body Newton's third law). No Baumgarte bias.
                Defaults to False.
            pgs_iterations (int, optional): Number of Gauss-Seidel iterations to apply per frame. Defaults to 12.
            pgs_beta (float, optional): ERP style position correction factor. Defaults to 0.2.
            pgs_cfm (float, optional): Compliance/regularization added to the Delassus diagonal. Defaults to 1.0e-6.
            contact_compliance (float, optional): Normal contact compliance [m/N] applied
                to articulated contact rows. Converted to an impulse-space diagonal term
                using ``compliance / dt^2``. Defaults to 0.0.
            pgs_omega (float, optional): Successive over-relaxation factor for the PGS sweep. Defaults to 1.0.
            max_constraints (int, optional): Maximum number of articulated contact constraint
                rows stored per world. Free rigid body contacts are stored separately, bounded by
                mf_max_constraints. Must be a multiple of 3 for contact triplet packing.
                Defaults to 33.
            pgs_warmstart (bool, optional): Re-use impulses from the previous frame when contacts persist. Defaults to False.
            mf_max_constraints (int, optional): Maximum number of matrix-free constraints per world. Defaults to 512.
            use_parallel_streams (bool, optional): Dispatch size groups on separate CUDA streams.
                Defaults to True.

        """
        super().__init__(model)

        pgs_mode = "matrix_free"
        friction_mode = "current"
        pgs_kernel = "fused_warp"
        cholesky_kernel = "auto"
        trisolve_kernel = "auto"
        hinv_jt_kernel = "auto"
        delassus_kernel = "auto"
        delassus_chunk_size = None
        pgs_chunk_size = 1
        small_dof_threshold = 12
        pgs_debug = False

        self.angular_damping = angular_damping
        self.update_mass_matrix_interval = update_mass_matrix_interval
        self.friction_smoothing = friction_smoothing
        self.enable_contact_friction = enable_contact_friction
        self.enable_joint_limits = enable_joint_limits
        self.enable_joint_velocity_limits = enable_joint_velocity_limits
        self.pgs_iterations = pgs_iterations
        self.pgs_beta = pgs_beta
        self.pgs_cfm = pgs_cfm
        self.contact_compliance = contact_compliance
        self.dense_contact_compliance = contact_compliance
        self.pgs_omega = pgs_omega
        self.max_constraints = max_constraints
        self.dense_max_constraints = max_constraints
        self.pgs_warmstart = pgs_warmstart
        self.pgs_mode = pgs_mode

        self.friction_mode = friction_mode
        self._friction_mode_id = int(FRICTION_MODE_CURRENT)
        self.mf_max_constraints = mf_max_constraints
        self._double_buffer = double_buffer
        self._nvtx = nvtx
        self.pgs_debug = pgs_debug
        self._pgs_convergence_log: list[np.ndarray] = []
        # Per-step, per-iteration NCP / MDP residual log for the matrix_free
        # debug path. Each entry is an ndarray of shape
        # ``[pgs_iterations, world_count, 6]`` holding
        # ``(r_compl, r_cone, r_gap, r_ds_compl, r_ds_dual, r_mdp_dir)``
        # per world per iteration. Populated only when ``pgs_debug`` is
        # ``True`` and ``pgs_mode == "matrix_free"``.
        self._pgs_ncp_residual_log: list[np.ndarray] = []
        if not model.device.is_cuda:
            raise ValueError("SolverFeatherPGS requires a CUDA device.")
        if not enable_contact_friction:
            raise ValueError("SolverFeatherPGS currently requires enable_contact_friction=True.")
        if max_constraints % 3 != 0:
            raise ValueError("max_constraints must be a multiple of 3 for fused-Warp contact triplet packing.")

        # Effort-limit clamp is always actuator-only: the explicit-PD drive bucket
        # (``aug_row_u0``) is clamped to ``+/- joint_effort_limit`` before it
        # is summed into ``joint_tau``. Matches MuJoCo's ``actuatorfrcrange``
        # and PhysX articulation drive ``maxForce`` conventions.
        self.effort_limit_mode = "actuator"

        self.cholesky_kernel = cholesky_kernel
        self.trisolve_kernel = trisolve_kernel
        self.hinv_jt_kernel = hinv_jt_kernel
        self.delassus_kernel = delassus_kernel
        self.pgs_kernel = pgs_kernel
        self.delassus_chunk_size = delassus_chunk_size
        self.pgs_chunk_size = pgs_chunk_size if pgs_chunk_size is not None else 1
        self.small_dof_threshold = small_dof_threshold
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
        max_constraints = self.dense_max_constraints

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
        # Per-world count of dense contact rows (excluding joint/velocity limits).
        # Used by the fused-Warp matrix-free path to split contact triplets from
        # later one-row constraints.
        self.dense_contact_row_count = wp.zeros((self.world_count,), dtype=wp.int32, device=device)
        self.contact_path = wp.zeros((max_contacts,), dtype=wp.int32, device=device, requires_grad=requires_grad)

        # Joint limit buffers (per-DOF tracking)
        if self.enable_joint_limits and model.joint_dof_count > 0:
            dof_count = model.joint_dof_count
            self.limit_slot = wp.full((dof_count,), -1, dtype=wp.int32, device=device, requires_grad=requires_grad)
            self.limit_sign = wp.zeros((dof_count,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        else:
            self.limit_slot = None
            self.limit_sign = None

        # Joint velocity-limit buffers (per-DOF tracking). Independent from the
        # joint-position-limit buffers so the two flags can be used separately.
        if self.enable_joint_velocity_limits and model.joint_dof_count > 0:
            dof_count = model.joint_dof_count
            self.velocity_limit_slot = wp.full(
                (dof_count,), -1, dtype=wp.int32, device=device, requires_grad=requires_grad
            )
            self.velocity_limit_sign = wp.zeros(
                (dof_count,), dtype=wp.float32, device=device, requires_grad=requires_grad
            )
        else:
            self.velocity_limit_slot = None
            self.velocity_limit_sign = None

    def _allocate_world_buffers(self, model):
        """Allocate world-level constraint system buffers for multi-articulation support."""
        if self.world_count == 0:
            return

        device = model.device
        requires_grad = model.requires_grad
        max_constraints = self.dense_max_constraints

        # Per-world constraint matrices and vectors. Matrix-free never assembles C.
        if self.pgs_mode != "matrix_free":
            self.C = wp.zeros(
                (self.world_count, max_constraints, max_constraints),
                dtype=wp.float32,
                device=device,
                requires_grad=requires_grad,
            )
        else:
            self.C = None

        # Matrix-free uses world-indexed J/Y for both dense and rigid phases.
        if self.pgs_mode == "matrix_free":
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
        else:
            self.J_world = None
            self.Y_world = None

        self.rhs = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        if self.pgs_kernel == "fused_warp":
            # Zach's fused-Warp path stores dense contact triplets as vec3.
            # The constructor validates that max_constraints is divisible by 3,
            # so this preserves the same flat impulse capacity for non-contact
            # one-row constraints while allowing a vec3 view for contact rows.
            max_constraints_padded = ((max_constraints + 2) // 3) * 3
            self.impulses = wp.zeros(
                (self.world_count, max_constraints_padded),
                dtype=wp.float32,
                device=device,
                requires_grad=requires_grad,
            )
            self._max_contact_triplets = max_constraints_padded // 3
            self.impulses_vec3 = self.impulses.reshape((self.world_count, self._max_contact_triplets, 3)).view(wp.vec3)
            self.diag_contact_vec3 = wp.zeros(
                (self.world_count, self._max_contact_triplets), dtype=wp.vec3, device=device
            )
            self.rhs_contact_vec3 = wp.zeros(
                (self.world_count, self._max_contact_triplets), dtype=wp.vec3, device=device
            )
        else:
            self.impulses = wp.zeros(
                (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
            )
            self._max_contact_triplets = 0
            self.impulses_vec3 = None
            self.diag_contact_vec3 = None
            self.rhs_contact_vec3 = None
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
        if self.pgs_mode == "dense" or (not self._has_free_rigid_bodies and self.pgs_mode != "matrix_free"):
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

        # World-relative DOF offsets for two-phase GS kernel
        self.mf_dof_a = wp.zeros((worlds, mf_max_c), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.mf_dof_b = wp.zeros((worlds, mf_max_c), dtype=wp.int32, device=device, requires_grad=requires_grad)

        # Packed MF metadata for two-phase GS kernel (int4 per constraint):
        #   .x = (dof_a << 16) | (dof_b & 0xFFFF)
        #   .y = __float_as_int(eff_mass_inv)
        #   .z = __float_as_int(rhs)
        #   .w = row_type | (row_parent << 16)
        self.mf_meta_packed = wp.zeros((worlds, mf_max_c * 4), dtype=wp.int32, device=device)
        self.mf_meta = (
            wp.zeros((worlds, mf_max_c), dtype=wp.vec4i, device=device) if self.pgs_kernel == "fused_warp" else None
        )

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
        max_c = self.dense_max_constraints
        mf_max_c = self.mf_max_constraints

        self._diag_metrics = wp.zeros((worlds, 4), dtype=wp.float32, device=device)
        # Per-world NCP / MDP residual scratch buffer ([worlds, 6]).
        # Populated by :func:`pgs_ncp_residuals_diagnostic_velocity` on the
        # matrix_free debug path and copied into :attr:`_pgs_ncp_residual_log`.
        self._diag_ncp_metrics = wp.zeros((worlds, 6), dtype=wp.float32, device=device)
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

    def reset_diagnostic_logs(self) -> None:
        """Clear per-step PGS diagnostic logs populated when ``pgs_debug=True``.

        :attr:`_pgs_convergence_log` and :attr:`_pgs_ncp_residual_log` are
        append-only across :meth:`step` calls so one solver instance can
        accumulate a trace across many frames.  A replay / sweep harness
        that reuses a single solver to scan ``pgs_iterations`` over a
        snapshot needs the logs reset between sweeps so each entry
        corresponds to exactly one replayed step.  ``pgs_warmstart``
        impulses and all device-side solver buffers are left untouched.
        """
        self._pgs_convergence_log = []
        self._pgs_ncp_residual_log = []

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
                    use_tiled = (self.cholesky_kernel == "tiled") or (
                        self.cholesky_kernel == "auto" and size > self.small_dof_threshold
                    )
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
                    use_tiled = (self.trisolve_kernel == "tiled") or (
                        self.trisolve_kernel == "auto" and size > self.small_dof_threshold
                    )
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

        if self.pgs_mode == "matrix_free":
            # Compute Y = H^-1 * J^T only (no Delassus C)
            with wp.ScopedTimer("S4_HinvJt_Diag_RHS", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                    with ctx:
                        use_tiled = (self.hinv_jt_kernel == "tiled") or (
                            self.hinv_jt_kernel == "auto" and size > self.small_dof_threshold
                        )
                        if use_tiled:
                            self._stage4_hinv_jt_tiled(size)
                        else:
                            self._stage4_hinv_jt_par_row(size)

                # Diagonal from J*Y (no full Delassus)
                self.diag.zero_()
                for size in self.size_groups:
                    self._stage4_diag_from_JY(size)
                self._stage4_finalize_world_diag_cfm()
                self._stage4_add_dense_contact_compliance(dt)

                # RHS = bias only (J*v recomputed per iteration)
                self._stage4_compute_rhs_world(dt)
                # NOTE: skip _stage4_accumulate_rhs_world — J*v_hat not baked into rhs

                if self.pgs_kernel == "fused_warp":
                    wp.launch(
                        pack_contact_triplets_vec3,
                        dim=(self.world_count, self._max_contact_triplets),
                        inputs=[self.diag],
                        outputs=[self.diag_contact_vec3],
                        device=self.model.device,
                    )
                    wp.launch(
                        pack_contact_triplets_vec3,
                        dim=(self.world_count, self._max_contact_triplets),
                        inputs=[self.rhs],
                        outputs=[self.rhs_contact_vec3],
                        device=self.model.device,
                    )

            # MF: compute mf_MiJt, mf_rhs, mf_eff_mass_inv, body maps
            if self._has_free_rigid_bodies:
                with wp.ScopedTimer("S4_MF_Setup", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                    self._mf_pgs_setup(state_aug, dt)

                    # MF: compute world-relative DOF offsets for two-phase GS kernel
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

                    if self.pgs_kernel == "fused_warp":
                        wp.launch(
                            pack_mf_meta,
                            dim=(self.world_count, self.mf_max_constraints),
                            inputs=[
                                self.mf_dof_a,
                                self.mf_dof_b,
                                self.mf_eff_mass_inv,
                                self.mf_rhs,
                                self.mf_row_type,
                                self.mf_row_parent,
                            ],
                            outputs=[self.mf_meta],
                            device=self.model.device,
                        )

        else:
            fused_ok = (
                self._is_one_art_per_world
                and self.hinv_jt_kernel != "par_row"
                and all(
                    (self.hinv_jt_kernel == "tiled")
                    or (self.hinv_jt_kernel == "auto" and size > self.small_dof_threshold)
                    for size in self.size_groups
                )
            )

            if fused_ok:
                for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                    with ctx:
                        self._stage4_hinv_jt_tiled_fused(size)
            else:
                self._stage4_zero_world_C()

                for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                    with ctx:
                        use_tiled = (self.hinv_jt_kernel == "tiled") or (
                            self.hinv_jt_kernel == "auto" and size > self.small_dof_threshold
                        )
                        if use_tiled:
                            self._stage4_hinv_jt_tiled(size)
                        else:
                            self._stage4_hinv_jt_par_row(size)

                for size in self.size_groups:
                    use_tiled_delassus = self.delassus_kernel != "par_row_col"
                    if use_tiled_delassus:
                        self._stage4_delassus_tiled(size)
                    else:
                        self._stage4_delassus_par_row_col(size)

                self._stage4_finalize_world_diag_cfm()

            self._stage4_add_dense_contact_compliance(dt)
            self._stage4_compute_rhs_world(dt)

            for size in self.size_groups:
                self._stage4_accumulate_rhs_world(size)

        # ══════════════════════════════════════════════════════════════
        # STAGE 5+6: PGS solve
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S5_PGS_Prep", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            self._stage5_prepare_impulses_world()

        if self.pgs_mode == "matrix_free":
            with wp.ScopedTimer("S5_GatherJY", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                # Gather J/Y from per-size-group arrays into world-indexed arrays
                # No J_world/Y_world zeroing needed: gather writes all DOFs unconditionally
                for size in self.size_groups:
                    n_arts = self.n_arts_by_size[size]
                    wp.launch(
                        gather_JY_to_world,
                        dim=int(n_arts * self.dense_max_constraints * size),
                        inputs=[
                            self.group_to_art[size],
                            self.art_to_world,
                            self.articulation_dof_start,
                            self.constraint_count,
                            self.world_dof_start,
                            self.J_by_size[size],
                            self.Y_by_size[size],
                            size,
                            self.dense_max_constraints,
                            n_arts,
                        ],
                        outputs=[self.J_world, self.Y_world],
                        device=self.model.device,
                    )

                # Initialize v_out = v_hat before GS loop
                self._stage6_prepare_world_velocity()

                if self.pgs_kernel != "fused_warp":
                    # Pack MF metadata into int4 structs for coalesced 128-bit loads
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

            # Two-phase GS kernel: split-style dense + MF in one pass
            with wp.ScopedTimer("S6_PGS_Solve", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
                mf_gs_kernel = TiledKernelFactory.get_pgs_solve_mf_gs_kernel(
                    self.dense_max_constraints,
                    self.mf_max_constraints,
                    self.max_world_dofs,
                    self.model.device,
                    friction_mode=self.friction_mode,
                )

                if self.pgs_kernel == "fused_warp":
                    fused_kernel = TiledKernelFactory.get_pgs_fused_warp_kernel(
                        self.dense_max_constraints,
                        self._max_contact_triplets,
                        self.mf_max_constraints,
                        self.max_world_dofs,
                        self.model.device,
                    )
                    wp.launch_tiled(
                        fused_kernel,
                        dim=[self.world_count],
                        inputs=[
                            # Dense
                            self.constraint_count,
                            self.dense_contact_row_count,
                            self.world_dof_start,
                            self.rhs,
                            self.rhs_contact_vec3,
                            self.diag,
                            self.diag_contact_vec3,
                            self.impulses_vec3,
                            self.impulses,
                            self.J_world,
                            self.Y_world,
                            self.row_mu,
                            # MF
                            self.mf_constraint_count,
                            self.mf_meta,
                            self.mf_impulses,
                            self.mf_J_a,
                            self.mf_J_b,
                            self.mf_MiJt_a,
                            self.mf_MiJt_b,
                            self.mf_row_mu,
                            # Shared
                            self.pgs_iterations,
                            self.pgs_omega,
                        ],
                        outputs=[self.v_out],
                        block_dim=32,
                        device=self.model.device,
                    )

                elif self.pgs_debug:
                    self._pgs_convergence_log.append([])
                    self._pgs_ncp_residual_log.append([])
                    for _pgs_dbg_iter in range(self.pgs_iterations):
                        # Snapshot impulses before this iteration
                        wp.copy(self._diag_prev_impulses, self.impulses)
                        if self._diag_prev_mf_impulses is not None:
                            wp.copy(self._diag_prev_mf_impulses, self.mf_impulses)

                        # Run 1 iteration
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
                                1,  # iterations=1
                                self.pgs_omega,
                            ],
                            outputs=[self.v_out],
                            block_dim=32,
                            device=self.model.device,
                        )

                        # Diagnostic kernel
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
                                self.dense_max_constraints,
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

                        # Sync and reduce across worlds
                        metrics_np = self._diag_metrics.numpy()
                        row = np.array(
                            [
                                np.max(metrics_np[:, 0]),  # max|delta_lambda|
                                np.sum(metrics_np[:, 1]),  # complementarity gap
                                np.sum(metrics_np[:, 2]),  # tangent residual
                                np.sum(metrics_np[:, 3]),  # FB merit
                            ]
                        )
                        self._pgs_convergence_log[-1].append(row)

                        # NCP / MDP residual diagnostic ([world, 6]):
                        # (r_compl, r_cone, r_gap, r_ds_compl, r_ds_dual, r_mdp_dir)
                        wp.launch(
                            pgs_ncp_residuals_diagnostic_velocity,
                            dim=self.world_count,
                            inputs=[
                                self.constraint_count,
                                self.world_dof_start,
                                self.rhs,
                                self.impulses,
                                self.row_type,
                                self.row_parent,
                                self.row_mu,
                                self.phi,
                                self.J_world,
                                self.dense_max_constraints,
                                self.max_world_dofs,
                                self.mf_constraint_count,
                                self.mf_rhs,
                                self.mf_impulses,
                                self.mf_row_type,
                                self.mf_row_parent,
                                self.mf_row_mu,
                                self.mf_phi,
                                self.mf_J_a,
                                self.mf_J_b,
                                self.mf_dof_a,
                                self.mf_dof_b,
                                self.mf_max_constraints,
                                self.v_out,
                            ],
                            outputs=[self._diag_ncp_metrics],
                            device=self.model.device,
                        )
                        ncp_np = self._diag_ncp_metrics.numpy().copy()
                        self._pgs_ncp_residual_log[-1].append(ncp_np)

                    self._pgs_convergence_log[-1] = np.array(self._pgs_convergence_log[-1])
                    # Stack per-iter [world, 6] arrays into [iters, world, 6].
                    self._pgs_ncp_residual_log[-1] = np.stack(self._pgs_ncp_residual_log[-1], axis=0)

                else:
                    wp.launch_tiled(
                        mf_gs_kernel,
                        dim=[self.world_count],
                        inputs=[
                            # Dense
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
                            # MF
                            self.mf_constraint_count,
                            self.mf_meta_packed,
                            self.mf_impulses,
                            self.mf_J_a,
                            self.mf_J_b,
                            self.mf_MiJt_a,
                            self.mf_MiJt_b,
                            self.mf_row_mu,
                            # Shared
                            self.pgs_iterations,
                            self.pgs_omega,
                        ],
                        outputs=[self.v_out],
                        block_dim=32,
                        device=self.model.device,
                    )
        elif self.pgs_mode == "split" and self._has_mixed_contacts:
            # Split mode with mixed contacts: interleaved dense and MF, 1 iteration each.
            self._mf_pgs_setup(state_aug, dt)
            self.v_mf_accum.zero_()

            for _pgs_iter in range(self.pgs_iterations):
                # Dense PGS (1 iteration, impulse space)
                self._dispatch_dense_pgs_solve(iterations=1)

                # Rebuild v_out = v_hat + Y*impulses + MF_corrections
                self._stage6_prepare_world_velocity()
                for size in self.size_groups:
                    self._stage6_apply_impulses_world(size)
                wp.launch(
                    vector_add_inplace,
                    dim=self.v_out.size,
                    inputs=[self.v_out, self.v_mf_accum],
                    device=self.model.device,
                )

                # Snapshot v_out, run MF, compute delta
                wp.copy(self.v_out_snap, self.v_out)
                self._mf_pgs_solve(iterations=1)

                # v_mf_accum += (v_out - v_out_snap); v_out_snap = delta
                wp.launch(
                    compute_delta_and_accumulate,
                    dim=self.v_out.size,
                    inputs=[self.v_out, self.v_out_snap, self.v_mf_accum],
                    device=self.model.device,
                )

                # Update dense rhs: world_rhs += J * delta_v_mf
                for size in self.size_groups:
                    n_arts = self.n_arts_by_size[size]
                    wp.launch(
                        rhs_accum_world_par_art,
                        dim=n_arts,
                        inputs=[
                            self.constraint_count,
                            self.dense_max_constraints,
                            self.art_to_world,
                            self.art_size,
                            self.art_group_idx,
                            self.articulation_dof_start,
                            self.v_out_snap,
                            self.group_to_art[size],
                            self.J_by_size[size],
                            size,
                        ],
                        outputs=[self.rhs],
                        device=self.model.device,
                    )

            # v_out is already final (includes both dense and MF corrections)

        else:
            # Dense or split without mixed contacts: dense PGS, then optional MF
            if self.pgs_debug:
                self._pgs_convergence_log.append([])
                for _pgs_dbg_iter in range(self.pgs_iterations):
                    prev_np = self.impulses.numpy().copy()
                    self._dispatch_dense_pgs_solve(iterations=1)
                    cur_np = self.impulses.numpy()
                    max_delta = float(np.max(np.abs(cur_np - prev_np)))
                    self._pgs_convergence_log[-1].append(np.array([max_delta, 0.0, 0.0, 0.0]))
                self._pgs_convergence_log[-1] = np.array(self._pgs_convergence_log[-1])
            else:
                self._dispatch_dense_pgs_solve(iterations=self.pgs_iterations)

            self._stage6_prepare_world_velocity()
            for size in self.size_groups:
                self._stage6_apply_impulses_world(size)

            if self.pgs_mode == "split" and self._has_free_rigid_bodies:
                self._stage6b_mf_pgs(state_aug, dt)

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
        """Populate ``state_aug.joint_tau`` and apply the effort-limit clamp.

        Torque-bucket ownership after this routine returns:

        - ``state_aug.joint_tau`` holds the rigid / passive / Coriolis /
          gravity / external / :attr:`~newton.Control.joint_f` contribution
          (populated by ``eval_rigid_tau``) summed with the explicit-PD
          drive contribution ``u0`` (added by
          :meth:`apply_augmented_joint_tau` from the augmented-row buffer
          ``self.aug_row_u0``).
        - ``self.aug_row_u0`` transiently owns the explicit-PD
          actuator-drive contribution per augmented row before it is
          accumulated into ``joint_tau``.
        - The implicit-PD drive response is carried by ``self.aug_row_K``
          and realized through ``H_tilde^{-1}`` during the linear solve.

        :attr:`~newton.Model.joint_effort_limit` is always applied as
        an **actuator-only** clamp on ``self.aug_row_u0`` *before* it is
        added to ``joint_tau``; the rigid / passive / external bucket is
        left uncapped (MuJoCo ``actuatorfrcrange`` / PhysX drive
        ``maxForce`` convention).

        """
        model = self.model

        if model.articulation_count:
            body_f = state_in.body_f if state_in.body_count else None
            # Evaluate joint torques. After this launch `joint_tau` owns
            # the rigid / passive / Coriolis / gravity / external /
            # `control.joint_f` bucket only; the actuator-drive bucket has
            # not been added yet.
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

            # Populate `aug_row_u0` (and `aug_row_K`) with the explicit-PD
            # actuator-drive bucket per augmented row.
            self.build_augmented_joint_targets(state_in, control, dt)

            self._stage1_drives_apply_augmented_tau(state_aug)

    def _stage1_drives_apply_augmented_tau(self, state_aug: State):
        """Clamp (if needed) and fold the augmented-row ``u0`` into ``joint_tau``.

        Factored out of :meth:`_stage1_drives` to keep the actuator-only
        effort-limit clamp isolated from the rigid / passive / external
        torque bucket.
        """
        model = self.model
        if model.articulation_count == 0:
            return

        if self.articulation_max_dofs > 0:
            # Actuator-only effort-limit clamp: cap the explicit-PD
            # drive bucket (``u0``) to ``+/- joint_effort_limit`` before
            # it is folded into ``joint_tau``. The rigid / passive /
            # external bucket living in ``joint_tau`` is left uncapped;
            # the implicit-PD drive response carried by
            # ``H_tilde^{-1}`` is not clamped here either.
            wp.launch(
                clamp_augmented_joint_u0,
                dim=model.articulation_count,
                inputs=[
                    self.articulation_max_dofs,
                    self.aug_row_counts,
                    self.aug_row_dof_index,
                    model.joint_effort_limit,
                ],
                outputs=[self.aug_row_u0],
                device=model.device,
            )

        # Accumulate (clamped) ``u0`` into ``joint_tau``.
        self.apply_augmented_joint_tau(None, state_aug, 0.0)

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
        max_constraints = self.dense_max_constraints
        mf_active = self._has_free_rigid_bodies and self.pgs_mode != "dense"

        # Zero world-level buffers (only arrays that require it)
        self.slot_counter.zero_()  # atomic-add counter
        self.dense_contact_row_count.zero_()

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

            # Snapshot contact-only dense row count before joint-limit and
            # velocity-limit rows are appended.
            wp.copy(self.dense_contact_row_count, self.slot_counter)

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

            # Allocate + populate joint velocity-limit rows (per-DOF clamp on
            # |qdot_i| against model.joint_velocity_limit). Launched *after*
            # joint-position-limit allocation so velocity-limit rows occupy
            # the last per-world slots — matching PhysX's documented
            # ordering where the vel-limit row fires after contact, drive,
            # friction, and positional-limit rows (physx-deep-dive §7).
            if self.enable_joint_velocity_limits and self.velocity_limit_slot is not None:
                wp.launch(
                    allocate_joint_velocity_limit_slots,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        self.articulation_dof_start,
                        self.articulation_H_rows,
                        model.joint_type,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_velocity_limit,
                        self.v_hat,
                        self.art_to_world,
                        max_constraints,
                    ],
                    outputs=[
                        self.velocity_limit_slot,
                        self.velocity_limit_sign,
                        self.slot_counter,
                    ],
                    device=model.device,
                )
                for size in self.size_groups:
                    n_arts = self.n_arts_by_size[size]
                    wp.launch(
                        populate_joint_velocity_limit_J_for_size,
                        dim=n_arts,
                        inputs=[
                            model.articulation_start,
                            self.articulation_dof_start,
                            model.joint_type,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_velocity_limit,
                            self.art_to_world,
                            self.velocity_limit_slot,
                            self.velocity_limit_sign,
                            self.group_to_art[size],
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

        # Joint velocity-limit fallback path: activates when there are no
        # contacts and the position-limit fallback did not already run us
        # through the velocity-limit dispatch inside the contact block.
        if self.enable_joint_velocity_limits and self.velocity_limit_slot is not None:
            has_contacts = (
                contacts is not None
                and getattr(contacts, "rigid_contact_count", None) is not None
                and contacts.rigid_contact_max > 0
            )
            if not has_contacts:
                # If the position-limit fallback above didn't zero J (because
                # enable_joint_limits is off), we need to zero it here so
                # prior-frame rows don't leak into the current solve.
                if not (self.enable_joint_limits and self.limit_slot is not None):
                    if self._H_bufs is None:
                        for size in self.size_groups:
                            self.J_by_size[size].zero_()
                wp.launch(
                    allocate_joint_velocity_limit_slots,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        self.articulation_dof_start,
                        self.articulation_H_rows,
                        model.joint_type,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_velocity_limit,
                        self.v_hat,
                        self.art_to_world,
                        max_constraints,
                    ],
                    outputs=[
                        self.velocity_limit_slot,
                        self.velocity_limit_sign,
                        self.slot_counter,
                    ],
                    device=model.device,
                )
                for size in self.size_groups:
                    n_arts = self.n_arts_by_size[size]
                    wp.launch(
                        populate_joint_velocity_limit_J_for_size,
                        dim=n_arts,
                        inputs=[
                            model.articulation_start,
                            self.articulation_dof_start,
                            model.joint_type,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            model.joint_velocity_limit,
                            self.art_to_world,
                            self.velocity_limit_slot,
                            self.velocity_limit_sign,
                            self.group_to_art[size],
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
        hinv_jt_kernel = TiledKernelFactory.get_hinv_jt_kernel(size, self.dense_max_constraints, model.device)
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
        hinv_jt_kernel = TiledKernelFactory.get_hinv_jt_fused_kernel(size, self.dense_max_constraints, model.device)
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
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.dense_max_constraints,
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
            dim=n_arts * self.dense_max_constraints * self.dense_max_constraints,
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.dense_max_constraints,
                n_arts,
            ],
            outputs=[self.C, self.diag],
            device=model.device,
        )

    def _stage4_delassus_tiled(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        delassus_kernel = TiledKernelFactory.get_delassus_kernel(
            size, self.dense_max_constraints, model.device, chunk_size=self.delassus_chunk_size
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

    def _stage4_add_dense_contact_compliance(self, dt: float):
        if self.dense_contact_compliance <= 0.0:
            return

        contact_alpha = float(self.dense_contact_compliance / (dt * dt))
        wp.launch(
            add_dense_contact_compliance_to_diag,
            dim=self.world_count,
            inputs=[self.constraint_count, self.row_type, contact_alpha],
            outputs=[self.diag],
            device=self.model.device,
        )

    def _stage4_diag_from_JY(self, size: int):
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            diag_from_JY_par_art,
            dim=n_arts * self.dense_max_constraints,
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.dense_max_constraints,
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
                self.dense_max_constraints,
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
                self.dense_max_constraints,
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
            inputs=[self.constraint_count, self.dense_max_constraints, warmstart_flag],
            outputs=[self.impulses],
            device=self.model.device,
        )

    def _dispatch_dense_pgs_solve(self, iterations: int):
        """Dispatch the dense PGS kernel with a given iteration count."""
        saved = self.pgs_iterations
        self.pgs_iterations = iterations
        if self.pgs_kernel == "tiled_row":
            self._stage5_pgs_solve_world_tiled_row()
        elif self.pgs_kernel == "tiled_contact":
            self._stage5_pgs_solve_world_tiled_contact()
        elif self.pgs_kernel == "streaming":
            self._stage5_pgs_solve_world_streaming()
        else:
            self._stage5_pgs_solve_world_loop()
        self.pgs_iterations = saved

    def _stage5_pgs_solve_world_tiled_row(self):
        pgs_kernel = TiledKernelFactory.get_pgs_solve_tiled_row_kernel(self.dense_max_constraints, self.model.device)
        wp.launch_tiled(
            pgs_kernel,
            dim=[self.world_count],
            inputs=[
                self.constraint_count,
                self.diag,
                self.C,
                self.rhs,
                self.impulses,
                self.pgs_iterations,
                self.pgs_omega,
                self.row_type,
                self.row_parent,
                self.row_mu,
            ],
            block_dim=32,
            device=self.model.device,
        )

    def _stage5_pgs_solve_world_loop(self):
        wp.launch(
            pgs_solve_loop,
            dim=self.world_count,
            inputs=[
                self.constraint_count,
                self.dense_max_constraints,
                self.diag,
                self.C,
                self.rhs,
                self.impulses,
                self.pgs_iterations,
                self.pgs_omega,
                self.row_type,
                self.row_parent,
                self.row_mu,
            ],
            device=self.model.device,
        )

    def _stage5_pgs_solve_world_tiled_contact(self):
        pgs_kernel = TiledKernelFactory.get_pgs_solve_tiled_contact_kernel(
            self.dense_max_constraints, self.model.device
        )
        wp.launch_tiled(
            pgs_kernel,
            dim=[self.world_count],
            inputs=[
                self.constraint_count,
                self.C,
                self.rhs,
                self.impulses,
                self.pgs_iterations,
                self.pgs_omega,
                self.row_mu,
            ],
            block_dim=32,
            device=self.model.device,
        )

    def _stage5_pgs_solve_world_streaming(self):
        pgs_kernel = TiledKernelFactory.get_pgs_solve_streaming_kernel(
            self.dense_max_constraints, self.model.device, pgs_chunk_size=self.pgs_chunk_size
        )
        wp.launch_tiled(
            pgs_kernel,
            dim=[self.world_count],
            inputs=[
                self.constraint_count,
                self.C,
                self.rhs,
                self.impulses,
                self.pgs_iterations,
                self.pgs_omega,
                self.row_mu,
            ],
            block_dim=32,
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
                self.dense_max_constraints,
                self.Y_by_size[size],
                self.impulses,
                self.v_hat,
            ],
            outputs=[self.v_out],
            device=model.device,
        )

    def _stage6b_mf_pgs(self, state_aug: State, dt: float):
        """Run matrix-free PGS for free rigid body contacts."""
        self._mf_pgs_setup(state_aug, dt)
        self._mf_pgs_solve(self.pgs_iterations)

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

    def _mf_pgs_solve(self, iterations: int):
        """MF PGS solve with given iteration count."""
        model = self.model

        # Build compact body map for standalone MF kernel
        wp.launch(
            build_mf_body_map,
            dim=self.world_count,
            inputs=[
                self.mf_constraint_count,
                self.mf_body_a,
                self.mf_body_b,
                self.body_to_articulation,
                self.articulation_dof_start,
                self.max_mf_bodies,
            ],
            outputs=[
                self.mf_body_list,
                self.mf_body_dof_start,
                self.mf_body_count,
                self.mf_local_body_a,
                self.mf_local_body_b,
            ],
            device=model.device,
        )

        if model.device.is_cuda:
            mf_pgs_kernel = TiledKernelFactory.get_pgs_solve_mf_kernel(
                self.mf_max_constraints, self.max_mf_bodies, model.device
            )
            wp.launch_tiled(
                mf_pgs_kernel,
                dim=[self.world_count],
                inputs=[
                    self.mf_constraint_count,
                    self.mf_body_count,
                    self.mf_body_dof_start,
                    self.mf_local_body_a,
                    self.mf_local_body_b,
                    self.mf_J_a,
                    self.mf_J_b,
                    self.mf_MiJt_a,
                    self.mf_MiJt_b,
                    self.mf_eff_mass_inv,
                    self.mf_rhs,
                    self.mf_row_type,
                    self.mf_row_parent,
                    self.mf_row_mu,
                    self.mf_impulses,
                    self.v_out,
                    iterations,
                    self.pgs_omega,
                ],
                block_dim=32,
                device=model.device,
            )
        else:
            # CPU fallback: use the loop kernel
            wp.launch(
                pgs_solve_mf_loop,
                dim=self.world_count,
                inputs=[
                    self.mf_constraint_count,
                    self.mf_body_a,
                    self.mf_body_b,
                    self.mf_MiJt_a,
                    self.mf_MiJt_b,
                    self.mf_J_a,
                    self.mf_J_b,
                    self.mf_eff_mass_inv,
                    self.mf_rhs,
                    self.mf_row_type,
                    self.mf_row_parent,
                    self.mf_row_mu,
                    self.body_to_articulation,
                    self.articulation_dof_start,
                    iterations,
                    self.pgs_omega,
                    self._friction_mode_id,
                ],
                outputs=[
                    self.mf_impulses,
                    self.v_out,
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
    _pgs_solve_tiled_row_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _pgs_solve_tiled_contact_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _pgs_solve_streaming_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _pgs_solve_mf_cache: ClassVar[dict[tuple[int, int, str], "wp.Kernel"]] = {}
    _pgs_solve_mf_gs_cache: ClassVar[dict[tuple[int, int, int, str, str], "wp.Kernel"]] = {}
    _pgs_fused_warp_cache: ClassVar[dict] = {}
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
    def get_pgs_solve_tiled_row_kernel(cls, max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled row-wise PGS world solve kernel for the given constraint count."""
        key = (max_constraints, device.arch)
        if key not in cls._pgs_solve_tiled_row_cache:
            cls._pgs_solve_tiled_row_cache[key] = cls._build_pgs_solve_tiled_row_kernel(max_constraints)
        return cls._pgs_solve_tiled_row_cache[key]

    @classmethod
    def _build_pgs_solve_tiled_row_kernel(cls, max_constraints: int) -> "wp.Kernel":
        """PGS world solve kernel that stages only the LOWER triangle of Delassus.

        Shared memory footprint drops from M*M to M*(M+1)/2 floats.
        Uses symmetry in dot: C(i,j) = L(i,j) if j<=i else L(j,i).
        """
        TILE_M = max_constraints
        TILE_M_SQ = TILE_M * TILE_M
        TILE_TRI = TILE_M * (TILE_M + 1) // 2

        ELEMS_PER_THREAD_1D = (TILE_M + 31) // 32

        def gen_load_1d(dst, src):
            return "\n".join(
                [
                    f"    {dst}[lane + {k * 32}] = {src}.data[off1 + lane + {k * 32}];"
                    for k in range(ELEMS_PER_THREAD_1D)
                    if (k * 32) < TILE_M
                ]
            )

        # Build a deterministic packed-lower-tri index order: row-major over (i, j<=i)
        # idx = i*(i+1)/2 + j
        tri_pairs = []
        for i in range(TILE_M):
            base = i * (i + 1) // 2
            for j in range(i + 1):
                tri_pairs.append((base + j, i, j))
        assert len(tri_pairs) == TILE_TRI

        load_code = "\n".join(
            [
                gen_load_1d("s_lam", "world_impulses"),
                gen_load_1d("s_rhs", "world_rhs"),
                gen_load_1d("s_diag", "world_diag"),
                gen_load_1d("s_rtype", "world_row_type"),
                gen_load_1d("s_parent", "world_row_parent"),
                gen_load_1d("s_mu", "world_row_mu"),
            ]
        )

        # Precompute lane's column indices (j_k) and their triangular bases (j_k*(j_k+1)/2)
        # so inside the dot we avoid multiply.
        precompute_j = []
        for k in range(ELEMS_PER_THREAD_1D):
            j = k * 32
            if j < TILE_M:
                precompute_j.append(
                    f"    const int j{k} = lane + {j};\n    const int jb{k} = (j{k} * (j{k} + 1)) >> 1;"
                )
        precompute_j_code = "\n".join(precompute_j)

        # Dot code: guarded on j_k < m
        dot_terms = []
        for k in range(ELEMS_PER_THREAD_1D):
            joff = k * 32
            if joff < TILE_M:
                dot_terms.append(
                    f"""    if (j{k} < m) {{
            // Use symmetry to fetch C(i, j{k}) from packed-lower shared.
            // base_i = i*(i+1)/2
            float cij = (j{k} <= i) ? s_Ctri[base_i + j{k}] : s_Ctri[jb{k} + i];
            my_sum += cij * s_lam[j{k}];
        }}"""
                )
        dot_code = "\n".join(["float my_sum = 0.0f;", "int base_i = (i * (i + 1)) >> 1;", *dot_terms])

        store_code = "\n".join(
            [
                f"    world_impulses.data[off1 + lane + {k * 32}] = s_lam[lane + {k * 32}];"
                for k in range(ELEMS_PER_THREAD_1D)
                if (k * 32) < TILE_M
            ]
        )

        snippet = f"""
    #if defined(__CUDA_ARCH__)
        const int TILE_M = {TILE_M};
        const int TILE_M_SQ = {TILE_M_SQ};
        const int TILE_TRI = {TILE_TRI};
        const unsigned MASK = 0xFFFFFFFF;

        int lane = threadIdx.x;

        int m = world_constraint_count.data[world];
        if (m == 0) return;

        // Packed LOWER triangle of C in row-major (i*(i+1)/2 + j), j<=i
        __shared__ float s_Ctri[TILE_TRI];

        __shared__ float s_lam[TILE_M];
        __shared__ float s_rhs[TILE_M];
        __shared__ float s_diag[TILE_M];
        __shared__ int   s_rtype[TILE_M];
        __shared__ int   s_parent[TILE_M];
        __shared__ float s_mu[TILE_M];

        int off1 = world * TILE_M;
        int off2 = world * TILE_M_SQ;

    {load_code}

        // Load only lower triangle from global full matrix into packed shared.
        // Work distribution: each lane walks rows; for each row i, lane loads j = lane, lane+32, lane+64...
        for (int i = 0; i < TILE_M; ++i) {{
            int base = (i * (i + 1)) >> 1; // packed base for row i
            for (int j = lane; j <= i; j += 32) {{
                s_Ctri[base + j] = world_C.data[off2 + i * TILE_M + j];
            }}
        }}
        __syncwarp();

    {precompute_j_code}

        for (int iter = 0; iter < iterations; iter++) {{
            for (int i = 0; i < m; i++) {{
                // NOTE: single-warp kernel; __syncwarp here is typically unnecessary unless divergence occurs
                // before the dot. If you want max perf, try removing it after verifying correctness.
                // __syncwarp();

                {dot_code}

                // Warp reduce my_sum
                my_sum += __shfl_down_sync(MASK, my_sum, 16);
                my_sum += __shfl_down_sync(MASK, my_sum, 8);
                my_sum += __shfl_down_sync(MASK, my_sum, 4);
                my_sum += __shfl_down_sync(MASK, my_sum, 2);
                my_sum += __shfl_down_sync(MASK, my_sum, 1);
                float dot_sum = __shfl_sync(MASK, my_sum, 0);

                float denom = s_diag[i];
                if (denom <= 0.0f) continue;

                float w_val = s_rhs[i] + dot_sum;
                float delta = -w_val / denom;
                float new_impulse = s_lam[i] + omega * delta;
                int row_type = s_rtype[i];

                // row_type 0=CONTACT, 3=JOINT_LIMIT, 4=JOINT_VELOCITY_LIMIT:
                // unilateral lambda >= 0 projector. The velocity-limit row
                // uses a signed Jacobian so one side of the bilateral
                // [-qdot_max, +qdot_max] box is active at a time.
                if (row_type == 0 || row_type == 3 || row_type == 4) {{
                    if (new_impulse < 0.0f) new_impulse = 0.0f;
                    s_lam[i] = new_impulse;
                }} else if (row_type == 2) {{
                    int parent_idx = s_parent[i];
                    float lambda_n = s_lam[parent_idx];
                    float mu = s_mu[i];
                    float radius = fmaxf(mu * lambda_n, 0.0f);

                    if (radius <= 0.0f) {{
                        s_lam[i] = 0.0f;
                    }} else {{
                        s_lam[i] = new_impulse;
                        int sib = (i == parent_idx + 1) ? (parent_idx + 2) : (parent_idx + 1);
                        float a = s_lam[i];
                        float b = s_lam[sib];
                        float mag = sqrtf(a * a + b * b);
                        if (mag > radius) {{
                            float scale = radius / mag;
                            s_lam[i] = a * scale;
                            s_lam[sib] = b * scale;
                        }}
                    }}
                }} else {{
                    s_lam[i] = new_impulse;
                }}
            }}
        }}

    {store_code}
    #endif
    """

        @wp.func_native(snippet)
        def pgs_solve_native(
            world: int,
            world_constraint_count: wp.array[int],
            world_diag: wp.array2d[float],
            world_C: wp.array3d[float],
            world_rhs: wp.array2d[float],
            world_impulses: wp.array2d[float],
            iterations: int,
            omega: float,
            world_row_type: wp.array2d[int],
            world_row_parent: wp.array2d[int],
            world_row_mu: wp.array2d[float],
        ): ...

        def pgs_solve_tiled_template(
            world_constraint_count: wp.array[int],
            world_diag: wp.array2d[float],
            world_C: wp.array3d[float],
            world_rhs: wp.array2d[float],
            world_impulses: wp.array2d[float],
            iterations: int,
            omega: float,
            world_row_type: wp.array2d[int],
            world_row_parent: wp.array2d[int],
            world_row_mu: wp.array2d[float],
        ):
            world, _lane = wp.tid()
            pgs_solve_native(
                world,
                world_constraint_count,
                world_diag,
                world_C,
                world_rhs,
                world_impulses,
                iterations,
                omega,
                world_row_type,
                world_row_parent,
                world_row_mu,
            )

        pgs_solve_tiled_template.__name__ = f"pgs_solve_tiled_row_{max_constraints}"
        pgs_solve_tiled_template.__qualname__ = f"pgs_solve_tiled_row_{max_constraints}"
        return wp.kernel(enable_backward=False, module="unique")(pgs_solve_tiled_template)

    @classmethod
    def get_pgs_solve_tiled_contact_kernel(cls, max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled contact-wise PGS world solve kernel using 3x3 block formulation."""
        key = (max_constraints, device.arch)
        if key not in cls._pgs_solve_tiled_contact_cache:
            cls._pgs_solve_tiled_contact_cache[key] = cls._build_pgs_solve_tiled_contact_kernel(max_constraints)
        return cls._pgs_solve_tiled_contact_cache[key]

    @classmethod
    def _build_pgs_solve_tiled_contact_kernel(cls, max_constraints: int) -> "wp.Kernel":
        """PGS world solve kernel using 3x3 block formulation.

        Stores only the LOWER triangle of block Delassus matrix.
        Each contact is a 3-vector (normal, tangent1, tangent2).
        Reduces serial depth from M to M/3.

        TILE_M can be any value (power of 2 recommended for other kernels).
        Runtime m must be divisible by 3.
        """
        TILE_M = max_constraints
        # Max contacts we can handle (rounded down)
        NUM_CONTACTS_MAX = TILE_M // 3
        # Actual max constraints we'll process (may be < TILE_M)
        TILE_M_USABLE = NUM_CONTACTS_MAX * 3

        # Lower triangle of block matrix (sized for max)
        NUM_BLOCKS_TRI = NUM_CONTACTS_MAX * (NUM_CONTACTS_MAX + 1) // 2
        BLOCK_TRI_FLOATS = NUM_BLOCKS_TRI * 9

        snippet = f"""
    #if defined(__CUDA_ARCH__)
        const int TILE_M = {TILE_M};
        const int TILE_M_USABLE = {TILE_M_USABLE};
        const int NUM_CONTACTS_MAX = {NUM_CONTACTS_MAX};
        const int BLOCK_TRI_FLOATS = {BLOCK_TRI_FLOATS};
        const unsigned MASK = 0xFFFFFFFF;

        int lane = threadIdx.x;

        int m = world_constraint_count.data[world];
        if (m == 0) return;

        // Clamp m to usable range and ensure divisible by 3
        if (m > TILE_M_USABLE) m = TILE_M_USABLE;
        int num_contacts = m / 3;

        // Shared memory (sized for max)
        __shared__ float s_Dtri[BLOCK_TRI_FLOATS];
        __shared__ float s_Dinv[NUM_CONTACTS_MAX * 9];
        __shared__ float s_lam[TILE_M_USABLE];
        __shared__ float s_rhs[TILE_M_USABLE];
        __shared__ float s_mu[NUM_CONTACTS_MAX];

        int off1 = world * TILE_M;
        int off2 = world * TILE_M * TILE_M;

        // ============ LOAD PHASE ============

        // Load lambda and rhs
        for (int i = lane; i < TILE_M_USABLE; i += 32) {{
            if (i < m) {{
                s_lam[i] = world_impulses.data[off1 + i];
                s_rhs[i] = world_rhs.data[off1 + i];
            }} else {{
                s_lam[i] = 0.0f;
                s_rhs[i] = 0.0f;
            }}
        }}

        // Load mu (one per contact, stored on tangent1 row)
        for (int c = lane; c < NUM_CONTACTS_MAX; c += 32) {{
            if (c < num_contacts) {{
                s_mu[c] = world_row_mu.data[off1 + c * 3 + 1];
            }}
        }}

        // Load lower triangle of block Delassus
        for (int c = 0; c < num_contacts; c++) {{
            int base_block = (c * (c + 1)) >> 1;
            int floats_in_row = (c + 1) * 9;

            for (int f = lane; f < floats_in_row; f += 32) {{
                int j = f / 9;
                int k = f % 9;
                int lr = k / 3;
                int lc = k % 3;
                int gr = c * 3 + lr;
                int gc = j * 3 + lc;
                s_Dtri[(base_block + j) * 9 + k] = world_C.data[off2 + gr * TILE_M + gc];
            }}
        }}
        __syncwarp();

        // Compute diagonal block inverses
        for (int c = lane; c < num_contacts; c += 32) {{
            int diag_block_idx = ((c * (c + 1)) >> 1) + c;
            const float* D = &s_Dtri[diag_block_idx * 9];
            float* Dinv = &s_Dinv[c * 9];

            float det = D[0] * (D[4] * D[8] - D[5] * D[7])
                    - D[1] * (D[3] * D[8] - D[5] * D[6])
                    + D[2] * (D[3] * D[7] - D[4] * D[6]);

            float inv_det = 1.0f / det;

            Dinv[0] = (D[4] * D[8] - D[5] * D[7]) * inv_det;
            Dinv[1] = (D[2] * D[7] - D[1] * D[8]) * inv_det;
            Dinv[2] = (D[1] * D[5] - D[2] * D[4]) * inv_det;
            Dinv[3] = (D[5] * D[6] - D[3] * D[8]) * inv_det;
            Dinv[4] = (D[0] * D[8] - D[2] * D[6]) * inv_det;
            Dinv[5] = (D[2] * D[3] - D[0] * D[5]) * inv_det;
            Dinv[6] = (D[3] * D[7] - D[4] * D[6]) * inv_det;
            Dinv[7] = (D[1] * D[6] - D[0] * D[7]) * inv_det;
            Dinv[8] = (D[0] * D[4] - D[1] * D[3]) * inv_det;
        }}
        __syncwarp();

        // ============ ITERATION PHASE ============

        for (int iter = 0; iter < iterations; iter++) {{
            for (int c = 0; c < num_contacts; c++) {{
                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;

                for (int j = lane; j < num_contacts; j += 32) {{
                    float l0 = s_lam[j * 3 + 0];
                    float l1 = s_lam[j * 3 + 1];
                    float l2 = s_lam[j * 3 + 2];

                    int block_off;
                    bool transpose;
                    if (j <= c) {{
                        block_off = (((c * (c + 1)) >> 1) + j) * 9;
                        transpose = false;
                    }} else {{
                        block_off = (((j * (j + 1)) >> 1) + c) * 9;
                        transpose = true;
                    }}

                    const float* B = &s_Dtri[block_off];

                    if (!transpose) {{
                        sum0 += B[0] * l0 + B[1] * l1 + B[2] * l2;
                        sum1 += B[3] * l0 + B[4] * l1 + B[5] * l2;
                        sum2 += B[6] * l0 + B[7] * l1 + B[8] * l2;
                    }} else {{
                        sum0 += B[0] * l0 + B[3] * l1 + B[6] * l2;
                        sum1 += B[1] * l0 + B[4] * l1 + B[7] * l2;
                        sum2 += B[2] * l0 + B[5] * l1 + B[8] * l2;
                    }}
                }}

                // Warp reduce
                sum0 += __shfl_down_sync(MASK, sum0, 16);
                sum1 += __shfl_down_sync(MASK, sum1, 16);
                sum2 += __shfl_down_sync(MASK, sum2, 16);
                sum0 += __shfl_down_sync(MASK, sum0, 8);
                sum1 += __shfl_down_sync(MASK, sum1, 8);
                sum2 += __shfl_down_sync(MASK, sum2, 8);
                sum0 += __shfl_down_sync(MASK, sum0, 4);
                sum1 += __shfl_down_sync(MASK, sum1, 4);
                sum2 += __shfl_down_sync(MASK, sum2, 4);
                sum0 += __shfl_down_sync(MASK, sum0, 2);
                sum1 += __shfl_down_sync(MASK, sum1, 2);
                sum2 += __shfl_down_sync(MASK, sum2, 2);
                sum0 += __shfl_down_sync(MASK, sum0, 1);
                sum1 += __shfl_down_sync(MASK, sum1, 1);
                sum2 += __shfl_down_sync(MASK, sum2, 1);

                if (lane == 0) {{
                    // Corrected sign: -(rhs + D*lambda)
                    float res0 = -(s_rhs[c * 3 + 0] + sum0);
                    float res1 = -(s_rhs[c * 3 + 1] + sum1);
                    float res2 = -(s_rhs[c * 3 + 2] + sum2);

                    const float* Dinv = &s_Dinv[c * 9];
                    float d0 = Dinv[0] * res0 + Dinv[1] * res1 + Dinv[2] * res2;
                    float d1 = Dinv[3] * res0 + Dinv[4] * res1 + Dinv[5] * res2;
                    float d2 = Dinv[6] * res0 + Dinv[7] * res1 + Dinv[8] * res2;

                    float new_n  = s_lam[c * 3 + 0] + omega * d0;
                    float new_t1 = s_lam[c * 3 + 1] + omega * d1;
                    float new_t2 = s_lam[c * 3 + 2] + omega * d2;

                    // Friction cone projection
                    new_n = fmaxf(new_n, 0.0f);

                    float mu = s_mu[c];
                    float radius = mu * new_n;

                    if (radius <= 0.0f) {{
                        new_t1 = 0.0f;
                        new_t2 = 0.0f;
                    }} else {{
                        float t_mag_sq = new_t1 * new_t1 + new_t2 * new_t2;
                        if (t_mag_sq > radius * radius) {{
                            float scale = radius * rsqrtf(t_mag_sq);
                            new_t1 *= scale;
                            new_t2 *= scale;
                        }}
                    }}

                    s_lam[c * 3 + 0] = new_n;
                    s_lam[c * 3 + 1] = new_t1;
                    s_lam[c * 3 + 2] = new_t2;
                }}
                __syncwarp();
            }}
        }}

        // ============ STORE PHASE ============

        for (int i = lane; i < TILE_M_USABLE; i += 32) {{
            if (i < m) {{
                world_impulses.data[off1 + i] = s_lam[i];
            }}
        }}
    #endif
    """

        @wp.func_native(snippet)
        def pgs_solve_contact_native(
            world: int,
            world_constraint_count: wp.array[int],
            world_C: wp.array3d[float],
            world_rhs: wp.array2d[float],
            world_impulses: wp.array2d[float],
            iterations: int,
            omega: float,
            world_row_mu: wp.array2d[float],
        ): ...

        def pgs_solve_tiled_contact_template(
            world_constraint_count: wp.array[int],
            world_C: wp.array3d[float],
            world_rhs: wp.array2d[float],
            world_impulses: wp.array2d[float],
            iterations: int,
            omega: float,
            world_row_mu: wp.array2d[float],
        ):
            world, _lane = wp.tid()
            pgs_solve_contact_native(
                world,
                world_constraint_count,
                world_C,
                world_rhs,
                world_impulses,
                iterations,
                omega,
                world_row_mu,
            )

        pgs_solve_tiled_contact_template.__name__ = f"pgs_solve_tiled_contact_{max_constraints}"
        pgs_solve_tiled_contact_template.__qualname__ = f"pgs_solve_tiled_contact_{max_constraints}"
        return wp.kernel(enable_backward=False, module="unique")(pgs_solve_tiled_contact_template)

    @classmethod
    def get_pgs_solve_streaming_kernel(
        cls, max_constraints: int, device: "wp.Device", pgs_chunk_size: int = 1
    ) -> "wp.Kernel":
        """Get or create a streaming contact-wise PGS world solve kernel."""
        key = (max_constraints, device.arch, pgs_chunk_size)
        if key not in cls._pgs_solve_streaming_cache:
            cls._pgs_solve_streaming_cache[key] = cls._build_pgs_solve_streaming_kernel(max_constraints, pgs_chunk_size)
        return cls._pgs_solve_streaming_cache[key]

    @classmethod
    def _build_pgs_solve_streaming_kernel(cls, max_constraints: int, pgs_chunk_size: int = 1) -> "wp.Kernel":
        """Streaming contact-wise PGS kernel that streams block-rows from global memory.

        Unlike tiled_contact which loads the entire Delassus matrix into shared memory,
        this kernel keeps only lambda and auxiliaries in shared memory and streams
        block-rows of C on demand. This enables handling much larger constraint counts
        (hundreds of contacts) at the cost of increased global memory bandwidth.

        When pgs_chunk_size > 1, multiple block-rows are preloaded into shared memory
        at once, reducing the number of global memory round-trips per PGS iteration.

        Algorithm:
        - Load lambda, rhs, mu, and compute diagonal block inverses once
        - For each PGS iteration:
            - For each chunk of pgs_chunk_size contacts:
                - Preload pgs_chunk_size block-rows of C into shared memory
                - For each contact c in the chunk:
                    - Compute block-row dot product with lambda (warp-parallel)
                    - Update lambda[c] with friction cone projection (lane 0)
        - Store final lambda back to global memory
        """
        TILE_M = max_constraints
        NUM_CONTACTS_MAX = TILE_M // 3
        TILE_M_USABLE = NUM_CONTACTS_MAX * 3
        PGS_CHUNK = pgs_chunk_size

        snippet = f"""
    #if defined(__CUDA_ARCH__)
        const int TILE_M = {TILE_M};
        const int TILE_M_USABLE = {TILE_M_USABLE};
        const int NUM_CONTACTS_MAX = {NUM_CONTACTS_MAX};
        const int PGS_CHUNK = {PGS_CHUNK};
        const unsigned MASK = 0xFFFFFFFF;

        int lane = threadIdx.x;

        int m = world_constraint_count.data[world];
        if (m == 0) return;

        // Clamp m to usable range and ensure divisible by 3
        if (m > TILE_M_USABLE) m = TILE_M_USABLE;
        int num_contacts = m / 3;

        // ═══════════════════════════════════════════════════════════════
        // SHARED MEMORY: lambda, rhs, mu, diagonal inverses, and
        // block-row buffer for PGS_CHUNK contacts at a time
        // ═══════════════════════════════════════════════════════════════
        __shared__ float s_lam[{TILE_M_USABLE}];
        __shared__ float s_rhs[{TILE_M_USABLE}];
        __shared__ float s_mu[{NUM_CONTACTS_MAX}];
        __shared__ float s_Dinv[{NUM_CONTACTS_MAX} * 9];
        __shared__ float s_block_rows[{PGS_CHUNK} * {NUM_CONTACTS_MAX} * 9];

        int off1 = world * TILE_M;
        int off2 = world * TILE_M * TILE_M;

        // ═══════════════════════════════════════════════════════════════
        // LOAD PHASE: Load persistent data into shared memory
        // ═══════════════════════════════════════════════════════════════

        // Load lambda and rhs (coalesced)
        for (int i = lane; i < TILE_M_USABLE; i += 32) {{
            if (i < m) {{
                s_lam[i] = world_impulses.data[off1 + i];
                s_rhs[i] = world_rhs.data[off1 + i];
            }} else {{
                s_lam[i] = 0.0f;
                s_rhs[i] = 0.0f;
            }}
        }}

        // Load mu (one per contact, stored on tangent1 row)
        for (int c = lane; c < NUM_CONTACTS_MAX; c += 32) {{
            if (c < num_contacts) {{
                s_mu[c] = world_row_mu.data[off1 + c * 3 + 1];
            }}
        }}
        __syncwarp();

        // Compute diagonal block inverses (each thread handles one contact)
        for (int c = lane; c < num_contacts; c += 32) {{
            // Load diagonal block D[c,c] from global memory
            int diag_row = c * 3;
            float D[9];
            for (int k = 0; k < 9; k++) {{
                int lr = k / 3;
                int lc = k % 3;
                D[k] = world_C.data[off2 + (diag_row + lr) * TILE_M + (diag_row + lc)];
            }}

            // Compute 3x3 inverse
            float det = D[0] * (D[4] * D[8] - D[5] * D[7])
                      - D[1] * (D[3] * D[8] - D[5] * D[6])
                      + D[2] * (D[3] * D[7] - D[4] * D[6]);

            float inv_det = 1.0f / det;
            float* Dinv = &s_Dinv[c * 9];

            Dinv[0] = (D[4] * D[8] - D[5] * D[7]) * inv_det;
            Dinv[1] = (D[2] * D[7] - D[1] * D[8]) * inv_det;
            Dinv[2] = (D[1] * D[5] - D[2] * D[4]) * inv_det;
            Dinv[3] = (D[5] * D[6] - D[3] * D[8]) * inv_det;
            Dinv[4] = (D[0] * D[8] - D[2] * D[6]) * inv_det;
            Dinv[5] = (D[2] * D[3] - D[0] * D[5]) * inv_det;
            Dinv[6] = (D[3] * D[7] - D[4] * D[6]) * inv_det;
            Dinv[7] = (D[1] * D[6] - D[0] * D[7]) * inv_det;
            Dinv[8] = (D[0] * D[4] - D[1] * D[3]) * inv_det;
        }}
        __syncwarp();

        // ═══════════════════════════════════════════════════════════════
        // ITERATION PHASE: Stream block-rows in chunks and solve
        // ═══════════════════════════════════════════════════════════════

        for (int iter = 0; iter < iterations; iter++) {{
            for (int chunk_start = 0; chunk_start < num_contacts; chunk_start += PGS_CHUNK) {{
                int chunk_end = min(chunk_start + PGS_CHUNK, num_contacts);
                int chunk_len = chunk_end - chunk_start;

                // ─────────────────────────────────────────────────────────
                // STREAM: Preload chunk_len block-rows of Delassus matrix
                // ─────────────────────────────────────────────────────────
                for (int ci = 0; ci < chunk_len; ci++) {{
                    int c = chunk_start + ci;
                    int c_row = c * 3;
                    float* row_base = &s_block_rows[ci * NUM_CONTACTS_MAX * 9];
                    for (int j = lane; j < num_contacts; j += 32) {{
                        int j_col = j * 3;
                        float* dst = &row_base[j * 9];
                        for (int k = 0; k < 9; k++) {{
                            int lr = k / 3;
                            int lc = k % 3;
                            dst[k] = world_C.data[off2 + (c_row + lr) * TILE_M + (j_col + lc)];
                        }}
                    }}
                }}
                __syncwarp();

                // ─────────────────────────────────────────────────────────
                // SOLVE: Process each contact in the chunk sequentially
                // ─────────────────────────────────────────────────────────
                for (int ci = 0; ci < chunk_len; ci++) {{
                    int c = chunk_start + ci;
                    const float* row_base = &s_block_rows[ci * NUM_CONTACTS_MAX * 9];

                    // Block-row dot product sum_j C[c,j] * lambda[j]
                    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;

                    for (int j = lane; j < num_contacts; j += 32) {{
                        float l0 = s_lam[j * 3 + 0];
                        float l1 = s_lam[j * 3 + 1];
                        float l2 = s_lam[j * 3 + 2];

                        const float* B = &row_base[j * 9];

                        sum0 += B[0] * l0 + B[1] * l1 + B[2] * l2;
                        sum1 += B[3] * l0 + B[4] * l1 + B[5] * l2;
                        sum2 += B[6] * l0 + B[7] * l1 + B[8] * l2;
                    }}

                    // Warp reduce
                    sum0 += __shfl_down_sync(MASK, sum0, 16);
                    sum1 += __shfl_down_sync(MASK, sum1, 16);
                    sum2 += __shfl_down_sync(MASK, sum2, 16);
                    sum0 += __shfl_down_sync(MASK, sum0, 8);
                    sum1 += __shfl_down_sync(MASK, sum1, 8);
                    sum2 += __shfl_down_sync(MASK, sum2, 8);
                    sum0 += __shfl_down_sync(MASK, sum0, 4);
                    sum1 += __shfl_down_sync(MASK, sum1, 4);
                    sum2 += __shfl_down_sync(MASK, sum2, 4);
                    sum0 += __shfl_down_sync(MASK, sum0, 2);
                    sum1 += __shfl_down_sync(MASK, sum1, 2);
                    sum2 += __shfl_down_sync(MASK, sum2, 2);
                    sum0 += __shfl_down_sync(MASK, sum0, 1);
                    sum1 += __shfl_down_sync(MASK, sum1, 1);
                    sum2 += __shfl_down_sync(MASK, sum2, 1);

                    // Update: Solve and project (lane 0 only)
                    if (lane == 0) {{
                        float res0 = -(s_rhs[c * 3 + 0] + sum0);
                        float res1 = -(s_rhs[c * 3 + 1] + sum1);
                        float res2 = -(s_rhs[c * 3 + 2] + sum2);

                        const float* Dinv = &s_Dinv[c * 9];
                        float d0 = Dinv[0] * res0 + Dinv[1] * res1 + Dinv[2] * res2;
                        float d1 = Dinv[3] * res0 + Dinv[4] * res1 + Dinv[5] * res2;
                        float d2 = Dinv[6] * res0 + Dinv[7] * res1 + Dinv[8] * res2;

                        float new_n  = s_lam[c * 3 + 0] + omega * d0;
                        float new_t1 = s_lam[c * 3 + 1] + omega * d1;
                        float new_t2 = s_lam[c * 3 + 2] + omega * d2;

                        // Friction cone projection
                        new_n = fmaxf(new_n, 0.0f);

                        float mu = s_mu[c];
                        float radius = mu * new_n;

                        if (radius <= 0.0f) {{
                            new_t1 = 0.0f;
                            new_t2 = 0.0f;
                        }} else {{
                            float t_mag_sq = new_t1 * new_t1 + new_t2 * new_t2;
                            if (t_mag_sq > radius * radius) {{
                                float scale = radius * rsqrtf(t_mag_sq);
                                new_t1 *= scale;
                                new_t2 *= scale;
                            }}
                        }}

                        s_lam[c * 3 + 0] = new_n;
                        s_lam[c * 3 + 1] = new_t1;
                        s_lam[c * 3 + 2] = new_t2;
                    }}
                    __syncwarp();
                }}
            }}
        }}

        // ═══════════════════════════════════════════════════════════════
        // STORE PHASE: Write final lambda back to global memory
        // ═══════════════════════════════════════════════════════════════
        for (int i = lane; i < TILE_M_USABLE; i += 32) {{
            if (i < m) {{
                world_impulses.data[off1 + i] = s_lam[i];
            }}
        }}
    #endif
    """

        @wp.func_native(snippet)
        def pgs_solve_streaming_native(
            world: int,
            world_constraint_count: wp.array[int],
            world_C: wp.array3d[float],
            world_rhs: wp.array2d[float],
            world_impulses: wp.array2d[float],
            iterations: int,
            omega: float,
            world_row_mu: wp.array2d[float],
        ): ...

        def pgs_solve_streaming_template(
            world_constraint_count: wp.array[int],
            world_C: wp.array3d[float],
            world_rhs: wp.array2d[float],
            world_impulses: wp.array2d[float],
            iterations: int,
            omega: float,
            world_row_mu: wp.array2d[float],
        ):
            world, _lane = wp.tid()
            pgs_solve_streaming_native(
                world,
                world_constraint_count,
                world_C,
                world_rhs,
                world_impulses,
                iterations,
                omega,
                world_row_mu,
            )

        pgs_solve_streaming_template.__name__ = f"pgs_solve_streaming_{max_constraints}_chunk{pgs_chunk_size}"
        pgs_solve_streaming_template.__qualname__ = f"pgs_solve_streaming_{max_constraints}_chunk{pgs_chunk_size}"
        return wp.kernel(enable_backward=False, module="unique")(pgs_solve_streaming_template)

    @classmethod
    def get_pgs_solve_mf_kernel(cls, mf_max_constraints: int, max_mf_bodies: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a streaming MF PGS kernel for free rigid body contacts."""
        key = (mf_max_constraints, max_mf_bodies, device.arch)
        if key not in cls._pgs_solve_mf_cache:
            cls._pgs_solve_mf_cache[key] = cls._build_pgs_solve_mf_kernel(mf_max_constraints, max_mf_bodies)
        return cls._pgs_solve_mf_cache[key]

    @classmethod
    def _build_pgs_solve_mf_kernel(cls, mf_max_constraints: int, max_mf_bodies: int) -> "wp.Kernel":
        """Matrix-free PGS with body velocities and impulses in shared memory.

        Uses one warp (32 threads) per world. Body velocities and impulses live
        in shared memory for the duration of all PGS iterations, eliminating
        global memory round-trips. J, MiJt, eff_mass_inv, and rhs are read
        from global memory per constraint (read-only, cache-friendly sequential access).
        """
        MF_MAX_C = mf_max_constraints
        MAX_BODIES = max_mf_bodies

        snippet = f"""
    #if defined(__CUDA_ARCH__)
        const int MF_MAX_C = {MF_MAX_C};
        const int MAX_BODIES = {MAX_BODIES};

        int lane = threadIdx.x;

        int m = mf_constraint_count.data[world];
        if (m == 0) return;
        if (m > MF_MAX_C) m = MF_MAX_C;

        int n_bodies = mf_body_count.data[world];
        if (n_bodies > MAX_BODIES) n_bodies = MAX_BODIES;

        // ═══════════════════════════════════════════════════════════════
        // SHARED MEMORY
        // ═══════════════════════════════════════════════════════════════
        __shared__ float s_vel[{MAX_BODIES * 6}];
        __shared__ float s_impulse[{MF_MAX_C}];
        __shared__ int s_dof_start[{MAX_BODIES}];

        int body_off = world * MAX_BODIES;
        int c_off = world * MF_MAX_C;

        // ═══════════════════════════════════════════════════════════════
        // LOAD PHASE
        // ═══════════════════════════════════════════════════════════════

        // Load body DOF starts and velocities
        for (int b = lane; b < n_bodies; b += 32) {{
            int dof = mf_body_dof_start.data[body_off + b];
            s_dof_start[b] = dof;
            for (int k = 0; k < 6; k++) {{
                s_vel[b * 6 + k] = v_out.data[dof + k];
            }}
        }}

        // Load impulses
        for (int i = lane; i < m; i += 32) {{
            s_impulse[i] = mf_impulses.data[c_off + i];
        }}
        __syncwarp();

        // ═══════════════════════════════════════════════════════════════
        // SOLVE PHASE (lane 0)
        // ═══════════════════════════════════════════════════════════════

        if (lane == 0) {{
            for (int iter = 0; iter < iterations; iter++) {{
                for (int i = 0; i < m; i++) {{
                    float eff_inv = mf_eff_mass_inv.data[c_off + i];
                    if (eff_inv <= 0.0f) continue;

                    int lba = mf_local_body_a.data[c_off + i];
                    int lbb = mf_local_body_b.data[c_off + i];

                    // Load J from global memory
                    int j_base = (c_off + i) * 6;
                    float ja0 = mf_J_a.data[j_base + 0];
                    float ja1 = mf_J_a.data[j_base + 1];
                    float ja2 = mf_J_a.data[j_base + 2];
                    float ja3 = mf_J_a.data[j_base + 3];
                    float ja4 = mf_J_a.data[j_base + 4];
                    float ja5 = mf_J_a.data[j_base + 5];

                    float jb0 = mf_J_b.data[j_base + 0];
                    float jb1 = mf_J_b.data[j_base + 1];
                    float jb2 = mf_J_b.data[j_base + 2];
                    float jb3 = mf_J_b.data[j_base + 3];
                    float jb4 = mf_J_b.data[j_base + 4];
                    float jb5 = mf_J_b.data[j_base + 5];

                    // Compute J * v from shared memory
                    float jv = 0.0f;
                    if (lba >= 0) {{
                        int va = lba * 6;
                        jv += ja0 * s_vel[va] + ja1 * s_vel[va+1] + ja2 * s_vel[va+2]
                            + ja3 * s_vel[va+3] + ja4 * s_vel[va+4] + ja5 * s_vel[va+5];
                    }}
                    if (lbb >= 0) {{
                        int vb = lbb * 6;
                        jv += jb0 * s_vel[vb] + jb1 * s_vel[vb+1] + jb2 * s_vel[vb+2]
                            + jb3 * s_vel[vb+3] + jb4 * s_vel[vb+4] + jb5 * s_vel[vb+5];
                    }}

                    // PGS update
                    float rhs_i = mf_rhs.data[c_off + i];
                    float delta = -(jv + rhs_i) * eff_inv;
                    float old_impulse = s_impulse[i];
                    float new_impulse = old_impulse + omega * delta;

                    int row_type = mf_row_type.data[c_off + i];

                    // Project: contact or joint limit
                    if (row_type == 0 || row_type == 3) {{
                        if (new_impulse < 0.0f) new_impulse = 0.0f;
                    }}
                    // Project: friction
                    else if (row_type == 2) {{
                        int parent_idx = mf_row_parent.data[c_off + i];
                        float lambda_n = s_impulse[parent_idx];
                        float mu = mf_row_mu.data[c_off + i];
                        float radius = fmaxf(mu * lambda_n, 0.0f);

                        if (radius <= 0.0f) {{
                            new_impulse = 0.0f;
                        }} else {{
                            int sib = (i == parent_idx + 1) ? parent_idx + 2 : parent_idx + 1;

                            s_impulse[i] = new_impulse;
                            float a_val = new_impulse;
                            float b_val = s_impulse[sib];
                            float mag = sqrtf(a_val * a_val + b_val * b_val);
                            if (mag > radius) {{
                                float scale = radius / mag;
                                new_impulse = a_val * scale;
                                float sib_new = b_val * scale;
                                float sib_delta = sib_new - b_val;
                                s_impulse[sib] = sib_new;

                                // Apply sibling correction to body velocities
                                int sib_lba = mf_local_body_a.data[c_off + sib];
                                int sib_lbb = mf_local_body_b.data[c_off + sib];
                                int sib_j_base = (c_off + sib) * 6;
                                if (sib_lba >= 0) {{
                                    int sva = sib_lba * 6;
                                    s_vel[sva+0] += mf_MiJt_a.data[sib_j_base+0] * sib_delta;
                                    s_vel[sva+1] += mf_MiJt_a.data[sib_j_base+1] * sib_delta;
                                    s_vel[sva+2] += mf_MiJt_a.data[sib_j_base+2] * sib_delta;
                                    s_vel[sva+3] += mf_MiJt_a.data[sib_j_base+3] * sib_delta;
                                    s_vel[sva+4] += mf_MiJt_a.data[sib_j_base+4] * sib_delta;
                                    s_vel[sva+5] += mf_MiJt_a.data[sib_j_base+5] * sib_delta;
                                }}
                                if (sib_lbb >= 0) {{
                                    int svb = sib_lbb * 6;
                                    s_vel[svb+0] += mf_MiJt_b.data[sib_j_base+0] * sib_delta;
                                    s_vel[svb+1] += mf_MiJt_b.data[sib_j_base+1] * sib_delta;
                                    s_vel[svb+2] += mf_MiJt_b.data[sib_j_base+2] * sib_delta;
                                    s_vel[svb+3] += mf_MiJt_b.data[sib_j_base+3] * sib_delta;
                                    s_vel[svb+4] += mf_MiJt_b.data[sib_j_base+4] * sib_delta;
                                    s_vel[svb+5] += mf_MiJt_b.data[sib_j_base+5] * sib_delta;
                                }}
                            }}
                        }}
                    }}

                    float delta_impulse = new_impulse - old_impulse;
                    s_impulse[i] = new_impulse;

                    // Apply velocity correction: v += MiJt * delta_impulse
                    int mijt_base = (c_off + i) * 6;
                    if (lba >= 0) {{
                        int va = lba * 6;
                        s_vel[va+0] += mf_MiJt_a.data[mijt_base+0] * delta_impulse;
                        s_vel[va+1] += mf_MiJt_a.data[mijt_base+1] * delta_impulse;
                        s_vel[va+2] += mf_MiJt_a.data[mijt_base+2] * delta_impulse;
                        s_vel[va+3] += mf_MiJt_a.data[mijt_base+3] * delta_impulse;
                        s_vel[va+4] += mf_MiJt_a.data[mijt_base+4] * delta_impulse;
                        s_vel[va+5] += mf_MiJt_a.data[mijt_base+5] * delta_impulse;
                    }}
                    if (lbb >= 0) {{
                        int vb = lbb * 6;
                        s_vel[vb+0] += mf_MiJt_b.data[mijt_base+0] * delta_impulse;
                        s_vel[vb+1] += mf_MiJt_b.data[mijt_base+1] * delta_impulse;
                        s_vel[vb+2] += mf_MiJt_b.data[mijt_base+2] * delta_impulse;
                        s_vel[vb+3] += mf_MiJt_b.data[mijt_base+3] * delta_impulse;
                        s_vel[vb+4] += mf_MiJt_b.data[mijt_base+4] * delta_impulse;
                        s_vel[vb+5] += mf_MiJt_b.data[mijt_base+5] * delta_impulse;
                    }}
                }}
            }}
        }}
        __syncwarp();

        // ═══════════════════════════════════════════════════════════════
        // STORE PHASE
        // ═══════════════════════════════════════════════════════════════

        // Write body velocities back to v_out
        for (int b = lane; b < n_bodies; b += 32) {{
            int dof = s_dof_start[b];
            for (int k = 0; k < 6; k++) {{
                v_out.data[dof + k] = s_vel[b * 6 + k];
            }}
        }}

        // Write impulses back
        for (int i = lane; i < m; i += 32) {{
            mf_impulses.data[c_off + i] = s_impulse[i];
        }}
    #endif
    """

        @wp.func_native(snippet)
        def pgs_solve_mf_native(
            world: int,
            mf_constraint_count: wp.array[int],
            mf_body_count: wp.array[int],
            mf_body_dof_start: wp.array2d[int],
            mf_local_body_a: wp.array2d[int],
            mf_local_body_b: wp.array2d[int],
            mf_J_a: wp.array3d[float],
            mf_J_b: wp.array3d[float],
            mf_MiJt_a: wp.array3d[float],
            mf_MiJt_b: wp.array3d[float],
            mf_eff_mass_inv: wp.array2d[float],
            mf_rhs: wp.array2d[float],
            mf_row_type: wp.array2d[int],
            mf_row_parent: wp.array2d[int],
            mf_row_mu: wp.array2d[float],
            mf_impulses: wp.array2d[float],
            v_out: wp.array[float],
            iterations: int,
            omega: float,
        ): ...

        def pgs_solve_mf_template(
            mf_constraint_count: wp.array[int],
            mf_body_count: wp.array[int],
            mf_body_dof_start: wp.array2d[int],
            mf_local_body_a: wp.array2d[int],
            mf_local_body_b: wp.array2d[int],
            mf_J_a: wp.array3d[float],
            mf_J_b: wp.array3d[float],
            mf_MiJt_a: wp.array3d[float],
            mf_MiJt_b: wp.array3d[float],
            mf_eff_mass_inv: wp.array2d[float],
            mf_rhs: wp.array2d[float],
            mf_row_type: wp.array2d[int],
            mf_row_parent: wp.array2d[int],
            mf_row_mu: wp.array2d[float],
            mf_impulses: wp.array2d[float],
            v_out: wp.array[float],
            iterations: int,
            omega: float,
        ):
            world, _lane = wp.tid()
            pgs_solve_mf_native(
                world,
                mf_constraint_count,
                mf_body_count,
                mf_body_dof_start,
                mf_local_body_a,
                mf_local_body_b,
                mf_J_a,
                mf_J_b,
                mf_MiJt_a,
                mf_MiJt_b,
                mf_eff_mass_inv,
                mf_rhs,
                mf_row_type,
                mf_row_parent,
                mf_row_mu,
                mf_impulses,
                v_out,
                iterations,
                omega,
            )

        name = f"pgs_solve_mf_{mf_max_constraints}_{max_mf_bodies}"
        pgs_solve_mf_template.__name__ = name
        pgs_solve_mf_template.__qualname__ = name
        return wp.kernel(enable_backward=False, module="unique")(pgs_solve_mf_template)

    @classmethod
    def get_pgs_fused_warp_kernel(
        cls,
        max_constraints: int,
        max_contact_triplets: int,
        mf_max_constraints: int,
        max_world_dofs: int,
        device: "wp.Device",
    ) -> "wp.Kernel":
        key = (max_constraints, max_contact_triplets, mf_max_constraints, max_world_dofs, device.arch)
        if key not in cls._pgs_fused_warp_cache:
            cls._pgs_fused_warp_cache[key] = cls._build_pgs_fused_warp_kernel(
                max_constraints, max_contact_triplets, mf_max_constraints, max_world_dofs
            )
        return cls._pgs_fused_warp_cache[key]

    @classmethod
    def _build_pgs_fused_warp_kernel(
        cls, max_constraints: int, max_contact_triplets: int, mf_max_constraints: int, max_world_dofs: int
    ) -> "wp.Kernel":
        """Fused two-phase GS PGS kernel — pure Warp tile API.

        Phase 1 (dense): tile-parallel dot/update over D DOFs using
        cooperative tile_load + tile_dot + tile_axpy. J/Y loads are
        software-pipelined (prefetch next row while computing current).
        Scoped @wp.func helpers limit register lifetimes. Contact metadata
        (diag, rhs) loaded as vec3 triplets to reduce broadcast loads.

        Phase 2 (MF): SIMT scalar code within tiled kernel. J/MiJt loads
        are software-pipelined and lane-parallel (lanes 0-5 body A, 6-11
        body B). Metadata packed into vec4i for single 16-byte loads.
        tile_extract (sync-free on shared) for velocity reads,
        tile_scatter_add(atomic=False) for velocity writes (lanes write
        distinct DOF indices), tile_scatter_masked for single-lane impulse
        writes.

        Both phases share s_v in shared memory. All PGS iterations run
        inside the kernel (no global round-trip for v between phases).
        One warp (32 threads) per world.
        """
        M_D = wp.constant(max_constraints)
        M_CT = wp.constant(max_contact_triplets)
        M_MF = mf_max_constraints
        M_MF_CONST = wp.constant(mf_max_constraints)
        D_val = max_world_dofs
        D = wp.constant(D_val)

        # Phase 1 helpers — scoped to limit register lifetimes
        @wp.func
        def dot_Jv(
            s_v: wp.tile(dtype=float, shape=(D_val,), storage="shared"),
            J_row: wp.tile(dtype=float, shape=(D_val,), storage="register"),
        ) -> float:
            """Compute J·v from a pre-loaded J row. Scoped — register tiles freed on return."""
            return wp.tile_extract(wp.tile_dot(J_row, s_v), 0)

        @wp.func
        def load_J_row(
            J_world: wp.array3d[float],
            world: int,
            row: int,
        ) -> wp.tile(dtype=float, shape=(D_val,), storage="register"):
            """Load a J/Y row into a register tile."""
            return wp.tile_load(J_world[world, row], shape=(D_val,), storage="register", bounds_check=False)

        @wp.func
        def velocity_update_preloaded(
            s_v: wp.tile(dtype=float, shape=(D_val,), storage="shared"),
            Y_row: wp.tile(dtype=float, shape=(D_val,), storage="register"),
            delta_impulse: float,
        ):
            """Apply s_v += Y * delta using a pre-loaded Y row."""
            wp.tile_axpy(delta_impulse, Y_row, s_v)

        @wp.func
        def velocity_update(
            s_v: wp.tile(dtype=float, shape=(D_val,), storage="shared"),
            Y_world: wp.array3d[float],
            world: int,
            row: int,
            delta_impulse: float,
        ):
            """Load Y row and apply s_v += Y * delta."""
            Y_row = wp.tile_load(Y_world[world, row], shape=(D_val,), storage="register", bounds_check=False)
            wp.tile_axpy(delta_impulse, Y_row, s_v)

        # Phase 2: SIMT within tiled kernel — each lane holds one element of the
        # 6-DOF body A/B operations (lanes 0-5 body A, 6-11 body B). Velocity reads
        # use tile_extract on shared s_v (sync-free). Impulse writes use
        # tile_scatter_masked; velocity writes use tile_scatter_add(atomic=False)
        # since body A and body B DOF ranges don't overlap within a constraint.
        @wp.func
        def pgs_mf_phase(
            world: int,
            lane: int,
            s_v: wp.tile(dtype=float, shape=(D_val,), storage="shared"),
            s_lam_mf: wp.tile(dtype=float, shape=(M_MF,), storage="shared"),
            mf_constraint_count: wp.array[int],
            mf_meta: wp.array2d[wp.vec4i],
            mf_J_a: wp.array3d[float],
            mf_J_b: wp.array3d[float],
            mf_MiJt_a: wp.array3d[float],
            mf_MiJt_b: wp.array3d[float],
            mf_row_mu: wp.array2d[float],
            omega: float,
        ):
            m_mf = mf_constraint_count[world]
            if m_mf > M_MF_CONST:
                m_mf = M_MF_CONST
            if m_mf == 0:
                return

            # Software-pipelined Phase 2: prefetch J/MiJt/metadata for constraint
            # i+1 while computing constraint i. Hides global memory latency.
            pre_Ja = float(0.0)
            pre_Jb = float(0.0)
            pre_MiJta = float(0.0)
            pre_MiJtb = float(0.0)
            pre_meta = wp.vec4i(0, 0, 0, 0)

            # Prefetch constraint 0
            if m_mf > 0:
                pre_meta = mf_meta[world, 0]
                if lane < 6:
                    pre_Ja = mf_J_a[world, 0, lane]
                    pre_MiJta = mf_MiJt_a[world, 0, lane]
                if lane >= 6 and lane < 12:
                    pre_Jb = mf_J_b[world, 0, lane - 6]
                    pre_MiJtb = mf_MiJt_b[world, 0, lane - 6]

            for i in range(m_mf):
                # Consume prefetched data
                cur_Ja = pre_Ja
                cur_Jb = pre_Jb
                cur_MiJta = pre_MiJta
                cur_MiJtb = pre_MiJtb
                meta = pre_meta

                # Prefetch i+1
                if i + 1 < m_mf:
                    pre_meta = mf_meta[world, i + 1]
                    if lane < 6:
                        pre_Ja = mf_J_a[world, i + 1, lane]
                        pre_MiJta = mf_MiJt_a[world, i + 1, lane]
                    if lane >= 6 and lane < 12:
                        pre_Jb = mf_J_b[world, i + 1, lane - 6]
                        pre_MiJtb = mf_MiJt_b[world, i + 1, lane - 6]

                # Unpack metadata (already prefetched)
                dof_a = meta[0] >> 16
                dof_b = (meta[0] << 16) >> 16  # sign-extend lower 16 bits
                mf_diag = wp.cast(meta[1], wp.float32)
                if mf_diag <= 0.0:
                    continue

                # J*v dot product — lane-parallel using prefetched J values
                my_jv = float(0.0)
                if lane < 6 and dof_a >= 0:
                    my_jv = cur_Ja * wp.tile_extract(s_v, dof_a + lane)
                if lane >= 6 and lane < 12 and dof_b >= 0:
                    my_jv = cur_Jb * wp.tile_extract(s_v, dof_b + lane - 6)
                # Reduce per-thread values: tile_full creates a register tile where
                # each thread's element is its own my_jv, then tile_sum reduces via shuffles.
                jv_tile = wp.tile_sum(wp.tile_full(shape=(32,), value=my_jv, dtype=float, storage="register"))
                jv = wp.tile_extract(jv_tile, 0)

                # PGS projection
                rhs_val = wp.cast(meta[2], wp.float32)
                residual = jv + rhs_val
                delta = -residual * mf_diag
                old_impulse = wp.tile_extract(s_lam_mf, i)
                new_impulse = old_impulse + omega * delta
                mf_rt = meta[3] & 0xFFFF

                if mf_rt == 0:
                    if new_impulse < 0.0:
                        new_impulse = 0.0
                elif mf_rt == 2:
                    mf_par = meta[3] >> 16
                    lambda_n = wp.tile_extract(s_lam_mf, mf_par)
                    mu = mf_row_mu[world, i]
                    radius = wp.max(mu * lambda_n, 0.0)

                    if radius <= 0.0:
                        new_impulse = 0.0
                    else:
                        sib = wp.where(i == mf_par + 1, mf_par + 2, mf_par + 1)
                        wp.tile_scatter_masked(s_lam_mf, i, new_impulse, lane == 0)
                        a_val = new_impulse
                        b_val = wp.tile_extract(s_lam_mf, sib)
                        mag = wp.sqrt(a_val * a_val + b_val * b_val)
                        if mag > radius:
                            scale = radius / mag
                            new_impulse = a_val * scale
                            sib_new = b_val * scale
                            sib_delta = sib_new - b_val
                            wp.tile_scatter_masked(s_lam_mf, sib, sib_new, lane == 0)

                            # Sibling velocity update — unpack sibling dofs from meta
                            sib_meta = mf_meta[world, sib]
                            sib_dof_a = sib_meta[0] >> 16
                            sib_dof_b = (sib_meta[0] << 16) >> 16
                            sib_idx = -1
                            sib_val = float(0.0)
                            if lane < 6 and sib_dof_a >= 0:
                                sib_idx = sib_dof_a + lane
                                sib_val = mf_MiJt_a[world, sib, lane] * sib_delta
                            elif lane >= 6 and lane < 12 and sib_dof_b >= 0:
                                sib_idx = sib_dof_b + lane - 6
                                sib_val = mf_MiJt_b[world, sib, lane - 6] * sib_delta
                            wp.tile_scatter_add(s_v, sib_idx, sib_val, sib_idx >= 0, atomic=False)

                delta_impulse = new_impulse - old_impulse
                wp.tile_scatter_masked(s_lam_mf, i, new_impulse, lane == 0)

                # Velocity update — lane-parallel scatter using prefetched MiJt
                if delta_impulse != 0.0:
                    idx = -1
                    val = float(0.0)
                    if lane < 6 and dof_a >= 0:
                        idx = dof_a + lane
                        val = cur_MiJta * delta_impulse
                    elif lane >= 6 and lane < 12 and dof_b >= 0:
                        idx = dof_b + lane - 6
                        val = cur_MiJtb * delta_impulse
                    wp.tile_scatter_add(s_v, idx, val, idx >= 0, atomic=False)

        def pgs_fused_warp(
            # Dense
            world_constraint_count: wp.array[int],
            dense_contact_row_count: wp.array[int],
            world_dof_start: wp.array[int],
            rhs_bias: wp.array2d[float],
            rhs_bias_vec3: wp.array2d[wp.vec3],
            world_diag: wp.array2d[float],
            world_diag_vec3: wp.array2d[wp.vec3],
            impulses_vec3: wp.array2d[wp.vec3],
            impulses_flat: wp.array2d[float],
            J_world: wp.array3d[float],
            Y_world: wp.array3d[float],
            world_row_mu: wp.array2d[float],
            # MF
            mf_constraint_count: wp.array[int],
            mf_meta: wp.array2d[wp.vec4i],
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
            world, thread = wp.tid()

            m_total = world_constraint_count[world]
            m_contact_rows = dense_contact_row_count[world]
            if m_total > M_D:
                m_total = M_D
            if m_contact_rows > M_D:
                m_contact_rows = M_D
            n_contacts = m_contact_rows // 3

            w_dof_start = world_dof_start[world]

            # ── LOAD PHASE ──
            s_v = wp.tile_load(v_out, shape=(D,), offset=(w_dof_start,), storage="shared", bounds_check=False)
            s_lam_contact = wp.tile_load(impulses_vec3[world], shape=(M_CT,), storage="shared", bounds_check=False)
            s_lam_mf = wp.tile_load(mf_impulses[world], shape=(M_MF_CONST,), storage="shared", bounds_check=False)
            # ── SOLVE PHASE ──
            for _iter in range(iterations):
                # ── Phase 1: Dense contacts (tile API, pipelined J+Y loads) ──
                # Prefetch first J and Y rows before loop
                pre_J = load_J_row(J_world, world, 0)
                pre_Y = load_J_row(Y_world, world, 0)

                for c in range(n_contacts):
                    row0 = c * 3
                    lam3 = wp.tile_extract(s_lam_contact, c)
                    diag3 = world_diag_vec3[world, c]
                    rhs3 = rhs_bias_vec3[world, c]

                    # Normal — consume prefetched J+Y, prefetch friction 1
                    cur_J = pre_J
                    cur_Y = pre_Y
                    pre_J = load_J_row(J_world, world, row0 + 1)
                    pre_Y = load_J_row(Y_world, world, row0 + 1)
                    if diag3[0] > 0.0:
                        jv = dot_Jv(s_v, cur_J)
                        new_n = wp.max(lam3[0] + omega * (-(jv + rhs3[0]) / diag3[0]), 0.0)
                        delta_n = new_n - lam3[0]
                        lam3 = wp.vec3(new_n, lam3[1], lam3[2])
                        if delta_n != 0.0:
                            velocity_update_preloaded(s_v, cur_Y, delta_n)

                    # Friction 1 — consume prefetched J+Y, prefetch friction 2
                    cur_J = pre_J
                    cur_Y = pre_Y
                    pre_J = load_J_row(J_world, world, row0 + 2)
                    pre_Y = load_J_row(Y_world, world, row0 + 2)
                    if diag3[1] > 0.0:
                        jv = dot_Jv(s_v, cur_J)
                        new_f1 = lam3[1] + omega * (-(jv + rhs3[1]) / diag3[1])
                        delta_f1 = new_f1 - lam3[1]
                        lam3 = wp.vec3(lam3[0], new_f1, lam3[2])
                        if delta_f1 != 0.0:
                            velocity_update_preloaded(s_v, cur_Y, delta_f1)

                    # Friction 2 — consume prefetched J+Y, prefetch next contact
                    cur_J = pre_J
                    cur_Y = pre_Y
                    if c + 1 < n_contacts:
                        pre_J = load_J_row(J_world, world, (c + 1) * 3)
                        pre_Y = load_J_row(Y_world, world, (c + 1) * 3)
                    if diag3[2] > 0.0:
                        jv = dot_Jv(s_v, cur_J)
                        new_f2 = lam3[2] + omega * (-(jv + rhs3[2]) / diag3[2])
                        delta_f2 = new_f2 - lam3[2]
                        lam3 = wp.vec3(lam3[0], lam3[1], new_f2)
                        if delta_f2 != 0.0:
                            velocity_update_preloaded(s_v, cur_Y, delta_f2)

                    # Friction cone projection (uses non-pipelined velocity_update
                    # for random-access sibling rows)
                    mu = world_row_mu[world, row0 + 1]
                    radius = wp.max(mu * lam3[0], 0.0)
                    if radius <= 0.0:
                        if lam3[1] != 0.0:
                            velocity_update(s_v, Y_world, world, row0 + 1, -lam3[1])
                        if lam3[2] != 0.0:
                            velocity_update(s_v, Y_world, world, row0 + 2, -lam3[2])
                        lam3 = wp.vec3(lam3[0], 0.0, 0.0)
                    else:
                        mag = wp.sqrt(lam3[1] * lam3[1] + lam3[2] * lam3[2])
                        if mag > radius:
                            scale = radius / mag
                            old_f1 = lam3[1]
                            old_f2 = lam3[2]
                            lam3 = wp.vec3(lam3[0], old_f1 * scale, old_f2 * scale)
                            velocity_update(s_v, Y_world, world, row0 + 1, lam3[1] - old_f1)
                            velocity_update(s_v, Y_world, world, row0 + 2, lam3[2] - old_f2)

                    wp.tile_scatter_masked(s_lam_contact, c, lam3, thread == 0)

                # ── Phase 1: Joint limits (tile API, scoped helpers) ──
                for i in range(m_contact_rows, m_total):
                    denom = world_diag[world, i]
                    if denom <= 0.0:
                        continue
                    J_row_i = load_J_row(J_world, world, i)
                    jv = dot_Jv(s_v, J_row_i)
                    old_impulse = impulses_flat[world, i]
                    new_impulse = wp.max(old_impulse + omega * (-(jv + rhs_bias[world, i]) / denom), 0.0)
                    delta_impulse = new_impulse - old_impulse
                    impulses_flat[world, i] = new_impulse
                    if delta_impulse != 0.0:
                        velocity_update(s_v, Y_world, world, i, delta_impulse)

                # ── Phase 2: MF constraints (SIMT Warp, shared tile access) ──
                pgs_mf_phase(
                    world,
                    thread,
                    s_v,
                    s_lam_mf,
                    mf_constraint_count,
                    mf_meta,
                    mf_J_a,
                    mf_J_b,
                    mf_MiJt_a,
                    mf_MiJt_b,
                    mf_row_mu,
                    omega,
                )

            # ── STORE PHASE ──
            wp.tile_store(v_out, s_v, offset=(w_dof_start,), bounds_check=False)
            wp.tile_store(impulses_vec3[world], s_lam_contact, bounds_check=False)
            wp.tile_store(mf_impulses[world], s_lam_mf, bounds_check=False)

        name = f"pgs_fused_warp_{max_constraints}_{max_contact_triplets}_{mf_max_constraints}_{max_world_dofs}"
        pgs_fused_warp.__name__ = name
        pgs_fused_warp.__qualname__ = name
        return wp.kernel(enable_backward=False, module="unique", launch_bounds=(32, 12))(pgs_fused_warp)

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
        cls,
        max_constraints: int,
        mf_max_constraints: int,
        max_world_dofs: int,
        device: "wp.Device",
        friction_mode: str = "current",
    ) -> "wp.Kernel":
        """Get or create a two-phase GS kernel for matrix-free articulated PGS.

        Phase 1 processes dense constraints (via J_world/Y_world at ``max_constraints``).
        Phase 2 processes MF constraints (via mf_J/mf_MiJt at ``mf_max_constraints``).
        Both phases share a single velocity vector in shared memory.

        ``friction_mode`` selects the MF friction-row projection:

        * ``"current"`` (default): isotropic Coulomb cone projection,
          matching :func:`friction_step_current`.
        * ``"bisection"``: RAISim-style bisection on λ_n (see
          :func:`friction_step_bisection`), ported from Miles Macklin's
          ``USE_BISECTION`` branch of ``raisim/kernels.py::gs_contact_sweep``.
        * ``"bisection_desaxce"``: ``"bisection"`` with the de Saxce
          max-dissipation bias (``μ · ‖c_T‖`` added to the normal
          target) — FPGS Friction Modes 6/13.
        * ``"coulomb_newton"``: Gilles Daviet's scalar bracketed-Newton
          on alpha (see :func:`friction_step_coulomb_newton`) — FPGS
          Friction Modes 7/13.
        """
        key = (max_constraints, mf_max_constraints, max_world_dofs, device.arch, friction_mode)
        if key not in cls._pgs_solve_mf_gs_cache:
            cls._pgs_solve_mf_gs_cache[key] = cls._build_pgs_solve_mf_gs_kernel(
                max_constraints, mf_max_constraints, max_world_dofs, friction_mode
            )
        return cls._pgs_solve_mf_gs_cache[key]

    @classmethod
    def _build_pgs_solve_mf_gs_kernel(
        cls,
        max_constraints: int,
        mf_max_constraints: int,
        max_world_dofs: int,
        friction_mode: str = "current",
    ) -> "wp.Kernel":
        """Two-phase GS PGS kernel: dense + matrix-free in one pass.

        Uses one warp (32 threads) per world.

        Phase 1 (dense): warp-parallel dot/update over D DOFs using J_world/Y_world.
        Phase 2 (MF): lanes 0-5 handle body_a, lanes 6-11 handle body_b (6 DOFs each).

        Shared memory layout:
          s_v[D] — world velocity
          s_lam_dense[M_D] + metadata — dense impulses and constraint info
          s_lam_mf[M_MF] — MF impulses (metadata read from global per constraint)
        """
        M_D = max_constraints
        M_MF = mf_max_constraints
        D = max_world_dofs

        # How many DOF elements each lane handles (ceil(D/32))
        ELEMS_PER_LANE = (D + 31) // 32

        # --- Code generation for dense phase (D-wide dot/update, software-pipelined) ---

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

        # --- MF friction-row projection -------------------------------------
        # ``friction_mode="current"`` keeps the legacy isotropic Coulomb
        # cone projection (matches :func:`friction_step_current`).
        # ``friction_mode="bisection"`` runs the RAISim bisection on λ_n
        # (matches :func:`friction_step_bisection`).
        # ``friction_mode="bisection_desaxce"`` runs the same bisection
        # with the de Saxce max-dissipation bias (``μ · ‖c_T‖`` added to
        # the normal target velocity) — FPGS Friction Modes 6/13.
        # ``friction_mode="coulomb_newton"`` runs Gilles Daviet's scalar
        # bracketed-Newton on alpha (matches
        # :func:`friction_step_coulomb_newton`) — FPGS Friction Modes
        # 7/13.  We inject one of these CUDA blocks into the MF phase;
        # the outer row loop handles the standard ``delta_impulse``
        # accounting for the first-friction row (and t2's delta
        # collapses to zero because ``s_lam_mf[i_t2]`` is pre-written
        # by the per-contact solve).
        if friction_mode in ("bisection", "bisection_desaxce"):
            mf_friction_block = """
                    // __BISECTION_LABEL__
                    int mf_par = packed_tp >> 16;
                    int i_t1 = mf_par + 1;
                    int i_t2 = mf_par + 2;

                    if (i != i_t1) {
                        // Second friction row of the triple: the bisection
                        // that ran at i == i_t1 already wrote s_lam_mf[i_t2].
                        // Collapse delta_impulse to zero so no further
                        // v_out correction is applied for this row.
                        new_impulse = s_lam_mf[i];
                    } else {
                        // First friction row of the triple. Run bisection on
                        // (λ_n, λ_t1, λ_t2), apply v_out corrections for the
                        // normal and t2 siblings in-place, and return
                        // new_λ_t1 as the ``new_impulse`` so the outer kernel
                        // applies the t1 v_out delta via its usual path.
                        int n_mf6 = mf6_base + mf_par * 6;
                        int t1_mf6 = mf6_base + i_t1 * 6;
                        int t2_mf6 = mf6_base + i_t2 * 6;

                        // Body indices are shared across the triple: read
                        // from the parent row's packed meta.
                        int parent_packed_dofs = mf_meta.data[off_meta + mf_par * 4];
                        int dof_a_par = parent_packed_dofs >> 16;
                        int dof_b_par = (parent_packed_dofs << 16) >> 16;
                        // FPGS stores rhs = beta * phi / dt (negative on
                        // penetration).  RAISim's bisection targets
                        // ``u_n >= b_n`` with ``b_n = -erp * gap / dt``
                        // (positive on penetration).  Flip the sign once
                        // so the rest of this block mirrors RAISim.
                        float target_vel_n = -__int_as_float(
                            mf_meta.data[off_meta + mf_par * 4 + 2]);
                        float mu = mf_row_mu.data[off_mf + i];

                        float old_lambda_n = s_lam_mf[mf_par];
                        float old_lambda_t1 = s_lam_mf[i_t1];
                        float old_lambda_t2 = s_lam_mf[i_t2];

                        float new_lambda_n = old_lambda_n;
                        float new_lambda_t1 = old_lambda_t1;
                        float new_lambda_t2 = old_lambda_t2;
                        float d_n_total = 0.0f;
                        float d_t2_total = 0.0f;

                        // Lane 0 runs the serial bisection; other lanes
                        // wait and then consume the broadcast results.
                        if (lane == 0) {
                            float u_n = 0.0f, u_t1 = 0.0f, u_t2 = 0.0f;
                            if (dof_a_par >= 0) {
                                for (int k = 0; k < 6; k++) {
                                    float va = s_v[dof_a_par + k];
                                    u_n  += mf_J_a.data[n_mf6  + k] * va;
                                    u_t1 += mf_J_a.data[t1_mf6 + k] * va;
                                    u_t2 += mf_J_a.data[t2_mf6 + k] * va;
                                }
                            }
                            if (dof_b_par >= 0) {
                                for (int k = 0; k < 6; k++) {
                                    float vb = s_v[dof_b_par + k];
                                    u_n  += mf_J_b.data[n_mf6  + k] * vb;
                                    u_t1 += mf_J_b.data[t1_mf6 + k] * vb;
                                    u_t2 += mf_J_b.data[t2_mf6 + k] * vb;
                                }
                            }

                            // __DESAXCE_BIAS__

                            float G_nn = 0.0f, G_nt1 = 0.0f, G_nt2 = 0.0f;
                            float G_t1t1 = 0.0f, G_t1t2 = 0.0f, G_t2t2 = 0.0f;
                            if (dof_a_par >= 0) {
                                for (int k = 0; k < 6; k++) {
                                    float Jna = mf_J_a.data[n_mf6  + k];
                                    float Jt1a = mf_J_a.data[t1_mf6 + k];
                                    float Jt2a = mf_J_a.data[t2_mf6 + k];
                                    float Mna = mf_MiJt_a.data[n_mf6  + k];
                                    float Mt1a = mf_MiJt_a.data[t1_mf6 + k];
                                    float Mt2a = mf_MiJt_a.data[t2_mf6 + k];
                                    G_nn   += Jna * Mna;
                                    G_nt1  += Jna * Mt1a;
                                    G_nt2  += Jna * Mt2a;
                                    G_t1t1 += Jt1a * Mt1a;
                                    G_t1t2 += Jt1a * Mt2a;
                                    G_t2t2 += Jt2a * Mt2a;
                                }
                            }
                            if (dof_b_par >= 0) {
                                for (int k = 0; k < 6; k++) {
                                    float Jnb = mf_J_b.data[n_mf6  + k];
                                    float Jt1b = mf_J_b.data[t1_mf6 + k];
                                    float Jt2b = mf_J_b.data[t2_mf6 + k];
                                    float Mnb = mf_MiJt_b.data[n_mf6  + k];
                                    float Mt1b = mf_MiJt_b.data[t1_mf6 + k];
                                    float Mt2b = mf_MiJt_b.data[t2_mf6 + k];
                                    G_nn   += Jnb * Mnb;
                                    G_nt1  += Jnb * Mt1b;
                                    G_nt2  += Jnb * Mt2b;
                                    G_t1t1 += Jt1b * Mt1b;
                                    G_t1t2 += Jt1b * Mt2b;
                                    G_t2t2 += Jt2b * Mt2b;
                                }
                            }

                            if (G_nn >= 1.0e-20f) {
                                // Check separating contact (λ_n = 0 solves it)
                                float u_n_at_zero = u_n + G_nn * (0.0f - old_lambda_n);
                                if (u_n_at_zero >= target_vel_n) {
                                    new_lambda_n = 0.0f;
                                    new_lambda_t1 = 0.0f;
                                    new_lambda_t2 = 0.0f;
                                } else {
                                    float lo = 0.0f;
                                    float hi = fmaxf(
                                        old_lambda_n * 2.0f,
                                        (target_vel_n - u_n) / G_nn + old_lambda_n);
                                    hi = fmaxf(hi, 1.0f);

                                    for (int _bi = 0; _bi < 20; _bi++) {
                                        float mid = 0.5f * (lo + hi);
                                        float d_n = mid - old_lambda_n;
                                        float ut1_eff = u_t1 + G_nt1 * d_n;
                                        float ut2_eff = u_t2 + G_nt2 * d_n;

                                        float det = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
                                        float d_t1 = 0.0f, d_t2 = 0.0f;
                                        if (fabsf(det) > 1.0e-20f) {
                                            d_t1 = (-ut1_eff * G_t2t2 + ut2_eff * G_t1t2) / det;
                                            d_t2 = ( ut1_eff * G_t1t2 - ut2_eff * G_t1t1) / det;
                                        }
                                        float trial_t1 = old_lambda_t1 + d_t1;
                                        float trial_t2 = old_lambda_t2 + d_t2;

                                        float flimit = mu * mid;
                                        float tmag = sqrtf(
                                            trial_t1 * trial_t1 + trial_t2 * trial_t2);
                                        if (tmag > flimit && tmag > 1.0e-20f) {
                                            float sc = flimit / tmag;
                                            trial_t1 *= sc;
                                            trial_t2 *= sc;
                                        }
                                        float d_t1_actual = trial_t1 - old_lambda_t1;
                                        float d_t2_actual = trial_t2 - old_lambda_t2;
                                        float u_n_trial = u_n + G_nn * d_n
                                            + G_nt1 * d_t1_actual
                                            + G_nt2 * d_t2_actual;
                                        if (u_n_trial < target_vel_n) lo = mid;
                                        else hi = mid;
                                    }
                                    new_lambda_n = 0.5f * (lo + hi);

                                    float d_n_final = new_lambda_n - old_lambda_n;
                                    float ut1_f = u_t1 + G_nt1 * d_n_final;
                                    float ut2_f = u_t2 + G_nt2 * d_n_final;
                                    float det_f = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2;
                                    float d_t1_f = 0.0f, d_t2_f = 0.0f;
                                    if (fabsf(det_f) > 1.0e-20f) {
                                        d_t1_f = (-ut1_f * G_t2t2 + ut2_f * G_t1t2) / det_f;
                                        d_t2_f = ( ut1_f * G_t1t2 - ut2_f * G_t1t1) / det_f;
                                    }
                                    new_lambda_t1 = old_lambda_t1 + d_t1_f;
                                    new_lambda_t2 = old_lambda_t2 + d_t2_f;

                                    float flimit_f = mu * new_lambda_n;
                                    float tmag_f = sqrtf(
                                        new_lambda_t1 * new_lambda_t1
                                        + new_lambda_t2 * new_lambda_t2);
                                    if (tmag_f > flimit_f && tmag_f > 1.0e-20f) {
                                        float sc_f = flimit_f / tmag_f;
                                        new_lambda_t1 *= sc_f;
                                        new_lambda_t2 *= sc_f;
                                    }
                                }
                                d_n_total = new_lambda_n - old_lambda_n;
                                d_t2_total = new_lambda_t2 - old_lambda_t2;
                                s_lam_mf[mf_par] = new_lambda_n;
                                s_lam_mf[i_t2]   = new_lambda_t2;
                            }
                        }

                        __syncwarp();
                        new_lambda_t1 = __shfl_sync(MASK, new_lambda_t1, 0);
                        d_n_total = __shfl_sync(MASK, d_n_total, 0);
                        d_t2_total = __shfl_sync(MASK, d_t2_total, 0);

                        // Apply the normal-row v_out delta (all lanes).
                        if (d_n_total != 0.0f) {
                            if (lane < 6 && dof_a_par >= 0) {
                                s_v[dof_a_par + lane] +=
                                    mf_MiJt_a.data[n_mf6 + lane] * d_n_total;
                            }
                            if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                                s_v[dof_b_par + lane - 6] +=
                                    mf_MiJt_b.data[n_mf6 + lane - 6] * d_n_total;
                            }
                        }
                        // Apply the t2-row v_out delta (all lanes).
                        if (d_t2_total != 0.0f) {
                            if (lane < 6 && dof_a_par >= 0) {
                                s_v[dof_a_par + lane] +=
                                    mf_MiJt_a.data[t2_mf6 + lane] * d_t2_total;
                            }
                            if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                                s_v[dof_b_par + lane - 6] +=
                                    mf_MiJt_b.data[t2_mf6 + lane - 6] * d_t2_total;
                            }
                        }
                        __syncwarp();
                        new_impulse = new_lambda_t1;
                    }
"""
            # Inject the per-mode label and the optional de Saxce bias
            # correction (``μ · ‖c_T‖`` added to ``target_vel_n`` once per
            # contact before the bisection) into the shared CUDA body.
            if friction_mode == "bisection_desaxce":
                mf_friction_block = mf_friction_block.replace(
                    "// __BISECTION_LABEL__",
                    '// friction_mode="bisection_desaxce": RAISim bisection + de Saxce bias (μ·‖c_T‖).',
                )
                mf_friction_block = mf_friction_block.replace(
                    "// __DESAXCE_BIAS__",
                    "{\n"
                    "                                float c_T_mag = sqrtf(u_t1 * u_t1 + u_t2 * u_t2);\n"
                    "                                target_vel_n = target_vel_n + mu * c_T_mag;\n"
                    "                            }",
                )
            else:
                mf_friction_block = mf_friction_block.replace(
                    "// __BISECTION_LABEL__",
                    '// friction_mode="bisection": RAISim-style bisection on λ_n.',
                )
                # Pure RAISim bisection: no de Saxce bias.
                mf_friction_block = mf_friction_block.replace(
                    "// __DESAXCE_BIAS__",
                    '// (no de Saxce bias; friction_mode="bisection".)',
                )
        elif friction_mode == "coulomb_newton":
            # Gilles Daviet's 1D Coulomb Newton (7/13).  CUDA port of
            # :func:`friction_step_coulomb_newton`; mirrors the
            # bisection block's structure — lane 0 runs the serial
            # scalar Newton, then broadcasts (new_t1, d_n, d_t2) to
            # every lane via ``__shfl_sync`` so the v_out deltas for
            # the normal / t2 rows are applied cooperatively.
            mf_friction_block = """
                    // friction_mode="coulomb_newton": Gilles Daviet's
                    // scalar bracketed-Newton on alpha.  Ported from
                    // ``coulomb_root_finding_warp.py::solve_coulomb`` —
                    // see :func:`friction_step_coulomb_newton` for the
                    // matrix-free row-data mapping.
                    int mf_par = packed_tp >> 16;
                    int i_t1 = mf_par + 1;
                    int i_t2 = mf_par + 2;

                    if (i != i_t1) {
                        // Second friction row of the triple: the
                        // Newton solve that ran at i == i_t1 already
                        // wrote s_lam_mf[i_t2].  Collapse
                        // delta_impulse to zero for this row.
                        new_impulse = s_lam_mf[i];
                    } else {
                        int n_mf6 = mf6_base + mf_par * 6;
                        int t1_mf6 = mf6_base + i_t1 * 6;
                        int t2_mf6 = mf6_base + i_t2 * 6;

                        int parent_packed_dofs = mf_meta.data[off_meta + mf_par * 4];
                        int dof_a_par = parent_packed_dofs >> 16;
                        int dof_b_par = (parent_packed_dofs << 16) >> 16;
                        float target_vel_n = -__int_as_float(
                            mf_meta.data[off_meta + mf_par * 4 + 2]);
                        float mu = mf_row_mu.data[off_mf + i];

                        float old_lambda_n = s_lam_mf[mf_par];
                        float old_lambda_t1 = s_lam_mf[i_t1];
                        float old_lambda_t2 = s_lam_mf[i_t2];

                        float new_lambda_n = old_lambda_n;
                        float new_lambda_t1 = old_lambda_t1;
                        float new_lambda_t2 = old_lambda_t2;
                        float d_n_total = 0.0f;
                        float d_t2_total = 0.0f;

                        if (lane == 0) {
                            float u_n = 0.0f, u_t1 = 0.0f, u_t2 = 0.0f;
                            if (dof_a_par >= 0) {
                                for (int k = 0; k < 6; k++) {
                                    float va = s_v[dof_a_par + k];
                                    u_n  += mf_J_a.data[n_mf6  + k] * va;
                                    u_t1 += mf_J_a.data[t1_mf6 + k] * va;
                                    u_t2 += mf_J_a.data[t2_mf6 + k] * va;
                                }
                            }
                            if (dof_b_par >= 0) {
                                for (int k = 0; k < 6; k++) {
                                    float vb = s_v[dof_b_par + k];
                                    u_n  += mf_J_b.data[n_mf6  + k] * vb;
                                    u_t1 += mf_J_b.data[t1_mf6 + k] * vb;
                                    u_t2 += mf_J_b.data[t2_mf6 + k] * vb;
                                }
                            }

                            float G_nn = 0.0f, G_nt1 = 0.0f, G_nt2 = 0.0f;
                            float G_t1t1 = 0.0f, G_t1t2 = 0.0f, G_t2t2 = 0.0f;
                            if (dof_a_par >= 0) {
                                for (int k = 0; k < 6; k++) {
                                    float Jna = mf_J_a.data[n_mf6  + k];
                                    float Jt1a = mf_J_a.data[t1_mf6 + k];
                                    float Jt2a = mf_J_a.data[t2_mf6 + k];
                                    float Mna = mf_MiJt_a.data[n_mf6  + k];
                                    float Mt1a = mf_MiJt_a.data[t1_mf6 + k];
                                    float Mt2a = mf_MiJt_a.data[t2_mf6 + k];
                                    G_nn   += Jna * Mna;
                                    G_nt1  += Jna * Mt1a;
                                    G_nt2  += Jna * Mt2a;
                                    G_t1t1 += Jt1a * Mt1a;
                                    G_t1t2 += Jt1a * Mt2a;
                                    G_t2t2 += Jt2a * Mt2a;
                                }
                            }
                            if (dof_b_par >= 0) {
                                for (int k = 0; k < 6; k++) {
                                    float Jnb = mf_J_b.data[n_mf6  + k];
                                    float Jt1b = mf_J_b.data[t1_mf6 + k];
                                    float Jt2b = mf_J_b.data[t2_mf6 + k];
                                    float Mnb = mf_MiJt_b.data[n_mf6  + k];
                                    float Mt1b = mf_MiJt_b.data[t1_mf6 + k];
                                    float Mt2b = mf_MiJt_b.data[t2_mf6 + k];
                                    G_nn   += Jnb * Mnb;
                                    G_nt1  += Jnb * Mt1b;
                                    G_nt2  += Jnb * Mt2b;
                                    G_t1t1 += Jt1b * Mt1b;
                                    G_t1t2 += Jt1b * Mt2b;
                                    G_t2t2 += Jt2b * Mt2b;
                                }
                            }

                            if (G_nn >= 1.0e-20f) {
                                // u_free = u_current - G * lambda_old
                                float u_free_n = u_n - (G_nn * old_lambda_n
                                    + G_nt1 * old_lambda_t1
                                    + G_nt2 * old_lambda_t2);
                                float u_free_t1 = u_t1 - (G_nt1 * old_lambda_n
                                    + G_t1t1 * old_lambda_t1
                                    + G_t1t2 * old_lambda_t2);
                                float u_free_t2 = u_t2 - (G_nt2 * old_lambda_n
                                    + G_t1t2 * old_lambda_t1
                                    + G_t2t2 * old_lambda_t2);

                                if (u_free_n < target_vel_n) {
                                    // Reference b vector shifted into the
                                    // positive-target frame (b_N < 0 on
                                    // penetrating contacts).
                                    float bN = u_free_n - target_vel_n;
                                    float bT0 = u_free_t1;
                                    float bT1 = u_free_t2;

                                    float WN = G_nn;
                                    float wNT0 = G_nt1;
                                    float wNT1 = G_nt2;
                                    // A_T = W_T - wNT wNT^T / WN.
                                    float AT00 = G_t1t1 - (wNT0 * wNT0) / WN;
                                    float AT01 = G_t1t2 - (wNT0 * wNT1) / WN;
                                    float AT10 = G_t1t2 - (wNT1 * wNT0) / WN;
                                    float AT11 = G_t2t2 - (wNT1 * wNT1) / WN;
                                    // c_T = bT - (bN / WN) * wNT.
                                    float cT0 = bT0 - (bN / WN) * wNT0;
                                    float cT1 = bT1 - (bN / WN) * wNT1;

                                    // Sticking check at alpha = 0.
                                    float inv_det0 = 1.0f /
                                        (AT00 * AT11 - AT01 * AT10);
                                    float s0_0 = (AT11 * cT0 - AT01 * cT1) * inv_det0;
                                    float s0_1 = (AT00 * cT1 - AT10 * cT0) * inv_det0;
                                    float norm_s0 = sqrtf(s0_0 * s0_0 + s0_1 * s0_1);
                                    float phi0 = norm_s0 - mu *
                                        (wNT0 * s0_0 + wNT1 * s0_1 - bN) / WN;

                                    float last_s0 = 0.0f, last_s1 = 0.0f;
                                    float alpha = 0.0f;

                                    if (phi0 <= 0.0f) {
                                        // Sticking: alpha = 0, s = s0.
                                        last_s0 = s0_0;
                                        last_s1 = s0_1;
                                        alpha = 0.0f;
                                    } else {
                                        // Bracket: double hi until phi(hi) < 0.
                                        float hi = 1.0f;
                                        for (int _e = 0; _e < 30; _e++) {
                                            float a_hi = AT00 + hi;
                                            float d_hi = AT11 + hi;
                                            float idet_hi = 1.0f /
                                                (a_hi * d_hi - AT01 * AT10);
                                            float sh0 = (d_hi * cT0 - AT01 * cT1) * idet_hi;
                                            float sh1 = (a_hi * cT1 - AT10 * cT0) * idet_hi;
                                            float ns_hi = sqrtf(sh0 * sh0 + sh1 * sh1);
                                            float phi_hi = ns_hi - mu *
                                                (wNT0 * sh0 + wNT1 * sh1 - bN) / WN;
                                            if (phi_hi < 0.0f) break;
                                            hi *= 2.0f;
                                        }

                                        float lo = 0.0f;
                                        float x = 0.5f * (lo + hi);
                                        float tol = 1.0e-6f + 1.0e-6f * phi0;

                                        for (int _it = 0; _it < 50; _it++) {
                                            float ax = AT00 + x;
                                            float dx = AT11 + x;
                                            float idet = 1.0f /
                                                (ax * dx - AT01 * AT10);
                                            float s0 = (dx * cT0 - AT01 * cT1) * idet;
                                            float s1 = (ax * cT1 - AT10 * cT0) * idet;
                                            float t0 = (dx * s0 - AT01 * s1) * idet;
                                            float t1 = (ax * s1 - AT10 * s0) * idet;
                                            float ns = sqrtf(s0 * s0 + s1 * s1);
                                            float fx = ns - mu *
                                                (wNT0 * s0 + wNT1 * s1 - bN) / WN;
                                            float dfx = -(s0 * t0 + s1 * t1) / ns
                                                + mu * (wNT0 * t0 + wNT1 * t1) / WN;

                                            last_s0 = s0;
                                            last_s1 = s1;

                                            if (fabsf(fx) < tol
                                                || fabsf(hi - lo) < 1.0e-6f * (1.0f + hi)) {
                                                alpha = x;
                                                break;
                                            }

                                            if (fx > 0.0f) lo = x;
                                            else hi = x;

                                            float x_new = 0.5f * (lo + hi);
                                            if (dfx != 0.0f) {
                                                float x_newton = x - fx / dfx;
                                                if (x_newton > lo && x_newton < hi) {
                                                    x_new = x_newton;
                                                }
                                            }
                                            x = x_new;
                                            alpha = x;
                                        }
                                    }

                                    // Recover r_T, r_N from the last s.
                                    float rT0 = -last_s0;
                                    float rT1 = -last_s1;
                                    float rN = -(wNT0 * rT0 + wNT1 * rT1 + bN) / WN;
                                    new_lambda_n = rN;
                                    new_lambda_t1 = rT0;
                                    new_lambda_t2 = rT1;
                                } else {
                                    // Separating contact: zero impulse.
                                    new_lambda_n = 0.0f;
                                    new_lambda_t1 = 0.0f;
                                    new_lambda_t2 = 0.0f;
                                }

                                d_n_total = new_lambda_n - old_lambda_n;
                                d_t2_total = new_lambda_t2 - old_lambda_t2;
                                s_lam_mf[mf_par] = new_lambda_n;
                                s_lam_mf[i_t2]   = new_lambda_t2;
                            }
                        }

                        __syncwarp();
                        new_lambda_t1 = __shfl_sync(MASK, new_lambda_t1, 0);
                        d_n_total = __shfl_sync(MASK, d_n_total, 0);
                        d_t2_total = __shfl_sync(MASK, d_t2_total, 0);

                        // Apply normal-row v_out delta (all lanes).
                        if (d_n_total != 0.0f) {
                            if (lane < 6 && dof_a_par >= 0) {
                                s_v[dof_a_par + lane] +=
                                    mf_MiJt_a.data[n_mf6 + lane] * d_n_total;
                            }
                            if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                                s_v[dof_b_par + lane - 6] +=
                                    mf_MiJt_b.data[n_mf6 + lane - 6] * d_n_total;
                            }
                        }
                        // Apply t2-row v_out delta (all lanes).
                        if (d_t2_total != 0.0f) {
                            if (lane < 6 && dof_a_par >= 0) {
                                s_v[dof_a_par + lane] +=
                                    mf_MiJt_a.data[t2_mf6 + lane] * d_t2_total;
                            }
                            if (lane >= 6 && lane < 12 && dof_b_par >= 0) {
                                s_v[dof_b_par + lane - 6] +=
                                    mf_MiJt_b.data[t2_mf6 + lane - 6] * d_t2_total;
                            }
                        }
                        __syncwarp();
                        new_impulse = new_lambda_t1;
                    }
"""
        else:
            mf_friction_block = """
                    // friction_mode="current": isotropic Coulomb cone clamp.
                    int mf_par = packed_tp >> 16;
                    float lambda_n = s_lam_mf[mf_par];
                    float mu = mf_row_mu.data[off_mf + i];
                    float radius = fmaxf(mu * lambda_n, 0.0f);

                    if (radius <= 0.0f) {
                        new_impulse = 0.0f;
                    } else {
                        int sib = (i == mf_par + 1) ? mf_par + 2 : mf_par + 1;
                        s_lam_mf[i] = new_impulse;
                        float a_val = new_impulse;
                        float b_val = s_lam_mf[sib];
                        float mag = sqrtf(a_val * a_val + b_val * b_val);
                        if (mag > radius) {
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
                            if (lane < 6 && sib_dof_a >= 0) {
                                s_v[sib_dof_a + lane] += mf_MiJt_a.data[sib_mf6 + lane] * sib_delta;
                            }
                            if (lane >= 6 && lane < 12 && sib_dof_b >= 0) {
                                s_v[sib_dof_b + lane - 6] += mf_MiJt_b.data[sib_mf6 + lane - 6] * sib_delta;
                            }
                        }
                    }
"""

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

            // ── Phase 1: Dense constraints (D-DOF warp-parallel, software-pipelined) ──

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

                // row_type 0=CONTACT, 3=JOINT_LIMIT, 4=JOINT_VELOCITY_LIMIT:
                // unilateral lambda >= 0 projector. The velocity-limit row
                // uses a signed Jacobian so one side of the bilateral
                // [-qdot_max, +qdot_max] box is active at a time.
                if (row_type == 0 || row_type == 3 || row_type == 4) {{
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

            // ── Phase 2: MF constraints (6-DOF per body, software-pipelined) ──

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
                    {mf_friction_block}
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

        name = f"pgs_solve_mf_gs_{max_constraints}_{mf_max_constraints}_{max_world_dofs}_{friction_mode}"
        pgs_solve_mf_gs_template.__name__ = name
        pgs_solve_mf_gs_template.__qualname__ = name
        return wp.kernel(enable_backward=False, module="unique")(pgs_solve_mf_gs_template)
