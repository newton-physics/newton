# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import BodyFlags, Contacts, Control, Model, State
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
    TILE_THREADS,
    TiledKernelFactory,
    add_dense_contact_compliance_to_diag,
    allocate_joint_limit_slots,
    allocate_joint_velocity_limit_slots,
    allocate_world_contact_slots,
    apply_augmented_joint_tau,
    apply_augmented_mass_diagonal_grouped,
    build_augmented_joint_rows,
    build_mass_update_mask,
    build_mf_contact_rows,
    clamp_augmented_joint_u0,
    compute_body_parent_f,
    compute_com_transforms,
    compute_composite_inertia,
    compute_contact_linear_force_from_impulses,
    compute_mf_body_Hinv,
    compute_mf_effective_mass_and_rhs,
    compute_mf_world_dof_offsets,
    compute_spatial_inertia,
    compute_velocity_predictor_and_seed_world_velocity,
    compute_world_contact_bias,
    convert_root_free_qd_local_to_world,
    convert_root_free_qd_world_to_local,
    copy_int_array_masked,
    crba_fill_par_dof,
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
    integrate_generalized_joints,
    pack_contact_linear_force_as_spatial,
    pack_contact_triplets_vec3,
    pack_mf_meta,
    populate_joint_limit_J_for_size,
    populate_joint_velocity_limit_J_for_size,
    populate_world_J_for_size,
    prepare_world_impulses,
    scatter_qdd_from_groups,
    scatter_world_velocity,
    update_articulation_origins,
    update_articulation_root_com_offsets,
    update_body_qd_from_featherstone,
    update_qdd_from_velocity,
    update_qdd_from_world_velocity,
)


class SolverFeatherPGS(SolverBase):
    """Private CUDA-only FeatherPGS prototype using one fused Warp path.

    The solver advances reduced-coordinate articulations with CRBA/RNEA and
    solves contact, joint-position limit, and joint-velocity limit rows through
    a matrix-free projected Gauss-Seidel sweep. Velocity limits are always
    considered for finite positive ``model.joint_velocity_limit`` entries.

    Unsupported model features intentionally fail early while this remains a
    private API. Kinematic bodies are not supported, and model mutations require
    recreating the solver. When requested on the model, ``State.body_parent_f``
    is populated from the solver's RNEA backward pass.
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
        max_constraints: int = 36,
        mf_max_constraints: int = 512,
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
            model.joint_velocity_limit: Joint velocity limits are always enforced
                as per-DOF PGS constraints when finite and positive. When
                ``|qdot_i| > qdot_max_i``, a single signed-Jacobian row projects
                ``qdot_i`` back onto the bilateral box
                ``[-qdot_max_i, +qdot_max_i]``. No Baumgarte bias.
            pgs_iterations (int, optional): Number of Gauss-Seidel iterations to apply per frame. Defaults to 12.
            pgs_beta (float, optional): ERP style position correction factor. Defaults to 0.2.
            pgs_cfm (float, optional): Compliance/regularization added to the Delassus diagonal. Defaults to 1.0e-6.
            contact_compliance (float, optional): Normal contact compliance [m/N] applied
                to articulated contact rows. Converted to an impulse-space diagonal term
                using ``compliance / dt^2``. Defaults to 0.0.
            pgs_omega (float, optional): Successive over-relaxation factor for the PGS sweep. Defaults to 1.0.
            max_constraints (int, optional): Maximum number of articulated contact constraint
                rows stored per world. Free rigid body contacts are stored separately, bounded by
                mf_max_constraints. Contact triplet storage is padded internally when needed.
                Defaults to 36.
            mf_max_constraints (int, optional): Maximum number of matrix-free constraints per world. Defaults to 512.

        """
        super().__init__(model)

        if update_mass_matrix_interval < 1:
            raise ValueError("update_mass_matrix_interval must be >= 1.")
        if max_constraints < 1:
            raise ValueError("max_constraints must be positive.")
        if mf_max_constraints < 1:
            raise ValueError("mf_max_constraints must be >= 1.")

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
        self._max_constraints_padded = ((max_constraints + 2) // 3) * 3
        self.mf_max_constraints = mf_max_constraints
        self._double_buffer = True
        self._nvtx = False
        if not model.device.is_cuda:
            raise ValueError("SolverFeatherPGS requires a CUDA device.")
        if not enable_contact_friction:
            raise ValueError("SolverFeatherPGS currently requires enable_contact_friction=True.")
        if model.body_count:
            body_flags = model.body_flags.numpy()
            if np.any((body_flags & int(BodyFlags.KINEMATIC)) != 0):
                raise NotImplementedError("SolverFeatherPGS does not support kinematic bodies yet.")

        # Effort-limit clamp is always actuator-only: the explicit-PD drive bucket
        # (``aug_row_u0``) is clamped to ``+/- joint_effort_limit`` before it
        # is summed into ``joint_tau``. Matches MuJoCo's ``actuatorfrcrange``
        # and PhysX articulation drive ``maxForce`` conventions.
        self.use_parallel_streams = True

        self._step = 0
        self._force_mass_update = False
        self._last_step_dt = None

        self._compute_articulation_metadata(model)

        self._allocate_common_buffers(model)
        self._allocate_buffers(model)
        self._allocate_world_buffers(model)
        self._allocate_mf_buffers(model)
        self._scatter_armature_to_groups(model)
        self._init_size_group_streams(model)
        self._dummy_is_free_rigid = wp.zeros((1,), dtype=wp.int32, device=model.device)
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
        self._validate_supported_free_joint_topology(model)
        self._compute_root_free_metadata(model)
        self._setup_size_grouping(model)
        self._setup_world_mapping(model)
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

    def _validate_supported_free_joint_topology(self, model):
        if not model.articulation_count or not model.joint_count:
            return

        articulation_start = model.articulation_start.numpy()
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        root_joints = {int(j) for j in articulation_start[:-1]}

        for joint in range(model.joint_count):
            jt = int(joint_type[joint])
            if jt != int(JointType.FREE) and jt != int(JointType.DISTANCE):
                continue
            if joint not in root_joints or int(joint_parent[joint]) != -1:
                raise NotImplementedError(
                    "SolverFeatherPGS only supports FREE and DISTANCE joints when they are root joints "
                    f"attached to the world; joint {joint} is not a supported floating root."
                )

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
        self.is_free_rigid = wp.array(is_free_rigid_np, dtype=wp.int32, device=model.device)

    def _compute_world_dof_mapping(self, model):
        """Compute per-world DOF metadata for consolidated J/Y and velocity arrays."""
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
        self.world_dof_count = wp.array(world_dof_counts, dtype=wp.int32, device=model.device)

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
        else:
            self.M_blocks = None
            self.mass_update_mask = None
            self.v_hat = None
            self.v_out = None
            self.qd_work = None

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
        self._contact_metadata_capacity = max_contacts
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

        # Joint velocity-limit buffers (per-DOF tracking). Velocity-limit
        # rows are always enabled for finite positive model limits.
        if model.joint_dof_count > 0:
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
        max_constraints = self.max_constraints
        max_constraints_padded = self._max_constraints_padded

        # Matrix-free uses world-indexed J/Y for both articulated and free-rigid rows.
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
        self.world_velocity = wp.zeros(
            (self.world_count, self.max_world_dofs),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )

        self.rhs = wp.zeros(
            (self.world_count, max_constraints_padded), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        # The fused-Warp path stores contact triplets as vec3.
        self.impulses = wp.zeros(
            (self.world_count, max_constraints_padded),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        self._max_contact_triplets = max_constraints_padded // 3
        self.impulses_vec3 = self.impulses.reshape((self.world_count, self._max_contact_triplets, 3)).view(wp.vec3)
        self.diag_contact_vec3 = wp.zeros((self.world_count, self._max_contact_triplets), dtype=wp.vec3, device=device)
        self.rhs_contact_vec3 = wp.zeros((self.world_count, self._max_contact_triplets), dtype=wp.vec3, device=device)
        self.diag = wp.zeros(
            (self.world_count, max_constraints_padded), dtype=wp.float32, device=device, requires_grad=requires_grad
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
        self.mf_meta = wp.zeros((worlds, mf_max_c), dtype=wp.vec4i, device=device)

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
    def notify_model_changed(self, flags: int) -> None:
        if flags:
            raise NotImplementedError("SolverFeatherPGS does not support model mutations; recreate the solver.")

    def _validate_contact_metadata_capacity(self, contacts: Contacts | None, caller: str) -> None:
        if contacts is None:
            return

        contact_capacity = int(contacts.rigid_contact_max)
        if contact_capacity > self._contact_metadata_capacity:
            raise ValueError(
                f"SolverFeatherPGS.{caller}() received Contacts with rigid_contact_max={contact_capacity}, "
                f"but its contact metadata was allocated for {self._contact_metadata_capacity}. "
                "Construct or configure the CollisionPipeline before creating SolverFeatherPGS, "
                "or recreate the solver after increasing model.rigid_contact_max."
            )

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ):
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("SolverFeatherPGS.step() requires a finite dt > 0.")
        self._validate_contact_metadata_capacity(contacts, "step")

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

        if state_out.body_parent_f is not None:
            state_out.body_parent_f.zero_()

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
                self._stage1_drives(state_in, state_aug, state_out, control, dt)

            self._stage1_crba(state_aug)
        # ══════════════════════════════════════════════════════════════
        # STAGE 2: Cholesky
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S2_Cholesky", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                with ctx:
                    self._stage2_cholesky_tiled(size)
        # ══════════════════════════════════════════════════════════════
        # STAGE 3: Triangular solve + v_hat
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S3_Trisolve_Vhat", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            self._stage3_zero_qdd(state_aug)
            for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                with ctx:
                    self._stage3_trisolve_tiled(size, state_aug)
            self._stage3_compute_v_hat(state_in, state_aug, dt)

        # ══════════════════════════════════════════════════════════════
        # STAGE 4: Build contact problem
        # ══════════════════════════════════════════════════════════════
        with wp.ScopedTimer("S4_ContactBuild", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            self._stage4_build_rows(state_in, state_aug, contacts)

        # Compute Y = H^-1 * J^T only (no dense Delassus matrix).
        with wp.ScopedTimer("S4_HinvJt_Diag_RHS", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                with ctx:
                    self._stage4_hinv_jt_tiled(size)

            self.diag.zero_()
            for size in self.size_groups:
                self._stage4_diag_from_JY(size)
            self._stage4_finalize_world_diag_cfm()
            self._stage4_add_dense_contact_compliance(dt)
            self._stage4_compute_rhs_world(dt)

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

        with wp.ScopedTimer("S6_PGS_Solve", print=False, use_nvtx=self._nvtx, synchronize=self._nvtx):
            fused_kernel = TiledKernelFactory.get_pgs_fused_warp_kernel(
                self.max_constraints,
                self._max_contact_triplets,
                self.mf_max_constraints,
                self.max_world_dofs,
                self.model.device,
            )
            wp.launch_tiled(
                fused_kernel,
                dim=[self.world_count],
                inputs=[
                    self.constraint_count,
                    self.dense_contact_row_count,
                    self.rhs,
                    self.rhs_contact_vec3,
                    self.diag,
                    self.diag_contact_vec3,
                    self.impulses_vec3,
                    self.impulses,
                    self.J_world,
                    self.Y_world,
                    self.row_mu,
                    self.mf_constraint_count,
                    self.mf_meta,
                    self.mf_impulses,
                    self.mf_J_a,
                    self.mf_J_b,
                    self.mf_MiJt_a,
                    self.mf_MiJt_b,
                    self.mf_row_mu,
                    self.pgs_iterations,
                    self.pgs_omega,
                ],
                outputs=[self.world_velocity],
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
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Populate Newton contact-force buffers from the last FeatherPGS solve."""
        if contacts is None or contacts.rigid_contact_count is None:
            return
        self._validate_contact_metadata_capacity(contacts, "update_contacts")

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
                # used by the triangular-solve gradient path
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

    def _stage1_drives(self, state_in: State, state_aug: State, state_out: State, control: Control, dt: float):
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

            if state_out.body_parent_f is not None and model.body_count:
                wp.launch(
                    compute_body_parent_f,
                    dim=model.body_count,
                    inputs=[
                        state_aug.body_q_com,
                        state_aug.body_f_s,
                        state_aug.body_ft_s,
                        body_f,
                        self.body_to_articulation,
                        self.articulation_origin,
                    ],
                    outputs=[state_out.body_parent_f],
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

    def _stage3_compute_v_hat(self, state_in: State, state_aug: State, dt: float):
        model = self.model
        if not model.joint_count:
            return
        wp.launch(
            compute_velocity_predictor_and_seed_world_velocity,
            dim=(self.world_count, self.max_world_dofs),
            inputs=[
                self.qd_work,
                state_aug.joint_qdd,
                self.world_dof_start,
                self.world_dof_count,
                dt,
            ],
            outputs=[self.v_hat, self.world_velocity],
            device=model.device,
        )

    def _stage4_build_rows(self, state_in: State, state_aug: State, contacts: Contacts):
        model = self.model
        max_constraints = self.max_constraints
        mf_active = self._has_free_rigid_bodies

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
        is_free_rigid = self.is_free_rigid if self.is_free_rigid is not None else self._dummy_is_free_rigid
        mf_slot_counter = self.mf_slot_counter

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
            # velocity-limit rows are appended, clamped through the same
            # capacity path as the final world constraint count.
            slots_per_contact_dense = 3 if self.enable_contact_friction else 1
            wp.launch(
                finalize_world_constraint_counts,
                dim=self.world_count,
                inputs=[self.slot_counter, max_constraints, slots_per_contact_dense],
                outputs=[self.dense_contact_row_count],
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

            # Allocate + populate joint velocity-limit rows (per-DOF clamp on
            # |qdot_i| against model.joint_velocity_limit). Launched *after*
            # joint-position-limit allocation so velocity-limit rows occupy
            # the last per-world slots — matching PhysX's documented
            # ordering where the vel-limit row fires after contact, drive,
            # friction, and positional-limit rows (physx-deep-dive §7).
            if self.velocity_limit_slot is not None:
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

        # Joint velocity-limit rows also need to be built when the contact
        # block did not run.
        if self.velocity_limit_slot is not None:
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
        if self.contact_compliance <= 0.0:
            return

        contact_alpha = float(self.contact_compliance / (dt * dt))
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

    def _stage5_prepare_impulses_world(self):
        wp.launch(
            prepare_world_impulses,
            dim=self.world_count,
            inputs=[self._max_constraints_padded],
            outputs=[self.impulses],
            device=self.model.device,
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
                scatter_world_velocity,
                dim=(self.world_count, self.max_world_dofs),
                inputs=[
                    self.world_velocity,
                    self.world_dof_start,
                    self.world_dof_count,
                ],
                outputs=[self.v_out],
                device=model.device,
            )
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
        else:
            wp.launch(
                update_qdd_from_world_velocity,
                dim=(self.world_count, self.max_world_dofs),
                inputs=[
                    state_in.joint_qd,
                    self.world_velocity,
                    self.world_dof_start,
                    self.world_dof_count,
                    1.0 / dt,
                ],
                outputs=[self.v_out, state_aug.joint_qdd],
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
