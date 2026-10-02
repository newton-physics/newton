# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent primal problem for the LOX rigid-body backend.

The problem owns persistent rows, capacity mappings, and the smooth body system;
its kernels live in :mod:`.problem_kernels`. Construction and the one-time joint
penalty-scale seeding may read arrays on the host; :meth:`LOXProblem.begin_time_step`
and :meth:`LOXProblem.build_smooth_system` use only fixed-size device operations so
they remain suitable for graph capture. Kamino contact vectors remain normal-last
throughout this boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import warp as wp

from ......sim import BodyFlags, JointType
from ...core.bodies import update_body_inertias
from ...core.data import DataKamino
from ...core.joints import DofActuationPath, JointActuationType, JointCorrectionMode
from ...core.model import ModelKamino
from ...core.types import to_warp_int32_array, vec6f
from ...geometry.contacts import ContactsKamino
from ...kinematics.constraints import update_constraints_info
from ...kinematics.jacobians import SparseSystemJacobians
from ...kinematics.joints import compute_joints_data
from ...kinematics.limits import LimitsKamino
from .problem_kernels import (
    _accumulate_aligned_joint_wrenches,
    _compute_structural_effective_mass,
    _count_box_rows,
    _enable_bodies_in_weighted_blocks,
    _finish_box_rows,
    _finish_contacts,
    _load_contacts,
    _load_dynamic_rows,
    _load_joint_frictions,
    _load_limits,
    _load_structural_rows,
    _mark_blocks_with_unilaterals,
    _reset_angular_reactions,
    _reset_effort_rows,
    _reset_effort_worlds,
    _reset_row_reactions,
    _scatter_structural_impulses,
    _update_effort_counters,
    _write_contact_outputs,
    _write_dynamic_outputs,
    _write_friction_outputs,
    _write_limit_outputs,
    make_update_structural_multipliers_kernel,
)
from .system import LOXSystem
from .types import (
    BoxRows,
    ContactRows,
    DynamicRows,
    EffortRows,
    LOXProblemData,
    SpatialContactRows,
    StructuralRows,
    capacity_offsets,
    segment_local_indices,
)

###
# Module interface
###

__all__ = ["LOXProblem"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_LOW_MODE_PERCENTILE = 0.02
"""Discrete percentile of the positive structural eigenvalues used as the low mode."""


###
# Functions
###


def _low_mode_eigenvalue(eigenvalues: np.ndarray) -> float:
    """Return the discrete low-mode percentile of sorted positive eigenvalues."""
    return float(eigenvalues[int(_LOW_MODE_PERCENTILE * eigenvalues.size)])


def _connected_component_labels(
    count: int,
    bid_a: np.ndarray,
    bid_b: np.ndarray,
) -> np.ndarray:
    """Compute minimum-vertex component labels with vectorized label propagation."""
    labels = np.arange(count, dtype=np.int32)
    if bid_a.size == 0:
        return labels
    while True:
        endpoint_labels = np.minimum(labels[bid_a], labels[bid_b])
        updated = labels.copy()
        np.minimum.at(updated, bid_a, endpoint_labels)
        np.minimum.at(updated, bid_b, endpoint_labels)
        updated = updated[updated]
        if np.array_equal(updated, labels):
            return labels
        labels = updated


def _body_row_incidence(
    body_a: np.ndarray, body_b: np.ndarray, body_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group the rows touching each body, ordered by body, row, then side.

    Args:
        body_a: Body A of each row, or ``-1``.
        body_b: Body B of each row, or ``-1``.
        body_count: Number of bodies.

    Returns:
        The first incidence of each body with the total appended, and the row
        and side (0 for A, 1 for B) of each incidence.
    """
    rows = np.arange(body_a.size, dtype=np.int32)
    incidence_body = np.concatenate((body_a[body_a >= 0], body_b[body_b >= 0])).astype(np.int32, copy=False)
    incidence_row = np.concatenate((rows[body_a >= 0], rows[body_b >= 0]))
    incidence_side = np.concatenate(
        (
            np.zeros(np.count_nonzero(body_a >= 0), dtype=np.int32),
            np.ones(np.count_nonzero(body_b >= 0), dtype=np.int32),
        )
    )
    order = np.lexsort((incidence_side, incidence_row, incidence_body))
    body_counts = np.bincount(incidence_body, minlength=body_count).astype(np.int32, copy=False)
    return capacity_offsets(body_counts), incidence_row[order], incidence_side[order]


###
# Interfaces
###


class LOXProblem:
    """Own rigid-body topology, constraint rows, and the smooth body system."""

    def __init__(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: SparseSystemJacobians,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        eliminate_fixed_world_islands: bool = True,
        selective_weights: bool = True,
        rotation_correction: JointCorrectionMode = JointCorrectionMode.TWOPI,
        joint_proximal_relaxation: float = 0.0,
        contact_compliance: bool = False,
        contact_restitution: bool = False,
        contact_spatial_friction: bool = False,
    ):
        self.model = model
        self.kamino_data = data
        """Kamino data of the model."""
        self.jacobians = jacobians
        self.limits = limits
        self.contacts = contacts
        self.device = wp.get_device(model.device)
        self.num_worlds = model.info.num_worlds
        self.eliminate_fixed_world_islands = eliminate_fixed_world_islands
        self.selective_weights = selective_weights
        self.rotation_correction = rotation_correction
        self.joint_proximal_relaxation = joint_proximal_relaxation
        self._joint_max_correction = math.inf
        self.contact_compliance = contact_compliance
        self.contact_restitution = contact_restitution
        self.contact_spatial_friction = contact_spatial_friction

        if jacobians._J_cts is None or jacobians._J_dofs is None:
            raise RuntimeError("Sparse Jacobians must be finalized before constructing the LOX problem.")
        self._sparse_jacobian_data = jacobians._J_cts.bsm.nzb_values
        self._sparse_dof_jacobian_data = jacobians._J_dofs.bsm.nzb_values
        self._sparse_limit_offsets = jacobians.limit_constraint_nzb_offsets

        self._data = LOXProblemData(
            self.num_worlds,
            model.size.sum_of_num_bodies,
            model.info.bodies_offset,
            model.info.num_bodies,
            self.device,
        )
        self.rebuild_dynamic_body_topology()

    ###
    # Properties
    ###

    @property
    def data(self) -> LOXProblemData:
        """Returns the constraint rows and per-step body data."""
        return self._data

    @property
    def body_velocity_begin(self) -> wp.array[vec6f]:
        """Begin-step body twists [m/s, rad/s].

        Kamino's body velocities, which the solve leaves unchanged until the
        integrator inputs are written.
        """
        return self.kamino_data.bodies.u_i.view(dtype=vec6f)

    ###
    # Public API
    ###

    def dynamic_body_topology_changed(self) -> bool:
        """Return whether current body flags require a different primal topology."""
        return self._classify_dynamic_bodies() != self.system.dynamic_bodies

    def rebuild_dynamic_body_topology(self) -> None:
        """Reallocate the body system and the rows after the set of dynamic bodies changed."""
        model = self.model
        body_counts = tuple(model.info.num_bodies.numpy().astype(np.int32, copy=False).tolist())
        dynamic_bodies = self._classify_dynamic_bodies()
        body_edges = self._build_body_edges(dynamic_bodies)
        body_components = self._build_body_components(dynamic_bodies, body_edges)
        self.system = LOXSystem(
            body_counts,
            model.bodies.wid,
            body_components=body_components,
            body_edges=body_edges,
            device=self.device,
        )

        dynamic_world, dynamic_dof = self._allocate_joint_rows()
        self.system.validate_body_pairs("dynamic rows", self._data.dynamic_rows.body_a, self._data.dynamic_rows.body_b)
        self.system.validate_body_pairs(
            "structural rows", self._data.structural_rows.body_a, self._data.structural_rows.body_b
        )
        self._allocate_effort_rows(dynamic_world, dynamic_dof)
        self._allocate_unilaterals(*self._allocate_joint_frictions())

    def validate_model_changed(self) -> None:
        """Validate LOX constraint topology derived from aliased model values."""
        source_model = self.model._model
        if source_model is None:
            return

        if source_model.joint_friction is not None:
            friction = source_model.joint_friction.numpy()
            if np.any(~np.isfinite(friction) | (friction < 0.0)):
                raise ValueError("Joint friction values must be finite and nonnegative.")
        effort_limit = source_model.joint_effort_limit.numpy()
        if np.any(np.isnan(effort_limit) | (effort_limit < 0.0)):
            raise ValueError("Joint effort limits must be nonnegative or positive infinity.")

    def prepare_joint_penalty_scale_seed(self, time_step: float) -> None:
        """Prepare the initial rigid state for LOX penalty-scale seeding."""
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("The joint penalty seeding time step must be finite and positive.")
        if np.any(self.kamino_data.time.steps.numpy() != 0):
            raise RuntimeError("joint penalty scale seeding must run before the first step or after a solver reset.")

        self.model.time.set_uniform_timestep(time_step)
        compute_joints_data(
            model=self.model,
            data=self.kamino_data,
            q_j_p=self.kamino_data.joints.q_j_p,
            correction=self.rotation_correction,
        )
        update_body_inertias(model=self.model.bodies, data=self.kamino_data.bodies)
        if self.limits is not None:
            self.limits.detect(q_j=self.kamino_data.joints.q_j)
        update_constraints_info(model=self.model, data=self.kamino_data)
        self.jacobians.build(
            model=self.model,
            data=self.kamino_data,
            limits=self.limits,
            contacts=None,
            reset_to_zero=True,
        )

    def estimate_joint_penalty_scale(self, default_scale: np.ndarray) -> np.ndarray:
        """Return the per-world reciprocal of the low-mode structural eigenvalue.

        For each body component, the structural Delassus ``D = J A^+ J^T`` uses
        the pseudo-inverse of the assembled smooth matrix ``A`` and is normalized by
        the structural effective masses as ``M^1/2 D M^1/2``. Massless bodies leave
        directions without inertia in ``A``, which the pseudo-inverse excludes from
        the estimate. The positive eigenvalues of all
        components of a world are pooled, and the discrete second-percentile value
        gives the world's scale; worlds without structural rows keep
        ``default_scale``.

        The smooth system and the structural rows must be assembled, e.g. after
        :meth:`prepare_joint_penalty_scale_seed` and :meth:`build_smooth_system`. This
        is a one-time initialization, so the Delassus blocks are computed on the host.

        Args:
            default_scale: Scale of each world without structural rows.

        Returns:
            The float32 penalty scale of each world.
        """
        system = self.system
        structural_rows = self._data.structural_rows
        smooth_matrix = system.data.matrix.numpy()
        matrix_offsets = system.data.info.mio.numpy()
        body_block = np.asarray(system.body_block_host, dtype=np.int32)
        body_local = np.asarray(system.body_local_host, dtype=np.int32)
        body_a = structural_rows.body_a.numpy()
        body_b = structural_rows.body_b.numpy()
        jacobian_a = structural_rows.jacobian_a.numpy().astype(np.float64)
        jacobian_b = structural_rows.jacobian_b.numpy().astype(np.float64)
        effective_mass = structural_rows.effective_mass.numpy().astype(np.float64)

        # Rows attach to the component of their dynamic endpoints, or to -1 between
        # prescribed bodies.
        block_a = np.where(body_a >= 0, body_block[body_a.clip(min=0)], -1)
        block_b = np.where(body_b >= 0, body_block[body_b.clip(min=0)], -1)
        row_block = np.where(block_a >= 0, block_a, block_b)

        # The Delassus is evaluated in float64, but its eigenvalues are only as
        # resolved as the float32 system.
        epsilon = np.finfo(np.float32).eps
        positive_by_world: list[list[np.ndarray]] = [[] for _ in range(len(default_scale))]
        for block in np.unique(row_block[row_block >= 0]):
            rows = np.flatnonzero(row_block == block)
            dimension = 6 * system.block_body_counts[block]
            offset = int(matrix_offsets[block])
            matrix = smooth_matrix[offset : offset + dimension * dimension].reshape(dimension, dimension)

            jacobian = np.zeros((rows.size, dimension))
            for row_body, row_block_side, row_jacobian in (
                (body_a, block_a, jacobian_a),
                (body_b, block_b, jacobian_b),
            ):
                for local_row, row in enumerate(rows):
                    if row_block_side[row] == block:
                        column = 6 * body_local[row_body[row]]
                        jacobian[local_row, column : column + 6] = row_jacobian[row]

            # Massless bodies leave exact null directions in the smooth matrix
            matrix_eigenvalues, matrix_eigenvectors = np.linalg.eigh(matrix.astype(np.float64))
            resolved = matrix_eigenvalues > np.finfo(np.float64).eps * dimension * max(matrix_eigenvalues[-1], 0.0)
            projected = jacobian @ matrix_eigenvectors[:, resolved]
            delassus = (projected / matrix_eigenvalues[resolved]) @ projected.T
            metric_sqrt = np.sqrt(effective_mass[rows])
            eigenvalues = np.linalg.eigvalsh(metric_sqrt[:, None] * delassus * metric_sqrt[None, :])
            positive = eigenvalues[eigenvalues > epsilon * rows.size * float(eigenvalues[-1])]
            world = system.block_world_host[block]
            if positive.size == 0:
                raise RuntimeError(
                    f"World {world} component {block} structural Delassus has no resolvable positive eigenvalue."
                )
            positive_by_world[world].append(positive)

        scales = np.asarray(default_scale, dtype=np.float64).copy()
        for world, component_eigenvalues in enumerate(positive_by_world):
            if not component_eigenvalues:
                continue
            positive = np.sort(np.concatenate(component_eigenvalues))
            scale = 1.0 / _low_mode_eigenvalue(positive)
            if not math.isfinite(scale) or scale > np.finfo(np.float32).max:
                raise RuntimeError(f"World {world} structural ALM penalty seed is not representable in float32.")
            scales[world] = scale
        return scales.astype(np.float32)

    def begin_time_step(
        self,
        time_step: wp.array[wp.float32],
        *,
        joint_max_correction: float,
        limit_stabilization_fraction: float,
        contact_stabilization_fraction: float,
        contact_dead_zone: float,
        structural_warmstart_factor: float,
    ) -> None:
        """Load the constraint rows of the time step and import the reactions of the previous time step once.

        Args:
            time_step: Time step of each world [s].
            joint_max_correction: Largest joint residual corrected in one step [m or rad].
            limit_stabilization_fraction: Fraction of the limit violation corrected in one step.
            contact_stabilization_fraction: Fraction of the contact penetration corrected in one step.
            contact_dead_zone: Symmetric contact distance dead zone of the stabilization [m].
            structural_warmstart_factor: Scale applied to the structural multipliers of the
                previous time step.
        """
        self._joint_max_correction = joint_max_correction
        if self._data.structural_rows.count > 0:
            if self._data.structural_rows.proximal_defect is not None:
                self._data.structural_rows.proximal_defect.zero_()
            wp.launch(
                _load_structural_rows,
                dim=self._data.structural_rows.count,
                inputs=[
                    self._data.structural_rows.sparse_a_index,
                    self._data.structural_rows.sparse_b_index,
                    self._sparse_jacobian_data,
                    structural_warmstart_factor,
                ],
                outputs=[
                    self._data.structural_rows.jacobian_a,
                    self._data.structural_rows.jacobian_b,
                    self._data.structural_rows.reaction,
                ],
                device=self.device,
            )
        self._update_unilaterals(
            time_step,
            limit_stabilization_fraction,
            contact_stabilization_fraction,
            contact_dead_zone,
        )
        self._select_weighted_bodies()

    def build_smooth_system(
        self,
        time_step: wp.array[wp.float32],
        joint_penalty_scale: wp.array[wp.float32] | None,
    ) -> None:
        """Assemble the frozen smooth primal system of the time step.

        :meth:`begin_time_step` must be called first. The matrix and right-hand side
        stay fixed during the iterations; the multiplier and effort-limit impulses are
        added to the candidate right-hand side of each iteration.

        Args:
            time_step: Time step of each world [s].
            joint_penalty_scale: Dimensionless structural ALM penalty scale of each world, or ``None``
                to leave the structural rows out of the smooth system.
        """
        data = self.kamino_data
        self.system.assemble_bodies(
            self.model.bodies.m_i,
            data.bodies.I_i,
            self.body_velocity_begin,
            data.bodies.w_e_i.view(dtype=vec6f),
            data.bodies.w_a_i.view(dtype=vec6f),
            self.model.gravity.vector,
            time_step,
        )
        if self._data.dynamic_rows.count > 0:
            wp.launch(
                _load_dynamic_rows,
                dim=self._data.dynamic_rows.count,
                inputs=[
                    self._data.dynamic_rows.world,
                    self._data.dynamic_rows.uses_dof_jacobian,
                    self._data.dynamic_rows.value_index,
                    self._data.dynamic_rows.dof_index,
                    self._data.dynamic_rows.sparse_a_index,
                    self._data.dynamic_rows.sparse_b_index,
                    self._sparse_jacobian_data,
                    self._sparse_dof_jacobian_data,
                    data.joints.m_j,
                    data.joints.dq_b_j,
                    self._data.dynamic_rows.effort_index,
                    self._data.effort_rows.value_index,
                    data.joints.inv_m_a,
                    data.joints.dq_b_a,
                    self.model.joints.a_j,
                    self.model.joints.k_p_j,
                    self.model.joints.dof_act_types,
                    data.joints.dq_j,
                    data.joints.tau_j,
                    self.model.joints.k_d_j,
                    self.model.joints.tau_j_max,
                    time_step,
                ],
                outputs=[
                    self._data.dynamic_rows.jacobian_a,
                    self._data.dynamic_rows.jacobian_b,
                    self._data.dynamic_rows.effective_inertia,
                    self._data.dynamic_rows.free_velocity,
                    self._data.effort_rows.intercept,
                    self._data.effort_rows.slope,
                    self._data.effort_rows.impulse_bound,
                ],
                device=self.device,
            )
            self.system.add_dynamic_rows(
                self._data.dynamic_rows.world,
                self._data.dynamic_rows.body_a,
                self._data.dynamic_rows.body_b,
                self._data.dynamic_rows.jacobian_a,
                self._data.dynamic_rows.jacobian_b,
                self._data.dynamic_rows.effective_inertia,
                self._data.dynamic_rows.free_velocity,
                prescribed_twist=self.body_velocity_begin,
            )
        if self._data.structural_rows.count > 0:
            wp.launch(
                _compute_structural_effective_mass,
                dim=self.model.size.sum_of_num_joints,
                inputs=[
                    self.model.joints.bid_B,
                    self.model.joints.bid_F,
                    self.model.joints.kinematic_cts_offset,
                    self.model.joints.num_kinematic_cts,
                    self._data.dynamic_rows.joint_offset,
                    self._data.dynamic_rows.joint_count,
                    self._data.structural_rows.jacobian_a,
                    self._data.structural_rows.jacobian_b,
                    self._data.dynamic_rows.jacobian_a,
                    self._data.dynamic_rows.jacobian_b,
                    self._data.dynamic_rows.effective_inertia,
                    self.model.bodies.inv_m_i,
                    data.bodies.inv_I_i,
                ],
                outputs=[self._data.structural_rows.effective_mass],
                device=self.device,
            )
            if joint_penalty_scale is not None:
                self.system.add_structural_rows(
                    self._data.structural_rows.world,
                    self._data.structural_rows.body_a,
                    self._data.structural_rows.body_b,
                    self._data.structural_rows.jacobian_a,
                    self._data.structural_rows.jacobian_b,
                    self._data.structural_rows.residual,
                    self._data.structural_rows.effective_mass,
                    time_step,
                    joint_penalty_scale,
                    self._data.structural_rows.penalty,
                    prescribed_twist=self.body_velocity_begin,
                    joint_max_correction=self._joint_max_correction,
                )
            rows = self._data.structural_rows
            rows.body_impulse.zero_()
            wp.launch(
                _scatter_structural_impulses,
                dim=rows.count,
                inputs=[
                    time_step,
                    rows.world,
                    rows.body_a,
                    rows.body_b,
                    rows.jacobian_a,
                    rows.jacobian_b,
                    rows.penalty,
                    rows.reaction,
                ],
                outputs=[rows.body_impulse],
                device=self.device,
            )

    def build_consensus_weights(self, sigma: float, beta: float) -> None:
        """Compute the frozen consensus weights ``W`` of the time step and assemble the weighted matrix ``A + W``.

        Args:
            sigma: Relative lower scale of the inertia-normalized body-weight clamp.
            beta: Normalized smooth-weight transition threshold of the body-weight heuristic.
        """
        self.system.build_weighted_matrix(self._data.body_weight_enabled, sigma=sigma, beta=beta)

    def update_structural_multipliers_from_twist(
        self,
        time_step: wp.array[wp.float32],
        structural_tolerance: float,
        global_twist: wp.array[vec6f],
        projected_twist: wp.array[vec6f],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        projected_fraction: float,
    ) -> None:
        """Update the structural multipliers from the candidate twist of each joint, in one pass over the joints.

        The per-world structural residual is accumulated into ``structural_rows.world_residual``,
        which the convergence check reads and clears.
        """
        if self._data.structural_rows.count == 0:
            return
        joints = self.model.joints
        wp.launch(
            make_update_structural_multipliers_kernel(self.rotation_correction, self.joint_proximal_relaxation > 0.0),
            dim=self.model.size.sum_of_num_joints,
            inputs=[
                time_step,
                structural_tolerance,
                joints.wid,
                joints.dof_type,
                joints.coords_offset,
                joints.dofs_offset,
                joints.kinematic_cts_offset,
                joints.num_kinematic_cts,
                joints.bid_B,
                joints.bid_F,
                joints.B_r_Bj,
                joints.F_r_Fj,
                joints.X_Bj,
                joints.X_Fj,
                self.kamino_data.bodies.q_i,
                self.kamino_data.joints.q_j_p,
                world_active,
                world_failed,
                self.system.data.body_vector_index,
                global_twist,
                projected_twist,
                projected_fraction,
                self._data.structural_rows.jacobian_a,
                self._data.structural_rows.jacobian_b,
                self._data.structural_rows.residual,
                self._data.structural_rows.penalty,
                self.joint_proximal_relaxation,
                self._joint_max_correction,
            ],
            outputs=[
                self._data.structural_rows.proximal_defect,
                self._data.structural_rows.candidate_residual,
                self._data.structural_rows.residual_velocity_scratch,
                self._data.joint_coordinate_scratch,
                self._data.joint_velocity_scratch,
                self._data.structural_rows.reaction,
                self._data.structural_rows.body_impulse,
                self._data.structural_rows.world_residual,
            ],
            device=self.device,
        )

    def update_effort_counters(
        self,
        time_step: wp.array[wp.float32],
        velocity_tolerance: float,
        projected_twist: wp.array[vec6f],
        world_active: wp.array[wp.bool],
    ) -> None:
        """Clamp the bounded drives at the projected twist and reduce their effort residuals.

        The convergence check consumes and clears the effort residuals.

        The counter-impulses enter the right-hand side of the next candidate solve and
        the reported efforts.
        """
        if not self.has_bounded_effort:
            return
        wp.launch(
            _update_effort_counters,
            dim=self._data.effort_rows.capacity,
            inputs=[
                self._data.effort_rows.world,
                self._data.effort_rows.dynamic_row_index,
                world_active,
                self._data.dynamic_rows.body_a,
                self._data.dynamic_rows.body_b,
                self._data.dynamic_rows.jacobian_a,
                self._data.dynamic_rows.jacobian_b,
                self._data.dynamic_rows.effective_inertia,
                projected_twist,
                self._data.effort_rows.intercept,
                self._data.effort_rows.slope,
                self._data.effort_rows.impulse_bound,
                time_step,
                velocity_tolerance,
            ],
            outputs=[
                self._data.effort_rows.counter,
                self._data.effort_rows.net_applied,
                self._data.effort_rows.world_residual_max,
            ],
            device=self.device,
        )

    def write_output_wrenches(
        self,
        inverse_time_step: wp.array[wp.float32],
        body_velocity: wp.array[vec6f],
        world_failed: wp.array[wp.bool],
    ) -> None:
        """Write force-valued constraint reactions to the Kamino body wrenches, zero for the failed worlds."""
        body_data = self.kamino_data.bodies
        joint_wrench = body_data.w_j_i.view(dtype=vec6f)
        limit_wrench = body_data.w_l_i.view(dtype=vec6f)
        contact_wrench = body_data.w_c_i.view(dtype=vec6f)
        body_data.w_j_i.zero_()
        body_data.w_l_i.zero_()
        body_data.w_c_i.zero_()
        if self._data.dynamic_rows.count > 0:
            wp.launch(
                _write_dynamic_outputs,
                dim=self._data.dynamic_rows.count,
                inputs=[
                    inverse_time_step,
                    self._data.dynamic_rows.world,
                    self._data.dynamic_rows.multiplier_index,
                    self._data.dynamic_rows.effort_index,
                    self._data.dynamic_rows.body_a,
                    self._data.dynamic_rows.body_b,
                    self._data.dynamic_rows.jacobian_a,
                    self._data.dynamic_rows.jacobian_b,
                    self._data.dynamic_rows.effective_inertia,
                    self._data.dynamic_rows.free_velocity,
                    self._data.effort_rows.counter,
                    self._data.effort_rows.net_applied,
                    self._data.effort_rows.value_index,
                    body_velocity,
                ],
                outputs=[self.kamino_data.joints.lambda_dyn_j, self.kamino_data.joints.lambda_tau_j, joint_wrench],
                device=self.device,
            )
        if self._data.box_rows.friction_capacity > 0:
            wp.launch(
                _write_friction_outputs,
                dim=self._data.box_rows.friction_capacity,
                inputs=[
                    inverse_time_step,
                    self._data.box_rows.friction_row,
                    self._data.friction_multiplier_index,
                    self._data.box_rows.world,
                    self._data.box_rows.body_a,
                    self._data.box_rows.body_b,
                    self._data.box_rows.jacobian_a,
                    self._data.box_rows.jacobian_b,
                    self._data.box_rows.reaction,
                    world_failed,
                ],
                outputs=[self.kamino_data.joints.lambda_f_j, joint_wrench],
                device=self.device,
            )
        if self._data.box_rows.limit_capacity > 0 and self.limits is not None:
            wp.launch(
                _write_limit_outputs,
                dim=self.limits.model_max_limits_host,
                inputs=[
                    self.limits.model_active_limits,
                    self.limits.model_max_limits_host,
                    self.limits.wid,
                    self.limits.lid,
                    self._data.box_rows.world_limit_capacity,
                    self._data.box_rows.world_limit_offset,
                    inverse_time_step,
                    self._data.box_rows.body_a,
                    self._data.box_rows.body_b,
                    self._data.box_rows.jacobian_a,
                    self._data.box_rows.jacobian_b,
                    self._data.box_rows.reaction,
                    self._data.box_rows.velocity,
                    world_failed,
                ],
                outputs=[self.limits.reaction, self.limits.velocity, limit_wrench],
                device=self.device,
            )
        if self._data.contact_rows.capacity > 0 and self.contacts is not None:
            wp.launch(
                _write_contact_outputs,
                dim=self.contacts.model_max_contacts_host,
                inputs=[
                    self.contacts.model_active_contacts,
                    self.contacts.model_max_contacts_host,
                    self.contacts.wid,
                    self._data.contact_rows.source_to_internal,
                    inverse_time_step,
                    self._data.contact_rows.body_a,
                    self._data.contact_rows.body_b,
                    self._data.contact_rows.jacobian_a,
                    self._data.contact_rows.jacobian_b,
                    self._data.contact_rows.reaction,
                    self._data.contact_rows.velocity,
                    self._data.contact_rows.frame,
                    self._data.contact_rows.angular_reaction,
                    world_failed,
                ],
                outputs=[self.contacts.reaction, self.contacts.velocity, self.contacts.mode, contact_wrench],
                device=self.device,
            )
            if self._data.contact_rows.angular_warmstarter is not None:
                self._data.contact_rows.angular_warmstarter.update(
                    self.contacts,
                    self._data.contact_rows.source_to_internal,
                    inverse_time_step,
                    self._data.contact_rows.angular_reaction,
                )
        if self._data.structural_rows.count > 0:
            wp.launch(
                _accumulate_aligned_joint_wrenches,
                dim=self._data.structural_rows.count,
                inputs=[
                    self._data.structural_rows.body_a,
                    self._data.structural_rows.body_b,
                    self._data.structural_rows.jacobian_a,
                    self._data.structural_rows.jacobian_b,
                    self._data.structural_rows.reaction,
                ],
                outputs=[joint_wrench],
                device=self.device,
            )

    def reset_structural_multipliers(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear the structural multiplier warm starts of the masked worlds, or of all worlds."""
        self._reset_row_reactions(self._data.structural_rows.world, self._data.structural_rows.reaction, world_mask)

    def reset_box_reactions(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear the joint-friction and limit warm starts of the masked worlds, or of all worlds."""
        self._reset_row_reactions(self._data.box_rows.world, self._data.box_rows.reaction, world_mask)

    def reset_angular_contact_reactions(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear the spin and rolling warm starts of spatial contacts in the masked worlds, or in all worlds."""
        rows = self._data.contact_rows
        if rows.angular_warmstarter is None:
            return
        rows.angular_warmstarter.reset(world_mask)
        if rows.capacity > 0:
            wp.launch(
                _reset_angular_reactions,
                dim=rows.capacity,
                inputs=[rows.world, world_mask],
                outputs=[rows.angular_reaction],
                device=self.device,
            )

    def reset_effort_counters(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear the finite actuator corrections of the masked worlds, or of all worlds."""
        if not self.has_bounded_effort:
            return
        wp.launch(
            _reset_effort_rows,
            dim=self._data.effort_rows.capacity,
            inputs=[self._data.effort_rows.world, world_mask],
            outputs=[
                self._data.effort_rows.intercept,
                self._data.effort_rows.slope,
                self._data.effort_rows.impulse_bound,
                self._data.effort_rows.counter,
                self._data.effort_rows.net_applied,
            ],
            device=self.device,
        )
        wp.launch(
            _reset_effort_worlds,
            dim=self.num_worlds,
            inputs=[world_mask],
            outputs=[self._data.effort_rows.world_residual_max],
            device=self.device,
        )

    ###
    # Internals - Topology and Allocation
    ###

    def _classify_dynamic_bodies(self) -> tuple[int, ...]:
        """Return dynamic bodies using state flags and fixed-tree world islands."""
        source_model = self.model._model
        body_count = self.model.size.sum_of_num_bodies
        if source_model is None:
            inverse_mass = self.model.bodies.inv_m_i.numpy()
            return tuple(np.flatnonzero(inverse_mass > 0.0).tolist())

        body_flags = source_model.body_flags.numpy().astype(np.int32, copy=False)
        if len(body_flags) != body_count:
            raise ValueError("Newton body flags must contain one entry per packed Kamino body.")
        if not self.eliminate_fixed_world_islands:
            return tuple(np.flatnonzero((body_flags & int(BodyFlags.KINEMATIC)) == 0).tolist())

        joint_type = source_model.joint_type.numpy().astype(np.int32, copy=False)
        joint_parent = source_model.joint_parent.numpy().astype(np.int32, copy=False)
        joint_child = source_model.joint_child.numpy().astype(np.int32, copy=False)
        articulation_start = source_model.articulation_start.numpy().astype(np.int32, copy=False)[:-1]
        articulation_end = source_model.articulation_end.numpy().astype(np.int32, copy=False)
        tree_delta = np.zeros(joint_type.size + 1, dtype=np.int32)
        np.add.at(tree_delta, articulation_start, 1)
        np.add.at(tree_delta, articulation_end, -1)
        tree_joint = np.cumsum(tree_delta[:-1]) != 0
        fixed_joint = tree_joint & (joint_type == int(JointType.FIXED)) & (joint_child >= 0)
        fixed_to_world = joint_child[fixed_joint & (joint_parent < 0)]
        fixed_pair = fixed_joint & (joint_parent >= 0)
        labels = _connected_component_labels(body_count, joint_parent[fixed_pair], joint_child[fixed_pair])
        prescribed_bodies = np.concatenate(
            (fixed_to_world, np.flatnonzero(body_flags & int(BodyFlags.KINEMATIC)).astype(np.int32))
        )
        prescribed_labels = np.unique(labels[prescribed_bodies])
        return tuple(np.flatnonzero(~np.isin(labels, prescribed_labels)).tolist())

    def _build_body_edges(self, dynamic_bodies: Sequence[int]) -> tuple[tuple[int, int], ...]:
        """Return fixed joint edges whose endpoints are both dynamic bodies."""
        joint_a = self.model.joints.bid_B.numpy().astype(np.int32, copy=False)
        joint_b = self.model.joints.bid_F.numpy().astype(np.int32, copy=False)
        dynamic_mask = np.zeros(self.model.size.sum_of_num_bodies, dtype=bool)
        dynamic_mask[np.asarray(dynamic_bodies, dtype=np.int32)] = True
        edge_mask = (joint_a >= 0) & (joint_b >= 0)
        edge_mask &= dynamic_mask[joint_a.clip(min=0)] & dynamic_mask[joint_b.clip(min=0)]
        return tuple(map(tuple, np.column_stack((joint_a[edge_mask], joint_b[edge_mask])).tolist()))

    def _build_body_components(
        self, dynamic_bodies: Sequence[int], body_edges: Sequence[tuple[int, int]]
    ) -> tuple[tuple[int, ...], ...]:
        """Return joint-connected dynamic-body components without contact edges."""
        dynamic_np = np.asarray(dynamic_bodies, dtype=np.int32)
        if dynamic_np.size == 0:
            return ()
        edges_np = np.asarray(body_edges, dtype=np.int32).reshape((-1, 2))
        labels = _connected_component_labels(
            self.model.size.sum_of_num_bodies,
            edges_np[:, 0],
            edges_np[:, 1],
        )[dynamic_np]
        order = np.argsort(labels, kind="stable")
        sorted_bodies = dynamic_np[order]
        sorted_labels = labels[order]
        boundaries = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
        return tuple(tuple(component.tolist()) for component in np.split(sorted_bodies, boundaries))

    def _allocate_joint_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """Allocate the dynamic and structural joint rows.

        Returns:
            The world and joint DOF of each dynamic row, on the host.
        """
        model = self.model
        joint_count = model.size.sum_of_num_joints
        joint_indices = np.arange(joint_count, dtype=np.int32)
        joint_world = model.joints.wid.numpy().astype(np.int32, copy=False)
        joint_body_a = model.joints.bid_B.numpy().astype(np.int32, copy=False)
        joint_body_b = model.joints.bid_F.numpy().astype(np.int32, copy=False)
        joint_dof_count = model.joints.num_dofs.numpy().astype(np.int32, copy=False)
        joint_dynamic_count = model.joints.num_dynamic_cts.numpy().astype(np.int32, copy=False)
        joint_effort_count = model.joints.num_effort_cts.numpy().astype(np.int32, copy=False)
        joint_dynamic_axis = model.joints.dynamic_cts_axis.numpy().astype(np.int32, copy=False)
        joint_structural_count = model.joints.num_kinematic_cts.numpy().astype(np.int32, copy=False)
        joint_dynamic_offset = model.joints.dynamic_cts_offset.numpy().astype(np.int32, copy=False)
        joint_dof_offset = model.joints.dofs_offset.numpy().astype(np.int32, copy=False)
        joint_dof_actuation_path = model.joints.dof_act_paths.numpy().astype(np.int32, copy=False)
        joint_kinematic_offset = model.joints.kinematic_cts_offset.numpy().astype(np.int32, copy=False)
        sparse_joint_offsets = self.jacobians._J_cts_joint_nzb_offsets.numpy().astype(np.int32, copy=False)
        sparse_dof_joint_offsets = self.jacobians._J_dofs_joint_nzb_offsets.numpy().astype(np.int32, copy=False)

        native_joint = np.repeat(joint_indices, joint_dynamic_count)
        native_offsets = capacity_offsets(joint_dynamic_count)
        native_local = segment_local_indices(joint_dynamic_count, native_offsets)
        native_axis = joint_dynamic_axis[joint_dynamic_offset[native_joint] + native_local]
        native_dof = joint_dof_offset[native_joint] + native_axis

        dof_joint = np.repeat(joint_indices, joint_dof_count)
        dof_offsets = capacity_offsets(joint_dof_count)
        dof_axis = segment_local_indices(joint_dof_count, dof_offsets)
        dof_index = joint_dof_offset[dof_joint] + dof_axis
        native_dof_mask = np.zeros(model.size.sum_of_num_joint_dofs, dtype=bool)
        native_dof_mask[native_dof] = True
        joint_retains_effort_rows = np.repeat(joint_effort_count > 0, joint_dof_count)
        extra_mask = (
            (joint_dof_actuation_path[dof_index] == int(DofActuationPath.EFFORT_CTS))
            & joint_retains_effort_rows[dof_index]
            & ~native_dof_mask[dof_index]
        )
        extra_joint = dof_joint[extra_mask]
        extra_axis = dof_axis[extra_mask]
        extra_dof = dof_index[extra_mask]
        extra_count = np.bincount(extra_joint, minlength=joint_count).astype(np.int32, copy=False)

        joint_dynamic_row_count = joint_dynamic_count + extra_count
        joint_dynamic_row_offset = capacity_offsets(joint_dynamic_row_count)
        dynamic_count = int(joint_dynamic_row_offset[-1])
        dynamic_world = np.empty(dynamic_count, dtype=np.int32)
        dynamic_joint = np.empty(dynamic_count, dtype=np.int32)
        dynamic_a_global = np.empty(dynamic_count, dtype=np.int32)
        dynamic_b_global = np.empty(dynamic_count, dtype=np.int32)
        dynamic_value_index = np.full(dynamic_count, -1, dtype=np.int32)
        dynamic_dof_index = np.empty(dynamic_count, dtype=np.int32)
        dynamic_multiplier_index = np.full(dynamic_count, -1, dtype=np.int32)
        dynamic_sparse_a_index = np.full(dynamic_count, -1, dtype=np.int32)
        dynamic_sparse_b_index = np.empty(dynamic_count, dtype=np.int32)
        dynamic_uses_dof_jacobian = np.ones(dynamic_count, dtype=bool)

        native_target = joint_dynamic_row_offset[native_joint] + native_local
        extra_offsets = capacity_offsets(extra_count)
        extra_local = segment_local_indices(extra_count, extra_offsets)
        extra_target = joint_dynamic_row_offset[extra_joint] + joint_dynamic_count[extra_joint] + extra_local
        for target, source_joint in ((native_target, native_joint), (extra_target, extra_joint)):
            dynamic_world[target] = joint_world[source_joint]
            dynamic_joint[target] = source_joint
            dynamic_a_global[target] = joint_body_a[source_joint]
            dynamic_b_global[target] = joint_body_b[source_joint]
        dynamic_value_index[native_target] = joint_dynamic_offset[native_joint] + native_local
        dynamic_dof_index[native_target] = native_dof
        dynamic_dof_index[extra_target] = extra_dof
        dynamic_multiplier_index[native_target] = joint_dynamic_offset[native_joint] + native_local
        # Kamino stores the base blocks of a joint's dynamic rows after their follower blocks
        native_sparse_a = sparse_joint_offsets[native_joint] + joint_dynamic_count[native_joint] + native_local
        dynamic_sparse_a_index[native_target] = np.where(joint_body_a[native_joint] >= 0, native_sparse_a, -1)
        dynamic_sparse_b_index[native_target] = sparse_joint_offsets[native_joint] + native_local
        extra_sparse_a = sparse_dof_joint_offsets[extra_joint] + joint_dof_count[extra_joint] + extra_axis
        dynamic_sparse_a_index[extra_target] = np.where(joint_body_a[extra_joint] >= 0, extra_sparse_a, -1)
        dynamic_sparse_b_index[extra_target] = sparse_dof_joint_offsets[extra_joint] + extra_axis
        dynamic_uses_dof_jacobian[native_target] = False

        self._data.dynamic_rows = DynamicRows(
            dynamic_world,
            dynamic_joint,
            dynamic_a_global,
            dynamic_b_global,
            dynamic_value_index,
            dynamic_dof_index,
            dynamic_multiplier_index,
            dynamic_sparse_a_index,
            dynamic_sparse_b_index,
            dynamic_uses_dof_jacobian,
            joint_dynamic_row_offset[:-1],
            joint_dynamic_row_count,
            self.device,
        )

        structural_joint = np.repeat(joint_indices, joint_structural_count)
        structural_offsets = capacity_offsets(joint_structural_count)
        structural_local = segment_local_indices(joint_structural_count, structural_offsets)
        structural_value_index = joint_kinematic_offset[structural_joint] + structural_local
        structural_count = int(structural_offsets[-1])
        if not np.array_equal(structural_value_index, np.arange(structural_count, dtype=np.int32)):
            raise RuntimeError("LOX structural rows must use Kamino's canonical kinematic-row order.")
        world_structural_count = model.info.num_joint_kinematic_cts.numpy().astype(np.int32, copy=False)
        if structural_count != int(world_structural_count.sum()):
            raise RuntimeError("LOX structural row count must match Kamino's kinematic constraints.")
        if any(
            array.shape[0] != structural_count
            for array in (self.kamino_data.joints.r_j, self.kamino_data.joints.lambda_kin_j)
        ):
            raise RuntimeError("Kamino kinematic residual and reaction arrays must use canonical row storage.")
        structural_world = joint_world[structural_joint]
        structural_a_global = joint_body_a[structural_joint]
        structural_b_global = joint_body_b[structural_joint]
        adjacent_body_count = np.where(joint_body_a >= 0, 2, 1)
        # The kinematic blocks of a joint follow the blocks of its dynamic rows
        structural_sparse_start = sparse_joint_offsets + adjacent_body_count * joint_dynamic_count
        structural_sparse_a_index = np.where(
            structural_a_global >= 0,
            structural_sparse_start[structural_joint] + joint_structural_count[structural_joint] + structural_local,
            -1,
        )
        structural_sparse_b_index = structural_sparse_start[structural_joint] + structural_local
        self._data.structural_rows = StructuralRows(
            structural_world,
            structural_a_global,
            structural_b_global,
            structural_sparse_a_index,
            structural_sparse_b_index,
            self.num_worlds,
            self.kamino_data.joints.r_j,
            self.kamino_data.joints.lambda_kin_j,
            model.size.sum_of_num_bodies,
            self.joint_proximal_relaxation > 0.0,
            self.device,
        )
        if self.joint_proximal_relaxation > 0.0:
            self._data.joint_coordinate_scratch = wp.zeros(
                model.size.sum_of_num_joint_coords, dtype=wp.float32, device=self.device
            )
            self._data.joint_velocity_scratch = wp.zeros(
                model.size.sum_of_num_joint_dofs, dtype=wp.float32, device=self.device
            )
        return dynamic_world, dynamic_dof_index

    def _allocate_effort_rows(self, dynamic_world: np.ndarray, dynamic_dof: np.ndarray) -> None:
        """Allocate one bounded-effort correction per actuated dynamic row driven through Kamino's effort constraints.

        Each correction maps its dynamic row to its Kamino effort index, and the corrections touching each
        body are indexed per body.
        """
        model = self.model
        effort_limits = model.joints.tau_j_max.numpy()
        if len(effort_limits) != model.size.sum_of_num_joint_dofs:
            raise ValueError("Joint effort limits must contain one value per joint DOF.")
        if np.any(np.isnan(effort_limits) | (effort_limits < 0.0)):
            raise ValueError("Joint effort limits must be nonnegative or positive infinity.")

        dof_actuation = model.joints.dof_act_types.numpy().astype(np.int32, copy=False)
        dof_actuation_path = model.joints.dof_act_paths.numpy().astype(np.int32, copy=False)
        joint_dof_offset = model.joints.dofs_offset.numpy().astype(np.int32, copy=False)
        effort_offset = model.joints.effort_cts_offset.numpy().astype(np.int32, copy=False)
        effort_axis = model.joints.effort_cts_axis.numpy().astype(np.int32, copy=False)
        dynamic_joint = self._data.dynamic_rows.joint.numpy()
        dynamic_a = self._data.dynamic_rows.body_a.numpy()
        dynamic_b = self._data.dynamic_rows.body_b.numpy()

        effort_mask = (dof_actuation[dynamic_dof] > int(JointActuationType.PASSIVE)) & (
            dof_actuation_path[dynamic_dof] == int(DofActuationPath.EFFORT_CTS)
        )
        effort_dynamic_row = np.flatnonzero(effort_mask).astype(np.int32)
        effort_world = dynamic_world[effort_dynamic_row]
        effort_joint = dynamic_joint[effort_dynamic_row]
        effort_dof = dynamic_dof[effort_dynamic_row]
        effort_counts = np.diff(effort_offset)
        source_joint = np.repeat(np.arange(model.size.sum_of_num_joints, dtype=np.int32), effort_counts)
        source_dof = joint_dof_offset[source_joint] + effort_axis
        dof_to_effort = np.full(model.size.sum_of_num_joint_dofs, -1, dtype=np.int32)
        dof_to_effort[source_dof] = np.arange(effort_axis.size, dtype=np.int32)
        effort_value_index = dof_to_effort[effort_dof]
        if np.any(effort_value_index < 0):
            missing = int(np.flatnonzero(effort_value_index < 0)[0])
            joint = int(effort_joint[missing])
            axis = int(effort_dof[missing] - joint_dof_offset[joint])
            raise RuntimeError(f"Missing effort row for joint {joint} axis {axis}.")

        effort_capacity = effort_dynamic_row.size
        self.has_bounded_effort = effort_capacity > 0
        dynamic_effort_index = np.full(self._data.dynamic_rows.count, -1, dtype=np.int32)
        dynamic_effort_index[effort_dynamic_row] = np.arange(effort_capacity, dtype=np.int32)
        self._data.dynamic_rows.effort_index.assign(dynamic_effort_index)

        body_offsets, incidence_effort, incidence_side = _body_row_incidence(
            dynamic_a[effort_dynamic_row], dynamic_b[effort_dynamic_row], model.size.sum_of_num_bodies
        )

        self._data.effort_rows = EffortRows(
            effort_world,
            effort_dynamic_row,
            effort_value_index,
            body_offsets,
            incidence_effort,
            incidence_side,
            self.num_worlds,
            self.device,
        )

    def _allocate_joint_frictions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Allocate the DOF mapping of one bounded row for each of Kamino's joint friction rows.

        Returns:
            The number of friction rows of each world, and the world, body A,
            and body B of each friction row, on the host.
        """
        source_model = self.model._model
        if source_model is None or source_model.joint_friction is None:
            friction_values = np.zeros(self.model.size.sum_of_num_joint_dofs, dtype=np.float32)
            self._data.joint_friction_force = wp.zeros(
                self.model.size.sum_of_num_joint_dofs,
                dtype=wp.float32,
                device=self.device,
            )
        else:
            friction_values = source_model.joint_friction.numpy()
            self._data.joint_friction_force = source_model.joint_friction
        if len(friction_values) != self.model.size.sum_of_num_joint_dofs:
            raise ValueError("Newton joint friction must contain one value per joint DOF.")
        if np.any(~np.isfinite(friction_values) | (friction_values < 0.0)):
            raise ValueError("Joint friction values must be finite and nonnegative.")

        joint_worlds = self.model.joints.wid.numpy().astype(np.int32, copy=False)
        joint_a = self.model.joints.bid_B.numpy().astype(np.int32, copy=False)
        joint_b = self.model.joints.bid_F.numpy().astype(np.int32, copy=False)
        joint_dof_offsets = self.model.joints.dofs_offset.numpy().astype(np.int32, copy=False)
        joint_dof_counts = self.model.joints.num_dofs.numpy().astype(np.int32, copy=False)
        joint_friction_counts = self.model.joints.num_friction_cts.numpy().astype(np.int32, copy=False)
        joint_friction_offsets = self.model.joints.friction_cts_offset.numpy().astype(np.int32, copy=False)
        sparse_offsets = self.jacobians._J_dofs_joint_nzb_offsets.numpy().astype(np.int32, copy=False)
        friction_axis = np.zeros(0, dtype=np.int32)
        if self.model.joints.friction_cts_axis is not None:
            friction_axis = self.model.joints.friction_cts_axis.numpy().astype(np.int32, copy=False)

        body_vector_index = np.asarray(self.system.body_vector_index_host, dtype=np.int32)
        a_is_dynamic = (joint_a >= 0) & (body_vector_index[joint_a.clip(min=0)] >= 0)
        b_is_dynamic = (joint_b >= 0) & (body_vector_index[joint_b.clip(min=0)] >= 0)
        active_joint_mask = (joint_friction_counts > 0) & (a_is_dynamic | b_is_dynamic)
        active_joints = np.flatnonzero(active_joint_mask).astype(np.int32)
        # Kamino allocates friction rows only for the DOFs with friction, and records the axis of each row
        row_joint = np.repeat(active_joints, joint_friction_counts[active_joints])
        local_row = segment_local_indices(joint_friction_counts[active_joints])
        friction_row = joint_friction_offsets[row_joint] + local_row
        local_dof = friction_axis[friction_row]
        row_world = joint_worlds[row_joint]
        order = np.argsort(row_world, kind="stable")
        row_joint = row_joint[order]
        local_dof = local_dof[order]
        friction_row = friction_row[order]
        row_world = row_world[order]
        counts = np.bincount(row_world, minlength=self.num_worlds).astype(np.int32, copy=False)
        self._data.friction_dof_index = to_warp_int32_array(joint_dof_offsets[row_joint] + local_dof, self.device)
        self._data.friction_sparse_a_index = to_warp_int32_array(
            np.where(
                joint_a[row_joint] >= 0,
                sparse_offsets[row_joint] + joint_dof_counts[row_joint] + local_dof,
                -1,
            ),
            self.device,
        )
        self._data.friction_sparse_b_index = to_warp_int32_array(sparse_offsets[row_joint] + local_dof, self.device)
        self._data.friction_multiplier_index = to_warp_int32_array(friction_row, self.device)
        return counts, row_world, joint_a[row_joint], joint_b[row_joint]

    def _allocate_unilaterals(
        self,
        world_friction_count: np.ndarray,
        friction_world: np.ndarray,
        friction_body_a: np.ndarray,
        friction_body_b: np.ndarray,
    ) -> None:
        """Allocate the box rows, the contact rows, and the body incidence of unilateral rows."""
        limit_capacities = np.zeros(self.num_worlds, dtype=np.int32)
        if self.limits is not None and self.limits.model_max_limits_host > 0:
            limit_capacities = np.asarray(self.limits.world_max_limits_host, dtype=np.int32)
        contact_capacities = np.zeros(self.num_worlds, dtype=np.int32)
        if self.contacts is not None:
            try:
                if self.contacts.model_max_contacts_host > 0:
                    contact_capacities = np.asarray(self.contacts.world_max_contacts_host, dtype=np.int32)
            except RuntimeError:
                self.contacts = None
        if len(limit_capacities) != self.num_worlds or len(contact_capacities) != self.num_worlds:
            raise ValueError("Unilateral container world capacities must match the model world count.")

        self._data.box_rows = BoxRows(
            world_friction_count,
            limit_capacities,
            friction_world,
            friction_body_a,
            friction_body_b,
            self.device,
        )
        compliant = self.contact_compliance
        if self.contact_spatial_friction:
            source_capacity = self.contacts.model_max_contacts_host if self.contacts is not None else 0
            self._data.contact_rows = SpatialContactRows(
                contact_capacities, self.device, source_capacity, compliant, self.contact_restitution
            )
        else:
            self._data.contact_rows = ContactRows(contact_capacities, self.device, compliant, self.contact_restitution)

        body_count = self.model.size.sum_of_num_bodies
        self._data.body_incidence = wp.zeros((body_count, 1), dtype=wp.int32, device=self.device)
        self._data.body_unilateral_count = self._data.body_incidence.reshape((body_count,))
        # Selective weights enable exactly the bodies touched by unilateral rows
        self._data.block_has_unilateral = None
        self._data.body_weight_enabled = self._data.body_unilateral_count
        if not self.selective_weights:
            self._data.block_has_unilateral = wp.zeros(self.system.num_blocks, dtype=wp.int32, device=self.device)
            self._data.body_weight_enabled = wp.zeros(body_count, dtype=wp.int32, device=self.device)

    ###
    # Internals - Time Step
    ###

    def _update_unilaterals(
        self,
        time_step: wp.array[wp.float32],
        limit_stabilization_fraction: float,
        contact_stabilization_fraction: float,
        contact_dead_zone: float,
    ) -> None:
        """Load the box rows and contacts of a time step, with their warm-start reactions.

        Also counts the active rows touching each body into :attr:`LOXProblemData.body_incidence`.
        """
        self._data.body_incidence.zero_()
        self._update_box_rows(time_step, limit_stabilization_fraction)
        self._update_contacts(
            time_step,
            contact_stabilization_fraction,
            contact_dead_zone,
        )

    def _update_box_rows(self, time_step: wp.array[wp.float32], limit_stabilization_fraction: float) -> None:
        """Load the friction rows and the detected limits, then clear the unused limit rows."""
        has_limits = self._data.box_rows.limit_capacity > 0 and self.limits is not None
        wp.launch(
            _count_box_rows,
            dim=self.num_worlds,
            inputs=[
                self.limits.world_active_limits if has_limits else None,
                self._data.box_rows.world_limit_capacity,
                self._data.box_rows.world_friction_count,
            ],
            outputs=[self._data.box_rows.world_count],
            device=self.device,
        )
        if self._data.box_rows.friction_capacity > 0:
            wp.launch(
                _load_joint_frictions,
                dim=self._data.box_rows.friction_capacity,
                inputs=[
                    self._data.box_rows.friction_row,
                    self._data.friction_dof_index,
                    self._data.friction_multiplier_index,
                    self._data.friction_sparse_a_index,
                    self._data.friction_sparse_b_index,
                    self._sparse_dof_jacobian_data,
                    self._data.joint_friction_force,
                    self.kamino_data.joints.lambda_f_j,
                    self._data.box_rows.world,
                    self._data.box_rows.body_a,
                    self._data.box_rows.body_b,
                    self.body_velocity_begin,
                    time_step,
                ],
                outputs=[
                    self._data.box_rows.jacobian_a,
                    self._data.box_rows.jacobian_b,
                    self._data.box_rows.lower,
                    self._data.box_rows.upper,
                    self._data.box_rows.reaction,
                    self._data.box_rows.velocity,
                ],
                device=self.device,
            )
        if has_limits:
            wp.launch(
                _load_limits,
                dim=self.limits.model_max_limits_host,
                inputs=[
                    self.limits.model_active_limits,
                    self.limits.model_max_limits_host,
                    self.limits.wid,
                    self.limits.lid,
                    self.limits.bids,
                    self.limits.r_q,
                    self.limits.reaction,
                    self.body_velocity_begin,
                    self._data.box_rows.world_limit_capacity,
                    self._data.box_rows.world_limit_offset,
                    self.system.data.body_vector_index,
                    self._sparse_limit_offsets,
                    self._sparse_jacobian_data,
                    time_step,
                    limit_stabilization_fraction,
                ],
                outputs=[
                    self._data.box_rows.body_a,
                    self._data.box_rows.body_b,
                    self._data.box_rows.jacobian_a,
                    self._data.box_rows.jacobian_b,
                    self._data.box_rows.bias,
                    self._data.box_rows.reaction,
                    self._data.box_rows.velocity,
                ],
                device=self.device,
            )
        if self._data.box_rows.capacity > 0:
            wp.launch(
                _finish_box_rows,
                dim=self._data.box_rows.capacity,
                inputs=[self._data.box_rows.world, self._data.box_rows.local, self._data.box_rows.world_count],
                outputs=[
                    self._data.box_rows.body_a,
                    self._data.box_rows.body_b,
                    self._data.box_rows.reaction,
                    self._data.box_rows.velocity,
                    self._data.body_incidence,
                ],
                device=self.device,
            )

    def _update_contacts(
        self,
        time_step: wp.array[wp.float32],
        stabilization_fraction: float,
        dead_zone: float,
    ) -> None:
        """Compact the detected contacts into the contact rows and clear the unused ones."""
        if self._data.contact_rows.capacity > 0 and self.contacts is not None:
            self._data.contact_rows.world_count.zero_()
            wp.launch(
                _load_contacts,
                dim=self.contacts.model_max_contacts_host,
                inputs=[
                    self.contacts.model_active_contacts,
                    self.contacts.model_max_contacts_host,
                    self.contacts.wid,
                    self.contacts.cid,
                    self.contacts.bid_AB,
                    self.contacts.position_A,
                    self.contacts.position_B,
                    self.contacts.frame,
                    self.contacts.gapfunc,
                    self.contacts.material,
                    self.contacts.angular_friction,
                    self.contacts.reaction,
                    self.kamino_data.bodies.q_i,
                    self.body_velocity_begin,
                    self._data.contact_rows.world_capacity,
                    self._data.contact_rows.world_count,
                    self._data.contact_rows.world_offset,
                    self.system.data.body_vector_index,
                    time_step,
                    stabilization_fraction,
                    dead_zone,
                    self.contacts.stiffness if self.contact_compliance else None,
                    self.contacts.damping if self.contact_compliance else None,
                    self.contact_restitution,
                    self._data.contact_rows.source_to_internal,
                ],
                outputs=[
                    self._data.contact_rows.body_a,
                    self._data.contact_rows.body_b,
                    self._data.contact_rows.jacobian_a,
                    self._data.contact_rows.jacobian_b,
                    self._data.contact_rows.bias,
                    self._data.contact_rows.normal_compliance,
                    self._data.contact_rows.restitution,
                    self._data.contact_rows.friction,
                    self._data.contact_rows.reaction,
                    self._data.contact_rows.velocity,
                    self._data.contact_rows.frame,
                    self._data.contact_rows.angular_friction,
                ],
                device=self.device,
            )
            if self._data.contact_rows.angular_warmstarter is not None:
                # Warm-start the spin and rolling impulses from the contacts of the previous time step
                self._data.contact_rows.angular_reaction.zero_()
                self._data.contact_rows.angular_warmstarter.warmstart(
                    self.contacts,
                    self._data.contact_rows.source_to_internal,
                    self.kamino_data.bodies.q_i,
                    time_step,
                    self._data.contact_rows.angular_reaction,
                )
        else:
            self._data.contact_rows.world_count.zero_()
        if self._data.contact_rows.capacity > 0:
            wp.launch(
                _finish_contacts,
                dim=self._data.contact_rows.capacity,
                inputs=[
                    self._data.contact_rows.world,
                    self._data.contact_rows.local,
                    self._data.contact_rows.world_count,
                ],
                outputs=[
                    self._data.contact_rows.body_a,
                    self._data.contact_rows.body_b,
                    self._data.contact_rows.reaction,
                    self._data.contact_rows.velocity,
                    self._data.body_incidence,
                ],
                device=self.device,
            )

    def _select_weighted_bodies(self) -> None:
        """Enable the splitting weight of the bodies touched by unilateral rows, or of their whole factor blocks.

        With selective weights, :attr:`body_weight_enabled` aliases :attr:`body_unilateral_count`.
        """
        if self.selective_weights:
            return
        body_block = self.system.data.body_block
        self._data.block_has_unilateral.zero_()
        wp.launch(
            _mark_blocks_with_unilaterals,
            dim=body_block.shape[0],
            inputs=[body_block, self._data.body_unilateral_count],
            outputs=[self._data.block_has_unilateral],
            device=self.device,
        )
        wp.launch(
            _enable_bodies_in_weighted_blocks,
            dim=body_block.shape[0],
            inputs=[body_block, self._data.block_has_unilateral],
            outputs=[self._data.body_weight_enabled],
            device=self.device,
        )

    ###
    # Internals - Resets
    ###

    def _reset_row_reactions(
        self,
        row_world: wp.array[wp.int32],
        reaction: wp.array[wp.float32],
        world_mask: wp.array[wp.bool] | None,
    ) -> None:
        if row_world.shape[0] > 0:
            wp.launch(
                _reset_row_reactions,
                dim=row_world.shape[0],
                inputs=[row_world, world_mask],
                outputs=[reaction],
                device=self.device,
            )
