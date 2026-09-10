# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent primal problem for the LOX rigid-body backend.

The problem owns persistent rows, capacity mappings, and splitting state.
Construction may read immutable model arrays on the host; :meth:`LOXProblem.update`
uses only fixed-size device operations so it remains suitable for graph capture.
Kamino contact vectors remain normal-last throughout this boundary.
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
from ...core.types import mat36f, vec6f
from ...geometry.contacts import ContactsKamino
from ...kinematics.constraints import update_constraints_info
from ...kinematics.jacobians import SparseSystemJacobians
from ...kinematics.joints import compute_joints_data
from ...kinematics.limits import LimitsKamino
from .adapter import (
    _accumulate_aligned_joint_wrenches,
    _accumulate_constraint_incidence,
    _clear_inactive_contacts,
    _clear_inactive_limits,
    _copy_clamped_world_counts,
    _mark_worlds_with_unilaterals,
    _prepare_contacts,
    _prepare_dynamic_rows,
    _prepare_joint_frictions,
    _prepare_limits,
    _prepare_structural_rows,
    _write_contact_outputs,
    _write_dynamic_outputs,
    _write_dynamic_outputs_with_effort,
    _write_friction_outputs,
    _write_limit_outputs,
)
from .kernels import (
    _blend_structural_candidate_twist,
    _clear_active_effort_residuals,
    _evaluate_lagged_contact_velocity_consistency,
    _evaluate_lagged_scalar_velocity_consistency,
    _include_dynamic_compliance_in_structural_effective_mass,
    _promote_effort_counters,
    _reduce_structural_candidate_residual,
    _reset_effort_rows_masked,
    _reset_effort_worlds_masked,
    _reset_friction_reactions_masked,
    _reset_structural_multipliers_masked,
    _scale_structural_reactions,
    _update_effort_counters,
    _update_structural_multipliers_from_candidate_rows,
    make_evaluate_candidate_structural_residual_kernel,
)
from .system import BatchedPrimalBodySystem
from .types import SplittingState

__all__ = ["LOXProblem"]

wp.set_module_options({"enable_backward": False})

_LAGGED_CONTACT_BLOCK_DIM = 64


def _capacity_offsets(capacities: Sequence[int] | np.ndarray) -> np.ndarray:
    capacities_np = np.asarray(capacities, dtype=np.int32)
    offsets = np.empty(capacities_np.size + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(capacities_np, out=offsets[1:])
    return offsets


def _segment_local_indices(counts: np.ndarray, offsets: np.ndarray | None = None) -> np.ndarray:
    """Return packed indices relative to the start of each variable-length segment."""
    if offsets is None:
        offsets = _capacity_offsets(counts)
    return np.arange(int(offsets[-1]), dtype=np.int32) - np.repeat(offsets[:-1], counts)


def _connected_component_labels(
    count: int,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Compute minimum-vertex component labels with vectorized label propagation."""
    labels = np.arange(count, dtype=np.int32)
    if first.size == 0:
        return labels
    while True:
        endpoint_labels = np.minimum(labels[first], labels[second])
        updated = labels.copy()
        np.minimum.at(updated, first, endpoint_labels)
        np.minimum.at(updated, second, endpoint_labels)
        updated = updated[updated]
        if np.array_equal(updated, labels):
            return labels
        labels = updated


class LOXProblem:
    """Own rigid-body topology, constraint rows, and nonlinear splitting state."""

    def __init__(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: SparseSystemJacobians,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        eliminate_fixed_world_islands: bool = True,
        projection_method: str = "jacobi",
        rotation_correction: JointCorrectionMode = JointCorrectionMode.TWOPI,
        joint_proximal_relaxation: float = 0.0,
    ):
        self.model = model
        self.data = data
        self.jacobians = jacobians
        self.limits = limits
        self.contacts = contacts
        self.device = wp.get_device(model.device)
        self.num_worlds = model.info.num_worlds
        self.eliminate_fixed_world_islands = eliminate_fixed_world_islands
        self.projection_method = projection_method
        self.rotation_correction = rotation_correction
        self._evaluate_candidate_structural_residual = make_evaluate_candidate_structural_residual_kernel(
            rotation_correction
        )
        self.joint_proximal_relaxation = joint_proximal_relaxation
        self._allocate_acceleration_storage = projection_method == "apgd"

        if jacobians._J_cts is None or jacobians._J_dofs is None:
            raise RuntimeError("Sparse Jacobians must be finalized before constructing the LOX problem.")
        self._sparse_jacobian_data = jacobians._J_cts.bsm.nzb_values
        self._sparse_dof_jacobian_data = jacobians._J_dofs.bsm.nzb_values
        self._sparse_limit_offsets = jacobians._J_cts_limit_nzb_offsets

        self.rebuild_dynamic_body_topology()

    def prepare_joint_penalty_scale_seed(self, time_step: float) -> None:
        """Prepare the initial rigid state for LOX penalty-scale seeding."""
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        if np.any(self.data.time.steps.numpy() != 0):
            raise RuntimeError("joint penalty scale seeding must run before the first step or after a solver reset.")

        self.model.time.set_uniform_timestep(time_step)
        compute_joints_data(
            model=self.model,
            data=self.data,
            q_j_p=self.data.joints.q_j_p,
            correction=self.rotation_correction,
        )
        update_body_inertias(model=self.model.bodies, data=self.data.bodies)
        if self.limits is not None:
            self.limits.detect(q_j=self.data.joints.q_j)
        update_constraints_info(model=self.model, data=self.data)
        self.jacobians.build(
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=None,
            reset_to_zero=True,
        )

    @staticmethod
    def _device_array(values: Sequence[int], device: wp.DeviceLike) -> wp.array[wp.int32]:
        return wp.array(values, dtype=wp.int32, device=device)

    def _build_body_edges(self, dynamic_bodies: Sequence[int]) -> tuple[tuple[int, int], ...]:
        """Return fixed joint edges whose endpoints are both dynamic bodies."""
        joint_first = self.model.joints.bid_B.numpy().astype(np.int32, copy=False)
        joint_second = self.model.joints.bid_F.numpy().astype(np.int32, copy=False)
        dynamic_mask = np.zeros(self.model.size.sum_of_num_bodies, dtype=bool)
        dynamic_mask[np.asarray(dynamic_bodies, dtype=np.int32)] = True
        edge_mask = (joint_first >= 0) & (joint_second >= 0)
        edge_mask &= dynamic_mask[joint_first.clip(min=0)] & dynamic_mask[joint_second.clip(min=0)]
        return tuple(map(tuple, np.column_stack((joint_first[edge_mask], joint_second[edge_mask])).tolist()))

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

    def dynamic_body_topology_changed(self) -> bool:
        """Return whether current body flags require a different primal topology."""
        return self._classify_dynamic_bodies() != self.system.dynamic_bodies

    def rebuild_dynamic_body_topology(self) -> None:
        """Reallocate topology-dependent data while preserving adapter identity."""
        model = self.model
        body_counts = tuple(model.info.num_bodies.numpy().astype(np.int32, copy=False).tolist())
        dynamic_bodies = self._classify_dynamic_bodies()
        body_edges = self._build_body_edges(dynamic_bodies)
        body_components = self._build_body_components(dynamic_bodies, body_edges)
        self.system = BatchedPrimalBodySystem(
            body_counts,
            body_components=body_components,
            dynamic_bodies=dynamic_bodies,
            body_edges=body_edges,
            device=self.device,
        )
        self.splitting = SplittingState(body_counts, device=self.device)
        self.world_active = self.splitting.world_active
        self.projected_twist = self.splitting.projected_twist
        self.body_velocity_begin = wp.zeros(model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        self.joint_velocity_begin = wp.zeros(model.size.sum_of_num_joint_dofs, dtype=wp.float32, device=self.device)
        self.body_linearization_twist = wp.zeros(model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        self._limit_stabilization_fraction = 0.01
        self._contact_stabilization_fraction = 0.01
        self._contact_dead_zone = 1.0e-6
        self._impact_velocity_threshold = 1.0e-3
        self._contact_recoverable_response = False
        self._uniform_joint_penalty_scale = wp.ones(self.num_worlds, dtype=wp.float32, device=self.device)

        self._allocate_joint_rows()
        self.system.validate_body_pairs("dynamic rows", self.dynamic_body_first_global, self.dynamic_body_second_global)
        self.system.validate_body_pairs(
            "structural rows", self.structural_body_first_global, self.structural_body_second_global
        )
        self._allocate_effort_rows()
        self._allocate_joint_frictions()
        self._allocate_unilaterals()
        self.world_lagged_velocity_residual = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.world_lagged_velocity_required = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)

    def _allocate_joint_rows(self) -> None:
        model = self.model
        joint_count = model.size.sum_of_num_joints
        joint_indices = np.arange(joint_count, dtype=np.int32)
        joint_world = model.joints.wid.numpy().astype(np.int32, copy=False)
        joint_body_first = model.joints.bid_B.numpy().astype(np.int32, copy=False)
        joint_body_second = model.joints.bid_F.numpy().astype(np.int32, copy=False)
        joint_dof_count = model.joints.num_dofs.numpy().astype(np.int32, copy=False)
        joint_dynamic_count = model.joints.num_dynamic_cts.numpy().astype(np.int32, copy=False)
        joint_effort_count = model.joints.num_effort_cts.numpy().astype(np.int32, copy=False)
        joint_dynamic_axis = model.joints.dynamic_cts_axis.numpy().astype(np.int32, copy=False)
        joint_structural_count = model.joints.num_kinematic_cts.numpy().astype(np.int32, copy=False)
        joint_dynamic_offset = model.joints.dynamic_cts_offset.numpy().astype(np.int32, copy=False)
        joint_dof_offset = model.joints.dofs_offset.numpy().astype(np.int32, copy=False)
        joint_dof_actuation_path = model.joints.dof_act_paths.numpy().astype(np.int32, copy=False)
        joint_structural_offset = model.joints.kinematic_cts_offset.numpy().astype(np.int32, copy=False)
        sparse_joint_offsets = self.jacobians._J_cts_joint_nzb_offsets.numpy().astype(np.int32, copy=False)
        sparse_dof_joint_offsets = self.jacobians._J_dofs_joint_nzb_offsets.numpy().astype(np.int32, copy=False)

        native_joint = np.repeat(joint_indices, joint_dynamic_count)
        native_offsets = _capacity_offsets(joint_dynamic_count)
        native_local = _segment_local_indices(joint_dynamic_count, native_offsets)
        native_axis = joint_dynamic_axis[joint_dynamic_offset[native_joint] + native_local]
        native_dof = joint_dof_offset[native_joint] + native_axis

        dof_joint = np.repeat(joint_indices, joint_dof_count)
        dof_offsets = _capacity_offsets(joint_dof_count)
        dof_axis = _segment_local_indices(joint_dof_count, dof_offsets)
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
        joint_dynamic_row_offset = _capacity_offsets(joint_dynamic_row_count)
        self.dynamic_row_count = int(joint_dynamic_row_offset[-1])
        dynamic_world = np.empty(self.dynamic_row_count, dtype=np.int32)
        dynamic_joint = np.empty(self.dynamic_row_count, dtype=np.int32)
        dynamic_first_global = np.empty(self.dynamic_row_count, dtype=np.int32)
        dynamic_second_global = np.empty(self.dynamic_row_count, dtype=np.int32)
        dynamic_value_index = np.full(self.dynamic_row_count, -1, dtype=np.int32)
        dynamic_dof_index = np.empty(self.dynamic_row_count, dtype=np.int32)
        dynamic_multiplier_index = np.full(self.dynamic_row_count, -1, dtype=np.int32)
        dynamic_sparse_first_index = np.full(self.dynamic_row_count, -1, dtype=np.int32)
        dynamic_sparse_second_index = np.empty(self.dynamic_row_count, dtype=np.int32)
        dynamic_uses_dof_jacobian = np.ones(self.dynamic_row_count, dtype=bool)

        native_target = joint_dynamic_row_offset[native_joint] + native_local
        extra_offsets = _capacity_offsets(extra_count)
        extra_local = _segment_local_indices(extra_count, extra_offsets)
        extra_target = joint_dynamic_row_offset[extra_joint] + joint_dynamic_count[extra_joint] + extra_local
        for target, source_joint in ((native_target, native_joint), (extra_target, extra_joint)):
            dynamic_world[target] = joint_world[source_joint]
            dynamic_joint[target] = source_joint
            dynamic_first_global[target] = joint_body_first[source_joint]
            dynamic_second_global[target] = joint_body_second[source_joint]
        dynamic_value_index[native_target] = joint_dynamic_offset[native_joint] + native_local
        dynamic_dof_index[native_target] = native_dof
        dynamic_dof_index[extra_target] = extra_dof
        dynamic_multiplier_index[native_target] = joint_dynamic_offset[native_joint] + native_local
        native_sparse_first = sparse_joint_offsets[native_joint] + joint_dof_count[native_joint] + native_local
        dynamic_sparse_first_index[native_target] = np.where(
            joint_body_first[native_joint] >= 0, native_sparse_first, -1
        )
        dynamic_sparse_second_index[native_target] = sparse_joint_offsets[native_joint] + native_local
        extra_sparse_first = sparse_dof_joint_offsets[extra_joint] + joint_dof_count[extra_joint] + extra_axis
        dynamic_sparse_first_index[extra_target] = np.where(joint_body_first[extra_joint] >= 0, extra_sparse_first, -1)
        dynamic_sparse_second_index[extra_target] = sparse_dof_joint_offsets[extra_joint] + extra_axis
        dynamic_uses_dof_jacobian[native_target] = False

        adapter_world_dynamic_count = np.bincount(dynamic_world, minlength=self.num_worlds).astype(np.int32, copy=False)
        dynamic_row_offsets = _capacity_offsets(adapter_world_dynamic_count)
        self.world_dynamic_row_offset = self._device_array(dynamic_row_offsets[:-1], self.device)
        self.world_dynamic_row_count = self._device_array(adapter_world_dynamic_count, self.device)
        self.dynamic_row_world = self._device_array(dynamic_world, self.device)
        self.dynamic_row_joint = self._device_array(dynamic_joint, self.device)
        self.dynamic_body_first_global = self._device_array(dynamic_first_global, self.device)
        self.dynamic_body_second_global = self._device_array(dynamic_second_global, self.device)
        self.dynamic_value_index = self._device_array(dynamic_value_index, self.device)
        self.dynamic_dof_index = self._device_array(dynamic_dof_index, self.device)
        self.dynamic_multiplier_index = self._device_array(dynamic_multiplier_index, self.device)
        self.dynamic_sparse_first_index = self._device_array(dynamic_sparse_first_index, self.device)
        self.dynamic_sparse_second_index = self._device_array(dynamic_sparse_second_index, self.device)
        self.dynamic_uses_dof_jacobian = wp.array(dynamic_uses_dof_jacobian, dtype=wp.bool, device=self.device)
        self.joint_dynamic_row_offset = self._device_array(joint_dynamic_row_offset[:-1], self.device)
        self.joint_dynamic_row_count = self._device_array(joint_dynamic_row_count, self.device)
        self.dynamic_jacobian_first = wp.zeros(self.dynamic_row_count, dtype=vec6f, device=self.device)
        self.dynamic_jacobian_second = wp.zeros(self.dynamic_row_count, dtype=vec6f, device=self.device)
        self.dynamic_effective_inertia = wp.zeros(self.dynamic_row_count, dtype=wp.float32, device=self.device)
        self.dynamic_free_velocity = wp.zeros(self.dynamic_row_count, dtype=wp.float32, device=self.device)

        structural_joint = np.repeat(joint_indices, joint_structural_count)
        structural_offsets = _capacity_offsets(joint_structural_count)
        structural_local = _segment_local_indices(joint_structural_count, structural_offsets)
        structural_value_index = joint_structural_offset[structural_joint] + structural_local
        self.structural_row_count = int(structural_offsets[-1])
        if not np.array_equal(structural_value_index, np.arange(self.structural_row_count, dtype=np.int32)):
            raise RuntimeError("LOX structural rows must use Kamino's canonical kinematic-row order.")
        world_structural_count = model.info.num_joint_kinematic_cts.numpy().astype(np.int32, copy=False)
        if self.structural_row_count != int(world_structural_count.sum()):
            raise RuntimeError("LOX structural row count must match Kamino's kinematic constraints.")
        if any(
            array.shape[0] != self.structural_row_count
            for array in (self.data.joints.r_j, self.data.joints.lambda_kin_j)
        ):
            raise RuntimeError("Kamino structural residual and reaction arrays must use canonical row storage.")
        self.world_structural_row_offset = model.info.joint_kinematic_cts_offset
        self.world_structural_row_count = model.info.num_joint_kinematic_cts
        structural_world = joint_world[structural_joint]
        structural_first_global = joint_body_first[structural_joint]
        structural_second_global = joint_body_second[structural_joint]
        adjacent_body_count = np.where(joint_body_first >= 0, 2, 1)
        structural_sparse_start = sparse_joint_offsets + np.where(
            joint_dynamic_count > 0, adjacent_body_count * joint_dof_count, 0
        )
        structural_sparse_first_index = np.where(
            structural_first_global >= 0,
            structural_sparse_start[structural_joint] + joint_structural_count[structural_joint] + structural_local,
            -1,
        )
        structural_sparse_second_index = structural_sparse_start[structural_joint] + structural_local
        self.structural_row_world = self._device_array(structural_world, self.device)
        self.structural_body_first_global = self._device_array(structural_first_global, self.device)
        self.structural_body_second_global = self._device_array(structural_second_global, self.device)
        self.structural_sparse_first_index = self._device_array(structural_sparse_first_index, self.device)
        self.structural_sparse_second_index = self._device_array(structural_sparse_second_index, self.device)
        self.structural_jacobian_first = wp.zeros(self.structural_row_count, dtype=vec6f, device=self.device)
        self.structural_jacobian_second = wp.zeros(self.structural_row_count, dtype=vec6f, device=self.device)
        self.structural_residual = self.data.joints.r_j
        self.structural_reaction = self.data.joints.lambda_kin_j
        self.structural_candidate_twist = wp.empty(model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        self.structural_candidate_residual = wp.zeros(self.structural_row_count, dtype=wp.float32, device=self.device)
        self.structural_proximal_defect = wp.zeros(self.structural_row_count, dtype=wp.float32, device=self.device)
        self.structural_residual_velocity_scratch = wp.zeros(
            self.structural_row_count, dtype=wp.float32, device=self.device
        )
        self.joint_coordinate_scratch = wp.zeros(
            model.size.sum_of_num_joint_coords, dtype=wp.float32, device=self.device
        )
        self.joint_velocity_scratch = wp.zeros(model.size.sum_of_num_joint_dofs, dtype=wp.float32, device=self.device)
        self.world_structural_residual = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.world_projected_structural_residual = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.structural_effective_mass = wp.zeros(self.structural_row_count, dtype=wp.float32, device=self.device)
        self.structural_penalty = wp.zeros(self.structural_row_count, dtype=wp.float32, device=self.device)

        static_constraint_count = np.zeros(model.size.sum_of_num_bodies, dtype=np.int32)
        structural_joints = joint_indices[joint_structural_count > 0]
        structural_first = joint_body_first[structural_joints]
        structural_second = joint_body_second[structural_joints]
        np.add.at(static_constraint_count, structural_first[structural_first >= 0], 1)
        second_mask = (structural_second >= 0) & (structural_second != structural_first)
        np.add.at(static_constraint_count, structural_second[second_mask], 1)
        self.static_body_constraint_count = wp.array(static_constraint_count, dtype=wp.int32, device=self.device)
        self.body_constraint_count = wp.zeros(model.size.sum_of_num_bodies, dtype=wp.int32, device=self.device)

        self._dynamic_row_world_host = dynamic_world
        self._dynamic_row_joint_host = dynamic_joint
        self._dynamic_dof_index_host = dynamic_dof_index
        self._dynamic_body_first_host = dynamic_first_global
        self._dynamic_body_second_host = dynamic_second_global

    def _allocate_effort_rows(self) -> None:
        """Allocate finite actuator corrections without changing projection incidence."""
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
        dynamic_world = self._dynamic_row_world_host
        dynamic_joint = self._dynamic_row_joint_host
        dynamic_dof = self._dynamic_dof_index_host
        dynamic_first = self._dynamic_body_first_host
        dynamic_second = self._dynamic_body_second_host

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

        self.effort_capacity = effort_dynamic_row.size
        self.has_bounded_effort = self.effort_capacity > 0
        dynamic_effort_index = np.full(self.dynamic_row_count, -1, dtype=np.int32)
        dynamic_effort_index[effort_dynamic_row] = np.arange(self.effort_capacity, dtype=np.int32)
        if not self.has_bounded_effort:
            self.dynamic_effort_index = self._device_array(dynamic_effort_index, self.device)
            self.effort_world = None
            self.effort_dynamic_row_index = None
            self.effort_value_index = self._device_array((), self.device)
            self.body_effort_offset = None
            self.body_effort_index = None
            self.body_effort_side = None
            self.effort_intercept = wp.zeros(0, dtype=wp.float32, device=self.device)
            self.effort_slope = wp.zeros(0, dtype=wp.float32, device=self.device)
            self.effort_impulse_bound = wp.zeros(0, dtype=wp.float32, device=self.device)
            self.effort_raw_impulse = None
            self.effort_counter_applied = None
            self.effort_counter_next = None
            self.effort_net_applied = None
            self.effort_net_target = None
            self.effort_velocity = None
            self.effort_residual = None
            self.world_effort_residual_max = None
            self.world_effort_defect_max = None
            return

        effort_indices = np.arange(self.effort_capacity, dtype=np.int32)
        effort_first = dynamic_first[effort_dynamic_row]
        effort_second = dynamic_second[effort_dynamic_row]
        incidence_body = np.concatenate((effort_first[effort_first >= 0], effort_second[effort_second >= 0]))
        incidence_effort = np.concatenate((effort_indices[effort_first >= 0], effort_indices[effort_second >= 0]))
        incidence_side = np.concatenate(
            (
                np.zeros(np.count_nonzero(effort_first >= 0), dtype=np.int32),
                np.ones(np.count_nonzero(effort_second >= 0), dtype=np.int32),
            )
        )
        incidence_order = np.lexsort((incidence_side, incidence_effort, incidence_body))
        incidence_body = incidence_body[incidence_order]
        incidence_effort = incidence_effort[incidence_order]
        incidence_side = incidence_side[incidence_order]
        body_counts = np.bincount(incidence_body, minlength=model.size.sum_of_num_bodies).astype(np.int32, copy=False)
        body_offsets = _capacity_offsets(body_counts)

        self.dynamic_effort_index = self._device_array(dynamic_effort_index, self.device)
        self.effort_world = self._device_array(effort_world, self.device)
        self.effort_dynamic_row_index = self._device_array(effort_dynamic_row, self.device)
        self.effort_value_index = self._device_array(effort_value_index, self.device)
        self.body_effort_offset = self._device_array(body_offsets, self.device)
        self.body_effort_index = self._device_array(incidence_effort, self.device)
        self.body_effort_side = self._device_array(incidence_side, self.device)
        self.effort_intercept = wp.zeros(self.effort_capacity, dtype=wp.float32, device=self.device)
        self.effort_slope = wp.zeros(self.effort_capacity, dtype=wp.float32, device=self.device)
        self.effort_impulse_bound = wp.zeros(self.effort_capacity, dtype=wp.float32, device=self.device)
        self.effort_raw_impulse = wp.zeros(self.effort_capacity, dtype=wp.float32, device=self.device)
        self.effort_counter_applied = wp.zeros(self.effort_capacity, dtype=wp.float32, device=self.device)
        self.effort_counter_next = wp.zeros(self.effort_capacity, dtype=wp.float32, device=self.device)
        self.effort_net_applied = wp.zeros(self.effort_capacity, dtype=wp.float32, device=self.device)
        self.effort_net_target = wp.zeros(self.effort_capacity, dtype=wp.float32, device=self.device)
        self.effort_velocity = wp.zeros(self.effort_capacity, dtype=wp.float32, device=self.device)
        self.effort_residual = wp.zeros(self.effort_capacity, dtype=wp.float32, device=self.device)
        self.world_effort_residual_max = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.world_effort_defect_max = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)

    def _allocate_joint_frictions(self) -> None:
        """Allocate one bounded scalar constraint for each frictional joint DOF."""
        source_model = self.model._model
        if source_model is None or source_model.joint_friction is None:
            friction_values = np.zeros(self.model.size.sum_of_num_joint_dofs, dtype=np.float32)
            self._joint_friction_force = wp.zeros(
                self.model.size.sum_of_num_joint_dofs,
                dtype=wp.float32,
                device=self.device,
            )
        else:
            friction_values = source_model.joint_friction.numpy()
            self._joint_friction_force = source_model.joint_friction
        if len(friction_values) != self.model.size.sum_of_num_joint_dofs:
            raise ValueError("Newton joint friction must contain one value per joint DOF.")
        if np.any(~np.isfinite(friction_values) | (friction_values < 0.0)):
            raise ValueError("Joint friction values must be finite and nonnegative.")

        joint_worlds = self.model.joints.wid.numpy().astype(np.int32, copy=False)
        joint_first = self.model.joints.bid_B.numpy().astype(np.int32, copy=False)
        joint_second = self.model.joints.bid_F.numpy().astype(np.int32, copy=False)
        joint_dof_offsets = self.model.joints.dofs_offset.numpy().astype(np.int32, copy=False)
        joint_dof_counts = self.model.joints.num_dofs.numpy().astype(np.int32, copy=False)
        joint_friction_counts = self.model.joints.num_friction_cts.numpy().astype(np.int32, copy=False)
        joint_friction_offsets = self.model.joints.friction_cts_offset.numpy().astype(np.int32, copy=False)
        sparse_offsets = self.jacobians._J_dofs_joint_nzb_offsets.numpy().astype(np.int32, copy=False)
        invalid_count = (joint_friction_counts != 0) & (joint_friction_counts != joint_dof_counts)
        if np.any(invalid_count):
            raise ValueError("Joint friction rows must match the joint DOF count.")

        body_vector_index = np.asarray(self.system.body_vector_index_host, dtype=np.int32)
        first_dynamic = (joint_first >= 0) & (body_vector_index[joint_first.clip(min=0)] >= 0)
        second_dynamic = (joint_second >= 0) & (body_vector_index[joint_second.clip(min=0)] >= 0)
        active_joint_mask = (joint_friction_counts > 0) & (first_dynamic | second_dynamic)
        active_joints = np.flatnonzero(active_joint_mask).astype(np.int32)
        row_joint = np.repeat(active_joints, joint_dof_counts[active_joints])
        active_counts = joint_dof_counts[active_joints]
        local_dof = _segment_local_indices(active_counts)
        row_world = joint_worlds[row_joint]
        order = np.argsort(row_world, kind="stable")
        row_joint = row_joint[order]
        local_dof = local_dof[order]
        row_world = row_world[order]
        counts = np.bincount(row_world, minlength=self.num_worlds).astype(np.int32, copy=False)
        offsets = _capacity_offsets(counts)
        self.friction_capacity = row_joint.size
        self.world_friction_offset = self._device_array(offsets[:-1], self.device)
        self.world_friction_count = self._device_array(counts, self.device)
        self.friction_world = self._device_array(row_world, self.device)
        self.friction_local = self._device_array(_segment_local_indices(counts, offsets), self.device)
        self.friction_dof_index = self._device_array(joint_dof_offsets[row_joint] + local_dof, self.device)
        self.friction_body_first = self._device_array(joint_first[row_joint], self.device)
        self.friction_body_second = self._device_array(joint_second[row_joint], self.device)
        self.friction_sparse_first_index = self._device_array(
            np.where(
                joint_first[row_joint] >= 0,
                sparse_offsets[row_joint] + joint_dof_counts[row_joint] + local_dof,
                -1,
            ),
            self.device,
        )
        self.friction_sparse_second_index = self._device_array(sparse_offsets[row_joint] + local_dof, self.device)
        self.friction_multiplier_index = self._device_array(joint_friction_offsets[row_joint] + local_dof, self.device)
        self.friction_jacobian_first = wp.zeros(self.friction_capacity, dtype=vec6f, device=self.device)
        self.friction_jacobian_second = wp.zeros(self.friction_capacity, dtype=vec6f, device=self.device)
        self.friction_impulse_bound = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.friction_reaction = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.friction_velocity = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.friction_residual = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.friction_physical_delassus = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.friction_projection_delassus = wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
        self.friction_acceleration_trial = (
            wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
            if self._allocate_acceleration_storage
            else None
        )
        self.friction_acceleration_previous = (
            wp.zeros(self.friction_capacity, dtype=wp.float32, device=self.device)
            if self._allocate_acceleration_storage
            else None
        )
        self.world_friction_residual_max = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)

    def _allocate_unilaterals(self) -> None:
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

        limit_offsets = _capacity_offsets(limit_capacities)
        contact_offsets = _capacity_offsets(contact_capacities)
        self.limit_capacity = int(limit_offsets[-1])
        self.contact_capacity = int(contact_offsets[-1])
        self._lagged_contact_block_count = 1
        self.world_limit_capacity = self._device_array(limit_capacities, self.device)
        self.world_limit_offset = self._device_array(limit_offsets[:-1], self.device)
        self.world_limit_count = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.world_contact_capacity = self._device_array(contact_capacities, self.device)
        self.world_contact_offset = self._device_array(contact_offsets[:-1], self.device)
        self.world_contact_count = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.limit_world = self._device_array(
            np.repeat(np.arange(self.num_worlds, dtype=np.int32), limit_capacities), self.device
        )
        self.limit_local = self._device_array(_segment_local_indices(limit_capacities, limit_offsets), self.device)
        self.contact_world = self._device_array(
            np.repeat(np.arange(self.num_worlds, dtype=np.int32), contact_capacities), self.device
        )
        self.contact_local = self._device_array(
            _segment_local_indices(contact_capacities, contact_offsets), self.device
        )

        self.limit_body_first = wp.full(self.limit_capacity, -1, dtype=wp.int32, device=self.device)
        self.limit_body_second = wp.full(self.limit_capacity, -1, dtype=wp.int32, device=self.device)
        self.limit_jacobian_first = wp.zeros(self.limit_capacity, dtype=vec6f, device=self.device)
        self.limit_jacobian_second = wp.zeros(self.limit_capacity, dtype=vec6f, device=self.device)
        self.limit_bias = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.limit_reaction = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.limit_velocity = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.limit_residual = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.limit_physical_delassus = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.limit_projection_delassus = wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
        self.limit_acceleration_trial = (
            wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
            if self._allocate_acceleration_storage
            else None
        )
        self.limit_acceleration_previous = (
            wp.zeros(self.limit_capacity, dtype=wp.float32, device=self.device)
            if self._allocate_acceleration_storage
            else None
        )
        self.world_limit_residual_max = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)

        self.contact_body_first = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.contact_body_second = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.contact_jacobian_first = wp.zeros(self.contact_capacity, dtype=mat36f, device=self.device)
        self.contact_jacobian_second = wp.zeros(self.contact_capacity, dtype=mat36f, device=self.device)
        self.contact_bias = wp.zeros(self.contact_capacity, dtype=wp.vec3f, device=self.device)
        self.contact_friction = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.contact_reaction = wp.zeros(self.contact_capacity, dtype=wp.vec3f, device=self.device)
        self.contact_velocity = wp.zeros(self.contact_capacity, dtype=wp.vec3f, device=self.device)
        self.contact_source_to_internal = wp.full(self.contact_capacity, -1, dtype=wp.int32, device=self.device)
        self.contact_residual = wp.zeros(self.contact_capacity, dtype=wp.float32, device=self.device)
        self.contact_physical_delassus = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.contact_prepared_delassus = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.contact_projection_delassus = wp.zeros(self.contact_capacity, dtype=wp.mat33f, device=self.device)
        self.contact_acceleration_trial = (
            wp.zeros(self.contact_capacity, dtype=wp.vec3f, device=self.device)
            if self._allocate_acceleration_storage
            else None
        )
        self.contact_acceleration_previous = (
            wp.zeros(self.contact_capacity, dtype=wp.vec3f, device=self.device)
            if self._allocate_acceleration_storage
            else None
        )
        self.world_physical_projection_status = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.world_jacobi_projection_status = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.projection_twist_delta = wp.zeros(self.model.size.sum_of_num_bodies, dtype=vec6f, device=self.device)
        self.world_contact_residual_max = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.projection_status = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.world_has_unilateral = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.body_has_unilateral = wp.zeros(self.model.size.sum_of_num_bodies, dtype=wp.int32, device=self.device)

    def begin_time_step(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        *,
        limit_stabilization_fraction: float,
        contact_stabilization_fraction: float,
        contact_dead_zone: float,
        impact_velocity_threshold: float,
        contact_recoverable_response: bool,
    ) -> None:
        """Freeze unilateral data and import begin-step velocities and reactions once."""
        self._time_step = time_step
        self._inverse_time_step = inverse_time_step
        self._limit_stabilization_fraction = limit_stabilization_fraction
        self._contact_stabilization_fraction = contact_stabilization_fraction
        self._contact_dead_zone = contact_dead_zone
        self._impact_velocity_threshold = impact_velocity_threshold
        self._contact_recoverable_response = contact_recoverable_response
        wp.copy(self.body_velocity_begin, self.data.bodies.u_i.view(dtype=vec6f))
        if self.dynamic_row_count > 0:
            wp.copy(self.joint_velocity_begin, self.data.joints.dq_j)
        self._update_unilaterals(
            time_step,
            limit_stabilization_fraction,
            contact_stabilization_fraction,
            contact_dead_zone,
            impact_velocity_threshold,
            contact_recoverable_response,
        )
        self._freeze_constraint_multiplicities()

    def _freeze_constraint_multiplicities(self) -> None:
        """Build combined entity incidence without a device-to-host readback."""
        wp.copy(self.body_constraint_count, self.static_body_constraint_count)
        self.body_has_unilateral.zero_()
        for entity_world, entity_local, world_count, body_first, body_second in (
            (
                self.friction_world,
                self.friction_local,
                self.world_friction_count,
                self.friction_body_first,
                self.friction_body_second,
            ),
            (
                self.limit_world,
                self.limit_local,
                self.world_limit_count,
                self.limit_body_first,
                self.limit_body_second,
            ),
            (
                self.contact_world,
                self.contact_local,
                self.world_contact_count,
                self.contact_body_first,
                self.contact_body_second,
            ),
        ):
            if entity_world.shape[0] > 0:
                wp.launch(
                    _accumulate_constraint_incidence,
                    dim=entity_world.shape[0],
                    inputs=[entity_world, entity_local, world_count, body_first, body_second],
                    outputs=[self.body_constraint_count, self.body_has_unilateral],
                    device=self.device,
                )

    def update(
        self,
        time_step: wp.array[wp.float32],
        joint_penalty_scale: float | wp.array[wp.float32] = 10.0,
        linearization_twist: wp.array[vec6f] | None = None,
        assemble_structural_penalty: bool = True,
    ) -> None:
        """Prepare one nonlinear evaluation and assemble the smooth primal system.

        :meth:`begin_time_step` must be called before the first nonlinear
        evaluation. It freezes the inertial and restitution velocities while
        this method continues to use the current body velocity for gyroscopic
        torque evaluation.

        If ``linearization_twist`` is omitted, the current body velocity is
        explicitly captured and used as the structural Newton linearization
        twist. Pass a packed zero twist for the first linearly implicit
        assembly about the begin-step pose.
        """
        if isinstance(joint_penalty_scale, wp.array):
            if joint_penalty_scale.ndim != 1 or joint_penalty_scale.dtype != wp.float32:
                raise ValueError("joint_penalty_scale must be a one-dimensional float32 array.")
            if joint_penalty_scale.shape[0] != self.num_worlds:
                raise ValueError("joint_penalty_scale must contain one entry per world.")
            world_joint_penalty_scale = joint_penalty_scale
        else:
            if joint_penalty_scale <= 0.0:
                raise ValueError("joint_penalty_scale must be positive.")
            self._uniform_joint_penalty_scale.fill_(joint_penalty_scale)
            world_joint_penalty_scale = self._uniform_joint_penalty_scale
        data = self.data
        if linearization_twist is None:
            wp.copy(self.body_linearization_twist, data.bodies.u_i.view(dtype=vec6f))
            linearization_twist = self.body_linearization_twist
        elif linearization_twist.shape[0] != self.model.size.sum_of_num_bodies:
            raise ValueError("linearization_twist must contain one entry per model body.")
        self.system.assemble_bodies(
            self.model.bodies.m_i,
            data.bodies.I_i,
            self.body_velocity_begin,
            data.bodies.u_i.view(dtype=vec6f),
            data.bodies.w_e_i.view(dtype=vec6f),
            data.bodies.w_a_i.view(dtype=vec6f),
            self.model.gravity.vector,
            time_step,
        )
        if self.dynamic_row_count > 0:
            wp.launch(
                _prepare_dynamic_rows,
                dim=self.dynamic_row_count,
                inputs=[
                    self.dynamic_row_world,
                    self.dynamic_uses_dof_jacobian,
                    self.dynamic_body_first_global,
                    self.dynamic_body_second_global,
                    self.dynamic_value_index,
                    self.dynamic_dof_index,
                    self.dynamic_sparse_first_index,
                    self.dynamic_sparse_second_index,
                    self._sparse_jacobian_data,
                    self._sparse_dof_jacobian_data,
                    data.joints.m_j,
                    data.joints.dq_b_j,
                    self.dynamic_effort_index,
                    self.effort_value_index,
                    data.joints.inv_m_a,
                    data.joints.dq_b_a,
                    self.model.joints.a_j,
                    self.model.joints.k_p_j,
                    self.model.joints.dof_act_types,
                    data.joints.dq_j,
                    self.joint_velocity_begin,
                    data.joints.tau_j,
                    self.model.joints.k_d_j,
                    self.model.joints.tau_j_max,
                    linearization_twist,
                    time_step,
                ],
                outputs=[
                    self.dynamic_jacobian_first,
                    self.dynamic_jacobian_second,
                    self.dynamic_effective_inertia,
                    self.dynamic_free_velocity,
                    self.effort_intercept,
                    self.effort_slope,
                    self.effort_impulse_bound,
                ],
                device=self.device,
            )
            self.system.add_dynamic_rows(
                self.dynamic_row_world,
                self.dynamic_body_first_global,
                self.dynamic_body_second_global,
                self.dynamic_jacobian_first,
                self.dynamic_jacobian_second,
                self.dynamic_effective_inertia,
                self.dynamic_free_velocity,
                prescribed_twist=self.body_velocity_begin,
            )
        if self.structural_row_count > 0:
            self.structural_proximal_defect.zero_()
            wp.launch(
                _prepare_structural_rows,
                dim=self.structural_row_count,
                inputs=[
                    self.structural_body_first_global,
                    self.structural_body_second_global,
                    self.structural_sparse_first_index,
                    self.structural_sparse_second_index,
                    self._sparse_jacobian_data,
                    self.model.bodies.inv_m_i,
                    data.bodies.inv_I_i,
                ],
                outputs=[
                    self.structural_jacobian_first,
                    self.structural_jacobian_second,
                    self.structural_effective_mass,
                ],
                device=self.device,
            )
            if self.dynamic_row_count > 0:
                wp.launch(
                    _include_dynamic_compliance_in_structural_effective_mass,
                    dim=self.model.size.sum_of_num_joints,
                    inputs=[
                        self.model.joints.bid_B,
                        self.model.joints.bid_F,
                        self.model.joints.kinematic_cts_offset,
                        self.model.joints.num_kinematic_cts,
                        self.joint_dynamic_row_offset,
                        self.joint_dynamic_row_count,
                        self.structural_jacobian_first,
                        self.structural_jacobian_second,
                        self.dynamic_jacobian_first,
                        self.dynamic_jacobian_second,
                        self.dynamic_effective_inertia,
                        self.model.bodies.inv_m_i,
                        data.bodies.inv_I_i,
                    ],
                    outputs=[self.structural_effective_mass],
                    device=self.device,
                )
            if assemble_structural_penalty:
                self.system.add_structural_rows(
                    self.structural_row_world,
                    self.structural_body_first_global,
                    self.structural_body_second_global,
                    self.structural_jacobian_first,
                    self.structural_jacobian_second,
                    self.structural_residual,
                    self.structural_reaction,
                    self.structural_effective_mass,
                    linearization_twist,
                    time_step,
                    world_joint_penalty_scale,
                    self.structural_penalty,
                    prescribed_twist=self.body_velocity_begin,
                )

    def _update_unilaterals(
        self,
        time_step: wp.array[wp.float32],
        limit_stabilization_fraction: float,
        contact_stabilization_fraction: float,
        contact_dead_zone: float,
        impact_velocity_threshold: float,
        contact_recoverable_response: bool,
        import_reactions: bool = True,
        update_counts: bool = True,
    ) -> None:
        if self.friction_capacity > 0:
            wp.launch(
                _prepare_joint_frictions,
                dim=self.friction_capacity,
                inputs=[
                    self.friction_world,
                    self.friction_body_first,
                    self.friction_body_second,
                    self.friction_dof_index,
                    self.friction_multiplier_index,
                    self.friction_sparse_first_index,
                    self.friction_sparse_second_index,
                    self._sparse_dof_jacobian_data,
                    self._joint_friction_force,
                    self.data.joints.lambda_f_j,
                    self.body_velocity_begin,
                    time_step,
                    import_reactions,
                ],
                outputs=[
                    self.friction_jacobian_first,
                    self.friction_jacobian_second,
                    self.friction_impulse_bound,
                    self.friction_reaction,
                    self.friction_velocity,
                ],
                device=self.device,
            )

        if self.limit_capacity > 0 and self.limits is not None:
            if update_counts:
                wp.launch(
                    _copy_clamped_world_counts,
                    dim=self.num_worlds,
                    inputs=[self.limits.world_active_limits, self.world_limit_capacity],
                    outputs=[self.world_limit_count],
                    device=self.device,
                )
            wp.launch(
                _prepare_limits,
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
                    self.world_limit_capacity,
                    self.world_limit_offset,
                    self.system.body_vector_index,
                    self._sparse_limit_offsets,
                    self._sparse_jacobian_data,
                    time_step,
                    limit_stabilization_fraction,
                    import_reactions,
                ],
                outputs=[
                    self.limit_body_first,
                    self.limit_body_second,
                    self.limit_jacobian_first,
                    self.limit_jacobian_second,
                    self.limit_bias,
                    self.limit_reaction,
                    self.limit_velocity,
                ],
                device=self.device,
            )
        elif update_counts:
            self.world_limit_count.zero_()

        if self.contact_capacity > 0 and self.contacts is not None:
            if update_counts:
                self.world_contact_count.zero_()
            wp.launch(
                _prepare_contacts,
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
                    self.contacts.reaction,
                    self.data.bodies.q_i,
                    self.body_velocity_begin,
                    self.world_contact_capacity,
                    self.world_contact_count,
                    self.world_contact_offset,
                    self.system.body_vector_index,
                    time_step,
                    contact_stabilization_fraction,
                    contact_dead_zone,
                    impact_velocity_threshold,
                    contact_recoverable_response,
                    import_reactions,
                    update_counts,
                    self.contact_source_to_internal,
                ],
                outputs=[
                    self.contact_body_first,
                    self.contact_body_second,
                    self.contact_jacobian_first,
                    self.contact_jacobian_second,
                    self.contact_bias,
                    self.contact_friction,
                    self.contact_reaction,
                    self.contact_velocity,
                ],
                device=self.device,
            )
        elif update_counts:
            self.world_contact_count.zero_()

        if self.limit_capacity > 0:
            wp.launch(
                _clear_inactive_limits,
                dim=self.limit_capacity,
                inputs=[self.limit_world, self.limit_local, self.world_limit_count],
                outputs=[
                    self.limit_body_first,
                    self.limit_body_second,
                    self.limit_reaction,
                    self.limit_velocity,
                ],
                device=self.device,
            )
        if self.contact_capacity > 0:
            wp.launch(
                _clear_inactive_contacts,
                dim=self.contact_capacity,
                inputs=[self.contact_world, self.contact_local, self.world_contact_count],
                outputs=[
                    self.contact_body_first,
                    self.contact_body_second,
                    self.contact_reaction,
                    self.contact_velocity,
                ],
                device=self.device,
            )

        if update_counts:
            wp.launch(
                _mark_worlds_with_unilaterals,
                dim=self.num_worlds,
                inputs=[self.world_contact_count, self.world_limit_count, self.world_friction_count],
                outputs=[self.world_has_unilateral],
                device=self.device,
            )

    def _evaluate_structural_candidate_pose_residual(
        self,
        time_step: wp.array[wp.float32],
        candidate_twist: wp.array[vec6f],
        linearization_twist: wp.array[vec6f],
        world_active: wp.array[wp.bool],
        candidate_residual: wp.array[wp.float32],
    ) -> None:
        """Evaluate exact structural residuals after integrating a candidate twist."""
        wp.launch(
            self._evaluate_candidate_structural_residual,
            dim=self.model.size.sum_of_num_joints,
            inputs=[
                time_step,
                self.model.joints.wid,
                self.model.joints.dof_type,
                self.model.joints.coords_offset,
                self.model.joints.dofs_offset,
                self.model.joints.kinematic_cts_offset,
                self.model.joints.bid_B,
                self.model.joints.bid_F,
                self.model.joints.B_r_Bj,
                self.model.joints.F_r_Fj,
                self.model.joints.X_Bj,
                self.model.joints.X_Fj,
                self.data.bodies.q_i,
                candidate_twist,
                linearization_twist,
                self.data.joints.q_j_p,
                world_active,
            ],
            outputs=[
                candidate_residual,
                self.structural_residual_velocity_scratch,
                self.joint_coordinate_scratch,
                self.joint_velocity_scratch,
            ],
            device=self.device,
        )

    def update_structural_multipliers_from_twist(
        self,
        time_step: wp.array[wp.float32],
        structural_tolerance: float,
        linearization_twist: wp.array[vec6f],
        global_twist: wp.array[vec6f],
        projected_twist: wp.array[vec6f],
        world_active: wp.array[wp.bool],
        projected_fraction: float = 0.0,
    ) -> None:
        """Update structural multipliers from exact candidate-pose residuals."""
        if not structural_tolerance > 0.0:
            raise ValueError("Structural tolerance must be positive.")
        if linearization_twist.shape[0] != self.model.size.sum_of_num_bodies:
            raise ValueError("linearization_twist must contain one entry per body.")
        if global_twist.shape[0] != self.model.size.sum_of_num_bodies:
            raise ValueError("global_twist must contain one entry per body.")
        if projected_twist.shape[0] != self.model.size.sum_of_num_bodies:
            raise ValueError("projected_twist must contain one entry per body.")
        if world_active.shape[0] != self.num_worlds:
            raise ValueError("world_active must contain one entry per world.")
        if not 0.0 <= projected_fraction <= 1.0:
            raise ValueError("projected_fraction must be in [0, 1].")
        self.world_structural_residual.zero_()
        self.world_projected_structural_residual.zero_()
        if self.structural_row_count == 0:
            return
        wp.launch(
            _blend_structural_candidate_twist,
            dim=self.model.size.sum_of_num_bodies,
            inputs=[global_twist, projected_twist, projected_fraction],
            outputs=[self.structural_candidate_twist],
            device=self.device,
        )
        if self.joint_proximal_relaxation > 0.0:
            self._evaluate_structural_candidate_pose_residual(
                time_step,
                self.structural_candidate_twist,
                linearization_twist,
                world_active,
                self.structural_candidate_residual,
            )
        wp.launch(
            _update_structural_multipliers_from_candidate_rows,
            dim=self.structural_row_count,
            inputs=[
                time_step,
                structural_tolerance,
                self.structural_row_world,
                world_active,
                self.projection_status,
                self.structural_body_first_global,
                self.structural_body_second_global,
                self.structural_jacobian_first,
                self.structural_jacobian_second,
                self.structural_candidate_residual,
                self.structural_proximal_defect,
                self.structural_residual,
                self.structural_penalty,
                linearization_twist,
                self.structural_candidate_twist,
                self.joint_proximal_relaxation,
                self.system.body_vector_index,
                self.structural_reaction,
            ],
            outputs=[
                self.world_structural_residual,
                self.system.right_hand_side,
            ],
            device=self.device,
        )
        wp.launch(
            _reduce_structural_candidate_residual,
            dim=self.structural_row_count,
            inputs=[
                structural_tolerance,
                self.structural_row_world,
                world_active,
                self.projection_status,
                self.structural_body_first_global,
                self.structural_body_second_global,
                self.structural_candidate_residual,
                self.system.body_vector_index,
            ],
            outputs=[self.world_projected_structural_residual],
            device=self.device,
        )

    def evaluate_lagged_velocity_consistency(
        self,
        velocity_tolerance: float,
        global_twist: wp.array[vec6f],
        projected_twist_previous: wp.array[vec6f],
        world_active: wp.array[wp.bool],
    ) -> None:
        """Evaluate prior projected constraint velocities using the current global solve."""
        if velocity_tolerance <= 0.0:
            raise ValueError("Velocity tolerance must be positive.")
        body_count = self.model.size.sum_of_num_bodies
        if global_twist.shape[0] != body_count or projected_twist_previous.shape[0] != body_count:
            raise ValueError("Twist arrays must contain one entry per packed body.")
        if world_active.shape[0] != self.num_worlds:
            raise ValueError("world_active must contain one entry per world.")

        self.world_lagged_velocity_residual.zero_()
        self.world_lagged_velocity_required.zero_()
        inverse_tolerance = 1.0 / velocity_tolerance

        scalar_rows = (
            (
                self.dynamic_row_count,
                self.dynamic_row_world,
                self.dynamic_body_first_global,
                self.dynamic_body_second_global,
                self.dynamic_jacobian_first,
                self.dynamic_jacobian_second,
            ),
            (
                self.structural_row_count,
                self.structural_row_world,
                self.structural_body_first_global,
                self.structural_body_second_global,
                self.structural_jacobian_first,
                self.structural_jacobian_second,
            ),
            (
                self.friction_capacity,
                self.friction_world,
                self.friction_body_first,
                self.friction_body_second,
                self.friction_jacobian_first,
                self.friction_jacobian_second,
            ),
            (
                self.limit_capacity,
                self.limit_world,
                self.limit_body_first,
                self.limit_body_second,
                self.limit_jacobian_first,
                self.limit_jacobian_second,
            ),
        )
        if self.contact_capacity == 0:
            for row_count, row_world, body_first, body_second, jacobian_first, jacobian_second in scalar_rows:
                if row_count <= 0:
                    continue
                wp.launch(
                    _evaluate_lagged_scalar_velocity_consistency,
                    dim=row_count,
                    inputs=[
                        row_world,
                        body_first,
                        body_second,
                        jacobian_first,
                        jacobian_second,
                        world_active,
                        global_twist,
                        projected_twist_previous,
                        inverse_tolerance,
                    ],
                    outputs=[self.world_lagged_velocity_required, self.world_lagged_velocity_residual],
                    device=self.device,
                )
        if self.contact_capacity > 0:
            wp.launch(
                _evaluate_lagged_contact_velocity_consistency,
                dim=(self.num_worlds, self._lagged_contact_block_count, _LAGGED_CONTACT_BLOCK_DIM),
                block_dim=_LAGGED_CONTACT_BLOCK_DIM,
                inputs=[
                    self.world_dynamic_row_offset,
                    self.world_dynamic_row_count,
                    self.dynamic_body_first_global,
                    self.dynamic_body_second_global,
                    self.dynamic_jacobian_first,
                    self.dynamic_jacobian_second,
                    self.world_structural_row_offset,
                    self.world_structural_row_count,
                    self.structural_body_first_global,
                    self.structural_body_second_global,
                    self.structural_jacobian_first,
                    self.structural_jacobian_second,
                    self.world_friction_offset,
                    self.world_friction_count,
                    self.friction_body_first,
                    self.friction_body_second,
                    self.friction_jacobian_first,
                    self.friction_jacobian_second,
                    self.world_limit_offset,
                    self.world_limit_count,
                    self.limit_body_first,
                    self.limit_body_second,
                    self.limit_jacobian_first,
                    self.limit_jacobian_second,
                    self.world_contact_offset,
                    self.world_contact_count,
                    self.contact_body_first,
                    self.contact_body_second,
                    self.contact_jacobian_first,
                    self.contact_jacobian_second,
                    world_active,
                    global_twist,
                    projected_twist_previous,
                    inverse_tolerance,
                    self._lagged_contact_block_count,
                ],
                outputs=[self.world_lagged_velocity_required, self.world_lagged_velocity_residual],
                device=self.device,
            )

    def reset_structural_multipliers(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear persistent structural multiplier warm starts."""
        if self.structural_row_count == 0:
            return
        if world_mask is None:
            self.structural_reaction.zero_()
        else:
            wp.launch(
                _reset_structural_multipliers_masked,
                dim=self.structural_row_count,
                inputs=[self.structural_row_world, world_mask, self.structural_reaction],
                device=self.device,
            )

    def reset_effort_counters(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear finite actuator correction warm starts and diagnostics."""
        if not self.has_bounded_effort:
            return
        if world_mask is None:
            self.effort_intercept.zero_()
            self.effort_slope.zero_()
            self.effort_impulse_bound.zero_()
            self.effort_raw_impulse.zero_()
            self.effort_counter_applied.zero_()
            self.effort_counter_next.zero_()
            self.effort_net_applied.zero_()
            self.effort_net_target.zero_()
            self.effort_velocity.zero_()
            self.effort_residual.zero_()
            self.world_effort_residual_max.zero_()
            self.world_effort_defect_max.zero_()
            return

        wp.launch(
            _reset_effort_rows_masked,
            dim=self.effort_capacity,
            inputs=[self.effort_world, world_mask],
            outputs=[
                self.effort_intercept,
                self.effort_slope,
                self.effort_impulse_bound,
                self.effort_raw_impulse,
                self.effort_counter_applied,
                self.effort_counter_next,
                self.effort_net_applied,
                self.effort_net_target,
                self.effort_velocity,
                self.effort_residual,
            ],
            device=self.device,
        )
        wp.launch(
            _reset_effort_worlds_masked,
            dim=self.num_worlds,
            inputs=[world_mask],
            outputs=[self.world_effort_residual_max, self.world_effort_defect_max],
            device=self.device,
        )

    def reset_friction_reactions(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Clear persistent joint-friction impulse warm starts."""
        if self.friction_capacity == 0:
            return
        if world_mask is None:
            self.friction_reaction.zero_()
            return
        wp.launch(
            _reset_friction_reactions_masked,
            dim=self.friction_capacity,
            inputs=[self.friction_world, world_mask],
            outputs=[self.friction_reaction],
            device=self.device,
        )

    def promote_effort_counters(self, world_active: wp.array[wp.bool]) -> None:
        """Promote the pending correction before solving its candidate system."""
        if not self.has_bounded_effort:
            return
        wp.launch(
            _promote_effort_counters,
            dim=self.effort_capacity,
            inputs=[self.effort_world, world_active, self.effort_counter_next],
            outputs=[self.effort_counter_applied],
            device=self.device,
        )

    def update_effort_counters(
        self,
        time_step: wp.array[wp.float32],
        velocity_tolerance: float,
        projected_twist: wp.array[vec6f],
        world_active: wp.array[wp.bool],
    ) -> None:
        """Evaluate finite actuator defects without applying the next correction."""
        if not self.has_bounded_effort:
            return
        if velocity_tolerance <= 0.0:
            raise ValueError("Velocity tolerance must be positive.")
        wp.launch(
            _clear_active_effort_residuals,
            dim=self.num_worlds,
            inputs=[world_active],
            outputs=[self.world_effort_residual_max, self.world_effort_defect_max],
            device=self.device,
        )
        wp.launch(
            _update_effort_counters,
            dim=self.effort_capacity,
            inputs=[
                self.effort_world,
                self.effort_dynamic_row_index,
                world_active,
                self.dynamic_body_first_global,
                self.dynamic_body_second_global,
                self.dynamic_jacobian_first,
                self.dynamic_jacobian_second,
                self.dynamic_effective_inertia,
                projected_twist,
                self.effort_intercept,
                self.effort_slope,
                self.effort_impulse_bound,
                self.effort_counter_applied,
                time_step,
                velocity_tolerance,
            ],
            outputs=[
                self.effort_counter_next,
                self.effort_raw_impulse,
                self.effort_net_applied,
                self.effort_net_target,
                self.effort_velocity,
                self.effort_residual,
                self.world_effort_residual_max,
                self.world_effort_defect_max,
            ],
            device=self.device,
        )

    def scale_structural_multipliers(self, scale: float) -> None:
        """Scale structural multipliers before using them as a warm start."""
        if not 0.0 <= scale <= 1.0:
            raise ValueError("scale must be in [0, 1].")
        if self.structural_row_count == 0:
            return
        wp.launch(
            _scale_structural_reactions,
            dim=self.structural_row_count,
            inputs=[scale, self.structural_reaction],
            device=self.device,
        )

    def write_outputs(
        self,
        time_step: wp.array[wp.float32],
        inverse_time_step: wp.array[wp.float32],
        body_velocity: wp.array[vec6f] | None = None,
        *,
        write_body_velocity: bool = True,
    ) -> None:
        """Write force-valued reactions and optionally the solved body velocities."""
        if body_velocity is None:
            body_velocity = self.projected_twist
        if body_velocity.shape[0] != self.model.size.sum_of_num_bodies:
            raise ValueError("body_velocity must contain one entry per model body.")
        if write_body_velocity:
            wp.copy(self.data.bodies.u_i.view(dtype=vec6f), body_velocity)
        body_data = self.data.bodies
        joint_wrench = body_data.w_j_i.view(dtype=vec6f)
        limit_wrench = body_data.w_l_i.view(dtype=vec6f)
        contact_wrench = body_data.w_c_i.view(dtype=vec6f)
        body_data.w_j_i.zero_()
        body_data.w_l_i.zero_()
        body_data.w_c_i.zero_()
        if self.dynamic_row_count > 0:
            if self.has_bounded_effort:
                wp.launch(
                    _write_dynamic_outputs_with_effort,
                    dim=self.dynamic_row_count,
                    inputs=[
                        inverse_time_step,
                        self.dynamic_row_world,
                        self.dynamic_multiplier_index,
                        self.dynamic_effort_index,
                        self.dynamic_body_first_global,
                        self.dynamic_body_second_global,
                        self.dynamic_jacobian_first,
                        self.dynamic_jacobian_second,
                        self.dynamic_effective_inertia,
                        self.dynamic_free_velocity,
                        self.effort_counter_applied,
                        self.effort_net_applied,
                        self.effort_value_index,
                        body_velocity,
                    ],
                    outputs=[self.data.joints.lambda_dyn_j, self.data.joints.lambda_tau_j, joint_wrench],
                    device=self.device,
                )
            else:
                wp.launch(
                    _write_dynamic_outputs,
                    dim=self.dynamic_row_count,
                    inputs=[
                        inverse_time_step,
                        self.dynamic_row_world,
                        self.dynamic_multiplier_index,
                        self.dynamic_body_first_global,
                        self.dynamic_body_second_global,
                        self.dynamic_jacobian_first,
                        self.dynamic_jacobian_second,
                        self.dynamic_effective_inertia,
                        self.dynamic_free_velocity,
                        body_velocity,
                    ],
                    outputs=[self.data.joints.lambda_dyn_j, joint_wrench],
                    device=self.device,
                )
        if self.friction_capacity > 0:
            wp.launch(
                _write_friction_outputs,
                dim=self.friction_capacity,
                inputs=[
                    inverse_time_step,
                    self.friction_world,
                    self.friction_multiplier_index,
                    self.friction_body_first,
                    self.friction_body_second,
                    self.friction_jacobian_first,
                    self.friction_jacobian_second,
                    self.friction_reaction,
                ],
                outputs=[self.data.joints.lambda_f_j, joint_wrench],
                device=self.device,
            )
        if self.limit_capacity > 0 and self.limits is not None:
            wp.launch(
                _write_limit_outputs,
                dim=self.limits.model_max_limits_host,
                inputs=[
                    self.limits.model_active_limits,
                    self.limits.model_max_limits_host,
                    self.limits.wid,
                    self.limits.lid,
                    self.world_limit_capacity,
                    self.world_limit_offset,
                    inverse_time_step,
                    self.limit_body_first,
                    self.limit_body_second,
                    self.limit_jacobian_first,
                    self.limit_jacobian_second,
                    self.limit_reaction,
                    self.limit_velocity,
                ],
                outputs=[self.limits.reaction, self.limits.velocity, limit_wrench],
                device=self.device,
            )
        if self.contact_capacity > 0 and self.contacts is not None:
            wp.launch(
                _write_contact_outputs,
                dim=self.contacts.model_max_contacts_host,
                inputs=[
                    self.contacts.model_active_contacts,
                    self.contacts.model_max_contacts_host,
                    self.contacts.wid,
                    self.contact_source_to_internal,
                    inverse_time_step,
                    self.contact_body_first,
                    self.contact_body_second,
                    self.contact_jacobian_first,
                    self.contact_jacobian_second,
                    self.contact_reaction,
                    self.contact_velocity,
                ],
                outputs=[self.contacts.reaction, self.contacts.velocity, self.contacts.mode, contact_wrench],
                device=self.device,
            )
        if self.structural_row_count > 0:
            wp.launch(
                _accumulate_aligned_joint_wrenches,
                dim=self.structural_row_count,
                inputs=[
                    self.structural_body_first_global,
                    self.structural_body_second_global,
                    self.structural_jacobian_first,
                    self.structural_jacobian_second,
                    self.structural_reaction,
                ],
                outputs=[joint_wrench],
                device=self.device,
            )

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
