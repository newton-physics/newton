# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared data and kernels for Kamino dual forward-dynamics solvers."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import warp as wp

from ..core.size import SizeKamino


class WarmStartMode(IntEnum):
    """Warm-start source used by a Kamino dual solver."""

    NONE = -1
    """Start without cached solution data."""

    INTERNAL = 0
    """Initialize from the solver's internally cached solution."""

    CONTAINERS = 1
    """Initialize from external joint, limit, and contact containers."""

    @classmethod
    def from_string(cls, value: str) -> WarmStartMode:
        """Convert a string to a warm-start mode."""
        try:
            return cls[value.upper()]
        except KeyError as error:
            raise ValueError(
                f"Invalid WarmStartMode: {value}. Valid options are: {[mode.name for mode in cls]}"
            ) from error

    @staticmethod
    def parse_usd_attribute(value: str, context: dict[str, Any] | None = None) -> str:
        """Parse a warm-start option imported from USD."""
        del context
        if not isinstance(value, str):
            raise TypeError("Parser expects input of type 'str'.")
        value = value.lower().strip()
        if value not in {"none", "internal", "containers"}:
            raise ValueError(f"Warmstart parameter '{value}' is not a valid option.")
        return value


class DualSolution:
    """Constraint impulse and post-event velocity arrays for a dual solver."""

    def __init__(self, size: SizeKamino | None = None):
        self.lambdas: wp.array[wp.float32] | None = None
        """Constraint impulses, shape ``(sum_of_max_total_cts,)``."""
        self.v_plus: wp.array[wp.float32] | None = None
        """Post-event constraint-space velocities, shape ``(sum_of_max_total_cts,)``."""
        if size is not None:
            self.finalize(size)

    def finalize(self, size: SizeKamino) -> None:
        """Allocate solution arrays for a model size."""
        self.lambdas = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.v_plus = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)

    def zero(self) -> None:
        """Reset all solution arrays to zero."""
        self.lambdas.zero_()
        self.v_plus.zero_()


@wp.kernel
def warmstart_joint_constraints(
    model_time_dt: wp.array[wp.float32],
    joint_wid: wp.array[wp.int32],
    joint_num_dynamic_cts: wp.array[wp.int32],
    joint_num_kinematic_cts: wp.array[wp.int32],
    joint_dynamic_cts_offset_joint_cts: wp.array[wp.int32],
    joint_kinematic_cts_offset_joint_cts: wp.array[wp.int32],
    joint_dynamic_cts_offset_total_cts: wp.array[wp.int32],
    joint_kinematic_cts_offset_total_cts: wp.array[wp.int32],
    joint_lambda_j: wp.array[wp.float32],
    problem_P: wp.array[wp.float32],
    x_0: wp.array[wp.float32],
    y_0: wp.array[wp.float32],
    z_0: wp.array[wp.float32],
):
    """Initialize bilateral constraint iterates from cached joint reactions."""
    jid = wp.tid()
    wid_j = joint_wid[jid]
    num_dynamic_cts_j = joint_num_dynamic_cts[jid]
    num_kinematic_cts_j = joint_num_kinematic_cts[jid]
    dt = model_time_dt[wid_j]
    joint_dyn_cts_start = joint_dynamic_cts_offset_joint_cts[jid]
    joint_kin_cts_start = joint_kinematic_cts_offset_joint_cts[jid]
    dyn_cts_row_start_j = joint_dynamic_cts_offset_total_cts[jid]
    kin_cts_row_start_j = joint_kinematic_cts_offset_total_cts[jid]

    # Convert cached forces to preconditioned impulses. Joint constraints do
    # not cache a post-event velocity, so their dual iterate starts at zero.
    for j in range(num_dynamic_cts_j):
        P_j = problem_P[dyn_cts_row_start_j + j]
        lambda_j = (dt / P_j) * joint_lambda_j[joint_dyn_cts_start + j]
        x_0[dyn_cts_row_start_j + j] = lambda_j
        y_0[dyn_cts_row_start_j + j] = lambda_j
        z_0[dyn_cts_row_start_j + j] = 0.0
    for j in range(num_kinematic_cts_j):
        P_j = problem_P[kin_cts_row_start_j + j]
        lambda_j = (dt / P_j) * joint_lambda_j[joint_kin_cts_start + j]
        x_0[kin_cts_row_start_j + j] = lambda_j
        y_0[kin_cts_row_start_j + j] = lambda_j
        z_0[kin_cts_row_start_j + j] = 0.0


@wp.kernel
def warmstart_limit_constraints(
    model_time_dt: wp.array[wp.float32],
    model_info_total_cts_offset: wp.array[wp.int32],
    data_info_limit_cts_group_offset: wp.array[wp.int32],
    limit_model_num_active: wp.array[wp.int32],
    limit_wid: wp.array[wp.int32],
    limit_lid: wp.array[wp.int32],
    limit_reaction: wp.array[wp.float32],
    limit_velocity: wp.array[wp.float32],
    problem_P: wp.array[wp.float32],
    x_0: wp.array[wp.float32],
    y_0: wp.array[wp.float32],
    z_0: wp.array[wp.float32],
):
    """Initialize limit constraint iterates from cached limit data."""
    lid = wp.tid()
    if lid >= limit_model_num_active[0]:
        return

    wid = limit_wid[lid]
    vio_l = model_info_total_cts_offset[wid] + data_info_limit_cts_group_offset[wid] + limit_lid[lid]
    P_l = problem_P[vio_l]
    # Reactions are cached as forces and velocities in physical units.
    lambda_l = limit_reaction[lid] * model_time_dt[wid] / P_l
    v_plus_l = limit_velocity[lid] * P_l
    x_0[vio_l] = lambda_l
    y_0[vio_l] = lambda_l
    z_0[vio_l] = v_plus_l


@wp.kernel
def warmstart_contact_constraints(
    model_time_dt: wp.array[wp.float32],
    model_info_total_cts_offset: wp.array[wp.int32],
    data_info_contact_cts_group_offset: wp.array[wp.int32],
    contact_model_num_contacts: wp.array[wp.int32],
    contact_wid: wp.array[wp.int32],
    contact_cid: wp.array[wp.int32],
    contact_material: wp.array[wp.vec2f],
    contact_reaction: wp.array[wp.vec3f],
    contact_velocity: wp.array[wp.vec3f],
    problem_P: wp.array[wp.float32],
    x_0: wp.array[wp.float32],
    y_0: wp.array[wp.float32],
    z_0: wp.array[wp.float32],
):
    """Initialize contact constraint iterates from cached contact data."""
    cid = wp.tid()
    if cid >= contact_model_num_contacts[0]:
        return

    wid = contact_wid[cid]
    vio_k = model_info_total_cts_offset[wid] + data_info_contact_cts_group_offset[wid] + 3 * contact_cid[cid]
    P_k = problem_P[vio_k]
    lambda_k = contact_reaction[cid] * model_time_dt[wid] / P_k
    v_plus_k = contact_velocity[cid] * P_k
    mu_k = contact_material[cid][0]
    # Apply the De Saxce correction to recover the solver's dual variable.
    v_plus_k.z += mu_k * wp.sqrt(v_plus_k.x * v_plus_k.x + v_plus_k.y * v_plus_k.y)

    for k in range(3):
        x_0[vio_k + k] = lambda_k[k]
        y_0[vio_k + k] = lambda_k[k]
        z_0[vio_k + k] = v_plus_k[k]


@wp.kernel
def apply_dual_preconditioner_to_solution(
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    solution_lambdas: wp.array[wp.float32],
    solution_v_plus: wp.array[wp.float32],
):
    """Convert cached physical solution values to preconditioned solver units."""
    wid, tid = wp.tid()
    if tid >= problem_dim[wid]:
        return
    v_i = problem_vio[wid] + tid
    P_i = problem_P[v_i]
    # Constraint impulses and velocities scale inversely under dual preconditioning.
    solution_lambdas[v_i] /= P_i
    solution_v_plus[v_i] *= P_i
