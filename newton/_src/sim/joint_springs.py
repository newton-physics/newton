# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint spring import compatibility and solver support checks."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .enums import JointType

if TYPE_CHECKING:
    from .builder import ModelBuilder
    from .model import Model


def finalize_legacy_joint_spring(builder: ModelBuilder, model: Model, attr: ModelBuilder.CustomAttribute) -> None:
    """Convert authored legacy values into core arrays without exposing runtime aliases."""
    is_rest = attr.name == "dof_springref"
    authored = attr.values or {}
    authored = authored.items() if isinstance(authored, dict) else enumerate(authored)
    authored = {d: value for d, value in authored if value is not None}
    if authored:
        replacement = "joint_rest_q (springref - ref, indexed by joint_q_start)" if is_rest else "joint_stiffness"
        warnings.warn(
            f"{attr.key} builder inputs are deprecated since Newton 1.7; use {replacement} instead.",
            DeprecationWarning,
            stacklevel=4,
        )
    entries = authored.copy()
    if is_rest:
        # Legacy stiffness implies MuJoCo's default springref=0. A core spring
        # with only dof_ref authored must keep its Newton rest coordinate.
        stiffness = builder.custom_attributes["mujoco:dof_passive_stiffness"].values or {}
        stiffness = stiffness.items() if isinstance(stiffness, dict) else enumerate(stiffness)
        for d, value in stiffness:
            if value is not None:
                entries.setdefault(d, attr.default)
    if not entries:
        return
    target = model.joint_rest_q if is_rest else model.joint_stiffness
    values = target.numpy()
    coord = None
    ref = builder.custom_attributes.get("mujoco:dof_ref")
    if is_rest:
        coord = np.full(model.joint_dof_count, -1, dtype=np.int32)
        for j, kind in enumerate(builder.joint_type):
            if kind in (JointType.REVOLUTE, JointType.PRISMATIC, JointType.D6):
                start = builder.joint_qd_start[j]
                size = sum(builder.joint_dof_dim[j])
                coord[start : start + size] = np.arange(builder.joint_q_start[j], builder.joint_q_start[j] + size)
    for d, value in entries.items():
        q = int(coord[d]) if is_rest else d
        if q < 0:
            continue
        if d not in authored and values[q] != 0.0:
            continue  # An explicit core rest coordinate takes precedence over a legacy default.
        offset = 0.0
        if is_rest and ref is not None:
            authored_ref = ref.values or {}
            if isinstance(authored_ref, dict):
                offset = authored_ref.get(d, ref.default)
            else:
                offset = authored_ref[d] if d < len(authored_ref) else ref.default
            if offset is None:
                offset = ref.default
        canonical = float(value) - float(offset)
        if values[q] != 0.0 and not np.isclose(values[q], canonical, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Conflicting core joint spring value and deprecated {attr.key} at DOF {d}")
        values[q] = canonical
    target.assign(values)


def warn_unsupported_joint_springs(model: Model, solver_name: str) -> None:
    """Report passive springs outside the scalar joint types supported by the native solvers."""
    active_dofs = np.flatnonzero(model.joint_stiffness.numpy())
    if active_dofs.size == 0:
        return
    starts = model.joint_qd_start.numpy()
    joints = np.searchsorted(starts, active_dofs, side="right") - 1
    kinds = model.joint_type.numpy()[joints]
    unsupported = kinds[~np.isin(kinds, (JointType.REVOLUTE, JointType.PRISMATIC, JointType.D6))]
    if unsupported.size:
        warnings.warn(
            f"{solver_name} ignores passive springs on {JointType(int(unsupported[0])).name} joints; "
            "only REVOLUTE, PRISMATIC and D6 springs are supported.",
            stacklevel=3,
        )
