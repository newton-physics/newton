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
    if not attr.values:
        return
    is_rest = attr.name == "dof_springref"
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
    entries = attr.values.items() if isinstance(attr.values, dict) else enumerate(attr.values)
    for d, value in entries:
        if value is None:
            continue
        q = int(coord[d]) if is_rest else d
        if q < 0:
            continue
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
    stiffness = model.joint_stiffness.numpy()
    starts = model.joint_qd_start.numpy()
    for j, kind in enumerate(model.joint_type.numpy()):
        if kind not in (JointType.REVOLUTE, JointType.PRISMATIC, JointType.D6):
            if np.any(stiffness[starts[j] : starts[j + 1]] != 0.0):
                warnings.warn(
                    f"{solver_name} ignores passive springs on {JointType(int(kind)).name} joints; "
                    "only REVOLUTE, PRISMATIC and D6 springs are supported.",
                    stacklevel=3,
                )
                return
