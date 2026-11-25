# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility for legacy MuJoCo spring attributes."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ...exceptions import NewtonDeprecationWarning
from ...sim.enums import JointType
from ...sim.model import Model
from ...utils.deprecation import UnavailableAttribute

if TYPE_CHECKING:
    from ...sim.builder import ModelBuilder


def _authored_values(attr: ModelBuilder.CustomAttribute) -> dict[int, float]:
    values = attr.values or {}
    entries = values.items() if isinstance(values, dict) else enumerate(values)
    return {d: value for d, value in entries if value is not None}


def finalize_stiffness(builder: ModelBuilder, model: Model, attr: ModelBuilder.CustomAttribute) -> None:
    """Convert legacy stiffness and retain a warning alias to its canonical array."""
    if not hasattr(model, "mujoco"):
        model.mujoco = Model.AttributeNamespace("mujoco")
    model.mujoco._add_deprecated_alias(
        attr.name,
        lambda: model.joint_stiffness,
        "model.mujoco.dof_passive_stiffness is deprecated since Newton 1.8; use model.joint_stiffness instead.",
    )
    model._set_attribute_spec(
        attr.key, replace(model._attribute_spec("joint_stiffness"), deprecated=True, alias_of="joint_stiffness")
    )
    authored = _authored_values(attr)
    if not authored:
        return
    warnings.warn(
        f"{attr.key} builder inputs are deprecated since Newton 1.8; use joint_stiffness instead.",
        NewtonDeprecationWarning,
        stacklevel=4,
    )
    stiffness = model.joint_stiffness.numpy()
    for d, value in authored.items():
        if stiffness[d] != 0.0 and not np.isclose(stiffness[d], value, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Conflicting core joint spring value and deprecated {attr.key} at DOF {d}")
        stiffness[d] = value
    model.joint_stiffness.assign(stiffness)


def finalize_springref(builder: ModelBuilder, model: Model, attr: ModelBuilder.CustomAttribute) -> None:
    """Convert absolute MuJoCo spring references once, then use Newton rest coordinates."""
    ref_attr = builder.custom_attributes["mujoco:dof_ref"]
    refs = _authored_values(ref_attr)
    # The legacy layout equals joint_rest_q only without multi-coordinate joints or reference offsets.
    reasons = []
    if model.joint_coord_count != model.joint_dof_count:
        reasons.append("joints with more coordinates than DOFs (free, ball, distance, or rod)")
    if any(refs.values()) or ref_attr.default:
        reasons.append("nonzero mujoco:dof_ref")
    if not reasons:
        model.mujoco._add_deprecated_alias(
            attr.name,
            lambda: model.joint_rest_q,
            "model.mujoco.dof_springref is deprecated since Newton 1.8; use model.joint_rest_q instead.",
        )
        model._set_attribute_spec(
            attr.key, replace(model._attribute_spec("joint_rest_q"), deprecated=True, alias_of="joint_rest_q")
        )
    else:
        model.mujoco._removed_attributes[attr.name] = UnavailableAttribute(
            "model.mujoco.dof_springref is deprecated since Newton 1.8 and unavailable for models with "
            f"{' or '.join(reasons)}; use model.joint_rest_q (springref - ref, indexed by joint_q_start) instead."
        )
    authored = _authored_values(attr)
    if authored:
        warnings.warn(
            f"{attr.key} builder inputs are deprecated since Newton 1.8; "
            "use joint_rest_q (springref - ref, indexed by joint_q_start) instead.",
            NewtonDeprecationWarning,
            stacklevel=4,
        )
    entries = authored.copy()
    # A legacy stiffness input implies MuJoCo's default springref=0.
    for d in _authored_values(builder.custom_attributes["mujoco:dof_passive_stiffness"]):
        entries.setdefault(d, attr.default)
    if not entries:
        return
    rest = model.joint_rest_q.numpy()
    for j, kind in enumerate(builder.joint_type):
        if kind not in (JointType.REVOLUTE, JointType.PRISMATIC, JointType.D6):
            continue
        for i in range(sum(builder.joint_dof_dim[j])):
            d = builder.joint_qd_start[j] + i
            if d not in entries:
                continue
            q = builder.joint_q_start[j] + i
            core_authored = rest[q] != 0.0 or builder._joint_rest_q_authored[q]
            if d not in authored and core_authored:
                continue
            value = float(entries[d]) - float(refs.get(d, ref_attr.default))
            if core_authored and not np.isclose(rest[q], value, rtol=1e-6, atol=1e-7):
                raise ValueError(f"Conflicting core joint spring value and deprecated {attr.key} at DOF {d}")
            rest[q] = value
    model.joint_rest_q.assign(rest)
