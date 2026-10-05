# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compatibility for MuJoCo's deprecated passive joint spring attributes."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .enums import JointType
from .model import Model

if TYPE_CHECKING:
    from .builder import ModelBuilder


@wp.kernel
def _sync_rest_coordinates(
    coord: wp.array[wp.int32],
    ref: wp.array[float],
    rest: wp.array[float],
    legacy: wp.array[float],
    previous_rest: wp.array[float],
    previous_legacy: wp.array[float],
    legacy_authority: wp.array[wp.int32],
):
    d = wp.tid()
    q = coord[d]
    if q < 0:
        return
    offset = float(0.0)
    if ref:
        offset = ref[d]
    # Explicit canonical edits take precedence when both arrays changed.
    if rest[q] != previous_rest[d]:
        legacy_authority[d] = 0
    elif legacy[d] != previous_legacy[d]:
        legacy_authority[d] = 1
    if legacy_authority[d] != 0:
        rest[q] = legacy[d] - offset
    else:
        legacy[d] = rest[q] + offset
    previous_rest[d] = rest[q]
    previous_legacy[d] = legacy[d]


class _MuJoCoSpringNamespace(Model.AttributeNamespace):
    """Keep legacy Warp arrays writable while resolving scalar rest coordinates on notification."""

    @property
    def dof_passive_stiffness(self) -> wp.array:
        warnings.warn(
            "mujoco.dof_passive_stiffness is deprecated in Newton 1.7; use Model.joint_stiffness instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._model.joint_stiffness

    @dof_passive_stiffness.setter
    def dof_passive_stiffness(self, value) -> None:
        self.dof_passive_stiffness.assign(value)

    @property
    def dof_springref(self) -> wp.array:
        warnings.warn(
            "mujoco.dof_springref is deprecated in Newton 1.7; use Model.joint_rest_q in Newton coordinates "
            "(springref - ref), and notify the solver after editing legacy spring references.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._sync()
        return self._springref

    @dof_springref.setter
    def dof_springref(self, value) -> None:
        self.dof_springref.assign(value)

    def _sync(self) -> None:
        model = self._model
        if model.joint_dof_count:
            wp.launch(
                _sync_rest_coordinates,
                dim=model.joint_dof_count,
                inputs=[
                    self._coord,
                    getattr(self, "dof_ref", None),
                    model.joint_rest_q,
                    self._springref,
                    self._previous_rest,
                    self._previous_legacy,
                    self._legacy_authority,
                ],
                device=model.device,
                record_tape=False,
            )


def finalize_joint_springs(builder: ModelBuilder, model: Model) -> None:
    """Resolve legacy authored values and install deprecated aliases after custom attributes finalize."""
    namespace = getattr(model, "mujoco", None)
    if namespace is None or not any(
        name in builder.custom_attributes for name in ("mujoco:dof_passive_stiffness", "mujoco:dof_springref")
    ):
        return
    count = model.joint_dof_count
    coord = np.full(count, -1, dtype=np.int32)
    for j, kind in enumerate(builder.joint_type):
        if kind in (JointType.REVOLUTE, JointType.PRISMATIC, JointType.D6):
            start = builder.joint_qd_start[j]
            size = sum(builder.joint_dof_dim[j])
            coord[start : start + size] = np.arange(builder.joint_q_start[j], builder.joint_q_start[j] + size)
    ref = getattr(namespace, "dof_ref", None)
    offsets = ref.numpy() if ref is not None else np.zeros(count, dtype=np.float32)
    stiffness = model.joint_stiffness.numpy()
    rest = model.joint_rest_q.numpy()
    authority = np.zeros(count, dtype=np.int32)
    for name, target in (("dof_passive_stiffness", stiffness), ("dof_springref", rest)):
        attr = builder.custom_attributes.get("mujoco:" + name)
        if attr is None:
            continue
        values = attr.values or {}
        if values:
            replacement_name = "joint_stiffness" if name == "dof_passive_stiffness" else "joint_rest_q"
            warnings.warn(
                f"mujoco:{name} is deprecated in Newton 1.7; use {replacement_name} instead. "
                "Scalar rest coordinates use springref - ref.",
                DeprecationWarning,
                stacklevel=3,
            )
        entries = values.items() if isinstance(values, dict) else enumerate(values)
        for d, value in entries:
            q = int(coord[d]) if name == "dof_springref" else d
            if q < 0:
                continue
            canonical = float(value) - offsets[d] if name == "dof_springref" else float(value)
            if target[q] != 0.0 and not np.isclose(target[q], canonical, rtol=1e-6, atol=1e-7):
                raise ValueError(f"Conflicting core joint spring value and deprecated mujoco:{name} at DOF {d}")
            target[q] = canonical
            if name == "dof_springref":
                authority[d] = 1
    model.joint_stiffness.assign(stiffness)
    model.joint_rest_q.assign(rest)
    previous_rest = np.zeros(count, dtype=np.float32)
    previous_rest[coord >= 0] = rest[coord[coord >= 0]]
    springref = previous_rest + offsets
    # Preserve unused legacy entries for unsupported reference-coordinate types.
    old_springref = namespace.__dict__.get("dof_springref")
    if old_springref is not None:
        springref[coord < 0] = old_springref.numpy()[coord < 0]
    replacement = _MuJoCoSpringNamespace("mujoco")
    replacement.__dict__.update(namespace.__dict__)
    replacement.__dict__.pop("dof_passive_stiffness", None)
    replacement.__dict__.pop("dof_springref", None)
    replacement._model = model
    replacement._coord = wp.array(coord, dtype=wp.int32, device=model.device)
    replacement._springref = wp.array(springref, dtype=wp.float32, device=model.device)
    replacement._previous_rest = wp.array(previous_rest, dtype=wp.float32, device=model.device)
    replacement._previous_legacy = wp.array(springref, dtype=wp.float32, device=model.device)
    replacement._legacy_authority = wp.array(authority, dtype=wp.int32, device=model.device)
    model.mujoco = replacement
    model._joint_spring_compat = replacement
    for name in ("mujoco:dof_passive_stiffness", "mujoco:dof_springref"):
        spec = model._attribute_spec(name)
        if spec is not None:
            model._set_attribute_spec(name, replace(spec, deprecated=True))


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
