# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample conservative task geometry and reduce only established outcomes.

These helpers read public model/state arrays outside the physics timer. Positive
distance between complete object and hand bounds proves separation at that
sample. Bounds overlap does not prove contact, grasp, or continued retention.
Task acceptance limits and interface-slip observations are not supplied here.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .metrics import _array, _rotate

__all__ = ["criterion_results", "sample_task"]


def _distance(a, b):
    """Return a conservative Euclidean separation between two closed bounds."""
    return float(np.linalg.norm(np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0)))


class _TaskSampler:
    """Cache fixed collision geometry; never modify the model or state."""

    def __init__(self, scene, protocol):
        self.model = scene.model
        self.protocol = protocol
        self.radius = _array(getattr(self.model, "particle_radius", None))
        self.mass = _array(getattr(self.model, "particle_mass", None))
        self.shape_body = _array(getattr(self.model, "shape_body", None)).astype(int)
        self.shape_type = _array(getattr(self.model, "shape_type", None)).astype(int)
        self.scale = _array(getattr(self.model, "shape_scale", None), columns=3)
        self.transforms = _array(getattr(self.model, "shape_transform", None), columns=7)
        self.margin = _array(getattr(self.model, "shape_margin", None))
        self.boxes = {}
        self.warnings = []
        self.hands = []
        raw = getattr(scene, "raw_scene", scene)
        specs = getattr(raw, "specs", None) or getattr(raw, "hands", None)
        if not specs and getattr(raw, "hand", None) is not None:
            specs = [raw.hand]
        for index, spec in enumerate(specs or []):
            ids = getattr(spec, "shape_ids", None)
            self.hands.append(
                {"label": getattr(spec, "label", str(index)), "shape_ids": [] if ids is None else list(ids)}
            )
        self.hands_complete = bool(self.hands) and all(hand["shape_ids"] for hand in self.hands)
        bodies = protocol.get("cable_body_indices", getattr(raw, "cable_bodies", []))
        bodies = [] if bodies is None else list(bodies)
        self.cable_shapes = np.flatnonzero(np.isin(self.shape_body, bodies)).tolist()
        self.cables_complete = all(body in self.shape_body[self.cable_shapes] for body in bodies)
        self.object_declared = len(self.radius) > 0 or bool(bodies)
        for shape in {i for hand in self.hands for i in hand["shape_ids"]} | set(self.cable_shapes):
            try:
                self.boxes[shape] = self._shape_box(shape)
            except (ValueError, IndexError, AttributeError) as exc:
                self.warnings.append(str(exc))
                if any(shape in hand["shape_ids"] for hand in self.hands):
                    self.hands_complete = False
                if shape in self.cable_shapes:
                    self.cables_complete = False

    def _shape_box(self, shape):
        """Bound original local collision geometry, including its physical margin."""
        from newton import GeoType  # noqa: PLC0415

        kind, scale = self.shape_type[shape], self.scale[shape]
        if kind in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
            source = self.model.shape_source[shape]
            vertices = _array(source.vertices, columns=3) * scale
            lo, hi = vertices.min(0), vertices.max(0)
        elif kind == int(GeoType.SPHERE):
            hi = np.repeat(scale[0], 3)
            lo = -hi
        elif kind == int(GeoType.CAPSULE):
            hi = np.array([scale[0], scale[0], scale[0] + scale[1]])
            lo = -hi
        elif kind in (int(GeoType.CYLINDER), int(GeoType.CONE)):
            hi = np.array([scale[0], scale[0], scale[1]])
            lo = -hi
        elif kind in (int(GeoType.BOX), int(GeoType.ELLIPSOID)):
            hi, lo = np.abs(scale), -np.abs(scale)
        else:
            raise ValueError(f"Task bounds do not support shape {shape}, type {kind}")
        margin = float(self.margin[shape]) if len(self.margin) else 0.0
        if not np.isfinite([*lo, *hi, margin]).all() or margin < 0:
            raise ValueError(f"Invalid task bounds for shape {shape}")
        local = np.array(list(itertools.product(*zip(lo - margin, hi + margin, strict=True))))
        transform = self.transforms[shape]
        return _rotate(transform[3:], local) + transform[:3]

    def _bounds(self, shapes, poses):
        points = []
        for shape in shapes:
            local = self.boxes[shape]
            body = self.shape_body[shape]
            points.append(local if body < 0 else _rotate(poses[body, 3:], local) + poses[body, :3])
        points = np.concatenate(points)
        return points.min(0), points.max(0)

    def sample(self, state, time_s):
        """Return a JSON-safe frame with explicit object and hand coverage."""
        q = _array(getattr(state, "particle_q", None), columns=3)
        poses = _array(getattr(state, "body_q", None), columns=7)
        finite = all(
            np.isfinite(_array(getattr(state, name, None))).all()
            for name in ("particle_q", "particle_qd", "body_q", "body_qd")
        )
        row = {
            "time_s": float(time_s),
            "finite": bool(finite),
            "hand_count": len(self.hands),
            "hands_complete": self.hands_complete,
            "object_complete": False,
            "warnings": list(self.warnings),
            "scope": "Whole particle shells and declared cable shape bounds versus complete declared hand bounds; positive distance proves sampled separation only.",
        }
        if not finite:
            return row
        objects = []
        particles_complete = len(q) == len(self.radius) and np.isfinite(self.radius).all() and np.all(self.radius >= 0)
        if len(q) and particles_complete:
            objects.append(((q - self.radius[:, None]).min(0), (q + self.radius[:, None]).max(0)))
            if (
                len(self.mass) == len(q)
                and np.isfinite(self.mass).all()
                and np.all(self.mass >= 0)
                and self.mass.sum() > 0
            ):
                row["center_of_mass_m"] = (np.sum(q * self.mass[:, None], axis=0) / self.mass.sum()).tolist()
        if self.cable_shapes and self.cables_complete:
            objects.append(self._bounds(self.cable_shapes, poses))
        row["object_complete"] = bool(objects and self.object_declared and particles_complete and self.cables_complete)
        if row["object_complete"]:
            whole = np.min([x[0] for x in objects], axis=0), np.max([x[1] for x in objects], axis=0)
            row["whole_object_bounds_m"] = [whole[0].tolist(), whole[1].tolist()]
            row["minimum_shell_z_lower_bound_m"] = float(whole[0][2])
            if self.hands_complete:
                hands = [self._bounds(hand["shape_ids"], poses) for hand in self.hands]
                row["hand_bounds_m"] = [[lo.tolist(), hi.tolist()] for lo, hi in hands]
                row["hand_separation_lower_bound_m"] = [_distance(whole, hand) for hand in hands]
        return row


def sample_task(scene, time_s: float, *, protocol=None) -> dict:
    """Sample a scene after a display frame, outside the measured physics time.

    Hand specs must enumerate all original collision ``shape_ids``. Cable body
    IDs must enumerate the complete measured arrangement. Unknown geometry
    invalidates coverage; it is never silently removed from separation bounds.
    """
    protocol = protocol if protocol is not None else scene.protocol
    sampler = getattr(scene, "_fidelity_task_sampler", None)
    if sampler is None:
        sampler = _TaskSampler(scene, protocol)
        scene._fidelity_task_sampler = sampler
    return sampler.sample(scene.state, time_s)


def _criterion(applicable, reason, *, status=None, evidence=None):
    return {
        "status": status or ("incomplete" if applicable else "not_applicable"),
        "applicable": bool(applicable),
        "reason": reason,
        "evidence": evidence or {},
    }


def _hold_witness(frames, start, end):
    if start is None or end is None:
        return None
    for row in sorted(frames, key=lambda item: item["time_s"]):
        gaps = row.get("hand_separation_lower_bound_m", [])
        count = row.get("hand_count", 0)
        if (
            row.get("finite") is True
            and row.get("hands_complete") is True
            and row.get("object_complete") is True
            and count > 0
            and len(gaps) == count
            and start <= row["time_s"] <= end
            and all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in gaps)
        ):
            return row
    return None


def criterion_results(frames, protocol, *, execution=None, measurements=None) -> dict:
    """Reduce required failures while retaining missing limits and task evidence.

    ``frames`` are outputs of :func:`sample_task`. Numerical completion is a
    separate criterion, never a task or fidelity pass. Unsupported execution
    excludes the overall result. No visual review is inferred by this reducer.
    """
    execution, measurements = execution or {}, measurements or {}
    categories = protocol.get("category_applicability", {})
    robot = protocol.get("family") == "Robot manipulation"
    defaults = {
        "grasp": robot,
        "hold": robot,
        "transport": robot,
        "place_release": robot,
        "penetration": bool(protocol.get("penetration_surfaces")),
        "soft_stability": bool(protocol.get("stationary_intervals")),
        "friction_separation": robot,
        "dynamic": bool(protocol.get("gravity_only_interval")),
    }
    result = {
        name: _criterion(
            categories.get(name, applicable), "Required measurements or acceptance limits are not defined."
        )
        for name, applicable in defaults.items()
    }
    if protocol.get("initial_hold_prepared"):
        result["grasp"] = _criterion(False, "Prepared hold; grasp acquisition is not exercised.")
    witness = _hold_witness(frames, protocol.get("hold_start"), protocol.get("hold_end"))
    if result["hold"]["applicable"] and witness is not None:
        result["hold"] = _criterion(
            True,
            "The complete object is separated from every declared hand during the required hold.",
            status="fail",
            evidence={"first_proven_separation_s": witness["time_s"], "sample": witness},
        )
    result["hold"]["evidence"]["frame_sample_count"] = len(frames)
    result["hold"]["evidence"]["scope"] = (
        "Sampled negative witnesses only; overlapping bounds do not establish retention or slip."
    )
    finite = measurements.get("finite", {})
    nonfinite = execution.get("status") == "nonfinite" or finite.get("all_samples_finite") is False
    if nonfinite:
        numerical_status, reason = "fail", "A nonfinite state was observed."
    elif execution.get("status") == "completed" and execution.get("completed") is True:
        numerical_status, reason = "pass", "The scheduled timeline completed without an observed nonfinite state."
    else:
        numerical_status, reason = "incomplete", "Complete finite execution has not been established."
    result["numerical"] = _criterion(True, reason, status=numerical_status, evidence=finite)
    if execution.get("status") == "unsupported":
        result["overall"] = _criterion(False, "This case is unsupported by the current builder.", status="unsupported")
    else:
        required = [row for row in result.values() if row["applicable"]]
        status = (
            "fail"
            if any(row["status"] == "fail" for row in required)
            else ("pass" if required and all(row["status"] == "pass" for row in required) else "incomplete")
        )
        result["overall"] = _criterion(
            True, "Any established required failure fails the case; missing criteria prevent a pass.", status=status
        )
    return result
