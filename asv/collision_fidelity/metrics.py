# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Streaming collision-fidelity measurements from public model and state arrays.

Call ``observe`` after every physical substep, with the resulting state and end
time. Time, position, velocity, mass and radius use SI units. Host transfers and
observer time must be excluded from physics timing. This module imports no GPU
backend for particle-only observations.

Protocol keys (all optional):

* ``window_s``: fixed nonoverlapping window length, default 0.5 s.
* ``stationary_intervals``: dictionaries with ``start_s``, ``end_s`` and an
  optional descriptive ``phase``. Explicit intervals take precedence over phase
  labels. Windows are anchored at each interval start and use endpoint samples
  in (start, end]. No substep may straddle a window boundary. Incomplete windows
  are discarded, never shifted to select a quieter segment.
* ``stationary_phases``: exact labels, default hold/rest/steady observation/
  recovery. Without intervals, a phase starts at the first sample's time-dt.
* ``frame_at_time``: optional callable returning ``position_m``, unit quaternion
  ``rotation_xyzw``, ``linear_velocity_m_s`` and ``angular_velocity_rad_s``.
  The default is a stationary world frame. No object-motion fit is performed.
* ``cable_body_indices``: explicit complete set of measured cable bodies. All capsule shapes
  on these bodies contribute 14 fixed surface points each (three four-point
  rings and two poles). Shape transforms and body COM offsets are respected.
* ``gravity_only_interval``: start_s/end_s/gravity_m_s2 and optional body_indices.
  All particle masses are included. Rigid masses require explicit body_indices;
  the caller must declare that this is the complete free system. The baseline
  must be observed exactly at start_s, via frame or observe.
* ``penetration_surfaces``: static physical planes (normal, offset_m) or boxes
  (center_m, half_extents_m, optional rotation_xyzw). Plane equation n.x=offset.
  These are sampled particle-shell gaps only; no swept crossing or full-surface
  guarantee. Radius comes from model.particle_radius or particle_radius_m.
  Optional shape_margin_m is subtracted in a separate zero-force-boundary gap;
  the nominal solid plus particle-radius gap is also retained.

Missing observations and unset pass limits remain explicit in ``result``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["FidelityObserver"]


def _array(value, *, columns=None):
    if value is None:
        return np.empty((0, columns)) if columns else np.empty(0)
    value = value.numpy() if hasattr(value, "numpy") else value
    result = np.asarray(value, dtype=np.float64)
    return result.reshape((-1, columns)) if columns else result


def _rotate(quaternion, points):
    """Rotate row vectors with XYZW unit quaternions, with NumPy broadcasting."""
    q = np.asarray(quaternion, dtype=np.float64)
    xyz, w = q[..., :3], q[..., 3:4]
    twice_cross = 2.0 * np.cross(xyz, points)
    return points + w * twice_cross + np.cross(xyz, twice_cross)


def _inverse_rotate(quaternion, points):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    q[..., :3] *= -1
    return _rotate(q, points)


class _Window:
    """Weighted Welford moments; memory depends on point count, not duration."""

    def __init__(self, start, end, phase):
        self.start, self.end, self.phase = start, end, phase
        self.weight = self.mean_t = self.m2_t = 0.0
        self.count = 0
        self.valid = True
        self.last_end = start
        self.mean_x = self.m2_x = self.cross_tx = self.speed2 = None

    def add(self, time, dt, q, v, finite):
        eps = 1e-8
        if abs(time - dt - self.last_end) > eps or time - dt < self.start - eps or time > self.end + eps:
            self.valid = False
        self.last_end = time
        if not finite or not len(q):
            self.valid = False
            return
        if self.mean_x is None:
            self.mean_x = np.zeros_like(q)
            self.m2_x = np.zeros(len(q))
            self.cross_tx = np.zeros_like(q)
            self.speed2 = np.zeros(len(q))
        if self.mean_x.shape != q.shape:
            self.valid = False
            return
        total = self.weight + dt
        delta_t = time - self.start - self.mean_t
        delta_x = q - self.mean_x
        self.mean_t += dt / total * delta_t
        self.mean_x += dt / total * delta_x
        self.m2_t += dt * delta_t * (time - self.start - self.mean_t)
        self.m2_x += dt * np.einsum("ij,ij->i", delta_x, q - self.mean_x)
        self.cross_tx += dt * delta_t * (q - self.mean_x)
        self.speed2 += dt * np.einsum("ij,ij->i", v, v)
        self.weight = total
        self.count += 1

    def summary(self):
        if not self.valid or self.count < 3 or abs(self.weight - (self.end - self.start)) > 1e-8 or self.m2_t <= 0:
            return None
        explained = np.einsum("ij,ij->i", self.cross_tx, self.cross_tx) / self.m2_t
        if not np.isfinite(explained).all() or not np.isfinite(self.m2_x).all() or not np.isfinite(self.speed2).all():
            return None
        residual = np.maximum(self.m2_x - explained, 0)
        # Roundoff from a pure linear trajectory is not resolved physical motion.
        residual[residual < 64 * np.finfo(float).eps * np.maximum(self.m2_x, explained)] = 0
        jitter = np.sqrt(residual / self.weight)
        speed = np.sqrt(self.speed2 / self.weight)
        ji, vi = int(jitter.argmax()), int(speed.argmax())
        return {
            "start_s": self.start,
            "end_s": self.end,
            "phase": self.phase,
            "substep_samples": self.count,
            "jitter_m": float(jitter[ji]),
            "rms_speed_m_s": float(speed[vi]),
            "position_sample_index": ji,
            "velocity_sample_index": vi,
        }


class FidelityObserver:
    """Collect raw local stability, declared free fall and sampled solid gaps.

    Experimental API. ``result`` is a non-mutating snapshot. It contains no
    numerical fidelity verdict and covers only the measurements named here.
    """

    def __init__(self, model, protocol: dict):
        self.model, self.protocol = model, dict(protocol)
        self.window_s = float(protocol.get("window_s", 0.5))
        if not math.isfinite(self.window_s) or self.window_s <= 0:
            raise ValueError("window_s must be finite and positive")
        self.intervals = protocol.get("stationary_intervals")
        if self.intervals is not None:
            previous_end = -math.inf
            for item in self.intervals:
                start, end = float(item["start_s"]), float(item["end_s"])
                if not math.isfinite(start + end) or start < previous_end or end <= start:
                    raise ValueError("Stationary intervals must be finite, ordered and nonoverlapping")
                previous_end = end
        self.stationary_phases = set(
            protocol.get("stationary_phases", ["hold", "rest", "steady observation", "recovery"])
        )
        self._phase = self._phase_start = self._window_key = self._window = None
        self._windows, self._discarded = [], 0
        self._last_time = None
        self._steps = self._frames = 0
        self._first_nonfinite = None
        self._nonfinite_count = 0
        self._particle_count = len(_array(getattr(self.model, "particle_mass", None)))
        self._samples = [{"kind": "particle", "particle_index": i} for i in range(self._particle_count)]
        self._warnings = []
        self._cable = self._prepare_cables()
        self._radius = _array(protocol.get("particle_radius_m", getattr(self.model, "particle_radius", None)))
        if not self._radius.size:
            self._radius = np.zeros(self._particle_count)
            if self._particle_count:
                self._warnings.append("Particle radius unavailable; penetration uses particle centers only.")
        if (
            self._radius.size not in (1, self._particle_count)
            or not np.isfinite(self._radius).all()
            or np.any(self._radius < 0)
        ):
            raise ValueError("Particle radii must be finite, nonnegative, and cover every vertex")
        self._surfaces = protocol.get("penetration_surfaces", [])
        for surface in self._surfaces:
            margin = float(surface.get("shape_margin_m", 0))
            if not math.isfinite(margin) or margin < 0:
                raise ValueError("Shape margin must be finite and nonnegative")
            if surface["kind"] == "plane":
                normal = np.asarray(surface["normal"], dtype=float)
                if (
                    normal.shape != (3,)
                    or not np.isfinite(normal).all()
                    or abs(np.linalg.norm(normal) - 1) > 1e-5
                    or not math.isfinite(float(surface["offset_m"]))
                ):
                    raise ValueError("Physical plane requires a finite offset and unit normal")
            elif surface["kind"] == "box":
                center, half = np.asarray(surface["center_m"]), np.asarray(surface["half_extents_m"])
                rotation = np.asarray(surface.get("rotation_xyzw", [0, 0, 0, 1]))
                if (
                    center.shape != (3,)
                    or half.shape != (3,)
                    or rotation.shape != (4,)
                    or not np.isfinite(np.concatenate((center, half, rotation))).all()
                    or np.any(half <= 0)
                    or abs(np.linalg.norm(rotation) - 1) > 1e-5
                ):
                    raise ValueError("Physical box requires finite center, positive extents and unit XYZW rotation")
            else:
                raise ValueError(f"Unsupported physical surface kind: {surface['kind']}")
        self._min_gap, self._gap_witness = None, None
        self._min_zero_force_gap, self._zero_force_witness = None, None
        self._freefall = protocol.get("gravity_only_interval")
        self._freefall_baseline = None
        self._freefall_max = None
        self._freefall_max_time = None
        self._freefall_samples = 0
        self._freefall_last_time = None
        self._freefall_missing = False
        self._freefall_mass = None
        self._freefall_reason = None
        if self._freefall:
            start, end = float(self._freefall["start_s"]), float(self._freefall["end_s"])
            gravity = np.asarray(self._freefall["gravity_m_s2"], dtype=float)
            if end <= start or not np.isfinite([start, end, *gravity]).all() or gravity.shape != (3,):
                raise ValueError("Invalid gravity_only_interval")

    @property
    def all_samples_finite(self):
        """Return the accumulated finite-state flag without materializing results."""
        return self._first_nonfinite is None

    def _prepare_cables(self):
        bodies = self.protocol.get("cable_body_indices", [])
        if bodies is None or not len(bodies):
            return None
        from newton import GeoType  # noqa: PLC0415  (Keep particle-only CPU observers lightweight.)

        shape_body = _array(getattr(self.model, "shape_body", None)).astype(int)
        shape_type = _array(getattr(self.model, "shape_type", None)).astype(int)
        scale = _array(getattr(self.model, "shape_scale", None), columns=3)
        transforms = _array(getattr(self.model, "shape_transform", None), columns=7)
        com = _array(getattr(self.model, "body_com", None), columns=3)
        ids, local = [], []
        found = set()
        for shape in np.flatnonzero(np.isin(shape_body, bodies)):
            if shape >= len(shape_type) or shape_type[shape] != int(GeoType.CAPSULE):
                continue
            radius, half = scale[shape, :2]
            ring = [[radius, 0], [0, radius], [-radius, 0], [0, -radius]]
            points = np.array(
                [[x, y, z] for z in (-half, 0, half) for x, y in ring] + [[0, 0, -half - radius], [0, 0, half + radius]]
            )
            points = _rotate(transforms[shape, 3:], points) + transforms[shape, :3]
            body = int(shape_body[shape])
            found.add(body)
            ids.extend([body] * len(points))
            local.extend(points)
            self._samples.extend(
                {"kind": "cable_surface", "body_index": body, "shape_index": int(shape), "local_sample_index": j}
                for j in range(len(points))
            )
        missing = sorted(set(map(int, bodies)) - found)
        if missing:
            self._warnings.append(f"No capsule surface samples identified for cable bodies {missing}.")
        if not ids:
            return None
        return np.asarray(ids, dtype=int), np.asarray(local), com

    def _read(self, state):
        arrays = {
            name: _array(getattr(state, name, None), columns=columns)
            for name, columns in (("particle_q", 3), ("particle_qd", 3), ("body_q", 7), ("body_qd", 6))
        }
        finite = all(np.isfinite(value).all() for value in arrays.values())
        q, v = arrays["particle_q"], arrays["particle_qd"]
        if q.shape != v.shape or len(q) != self._particle_count:
            raise ValueError("Particle sample identities or velocity coverage changed")
        if self._cable is not None:
            ids, local, com = self._cable
            poses, velocities = arrays["body_q"][ids], arrays["body_qd"][ids]
            world = _rotate(poses[:, 3:], local) + poses[:, :3]
            world_com = _rotate(poses[:, 3:], com[ids]) + poses[:, :3]
            point_velocity = velocities[:, :3] + np.cross(velocities[:, 3:], world - world_com)
            q, v = np.concatenate((q, world)), np.concatenate((v, point_velocity))
        return q, v, finite

    def _relative(self, q, v, time):
        callback = self.protocol.get("frame_at_time")
        if callback is None:
            return q, v
        frame = callback(time)
        origin = np.asarray(frame["position_m"], dtype=float)
        rotation = np.asarray(frame["rotation_xyzw"], dtype=float)
        linear = np.asarray(frame["linear_velocity_m_s"], dtype=float)
        angular = np.asarray(frame["angular_velocity_rad_s"], dtype=float)
        if (
            not np.isfinite(np.concatenate((origin, rotation, linear, angular))).all()
            or abs(np.linalg.norm(rotation) - 1) > 1e-5
        ):
            raise ValueError("Frame must have finite pose/velocities and a unit XYZW quaternion")
        return _inverse_rotate(rotation, q - origin), _inverse_rotate(
            rotation, v - linear - np.cross(angular, q - origin)
        )

    def _window_for(self, time, dt, phase):
        start = end = None
        key_prefix = phase
        if self.intervals is not None:
            for index, item in enumerate(self.intervals):
                if float(item["start_s"]) < time - 1e-9 <= float(item["end_s"]):
                    start, end = float(item["start_s"]), float(item["end_s"])
                    phase = item.get("phase", phase)
                    key_prefix = index
                    break
        elif phase in self.stationary_phases:
            if phase != self._phase:
                self._phase_start = time - dt
            start = self._phase_start
        self._phase = phase
        if start is None:
            return None
        index = max(0, int(math.floor((time - start - 1e-9) / self.window_s)))
        window_start = start + index * self.window_s
        window_end = window_start + self.window_s
        if end is not None and window_end > end + 1e-8:
            window_end = end  # Report the trailing partial window, but never score it.
        return (key_prefix, start, index), window_start, window_end, phase

    def _finish_window(self):
        if self._window is None:
            return
        value = self._window.summary()
        if value is not None and abs(self._window.end - self._window.start - self.window_s) < 1e-8:
            self._windows.append(value)
        else:
            self._discarded += 1
        self._window = self._window_key = None

    def observe(self, state, time_s: float, dt_s: float, *, phase: str = ""):
        """Sample after a physical substep; initial dt=0 seeds only the baseline."""
        time, dt = float(time_s), float(dt_s)
        if dt == 0 and self._last_time is None and math.isfinite(time):
            self.frame(state, time)
            self._last_time = time
            return
        if not math.isfinite(time + dt) or dt <= 0 or (self._last_time is not None and time <= self._last_time):
            raise ValueError("Observe requires increasing finite times and positive physical dt")
        q, v, finite = self._read(state)
        self._steps += 1
        self._last_time = time
        if not finite:
            self._nonfinite_count += 1
            if self._first_nonfinite is None:
                self._first_nonfinite = time
        spec = self._window_for(time, dt, str(phase))
        if spec is None or spec[0] != self._window_key:
            self._finish_window()
            if spec is not None:
                self._window_key = spec[0]
                self._window = _Window(*spec[1:])
        if self._window is not None:
            if finite:
                relative_q, relative_v = self._relative(q, v, time)
            else:
                relative_q, relative_v = q, v
            self._window.add(time, dt, relative_q, relative_v, finite)
        if finite:
            self._observe_gaps(q[: self._particle_count], time)
            self._observe_freefall(state, time, dt)
        elif self._freefall and self._freefall["start_s"] <= time <= self._freefall["end_s"]:
            self._freefall_missing = True

    def frame(self, state, time_s: float):
        """Optionally seed a free-fall boundary; do not duplicate substep samples."""
        self._frames += 1
        _, _, finite = self._read(state)
        if not finite and self._first_nonfinite is None:
            self._first_nonfinite = float(time_s)
        self._observe_freefall(state, float(time_s), None)

    def _observe_gaps(self, particles, time):
        if not len(particles):
            return
        for index, surface in enumerate(self._surfaces):
            if surface["kind"] == "plane":
                normal = np.asarray(surface["normal"], dtype=float)
                if abs(np.linalg.norm(normal) - 1) > 1e-5:
                    raise ValueError("Physical plane normal must be unit length")
                distance = particles @ normal - float(surface["offset_m"])
            elif surface["kind"] == "box":
                local = particles - np.asarray(surface["center_m"], dtype=float)
                local = _inverse_rotate(surface.get("rotation_xyzw", [0, 0, 0, 1]), local)
                excess = np.abs(local) - np.asarray(surface["half_extents_m"], dtype=float)
                distance = np.linalg.norm(np.maximum(excess, 0), axis=1) + np.minimum(excess.max(axis=1), 0)
            else:
                raise ValueError(f"Unsupported physical surface kind: {surface['kind']}")
            gap = distance - self._radius
            point = int(gap.argmin())
            if self._min_gap is None or gap[point] < self._min_gap:
                self._min_gap = float(gap[point])
                self._gap_witness = {"time_s": time, "particle_index": point, "surface_index": index}
            zero_force_gap = gap[point] - float(surface.get("shape_margin_m", 0))
            if self._min_zero_force_gap is None or zero_force_gap < self._min_zero_force_gap:
                self._min_zero_force_gap = float(zero_force_gap)
                self._zero_force_witness = {"time_s": time, "particle_index": point, "surface_index": index}

    def _observe_freefall(self, state, time, dt):
        interval = self._freefall
        if not interval or time < interval["start_s"] - 1e-8 or time > interval["end_s"] + 1e-8:
            return
        mass = _array(getattr(self.model, "particle_mass", None))
        velocity = _array(getattr(state, "particle_qd", None), columns=3)
        body_mass = _array(getattr(self.model, "body_mass", None))
        if np.any(body_mass > 0) and "body_indices" not in interval:
            self._freefall_reason = "Rigid masses exist; declare the complete free-system body_indices explicitly."
            return
        ids = np.asarray(interval.get("body_indices", []), dtype=int)
        if len(ids):
            mass = np.concatenate((mass, body_mass[ids]))
            velocity = np.concatenate((velocity, _array(getattr(state, "body_qd", None), columns=6)[ids, :3]))
        if (
            not len(mass)
            or mass.sum() <= 0
            or not np.isfinite(velocity).all()
            or not np.isfinite(mass).all()
            or np.any(mass < 0)
        ):
            self._freefall_reason = "Complete finite masses and velocities are unavailable."
            return
        self._freefall_mass = float(mass.sum())
        com_velocity = np.sum(mass[:, None] * velocity, axis=0) / self._freefall_mass
        if abs(time - interval["start_s"]) < 1e-8:
            if self._freefall_baseline is None:
                self._freefall_baseline = com_velocity.copy()
                self._freefall_last_time = time
            return
        if dt is None:  # Frames cannot replace absent physical-substep samples.
            return
        if self._freefall_baseline is None:
            self._freefall_reason = "No mass-weighted velocity baseline observed at declared interval start."
            return
        if abs(time - dt - self._freefall_last_time) > 1e-8:
            self._freefall_missing = True
        expected = self._freefall_baseline + np.asarray(interval["gravity_m_s2"]) * (time - interval["start_s"])
        error = float(np.linalg.norm(com_velocity - expected))
        if self._freefall_max is None or error > self._freefall_max:
            self._freefall_max, self._freefall_max_time = error, time
        self._freefall_samples += 1
        self._freefall_last_time = time

    def result(self) -> dict:
        """Return JSON-safe raw values, observation coverage and explicit limitations."""
        windows = list(self._windows)
        discarded = self._discarded
        if self._window is not None:
            value = self._window.summary()
            if value is not None and abs(self._window.end - self._window.start - self.window_s) < 1e-8:
                windows.append(value)
            else:
                discarded += 1
        position = max(windows, key=lambda x: x["jitter_m"]) if windows else None
        velocity = max(windows, key=lambda x: x["rms_speed_m_s"]) if windows else None
        declared_count = (
            None
            if self.intervals is None
            else sum(
                int(math.floor((float(item["end_s"]) - float(item["start_s"]) + 1e-8) / self.window_s))
                for item in self.intervals
            )
        )
        freefall_complete = bool(
            self._freefall
            and self._freefall_max is not None
            and not self._freefall_missing
            and self._freefall_reason is None
            and abs(self._freefall_last_time - self._freefall["end_s"]) < 1e-8
        )
        return {
            "schema_version": 1,
            "sampling": {
                "physical_substep_samples": self._steps,
                "frame_calls": self._frames,
                "particle_vertex_count": self._particle_count,
                "cable_surface_sample_count": len(self._samples) - self._particle_count,
                "material_sample_count": len(self._samples),
                "sample_identities_fixed": True,
                "cable_surface_scope": "14 material points per identified capsule; not a continuous surface maximum",
                "frame": "prescribed callback" if self.protocol.get("frame_at_time") else "stationary world",
                "all_soft_points_scope": "all particle vertices plus identified cable surface samples; no contact filtering",
            },
            "finite": {
                "all_samples_finite": self._first_nonfinite is None,
                "nonfinite_substep_count": self._nonfinite_count,
                "first_nonfinite_time_s": self._first_nonfinite,
            },
            "stability": {
                "status": "measured" if windows else "incomplete",
                "criterion_status": "incomplete",
                "limit": None,
                "window_s": self.window_s,
                "window_overlap_s": 0,
                "complete_window_count": len(windows),
                "discarded_window_count": discarded,
                "jitter_m": position["jitter_m"] if position else None,
                "declared_complete_window_count": declared_count,
                "missing_or_invalid_declared_window_count": None
                if declared_count is None
                else max(0, declared_count - len(windows)),
                "rms_speed_m_s": velocity["rms_speed_m_s"] if velocity else None,
                "worst_position_sample": self._samples[position["position_sample_index"]] if position else None,
                "worst_velocity_sample": self._samples[velocity["velocity_sample_index"]] if velocity else None,
                "windows": windows,
                "position_method": "worst local linearly detrended vector displacement RMS",
                "velocity_method": "worst local relative-speed RMS; mean velocity is retained",
                "scope": "complete stationary windows only; unobserved or invalid windows are not scored",
            },
            "freefall": {
                "status": "not_declared" if not self._freefall else "measured" if freefall_complete else "incomplete",
                "criterion_status": "incomplete",
                "limit": None,
                "max_mass_weighted_velocity_error_m_s": self._freefall_max,
                "worst_time_s": self._freefall_max_time,
                "total_mass_kg": self._freefall_mass,
                "substep_samples": self._freefall_samples,
                "interval": self._freefall,
                "baseline_velocity_m_s": self._freefall_baseline.tolist()
                if self._freefall_baseline is not None
                else None,
                "missing_substeps": self._freefall_missing,
                "reason": self._freefall_reason,
            },
            "penetration": {
                "status": "sampled" if self._min_gap is not None else "not_measured",
                "criterion_status": "incomplete",
                "limit": None,
                "minimum_sampled_particle_shell_gap_m": self._min_gap,
                "minimum_sampled_zero_force_gap_m": self._min_zero_force_gap,
                "zero_force_witness": self._zero_force_witness,
                "zero_force_boundary": "nominal surface plus particle radius plus declared shape_margin_m; excludes detection gap",
                "rigid_soft_penetration_m": max(0, -self._min_gap) if self._min_gap is not None else None,
                "witness": self._gap_witness,
                "surface_count": len(self._surfaces),
                "swept_crossings_checked": False,
                "scope": "particle centers minus physical radius against declared static planes/boxes only; no robot, self or cable penetration claim",
            },
            "unimplemented_criteria": [
                "grasp acquisition",
                "10/60 s retention and interface slip",
                "transport target",
                "placement and sustained release",
                "friction slide/separation",
                "full-surface penetration and swept crossing",
            ],
            "warnings": list(self._warnings),
        }
