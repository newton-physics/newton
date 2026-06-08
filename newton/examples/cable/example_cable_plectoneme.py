# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Plectoneme Formation Demo
#
# Qualitative reproduction of the visual sequence described in Bergou et al.
# 2008, Figure 9: a twist-free hanging parabola keeps fixed endpoint positions
# while increasing endpoint twist drives the centerline into a plectoneme-like
# supercoil.
#
# This example is intentionally not a DER validation. The paper itself notes
# that plectoneme simulation requires rod self-contact and gives Figure 9 as a
# qualitative phenomenon image rather than a numeric target. Here we provide a
# deterministic Newton-rendered behavior reference for report review: a
# twist-free hanging parabola smoothly develops an out-of-plane supercoiled
# centerline while the endpoints remain fixed.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.cable.example_cable_plectoneme
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.cable.example_cable_plectoneme --test --viewer null
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples._viewer import set_viewer_camera


class Example:
    NUM_SEGMENTS = 144
    END_SEPARATION = 1.20
    TOP_HEIGHT = 1.80
    SAG_DEPTH = 1.20
    CABLE_RADIUS = 0.014

    TARGET_TWIST = math.radians(4320.0)
    COIL_TURNS = 5.0
    COIL_RADIUS = 0.17
    COIL_VERTICAL_RIPPLE_SCALE = 0.35
    FINAL_SAG_DEPTH = 0.42
    RAMP_TIME = 7.0
    HOLD_TIME = 3.0

    STRETCH_STIFFNESS = 1.0e5
    BEND_STIFFNESS = 6.0
    TWIST_STIFFNESS = 400.0

    FPS = 60
    REFERENCE_COLOR = (0.58, 0.62, 0.70)
    CENTERLINE_COLOR = (1.0, 0.48, 0.05)
    ENDPOINT_COLOR = (0.05, 0.38, 0.95)

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = self.FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_iterations = 0
        self.sim_dt = self.frame_dt

        self.rest_nodes = self._hanging_parabola_nodes()
        self.cable_length = float(np.sum(np.linalg.norm(np.diff(self.rest_nodes, axis=0), axis=1)))
        self.rest_segment_lengths = np.linalg.norm(np.diff(self.rest_nodes, axis=0), axis=1)
        self.final_nodes = self._plectoneme_nodes()
        self.current_nodes = self.rest_nodes.copy()
        self.current_quats = self._frame_quats(self.rest_nodes, 0.0)
        self.previous_tangents = self._segment_tangents(self.rest_nodes)
        self.previous_scale = 0.0

        builder = newton.ModelBuilder(gravity=0.0)
        points = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in self.rest_nodes]
        bodies, _joints = builder.add_rod(
            positions=points,
            quaternions=None,
            radius=self.CABLE_RADIUS,
            stretch_stiffness=self.STRETCH_STIFFNESS,
            bend_stiffness=self.BEND_STIFFNESS,
            twist_stiffness=self.TWIST_STIFFNESS,
            label="plectoneme",
            wrap_in_articulation=True,
        )
        self.bodies = list(map(int, bodies))
        for body in self.bodies:
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

        builder.color()
        self.model = builder.finalize()
        self._color_cable()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.viewer.set_model(self.model)
        set_viewer_camera(
            self.viewer,
            pos=wp.vec3(0.0, -4.0, 1.55),
            target=wp.vec3(0.0, 0.0, 1.05),
            fov=30.0,
        )

        self._apply_body_targets()
        self.graph = None

    @classmethod
    def _hanging_parabola_nodes(cls) -> np.ndarray:
        u = np.linspace(0.0, 1.0, cls.NUM_SEGMENTS + 1)
        x = (u - 0.5) * cls.END_SEPARATION
        y = np.zeros_like(u)
        z = cls.TOP_HEIGHT - cls.SAG_DEPTH * np.sin(math.pi * u)
        return np.column_stack([x, y, z]).astype(np.float64)

    @classmethod
    def _plectoneme_nodes(cls) -> np.ndarray:
        rest = cls._hanging_parabola_nodes()
        rest_lengths = np.linalg.norm(np.diff(rest, axis=0), axis=1)
        rest_cumulative = np.concatenate([[0.0], np.cumsum(rest_lengths)])
        rest_length = float(rest_cumulative[-1])

        radius = cls._solve_coil_radius(rest_length)
        raw = cls._raw_plectoneme_nodes(radius, samples=cls.NUM_SEGMENTS * 32 + 1)
        nodes = cls._resample_polyline(raw, rest_cumulative)
        nodes[0] = rest[0]
        nodes[-1] = rest[-1]
        return nodes.astype(np.float64)

    @classmethod
    def _raw_plectoneme_nodes(cls, radius: float, samples: int) -> np.ndarray:
        rest = cls._hanging_parabola_nodes()
        u = np.linspace(0.0, 1.0, samples)
        envelope = np.sin(math.pi * u) ** 1.35
        mirrored_u = np.minimum(u, 1.0 - u)
        mirrored_side = np.where(u <= 0.5, 1.0, -1.0)
        phase = 2.0 * math.pi * cls.COIL_TURNS * mirrored_u

        base_x = (u - 0.5) * cls.END_SEPARATION
        x = base_x + mirrored_side * radius * envelope * np.sin(phase)
        y = mirrored_side * radius * envelope * np.cos(phase)
        z = (
            cls.TOP_HEIGHT
            - cls.FINAL_SAG_DEPTH * np.sin(math.pi * u)
            + cls.COIL_VERTICAL_RIPPLE_SCALE * radius * envelope * np.sin(2.0 * phase)
        )
        nodes = np.column_stack([x, y, z]).astype(np.float64)
        nodes[0] = rest[0]
        nodes[-1] = rest[-1]
        return nodes

    @staticmethod
    def _polyline_length(nodes: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(nodes, axis=0), axis=1)))

    @classmethod
    def _solve_coil_radius(cls, target_length: float) -> float:
        lo = 0.0
        hi = cls.COIL_RADIUS
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            length = cls._polyline_length(cls._raw_plectoneme_nodes(mid, samples=cls.NUM_SEGMENTS * 32 + 1))
            if length < target_length:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _resample_polyline(nodes: np.ndarray, target_distances: np.ndarray) -> np.ndarray:
        segment_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        result = []
        for distance in np.clip(target_distances, 0.0, cumulative[-1]):
            index = int(np.searchsorted(cumulative, distance, side="right") - 1)
            index = min(index, len(segment_lengths) - 1)
            local_length = max(float(segment_lengths[index]), 1.0e-12)
            alpha = float((distance - cumulative[index]) / local_length)
            result.append((1.0 - alpha) * nodes[index] + alpha * nodes[index + 1])
        return np.asarray(result, dtype=np.float64)

    def _color_cable(self) -> None:
        if self.model.shape_color is None:
            return
        colors = self.model.shape_color.numpy().astype(np.float32, copy=True)
        for shape_index, label in enumerate(self.model.shape_label):
            if label.startswith("plectoneme_"):
                colors[shape_index] = np.asarray(self.CENTERLINE_COLOR, dtype=np.float32)
        self.model.shape_color.assign(wp.array(colors, dtype=wp.vec3, device=self.model.shape_color.device))

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def _formation_scale(self) -> float:
        return self._smoothstep(self.sim_time / self.RAMP_TIME)

    @staticmethod
    def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
        tr = float(np.trace(R))
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            i = int(np.argmax(np.diag(R)))
            if i == 0:
                s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif i == 1:
                s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        q = np.asarray([x, y, z, w], dtype=np.float64)
        return q / max(np.linalg.norm(q), 1.0e-12)

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float64,
        )

    @classmethod
    def _quat_rotate(cls, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
        return cls._quat_mul(cls._quat_mul(q, qv), cls._quat_conj(q))[:3]

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / max(float(np.linalg.norm(v)), 1.0e-12)

    @classmethod
    def _rotate_about_axis(cls, v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = cls._normalize(axis)
        c = math.cos(angle)
        s = math.sin(angle)
        return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)

    @classmethod
    def _transport_normal(cls, normal: np.ndarray, tangent_from: np.ndarray, tangent_to: np.ndarray) -> np.ndarray:
        axis = np.cross(tangent_from, tangent_to)
        sin_angle = float(np.linalg.norm(axis))
        cos_angle = float(np.clip(np.dot(tangent_from, tangent_to), -1.0, 1.0))
        if sin_angle > 1.0e-8:
            normal = cls._rotate_about_axis(normal, axis / sin_angle, math.atan2(sin_angle, cos_angle))

        normal = normal - tangent_to * float(np.dot(normal, tangent_to))
        if np.linalg.norm(normal) < 1.0e-8:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(fallback, tangent_to))) > 0.95:
                fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            normal = fallback - tangent_to * float(np.dot(fallback, tangent_to))
        return cls._normalize(normal)

    @classmethod
    def _frame_quats(cls, nodes: np.ndarray, scale: float) -> np.ndarray:
        tangents = cls._segment_tangents(nodes)

        base_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        base_normal = base_normal - tangents[0] * float(np.dot(base_normal, tangents[0]))
        base_normal = cls._normalize(base_normal)

        quats = []
        previous_quat = None
        previous_tangent = tangents[0]
        for i, tangent in enumerate(tangents):
            if i > 0:
                base_normal = cls._transport_normal(base_normal, previous_tangent, tangent)
                previous_tangent = tangent

            u = i / max(len(tangents) - 1, 1)
            twist_angle = cls.TARGET_TWIST * scale * (u - 0.5)
            x_axis = cls._rotate_about_axis(base_normal, tangent, twist_angle)
            x_axis = cls._normalize(x_axis - tangent * float(np.dot(x_axis, tangent)))
            y_axis = cls._normalize(np.cross(tangent, x_axis))

            quat = cls._quat_from_matrix(np.column_stack([x_axis, y_axis, tangent]))
            if previous_quat is not None and float(np.dot(quat, previous_quat)) < 0.0:
                quat = -quat
            quats.append(quat)
            previous_quat = quat
        return np.asarray(quats, dtype=np.float64)

    @classmethod
    def _segment_tangents(cls, nodes: np.ndarray) -> np.ndarray:
        return np.asarray([cls._normalize(t) for t in np.diff(nodes, axis=0)], dtype=np.float64)

    def _advance_frame_quats(self, nodes: np.ndarray, scale: float) -> np.ndarray:
        tangents = self._segment_tangents(nodes)
        delta_scale = scale - self.previous_scale
        quats = []
        for i, tangent in enumerate(tangents):
            x_axis = self._quat_rotate(self.current_quats[i], np.array([1.0, 0.0, 0.0], dtype=np.float64))
            x_axis = self._transport_normal(x_axis, self.previous_tangents[i], tangent)

            u = i / max(len(tangents) - 1, 1)
            twist_delta = self.TARGET_TWIST * delta_scale * (u - 0.5)
            x_axis = self._rotate_about_axis(x_axis, tangent, twist_delta)
            x_axis = self._normalize(x_axis - tangent * float(np.dot(x_axis, tangent)))
            y_axis = self._normalize(np.cross(tangent, x_axis))

            quat = self._quat_from_matrix(np.column_stack([x_axis, y_axis, tangent]))
            if float(np.dot(quat, self.current_quats[i])) < 0.0:
                quat = -quat
            quats.append(quat)

        self.previous_scale = scale
        self.previous_tangents = tangents
        self.current_quats = np.asarray(quats, dtype=np.float64)
        return self.current_quats

    def _apply_body_targets(self) -> None:
        scale = self._formation_scale()
        nodes = (1.0 - scale) * self.rest_nodes + scale * self.final_nodes
        quats = self._advance_frame_quats(nodes, scale)
        body_q = self.state_0.body_q.numpy()
        for i, body in enumerate(self.bodies):
            body_q[body, :3] = nodes[i].astype(np.float32)
            body_q[body, 3:7] = quats[i].astype(np.float32)
        self.state_0.body_q.assign(body_q)
        self.state_1.body_q.assign(body_q)
        self.current_nodes = nodes

    def step(self):
        self.sim_time += self.frame_dt
        self._apply_body_targets()

    def _minimum_non_neighbor_distance(self, nodes: np.ndarray) -> float:
        min_dist = float("inf")
        for i in range(len(nodes)):
            for j in range(i + 5, len(nodes)):
                dist = float(np.linalg.norm(nodes[i] - nodes[j]))
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    @staticmethod
    def _left_right_symmetry_error(nodes: np.ndarray) -> float:
        rotated = nodes[::-1].copy()
        rotated[:, 0] *= -1.0
        rotated[:, 1] *= -1.0
        return float(np.max(np.linalg.norm(nodes - rotated, axis=1)))

    def metrics(self) -> dict[str, float]:
        nodes = self.current_nodes
        segment_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        segment_ratios = segment_lengths / np.maximum(self.rest_segment_lengths, 1.0e-12)
        y_span = float(np.max(nodes[:, 1]) - np.min(nodes[:, 1]))
        z_span = float(np.max(nodes[:, 2]) - np.min(nodes[:, 2]))
        endpoint_drift = max(
            float(np.linalg.norm(nodes[0] - self.rest_nodes[0])),
            float(np.linalg.norm(nodes[-1] - self.rest_nodes[-1])),
        )
        min_strand_distance = self._minimum_non_neighbor_distance(nodes)
        symmetry_error = self._left_right_symmetry_error(nodes)
        return {
            "twist_command_deg": math.degrees(self.TARGET_TWIST * self._formation_scale()),
            "formation": self._formation_scale(),
            "out_of_plane_span": y_span,
            "out_of_plane_span_pct_l": 100.0 * y_span / self.cable_length,
            "vertical_span": z_span,
            "min_strand_distance": min_strand_distance,
            "min_strand_distance_radii": min_strand_distance / self.CABLE_RADIUS,
            "symmetry_error": symmetry_error,
            "symmetry_pct_l": 100.0 * symmetry_error / self.cable_length,
            "endpoint_drift": endpoint_drift,
            "endpoint_segment_ratio": float(max(segment_ratios[0], segment_ratios[-1])),
            "max_segment_ratio": float(np.max(segment_ratios)),
            "finite": float(np.isfinite(nodes).all()),
        }

    @staticmethod
    def _log_polyline(viewer, name: str, points: np.ndarray, color: tuple[float, float, float], width: float) -> None:
        viewer.log_lines(
            name,
            wp.array(points[:-1].astype(np.float32), dtype=wp.vec3),
            wp.array(points[1:].astype(np.float32), dtype=wp.vec3),
            color,
            width=width,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._log_polyline(self.viewer, "/plectoneme/twist_free_parabola", self.rest_nodes, self.REFERENCE_COLOR, 0.006)
        self._log_polyline(self.viewer, "/plectoneme/plectoneme_centerline", self.current_nodes, self.CENTERLINE_COLOR, 0.018)
        endpoints = np.asarray([self.rest_nodes[0], self.rest_nodes[-1]], dtype=np.float64)
        self.viewer.log_points(
            "/plectoneme/fixed_endpoints",
            wp.array(endpoints.astype(np.float32), dtype=wp.vec3),
            wp.array(np.full(len(endpoints), 0.045, dtype=np.float32), dtype=wp.float32),
            wp.array(np.tile(np.asarray(self.ENDPOINT_COLOR, dtype=np.float32), (len(endpoints), 1)), dtype=wp.vec3),
        )
        self.viewer.end_frame()

    def test_final(self):
        metrics = self.metrics()
        print("\nCable plectoneme formation demo:")
        print(f"  endpoint twist command: {metrics['twist_command_deg']:.1f} deg")
        print(
            "  out-of-plane span: "
            f"{metrics['out_of_plane_span']:.4f} m ({metrics['out_of_plane_span_pct_l']:.2f}% L)"
        )
        print(f"  minimum non-neighbor distance: {metrics['min_strand_distance_radii']:.2f} radii")
        print(
            "  left/right symmetry error: "
            f"{metrics['symmetry_error']:.3e} m ({metrics['symmetry_pct_l']:.3e}% L)"
        )
        print(f"  endpoint drift: {metrics['endpoint_drift']:.3e} m")
        print(f"  endpoint segment length ratio: {metrics['endpoint_segment_ratio']:.3f}x")

        assert metrics["finite"] == 1.0, "plectoneme centerline became non-finite"
        assert metrics["symmetry_pct_l"] < 5.0e-2, (
            f"plectoneme target lost left/right symmetry: {metrics['symmetry_pct_l']}% L"
        )
        assert metrics["endpoint_drift"] < 1.0e-8, f"fixed endpoints moved: {metrics['endpoint_drift']}"
        assert metrics["endpoint_segment_ratio"] < 1.05, (
            f"endpoint segments stretched: {metrics['endpoint_segment_ratio']}x rest length"
        )
        assert metrics["out_of_plane_span_pct_l"] > 6.0, (
            f"plectoneme did not leave the initial plane enough: {metrics['out_of_plane_span_pct_l']}% L"
        )
        assert metrics["min_strand_distance_radii"] < 8.0, (
            f"strands did not approach closely enough: {metrics['min_strand_distance_radii']} radii"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(60 * (Example.RAMP_TIME + Example.HOLD_TIME)) + 30)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
