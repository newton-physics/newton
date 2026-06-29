# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Michell Threshold Validation
#
# Twisted-ring threshold verification for elastic rod stability.
#
# A closed isotropic ring with bend stiffness and twist stiffness should lose
# planar stability when the imposed total material-frame twist exceeds
#
#   critical_twist = 2*pi*sqrt(3*bend_stiffness/twist_stiffness)
#
# The example fixes the imposed twist to one full material-frame turn and
# sweeps twist_stiffness / bend_stiffness. This moves the analytical critical
# twist around the fixed one-turn load and matches the report protocol.
#
# Rows at or below the threshold are fixed planar references. Rows above the
# threshold remain dynamic; the sweep verifies that supercritical rings develop
# a visible out-of-plane response, while exact post-buckling branch and
# amplitude are not treated as monotonic calibration targets.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_michell_threshold
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_michell_threshold --test --viewer null
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd._viewer import node_xyz, set_viewer_camera


def _cable_coplanarity(points) -> float:
    """Scale-free coplanarity metric: 0 for planar centerlines, growing out-of-plane."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] <= 2:
        return 0.0
    centered = pts - np.mean(pts, axis=0)
    moment = centered.T @ centered
    trace = float(np.trace(moment))
    if trace <= 0.0:
        return 0.0
    eigvals = np.linalg.eigvalsh(moment)
    return float(3.0 * max(float(eigvals[0]), 0.0) / trace)


class Example:
    NUM_SEGMENTS = 48
    RING_RADIUS = 0.55
    CABLE_RADIUS = 0.01

    STRETCH_STIFFNESS = 1.0e7
    BEND_STIFFNESS = 50.0
    CRITICAL_TWIST_TO_BEND = 3.0
    TWIST_STIFFNESS = BEND_STIFFNESS * CRITICAL_TWIST_TO_BEND
    TOTAL_TWIST = 2.0 * math.pi

    # These ratios give total_twist / critical_twist values of roughly
    # 0.71x, 0.86x, 0.95x, 1.00x, 1.05x, 1.18x, and 1.41x.
    TWIST_TO_BEND_RATIOS = (1.5, 2.2, 2.7, 3.0, 3.3, 4.2, 6.0)
    CLEARLY_SUPERCRITICAL_FACTOR = 1.35
    CASE_SPACING = 1.05
    SEED_AMPLITUDE = 1.0e-3
    STABLE_COPLANARITY_MAX = 5.0e-3
    WRITHE_COPLANARITY_MIN = 5.0e-2

    FPS = 60
    RUN_TIME = 6.0

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = self.FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_iterations = 28
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.critical_twist = self.TOTAL_TWIST

        builder = newton.ModelBuilder(gravity=0.0)
        self.cases = []
        self.cable_bodies: list[int] = []
        x_offsets = self._case_offsets(len(self.TWIST_TO_BEND_RATIOS))
        for twist_to_bend, x_offset in zip(self.TWIST_TO_BEND_RATIOS, x_offsets, strict=True):
            twist_stiffness = self.BEND_STIFFNESS * float(twist_to_bend)
            case_critical_twist = self.michell_critical_twist(self.BEND_STIFFNESS, twist_stiffness)
            factor = self.TOTAL_TWIST / case_critical_twist
            label = f"{factor:.2f}x"
            points, quats = self._ring_points_and_quats(x_offset)
            bodies, _joints = builder.add_rod(
                positions=points,
                quaternions=quats,
                radius=self.CABLE_RADIUS,
                stretch_stiffness=self.STRETCH_STIFFNESS,
                stretch_damping=0.0,
                bend_stiffness=self.BEND_STIFFNESS,
                bend_damping=0.0,
                twist_stiffness=twist_stiffness,
                twist_damping=0.0,
                closed=True,
                label=f"michell_threshold_{label}",
                wrap_in_articulation=True,
                body_frame_origin="com",
            )
            self.cable_bodies.extend(int(b) for b in bodies)
            is_dynamic = factor > 1.0
            if not is_dynamic:
                self._make_bodies_kinematic(builder, bodies)
            self.cases.append(
                {
                    "label": label,
                    "bodies": list(map(int, bodies)),
                    "factor": float(factor),
                    "total_twist": self.TOTAL_TWIST,
                    "critical_twist": case_critical_twist,
                    "twist_stiffness": twist_stiffness,
                    "twist_to_bend": float(twist_to_bend),
                    "expected": self._expected_outcome(float(factor)),
                    "dynamic": is_dynamic,
                }
            )

        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        body_q = self.state_0.body_q.numpy()
        for case in self.cases:
            case["rest_pos"] = np.asarray(
                [node_xyz(body_q[b], self._ring_segment_length()) for b in case["bodies"]],
                dtype=np.float64,
            )
            case["rest_q"] = [np.asarray(body_q[b][3:7], dtype=np.float64) for b in case["bodies"]]

        self._apply_initial_twist_and_seed()
        self.viewer.set_model(self.model)
        self.viewer.set_picking_linear_only_bodies(self.cable_bodies)
        set_viewer_camera(
            self.viewer,
            pos=wp.vec3(0.0, -7.2, 2.35),
            target=wp.vec3(0.0, 0.0, 0.60),
            fov=42.0,
        )
        self.graph = None
        self.capture()

    @staticmethod
    def michell_critical_twist(bend_stiffness: float, twist_stiffness: float) -> float:
        """Critical total twist for the Michell/Zajac ring instability."""
        return 2.0 * math.pi * math.sqrt(3.0 * float(bend_stiffness) / float(twist_stiffness))

    @classmethod
    def _case_offsets(cls, count: int) -> list[float]:
        center = 0.5 * float(count - 1)
        return [(float(i) - center) * cls.CASE_SPACING for i in range(count)]

    @classmethod
    def _ring_segment_length(cls) -> float:
        return 2.0 * cls.RING_RADIUS * math.sin(math.pi / cls.NUM_SEGMENTS)

    @classmethod
    def _expected_outcome(cls, critical_twist_factor: float) -> str:
        if math.isclose(critical_twist_factor, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            return "critical"
        if critical_twist_factor < 1.0:
            return "stable"
        if critical_twist_factor < cls.CLEARLY_SUPERCRITICAL_FACTOR:
            return "near-threshold"
        return "writhe"

    @classmethod
    def _case_color(cls, critical_twist_factor: float) -> tuple[float, float, float]:
        if math.isclose(critical_twist_factor, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            return (0.58, 0.40, 0.95)
        if critical_twist_factor < 1.0:
            return (0.0, 0.70, 1.0)
        if critical_twist_factor < cls.CLEARLY_SUPERCRITICAL_FACTOR:
            return (1.0, 0.75, 0.05)
        return (1.0, 0.25, 0.15)

    @staticmethod
    def _make_bodies_kinematic(builder: newton.ModelBuilder, bodies) -> None:
        for body_id in bodies:
            body = int(body_id)
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

    @classmethod
    def _ring_points_and_quats(cls, x_offset: float) -> tuple[list[wp.vec3], list[wp.quat]]:
        theta = np.linspace(0.0, 2.0 * math.pi, cls.NUM_SEGMENTS + 1, endpoint=True)
        points = [
            wp.vec3(float(x_offset + cls.RING_RADIUS * math.cos(t)), float(cls.RING_RADIUS * math.sin(t)), 0.6)
            for t in theta
        ]

        quats = []
        for i in range(cls.NUM_SEGMENTS):
            mid = 0.5 * (theta[i] + theta[i + 1])
            radial = np.array([math.cos(mid), math.sin(mid), 0.0], dtype=np.float64)
            tangent = np.array([-math.sin(mid), math.cos(mid), 0.0], dtype=np.float64)
            x_axis = -radial
            y_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            z_axis = tangent
            quats.append(cls._quat_from_matrix(np.column_stack([x_axis, y_axis, z_axis])))
        return points, quats

    @staticmethod
    def _quat_from_matrix(R: np.ndarray) -> wp.quat:
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
        return wp.quat(float(x), float(y), float(z), float(w))

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

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    @classmethod
    def _quat_rotate(cls, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
        return cls._quat_mul(cls._quat_mul(q, qv), cls._quat_conj(q))[:3]

    @staticmethod
    def _axis_quat(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64)
        axis /= max(np.linalg.norm(axis), 1.0e-12)
        s = math.sin(0.5 * angle)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle)], dtype=np.float64)

    def _apply_initial_twist_and_seed(self) -> None:
        body_q = self.state_0.body_q.numpy()
        for case in self.cases:
            bodies = case["bodies"]
            total_twist = case["total_twist"]
            for i, body in enumerate(bodies):
                rest_q = np.asarray(body_q[body][3:7], dtype=np.float64)
                if total_twist != 0.0:
                    tangent = self._quat_rotate(rest_q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
                    phi = total_twist * i / len(bodies)
                    body_q[body][3:7] = self._quat_mul(self._axis_quat(tangent, phi), rest_q).astype(np.float32)

                # Every ring gets the same deterministic perturbation. Reference
                # rows keep it static; dynamic rows can amplify it.
                body_q[body][2] += self.SEED_AMPLITUDE * math.sin(6.0 * math.pi * i / len(bodies))

        self.state_0.body_q.assign(body_q)
        self.state_1.body_q.assign(body_q)

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self) -> None:
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def _current_points(self, case: dict) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        segment_length = self._ring_segment_length()
        return np.asarray([node_xyz(body_q[b], segment_length) for b in case["bodies"]], dtype=np.float64)

    @staticmethod
    def _best_fit_plane_offsets(points: np.ndarray) -> np.ndarray:
        centered = points - np.mean(points, axis=0)
        if len(centered) <= 2:
            return np.zeros(len(centered), dtype=np.float64)
        moment = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(moment)
        normal = eigvecs[:, int(np.argmin(eigvals))]
        return centered @ normal

    def _metrics(self, case: dict) -> dict[str, float | bool | str]:
        points = self._current_points(case)
        z = points[:, 2]
        plane_offsets = self._best_fit_plane_offsets(points)
        return {
            "factor": case["factor"],
            "critical_twist_factor": case["factor"],
            "total_twist": case["total_twist"],
            "critical_twist": case["critical_twist"],
            "twist_stiffness": case["twist_stiffness"],
            "twist_to_bend": case["twist_to_bend"],
            "z_range": float(np.max(z) - np.min(z)),
            "z_std": float(np.std(z)),
            "plane_rms": float(np.sqrt(np.mean(plane_offsets * plane_offsets))),
            "plane_span": float(np.max(plane_offsets) - np.min(plane_offsets)),
            "coplanarity": float(_cable_coplanarity(points)),
            "finite": bool(np.isfinite(points).all()),
            "expected": case["expected"],
        }

    @staticmethod
    def _log_polyline(viewer, name: str, points: np.ndarray, color: tuple[float, float, float], width: float) -> None:
        closed = np.vstack([points, points[0]])
        viewer.log_lines(
            name,
            wp.array(closed[:-1].astype(np.float32), dtype=wp.vec3),
            wp.array(closed[1:].astype(np.float32), dtype=wp.vec3),
            color,
            width=width,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for case in self.cases:
            self._log_polyline(
                self.viewer,
                f"/michell_threshold/rest/{case['label']}",
                case["rest_pos"] + np.array([0.0, 0.0, -0.04]),
                (0.30, 0.30, 0.30),
                0.006,
            )
            self._log_polyline(
                self.viewer,
                f"/michell_threshold/current/{case['label']}",
                self._current_points(case),
                self._case_color(case["factor"]),
                0.012,
            )
        self.viewer.end_frame()

    def test_final(self):
        metrics = {case["label"]: self._metrics(case) for case in self.cases}

        assert all(bool(row["finite"]) for row in metrics.values()), f"non-finite ring positions: {metrics}"

        stable = {label: row for label, row in metrics.items() if row["expected"] == "stable"}
        above_threshold = {label: row for label, row in metrics.items() if row["factor"] > 1.0}
        max_stable_coplanarity = max(float(row["coplanarity"]) for row in stable.values())
        max_above_threshold_coplanarity = max(float(row["coplanarity"]) for row in above_threshold.values())

        assert max_stable_coplanarity < self.STABLE_COPLANARITY_MAX, (
            f"clearly subcritical rings lost planarity: {metrics}"
        )
        assert max_above_threshold_coplanarity > self.WRITHE_COPLANARITY_MIN, (
            f"above-threshold sweep did not produce a visible writhe response: {metrics}"
        )
        assert max_above_threshold_coplanarity > 10.0 * max(max_stable_coplanarity, 1.0e-8), (
            f"above-threshold rings should become much less planar than stable rings: {metrics}"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(Example.FPS * Example.RUN_TIME))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
