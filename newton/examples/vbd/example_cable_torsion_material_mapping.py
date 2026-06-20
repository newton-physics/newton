# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Torsion Material Mapping Validation
#
# The test starts from material properties -- E or explicit G, nu, radius, and
# segment length -- and converts them into Newton's per-joint twist stiffness.
# The rod centerline endpoints are fixed while one end is rotated about the rod
# tangent. In Newton VBD cable terms this is a fixed body position with a driven
# endpoint quaternion. This is an isolated circular-shaft material-mapping
# validation, not an asymmetry or routed twist-transfer reproduction.
#
# The scene renders:
#   1. cyan analytic ticks on the straight reference centerline
#   2. orange simulated material-frame ticks overlaid on the same line
#
# The generated web video adds two live error traces:
#   - twist profile error: deviation from the analytical target profile in degrees
#   - bend leakage: transverse centerline motion as percent of cable length
#
# Newton's cable joint stores the already-discretized twist stiffness, not a
# separate material G field. This example derives that solver input from the
# mechanical torsion law:
#
#   G = E / (2 * (1 + nu))
#   J = pi * r^4 / 2
#   twist_stiffness = GJ / h
#
# and verifies that pure endpoint quaternion twist does not bend the cable.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_torsion_material_mapping
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_torsion_material_mapping --test --viewer null
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd._viewer import node_xyz, set_viewer_camera


@wp.kernel
def _spin_tip_kernel(
    tip_body: int,
    twist_rate: wp.array[float],
    dt: float,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    X = body_q0[tip_body]
    pos = wp.transform_get_translation(X)
    rot = wp.transform_get_rotation(X)
    axis_world = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    dq = wp.quat_from_axis_angle(axis_world, twist_rate[0] * dt)
    X_new = wp.transform(pos, wp.mul(dq, rot))
    body_q0[tip_body] = X_new
    body_q1[tip_body] = X_new


class Example:
    NUM_ELEMENTS = 24
    SEGMENT_LENGTH = 0.08
    CABLE_RADIUS = 0.012

    YOUNGS_MODULUS = 2.0e9
    POISSONS_RATIO = 0.30

    TARGET_TIP_TWIST = math.radians(90.0)
    RAMP_TIME = 3.0
    HOLD_TIME = 5.0
    MATERIAL_SCALING_CASES = (
        # label, group, E, radius, segment length, nu, explicit G
        ("baseline", "baseline", YOUNGS_MODULUS, CABLE_RADIUS, SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("E x0.25", "E", 0.25 * YOUNGS_MODULUS, CABLE_RADIUS, SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("E x2", "E", 2.0 * YOUNGS_MODULUS, CABLE_RADIUS, SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("E x8", "E", 8.0 * YOUNGS_MODULUS, CABLE_RADIUS, SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("r x0.60", "r", YOUNGS_MODULUS, 0.60 * CABLE_RADIUS, SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("r x0.80", "r", YOUNGS_MODULUS, 0.80 * CABLE_RADIUS, SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("r x1.25", "r", YOUNGS_MODULUS, 1.25 * CABLE_RADIUS, SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("r x1.60", "r", YOUNGS_MODULUS, 1.60 * CABLE_RADIUS, SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("h x0.50", "h", YOUNGS_MODULUS, CABLE_RADIUS, 0.50 * SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("h x2", "h", YOUNGS_MODULUS, CABLE_RADIUS, 2.0 * SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("h x4", "h", YOUNGS_MODULUS, CABLE_RADIUS, 4.0 * SEGMENT_LENGTH, POISSONS_RATIO, None),
        ("nu 0.00", "nu", YOUNGS_MODULUS, CABLE_RADIUS, SEGMENT_LENGTH, 0.00, None),
        ("nu 0.20", "nu", YOUNGS_MODULUS, CABLE_RADIUS, SEGMENT_LENGTH, 0.20, None),
        ("nu 0.45", "nu", YOUNGS_MODULUS, CABLE_RADIUS, SEGMENT_LENGTH, 0.45, None),
        (
            "G x0.50",
            "G",
            YOUNGS_MODULUS,
            CABLE_RADIUS,
            SEGMENT_LENGTH,
            None,
            0.50 * YOUNGS_MODULUS / (2.0 * (1.0 + POISSONS_RATIO)),
        ),
        (
            "G x2",
            "G",
            YOUNGS_MODULUS,
            CABLE_RADIUS,
            SEGMENT_LENGTH,
            None,
            2.0 * YOUNGS_MODULUS / (2.0 * (1.0 + POISSONS_RATIO)),
        ),
    )

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.cable_length = self.NUM_ELEMENTS * self.SEGMENT_LENGTH
        self.effective_twist_length = (self.NUM_ELEMENTS - 1) * self.SEGMENT_LENGTH

        (
            self.stretch_stiffness,
            self.bend_stiffness,
            self.twist_stiffness,
        ) = newton.utils.create_cable_stiffness_from_elastic_moduli(
            self.YOUNGS_MODULUS,
            self.CABLE_RADIUS,
            self.SEGMENT_LENGTH,
            poissons_ratio=self.POISSONS_RATIO,
        )

        self.stretch_damping = self.stretch_stiffness
        self.bend_damping = 4.0 * self.bend_stiffness
        self.twist_damping = 4.0 * self.twist_stiffness

        self.shear_modulus = self.YOUNGS_MODULUS / (2.0 * (1.0 + self.POISSONS_RATIO))
        self.polar_inertia = 0.5 * math.pi * self.CABLE_RADIUS**4

        builder = newton.ModelBuilder(gravity=0.0)
        points = newton.utils.create_straight_cable_points(
            start=wp.vec3(-0.5 * self.cable_length, 0.0, 0.45),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=self.cable_length,
            num_segments=self.NUM_ELEMENTS,
        )
        quats = newton.utils.create_parallel_transport_cable_quaternions(points)

        bodies, _joints = builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=self.CABLE_RADIUS,
            stretch_stiffness=self.stretch_stiffness,
            stretch_damping=self.stretch_damping,
            bend_stiffness=self.bend_stiffness,
            bend_damping=self.bend_damping,
            twist_stiffness=self.twist_stiffness,
            twist_damping=self.twist_damping,
            label="torsion_material_mapping",
            wrap_in_articulation=True,
            body_frame_origin="com",
        )

        self.bodies = list(map(int, bodies))
        self.cable_bodies = list(map(int, bodies))
        self.root_body = self.bodies[0]
        self.tip_body = self.bodies[-1]

        for body in (self.root_body, self.tip_body):
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        body_q = self.state_0.body_q.numpy()
        self.rest_pos = np.asarray([node_xyz(body_q[b], self.SEGMENT_LENGTH) for b in self.bodies], dtype=np.float64)
        self.rest_q = [np.asarray(body_q[b][3:7], dtype=np.float64) for b in self.bodies]
        self._twist_rate = self.TARGET_TIP_TWIST / self.RAMP_TIME
        self._twist_rate_np = np.zeros(1, dtype=np.float32)
        self._twist_rate_wp = wp.array(self._twist_rate_np, dtype=float)

        self.viewer.set_model(self.model)
        self.viewer.set_picking_linear_only_bodies(self.cable_bodies)
        set_viewer_camera(
            self.viewer,
            pos=wp.vec3(0.0, -3.4, 1.15),
            target=wp.vec3(0.0, 0.0, 0.45),
            fov=30.0,
            show_joints=False,
        )
        self.graph = None
        self.capture()

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

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
        return Example._quat_mul(Example._quat_mul(q, qv), Example._quat_conj(q))[:3]

    @staticmethod
    def _quat_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        if w < 0.0:
            x, y, z, w = -x, -y, -z, -w
        w = max(-1.0, min(1.0, w))
        angle = 2.0 * math.acos(w)
        s = math.sqrt(max(0.0, 1.0 - w * w))
        if s < 1.0e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
        return np.array([x / s, y / s, z / s], dtype=np.float64), angle

    @classmethod
    def material_scaling_validation(cls) -> dict[str, np.ndarray | float | list[str]]:
        labels = []
        groups = []
        youngs_moduli = []
        radii = []
        segment_lengths = []
        poissons_ratios = []
        shear_moduli = []
        polar_inertias = []
        twist_stiffnesses = []
        formula_stiffnesses = []

        for (
            label,
            group,
            youngs_modulus,
            radius,
            segment_length,
            poissons_ratio,
            shear_modulus_in,
        ) in cls.MATERIAL_SCALING_CASES:
            stiffness_kwargs = (
                {"shear_modulus": shear_modulus_in}
                if shear_modulus_in is not None
                else {"poissons_ratio": poissons_ratio}
            )
            _stretch, _bend, twist = newton.utils.create_cable_stiffness_from_elastic_moduli(
                youngs_modulus,
                radius,
                segment_length,
                **stiffness_kwargs,
            )
            shear_modulus = (
                float(shear_modulus_in)
                if shear_modulus_in is not None
                else youngs_modulus / (2.0 * (1.0 + poissons_ratio))
            )
            polar_inertia = 0.5 * math.pi * radius**4
            formula_twist = shear_modulus * polar_inertia / segment_length

            labels.append(label)
            groups.append(group)
            youngs_moduli.append(float(youngs_modulus))
            radii.append(float(radius))
            segment_lengths.append(float(segment_length))
            poissons_ratios.append(float("nan") if poissons_ratio is None else float(poissons_ratio))
            shear_moduli.append(float(shear_modulus))
            polar_inertias.append(float(polar_inertia))
            twist_stiffnesses.append(float(twist))
            formula_stiffnesses.append(float(formula_twist))

        twist_stiffnesses_np = np.asarray(twist_stiffnesses, dtype=np.float64)
        formula_stiffnesses_np = np.asarray(formula_stiffnesses, dtype=np.float64)

        formula_relative_error = np.abs(twist_stiffnesses_np / np.maximum(formula_stiffnesses_np, 1.0e-30) - 1.0)
        predicted_scale = formula_stiffnesses_np / formula_stiffnesses_np[0]
        helper_scale = twist_stiffnesses_np / twist_stiffnesses_np[0]

        return {
            "labels": labels,
            "groups": groups,
            "youngs_modulus": np.asarray(youngs_moduli, dtype=np.float64),
            "radius": np.asarray(radii, dtype=np.float64),
            "segment_length": np.asarray(segment_lengths, dtype=np.float64),
            "poissons_ratio": np.asarray(poissons_ratios, dtype=np.float64),
            "shear_modulus": np.asarray(shear_moduli, dtype=np.float64),
            "polar_inertia": np.asarray(polar_inertias, dtype=np.float64),
            "formula_stiffness": formula_stiffnesses_np,
            "helper_stiffness": twist_stiffnesses_np,
            "predicted_scale": predicted_scale,
            "helper_scale": helper_scale,
            "case_count": len(labels),
            "max_formula_relative_error": float(np.max(formula_relative_error)),
            "max_scale_relative_error": float(
                np.max(np.abs(helper_scale / np.maximum(predicted_scale, 1.0e-30) - 1.0))
            ),
        }

    def _commanded_tip_twist(self) -> float:
        ramp_fraction = min(max(self.sim_time / self.RAMP_TIME, 0.0), 1.0)
        return ramp_fraction * self.TARGET_TIP_TWIST

    def _update_twist_rate(self, twist_rate: float) -> None:
        self._twist_rate_np[0] = twist_rate
        self._twist_rate_wp.assign(self._twist_rate_np)

    def _simulate_substeps(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _spin_tip_kernel,
                dim=1,
                inputs=[self.tip_body, self._twist_rate_wp, self.sim_dt],
                outputs=[self.state_0.body_q, self.state_1.body_q],
            )
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self) -> None:
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate_substeps()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self, twist_rate: float) -> None:
        self._update_twist_rate(twist_rate)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()

    def step(self):
        t0 = self.sim_time
        t1 = min(t0 + self.frame_dt, self.RAMP_TIME)
        twist_rate = self._twist_rate if t1 > t0 else 0.0
        if 0.0 < t1 - t0 < self.frame_dt:
            twist_rate *= (t1 - t0) / self.frame_dt
        self.simulate(twist_rate)
        self.sim_time += self.frame_dt

    def _measure_twists(self) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        twists = []
        for body, rest_q in zip(self.bodies, self.rest_q, strict=True):
            q_now = np.asarray(body_q[body][3:7], dtype=np.float64)
            q_delta = self._quat_mul(q_now, self._quat_conj(rest_q))
            axis, angle = self._quat_axis_angle(q_delta)
            tangent = self._quat_rotate(rest_q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
            twists.append(float(np.dot(axis, tangent) * angle))
        twists_np = np.asarray(twists, dtype=np.float64)
        if twists_np[-1] < twists_np[0]:
            twists_np = -twists_np
        return twists_np

    def _current_pos(self) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        return np.asarray([node_xyz(body_q[b], self.SEGMENT_LENGTH) for b in self.bodies], dtype=np.float64)

    @staticmethod
    def _log_polyline(viewer, name: str, points: np.ndarray, color: tuple[float, float, float], width: float) -> None:
        viewer.log_lines(
            name,
            wp.array(points[:-1].astype(np.float32), dtype=wp.vec3),
            wp.array(points[1:].astype(np.float32), dtype=wp.vec3),
            color,
            width=width,
        )

    def _log_ticks(
        self,
        name: str,
        positions: np.ndarray,
        twists: np.ndarray,
        color: tuple[float, float, float],
        tick_len: float,
        width: float,
    ) -> None:
        starts = []
        ends = []
        for p, rest_q, twist in zip(positions, self.rest_q, twists, strict=True):
            tangent = self._quat_rotate(rest_q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
            normal = self._quat_rotate(rest_q, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            q_twist = np.array(
                [
                    tangent[0] * math.sin(0.5 * twist),
                    tangent[1] * math.sin(0.5 * twist),
                    tangent[2] * math.sin(0.5 * twist),
                    math.cos(0.5 * twist),
                ],
                dtype=np.float64,
            )
            normal_twisted = self._quat_rotate(q_twist, normal)
            starts.append(p - 0.5 * tick_len * normal_twisted)
            ends.append(p + 0.5 * tick_len * normal_twisted)

        self.viewer.log_lines(
            name,
            wp.array(np.asarray(starts, dtype=np.float32), dtype=wp.vec3),
            wp.array(np.asarray(ends, dtype=np.float32), dtype=wp.vec3),
            color,
            width=width,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        current_pos = self._current_pos()
        measured_twists = self._measure_twists()
        analytic_twists = np.linspace(0.0, self._commanded_tip_twist(), len(self.bodies), dtype=np.float64)

        self._log_ticks(
            "/torsion_material_mapping/analytic_ticks",
            self.rest_pos,
            analytic_twists,
            (0.0, 0.9, 1.0),
            0.18,
            0.014,
        )
        self._log_ticks(
            "/torsion_material_mapping/simulated_ticks",
            current_pos,
            measured_twists,
            (1.0, 0.55, 0.05),
            0.13,
            0.010,
        )
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.isfinite(body_q).all(), "non-finite body transforms"
        assert np.isfinite(body_qd).all(), "non-finite body velocities"

        twists = self._measure_twists()
        current_pos = self._current_pos()

        expected_profile = np.linspace(0.0, self.TARGET_TIP_TWIST, len(twists), dtype=np.float64)
        profile_err = twists - expected_profile
        profile_rms = float(np.sqrt(np.mean(profile_err**2)))
        max_profile_err = float(np.max(np.abs(profile_err)))

        transverse = current_pos - self.rest_pos
        transverse[:, 0] = 0.0
        max_transverse = float(np.max(np.linalg.norm(transverse, axis=1)))

        material_scaling = self.material_scaling_validation()

        assert abs(twists[0]) < math.radians(0.1), f"root should remain untwisted: {math.degrees(twists[0])} deg"
        assert abs(twists[-1] - self.TARGET_TIP_TWIST) < math.radians(1.0), (
            f"tip drive missed target: measured {math.degrees(twists[-1])} deg"
        )
        # Current 10-iteration baseline is about 0.015 deg RMS after the
        # clamped ramp; keep a small margin for platform-level solver noise.
        assert profile_rms < math.radians(0.05), f"twist profile is not linear enough: {math.degrees(profile_rms)} deg"
        assert max_profile_err < math.radians(0.10), (
            f"twist profile max error too large: {math.degrees(max_profile_err)} deg"
        )
        assert max_transverse / self.cable_length < 5.0e-4, (
            f"pure endpoint quaternion twist leaked into bend: {max_transverse / self.cable_length}"
        )
        assert material_scaling["max_formula_relative_error"] < 1.0e-12, (
            f"helper does not match GJ/h: {material_scaling['max_formula_relative_error']}"
        )
        assert material_scaling["max_scale_relative_error"] < 1.0e-9, (
            f"material scaling response is wrong: {material_scaling['max_scale_relative_error']}"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(60 * (Example.RAMP_TIME + Example.HOLD_TIME)) + 30)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
