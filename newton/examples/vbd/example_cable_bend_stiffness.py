# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Bend Stiffness Validation
#
# Three horizontal cantilever cables sit side by side along Y. Each cable has
# the same geometry, stretch stiffness, and load, but a different bend
# stiffness. The root body is kinematic; the tip body receives the same
# transverse force in -Z.
#
# After settling, the example asserts the bend response:
#
#     delta_i * k_bend_i  must be approximately constant across cables.
#
# This is the primary check. It avoids depending on a continuum constant-factor
# calibration and verifies the discrete cable path behaves like a linear bend
# spring with the correct stiffness ratios.
#
# A reference Euler-Bernoulli deflection (delta = F * L^3 / (3 * E*I)) is also
# rendered using E*I = bend_stiffness * segment_length. The discrete rod has a
# slightly different effective continuum stiffness, so that overlay is a guide
# rather than the pass/fail gate.
#
# Other assertions:
#   - Deflection decreases monotonically as bend stiffness increases.
#   - Tip stays close to its original Y, so the bend load does not leak sideways.
#   - Tip X foreshortening remains within a geometric bound.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_bend_stiffness
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.vbd.example_cable_bend_stiffness --test --viewer null
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd._viewer import node_xyz, set_viewer_camera


class Example:
    """Cantilever bend stiffness validation."""

    NUM_ELEMENTS = 16
    SEGMENT_LENGTH = 0.10
    CABLE_RADIUS = 0.01
    TIP_FORCE_MAX = 0.5  # N, applied in -Z at the tip body
    BEND_STIFFNESS_VALUES = (100.0, 300.0, 900.0)  # 3x ratio between consecutive cables
    HOOKE_REFERENCE_DELTA_TIMES_K = 41.0
    Y_SEPARATION = 0.40

    # Slow ramp + long hold reaches a quiet steady state without oscillating.
    RAMP_TIME = 2.0  # seconds: linear ramp 0 -> TIP_FORCE_MAX
    HOLD_TIME = 6.0  # seconds: hold at TIP_FORCE_MAX before measuring

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
        self.num_cables = len(self.BEND_STIFFNESS_VALUES)

        builder = newton.ModelBuilder(gravity=0.0)

        self.tip_bodies: list[int] = []
        self.tip_rest_z: list[float] = []

        for i, bend_stiffness in enumerate(self.BEND_STIFFNESS_VALUES):
            y_pos = (i - (self.num_cables - 1) * 0.5) * self.Y_SEPARATION
            start = wp.vec3(0.0, y_pos, 0.0)
            points = newton.utils.create_straight_cable_points(
                start=start,
                direction=wp.vec3(1.0, 0.0, 0.0),
                length=self.cable_length,
                num_segments=self.NUM_ELEMENTS,
            )
            quats = newton.utils.create_parallel_transport_cable_quaternions(points)

            # Twist != bend exercises the split stiffness path while the applied
            # load remains pure bending.
            twist_stiffness = bend_stiffness * 0.77
            bend_damping = bend_stiffness
            twist_damping = twist_stiffness

            rod_bodies, _ = builder.add_rod(
                positions=points,
                quaternions=quats,
                radius=self.CABLE_RADIUS,
                stretch_stiffness=1.0e6,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                twist_stiffness=twist_stiffness,
                twist_damping=twist_damping,
                label=f"cantilever_k{int(bend_stiffness)}",
                body_frame_origin="com",
            )
            # Zero mass + zero inertia in Newton's VBD makes the root kinematic.
            root_body = rod_bodies[0]
            builder.body_mass[root_body] = 0.0
            builder.body_inv_mass[root_body] = 0.0
            builder.body_inertia[root_body] = wp.mat33(0.0)
            builder.body_inv_inertia[root_body] = wp.mat33(0.0)

            self.tip_bodies.append(int(rod_bodies[-1]))

        builder.color()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self._tip_rest_x: list[float] = []
        self._tip_rest_y: list[float] = []
        body_q = self.state_0.body_q.numpy()
        for tip in self.tip_bodies:
            node = node_xyz(body_q[tip], self.SEGMENT_LENGTH)
            self._tip_rest_x.append(float(node[0]))
            self._tip_rest_y.append(float(node[1]))
            self.tip_rest_z.append(float(node[2]))

        self.viewer.set_model(self.model)
        set_viewer_camera(
            self.viewer,
            pos=wp.vec3(0.5 * self.cable_length, -3.1, 0.85),
            target=wp.vec3(0.5 * self.cable_length, 0.0, -0.02),
            fov=32.0,
        )

        # body_f is a spatial_vector per body: (f_x, f_y, f_z, tau_x, tau_y, tau_z)
        # in world frame. The buffer is updated each frame during the load ramp.
        self._wrench_np = np.zeros((self.model.body_count, 6), dtype=np.float32)
        self.tip_wrench = wp.array(self._wrench_np, dtype=wp.spatial_vector)
        self.graph = None
        self.capture()

    def _force_at_time(self, t: float) -> float:
        """Linear ramp from 0 to TIP_FORCE_MAX over RAMP_TIME, then hold."""
        if t <= 0.0:
            return 0.0
        if t >= self.RAMP_TIME:
            return self.TIP_FORCE_MAX
        return self.TIP_FORCE_MAX * (t / self.RAMP_TIME)

    def _update_tip_wrench(self, force_now: float) -> None:
        self._wrench_np.fill(0.0)
        for tip in self.tip_bodies:
            self._wrench_np[tip, 2] = -force_now
        self.tip_wrench.assign(self._wrench_np)

    def _simulate_substeps(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_0.body_f.assign(self.tip_wrench)
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

    def simulate(self, F_now: float) -> None:
        self._update_tip_wrench(F_now)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()

    def step(self):
        F_now = self._force_at_time(self.sim_time)
        self.simulate(F_now)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._log_hooke_reference()
        self.viewer.end_frame()

    @staticmethod
    def _log_polyline(viewer, name: str, points: np.ndarray, color: tuple[float, float, float], width: float) -> None:
        viewer.log_lines(
            name,
            wp.array(points[:-1].astype(np.float32), dtype=wp.vec3),
            wp.array(points[1:].astype(np.float32), dtype=wp.vec3),
            color,
            width=width,
        )

    def _log_point_markers(
        self,
        name: str,
        points: np.ndarray,
        color: tuple[float, float, float],
        radius: float,
    ) -> None:
        self.viewer.log_points(
            name,
            wp.array(points.astype(np.float32), dtype=wp.vec3),
            wp.array(np.full(len(points), radius, dtype=np.float32), dtype=wp.float32),
            wp.array(np.tile(np.asarray(color, dtype=np.float32), (len(points), 1)), dtype=wp.vec3),
        )

    def _log_hooke_reference(self) -> None:
        curve_color = (0.0, 0.85, 0.35)
        marker_color = (0.0, 1.0, 0.45)
        markers = []
        for i, bend_stiffness in enumerate(self.BEND_STIFFNESS_VALUES):
            x_tip = self._tip_rest_x[i]
            y = self._tip_rest_y[i]
            z0 = self.tip_rest_z[i]
            delta = self.HOOKE_REFERENCE_DELTA_TIMES_K / bend_stiffness
            xs = np.linspace(0.0, x_tip, 32, dtype=np.float64)
            s = xs / max(x_tip, 1.0e-12)
            zs = z0 - delta * (s * s * (3.0 - s) * 0.5)
            points = np.column_stack((xs, np.full_like(xs, y), zs))
            self._log_polyline(self.viewer, f"/bend_reference/hooke_curve_{i}", points, curve_color, 0.014)

            tip = np.array([x_tip, y, z0 - delta], dtype=np.float64)
            markers.append(tip)
            bar = np.array(
                [
                    [x_tip - 0.055, y, z0 - delta],
                    [x_tip + 0.055, y, z0 - delta],
                ],
                dtype=np.float64,
            )
            self._log_polyline(self.viewer, f"/bend_reference/tip_bar_{i}", bar, marker_color, 0.022)

        self._log_point_markers("/bend_reference/tip_markers", np.asarray(markers), marker_color, 0.018)

    def _measured_tip_state(self) -> list[tuple[float, float, float]]:
        """Return per-cable (deflection_z, displacement_x, displacement_y) at the tip."""
        body_q = self.state_0.body_q.numpy()
        out = []
        for i in range(self.num_cables):
            tip_pos = node_xyz(body_q[self.tip_bodies[i]], self.SEGMENT_LENGTH)
            dz = self.tip_rest_z[i] - float(tip_pos[2])
            dx = float(tip_pos[0] - self._tip_rest_x[i])
            dy = float(tip_pos[1] - self._tip_rest_y[i])
            out.append((dz, dx, dy))
        return out

    def _eb_reference_deflection(self, bend_stiffness: float) -> float:
        L = self.cable_length
        EI = bend_stiffness * self.SEGMENT_LENGTH
        return self.TIP_FORCE_MAX * L**3 / (3.0 * EI)

    def test_final(self):
        states = self._measured_tip_state()
        deflections = [s[0] for s in states]
        dispx = [s[1] for s in states]
        dispy = [s[2] for s in states]

        assert all(np.isfinite(d) for d in deflections), f"non-finite deflections: {deflections}"

        for i in range(self.num_cables - 1):
            assert deflections[i] > deflections[i + 1], (
                f"deflection should decrease with bend stiffness, got {deflections[i]:.4f} <= {deflections[i + 1]:.4f}"
            )

        invariants = [d * k for d, k in zip(deflections, self.BEND_STIFFNESS_VALUES, strict=True)]
        ref = invariants[0]
        for i, inv in enumerate(invariants):
            rel = abs(inv - ref) / abs(ref)
            assert rel < 0.10, (
                f"cable {i}: delta*k = {inv:.4f} differs from cable 0 ({ref:.4f}) "
                f"by {100 * rel:.1f}% (>10%); bend stiffness is not behaving linearly"
            )

        for i, dy in enumerate(dispy):
            rel = abs(dy) / self.cable_length
            assert rel < 0.005, (
                f"cable {i}: tip Y drifted {dy:+.4f} m ({100 * rel:.2f}% of L); "
                f"pure transverse force should not move the tip in Y"
            )

        for i, (d, x) in enumerate(zip(deflections, dispx, strict=True)):
            geometric_bound = (d * d) / (2.0 * self.cable_length) + 0.05 * self.cable_length
            assert -geometric_bound < x < 0.005 * self.cable_length, (
                f"cable {i}: tip X displacement {x:+.4f} m outside expected "
                f"foreshortening range [-{geometric_bound:.4f}, {0.005 * self.cable_length:.4f}]"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(60 * (Example.RAMP_TIME + Example.HOLD_TIME)) + 30)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
