# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example VBD Rigid Contact Friction

Compare sliding, torsional and rolling resistance with independent Coulomb
references. A sliding box decelerates at mu*g. A spinning solid sphere
decelerates at mu_torsional*m*g/I, and a sphere rolling without slip
decelerates at mu_rolling*g/(1.4*r). The angular coefficients have units of
meters; all lanes use isotropic material friction. Directional contact response
comes from body inertia, not a direction-dependent material coefficient.

The box and the two spheres with angular resistance stop. Two control
spheres with their angular coefficient set to zero keep spinning or rolling.
White markers show the reference positions, and colored spokes make sphere
rotation visible. The references use backward-Euler displacement with
velocity clamped to zero at stopping; they are independent of VBD's solver.
The free-spin control accounts for the normalized-Euler quaternion step:
its reconstructed rate is ``2*atan(h*omega/2)/h``, slightly below ``omega``.

Two additional setups test static holding over two seconds. A sphere has
axle-like inertia, so its contact responds about twenty times more readily
across the axle than along it. An oblique push at ground level, at half the
friction limit, admits a stationary reference. Three boxes of one color
group are stacked on a 20 degree ramp, where tan(20 deg) is below mu.

Success criteria:
- The sliding box stops at the white reference marker.
- The sphere with torsional resistance stops spinning; its zero-coefficient control keeps spinning.
- The sphere with rolling resistance stops at its reference marker without slipping; its control keeps rolling.
- The measured velocities and travel agree with the independent Coulomb references.
- Over the first two seconds, the axle surface-displacement bound stays below 2 mm and stack creep below 0.5 mm.

Runs:
- `python -m newton.examples rigid_contact_friction`
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.free_spin_reference = 10.0
        self.radius = 0.1
        self.gravity = 9.81
        self.initial = np.array([2.0, 10.0, 10.0, 0.6, 0.6])
        self.deceleration = np.array(
            [0.5 * self.gravity, 0.005 * self.gravity / 0.004, 0.0, 0.005 * self.gravity / (1.4 * self.radius), 0.0]
        )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -self.gravity))
        ground_cfg = builder.ShapeConfig(ke=1.0e5, kd=100.0, mu=0.5, mu_torsional=0.005, mu_rolling=0.005)
        builder.add_ground_plane(cfg=ground_cfg)
        self.rows = (0.9, 0.4, 0.0, -0.5, -0.9)
        labels = ("Sliding", "Torsion on", "Torsion off", "Rolling on", "Rolling off")
        for index, y in enumerate(self.rows):
            is_box = index == 0
            half_height = 0.05 if is_box else self.radius
            body = builder.add_body(xform=wp.transform((-0.7, y, half_height), wp.quat_identity()), label=labels[index])
            cfg = builder.ShapeConfig(
                ke=1.0e5,
                kd=100.0,
                mu=0.0 if index in (1, 2) else 0.5,
                mu_torsional=0.005 if index == 1 else 0.0,
                mu_rolling=0.005 if index == 3 else 0.0,
            )
            if is_box:
                cfg.density = 1.0 / (0.2 * 0.2 * 0.1)
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=half_height, cfg=cfg)
            else:
                cfg.density = 1.0 / (4.0 / 3.0 * math.pi * self.radius**3)
                builder.add_shape_sphere(body, radius=self.radius, cfg=cfg)

        # Holding lanes. The 1 kg sphere has axle-like inertia along x: its moment about
        # the axle is one thirtieth of a solid sphere's, the other two are unchanged.
        # It rests at x = 0, where single precision resolves its small holding displacements.
        hold_cfg = builder.ShapeConfig(ke=1.0e5, kd=100.0, mu=0.5, mu_torsional=0.0, mu_rolling=0.0, density=0.0)
        moment = 0.4 * self.radius**2
        self.axle = builder.add_body(
            xform=wp.transform((0.0, -1.4, self.radius), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.diag([moment / 30.0, moment, moment])),
            lock_inertia=True,
            label="Directional response: axle sphere",
        )
        builder.add_shape_sphere(self.axle, radius=self.radius, cfg=hold_cfg)
        slope = math.radians(20.0)
        self.ramp_normal = np.array([math.sin(slope), 0.0, math.cos(slope)])
        self.downhill = np.array([math.cos(slope), 0.0, -math.sin(slope)])
        tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), slope)
        ramp_top = np.array([-0.3, 1.6, 0.3])
        builder.add_shape_box(
            -1, xform=wp.transform(ramp_top - 0.02 * self.ramp_normal, tilt), hx=0.45, hy=0.2, hz=0.02, cfg=hold_cfg
        )
        hold_cfg.density = 1.0 / (0.2 * 0.2 * 0.1)
        self.stack = []
        for level in range(3):
            body = builder.add_body(
                xform=wp.transform(ramp_top + (0.05 + 0.1 * level) * self.ramp_normal, tilt),
                label=f"Same-color ramp stack {level + 1}",
            )
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.05, cfg=hold_cfg)
            self.stack.append(body)

        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=10)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.push = wp.zeros(self.model.body_count, dtype=wp.spatial_vector)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.4, -3.3, 2.5), -34.0, 120.0)

        # Acquire stationary normal support before applying the measured motion.
        for _ in range(6):
            self.simulate()
        self.start = self.state_0.body_q.numpy().copy()
        velocity = np.zeros((self.model.body_count, 6), dtype=np.float32)
        velocity[0, 0] = self.initial[0]
        velocity[1:3, 5] = self.initial[1:3]
        velocity[3:5, 0] = self.initial[3:5]
        velocity[3:5, 4] = self.initial[3:5] / self.radius
        self.state_0.body_qd.assign(velocity)
        # Push the axle sphere at its contact point: a force and its moment about the center.
        push = np.zeros((self.model.body_count, 6), dtype=np.float32)
        push[self.axle, :3] = 0.5 * 0.5 * self.gravity * np.array([0.6, 0.8, 0.0])
        push[self.axle, 3:] = np.cross([0.0, 0.0, -self.radius], push[self.axle, :3])
        self.push.assign(push)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        self.max_velocity_error = np.zeros(5)
        self.max_rolling_slip = 0.0
        self.checked_frames = 0
        self.max_hold_displacement = np.zeros(2)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.copy(self.state_0.body_f, self.push)
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.frame += 1
        self.sim_time = self.frame * self.frame_dt
        for _ in range(self.sim_substeps):
            self.free_spin_reference = 2.0 * math.atan(0.5 * self.sim_dt * self.free_spin_reference) / self.sim_dt

    def _reference(self):
        """Return velocities and travel from constant Coulomb deceleration and backward Euler."""
        steps = self.frame * self.sim_substeps
        velocity = np.maximum(self.initial - self.deceleration * steps * self.sim_dt, 0.0)
        velocity[2] = self.free_spin_reference
        moving = np.full(5, steps, dtype=np.int64)
        active = self.deceleration > 0.0
        moving[active] = np.minimum(steps, np.floor(self.initial[active] / (self.deceleration[active] * self.sim_dt)))
        travel = self.sim_dt * (moving * self.initial - 0.5 * self.deceleration * self.sim_dt * moving * (moving + 1))
        return velocity, travel

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        body_q = self.state_0.body_q.numpy()
        _, travel = self._reference()
        starts, ends = [], []
        for index in (0, 3, 4):
            x = self.start[index, 0] + travel[index]
            y = self.rows[index]
            starts.append(wp.vec3(x, y - 0.15, 0.015))
            ends.append(wp.vec3(x, y + 0.15, 0.015))
        # The holding lanes keep their start: under the axle sphere, and at the downhill face of the stack.
        axle_start = self.start[self.axle, :3]
        starts.append(wp.vec3(axle_start[0], axle_start[1] - 0.15, 0.015))
        ends.append(wp.vec3(axle_start[0], axle_start[1] + 0.15, 0.015))
        face = self.start[self.stack[0], :3] + 0.1 * self.downhill - 0.035 * self.ramp_normal
        starts.append(wp.vec3(face[0], face[1] - 0.15, face[2]))
        ends.append(wp.vec3(face[0], face[1] + 0.15, face[2]))
        for side in (-0.1, 0.1):
            foot = face + np.array([0.0, side, 0.0])
            starts.append(wp.vec3(foot))
            ends.append(wp.vec3(foot + 0.32 * self.ramp_normal))
        self.viewer.log_lines(
            "reference", wp.array(starts, dtype=wp.vec3), wp.array(ends, dtype=wp.vec3), (1.0, 1.0, 1.0), width=0.006
        )
        starts, ends = [], []
        for index in (1, 2, 3, 4):
            transform = wp.transform(body_q[index, :3], body_q[index, 3:])
            starts.append(wp.vec3(body_q[index, :3]))
            ends.append(wp.transform_point(transform, wp.vec3(self.radius * 1.3, 0.0, 0.0)))
        transform = wp.transform(body_q[self.axle, :3], body_q[self.axle, 3:])
        starts.append(wp.transform_point(transform, wp.vec3(-self.radius * 1.3, 0.0, 0.0)))
        ends.append(wp.transform_point(transform, wp.vec3(self.radius * 1.3, 0.0, 0.0)))
        self.viewer.log_lines(
            "rotation", wp.array(starts, dtype=wp.vec3), wp.array(ends, dtype=wp.vec3), (1.0, 0.55, 0.1), width=0.008
        )
        self.viewer.end_frame()

    def _holding_displacement(self, body_q):
        """Return an axle surface-displacement bound and the largest downhill stack displacement [m]."""
        displacement = body_q[:, :3].astype(float) - self.start[:, :3]
        current = body_q[self.axle, 3:].astype(float)
        initial = self.start[self.axle, 3:].astype(float)
        current /= np.linalg.norm(current)
        initial /= np.linalg.norm(initial)
        turn = 2.0 * math.acos(min(1.0, abs(float(current @ initial))))
        # Translation plus the rotation arc bounds surface displacement; it is not measured contact slip.
        axle_bound = float(np.linalg.norm(displacement[self.axle])) + self.radius * turn
        stack_creep = float(np.abs(displacement[self.stack] @ self.downhill).max())
        return np.array([axle_bound, stack_creep])

    def gui(self, ui):
        """Identify the controls and display the two holding measures."""
        ui.text("Sliding: braking accuracy control")
        ui.text("Torsion and rolling: enabled / disabled pairs")
        ui.text("Axle sphere: directional response, isotropic material")
        ui.text("Ramp stack: sliding convergence, shared body color")
        ui.text("White lines: references; orange spokes: rotation")
        axle_bound, stack_creep = self._holding_displacement(self.state_0.body_q.numpy())
        ui.separator()
        ui.text(f"Axle displacement bound: {1000 * axle_bound:.3f} mm")
        ui.text(f"Stack downhill displacement: {1000 * stack_creep:.3f} mm")

    def test_post_step(self):
        """Check motion throughout the trajectory, including the stopping transitions."""
        if self.checked_frames == self.frame:
            return
        qd = self.state_0.body_qd.numpy()
        measured = np.array([qd[0, 0], qd[1, 5], qd[2, 5], qd[3, 0], qd[4, 0]])
        reference, _ = self._reference()
        self.max_velocity_error = np.maximum(self.max_velocity_error, np.abs(measured - reference) / self.initial)
        slip = qd[3:5, :2] + self.radius * np.column_stack((-qd[3:5, 4], qd[3:5, 3]))
        self.max_rolling_slip = max(self.max_rolling_slip, float(np.linalg.norm(slip, axis=1).max()))
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Contact poses must remain finite"
        # Use the same holding interval in every valid run; finite-budget creep grows with run length.
        if self.frame <= 2 * self.fps:
            self.max_hold_displacement = np.maximum(self.max_hold_displacement, self._holding_displacement(body_q))
        self.checked_frames = self.frame
        assert np.isfinite(qd).all(), "Contact velocities must remain finite"

    def test_final(self):
        """Verify active resistance and disabled controls against the independent references."""
        assert self.sim_time >= 2.0, "Run at least 120 frames to include all stopping transitions"
        assert self.checked_frames == self.frame, "Run with --test to check the frame-sampled velocity path"
        assert np.max(self.max_velocity_error) < 0.01, f"Velocity error by lane: {self.max_velocity_error}"
        assert self.max_rolling_slip < 0.001, f"Rolling contact slip: {self.max_rolling_slip} m/s"
        body_q = self.state_0.body_q.numpy()
        _, travel = self._reference()
        for index in (0, 3, 4):
            displacement = body_q[index, 0] - self.start[index, 0]
            assert abs(displacement - travel[index]) < 0.005, (
                f"Lane {index}: travel {displacement}, reference {travel[index]}"
            )
        axle_bound, stack_creep = self.max_hold_displacement
        assert axle_bound < 2.0e-3, f"Axle surface-displacement bound reached {axle_bound} m"
        assert stack_creep < 5.0e-4, f"Stacked boxes crept {stack_creep} m along the ramp"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=150)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
