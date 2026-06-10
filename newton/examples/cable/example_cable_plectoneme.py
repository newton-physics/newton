# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Cable Plectoneme Formation
#
# A cable hangs between two fixed endpoints and supercoils into a plectoneme
# when the endpoints are twisted:
#
#   1. the cable sags under gravity into a smooth loop (initialized on a
#      hanging arc to avoid a snap-in transient);
#   2. the two endpoints are gradually counter-twisted about the cable tangent;
#   3. past the buckling threshold the centerline leaves the plane and folds
#      back on itself into a plectoneme held open by rod self-contact.
#
# Plectoneme formation requires self-contact (the strands press against each
# other), so this uses hard-history contact at the true capsule radius with a
# healthy substep/iteration budget.
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


@wp.kernel
def _drive_endpoints_kernel(
    root_body: int,
    tip_body: int,
    root_pos: wp.vec3,
    tip_pos: wp.vec3,
    root_rest_rot: wp.quat,
    tip_rest_rot: wp.quat,
    twist_angle: wp.array(dtype=wp.float32),
    body_q0: wp.array(dtype=wp.transform),
    body_q1: wp.array(dtype=wp.transform),
):
    # Counter-twist: split the total end-to-end twist symmetrically. The angle
    # lives in a device array so the simulate() loop can be captured into a
    # CUDA graph and replayed with the angle updated from the host each frame.
    angle = twist_angle[0]
    root_axis = wp.quat_rotate(root_rest_rot, wp.vec3(0.0, 0.0, 1.0))
    tip_axis = wp.quat_rotate(tip_rest_rot, wp.vec3(0.0, 0.0, 1.0))
    root_rot = wp.mul(wp.quat_from_axis_angle(root_axis, -0.5 * angle), root_rest_rot)
    tip_rot = wp.mul(wp.quat_from_axis_angle(tip_axis, 0.5 * angle), tip_rest_rot)
    root_pose = wp.transform(root_pos, root_rot)
    tip_pose = wp.transform(tip_pos, tip_rot)

    body_q0[root_body] = root_pose
    body_q1[root_body] = root_pose
    body_q0[tip_body] = tip_pose
    body_q1[tip_body] = tip_pose


class Example:
    # Geometry of the hanging span.
    NUM_ELEMENTS = 80
    END_SEPARATION = 1.20
    TOP_HEIGHT = 1.80
    SAG_DEPTH = 0.90

    # Total counter-twist applied end-to-end (tip turns - root turns).
    TWIST_TURNS = 6.0

    SETTLE_TIME = 2.0
    TWIST_TIME = 8.0
    HOLD_TIME = 3.0
    TOTAL_TIME = SETTLE_TIME + TWIST_TIME + HOLD_TIME

    # Twist-dominated material: soft bend, stiff twist -> the rod prefers to
    # supercoil rather than store all twist in material rotation.
    STRETCH_STIFFNESS = 1.0e6
    BEND_STIFFNESS = 6.0
    TWIST_STIFFNESS = 400.0
    BEND_DAMPING = 0.02
    TWIST_DAMPING = 0.02
    GRAVITY = -9.81

    # Self-contact (true radius, hard-history). Radius is kept below half the
    # segment length so rest-state neighbours do not overlap; the gap catches
    # approaching strands before they cross.
    CONTACT_STIFFNESS = 5.0e4
    CONTACT_DAMPING = 0.0
    CONTACT_TOPOLOGICAL_FILTER_SPAN = 2

    FPS = 60
    SIM_SUBSTEPS = 8
    SIM_ITERATIONS = 20

    # Symmetry-breaking seed so the supercoil picks a deterministic handedness.
    SEED_BODY_OFFSET_Y = 1.0e-3

    MIN_OUT_OF_PLANE_SPAN_RADII = 3.0

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = self.FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(getattr(args, "substeps", None) or self.SIM_SUBSTEPS)
        # The captured substep loop ping-pongs state_0/state_1, so an even count
        # keeps the buffers in their original roles after each frame replay.
        if self.sim_substeps % 2 == 1:
            self.sim_substeps += 1
        self.sim_iterations = int(getattr(args, "iterations", None) or self.SIM_ITERATIONS)
        self.sim_dt = self.frame_dt / self.sim_substeps

        nodes = self._hanging_arc_nodes()
        seg_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        self.rest_segment_lengths = seg_lengths.copy()
        self.segment_length = float(np.mean(seg_lengths))
        self.cable_length = float(np.sum(seg_lengths))
        # Keep the capsule thinner than half a segment so neighbours are clear
        # at rest, but thick enough to form a visible, contact-bearing coil.
        self.cable_radius = 0.42 * self.segment_length
        self.contact_gap = 0.6 * self.segment_length

        self.twist_turns = self.TWIST_TURNS
        self.target_twist = 2.0 * math.pi * self.twist_turns

        points = [wp.vec3(*p) for p in nodes]
        builder = newton.ModelBuilder(gravity=self.GRAVITY)
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=self.CONTACT_STIFFNESS,
            kd=self.CONTACT_DAMPING,
            gap=self.contact_gap,
        )
        bodies, joints = builder.add_rod(
            positions=points,
            quaternions=None,
            radius=self.cable_radius,
            cfg=shape_cfg,
            stretch_stiffness=self.STRETCH_STIFFNESS,
            stretch_damping=0.0,
            bend_stiffness=self.BEND_STIFFNESS,
            bend_damping=self.BEND_DAMPING,
            twist_stiffness=self.TWIST_STIFFNESS,
            twist_damping=self.TWIST_DAMPING,
            label="plectoneme",
            wrap_in_articulation=False,
        )
        self.bodies = list(map(int, bodies))
        self.joints = list(map(int, joints))
        self._filter_near_rod_collision_pairs(builder, self.bodies, self.CONTACT_TOPOLOGICAL_FILTER_SPAN)

        self.root_body = self.bodies[0]
        self.tip_body = self.bodies[-1]
        for body in (self.root_body, self.tip_body):
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)
        builder.add_articulation(self.joints, label="plectoneme_articulation")

        builder.color()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_contact_hard=True,
            rigid_contact_history=True,
            rigid_contact_stick_motion_eps=0.0,
            rigid_contact_stick_freeze_translation_eps=0.0,
            rigid_contact_stick_freeze_angular_eps=0.0,
            rigid_body_contact_buffer_size=1024,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.contacts = self.model.contacts(collision_pipeline=pipeline)

        body_q = self.state_0.body_q.numpy()
        self.rest_pos = np.asarray([body_q[b][:3] for b in self.bodies], dtype=np.float64)
        self.root_rest_pos = self.rest_pos[0].copy()
        self.tip_rest_pos = self.rest_pos[-1].copy()
        self.root_rest_rot = wp.quat(*body_q[self.root_body][3:7])
        self.tip_rest_rot = wp.quat(*body_q[self.tip_body][3:7])

        # Symmetry-breaking seed at the mid body.
        mid = self.NUM_ELEMENTS // 2
        body_q_np = self.state_0.body_q.numpy()
        body_q_np[self.bodies[mid], 1] += self.SEED_BODY_OFFSET_Y
        self.state_0.body_q.assign(body_q_np)
        self.state_1.body_q.assign(body_q_np)

        # End-to-end twist angle, kept in a device array so the simulate() loop
        # can be captured into a CUDA graph and replayed with a host update.
        self.twist_angle = wp.array([0.0], dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.0, -4.0, 1.30), pitch=0.0, yaw=0.0)
            if hasattr(self.viewer, "camera"):
                self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 1.05))
                self.viewer.camera.fov = 35.0

        self.graph = None
        self.capture()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @classmethod
    def _hanging_arc_nodes(cls) -> np.ndarray:
        u = np.linspace(0.0, 1.0, cls.NUM_ELEMENTS + 1)
        x = (u - 0.5) * cls.END_SEPARATION
        y = np.zeros_like(u)
        z = cls.TOP_HEIGHT - cls.SAG_DEPTH * np.sin(math.pi * u)
        return np.column_stack([x, y, z]).astype(np.float64)

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = min(1.0, max(0.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def _filter_near_rod_collision_pairs(self, builder, bodies: list[int], span: int) -> None:
        for i, body_i in enumerate(bodies):
            for j in range(i + 1, min(len(bodies), i + span + 1)):
                body_j = bodies[j]
                for shape_i in builder.body_shapes.get(body_i, []):
                    for shape_j in builder.body_shapes.get(body_j, []):
                        builder.add_shape_collision_filter_pair(int(shape_i), int(shape_j))

    def _command(self, t: float) -> tuple[float, str]:
        t = float(t)
        if t <= self.SETTLE_TIME:
            return 0.0, "settle"
        t -= self.SETTLE_TIME
        a = self._smoothstep(t / self.TWIST_TIME)
        if t <= self.TWIST_TIME:
            return self.target_twist * a, "twist"
        return self.target_twist, "twist hold"

    def _apply_command(self) -> None:
        wp.launch(
            _drive_endpoints_kernel,
            dim=1,
            inputs=[
                self.root_body,
                self.tip_body,
                wp.vec3(*self.root_rest_pos),
                wp.vec3(*self.tip_rest_pos),
                self.root_rest_rot,
                self.tip_rest_rot,
                self.twist_angle,
            ],
            outputs=[self.state_0.body_q, self.state_1.body_q],
            device=self.model.device,
        )

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def capture(self) -> None:
        """Capture the substep loop into a CUDA graph for fast replay."""
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self._apply_command()
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.set_rigid_history_update(True)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # The twist ramp is smooth, so a single per-frame angle (held across the
        # frame's substeps) is sufficient and lets the substep loop be a graph.
        angle, _stage = self._command(self.sim_time)
        self.twist_angle.assign(np.array([angle], dtype=np.float32))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    def current_points(self) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()
        return np.asarray([body_q[b][:3] for b in self.bodies], dtype=np.float64)

    def _min_non_neighbor_distance(self, points: np.ndarray, span: int) -> float:
        pts = np.asarray(points, dtype=np.float64)
        best = float("inf")
        n = len(pts)
        for i in range(n):
            for j in range(i + span + 1, n):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d < best:
                    best = d
        return best

    def metrics(self) -> dict[str, float]:
        pts = self.current_points()
        angle, _stage = self._command(self.sim_time)
        min_strand = self._min_non_neighbor_distance(pts, self.CONTACT_TOPOLOGICAL_FILTER_SPAN)
        out_of_plane_span = float(np.ptp(pts[:, 1]))
        endpoint_drift = max(
            float(np.linalg.norm(pts[0] - self.root_rest_pos)),
            float(np.linalg.norm(pts[-1] - self.tip_rest_pos)),
        )
        seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        endpoint_segment_ratio = max(
            float(seg_lengths[0] / max(self.rest_segment_lengths[0], 1.0e-12)),
            float(seg_lengths[-1] / max(self.rest_segment_lengths[-1], 1.0e-12)),
        )
        return {
            "twist_command_deg": math.degrees(float(angle)),
            "target_twist_deg": math.degrees(self.target_twist),
            "out_of_plane_span": out_of_plane_span,
            "out_of_plane_span_radii": float(out_of_plane_span / max(self.cable_radius, 1.0e-12)),
            "min_strand_distance": float(min_strand),
            "min_strand_distance_radii": float(min_strand / max(self.cable_radius, 1.0e-12)),
            "endpoint_drift": endpoint_drift,
            "endpoint_segment_ratio": endpoint_segment_ratio,
            "formation": min(1.0, out_of_plane_span / (self.MIN_OUT_OF_PLANE_SPAN_RADII * self.cable_radius)),
        }

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_final(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--iterations", dest="iterations", type=int, default=None)
    parser.add_argument("--substeps", dest="substeps", type=int, default=None)
    parser.set_defaults(num_frames=int(Example.FPS * Example.TOTAL_TIME))
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
