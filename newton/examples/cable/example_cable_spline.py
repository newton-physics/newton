# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Spline
#
# Demonstrates ModelBuilder.add_cable_spline(): cables are authored as
# Catmull-Rom splines through control points and assembled into capsule
# chains with rotation-minimizing frames. Five pre-shaped cables are
# dropped onto the ground: two open ropes (a single and a double overhand
# knot) and three closed loops (trefoil, figure-eight knot, and a
# cinquefoil torus knot). Because cable strain is measured relative to
# the built configuration, all cables hold their knotted rest shapes
# while settling under gravity and contact.
#
# Run interactively:
#   uv run --extra examples python -m newton.examples.cable.example_cable_spline
#
# Run as a test:
#   uv run --extra examples python -m newton.examples.cable.example_cable_spline --test --viewer null
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


def create_overhand_knot_control_points(scale: float, height: float) -> list[wp.vec3]:
    """Control polygon of a rope tied into a loose overhand knot, with straight leads."""
    points = [
        [2.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 1.5],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -0.75],
        [0.0, 0.0, -0.75],
        [0.0, 0.0, 1.25],
        [-1.0, 0.0, 1.5],
        [-2.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0],
    ]
    return [wp.vec3(scale * p[0], scale * p[1], scale * p[2] + height) for p in points]


def create_double_overhand_knot_control_points(scale: float, center: wp.vec3) -> list[wp.vec3]:
    """Control polygon of a rope tied into two chained overhand knots, with straight leads."""
    core = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 1.5],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -0.75],
            [0.0, 0.0, -0.75],
            [0.0, 0.0, 1.25],
            [-1.0, 0.0, 1.5],
            [-2.0, 0.0, 0.0],
        ]
    )
    points = [np.array([2.0, 0.0, 0.0]), np.array([1.4, 0.0, 0.0])]
    points += list(core)
    points += list(core + np.array([-3.5, 0.0, 0.0]))
    points += [np.array([-6.5, 0.0, 0.0])]
    return [center + wp.vec3(*(scale * p)) for p in points]


def create_trefoil_control_points(scale: float, center: wp.vec3, num_points: int = 24) -> list[wp.vec3]:
    """Control points sampled on a trefoil knot (a closed curve)."""
    ts = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    return [
        center
        + wp.vec3(
            scale * float(np.sin(t) + 2.0 * np.sin(2.0 * t)),
            scale * float(np.cos(t) - 2.0 * np.cos(2.0 * t)),
            scale * float(-np.sin(3.0 * t)),
        )
        for t in ts
    ]


def create_figure_eight_knot_control_points(scale: float, center: wp.vec3, num_points: int = 32) -> list[wp.vec3]:
    """Control points sampled on a figure-eight knot (the closed 4_1 knot)."""
    ts = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    return [
        center
        + wp.vec3(
            scale * float((2.0 + np.cos(2.0 * t)) * np.cos(3.0 * t)),
            scale * float((2.0 + np.cos(2.0 * t)) * np.sin(3.0 * t)),
            scale * float(np.sin(4.0 * t)),
        )
        for t in ts
    ]


def create_torus_knot_control_points(
    p: int, q: int, scale: float, center: wp.vec3, radius_major: float = 3.0, num_points: int = 40
) -> list[wp.vec3]:
    """Control points sampled on a (p, q) torus knot, e.g. (2, 5) for the cinquefoil."""
    ts = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    return [
        center
        + wp.vec3(
            scale * float((radius_major + np.cos(q * t)) * np.cos(p * t)),
            scale * float((radius_major + np.cos(q * t)) * np.sin(p * t)),
            scale * float(np.sin(q * t)),
        )
        for t in ts
    ]


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        # Simulation cadence
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.cable_radius = 0.015

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.0

        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            mu=1.0e9,
            ke=builder.default_shape_cfg.ke,
            kd=builder.default_shape_cfg.kd,
        )
        builder.add_ground_plane(cfg=ground_cfg)

        # Rope material: per-joint stiffness derived from Young's modulus and geometry.
        youngs_modulus = 5.0e6
        segment_length = 0.04
        stretch_stiffness, bend_stiffness = newton.utils.rod_stiffness_from_elastic_moduli(
            youngs_modulus, self.cable_radius, segment_length
        )

        # Each knot starts a few centimeters above the ground and keeps enough clearance
        # between crossing strands for the chosen capsule radius.
        cable_specs: list[dict] = [
            {
                "label": "overhand_knot",
                "control_points": create_overhand_knot_control_points(scale=0.15, height=0.17),
                "num_segments": 64,
                "normal_hint": wp.vec3(0.0, 0.0, 1.0),
            },
            {
                "label": "double_overhand_knot",
                "control_points": create_double_overhand_knot_control_points(
                    scale=0.15, center=wp.vec3(0.3, -0.7, 0.2)
                ),
                "segment_length": 0.045,
                "normal_hint": wp.vec3(0.0, 0.0, 1.0),
            },
            {
                "label": "trefoil_loop",
                "control_points": create_trefoil_control_points(scale=0.1, center=wp.vec3(0.0, 0.9, 0.155)),
                "segment_length": segment_length,
                "closed": True,
                "twist_total": 2.0 * math.pi,
            },
            {
                "label": "figure_eight_loop",
                "control_points": create_figure_eight_knot_control_points(scale=0.07, center=wp.vec3(0.0, 1.9, 0.125)),
                "segment_length": segment_length,
                "closed": True,
            },
            {
                "label": "cinquefoil_loop",
                "control_points": create_torus_knot_control_points(2, 5, scale=0.06, center=wp.vec3(0.0, 2.9, 0.115)),
                "segment_length": segment_length,
                "closed": True,
            },
        ]

        self.cables: list[tuple[str, list[int]]] = []
        for spec in cable_specs:
            label = spec.pop("label")
            bodies, _ = builder.add_cable_spline(
                spec.pop("control_points"),
                radius=self.cable_radius,
                stretch_stiffness=stretch_stiffness,
                stretch_damping=1.0e0,
                bend_stiffness=bend_stiffness,
                bend_damping=1.0e0,
                label=label,
                body_frame_origin="com",
                **spec,
            )
            self.cables.append((label, bodies))

        builder.color()

        self.model = builder.finalize()
        # Size persistent contact history before graph capture.
        self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching="sticky")
        self.contacts = self.collision_pipeline.contacts()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_compliant_alm=True,
            rigid_body_contact_buffer_size=256,
            rigid_contact_history=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        """Capture simulation loop into a graph for optimal performance."""
        with wp.ScopedCapture() as cap:
            self.simulate()
        self.graph = cap.graph

    def simulate(self):
        """Execute all simulation substeps for one frame."""
        for _substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state to the viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify all cables settled without artifacts and kept their knotted shapes."""
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        assert np.isfinite(body_q).all(), "Non-finite positions"
        assert np.isfinite(body_qd).all(), "Non-finite velocities"
        assert (np.abs(body_qd) < 5.0e2).all(), "Velocities too large"

        ground_tolerance = 0.05
        min_z = body_q[:, 2].min()
        assert min_z > -ground_tolerance, f"Cable penetrated ground: min_z={min_z:.4f}"

        # Knots must not unravel: consecutive segment midpoints stay a bounded distance apart
        # (stretch would show up as gaps between capsule origins).
        for label, bodies in self.cables:
            positions = body_q[bodies, :3]
            gaps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            assert gaps.max() < 6.0 * self.cable_radius, f"{label}: cable segments separated: {gaps.max():.4f} m"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
