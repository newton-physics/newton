# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Pile
#
# Demonstrates complex cable-to-cable contact and settling behavior.
# Creates a pile of cables with alternating
# orientations (X/Y axis) and sinusoidal waviness. Tests multi-body contact
# resolution, stacking stability, and friction in dense cable assemblies.
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(
        self,
        viewer,
        args=None,
        slope_enabled: bool = False,
        slope_angle_deg: float = 20.0,
        slope_mu: float | None = None,
        layers: int = 10,
        lanes_per_layer: int = 10,
    ):
        self.viewer = viewer
        self.args = args

        # Simulation cadence
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable pile parameters
        self.num_elements = 40
        segment_length = 0.05
        self.cable_length = self.num_elements * segment_length
        cable_radius = 0.012
        bend_stiffness = 2.0e1

        # Layers and lanes
        self.layers = layers
        self.lanes_per_layer = lanes_per_layer
        lane_spacing = max(8.0 * cable_radius, 0.15)
        layer_gap = cable_radius * 6.0

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.0

        rod_bodies_all: list[int] = []

        # Material properties
        builder.default_shape_cfg.mu = 1.0e0
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0

        cable_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=builder.default_shape_cfg.density,
            ke=builder.default_shape_cfg.ke,
            kd=builder.default_shape_cfg.kd,
            kf=builder.default_shape_cfg.kf,
            ka=builder.default_shape_cfg.ka,
            mu=builder.default_shape_cfg.mu,
            restitution=builder.default_shape_cfg.restitution,
        )

        # Ground plane (optionally sloped for friction tests)
        if slope_enabled:
            angle = math.radians(slope_angle_deg)
            rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)

            slope_cfg = builder.default_shape_cfg
            if slope_mu is not None:
                slope_cfg = newton.ModelBuilder.ShapeConfig(
                    density=builder.default_shape_cfg.density,
                    ke=builder.default_shape_cfg.ke,
                    kd=builder.default_shape_cfg.kd,
                    kf=builder.default_shape_cfg.kf,
                    ka=builder.default_shape_cfg.ka,
                    mu=slope_mu,
                    restitution=builder.default_shape_cfg.restitution,
                )

            builder.add_shape_plane(
                width=10.0,
                length=10.0,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), rot),
                body=-1,
                cfg=slope_cfg,
            )
        else:
            ground_cfg = newton.ModelBuilder.ShapeConfig(
                mu=1.0e9,
                ke=builder.default_shape_cfg.ke,
                kd=builder.default_shape_cfg.kd,
            )
            builder.add_ground_plane(cfg=ground_cfg)

        # Build layered lanes of cables with alternating orientations
        for layer in range(self.layers):
            orient = "x" if (layer % 2 == 0) else "y"
            z0 = 0.3 + layer * layer_gap
            for lane in range(self.lanes_per_layer):
                offset = (lane - (self.lanes_per_layer - 1) * 0.5) * lane_spacing
                if orient == "x":
                    start = wp.vec3(0.0, offset, z0)
                else:
                    start = wp.vec3(offset, 0.0, z0)

                wav = 0.5
                twist = 0.0

                dir_vec = wp.vec3(1.0, 0.0, 0.0) if orient == "x" else wp.vec3(0.0, 1.0, 0.0)
                ortho_vec = wp.vec3(0.0, 1.0, 0.0) if orient == "x" else wp.vec3(1.0, 0.0, 0.0)

                cable_length = float(self.cable_length)
                start0 = start - 0.5 * cable_length * dir_vec
                pts = newton.utils.create_straight_cable_points(
                    start=start0,
                    direction=dir_vec,
                    length=cable_length,
                    num_segments=int(self.num_elements),
                )

                # Sinusoidal waviness along orthogonal axis
                cycles = 2.0
                waviness_scale = 0.05
                if wav > 0.0:
                    for i in range(len(pts)):
                        t = i / self.num_elements
                        phase = 2.0 * math.pi * cycles * t
                        amp = wav * cable_length * waviness_scale
                        pts[i] = pts[i] + ortho_vec * (amp * math.sin(phase))

                edge_q = newton.utils.create_parallel_transport_cable_quaternions(pts, twist_total=float(twist))

                rod_bodies, _rod_joints = builder.add_rod(
                    positions=pts,
                    quaternions=edge_q,
                    radius=cable_radius,
                    cfg=cable_shape_cfg,
                    bend_stiffness=bend_stiffness,
                    bend_damping=1.0e0,
                    label=f"cable_l{layer}_{lane}",
                )
                rod_bodies_all.extend(rod_bodies)

        builder.color()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_body_contact_buffer_size=256,
            rigid_contact_history=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        pipeline = newton.CollisionPipeline(self.model, contact_matching="latest")
        self.contacts = self.model.contacts(collision_pipeline=pipeline)

        self.viewer.set_model(self.model)

        if hasattr(self.viewer, "picking"):
            ps = self.viewer.picking.pick_state.numpy()
            ps[0]["pick_stiffness"] = 20.0
            ps[0]["pick_damping"] = 0.0
            self.viewer.picking.pick_state.assign(ps)

        self.capture()

    def capture(self):
        """Capture simulation loop into a CUDA graph for optimal GPU performance."""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
        else:
            self.graph = None

    def simulate(self):
        """Execute all simulation substeps for one frame."""
        for _substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)

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
        """Test cable pile simulation for stability and correctness (called after simulation)."""
        cable_radius = 0.012
        cable_diameter = 2.0 * cable_radius

        tolerance = 0.5

        max_z_settled = self.layers * cable_diameter + tolerance
        ground_tolerance = tolerance

        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            body_positions = self.state_0.body_q.numpy()
            body_velocities = self.state_0.body_qd.numpy()

            assert np.isfinite(body_positions).all(), "Non-finite positions"
            assert np.isfinite(body_velocities).all(), "Non-finite velocities"

            z_positions = body_positions[:, 2]
            min_z = np.min(z_positions)
            max_z_actual = np.max(z_positions)

            assert min_z > -ground_tolerance, (
                f"Cables penetrated ground too much: min_z={min_z:.3f} < {-ground_tolerance:.3f}"
            )
            assert max_z_actual < max_z_settled, (
                f"Pile too high: max_z={max_z_actual:.3f} > expected {max_z_settled:.3f} "
                f"({self.layers} layers x {cable_diameter:.3f}m diameter + tolerance)"
            )

            assert (np.abs(body_velocities) < 5e2).all(), "Velocities too large"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
