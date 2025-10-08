# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import warp as wp

import newton
import newton.examples


def create_cable_geometry(
    start_pos: wp.vec3,
    num_elements: int,
    length: float,
    orientation: str = "x",
    waviness: float = 0.0,
    twist_total: float = 0.0,
) -> tuple[list[wp.vec3], np.ndarray, list[wp.quat]]:
    """
    Build a cable polyline with optional sinusoidal waviness and distributed twist.

    orientation: "x" lays the cable along +X, "y" along +Y.
    waviness: amplitude relative to segment length for a simple sine profile.
    twist_total: total radians of twist distributed uniformly along the cable.
    """
    num_points = num_elements + 1
    points: list[wp.vec3] = []

    dir_vec = wp.vec3(1.0, 0.0, 0.0) if orientation == "x" else wp.vec3(0.0, 1.0, 0.0)
    ortho_vec = wp.vec3(0.0, 1.0, 0.0) if orientation == "x" else wp.vec3(1.0, 0.0, 0.0)

    # Center the cable around start_pos to avoid initial overlaps with neighbors
    start_center = start_pos - 0.5 * length * dir_vec
    # Use a few sinusoidal cycles along the cable with a visible amplitude
    cycles = 2.0
    waviness_scale = 0.05  # amplitude fraction of length when waviness=1.0
    for i in range(num_points):
        t = i / num_elements
        base = start_center + dir_vec * (length * t)
        if waviness > 0.0:
            phase = 2.0 * math.pi * cycles * t
            amp = waviness * length * waviness_scale
            base = base + ortho_vec * (amp * math.sin(phase))
        points.append(base)

    # Build edges
    edge_indices: list[int] = []
    for i in range(num_elements):
        edge_indices.extend([i, i + 1])
    edge_indices = np.array(edge_indices, dtype=np.int32)

    # Parallel-transported quaternions; distribute twist_total uniformly
    quats: list[wp.quat] = []
    if num_elements > 0:
        local_axis = wp.vec3(0.0, 0.0, 1.0)  # internal capsule axis is +Z
        from_direction = local_axis
        twist_step = twist_total / num_elements if num_elements > 0 else 0.0
        for i in range(num_elements):
            p0 = points[i]
            p1 = points[i + 1]
            to_direction = wp.normalize(p1 - p0)
            dq = wp.quat_between_vectors(from_direction, to_direction)
            base_q = dq if i == 0 else wp.mul(dq, quats[i - 1])
            if twist_total != 0.0:
                twist_q = wp.quat_from_axis_angle(to_direction, twist_step)
                base_q = wp.mul(twist_q, base_q)
            quats.append(base_q)
            from_direction = to_direction

    return points, edge_indices, quats


class Example:
    def __init__(
        self,
        viewer,
        slope_enabled: bool = False,
        slope_angle_deg: float = 20.0,
        slope_mu: float | None = None,
    ):
        # Simulation cadence
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable pile parameters
        self.num_elements = 40
        self.cable_length = 2.0
        cable_radius = 0.012

        # Layers and lanes
        layers = 4
        lanes_per_layer = 10
        # Increase spacing to accommodate lateral waviness without initial intersections
        lane_spacing = max(8.0 * cable_radius, 0.15)
        layer_gap = cable_radius * 6.0

        builder = newton.ModelBuilder()

        rod_bodies_all: list[int] = []

        # Set default material properties before adding any shapes
        # Default config will be used by plane and any shape without explicit cfg
        builder.default_shape_cfg.ke = 1.0e4  # Contact stiffness (used by plane)
        builder.default_shape_cfg.kd = 1.0e-1  # Contact damping
        builder.default_shape_cfg.mu = 1.0e2  # Friction coefficient

        cable_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=builder.default_shape_cfg.density,
            ke=builder.default_shape_cfg.ke,
            kd=builder.default_shape_cfg.kd,
            kf=builder.default_shape_cfg.kf,
            ka=builder.default_shape_cfg.ka,
            mu=builder.default_shape_cfg.mu,
            restitution=builder.default_shape_cfg.restitution,
        )

        # Generate a ground plane (optionally sloped for friction tests)
        if slope_enabled:
            angle = math.radians(slope_angle_deg)
            rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)

            # Optionally override friction for the slope without changing defaults for other shapes
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
            builder.add_ground_plane()

        # Build layered lanes of cables with alternating orientations (no initial intersections)
        for layer in range(layers):
            orient = "x" if (layer % 2 == 0) else "y"
            z0 = 0.3 + layer * layer_gap
            # Generate parallel lanes along the orthogonal axis only
            for lane in range(lanes_per_layer):
                offset = (lane - (lanes_per_layer - 1) * 0.5) * lane_spacing
                if orient == "x":
                    start = wp.vec3(0.0, offset, z0)
                else:
                    start = wp.vec3(offset, 0.0, z0)

                # Regular waviness and no twist for repeatable layout across layers
                wav = 0.5
                twist = 0.0
                pts, edges, edge_q = create_cable_geometry(
                    start_pos=start,
                    num_elements=self.num_elements,
                    length=self.cable_length,
                    orientation=orient,
                    waviness=wav,
                    twist_total=twist,
                )

                rod_bodies, rod_joints = builder.add_rod_mesh(
                    positions=pts,
                    quaternions=edge_q,
                    radius=cable_radius,
                    cfg=cable_shape_cfg,
                    bend_stiffness=1.0e2,
                    bend_damping=1.0e-4,
                    stretch_stiffness=1.0e6,
                    stretch_damping=1.0e-4,
                    key=f"cable_l{layer}_{lane}",
                )
                rod_bodies_all.extend(rod_bodies)

        # Color bodies for VBD/AVBD coloring
        builder.color()

        # Finalize model
        self.model = builder.finalize()

        # Solver
        self.solver = newton.solvers.SolverAVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=0.1,
        )

        # States and viewer
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer = viewer
        self.viewer.set_model(self.model)

        # Optional capture for CUDA
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self):
        """Test cable pile simulation for stability and correctness."""
        test_steps = 20
        cable_radius = 0.012
        cable_diameter = 2.0 * cable_radius  # 0.024m
        layers = 4

        # Use same tolerance for ground penetration and pile height offset
        tolerance = 0.1  # Soft contacts allow some penetration and gaps

        # Calculate maximum expected height for SETTLED pile
        # After gravity settles the pile, cables should be stacked:
        # 4 layers = 4 cable diameters high (approximately)
        # Plus compression tolerance and contact gaps
        max_z_settled = layers * cable_diameter + tolerance
        ground_tolerance = tolerance

        for i in range(test_steps):
            self.simulate()

            # Test 1: Check for numerical stability
            if self.state_0.body_q is not None and self.state_0.body_qd is not None:
                body_positions = self.state_0.body_q.numpy()
                body_velocities = self.state_0.body_qd.numpy()

                assert np.isfinite(body_positions).all(), f"Non-finite positions at step {i}"
                assert np.isfinite(body_velocities).all(), f"Non-finite velocities at step {i}"

                # Test 2: Check physical bounds - cables should stay within pile height
                z_positions = body_positions[:, 2]  # Z positions (Newton uses Z-up)
                min_z = np.min(z_positions)
                max_z_actual = np.max(z_positions)

                assert min_z > -ground_tolerance, (
                    f"Cables penetrated ground too much at step {i}: min_z={min_z:.3f} < {-ground_tolerance:.3f}"
                )
                assert max_z_actual < max_z_settled, (
                    f"Pile too high at step {i}: max_z={max_z_actual:.3f} > expected {max_z_settled:.3f} "
                    f"(4 layers x {cable_diameter:.3f}m diameter + tolerance)"
                )

                # Test 3: Velocity should be reasonable (pile shouldn't explode)
                assert (np.abs(body_velocities) < 5e2).all(), f"Velocities too large at step {i}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()

    # Enable sloped plane by passing parameters here, e.g.:
    # example = Example(viewer, slope_enabled=True, slope_angle_deg=20.0, slope_mu=0.8)
    example = Example(viewer)
    newton.examples.run(example, args)
