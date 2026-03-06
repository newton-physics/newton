# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Basic Conveyor
#
# Baggage-claim style conveyor built from one rotating belt mesh attached
# to a kinematic root link with a prescribed revolute joint motion. Two
# static annular boundary meshes keep dynamic "bags" on the belt.
#
# Command: uv run -m newton.examples basic_conveyor
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples

BELT_CENTER_Z = 0.55
BELT_RING_RADIUS = 1.8
BELT_HALF_WIDTH = 0.24
BELT_HALF_THICKNESS = 0.04
BELT_MESH_SEGMENTS = 96
RAIL_WALL_THICKNESS = 0.035
RAIL_HEIGHT = 0.16
RAIL_BASE_OVERLAP = 0.01
BAG_COUNT = 18
BAG_LANE_OFFSETS = (-0.12, 0.0, 0.12)
BAG_DROP_CLEARANCE = 0.035
BELT_SPEED = 0.75  # tangential belt speed [m/s]
BELT_COLLISION_GROUP = 7
RAIL_COLLISION_GROUP = 3
BAG_COLLISION_GROUP_BASE = 100


def create_annular_prism_mesh(
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    segments: int,
    *,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float,
) -> newton.Mesh:
    """Create a closed ring prism mesh centered at the origin."""
    if segments < 3:
        raise ValueError("segments must be >= 3")
    if inner_radius <= 0.0 or outer_radius <= inner_radius:
        raise ValueError("Expected 0 < inner_radius < outer_radius")
    if z_max <= z_min:
        raise ValueError("Expected z_max > z_min")

    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False, dtype=np.float32)
    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)

    inner_top = np.stack(
        (
            inner_radius * cos_theta,
            inner_radius * sin_theta,
            np.full(segments, z_max, dtype=np.float32),
        ),
        axis=1,
    )
    outer_top = np.stack(
        (
            outer_radius * cos_theta,
            outer_radius * sin_theta,
            np.full(segments, z_max, dtype=np.float32),
        ),
        axis=1,
    )
    inner_bottom = np.stack(
        (
            inner_radius * cos_theta,
            inner_radius * sin_theta,
            np.full(segments, z_min, dtype=np.float32),
        ),
        axis=1,
    )
    outer_bottom = np.stack(
        (
            outer_radius * cos_theta,
            outer_radius * sin_theta,
            np.full(segments, z_min, dtype=np.float32),
        ),
        axis=1,
    )

    vertices = np.vstack((inner_top, outer_top, inner_bottom, outer_bottom)).astype(np.float32)

    it_offset = 0
    outer_top_offset = segments
    ib_offset = 2 * segments
    ob_offset = 3 * segments

    indices: list[int] = []
    for i in range(segments):
        j = (i + 1) % segments

        it_i = it_offset + i
        it_j = it_offset + j
        outer_top_i = outer_top_offset + i
        outer_top_j = outer_top_offset + j
        ib_i = ib_offset + i
        ib_j = ib_offset + j
        ob_i = ob_offset + i
        ob_j = ob_offset + j

        # Top face (+Z)
        indices.extend((it_i, outer_top_i, outer_top_j, it_i, outer_top_j, it_j))
        # Bottom face (-Z)
        indices.extend((ib_i, ib_j, ob_j, ib_i, ob_j, ob_i))
        # Outer face (+radial)
        indices.extend((ob_i, ob_j, outer_top_j, ob_i, outer_top_j, outer_top_i))
        # Inner face (-radial)
        indices.extend((ib_i, it_i, it_j, ib_i, it_j, ib_j))

    mesh = newton.Mesh(vertices=vertices, indices=np.asarray(indices, dtype=np.int32), compute_inertia=False)
    mesh.color = color
    mesh.roughness = roughness
    mesh.metallic = metallic
    return mesh


@wp.kernel
def set_conveyor_belt_state(
    belt_joint_q_start: int,
    belt_joint_qd_start: int,
    sim_time: wp.array(dtype=wp.float32),
    belt_angular_speed: float,
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    """Set prescribed state for the belt's revolute root joint."""
    angle = belt_angular_speed * sim_time[0]
    joint_q[belt_joint_q_start] = angle
    joint_qd[belt_joint_qd_start] = belt_angular_speed


@wp.kernel
def advance_time(sim_time: wp.array(dtype=wp.float32), dt: float):
    sim_time[0] = sim_time[0] + dt


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        belt_speed = float(args.belt_speed) if args is not None and hasattr(args, "belt_speed") else BELT_SPEED
        self.belt_angular_speed = belt_speed / BELT_RING_RADIUS

        builder = newton.ModelBuilder()

        ground_shape = builder.add_ground_plane()

        # Visual-only center island for a baggage-claim look.
        island_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=False, has_particle_collision=False, density=0.0
        )
        builder.add_shape_cylinder(
            body=-1,
            radius=0.9,
            half_height=0.08,
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, BELT_CENTER_Z - BELT_HALF_THICKNESS - 0.08),
                q=wp.quat_identity(),
            ),
            cfg=island_cfg,
        )

        belt_cfg = newton.ModelBuilder.ShapeConfig(
            mu=1.2,
            ke=3.0e3,
            kd=150.0,
            collision_group=BELT_COLLISION_GROUP,
        )
        rail_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.8,
            ke=3.0e3,
            kd=150.0,
            collision_group=RAIL_COLLISION_GROUP,
        )
        bag_cfg = newton.ModelBuilder.ShapeConfig(mu=1.0, ke=2.0e3, kd=180.0, restitution=0.0)

        belt_inner_radius = BELT_RING_RADIUS - BELT_HALF_WIDTH
        belt_outer_radius = BELT_RING_RADIUS + BELT_HALF_WIDTH

        belt_mesh = create_annular_prism_mesh(
            inner_radius=belt_inner_radius,
            outer_radius=belt_outer_radius,
            z_min=-BELT_HALF_THICKNESS,
            z_max=BELT_HALF_THICKNESS,
            segments=BELT_MESH_SEGMENTS,
            color=(0.09, 0.09, 0.09),  # dark gray rubber
            roughness=0.94,
            metallic=0.02,
        )
        rail_inner_mesh = create_annular_prism_mesh(
            inner_radius=belt_inner_radius - RAIL_WALL_THICKNESS,
            outer_radius=belt_inner_radius,
            z_min=BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP,
            z_max=BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP + RAIL_HEIGHT,
            segments=BELT_MESH_SEGMENTS,
            color=(0.66, 0.69, 0.74),  # brushed metal
            roughness=0.24,
            metallic=0.9,
        )
        rail_outer_mesh = create_annular_prism_mesh(
            inner_radius=belt_outer_radius,
            outer_radius=belt_outer_radius + RAIL_WALL_THICKNESS,
            z_min=BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP,
            z_max=BELT_HALF_THICKNESS - RAIL_BASE_OVERLAP + RAIL_HEIGHT,
            segments=BELT_MESH_SEGMENTS,
            color=(0.66, 0.69, 0.74),  # brushed metal
            roughness=0.24,
            metallic=0.9,
        )

        self.belt_body = builder.add_link(
            mass=15.0,
            is_kinematic=True,
            label="conveyor_belt",
        )
        self.belt_shape = builder.add_shape_mesh(
            self.belt_body,
            mesh=belt_mesh,
            cfg=belt_cfg,
            label="conveyor_belt_mesh",
        )
        self.belt_joint = builder.add_joint_revolute(
            parent=-1,
            child=self.belt_body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, BELT_CENTER_Z), q=wp.quat_identity()),
            label="conveyor_belt_joint",
        )
        qd_start = builder.joint_qd_start[self.belt_joint]
        builder.joint_qd[qd_start] = self.belt_angular_speed
        builder.add_articulation([self.belt_joint], label="conveyor_belt")

        for rail_mesh, rail_label in (
            (rail_inner_mesh, "conveyor_rail_inner"),
            (rail_outer_mesh, "conveyor_rail_outer"),
        ):
            rail_shape = builder.add_shape_mesh(
                body=-1,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, BELT_CENTER_Z), q=wp.quat_identity()),
                mesh=rail_mesh,
                cfg=rail_cfg,
                label=rail_label,
            )
            builder.add_shape_collision_filter_pair(self.belt_shape, rail_shape)
        # Belt should only interact with dynamic rigid bags.
        builder.add_shape_collision_filter_pair(self.belt_shape, ground_shape)

        self.bag_bodies = []
        belt_top_z = BELT_CENTER_Z + BELT_HALF_THICKNESS
        bag_angles = np.linspace(0.0, 2.0 * math.pi, BAG_COUNT, endpoint=False, dtype=np.float32)

        for i, angle in enumerate(bag_angles):
            lane_idx = i % len(BAG_LANE_OFFSETS)
            radial_offset = BAG_LANE_OFFSETS[lane_idx]
            radius = BELT_RING_RADIUS + radial_offset
            bag_x = radius * math.cos(angle)
            bag_y = radius * math.sin(angle)
            bag_yaw = angle + 0.5 * math.pi

            shape_type = i % 3
            if shape_type == 0:
                bag_vertical_extent = 0.08  # box hz
            elif shape_type == 1:
                bag_vertical_extent = 0.08  # horizontal capsule radius
            else:
                bag_vertical_extent = 0.11  # sphere radius

            # Spawn just above the belt to avoid initial interpenetration and large bounces.
            bag_z = belt_top_z + bag_vertical_extent + BAG_DROP_CLEARANCE
            bag_body = builder.add_link(
                xform=wp.transform(
                    p=wp.vec3(bag_x, bag_y, bag_z),
                    q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), bag_yaw),
                ),
                mass=2.8 + 0.1 * i,
                label=f"bag_{i}",
            )
            # Important: negative groups collide with everything except the exact same
            # negative group. Use distinct groups per bag so bag-bag collisions are enabled.
            bag_shape_cfg = bag_cfg.copy()
            bag_shape_cfg.collision_group = -(BAG_COLLISION_GROUP_BASE + i)

            if shape_type == 0:
                builder.add_shape_box(bag_body, hx=0.18, hy=0.12, hz=0.08, cfg=bag_shape_cfg)
            elif shape_type == 1:
                builder.add_shape_capsule(
                    bag_body,
                    radius=0.08,
                    half_height=0.15,
                    xform=wp.transform(q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)),
                    cfg=bag_shape_cfg,
                )
            else:
                builder.add_shape_sphere(bag_body, radius=0.11, cfg=bag_shape_cfg)

            builder.add_articulation([builder.add_joint_free(bag_body)], label=f"bag_{i}")
            self.bag_bodies.append(bag_body)

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Ensure body state is initialized from model joint buffers.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        q_starts = self.model.joint_q_start.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        self.belt_joint_q_start = int(q_starts[self.belt_joint])
        self.belt_joint_qd_start = int(qd_starts[self.belt_joint])
        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.7, -1.3, 5.0), -60.0, -200.0)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                set_conveyor_belt_state,
                dim=1,
                inputs=[
                    self.belt_joint_q_start,
                    self.belt_joint_qd_start,
                    self.sim_time_wp,
                    self.belt_angular_speed,
                ],
                outputs=[self.state_0.joint_q, self.state_0.joint_qd],
                device=self.model.device,
            )

            # Only update maximal coordinates of the kinematic bodies (the conveyor belt)
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            wp.launch(advance_time, dim=1, inputs=[self.sim_time_wp, self.sim_dt], device=self.model.device)

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

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        belt_z = float(body_q[self.belt_body][2])
        assert abs(belt_z - BELT_CENTER_Z) < 0.15, f"Belt body drifted off the conveyor plane: z={belt_z:.4f}"

        for body_idx in self.bag_bodies:
            x = float(body_q[body_idx][0])
            y = float(body_q[body_idx][1])
            z = float(body_q[body_idx][2])
            assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z), f"Bag {body_idx} has non-finite pose values."
            assert z > -0.5, f"Bag body {body_idx} fell through the floor: z={z:.4f}"
            assert abs(x) < 4.0 and abs(y) < 4.0, f"Bag body {body_idx} left the scene bounds: ({x:.3f}, {y:.3f})"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--belt-speed",
        type=float,
        default=BELT_SPEED,
        help="Conveyor tangential speed [m/s].",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
