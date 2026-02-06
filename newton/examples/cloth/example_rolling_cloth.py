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
# Treadmill Cloth Simulation
#
# A rolled cloth mesh that unrolls as the inner seam rotates.
# Command: python -m newton.examples rolling_cloth
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags


@wp.kernel
def rotate_cylinder(
    angular_speed: float,
    dt: float,
    t: float,
    center_x: float,
    center_z: float,
    q0: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int64),
    q1: wp.array(dtype=wp.vec3),
):
    """Rotate cylinder vertices around their center axis."""
    i = wp.tid()
    particle_index = indices[i]
    c0 = wp.cos(-angular_speed * (t - dt))
    s0 = wp.sin(-angular_speed * (t - dt))
    c1 = wp.cos(angular_speed * t)
    s1 = wp.sin(angular_speed * t)

    # Translate to center, rotate, translate back
    x0 = q0[particle_index][0] - center_x
    y0 = q0[particle_index][1]
    z0 = q0[particle_index][2] - center_z

    # Undo previous rotation
    rx = c0 * x0 + s0 * z0
    rz = -s0 * x0 + c0 * z0

    # Apply new rotation
    x1 = c1 * rx + s1 * rz
    z1 = -s1 * rx + c1 * rz

    # Translate back
    q0[particle_index][0] = x1 + center_x
    q0[particle_index][1] = y0
    q0[particle_index][2] = z1 + center_z
    q1[particle_index] = q0[particle_index]


def rolled_cloth_mesh(
    length=500.0,
    width=100.0,
    nu=200,
    nv=15,
    inner_radius=10.0,
    thickness=0.4,
    target_x=None,
    target_y=None,
    extension_segments=10,
):
    """
    Create a rolled cloth mesh with optional extension to a target point.

    Args:
        target_x, target_y: Target position in local coords (before rotation).
                           If provided, extension goes directly to this point.
        extension_segments: Number of rows for extension
    """
    verts = []
    faces = []

    # Create the spiral part
    for i in range(nu):
        u = length * i / (nu - 1)
        theta = u / inner_radius
        r = inner_radius + (thickness / (2.0 * np.pi)) * theta

        for j in range(nv):
            v = width * (j / (nv - 1) - 0.5)  # Center around z=0 for XZ plane symmetry
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = v
            verts.append([x, y, z])

    # Get outer edge position
    last_theta = length / inner_radius
    last_r = inner_radius + (thickness / (2.0 * np.pi)) * last_theta
    outer_x = last_r * np.cos(last_theta)
    outer_y = last_r * np.sin(last_theta)

    # Add extension rows if target is provided
    ext_rows = 0
    if target_x is not None and target_y is not None:
        # Direction from outer edge to target
        dx = target_x - outer_x
        dy = target_y - outer_y
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > 1.0:
            ext_rows = extension_segments
            for i in range(1, ext_rows + 1):
                t = i / ext_rows
                ext_x = outer_x + t * dx
                ext_y = outer_y + t * dy

                for j in range(nv):
                    v = width * (j / (nv - 1) - 0.5)  # Center around z=0 for XZ plane symmetry
                    verts.append([ext_x, ext_y, v])

    total_rows = nu + ext_rows

    def idx(i, j):
        return i * nv + j

    for i in range(total_rows - 1):
        for j in range(nv - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            faces.append([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])

    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32), nu, ext_rows


def cylinder_mesh(radius=9.5, height=120.0, segments=64):
    """Create a cylinder mesh (side walls only)."""
    verts = []
    faces = []

    for i in range(segments):
        t0 = 2 * math.pi * i / segments
        t1 = 2 * math.pi * (i + 1) / segments

        x0, z0 = radius * math.cos(t0), radius * math.sin(t0)
        x1, z1 = radius * math.cos(t1), radius * math.sin(t1)

        y0 = -height * 0.5
        y1 = height * 0.5

        base = len(verts)

        verts += [
            [x0, y0, z0],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z0],
        ]

        faces += [
            [base + 0, base + 1, base + 2],
            [base + 0, base + 2, base + 3],
        ]

    return (
        np.array(verts, np.float32),
        np.array(faces, np.int32),
    )


class Example:
    def __init__(
        self,
        viewer,
        args=None,
        cloth_length=800.0,
        cloth_nu=300,
        cloth_thickness=0.4,
        angular_speed=-2 * np.pi,
        spin_duration=20.0,
    ):
        self.viewer = viewer
        self.sim_time = 0.0
        self.args = args

        # Visualization scale: simulation is in cm, visualization in meters
        self.viz_scale = 0.01

        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 10

        # Cloth parameters
        self.cloth_thickness = cloth_thickness
        self.nv = 15  # vertices per row

        # Cylinder properties - cylinders are now further apart
        self.cyl1_radius = 9.9
        self.cyl2_radius = 14.9
        self.cyl1_center = (-27.2, 7.4)  # (X, Z)
        self.cyl2_center = (40.0, 0.0)  # (X, Z) - moved further right

        # Cloth position offset
        cloth_offset_x = self.cyl1_center[0]  # -27.2
        cloth_offset_z = self.cyl1_center[1]  # 7.4

        # Calculate target position for extension (cylinder 2's left side)
        # in LOCAL coordinates (before 90° rotation around X)
        # World target: (cyl2_x - radius - offset, cyl2_z)
        # Local coords: local_x = world_x - cloth_offset_x, local_y = world_z - cloth_offset_z
        self_contact_radius = 0.40
        attach_offset = self.cloth_thickness + self_contact_radius
        target_world_x = self.cyl2_center[0] - self.cyl2_radius - attach_offset
        target_world_z = self.cyl2_center[1]

        target_local_x = target_world_x - cloth_offset_x
        target_local_y = target_world_z - cloth_offset_z

        # Build model with zero gravity
        builder = newton.ModelBuilder(gravity=0.0)

        # Generate cloth mesh with extension going directly to target
        self.cloth_verts, self.cloth_faces, self.spiral_rows, self.ext_rows = rolled_cloth_mesh(
            length=cloth_length,
            nu=cloth_nu,
            thickness=self.cloth_thickness,
            target_x=target_local_x,
            target_y=target_local_y,
            extension_segments=20,
        )
        self.cloth_faces_flat = self.cloth_faces.reshape(-1)
        self.num_cloth_verts = len(self.cloth_verts)
        self.total_rows = self.spiral_rows + self.ext_rows

        # Generate cylinder meshes
        cylinder_segments = 128
        self.cyl1_verts, self.cyl1_faces = cylinder_mesh(radius=self.cyl1_radius, segments=cylinder_segments)
        self.cyl2_verts, self.cyl2_faces = cylinder_mesh(radius=self.cyl2_radius, segments=cylinder_segments)
        self.num_cyl1_verts = len(self.cyl1_verts)
        self.num_cyl2_verts = len(self.cyl2_verts)

        # Add cloth mesh
        builder.add_cloth_mesh(
            pos=wp.vec3(-27.2, 50.0, 7.4),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -np.pi / 2),
            scale=1.0,
            vertices=self.cloth_verts,
            indices=self.cloth_faces_flat,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
            particle_radius=0.5,
        )

        # Add first cylinder
        builder.add_cloth_mesh(
            pos=wp.vec3(self.cyl1_center[0], 50.0, self.cyl1_center[1]),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.0),
            scale=1.0,
            vertices=self.cyl1_verts,
            indices=self.cyl1_faces.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1e2,
            edge_kd=0.0,
        )

        # Add second cylinder
        builder.add_cloth_mesh(
            pos=wp.vec3(self.cyl2_center[0], 50.0, self.cyl2_center[1]),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.0),
            scale=1.0,
            vertices=self.cyl2_verts,
            indices=self.cyl2_faces.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=1.0e-5,
            edge_ke=1,
            edge_kd=0.01,
        )

        # Add ground plane
        builder.add_ground_plane(-1.0)

        # Color for VBD solver
        builder.color(include_bending=False)

        # Finalize model
        self.model = builder.finalize()
        self.model.soft_contact_ke = 2.0e5
        self.model.soft_contact_kd = 1.0e-5
        self.model.soft_contact_mu = 0.1

        # Fix outer edge of cloth to cylinder 2 and set up cylinder rotation
        # Outer edge = last row (end of extension), attached to cylinder 2's leftmost line
        last_row = self.total_rows - 1
        self.fixed_point_indices = [last_row * self.nv + i for i in range(self.nv)]

        # Position the outer edge at cylinder 2's leftmost line
        # This avoids penetration by placing cloth on the surface facing the spiral
        # Offset = cloth thickness + self_contact_radius to allow air gap
        positions = self.model.particle_q.numpy()
        attach_offset = self_contact_radius * 1.2
        left_x = self.cyl2_center[0] - self.cyl2_radius - attach_offset
        for idx in self.fixed_point_indices:
            positions[idx][0] = left_x
            positions[idx][2] = self.cyl2_center[1]  # Align Z with cylinder center
        self.model.particle_q = wp.array(positions, dtype=wp.vec3)

        # Fix the outer edge vertices (kinematic, attached to cylinder 2)
        if len(self.fixed_point_indices):
            flags = self.model.particle_flags.numpy()
            for fixed_vertex_id in self.fixed_point_indices:
                flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE
            self.model.particle_flags = wp.array(flags)

        self.fixed_point_indices = wp.array(self.fixed_point_indices)

        # Store cylinder vertex indices for rotation
        cyl1_start = self.num_cloth_verts
        cyl1_end = cyl1_start + self.num_cyl1_verts
        cyl2_start = cyl1_end
        cyl2_end = cyl2_start + self.num_cyl2_verts

        cyl1_idx_list = list(range(cyl1_start, cyl1_end))
        cyl2_idx_list = list(range(cyl2_start, cyl2_end))
        self.num_cyl1_indices = len(cyl1_idx_list)
        self.num_cyl2_indices = len(cyl2_idx_list)
        self.cyl1_indices = wp.array(cyl1_idx_list, dtype=wp.int64)
        self.cyl2_indices = wp.array(cyl2_idx_list, dtype=wp.int64)

        # Make all cylinder vertices static (kinematic, not simulated)
        flags = self.model.particle_flags.numpy()
        for id in range(self.num_cloth_verts, len(builder.particle_q)):
            flags[id] = flags[id] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)

        # Rotation parameters - match linear velocity at surface
        # v = omega * r, so for same v: omega2 = omega1 * r1 / r2
        self.angular_speed = angular_speed  # rad/sec
        linear_velocity = abs(self.angular_speed) * self.cyl1_radius
        self.angular_speed_cyl1 = linear_velocity / self.cyl1_radius  # = angular_speed
        self.angular_speed_cyl2 = linear_velocity / self.cyl2_radius  # slower due to larger radius
        self.spin_duration = spin_duration  # seconds

        # Create solver
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.40,
            particle_self_contact_margin=0.6,
            particle_vertex_contact_buffer_size=48,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=5,
            particle_topological_contact_filter_threshold=2,
        )

        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create a state for visualization (will be scaled at render time)
        self.viz_state = self.model.state()

        # Also update state_0 positions (will be scaled later in _scale_model_for_visualization)
        state_positions = self.state_0.particle_q.numpy()
        for idx in range(len(self.fixed_point_indices.numpy())):
            state_positions[self.fixed_point_indices.numpy()[idx]][0] = left_x
            state_positions[self.fixed_point_indices.numpy()[idx]][2] = self.cyl2_center[1]
        self.state_0.particle_q = wp.array(state_positions, dtype=wp.vec3)

        # Disable collision detection (matches original)
        self.contacts = None

        self.viewer.set_model(self.model)

        # Set up camera
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.set_camera(pos=wp.vec3(-0.02, -2.33, 0.69), pitch=-15.9, yaw=-264.8)
            self.viewer.camera.fov = 67.0

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.solver.rebuild_bvh(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply rotation during spin duration
            if self.sim_time < self.spin_duration:
                # Rotate cloth outer edge (attached to cylinder 2's left side)
                wp.launch(
                    kernel=rotate_cylinder,
                    dim=len(self.fixed_point_indices),
                    inputs=[
                        self.angular_speed_cyl2,  # Same speed as cylinder 2
                        self.sim_dt,
                        self.sim_time,
                        self.cyl2_center[0],  # Rotate around cylinder 2's center
                        self.cyl2_center[1],
                        self.state_0.particle_q,
                        self.fixed_point_indices,
                        self.state_1.particle_q,
                    ],
                )

                # Rotate cylinder 1 (around its center, faster due to smaller radius)
                wp.launch(
                    kernel=rotate_cylinder,
                    dim=self.num_cyl1_indices,
                    inputs=[
                        self.angular_speed_cyl1,
                        self.sim_dt,
                        self.sim_time,
                        self.cyl1_center[0],
                        self.cyl1_center[1],
                        self.state_0.particle_q,
                        self.cyl1_indices,
                        self.state_1.particle_q,
                    ],
                )

                # Rotate cylinder 2 (around its center, slower due to larger radius)
                wp.launch(
                    kernel=rotate_cylinder,
                    dim=self.num_cyl2_indices,
                    inputs=[
                        self.angular_speed_cyl2,
                        self.sim_dt,
                        self.sim_time,
                        self.cyl2_center[0],
                        self.cyl2_center[1],
                        self.state_0.particle_q,
                        self.cyl2_indices,
                        self.state_1.particle_q,
                    ],
                )

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        # Scale positions from cm to meters for visualization and flip Z axis
        positions = self.state_0.particle_q.numpy()
        scaled_positions = positions * self.viz_scale
        self.viz_state.particle_q = wp.array(scaled_positions, dtype=wp.vec3)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.end_frame()

    def test_final(self):
        """Test that cloth has unrolled from cylinder 1 to cylinder 2."""
        # Get cloth particle positions (exclude cylinder particles)
        particle_q = self.state_0.particle_q.numpy()
        cloth_q = particle_q[: self.num_cloth_verts]

        # Calculate center of mass
        com = np.mean(cloth_q, axis=0)

        # Initial COM is at X ≈ -25.72 (near cylinder 1 at X=-27.2)
        # Cylinder 2 is at X=40.0
        # After ~12.7 revolutions (800 frames), expect COM to shift significantly
        initial_com_x = -25.72
        expected_shift = 40.0 - initial_com_x  # ≈ 65.72
        min_shift = expected_shift * 0.6  # ≈ 39.4 (60% progress is reasonable)

        actual_shift = com[0] - initial_com_x

        assert actual_shift > min_shift, (
            f"Cloth not unrolled enough: shift={actual_shift:.1f} < {min_shift:.1f}, COM X={com[0]:.1f}"
        )

        # Ensure bbox hasn't exploded
        bbox_size = np.linalg.norm(np.max(cloth_q, axis=0) - np.min(cloth_q, axis=0))
        assert bbox_size < 150.0, f"Bbox exploded: size={bbox_size:.2f}"


if __name__ == "__main__":
    # Create parser with base arguments
    parser = newton.examples.create_parser()

    parser.add_argument("--cloth-length", type=float, default=800.0, help="Length of cloth spiral")
    parser.add_argument("--cloth-nu", type=int, default=300, help="Number of rows along cloth length")
    parser.add_argument("--cloth-thickness", type=float, default=0.4, help="Thickness of rolled cloth mesh")
    parser.add_argument("--angular-speed", type=float, default=-2 * np.pi, help="Base rotation speed (rad/sec)")
    parser.add_argument("--spin-duration", type=float, default=20.0, help="Duration of spinning (seconds)")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(
        viewer=viewer,
        args=args,
        cloth_length=args.cloth_length,
        cloth_nu=args.cloth_nu,
        cloth_thickness=args.cloth_thickness,
        angular_speed=args.angular_speed,
        spin_duration=args.spin_duration,
    )

    newton.examples.run(example, args)
