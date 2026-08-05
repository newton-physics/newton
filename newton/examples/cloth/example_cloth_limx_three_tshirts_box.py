# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Throw three mutually colliding LIMX T-shirts into an open box."""

import math

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils


def _rotation_matrix_xyz(x_angle: float, y_angle: float, z_angle: float) -> np.ndarray:
    cosine_x = math.cos(x_angle)
    sine_x = math.sin(x_angle)
    cosine_y = math.cos(y_angle)
    sine_y = math.sin(y_angle)
    cosine_z = math.cos(z_angle)
    sine_z = math.sin(z_angle)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine_x, -sine_x], [0.0, sine_x, cosine_x]],
        dtype=np.float32,
    )
    rotation_y = np.array(
        [[cosine_y, 0.0, sine_y], [0.0, 1.0, 0.0], [-sine_y, 0.0, cosine_y]],
        dtype=np.float32,
    )
    rotation_z = np.array(
        [[cosine_z, -sine_z, 0.0], [sine_z, cosine_z, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return rotation_z @ rotation_y @ rotation_x


def _load_garment() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
    if stage is None:
        raise RuntimeError("Failed to open unisex_shirt.usd")
    shirt_prim = stage.GetPrimAtPath("/root/shirt")
    if not shirt_prim.IsValid():
        raise RuntimeError("unisex_shirt.usd does not contain /root/shirt")
    shirt_mesh = newton.usd.get_mesh(shirt_prim)

    source_positions = 0.01 * np.asarray(shirt_mesh.vertices, dtype=np.float32)
    triangles = np.asarray(shirt_mesh.indices, dtype=np.int32).reshape(-1, 3)
    local_center = 0.5 * (source_positions.min(axis=0) + source_positions.max(axis=0))
    local_positions = source_positions - local_center

    triangle_vertices = local_positions[triangles]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        ),
        axis=1,
    )
    masses = np.zeros(len(local_positions), dtype=np.float32)
    for corner in range(3):
        np.add.at(masses, triangles[:, corner], 0.3 * triangle_areas / 3.0)
    if not np.isfinite(masses).all() or np.any(masses <= 0.0):
        raise ValueError("T-shirt mesh must give every particle a finite positive area-weighted mass")
    return local_positions, triangles, masses


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt
        self.sim_time = 0.0

        self.box_min = np.array([-0.42, -0.34], dtype=np.float32)
        self.box_max = np.array([0.42, 0.34], dtype=np.float32)
        self.box_floor = 0.45
        self.box_wall_top = 1.15

        local_positions, local_triangles, local_masses = _load_garment()
        self.garment_vertex_count = len(local_positions)
        self.garment_triangle_count = len(local_triangles)

        configurations = (
            (
                np.array([-0.14, -0.08, 1.55], dtype=np.float32),
                (math.radians(11.0), math.radians(-9.0), math.radians(7.0)),
                np.array([0.18, 0.08, -0.70], dtype=np.float32),
                np.array([0.30, -0.20, 0.55], dtype=np.float32),
            ),
            (
                np.array([0.14, 0.08, 2.05], dtype=np.float32),
                (math.radians(-13.0), math.radians(8.0), math.radians(-12.0)),
                np.array([-0.12, 0.04, -0.90], dtype=np.float32),
                np.array([-0.25, 0.35, -0.45], dtype=np.float32),
            ),
            (
                np.array([-0.04, 0.10, 2.55], dtype=np.float32),
                (math.radians(8.0), math.radians(14.0), math.radians(16.0)),
                np.array([0.04, -0.10, -1.05], dtype=np.float32),
                np.array([0.20, 0.25, 0.60], dtype=np.float32),
            ),
        )
        self.garment_count = getattr(args, "garment_count", 3)
        if not 1 <= self.garment_count <= len(configurations):
            raise ValueError(f"garment_count must be between 1 and {len(configurations)}")

        positions_per_garment = []
        velocities_per_garment = []
        triangles_per_garment = []
        dihedrals_per_garment = []
        local_edge_rows = newton.utils.MeshAdjacency(local_triangles).edge_indices
        local_interior_edges = local_edge_rows[local_edge_rows[:, 1] >= 0]
        local_dihedrals = local_interior_edges[:, [2, 3, 0, 1]]
        for garment, (center, angles, linear_velocity, angular_velocity) in enumerate(
            configurations[: self.garment_count]
        ):
            rotation = _rotation_matrix_xyz(*angles)
            positions = local_positions @ rotation.T + center
            velocities = linear_velocity + np.cross(
                np.broadcast_to(angular_velocity, positions.shape),
                positions - center,
            )
            vertex_offset = garment * self.garment_vertex_count
            positions_per_garment.append(positions)
            velocities_per_garment.append(velocities)
            triangles_per_garment.append(local_triangles + vertex_offset)
            dihedrals_per_garment.append(local_dihedrals + vertex_offset)

        rest_positions = np.concatenate(positions_per_garment).astype(np.float32)
        velocities = np.concatenate(velocities_per_garment).astype(np.float32)
        masses = np.tile(local_masses, self.garment_count)
        triangles = np.concatenate(triangles_per_garment).astype(np.int32)
        dihedral_indices = np.concatenate(dihedrals_per_garment).astype(np.int32)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particles(
            pos=[wp.vec3(*position) for position in rest_positions],
            vel=[wp.vec3(*velocity) for velocity in velocities],
            mass=masses.tolist(),
            radius=[0.006] * len(rest_positions),
        )
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        self._add_box_shapes(builder)
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        particle_count = self.model.particle_count
        constraints = [
            newton.solvers.ConstraintTriangleElastic(
                triangle_indices=triangles,
                inverse_rest_matrices=self.model.tri_poses.numpy(),
                rest_areas=self.model.tri_areas.numpy(),
                stiffnesses=[wp.vec3(1.0e4, 1.0e4, 1.0e3)] * len(triangles),
                particle_count=particle_count,
                device=self.model.device,
            ),
            newton.solvers.ConstraintDihedralBending(
                dihedral_indices=dihedral_indices,
                rest_positions=rest_positions,
                stiffness=1.0e-5,
                particle_count=particle_count,
                device=self.model.device,
            ),
        ]
        self.self_collision = newton.solvers.ConstraintSelfCollision(
            self.model,
            thickness=0.003,
            stiffness=None,
            max_contacts=393216,
            stiffness_factors=(0.5, 0.3, 1.5),
            friction=0.4,
            friction_epsilon=1.0e-2,
        )
        plane_parameters = (
            ((0.0, 0.0, 1.0), self.box_floor),
            ((1.0, 0.0, 0.0), float(self.box_min[0])),
            ((-1.0, 0.0, 0.0), -float(self.box_max[0])),
            ((0.0, 1.0, 0.0), float(self.box_min[1])),
            ((0.0, -1.0, 0.0), -float(self.box_max[1])),
        )
        self.box_contacts = [
            newton.solvers.ConstraintStaticPlaneContact(
                normal=normal,
                offset=offset,
                thickness=0.006,
                stiffness=2.0e4,
                normal_damping=0.5,
                friction=0.4,
                friction_epsilon=1.0e-4,
                particle_count=particle_count,
                device=self.model.device,
            )
            for normal, offset in plane_parameters
        ]
        dynamic_constraints = newton.solvers.ConstraintGroupDynamic([self.self_collision, *self.box_contacts])
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            constraints,
            nonlinear_iterations=1,
            linear_iterations=50,
            velocity_damping=1.0,
            dynamic_operator=dynamic_constraints,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.0, -2.3, 1.9), -19.0, 131.0)
        self.capture()

    def _add_box_shapes(self, builder: newton.ModelBuilder) -> None:
        wall_center_z = 0.5 * (self.box_floor + self.box_wall_top)
        wall_half_height = 0.5 * (self.box_wall_top - self.box_floor)
        wall_thickness = 0.025
        box_center = 0.5 * (self.box_min + self.box_max)
        box_half_size = 0.5 * (self.box_max - self.box_min)
        outer_half_size = box_half_size + 2.0 * wall_thickness
        box_color = wp.vec3(0.34, 0.22, 0.12)
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(*box_center, self.box_floor - 0.05), wp.quat_identity()),
            hx=float(outer_half_size[0]),
            hy=float(outer_half_size[1]),
            hz=0.05,
            color=box_color,
        )
        for x_position in (self.box_min[0] - wall_thickness, self.box_max[0] + wall_thickness):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x_position, box_center[1], wall_center_z), wp.quat_identity()),
                hx=wall_thickness,
                hy=float(outer_half_size[1]),
                hz=wall_half_height,
                color=box_color,
            )
        for y_position in (self.box_min[1] - wall_thickness, self.box_max[1] + wall_thickness):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(box_center[0], y_position, wall_center_z), wp.quat_identity()),
                hx=float(outer_half_size[0]),
                hy=wall_thickness,
                hz=wall_half_height,
                color=box_color,
            )

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.state_0.assign(self.state_1)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("LIMX three-T-shirt state contains non-finite values")
        if float(positions[:, 2].min()) < self.box_floor - 0.04:
            raise AssertionError("A T-shirt penetrated catastrophically below the box floor")
        below_wall_top = positions[:, 2] <= self.box_wall_top
        contained_positions = positions[below_wall_top]
        if len(contained_positions) > 0 and (
            float(contained_positions[:, 0].min()) < float(self.box_min[0]) - 0.04
            or float(contained_positions[:, 0].max()) > float(self.box_max[0]) + 0.04
            or float(contained_positions[:, 1].min()) < float(self.box_min[1]) - 0.04
            or float(contained_positions[:, 1].max()) > float(self.box_max[1]) + 0.04
        ):
            raise AssertionError("A T-shirt escaped catastrophically through a box wall")

    def test_final(self):
        self.test_post_step()
        if self.sim_time <= 0.0:
            raise AssertionError("LIMX three-T-shirt box scene did not advance")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--garment-count",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Number of T-shirts to throw into the box.",
    )
    parser.set_defaults(num_frames=800)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
