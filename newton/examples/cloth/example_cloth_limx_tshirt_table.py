# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Throw a self-colliding LIMX T-shirt onto a static table."""

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


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt
        self.sim_time = 0.0
        self.table_top = 0.65

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
        rotation = _rotation_matrix_xyz(math.radians(12.0), math.radians(-8.0), math.radians(5.0))
        shirt_center = np.array([0.0, 0.0, 1.05], dtype=np.float32)
        rest_positions = local_positions @ rotation.T + shirt_center

        triangle_vertices = rest_positions[triangles]
        triangle_areas = 0.5 * np.linalg.norm(
            np.cross(
                triangle_vertices[:, 1] - triangle_vertices[:, 0],
                triangle_vertices[:, 2] - triangle_vertices[:, 0],
            ),
            axis=1,
        )
        masses = np.zeros(len(rest_positions), dtype=np.float32)
        for corner in range(3):
            np.add.at(masses, triangles[:, corner], 0.3 * triangle_areas / 3.0)
        if not np.isfinite(masses).all() or np.any(masses <= 0.0):
            raise ValueError("T-shirt mesh must give every particle a finite positive area-weighted mass")

        linear_velocity = np.array([0.25, 0.08, -0.60], dtype=np.float32)
        angular_velocity = np.array([0.35, -0.20, 0.65], dtype=np.float32)
        velocities = linear_velocity + np.cross(
            np.broadcast_to(angular_velocity, rest_positions.shape),
            rest_positions - shirt_center,
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particles(
            pos=[wp.vec3(*position) for position in rest_positions],
            vel=[wp.vec3(*velocity) for velocity in velocities],
            mass=masses.tolist(),
            radius=[0.006] * len(rest_positions),
        )
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.60), wp.quat_identity()),
            hx=0.55,
            hy=0.45,
            hz=0.05,
            color=wp.vec3(0.38, 0.24, 0.12),
        )
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        particle_count = self.model.particle_count
        edge_rows = newton.utils.MeshAdjacency(triangles).edge_indices
        interior_edge_rows = edge_rows[edge_rows[:, 1] >= 0]
        dihedral_indices = interior_edge_rows[:, [2, 3, 0, 1]]
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
                stiffness=1.0e-4,
                particle_count=particle_count,
                device=self.model.device,
            ),
        ]
        self.self_collision = newton.solvers.ConstraintSelfCollision(
            self.model,
            thickness=0.006,
            stiffness=None,
            max_contacts=131072,
            stiffness_factors=(0.5, 0.1, 1.5),
        )
        self.table_contact = newton.solvers.ConstraintStaticPlaneContact(
            normal=(0.0, 0.0, 1.0),
            offset=self.table_top,
            thickness=0.006,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.4,
            friction_epsilon=1.0e-4,
            particle_count=particle_count,
            device=self.model.device,
        )
        dynamic_constraints = newton.solvers.ConstraintGroupDynamic([self.self_collision, self.table_contact])
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
        self.initial_positions = rest_positions.copy()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.65, -1.75, 1.35), 8.0, 132.0)
        self.capture()

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
            raise AssertionError("LIMX T-shirt state contains non-finite values")
        if float(positions[:, 2].min()) < self.table_top - 0.03:
            raise AssertionError("LIMX T-shirt penetrated catastrophically below the table")

    def test_final(self):
        self.test_post_step()
        if self.sim_time <= 0.0:
            raise AssertionError("LIMX T-shirt scene did not advance")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
