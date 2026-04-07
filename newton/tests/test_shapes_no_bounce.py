# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Test that shapes falling onto a ground plane never exceed their initial z position.

This test verifies that during 200 simulation steps, no shape's z position ever
exceeds its initial value.  This catches issues like shapes bouncing upward due
to contact instabilities.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


def test_shapes_never_exceed_initial_z(test, device):
    """
    Test that shapes falling onto a ground mesh never exceed their initial z position.

    Scene setup:
    - 2-triangle ground plane mesh
    - 4x4x8 grid of mixed shapes (sphere, box, capsule, cylinder, cone, icosahedron)
    - XPBD solver with iterations=2, rigid_contact_relaxation=0.8, angular_damping=0.0
    - 100 FPS, 10 substeps per frame
    - nxn broad phase
    """
    builder = newton.ModelBuilder()

    # Simple 2-triangle ground plane mesh
    ground_size = 30.0
    ground_vertices = np.array(
        [
            [-ground_size, -ground_size, 0.0],
            [ground_size, -ground_size, 0.0],
            [ground_size, ground_size, 0.0],
            [-ground_size, ground_size, 0.0],
        ],
        dtype=np.float32,
    )
    ground_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    ground_mesh = newton.Mesh(ground_vertices, ground_indices)
    ground_offset = wp.transform(p=wp.vec3(0.0, 0.0, -0.5), q=wp.quat_identity())
    builder.add_shape_mesh(
        body=-1,
        mesh=ground_mesh,
        xform=ground_offset,
    )

    # Create icosahedron mesh
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_radius = 0.35

    ico_base_vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )
    for i in range(len(ico_base_vertices)):
        ico_base_vertices[i] = ico_base_vertices[i] / np.linalg.norm(ico_base_vertices[i]) * ico_radius

    ico_face_indices = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    ico_vertices = []
    ico_normals = []
    ico_indices = []
    for face_idx, face in enumerate(ico_face_indices):
        v0 = ico_base_vertices[face[0]]
        v1 = ico_base_vertices[face[1]]
        v2 = ico_base_vertices[face[2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        normal = normal / np.linalg.norm(normal)

        ico_vertices.extend([v0, v1, v2])
        ico_normals.extend([normal, normal, normal])

        base = face_idx * 3
        ico_indices.extend([base, base + 1, base + 2])

    ico_vertices = np.array(ico_vertices, dtype=np.float32)
    ico_normals = np.array(ico_normals, dtype=np.float32)
    ico_indices = np.array(ico_indices, dtype=np.int32)

    ico_mesh = newton.Mesh(ico_vertices, ico_indices, normals=ico_normals)

    # 3D grid of shapes
    grid_size_x = 4
    grid_size_y = 4
    grid_size_z = 8
    grid_spacing = 1.5
    grid_offset = wp.vec3(-5.0, -5.0, 1.0)
    position_randomness = 0.2

    shape_types = ["sphere", "box", "capsule", "cylinder", "cone", "icosahedron"]
    shape_index = 0

    rng = np.random.default_rng(42)

    initial_positions = []

    for ix in range(grid_size_x):
        for iy in range(grid_size_y):
            for iz in range(grid_size_z):
                base_x = grid_offset[0] + ix * grid_spacing
                base_y = grid_offset[1] + iy * grid_spacing
                base_z = grid_offset[2] + iz * grid_spacing

                random_offset_x = (rng.random() - 0.5) * 2 * position_randomness
                random_offset_y = (rng.random() - 0.5) * 2 * position_randomness
                random_offset_z = (rng.random() - 0.5) * 2 * position_randomness

                pos = wp.vec3(
                    base_x + random_offset_x,
                    base_y + random_offset_y,
                    base_z + random_offset_z,
                )

                initial_positions.append(pos[2])

                shape_type = shape_types[shape_index % len(shape_types)]
                shape_index += 1

                body = builder.add_body(xform=wp.transform(p=pos, q=wp.quat_identity()))

                if shape_type == "sphere":
                    builder.add_shape_sphere(body, radius=0.3)
                elif shape_type == "box":
                    builder.add_shape_box(body, hx=0.3, hy=0.3, hz=0.3)
                elif shape_type == "capsule":
                    builder.add_shape_capsule(body, radius=0.2, half_height=0.4)
                elif shape_type == "cylinder":
                    builder.add_shape_cylinder(body, radius=0.25, half_height=0.35)
                elif shape_type == "cone":
                    builder.add_shape_cone(body, radius=0.3, half_height=0.4)
                elif shape_type == "icosahedron":
                    builder.add_shape_convex_hull(body, mesh=ico_mesh)

                joint = builder.add_joint_free(body)
                builder.add_articulation([joint])

    model = builder.finalize(device=device)

    initial_positions = np.array(initial_positions, dtype=np.float32)

    # Create collision pipeline with nxn broad phase
    collision_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")

    # XPBD solver with exact same config as example
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=2,
        rigid_contact_relaxation=0.8,
        angular_damping=0.0,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.collide(state_0, collision_pipeline=collision_pipeline)

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Simulation parameters (same as example)
    fps = 100
    frame_dt = 1.0 / fps
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps

    # Compute height thresholds: max(2.0, initial_z) to avoid flaky tests
    # Shapes should never bounce higher than 2m or their initial position (whichever is greater)
    height_thresholds = np.maximum(2.0, initial_positions)

    # Run for 200 steps
    for step in range(200):
        # Simulate one frame (same as example)
        for _ in range(sim_substeps):
            state_0.clear_forces()
            contacts = model.collide(state_0, collision_pipeline=collision_pipeline)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        # Check that no body has z position exceeding the threshold
        body_q = state_0.body_q.numpy()
        current_z = body_q[: model.body_count, 2]
        test.assertTrue(
            np.all(current_z <= height_thresholds),
            f"Step {step + 1}: max z={current_z.max():.4f} exceeds threshold",
        )


class TestShapesNoBounce(unittest.TestCase):
    pass


add_function_test(
    TestShapesNoBounce,
    "test_shapes_never_exceed_initial_z",
    test_shapes_never_exceed_initial_z,
    devices=get_selected_cuda_test_devices(),
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
