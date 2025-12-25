# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

import itertools
import os
import unittest

import numpy as np
import warp as wp
import warp.examples

from newton import ModelBuilder
from newton._src.sim.graph_coloring import (
    ColoringAlgorithm,
    construct_trimesh_graph_edges,
    convert_to_color_groups,
    validate_graph_coloring,
)
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, assert_np_equal, get_test_devices


def create_lattice_grid(N):
    size = 10
    position = (0, 0)

    X = np.linspace(-0.5 * size + position[0], 0.5 * size + position[0], N)
    Y = np.linspace(-0.5 * size + position[1], 0.5 * size + position[1], N)

    X, Y = np.meshgrid(X, Y)

    Z = []
    for _i in range(N):
        Z.append(np.linspace(0, size, N))

    Z = np.array(Z)

    vs = []
    for i, j in itertools.product(range(N), range(N)):
        vs.append(wp.vec3((X[i, j], Y[i, j], Z[i, j])))

    fs = []
    for i, j in itertools.product(range(0, N - 1), range(0, N - 1)):
        vId = j + i * N

        if (j + i) % 2:
            fs.extend(
                [
                    vId,
                    vId + N + 1,
                    vId + 1,
                ]
            )
            fs.extend(
                [
                    vId,
                    vId + N,
                    vId + N + 1,
                ]
            )
        else:
            fs.extend(
                [
                    vId,
                    vId + N,
                    vId + 1,
                ]
            )
            fs.extend(
                [
                    vId + N,
                    vId + N + 1,
                    vId + 1,
                ]
            )

    return vs, fs


def color_lattice_grid(num_x, num_y):
    colors = []
    for _ in range(4):
        colors.append([])

    for xi in range(num_x + 1):
        for yi in range(num_y + 1):
            node_dx = yi * (num_x + 1) + xi

            a = 1 if xi % 2 else 0
            b = 1 if yi % 2 else 0

            c = b * 2 + a

            colors[c].append(node_dx)

    color_groups = [np.array(group) for group in colors]

    return color_groups


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_coloring_trimesh(test, device):
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    with wp.ScopedDevice(device):
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        vertices = np.array(usd_geom.GetPointsAttr().Get())
        faces = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        builder = ModelBuilder()

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=[wp.vec3(p) for p in vertices],
            indices=faces.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
        )

        model = builder.finalize()

        particle_colors = wp.empty(shape=(model.particle_count), dtype=int, device="cpu")

        edge_indices_cpu = wp.array(model.edge_indices.numpy()[:, 2:], dtype=int, device="cpu")

        # coloring without bending
        num_colors_greedy = wp.context.runtime.core.wp_graph_coloring(
            model.particle_count,
            edge_indices_cpu.__ctype__(),
            ColoringAlgorithm.GREEDY.value,
            particle_colors.__ctype__(),
        )
        wp.launch(
            kernel=validate_graph_coloring,
            inputs=[edge_indices_cpu, particle_colors],
            dim=edge_indices_cpu.shape[0],
            device="cpu",
        )

        num_colors_mcs = wp.context.runtime.core.wp_graph_coloring(
            model.particle_count,
            edge_indices_cpu.__ctype__(),
            ColoringAlgorithm.MCS.value,
            particle_colors.__ctype__(),
        )
        wp.launch(
            kernel=validate_graph_coloring,
            inputs=[edge_indices_cpu, particle_colors],
            dim=edge_indices_cpu.shape[0],
            device="cpu",
        )

        # coloring with bending
        edge_indices_cpu_with_bending = construct_trimesh_graph_edges(model.edge_indices, True)
        num_colors_greedy = wp.context.runtime.core.wp_graph_coloring(
            model.particle_count,
            edge_indices_cpu_with_bending.__ctype__(),
            ColoringAlgorithm.GREEDY.value,
            particle_colors.__ctype__(),
        )
        wp.context.runtime.core.wp_balance_coloring(
            model.particle_count,
            edge_indices_cpu_with_bending.__ctype__(),
            num_colors_greedy,
            1.1,
            particle_colors.__ctype__(),
        )
        wp.launch(
            kernel=validate_graph_coloring,
            inputs=[edge_indices_cpu_with_bending, particle_colors],
            dim=edge_indices_cpu_with_bending.shape[0],
            device="cpu",
        )

        num_colors_mcs = wp.context.runtime.core.wp_graph_coloring(
            model.particle_count,
            edge_indices_cpu_with_bending.__ctype__(),
            ColoringAlgorithm.MCS.value,
            particle_colors.__ctype__(),
        )
        max_min_ratio = wp.context.runtime.core.wp_balance_coloring(
            model.particle_count,
            edge_indices_cpu_with_bending.__ctype__(),
            num_colors_mcs,
            1.1,
            particle_colors.__ctype__(),
        )
        wp.launch(
            kernel=validate_graph_coloring,
            inputs=[edge_indices_cpu_with_bending, particle_colors],
            dim=edge_indices_cpu_with_bending.shape[0],
            device="cpu",
        )

        color_categories_balanced = convert_to_color_groups(num_colors_mcs, particle_colors)

        color_sizes = np.array([c.shape[0] for c in color_categories_balanced], dtype=np.float32)
        test.assertTrue(np.max(color_sizes) / np.min(color_sizes) <= max_min_ratio)

        # test if the color balance can quit from equilibrium
        builder = ModelBuilder()

        vs, fs = create_lattice_grid(100)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
        )

        builder.color(include_bending=True)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_combine_coloring(test, device):
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    with wp.ScopedDevice(device):
        builder1 = ModelBuilder()
        usd_stage = Usd.Stage.Open(os.path.join(wp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        vertices = np.array(usd_geom.GetPointsAttr().Get())
        faces = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        builder1.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=[wp.vec3(p) for p in vertices],
            indices=faces.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
        )

        builder1.add_cloth_grid(
            pos=wp.vec3(0.0, 4.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=50,
            dim_y=100,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            fix_left=True,
        )
        builder1.color()

        builder2 = ModelBuilder()
        builder2.add_cloth_grid(
            pos=wp.vec3(0.0, 4.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=50,
            dim_y=100,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            # to include bending in coloring
            edge_ke=100000,
            fix_left=True,
        )
        builder2.color()

        builder3 = ModelBuilder()
        builder3.add_cloth_grid(
            pos=wp.vec3(0.0, 4.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=50,
            dim_y=100,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            fix_left=True,
        )

        builder3.set_coloring(
            color_lattice_grid(50, 100),
        )

        builder1.add_world(builder2)
        builder1.add_world(builder3)

        model = builder2.finalize()

        particle_number_colored = np.full((model.particle_count), -1, dtype=int)
        particle_colors = np.full((model.particle_count), -1, dtype=int)
        for color, color_group in enumerate(model.particle_color_groups):
            particle_number_colored[color_group.numpy()] += 1
            particle_colors[color_group.numpy()] = color

        # all particles has been colored exactly once
        assert_np_equal(particle_number_colored, 0)

        edge_indices_cpu = wp.array(model.edge_indices.numpy()[:, 2:], dtype=int, device="cpu")
        wp.launch(
            kernel=validate_graph_coloring,
            inputs=[edge_indices_cpu, wp.array(particle_colors, dtype=int, device="cpu")],
            dim=edge_indices_cpu.shape[0],
            device="cpu",
        )


def test_coloring_rigid_body_cable_chain(test, device):
    """Test rigid body coloring for a cable chain (linear connectivity)."""
    with wp.ScopedDevice(device):
        builder = ModelBuilder()

        # Create a cable chain with 10 elements
        num_elements = 10
        cable_length = 2.0
        segment_length = cable_length / num_elements

        points = []
        for i in range(num_elements + 1):
            x = i * segment_length
            points.append(wp.vec3(x, 0.0, 1.0))

        # Create orientation (align capsule +Z with +X direction)
        rot_z_to_x = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(1.0, 0.0, 0.0))
        edge_q = [rot_z_to_x] * num_elements

        # Add cable using rod (creates bodies + cable joints)
        _rod_bodies, _rod_joints = builder.add_rod(
            positions=points,
            quaternions=edge_q,
            radius=0.05,
            bend_stiffness=1.0e2,
            bend_damping=1.0e-2,
            stretch_stiffness=1.0e6,
            stretch_damping=1.0e-2,
            key="test_cable",
        )

        # Apply coloring
        builder.color()

        # Finalize model
        model = builder.finalize()

        # Verify coloring exists
        test.assertGreater(len(model.body_color_groups), 0, "No body color groups generated")

        # Verify all bodies are colored exactly once
        body_color_count = np.zeros(model.body_count, dtype=int)
        for color_group in model.body_color_groups:
            color_group_np = color_group.numpy()
            test.assertTrue(np.all(color_group_np >= 0), "Invalid body index in color group")
            test.assertTrue(np.all(color_group_np < model.body_count), "Body index out of range")
            body_color_count[color_group_np] += 1

        test.assertTrue(np.all(body_color_count == 1), "Each body must be colored exactly once")

        # Verify adjacent bodies (connected by joints) have different colors
        body_colors = np.full(model.body_count, -1, dtype=int)
        for color_idx, color_group in enumerate(model.body_color_groups):
            body_colors[color_group.numpy()] = color_idx

        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()

        for i in range(len(joint_parent)):
            parent = joint_parent[i]
            child = joint_child[i]
            if parent >= 0 and child >= 0:  # Exclude world connections (-1)
                test.assertNotEqual(
                    body_colors[parent],
                    body_colors[child],
                    f"Joint {i}: parent body {parent} and child body {child} have same color",
                )

        # For a linear chain, expect 2 colors (alternating pattern)
        # This is optimal for cable chains
        test.assertLessEqual(
            len(model.body_color_groups),
            2,
            f"Cable chain should use at most 2 colors, got {len(model.body_color_groups)}",
        )


def test_coloring_rigid_body_color_algorithms(test, device):
    """Test different coloring algorithms (MCS vs GREEDY) for rigid bodies."""
    with wp.ScopedDevice(device):
        # Create a more complex cable structure for algorithm comparison
        builder_mcs = ModelBuilder()
        builder_greedy = ModelBuilder()

        num_elements = 20
        points = []
        for i in range(num_elements + 1):
            points.append(wp.vec3(float(i) * 0.1, 0.0, 1.0))

        rot_z_to_x = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(1.0, 0.0, 0.0))
        edge_q = [rot_z_to_x] * num_elements

        for b in (builder_mcs, builder_greedy):
            b.add_rod(
                positions=points,
                quaternions=edge_q,
                radius=0.05,
                bend_stiffness=1.0e2,
                bend_damping=1.0e-2,
                stretch_stiffness=1.0e6,
                stretch_damping=1.0e-2,
                key="test_cable",
            )

        # Test MCS algorithm
        builder_mcs.body_color_groups = []  # Reset
        builder_mcs.color(coloring_algorithm=ColoringAlgorithm.MCS)
        model_mcs = builder_mcs.finalize()

        # Test GREEDY algorithm
        builder_greedy.body_color_groups = []  # Reset
        builder_greedy.color(coloring_algorithm=ColoringAlgorithm.GREEDY)
        model_greedy = builder_greedy.finalize()

        # Both should produce valid colorings
        test.assertGreater(len(model_mcs.body_color_groups), 0, "MCS produced no colors")
        test.assertGreater(len(model_greedy.body_color_groups), 0, "GREEDY produced no colors")

        # Verify both colorings are valid (connected bodies have different colors)
        for model, name in [(model_mcs, "MCS"), (model_greedy, "GREEDY")]:
            body_colors = np.full(model.body_count, -1, dtype=int)
            for color_idx, color_group in enumerate(model.body_color_groups):
                body_colors[color_group.numpy()] = color_idx

            joint_parent = model.joint_parent.numpy()
            joint_child = model.joint_child.numpy()

            for i in range(len(joint_parent)):
                parent = joint_parent[i]
                child = joint_child[i]
                if parent >= 0 and child >= 0:
                    test.assertNotEqual(
                        body_colors[parent], body_colors[child], f"{name}: Joint {i} connects bodies with same color"
                    )


def test_coloring_rigid_body_no_joints(test, device):
    """Test rigid body coloring when there are no joints (all bodies independent)."""
    with wp.ScopedDevice(device):
        builder = ModelBuilder()

        # Add 5 independent bodies (no joints)
        for i in range(5):
            body = builder.add_body(xform=wp.transform(wp.vec3(float(i), 0.0, 1.0), wp.quat_identity()))
            builder.add_shape_capsule(body, radius=0.05, half_height=0.25)

        # Apply coloring
        builder.color()

        # Finalize model
        model = builder.finalize()

        # With no joints, all bodies can have the same color
        test.assertEqual(len(model.body_color_groups), 1, "Expected 1 color group for independent bodies")
        test.assertEqual(model.body_color_groups[0].size, 5, "All 5 bodies should be in same color group")


devices = get_test_devices()


class TestColoring(unittest.TestCase):
    pass


add_function_test(TestColoring, "test_coloring_trimesh", test_coloring_trimesh, devices=devices, check_output=False)
add_function_test(TestColoring, "test_combine_coloring", test_combine_coloring, devices=devices)

# Rigid body coloring tests
add_function_test(
    TestColoring, "test_coloring_rigid_body_cable_chain", test_coloring_rigid_body_cable_chain, devices=devices
)
add_function_test(
    TestColoring,
    "test_coloring_rigid_body_color_algorithms",
    test_coloring_rigid_body_color_algorithms,
    devices=devices,
)
add_function_test(
    TestColoring, "test_coloring_rigid_body_no_joints", test_coloring_rigid_body_no_joints, devices=devices
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
