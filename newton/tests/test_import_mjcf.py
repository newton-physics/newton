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

import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry.types import GeoType
from newton._src.sim.builder import ShapeFlags
from newton.solvers import SolverMuJoCo


class TestImportMjcf(unittest.TestCase):
    def test_humanoid_mjcf(self):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 123.0
        builder.default_shape_cfg.kd = 456.0
        builder.default_shape_cfg.mu = 789.0
        builder.default_joint_cfg.armature = 42.0
        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")
        builder.add_mjcf(
            mjcf_filename,
            ignore_names=["floor", "ground"],
            up_axis="Z",
        )
        # Filter out sites when checking shape material properties (sites don't have these attributes)
        non_site_indices = [i for i, flags in enumerate(builder.shape_flags) if not (flags & ShapeFlags.SITE)]
        self.assertTrue(all(np.array(builder.shape_material_ke)[non_site_indices] == 123.0))
        self.assertTrue(all(np.array(builder.shape_material_kd)[non_site_indices] == 456.0))
        self.assertTrue(all(np.array(builder.shape_material_mu)[non_site_indices] == 789.0))
        self.assertTrue(all(np.array(builder.joint_armature[:6]) == 0.0))
        self.assertEqual(
            builder.joint_armature[6:],
            [
                0.02,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.007,
                0.006,
                0.006,
                0.01,
                0.01,
                0.01,
                0.007,
                0.006,
                0.006,
                0.01,
                0.01,
                0.006,
                0.01,
                0.01,
                0.006,
            ],
        )
        assert builder.body_count == 13

    def test_mjcf_maxhullvert_parsing(self):
        """Test that maxhullvert is parsed from MJCF files"""
        # Create a temporary MJCF file with maxhullvert attribute
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <asset>
        <mesh name="mesh1" file="mesh1.obj" maxhullvert="32"/>
        <mesh name="mesh2" file="mesh2.obj" maxhullvert="128"/>
        <mesh name="mesh3" file="mesh3.obj"/>
    </asset>
    <worldbody>
        <body>
            <geom type="mesh" mesh="mesh1"/>
            <geom type="mesh" mesh="mesh2"/>
            <geom type="mesh" mesh="mesh3"/>
        </body>
    </worldbody>
</mujoco>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = os.path.join(tmpdir, "test.xml")

            # Create dummy mesh files
            for i in range(1, 4):
                mesh_path = os.path.join(tmpdir, f"mesh{i}.obj")
                with open(mesh_path, "w") as f:
                    # Simple triangle mesh
                    f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)

            # Parse MJCF
            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf_path, parse_meshes=True)
            model = builder.finalize()

            # Check that meshes have correct maxhullvert values
            # Note: This assumes meshes are added in order they appear in MJCF
            meshes = [model.shape_source[i] for i in range(3) if hasattr(model.shape_source[i], "maxhullvert")]

            if len(meshes) >= 3:
                self.assertEqual(meshes[0].maxhullvert, 32)
                self.assertEqual(meshes[1].maxhullvert, 128)
                self.assertEqual(meshes[2].maxhullvert, 64)  # Default value

    def test_inertia_rotation(self):
        """Test that inertia tensors are properly rotated using sandwich product R @ I @ R.T"""

        # Test case 1: Diagonal inertia with rotation
        mjcf_diagonal = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_diagonal">
    <worldbody>
        <body>
            <inertial pos="0 0 0" quat="0.7071068 0 0 0.7071068"
                      mass="1.0" diaginertia="1.0 2.0 3.0"/>
        </body>
    </worldbody>
</mujoco>
"""

        # Test case 2: Full inertia with rotation
        mjcf_full = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_full">
    <worldbody>
        <body>
            <inertial pos="0 0 0" quat="0.7071068 0 0 0.7071068"
                      mass="1.0" fullinertia="1.0 2.0 3.0 0.1 0.2 0.3"/>
        </body>
    </worldbody>
</mujoco>
"""

        # Test diagonal inertia rotation
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_diagonal, ignore_inertial_definitions=False)
        model = builder.finalize()

        # The quaternion (0.7071068, 0, 0, 0.7071068) in MuJoCo WXYZ format represents a 90-degree rotation around Z-axis
        # This transforms the diagonal inertia [1, 2, 3] to [2, 1, 3] via sandwich product R @ I @ R.T
        expected_diagonal = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])

        actual_inertia = model.body_inertia.numpy()[0]
        # The validation may add a small epsilon for numerical stability
        # Check that the values are close within a reasonable tolerance
        np.testing.assert_allclose(actual_inertia, expected_diagonal, rtol=1e-5, atol=1e-5)

        # Test full inertia rotation
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_full, ignore_inertial_definitions=False)
        model = builder.finalize()

        # For full inertia, we need to compute the expected result manually
        # Original inertia matrix:
        # [1.0  0.1  0.2]
        # [0.1  2.0  0.3]
        # [0.2  0.3  3.0]

        # The quaternion (0.7071068, 0, 0, 0.7071068) transforms the inertia
        # We need to use the same quaternion-to-matrix conversion as the MJCF importer

        original_inertia = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])

        # For full inertia, calculate the expected result analytically using the same quaternion
        # Original inertia matrix:
        # [1.0  0.1  0.2]
        # [0.1  2.0  0.3]
        # [0.2  0.3  3.0]

        # The quaternion (0.7071068, 0, 0, 0.7071068) in MuJoCo WXYZ format represents a 90-degree rotation around Z-axis
        # Calculate the expected result analytically using the correct rotation matrix
        # For a 90-degree Z-axis rotation: R = [0 -1 0; 1 0 0; 0 0 1]

        original_inertia = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])

        # Rotation matrix for 90-degree rotation around Z-axis
        rotation_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        expected_full = rotation_matrix @ original_inertia @ rotation_matrix.T

        actual_inertia = model.body_inertia.numpy()[0]

        # The original inertia violates the triangle inequality, so validation will correct it
        # The eigenvalues are [0.975, 1.919, 3.106], which violates I1 + I2 >= I3
        # The validation adds ~0.212 to all eigenvalues to fix this
        # We check that:
        # 1. The rotation structure is preserved (off-diagonal elements match)
        # 2. The diagonal has been increased by approximately the same amount

        # Check off-diagonal elements are preserved
        np.testing.assert_allclose(actual_inertia[0, 1], expected_full[0, 1], atol=1e-6)
        np.testing.assert_allclose(actual_inertia[0, 2], expected_full[0, 2], atol=1e-6)
        np.testing.assert_allclose(actual_inertia[1, 2], expected_full[1, 2], atol=1e-6)

        # Check that diagonal elements have been increased by approximately the same amount
        corrections = np.diag(actual_inertia - expected_full)
        np.testing.assert_allclose(corrections, corrections[0], rtol=1e-3)

        # Verify that the rotation was actually applied (not just identity)
        assert not np.allclose(actual_inertia, original_inertia, atol=1e-6)

    def test_single_body_transform(self):
        """Test 1: Single body with pos/quat → verify body_q matches expected world transform."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="test_body" pos="1.0 2.0 3.0" quat="0.7071068 0 0 0.7071068">
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Expected: translation (1, 2, 3) + 90° rotation around Z
        body_idx = model.body_key.index("test_body")
        body_q = model.body_q.numpy()
        body_pos = body_q[body_idx, :3]
        body_quat = body_q[body_idx, 3:]

        np.testing.assert_allclose(body_pos, [1.0, 2.0, 3.0], atol=1e-6)
        # MJCF quat is [w, x, y, z], body_q quat is [x, y, z, w]
        # So [0.7071068, 0, 0, 0.7071068] becomes [0, 0, 0.7071068, 0.7071068]
        np.testing.assert_allclose(body_quat, [0, 0, 0.7071068, 0.7071068], atol=1e-6)

    def test_root_body_with_custom_xform(self):
        """Test 1: Root body with custom xform parameter (with rotation) → verify transform is properly applied."""
        # Add a 45-degree rotation around Z to the body
        angle_body = np.pi / 4
        quat_body = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_body)
        # wp.quat_from_axis_angle returns [x, y, z, w]
        # MJCF expects [w, x, y, z]
        quat_body_mjcf = f"{quat_body[3]} {quat_body[0]} {quat_body[1]} {quat_body[2]}"
        mjcf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="test_body" pos="0.5 0.5 0.0" quat="{quat_body_mjcf}">
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        # Custom xform: translate by (10, 20, 30) and rotate 90 deg around Z
        angle_xform = np.pi / 2
        quat_xform = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_xform)
        custom_xform = wp.transform(wp.vec3(10.0, 20.0, 30.0), quat_xform)

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, xform=custom_xform)
        model = builder.finalize()

        # Compose transforms using warp
        body_xform = wp.transform(wp.vec3(0.5, 0.5, 0.0), quat_body)
        expected_xform = wp.transform_multiply(custom_xform, body_xform)
        expected_pos = expected_xform.p
        expected_quat = expected_xform.q

        body_idx = model.body_key.index("test_body")
        body_q = model.body_q.numpy()
        body_pos = body_q[body_idx, :3]
        body_quat = body_q[body_idx, 3:]

        np.testing.assert_allclose(body_pos, expected_pos, atol=1e-6)
        np.testing.assert_allclose(body_quat, expected_quat, atol=1e-6)

    def test_multiple_bodies_hierarchy(self):
        """Test 1: Multiple bodies in hierarchy → verify child transforms are correctly composed."""
        # Root is translated and rotated (45 deg around Z)
        angle_root = np.pi / 4
        quat_root = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle_root)
        # MJCF expects [w, x, y, z]
        quat_root_mjcf = f"{quat_root[3]} {quat_root[0]} {quat_root[1]} {quat_root[2]}"
        mjcf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="root" pos="2 3 0" quat="{quat_root_mjcf}">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child" pos="1 0 0" quat="0.7071068 0 0 0.7071068">
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Get all body transforms at once
        body_q = model.body_q.numpy()

        # Root: (2, 3, 0), 45 deg Z
        root_idx = model.body_key.index("root")
        root_pos = body_q[root_idx, :3]
        root_quat = body_q[root_idx, 3:]
        np.testing.assert_allclose(root_pos, [2, 3, 0], atol=1e-6)
        np.testing.assert_allclose(root_quat, quat_root, atol=1e-6)

        # Child: (1, 0, 0) in root frame, 90° Z rotation
        child_idx = model.body_key.index("child")
        child_pos = body_q[child_idx, :3]
        child_quat = body_q[child_idx, 3:]

        # Compose transforms using warp
        quat_child_mjcf = np.array([0.7071068, 0, 0, 0.7071068])
        # MJCF: [w, x, y, z] → warp: [x, y, z, w]
        quat_child = np.array([quat_child_mjcf[1], quat_child_mjcf[2], quat_child_mjcf[3], quat_child_mjcf[0]])
        child_xform = wp.transform(wp.vec3(1.0, 0.0, 0.0), quat_child)
        root_xform = wp.transform(wp.vec3(2.0, 3.0, 0.0), quat_root)
        expected_xform = wp.transform_multiply(root_xform, child_xform)
        expected_pos = expected_xform.p
        expected_quat = expected_xform.q

        np.testing.assert_allclose(child_pos, expected_pos, atol=1e-6)
        np.testing.assert_allclose(child_quat, expected_quat, atol=1e-6)

    def test_floating_base_transform(self):
        """Test 2: Floating base body → verify joint_q contains correct world coordinates, including rotation."""
        # Add a rotation: 90 deg about Z axis
        angle = np.pi / 2
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
        # MJCF expects [w, x, y, z]
        quat_mjcf = f"{quat[3]} {quat[0]} {quat[1]} {quat[2]}"
        mjcf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="floating_body" pos="2.0 3.0 4.0" quat="{quat_mjcf}">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # For floating base, joint_q should contain the body's world transform
        body_idx = model.body_key.index("floating_body")
        joint_idx = model.joint_key.index("floating_body_freejoint")

        # Get joint arrays at once
        joint_q_start = model.joint_q_start.numpy()
        joint_q = model.joint_q.numpy()

        joint_start = joint_q_start[joint_idx]

        # Extract position and orientation from joint_q
        joint_pos = [joint_q[joint_start + 0], joint_q[joint_start + 1], joint_q[joint_start + 2]]
        # Extract quaternion from joint_q (warp: [x, y, z, w])
        joint_quat = [
            joint_q[joint_start + 3],
            joint_q[joint_start + 4],
            joint_q[joint_start + 5],
            joint_q[joint_start + 6],
        ]

        # Should match the body's world transform
        body_q = model.body_q.numpy()
        body_pos = body_q[body_idx, :3]
        body_quat = body_q[body_idx, 3:]
        np.testing.assert_allclose(joint_pos, body_pos, atol=1e-6)
        np.testing.assert_allclose(joint_quat, body_quat, atol=1e-6)

    def test_chain_with_rotations(self):
        """Test 3: Chain of bodies with different pos/quat → verify each body's world transform."""
        # Test chain with cumulative rotations
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="link1" pos="1 0 0" quat="0.7071068 0 0 0.7071068">
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="link2" pos="0 1 0" quat="0.7071068 0 0.7071068 0">
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Get all body transforms at once
        body_q = model.body_q.numpy()

        # Verify each link's world transform
        base_idx = model.body_key.index("base")
        link1_idx = model.body_key.index("link1")
        link2_idx = model.body_key.index("link2")

        # Base: identity
        base_pos = body_q[base_idx, :3]
        base_quat = body_q[base_idx, 3:]
        np.testing.assert_allclose(base_pos, [0, 0, 0], atol=1e-6)
        # Identity quaternion in [x, y, z, w] format is [0, 0, 0, 1]
        np.testing.assert_allclose(base_quat, [0, 0, 0, 1], atol=1e-6)

        # Link1: base * link1_local
        link1_pos = body_q[link1_idx, :3]
        link1_quat = body_q[link1_idx, 3:]

        # Expected: base_xform * link1_local_xform
        base_xform = wp.transform(wp.vec3(0, 0, 0), wp.quat(0, 0, 0, 1))
        link1_local_xform = wp.transform(wp.vec3(1, 0, 0), wp.quat(0, 0, 0.7071068, 0.7071068))
        expected_link1_xform = wp.transform_multiply(base_xform, link1_local_xform)

        np.testing.assert_allclose(link1_pos, expected_link1_xform.p, atol=1e-6)
        np.testing.assert_allclose(link1_quat, expected_link1_xform.q, atol=1e-6)

        # Link2: base * link1_local * link2_local
        link2_pos = body_q[link2_idx, :3]
        link2_quat = body_q[link2_idx, 3:]

        # Expected: link1_world_xform * link2_local_xform
        link2_local_xform = wp.transform(wp.vec3(0, 1, 0), wp.quat(0, 0.7071068, 0, 0.7071068))
        expected_link2_xform = wp.transform_multiply(expected_link1_xform, link2_local_xform)

        np.testing.assert_allclose(link2_pos, expected_link2_xform.p, atol=1e-6)
        np.testing.assert_allclose(link2_quat, expected_link2_xform.q, atol=1e-6)

    def test_bodies_with_scale(self):
        """Test 3: Bodies with scale → verify scaling is applied at each level."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="root" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child" pos="2 0 0">
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        # Parse with scale=2.0
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, scale=2.0)
        model = builder.finalize()

        # Get all body transforms at once
        body_q = model.body_q.numpy()

        # Verify scaling is applied correctly
        root_idx = model.body_key.index("root")
        child_idx = model.body_key.index("child")

        # Root: no change
        root_pos = body_q[root_idx, :3]
        np.testing.assert_allclose(root_pos, [0, 0, 0], atol=1e-6)

        # Child: position scaled by 2.0
        child_pos = body_q[child_idx, :3]
        np.testing.assert_allclose(child_pos, [4, 0, 0], atol=1e-6)  # 2 * 2 = 4

    def test_tree_hierarchy_with_branching(self):
        """Test 3: Tree hierarchy with branching → verify transforms are correctly composed in all branches."""
        # Test a tree structure: root -> branch1 -> leaf1, and root -> branch2 -> leaf2
        # This tests that transforms are properly composed in parallel branches
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="root" pos="0 0 0" quat="0.7071068 0 0 0.7071068">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="branch1" pos="1 0 0" quat="0.7071068 0 0.7071068 0">
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="leaf1" pos="0 1 0" quat="1 0 0 0">
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
            <body name="branch2" pos="-1 0 0" quat="0.7071068 0.7071068 0 0">
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="leaf2" pos="0 0 1" quat="0.7071068 0 0 0.7071068">
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Get all body transforms at once
        body_q = model.body_q.numpy()

        # Verify transforms in all branches
        root_idx = model.body_key.index("root")
        branch1_idx = model.body_key.index("branch1")
        branch2_idx = model.body_key.index("branch2")
        leaf1_idx = model.body_key.index("leaf1")
        leaf2_idx = model.body_key.index("leaf2")

        # Root: (0, 0, 0), 90° Z rotation
        root_pos = body_q[root_idx, :3]
        root_quat = body_q[root_idx, 3:]
        np.testing.assert_allclose(root_pos, [0, 0, 0], atol=1e-6)
        # MJCF quat [0.7071068, 0, 0, 0.7071068] becomes [0, 0, 0.7071068, 0.7071068] in body_q
        np.testing.assert_allclose(root_quat, [0, 0, 0.7071068, 0.7071068], atol=1e-6)

        # Branch1: root * branch1_local
        branch1_pos = body_q[branch1_idx, :3]
        branch1_quat = body_q[branch1_idx, 3:]

        # Calculate expected using warp transforms
        root_xform = wp.transform(wp.vec3(0, 0, 0), wp.quat(0, 0, 0.7071068, 0.7071068))
        # MJCF quat "0.7071068 0 0.7071068 0" is [w, x, y, z] -> convert to [x, y, z, w]
        branch1_local_quat = wp.quat(0, 0.7071068, 0, 0.7071068)
        branch1_local_xform = wp.transform(wp.vec3(1, 0, 0), branch1_local_quat)
        expected_branch1_xform = wp.transform_multiply(root_xform, branch1_local_xform)

        np.testing.assert_allclose(branch1_pos, expected_branch1_xform.p, atol=1e-6)
        np.testing.assert_allclose(branch1_quat, expected_branch1_xform.q, atol=1e-6)

        # Leaf1: root * branch1_local * leaf1_local
        leaf1_pos = body_q[leaf1_idx, :3]
        leaf1_quat = body_q[leaf1_idx, 3:]

        # MJCF quat "1 0 0 0" is [w, x, y, z] -> convert to [x, y, z, w]
        leaf1_local_quat = wp.quat(0, 0, 0, 1)  # Identity quaternion
        leaf1_local_xform = wp.transform(wp.vec3(0, 1, 0), leaf1_local_quat)
        expected_leaf1_xform = wp.transform_multiply(expected_branch1_xform, leaf1_local_xform)

        np.testing.assert_allclose(leaf1_pos, expected_leaf1_xform.p, atol=1e-6)
        np.testing.assert_allclose(leaf1_quat, expected_leaf1_xform.q, atol=1e-6)

        # Branch2: root * branch2_local
        branch2_pos = body_q[branch2_idx, :3]
        branch2_quat = body_q[branch2_idx, 3:]

        # MJCF quat "0.7071068 0.7071068 0 0" is [w, x, y, z] -> convert to [x, y, z, w]
        branch2_local_quat = wp.quat(0.7071068, 0, 0, 0.7071068)
        branch2_local_xform = wp.transform(wp.vec3(-1, 0, 0), branch2_local_quat)
        expected_branch2_xform = wp.transform_multiply(root_xform, branch2_local_xform)

        np.testing.assert_allclose(branch2_pos, expected_branch2_xform.p, atol=1e-6)
        np.testing.assert_allclose(branch2_quat, expected_branch2_xform.q, atol=1e-6)

        # Leaf2: root * branch2_local * leaf2_local
        leaf2_pos = body_q[leaf2_idx, :3]
        leaf2_quat = body_q[leaf2_idx, 3:]

        # MJCF quat "0.7071068 0 0 0.7071068" is [w, x, y, z] -> convert to [x, y, z, w]
        leaf2_local_quat = wp.quat(0, 0, 0.7071068, 0.7071068)
        leaf2_local_xform = wp.transform(wp.vec3(0, 0, 1), leaf2_local_quat)
        expected_leaf2_xform = wp.transform_multiply(expected_branch2_xform, leaf2_local_xform)

        np.testing.assert_allclose(leaf2_pos, expected_leaf2_xform.p, atol=1e-6)
        np.testing.assert_allclose(leaf2_quat, expected_leaf2_xform.q, atol=1e-6)

    def test_cylinder_shapes_preserved(self):
        """Test that cylinder geometries are properly imported as cylinders, not capsules."""
        # Create MJCF content with cylinder geometry
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="cylinder_test">
    <worldbody>
        <body name="test_body">
            <geom type="cylinder" size="0.5 1.0" />
            <geom type="cylinder" size="0.3 0.8" fromto="0 0 0 1 0 0" />
            <geom type="capsule" size="0.2 0.5" />
            <geom type="box" size="0.4 0.4 0.4" />
        </body>
    </worldbody>
</mujoco>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = os.path.join(tmpdir, "cylinder_test.xml")
            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)

            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf_path)

            # Check that we have the correct number of shapes
            self.assertEqual(builder.shape_count, 4)

            # Check shape types
            shape_types = list(builder.shape_type)

            # First two shapes should be cylinders
            self.assertEqual(shape_types[0], GeoType.CYLINDER)
            self.assertEqual(shape_types[1], GeoType.CYLINDER)

            # Third shape should be capsule
            self.assertEqual(shape_types[2], GeoType.CAPSULE)

            # Fourth shape should be box
            self.assertEqual(shape_types[3], GeoType.BOX)

    def test_cylinder_properties_preserved(self):
        """Test that cylinder properties (radius, height) are correctly imported."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="cylinder_props_test">
    <worldbody>
        <body name="test_body">
            <geom type="cylinder" size="0.75 1.5" />
        </body>
    </worldbody>
</mujoco>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = os.path.join(tmpdir, "cylinder_props.xml")
            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)

            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf_path)

            # Check shape properties
            self.assertEqual(builder.shape_count, 1)
            self.assertEqual(builder.shape_type[0], GeoType.CYLINDER)

            # Check that radius and half_height are preserved
            # shape_scale stores (radius, half_height, 0) for cylinders
            shape_scale = builder.shape_scale[0]
            self.assertAlmostEqual(shape_scale[0], 0.75)  # radius
            self.assertAlmostEqual(shape_scale[1], 1.5)  # half_height

    def test_solreflimit_parsing(self):
        """Test that solreflimit joint attribute is correctly parsed and converted to limit_ke/limit_kd."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="solreflimit_test">
    <worldbody>
        <!-- Joint with standard mode solreflimit -->
        <body name="body1" pos="0 0 1">
            <joint name="joint1" type="hinge" axis="0 0 1" range="-45 45" solreflimit="0.03 0.9"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>

        <!-- Joint with direct mode solreflimit (negative values) -->
        <body name="body2" pos="1 0 1">
            <joint name="joint2" type="hinge" axis="0 0 1" range="-30 30" solreflimit="-100 -1"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>

        <!-- Joint without solreflimit (should use defaults) -->
        <body name="body3" pos="2 0 1">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-60 60"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = os.path.join(tmpdir, "solreflimit_test.xml")
            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)

            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf_path)
            model = builder.finalize()

            # Test we have 3 joints
            self.assertEqual(model.joint_count, 3)
            self.assertEqual(len(model.joint_limit_ke), 3)
            self.assertEqual(len(model.joint_limit_kd), 3)

            # Convert warp arrays to numpy for testing
            joint_limit_ke = model.joint_limit_ke.numpy()
            joint_limit_kd = model.joint_limit_kd.numpy()

            # Test joint1: standard mode solreflimit="0.03 0.9"
            # Expected: ke = 1/(0.03^2 * 0.9^2) = 1371.7421..., kd = 2.0/0.03 = 66.(6)
            expected_ke_1 = 1.0 / (0.03 * 0.03 * 0.9 * 0.9)
            expected_kd_1 = 2.0 / 0.03
            self.assertAlmostEqual(joint_limit_ke[0], expected_ke_1, places=2)
            self.assertAlmostEqual(joint_limit_kd[0], expected_kd_1, places=2)

            # Test joint2: direct mode solreflimit="-100 -1"
            # Expected: ke = 100, kd = 1
            self.assertAlmostEqual(joint_limit_ke[1], 100.0, places=2)
            self.assertAlmostEqual(joint_limit_kd[1], 1.0, places=2)

            # Test joint3: no solreflimit (should use default 0.02, 1.0)
            # Expected: ke = 1/(0.02^2 * 1.0^2) = 2500.0, kd = 2.0/0.02 = 100.0
            expected_ke_3 = 1.0 / (0.02 * 0.02 * 1.0 * 1.0)
            expected_kd_3 = 2.0 / 0.02
            self.assertAlmostEqual(joint_limit_ke[2], expected_ke_3, places=2)
            self.assertAlmostEqual(joint_limit_kd[2], expected_kd_3, places=2)

    def test_solimplimit_parsing(self):
        """Test that solimplimit attribute is parsed correctly from MJCF."""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="body1">
            <joint name="joint1" type="hinge" axis="0 1 0" solimplimit="0.89 0.9 0.01 2.1 1.8" range="-45 45" />
            <joint name="joint2" type="hinge" axis="1 0 0" range="-30 30" />
            <geom type="box" size="0.1 0.1 0.1" />
        </body>
        <body name="body2">
            <joint name="joint3" type="hinge" axis="0 0 1" solimplimit="0.8 0.85 0.002 0.6 1.5" range="-90 90" />
            <geom type="sphere" size="0.05" />
        </body>
    </worldbody>
</mujoco>
"""

        builder = newton.ModelBuilder()

        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Check if solimplimit custom attribute exists
        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "solimplimit"), "Model should have solimplimit attribute")

        solimplimit = model.mujoco.solimplimit.numpy()

        # Newton model has only 2 joints because it combines the ones under the same body into a single joint
        self.assertEqual(model.joint_count, 2, "Should have 2 joints")

        # Find joints by name
        joint_names = model.joint_key
        joint1_idx = joint_names.index("joint1_joint2")
        joint2_idx = joint_names.index("joint3")

        # For the merged joint (joint1_idx), both joint1 and joint2 should be present in the qd array.
        # We don't know the order, but both expected values should be present at joint1_idx and joint1_idx + 1.
        joint1_qd_start = model.joint_qd_start.numpy()[joint1_idx]
        # The joint should have 2 DoFs (since joint1 and joint2 are merged)
        self.assertEqual(model.joint_dof_dim.numpy()[joint1_idx, 1], 2)
        expected_joint1 = [0.89, 0.9, 0.01, 2.1, 1.8]  # from joint1
        expected_joint2 = [0.9, 0.95, 0.001, 0.5, 2.0]  # from joint2 (default values)
        val_qd_0 = solimplimit[joint1_qd_start, :]
        val_qd_1 = solimplimit[joint1_qd_start + 1, :]

        # Helper to check if two arrays match within tolerance
        def arrays_match(arr, expected, tol=1e-4):
            return all(abs(arr[i] - expected[i]) < tol for i in range(len(expected)))

        # The two DoFs should be exactly one joint1 and one default, in _some_ order
        if arrays_match(val_qd_0, expected_joint1):
            self.assertTrue(
                arrays_match(val_qd_1, expected_joint2), "Second DoF should have default solimplimit values"
            )
        elif arrays_match(val_qd_0, expected_joint2):
            self.assertTrue(
                arrays_match(val_qd_1, expected_joint1), "Second DoF should have joint1's solimplimit values"
            )
        else:
            self.fail(f"First DoF solimplimit {val_qd_0.tolist()} doesn't match either expected value")

        # Test joint3: explicit solimplimit with different values
        joint3_qd_start = model.joint_qd_start.numpy()[joint2_idx]
        expected_joint3 = [0.8, 0.85, 0.002, 0.6, 1.5]
        for i, expected in enumerate(expected_joint3):
            self.assertAlmostEqual(
                solimplimit[joint3_qd_start, i], expected, places=4, msg=f"joint3 solimplimit[{i}] should be {expected}"
            )

        # Test with MuJoCo solver - verify jnt_solimp values match using the mapping
        solver = SolverMuJoCo(model, separate_worlds=False)

        # MuJoCo's jnt_solimp should match our solimplimit values
        jnt_solimp = solver.mjw_model.jnt_solimp.numpy()
        joint_mjc_dof_start = solver.joint_mjc_dof_start.numpy()

        # For each Newton joint, verify its DOFs map correctly to MuJoCo
        for newton_joint_idx in range(model.joint_count):
            mjc_dof_start = joint_mjc_dof_start[newton_joint_idx]
            newton_dof_start = model.joint_qd_start.numpy()[newton_joint_idx]
            dof_count = model.joint_dof_dim.numpy()[newton_joint_idx].sum()

            # Check each DOF in this joint
            for dof_offset in range(dof_count):
                newton_dof_idx = newton_dof_start + dof_offset
                mjc_dof_idx = mjc_dof_start + dof_offset

                # Get expected solimplimit from Newton model
                expected_solimp = solimplimit[newton_dof_idx, :].tolist()

                # Get actual jnt_solimp from MuJoCo
                actual_solimp = jnt_solimp[0, mjc_dof_idx, :].tolist()
                # Verify they match
                self.assertTrue(
                    arrays_match(actual_solimp, expected_solimp),
                    f"MuJoCo jnt_solimp[{mjc_dof_idx}] = {actual_solimp} doesn't match "
                    f"Newton solimplimit[{newton_dof_idx}] = {expected_solimp} "
                    f"for joint {newton_joint_idx} DOF {dof_offset}",
                )

    def test_solreffriction_parsing(self):
        """Test that solreffriction attribute is parsed correctly from MJCF."""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="body1">
            <joint name="joint1" type="hinge" axis="0 1 0" solreffriction="0.01 0.5" range="-45 45" />
            <joint name="joint2" type="hinge" axis="1 0 0" range="-30 30" />
            <geom type="box" size="0.1 0.1 0.1" />
        </body>
        <body name="body2">
            <joint name="joint3" type="hinge" axis="0 0 1" solreffriction="0.05 2.0" range="-90 90" />
            <geom type="sphere" size="0.05" />
        </body>
    </worldbody>
</mujoco>
"""

        builder = newton.ModelBuilder()

        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Check if solreffriction custom attribute exists
        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "solreffriction"), "Model should have solreffriction attribute")

        solreffriction = model.mujoco.solreffriction.numpy()

        # Newton model has only 2 joints because it combines the ones under the same body into a single joint
        self.assertEqual(model.joint_count, 2, "Should have 2 joints")

        # Find joints by name
        joint_names = model.joint_key
        joint1_idx = joint_names.index("joint1_joint2")
        joint2_idx = joint_names.index("joint3")

        # For the merged joint (joint1_idx), both joint1 and joint2 should be present in the qd array.
        joint1_qd_start = model.joint_qd_start.numpy()[joint1_idx]
        # The joint should have 2 DoFs (since joint1 and joint2 are merged)
        self.assertEqual(model.joint_dof_dim.numpy()[joint1_idx, 1], 2)
        expected_joint1 = [0.01, 0.5]  # from joint1
        expected_joint2 = [0.02, 1.0]  # from joint2 (default values)
        val_qd_0 = solreffriction[joint1_qd_start, :]
        val_qd_1 = solreffriction[joint1_qd_start + 1, :]

        # Helper to check if two arrays match within tolerance
        def arrays_match(arr, expected, tol=1e-4):
            return all(abs(arr[i] - expected[i]) < tol for i in range(len(expected)))

        # The two DoFs should be exactly one joint1 and one default, in _some_ order
        if arrays_match(val_qd_0, expected_joint1):
            self.assertTrue(
                arrays_match(val_qd_1, expected_joint2), "Second DoF should have default solreffriction values"
            )
        elif arrays_match(val_qd_0, expected_joint2):
            self.assertTrue(
                arrays_match(val_qd_1, expected_joint1), "Second DoF should have joint1's solreffriction values"
            )
        else:
            self.fail(f"First DoF solreffriction {val_qd_0.tolist()} doesn't match either expected value")

        # Test joint3: explicit solreffriction with different values
        joint3_qd_start = model.joint_qd_start.numpy()[joint2_idx]
        expected_joint3 = [0.05, 2.0]
        for i, expected in enumerate(expected_joint3):
            self.assertAlmostEqual(
                solreffriction[joint3_qd_start, i],
                expected,
                places=4,
                msg=f"joint3 solreffriction[{i}] should be {expected}",
            )

        # Test with MuJoCo solver - verify dof_solref values match using the mapping
        solver = SolverMuJoCo(model, separate_worlds=False)

        # MuJoCo's dof_solref should match our solreffriction values
        mjw_dof_solref = solver.mjw_model.dof_solref.numpy()
        joint_mjc_dof_start = solver.joint_mjc_dof_start.numpy()

        # For each Newton joint, verify its DOFs map correctly to MuJoCo
        for newton_joint_idx in range(model.joint_count):
            mjc_dof_start = joint_mjc_dof_start[newton_joint_idx]
            newton_dof_start = model.joint_qd_start.numpy()[newton_joint_idx]
            dof_count = model.joint_dof_dim.numpy()[newton_joint_idx].sum()

            # Check each DOF in this joint
            for dof_offset in range(dof_count):
                newton_dof_idx = newton_dof_start + dof_offset
                mjc_dof_idx = mjc_dof_start + dof_offset

                # Get expected solreffriction from Newton model
                expected_solref = solreffriction[newton_dof_idx, :].tolist()

                # Get actual dof_solref from MuJoCo
                actual_solref = mjw_dof_solref[0, mjc_dof_idx, :].tolist()
                # Verify they match
                self.assertTrue(
                    arrays_match(actual_solref, expected_solref),
                    f"MuJoCo dof_solref[{mjc_dof_idx}] = {actual_solref} doesn't match "
                    f"Newton solreffriction[{newton_dof_idx}] = {expected_solref} "
                    f"for joint {newton_joint_idx} DOF {dof_offset}",
                )

    def test_granular_loading_flags(self):
        """Test granular control over sites and visual shapes loading."""
        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")

        # Test 1: Load all (default behavior)
        builder_all = newton.ModelBuilder()
        builder_all.add_mjcf(mjcf_filename, ignore_names=["floor", "ground"], up_axis="Z")
        count_all = builder_all.shape_count

        # Test 2: Load sites only, no visual shapes
        builder_sites_only = newton.ModelBuilder()
        builder_sites_only.add_mjcf(
            mjcf_filename, parse_sites=True, parse_visuals=False, ignore_names=["floor", "ground"], up_axis="Z"
        )
        count_sites_only = builder_sites_only.shape_count

        # Test 3: Load visual shapes only, no sites
        builder_visuals_only = newton.ModelBuilder()
        builder_visuals_only.add_mjcf(
            mjcf_filename, parse_sites=False, parse_visuals=True, ignore_names=["floor", "ground"], up_axis="Z"
        )
        count_visuals_only = builder_visuals_only.shape_count

        # Test 4: Load neither (physics collision shapes only)
        builder_physics_only = newton.ModelBuilder()
        builder_physics_only.add_mjcf(
            mjcf_filename, parse_sites=False, parse_visuals=False, ignore_names=["floor", "ground"], up_axis="Z"
        )
        count_physics_only = builder_physics_only.shape_count

        # Verify behavior
        # When loading all, should have most shapes
        self.assertEqual(count_all, 41, "Loading all should give 41 shapes (sites + visuals + collision)")

        # Sites only should have sites + collision shapes
        self.assertEqual(count_sites_only, 41, "Sites only should give 41 shapes (22 sites + 19 collision)")

        # Visuals only should have collision shapes only (no sites)
        self.assertEqual(count_visuals_only, 19, "Visuals only should give 19 shapes (collision only, no sites)")

        # Physics only should have collision shapes only
        self.assertEqual(count_physics_only, 19, "Physics only should give 19 shapes (collision only)")

        # Verify that sites are actually filtered
        self.assertLess(count_visuals_only, count_all, "Excluding sites should reduce shape count")
        self.assertLess(count_physics_only, count_all, "Excluding sites and visuals should reduce shape count")

    def test_parse_sites_backward_compatibility(self):
        """Test that parse_sites parameter works and maintains backward compatibility."""
        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")

        # Default (should parse sites)
        builder1 = newton.ModelBuilder()
        builder1.add_mjcf(mjcf_filename, ignore_names=["floor", "ground"], up_axis="Z")

        # Explicitly enable sites
        builder2 = newton.ModelBuilder()
        builder2.add_mjcf(mjcf_filename, parse_sites=True, ignore_names=["floor", "ground"], up_axis="Z")

        # Should have same count
        self.assertEqual(builder1.shape_count, builder2.shape_count, "Default should parse sites")

        # Explicitly disable sites
        builder3 = newton.ModelBuilder()
        builder3.add_mjcf(mjcf_filename, parse_sites=False, ignore_names=["floor", "ground"], up_axis="Z")

        # Should have fewer shapes
        self.assertLess(builder3.shape_count, builder1.shape_count, "Disabling sites should reduce shape count")

    def test_parse_visuals_vs_hide_visuals(self):
        """Test the distinction between parse_visuals (loading) and hide_visuals (visibility)."""
        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")

        # Test 1: parse_visuals=False (don't load)
        builder_no_load = newton.ModelBuilder()
        builder_no_load.add_mjcf(
            mjcf_filename, parse_visuals=False, parse_sites=False, ignore_names=["floor", "ground"], up_axis="Z"
        )

        # Test 2: hide_visuals=True (load but hide)
        builder_hidden = newton.ModelBuilder()
        builder_hidden.add_mjcf(
            mjcf_filename, hide_visuals=True, parse_sites=False, ignore_names=["floor", "ground"], up_axis="Z"
        )

        # Note: nv_humanoid.xml doesn't have separate visual-only geometries
        # so both will have the same count (collision shapes only)
        # The important thing is that neither crashes and the API works correctly
        self.assertEqual(
            builder_no_load.shape_count,
            builder_hidden.shape_count,
            "For nv_humanoid.xml, both should have same count (no separate visuals)",
        )

        # Verify parse_visuals=False doesn't crash
        self.assertGreater(builder_no_load.shape_count, 0, "Should still load collision shapes")
        # Verify hide_visuals=True doesn't crash
        self.assertGreater(builder_hidden.shape_count, 0, "Should still load collision shapes")

    def test_mjcf_gravcomp(self):
        """Test parsing of gravcomp from MJCF"""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <body name="body1" gravcomp="0.5">
                    <geom type="sphere" size="0.1" />
                </body>
                <body name="body2" gravcomp="1.0">
                    <geom type="sphere" size="0.1" />
                </body>
                <body name="body3">
                    <geom type="sphere" size="0.1" />
                </body>
            </worldbody>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        # Register gravcomp
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "gravcomp"))

        gravcomp = model.mujoco.gravcomp.numpy()

        # Bodies are added in order
        self.assertAlmostEqual(gravcomp[0], 0.5)
        self.assertAlmostEqual(gravcomp[1], 1.0)
        self.assertAlmostEqual(gravcomp[2], 0.0)  # Default

    def test_joint_stiffness_damping(self):
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="stiffness_damping_comprehensive_test">
    <worldbody>
        <body name="body1" pos="0 0 1">
            <joint name="joint1" type="hinge" axis="0 0 1" stiffness="0.05" damping="0.5" range="-45 45"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="body2" pos="1 0 1">
            <joint name="joint2" type="hinge" axis="0 1 0" range="-30 30"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="body3" pos="2 0 1">
            <joint name="joint3" type="hinge" axis="1 0 0" stiffness="0.1" damping="0.8" range="-60 60"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="body4" pos="3 0 1">
            <joint name="joint4" type="hinge" axis="0 1 0" stiffness="0.02" damping="0.3" range="-90 90"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <position joint="joint1" kp="10000.0" kv="2000.0"/>
        <velocity joint="joint1" kv="500.0"/>
        <position joint="joint2" kp="5000.0" kv="1000.0"/>
        <velocity joint="joint3" kv="800.0"/>
        <velocity joint="joint4" kv="3000.0"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "dof_passive_stiffness"))
        self.assertTrue(hasattr(model.mujoco, "dof_passive_damping"))

        joint_names = model.joint_key
        joint_qd_start = model.joint_qd_start.numpy()
        joint_stiffness = model.mujoco.dof_passive_stiffness.numpy()
        joint_damping = model.mujoco.dof_passive_damping.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_target_kd = model.joint_target_kd.numpy()

        expected_values = {
            "joint1": {"stiffness": 0.05, "damping": 0.5, "target_ke": 10000.0, "target_kd": 500.0},
            "joint2": {"stiffness": 0.0, "damping": 0.0, "target_ke": 5000.0, "target_kd": 1000.0},
            "joint3": {"stiffness": 0.1, "damping": 0.8, "target_ke": 0.0, "target_kd": 800.0},
            "joint4": {"stiffness": 0.02, "damping": 0.3, "target_ke": 0.0, "target_kd": 3000.0},
        }

        for joint_name, expected in expected_values.items():
            joint_idx = joint_names.index(joint_name)
            dof_idx = joint_qd_start[joint_idx]
            self.assertAlmostEqual(joint_stiffness[dof_idx], expected["stiffness"], places=4)
            self.assertAlmostEqual(joint_damping[dof_idx], expected["damping"], places=4)
            self.assertAlmostEqual(joint_target_ke[dof_idx], expected["target_ke"], places=1)
            self.assertAlmostEqual(joint_target_kd[dof_idx], expected["target_kd"], places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
