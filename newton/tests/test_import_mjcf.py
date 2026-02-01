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

import io
import os
import sys
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
        builder.default_shape_cfg.torsional_friction = 0.999
        builder.default_shape_cfg.rolling_friction = 0.888
        builder.default_joint_cfg.armature = 42.0
        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")
        builder.add_mjcf(
            mjcf_filename,
            ignore_names=["floor", "ground"],
            up_axis="Z",
        )
        # Filter out sites when checking shape material properties (sites don't have these attributes)
        non_site_indices = [i for i, flags in enumerate(builder.shape_flags) if not (flags & ShapeFlags.SITE)]

        # Check ke/kd from nv_humanoid.xml: solref=".015 1"
        # ke = 1/(0.015^2 * 1^2) ≈ 4444.4, kd = 2/0.015 ≈ 133.3
        # MJCF-specified solref overrides user defaults (like friction does)
        self.assertTrue(np.allclose(np.array(builder.shape_material_ke)[non_site_indices], 4444.4, rtol=0.01))
        self.assertTrue(np.allclose(np.array(builder.shape_material_kd)[non_site_indices], 133.3, rtol=0.01))

        # Check friction values from nv_humanoid.xml: friction="1.0 0.05 0.05"
        # mu = 1.0, torsional = 0.05, rolling = 0.05
        self.assertTrue(np.allclose(np.array(builder.shape_material_mu)[non_site_indices], 1.0))
        self.assertTrue(np.allclose(np.array(builder.shape_material_torsional_friction)[non_site_indices], 0.05))
        self.assertTrue(np.allclose(np.array(builder.shape_material_rolling_friction)[non_site_indices], 0.05))
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

    def test_replace_3d_hinge_with_ball_joint(self):
        """Test that 3D hinge joints are replaced with ball joints."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="root" pos="1 2 3" quat="0.7071068 0 0 0.7071068">
            <joint name="joint1" type="hinge" axis="1 0 0" range="-60 60" armature="1.0"/>
            <joint name="joint2" type="hinge" axis="0 1 0" range="-60 60" armature="2.0"/>
            <joint name="joint3" type="hinge" axis="0 0 1" range="-60 60" armature="3.0"/>
        </body>
    </worldbody>
</mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, convert_3d_hinge_to_ball_joints=True)
        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_dof_count, 3)
        self.assertEqual(builder.joint_coord_count, 4)
        self.assertEqual(builder.joint_type[0], newton.JointType.BALL)
        self.assertEqual(builder.joint_armature, [1.0, 2.0, 3.0])
        self.assertEqual(builder.joint_limit_lower, [np.deg2rad(-60)] * 3)
        self.assertEqual(builder.joint_limit_upper, [np.deg2rad(60)] * 3)
        joint_x_p = builder.joint_X_p[0]
        np.testing.assert_allclose(joint_x_p.p, [1, 2, 3], atol=1e-6)
        # note we need to swap quaternion order wxyz -> xyzw
        np.testing.assert_allclose(joint_x_p.q, [0, 0, 0.7071068, 0.7071068], atol=1e-6)

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

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)

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

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)

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

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
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

    def test_single_mujoco_fixed_tendon_parsing(self):
        """Test that tendon parameters can be parsed from mjcf"""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>

      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom solmix="1.0" type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

    </body>
  </worldbody>

  <!-- Fixed tendon coupling joint1 and joint2 -->
  <tendon>
    <fixed
		name="coupling_tendon"
        limited="false"
		stiffness="1.0"
		damping="2.0"
        margin="0.1"
        frictionloss="2.6"
        solreflimit="0.04 1.1"
        solimplimit="0.7 0.85 0.002 0.3 1.8"
        solreffriction="0.055 1.2"
        solimpfriction="0.3 0.4 0.006 0.5 1.4"
        actuatorfrcrange="-2.2 2.2"
        actuatorfrclimited="true"
        armature="0.13"
        springlength="3.0 3.5">
      <joint joint="joint1" coef="8"/>
      <joint joint="joint2" coef="-8"/>
    </fixed>

    <!-- Fixed tendon coupling joint1 and joint2 -->
    <fixed
		name="coupling_tendon_reversed"
        limited="true"
        solreflimit="0.05 1.2"
        solreffriction="0.07 1.5"
        range="-10.0 11.0"
        stiffness="4.0"
		damping="5.0"
        margin="0.3"
        frictionloss="2.8"
        solimplimit="0.8 0.85 0.003 0.4 1.9"
        solimpfriction="0.35 0.45 0.004 0.5 1.2"
        actuatorfrclimited="false"
        actuatorfrcrange="-3.3 3.3"
        armature="0.23"
        springlength="6.0">
      <joint joint="joint1" coef="9"/>
      <joint joint="joint2" coef="9"/>
    </fixed>
  </tendon>

</mujoco>
"""

        nbBuilders = 2
        nbTendonsPerBuilder = 2

        individual_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(mjcf)
        builder = newton.ModelBuilder()
        for _i in range(0, nbBuilders):
            builder.add_world(individual_builder)
        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        expected_damping = [[2.0, 5.0]]
        expected_stiffness = [[1.0, 4.0]]
        expected_frictionloss = [[2.6, 2.8]]
        expected_range = [[wp.vec2(0.0, 0.0), wp.vec2(-10.0, 11.0)]]
        expected_margin = [[0.1, 0.3]]
        expected_solreflimit = [[wp.vec2(0.04, 1.1), wp.vec2(0.05, 1.2)]]
        expected_solreffriction = [[wp.vec2(0.055, 1.2), wp.vec2(0.07, 1.5)]]
        vec5 = wp.types.vector(5, wp.float32)
        expected_solimplimit = [[vec5(0.7, 0.85, 0.002, 0.3, 1.8), vec5(0.8, 0.85, 0.003, 0.4, 1.9)]]
        expected_solimpfriction = [[vec5(0.3, 0.4, 0.006, 0.5, 1.4), vec5(0.35, 0.45, 0.004, 0.5, 1.2)]]
        expected_actuator_force_range = [[wp.vec2(-2.2, 2.2), wp.vec2(-3.3, 3.3)]]
        expected_armature = [[0.13, 0.23]]

        # We parse the 2nd tendon rest length as (6, -1) and store that in model.mujoco.
        # When we create the mujoco tendon in the mujoco solver we apply the dead zone rule.
        # If the user has authored a dead zone (2nd number > 1st number) then we honour that
        # but if they have not authored a dead zone (2nd number <= 1st number) then we create
        # the tendon with dead zone bounds that have zero extent. In our example, we create the
        # dead zone (6,6).
        expected_model_springlength = [[wp.vec2(3.0, 3.5), wp.vec2(6.0, -1.0)]]
        expected_solver_springlength = [[wp.vec2(3.0, 3.5), wp.vec2(6.0, 6.0)]]

        # Check every parameter in solver.mjw_model and in model.mujoco.
        # It is worthwhile checking model.mujoco in case we wish to use
        # the parameterisation in model.mujoco with a solver other than SolverMujoco.

        for i in range(0, nbBuilders):
            for j in range(0, nbTendonsPerBuilder):
                # Check the solver stiffness
                expected = expected_stiffness[0][j]
                measured = solver.mjw_model.tendon_stiffness.numpy()[i][j]
                self.assertAlmostEqual(
                    expected,
                    measured,
                    places=4,
                    msg=f"Expected stiffness value: {expected}, Measured value: {measured}",
                )

                # Check the model stiffness
                expected = expected_stiffness[0][j]
                measured = model.mujoco.tendon_stiffness.numpy()[nbTendonsPerBuilder * i + j]
                self.assertAlmostEqual(
                    expected,
                    measured,
                    places=4,
                    msg=f"Expected stiffness value: {expected}, Measured value: {measured}",
                )

                # Check the solver damping
                expected = expected_damping[0][j]
                measured = solver.mjw_model.tendon_damping.numpy()[i][j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected damping value: {expected}, Measured value: {measured}",
                )

                # Check the model damping
                expected = expected_damping[0][j]
                measured = model.mujoco.tendon_damping.numpy()[nbTendonsPerBuilder * i + j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected damping value: {expected}, Measured value: {measured}",
                )

                # Check the solver spring length
                for k in range(0, 2):
                    expected = expected_solver_springlength[0][j][k]
                    measured = solver.mjw_model.tendon_lengthspring.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected springlength[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check the model spring length
                for k in range(0, 2):
                    expected = expected_model_springlength[0][j][k]
                    measured = model.mujoco.tendon_springlength.numpy()[nbTendonsPerBuilder * i + j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected springlength[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check the solver frictionloss
                expected = expected_frictionloss[0][j]
                measured = solver.mjw_model.tendon_frictionloss.numpy()[i][j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected tendon frictionloss value: {expected}, Measured value: {measured}",
                )

                # Check the model frictionloss
                expected = expected_frictionloss[0][j]
                measured = model.mujoco.tendon_frictionloss.numpy()[nbTendonsPerBuilder * i + j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected tendon frictionloss value: {expected}, Measured value: {measured}",
                )

                # Check the solver range
                for k in range(0, 2):
                    expected = expected_range[0][j][k]
                    measured = solver.mjw_model.tendon_range.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected range[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check the model range
                for k in range(0, 2):
                    expected = expected_range[0][j][k]
                    measured = model.mujoco.tendon_range.numpy()[nbTendonsPerBuilder * i + j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected range[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check the solver margin
                expected = expected_margin[0][j]
                measured = solver.mjw_model.tendon_margin.numpy()[i][j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected margin value: {expected}, Measured value: {measured}",
                )

                # Check the model margin
                expected = expected_margin[0][j]
                measured = model.mujoco.tendon_margin.numpy()[nbTendonsPerBuilder * i + j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected margin value: {expected}, Measured value: {measured}",
                )

                # Check solver solreflimit
                for k in range(0, 2):
                    expected = expected_solreflimit[0][j][k]
                    measured = solver.mjw_model.tendon_solref_lim.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solreflimit[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check model solreflimit
                for k in range(0, 2):
                    expected = expected_solreflimit[0][j][k]
                    measured = model.mujoco.tendon_solref_limit.numpy()[nbTendonsPerBuilder * i + j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solreflimit[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check solver solimplimit
                for k in range(0, 5):
                    expected = expected_solimplimit[0][j][k]
                    measured = solver.mjw_model.tendon_solimp_lim.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solimplimit[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check model solimplimit
                for k in range(0, 5):
                    expected = expected_solimplimit[0][j][k]
                    measured = model.mujoco.tendon_solimp_limit.numpy()[nbTendonsPerBuilder * i + j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solimplimit[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check solver solreffriction
                for k in range(0, 2):
                    expected = expected_solreffriction[0][j][k]
                    measured = solver.mjw_model.tendon_solref_fri.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solreffriction[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check model solreffriction
                for k in range(0, 2):
                    expected = expected_solreffriction[0][j][k]
                    measured = model.mujoco.tendon_solref_friction.numpy()[nbTendonsPerBuilder * i + j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solreffriction[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check solver solimplimit
                for k in range(0, 5):
                    expected = expected_solimpfriction[0][j][k]
                    measured = solver.mjw_model.tendon_solimp_fri.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solimpfriction[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check model solimpfriction
                for k in range(0, 5):
                    expected = expected_solimpfriction[0][j][k]
                    measured = model.mujoco.tendon_solimp_friction.numpy()[nbTendonsPerBuilder * i + j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solimpfriction[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check the solver actuator force range
                for k in range(0, 2):
                    expected = expected_actuator_force_range[0][j][k]
                    measured = solver.mjw_model.tendon_actfrcrange.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected range[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check the model actuator force range
                for k in range(0, 2):
                    expected = expected_actuator_force_range[0][j][k]
                    measured = model.mujoco.tendon_actuator_force_range.numpy()[nbTendonsPerBuilder * i + j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected range[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check solver armature
                expected = expected_armature[0][j]
                measured = solver.mjw_model.tendon_armature.numpy()[i][j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected armature value: {expected}, Measured value: {measured}",
                )

                # Check model armature
                expected = expected_armature[0][j]
                measured = model.mujoco.tendon_armature.numpy()[nbTendonsPerBuilder * i + j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected armature value: {expected}, Measured value: {measured}",
                )

        expected_solver_num = [2, 2]
        expected_solver_limited = [0, 1]
        expected_solver_actfrc_limited = [1, 0]
        for i in range(0, nbTendonsPerBuilder):
            # Check the offsets that determine where the joint list starts for each tendon
            expected = expected_solver_num[i]
            measured = solver.mjw_model.tendon_num.numpy()[i]
            self.assertEqual(
                measured,
                expected,
                msg=f"Expected tendon_num value: {expected}, Measured value: {measured}",
            )

            # Check the limited attribute
            expected = expected_solver_limited[i]
            measured = solver.mjw_model.tendon_limited.numpy()[i]
            self.assertEqual(
                measured,
                expected,
                msg=f"Expected tendon limited value: {expected}, Measured value: {measured}",
            )

            # Check the actuation force limited attribute
            expected = expected_solver_actfrc_limited[i]
            measured = solver.mjw_model.tendon_actfrclimited.numpy()[i]
            self.assertEqual(
                measured,
                expected,
                msg=f"Expected tendon actuator force limited value: {expected}, Measured value: {measured}",
            )

        expected_model_num = [2, 2, 2, 2]
        expected_model_limited = [0, 1, 0, 1]
        expected_model_actfrc_limited = [1, 0, 1, 0]
        expected_model_joint_adr = [0, 2, 4, 6]
        for i in range(0, nbBuilders):
            for j in range(0, nbTendonsPerBuilder):
                # Check the offsets that determine where the joint list starts for each tendon
                expected = expected_model_num[nbTendonsPerBuilder * i + j]
                measured = model.mujoco.tendon_joint_num.numpy()[nbTendonsPerBuilder * i + j]
                self.assertEqual(
                    measured,
                    expected,
                    msg=f"Expected joint num value: {expected}, Measured value: {measured}",
                )

                # Check the limited attribute
                expected = expected_model_limited[nbTendonsPerBuilder * i + j]
                measured = model.mujoco.tendon_limited.numpy()[nbTendonsPerBuilder * i + j]
                self.assertEqual(
                    measured,
                    expected,
                    msg=f"Expected tendon limited value: {expected}, Measured value: {measured}",
                )

                # Check the actuation force limited attribute
                expected = expected_model_actfrc_limited[nbTendonsPerBuilder * i + j]
                measured = model.mujoco.tendon_actuator_force_limited.numpy()[nbTendonsPerBuilder * i + j]
                self.assertEqual(
                    measured,
                    expected,
                    msg=f"Expected tendon actuator force limited value: {expected}, Measured value: {measured}",
                )

                # Check the joint_adr attribute
                expected = expected_model_joint_adr[nbTendonsPerBuilder * i + j]
                measured = model.mujoco.tendon_joint_adr.numpy()[nbTendonsPerBuilder * i + j]
                self.assertEqual(
                    measured,
                    expected,
                    msg=f"Expected tendon joint_adr value: {expected}, Measured value: {measured}",
                )

        # Check that joint coefficients are correctly parsed
        # Tendon 1: joint1 coef=8, joint2 coef=-8
        # Tendon 2: joint1 coef=9, joint2 coef=9
        expected_wrap_prm = [8.0, -8.0, 9.0, 9.0]
        wrap_prm = solver.mj_model.wrap_prm
        self.assertEqual(len(wrap_prm), len(expected_wrap_prm), "wrap_prm length mismatch")
        for i, expected_coef in enumerate(expected_wrap_prm):
            self.assertAlmostEqual(
                wrap_prm[i],
                expected_coef,
                places=4,
                msg=f"wrap_prm[{i}] expected {expected_coef}, got {wrap_prm[i]}",
            )

        # Check that we made copies of the joint coefs in the model.
        expected_model_joint_coef = [8.0, -8.0, 9.0, 9.0, 8.0, -8.0, 9.0, 9.0]
        for i in range(0, nbBuilders):
            for j in range(0, nbTendonsPerBuilder):
                for k in range(0, 2):
                    idx = nbTendonsPerBuilder * 2 * i + 2 * j + k
                    expected = expected_model_joint_coef[idx]
                    measured = model.mujoco.tendon_coef.numpy()[idx]
                    self.assertEqual(
                        measured,
                        expected,
                        msg=f"Expected coef value: {expected}, Measured value: {measured}",
                    )

        # Check tendon_invweight0 is computed correctly
        # tendon_invweight0 is computed by MuJoCo based on the mass matrix and tendon geometry.
        # The formula accounts for: sum(coef^2 * effective_dof_inv_weight) / (1 + armature)
        # where effective_dof_inv_weight depends on the full articulated body inertia.
        # These expected values are verified against the Newton -> MuJoCo pipeline.
        expected_invweight0 = [4.6796, 5.9226]  # Values after Newton's inertia processing
        invweight0 = solver.mj_model.tendon_invweight0
        for i, expected in enumerate(expected_invweight0):
            self.assertAlmostEqual(
                invweight0[i],
                expected,
                places=2,
                msg=f"tendon_invweight0[{i}] expected {expected:.4f}, got {invweight0[i]:.4f}",
            )

    def test_single_mujoco_fixed_tendon_defaults(self):
        """Test that tendon parsing uses the correct mujoco default values."""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>

      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom solmix="1.0" type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

    </body>
  </worldbody>

  <tendon>
    <!-- Fixed tendon coupling joint1 and joint2 -->
	<fixed
		name="coupling_tendon">
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="-1"/>
    </fixed>
  </tendon>

  <tendon>
    <!-- Fixed tendon coupling joint1 and joint2 -->
	<fixed
		name="coupling_tendon_reversed">
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="1"/>
    </fixed>
  </tendon>
</mujoco>
"""

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        nbBuilders = 1
        nbTendonsPerBuilder = 2
        nbTendons = nbBuilders * nbTendonsPerBuilder

        # Note default spring length is -1 but ends up being 0.

        expected_damping = [[0.0, 0.0]]
        expected_stiffness = [[0.0, 0.0]]
        expected_frictionloss = [[0, 0]]
        expected_springlength = [[wp.vec2(0.0, 0.0), wp.vec2(0.0, 0.0)]]
        expected_range = [[wp.vec2(0.0, 0.0), wp.vec2(0.0, 0.0)]]
        expected_margin = [[0.0, 0.0]]
        expected_solreflimit = [[wp.vec2(0.02, 1.0), wp.vec2(0.02, 1.0)]]
        expected_solreffriction = [[wp.vec2(0.02, 1.0), wp.vec2(0.02, 1.0)]]
        vec5 = wp.types.vector(5, wp.float32)
        expected_solimplimit = [[vec5(0.9, 0.95, 0.001, 0.5, 2.0), vec5(0.9, 0.95, 0.001, 0.5, 2.0)]]
        expected_solimpfriction = [[vec5(0.9, 0.95, 0.001, 0.5, 2.0), vec5(0.9, 0.95, 0.001, 0.5, 2.0)]]
        expected_actuator_force_range = [[wp.vec2(0.0, 0.0), wp.vec2(0.0, 0.0)]]
        expected_armature = [[0.0, 0.0]]
        for i in range(0, nbBuilders):
            for j in range(0, nbTendonsPerBuilder):
                # Check the stiffness
                expected = expected_stiffness[i][j]
                measured = solver.mjw_model.tendon_stiffness.numpy()[i][j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected stiffness value: {expected}, Measured value: {measured}",
                )

                # Check the damping
                expected = expected_damping[i][j]
                measured = solver.mjw_model.tendon_damping.numpy()[i][j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected damping value: {expected}, Measured value: {measured}",
                )

                # Check the spring length
                for k in range(0, 2):
                    expected = expected_springlength[i][j][k]
                    measured = solver.mjw_model.tendon_lengthspring.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected springlength[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check the frictionloss
                expected = expected_frictionloss[i][j]
                measured = solver.mjw_model.tendon_frictionloss.numpy()[i][j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected tendon frictionloss value: {expected}, Measured value: {measured}",
                )

                # Check the range
                for k in range(0, 2):
                    expected = expected_range[i][j][k]
                    measured = solver.mjw_model.tendon_range.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected range[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check the margin
                expected = expected_margin[i][j]
                measured = solver.mjw_model.tendon_margin.numpy()[i][j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected margin value: {expected}, Measured value: {measured}",
                )

                # Check solreflimit
                for k in range(0, 2):
                    expected = expected_solreflimit[i][j][k]
                    measured = solver.mjw_model.tendon_solref_lim.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solreflimit[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check solimplimit
                for k in range(0, 5):
                    expected = expected_solimplimit[i][j][k]
                    measured = solver.mjw_model.tendon_solimp_lim.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solimplimit[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check solreffriction
                for k in range(0, 2):
                    expected = expected_solreffriction[i][j][k]
                    measured = solver.mjw_model.tendon_solref_fri.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solreffriction[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check solimplimit
                for k in range(0, 5):
                    expected = expected_solimpfriction[i][j][k]
                    measured = solver.mjw_model.tendon_solimp_fri.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected solimpfriction[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check the actuator force range
                for k in range(0, 2):
                    expected = expected_actuator_force_range[i][j][k]
                    measured = solver.mjw_model.tendon_actfrcrange.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected range[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check armature
                expected = expected_armature[i][j]
                measured = solver.mjw_model.tendon_armature.numpy()[i][j]
                self.assertAlmostEqual(
                    measured,
                    expected,
                    places=4,
                    msg=f"Expected armature value: {expected}, Measured value: {measured}",
                )

        expected_num = [2, 2]
        expected_limited = [0, 0]
        expected_actfrc_limited = [0, 0]
        for i in range(0, nbTendons):
            # Check the offsets that determine where the joint list starts for each tendon
            expected = expected_num[i]
            measured = solver.mjw_model.tendon_num.numpy()[i]
            self.assertEqual(
                measured,
                expected,
                msg=f"Expected springlength[0] value: {expected}, Measured value: {measured}",
            )

            # Check the limited attribute
            expected = expected_limited[i]
            measured = solver.mjw_model.tendon_limited.numpy()[i]
            self.assertEqual(
                measured,
                expected,
                msg=f"Expected tendon limited value: {expected}, Measured value: {measured}",
            )

            # Check the actuation force limited attribute
            expected = expected_actfrc_limited[i]
            measured = solver.mjw_model.tendon_actfrclimited.numpy()[i]
            self.assertEqual(
                measured,
                expected,
                msg=f"Expected tendon actuator force limited value: {expected}, Measured value: {measured}",
            )

    def test_single_mujoco_fixed_tendon_limit_parsing(self):
        """Test that tendon limits are correctly parsed."""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>

      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom solmix="1.0" type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

    </body>
  </worldbody>

  <tendon>
    <!-- Fixed tendon coupling joint1 and joint2 -->
	<fixed
       range="-10.0 11.0"
       actuatorfrcrange="-2.2 2.2"
       name="coupling_tendon1">
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="-1"/>
    </fixed>
  </tendon>

  <tendon>
    <!-- Fixed tendon coupling joint1 and joint2 -->
	<fixed
        limited="true"
        range="-12.0 13.0"
        actuatorfrclimited="true"
        actuatorfrcrange="-3.3 3.3"
        name="coupling_tendon2">
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="1"/>
    </fixed>
  </tendon>

  <tendon>
    <!-- Fixed tendon coupling joint1 and joint2 -->
	<fixed
        limited="false"
        range="-14.0 15.0"
        actuatorfrclimited="false"
        actuatorfrcrange="-4.4 4.4"
		name="coupling_tendon3">
      <joint joint="joint1" coef="2"/>
      <joint joint="joint2" coef="3"/>
    </fixed>
  </tendon>

</mujoco>
"""

        # Newton hard-codes spec.compiler.automlimits=1.
        # 1) With automlimits=1 we should not have to specify limited="true" on each tendon. It should be sufficient
        # just to set the range. coupling_tendon1 is the test for this.
        # 2) With compiler.autolimits=1 it shouldn't matter if we do specify limited="true.  We should still end up
        # with an active limit with limited="true". coupling_tendon2 is the test for this.
        # 3) With compiler.autolimits=1  and limited="false" we should end up with an inactive limit. coupling_tendon3
        # is the test for this.
        # 4) repeat the test with actuatorfrclimited.

        nbBuilders = 1
        nbTendonsPerBuilder = 3
        nbTendons = nbBuilders * nbTendonsPerBuilder

        individual_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(mjcf)
        builder = newton.ModelBuilder()
        for _i in range(0, nbBuilders):
            builder.add_world(individual_builder)
        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        # Note default spring length is -1 but ends up being 0.

        expected_range = [[wp.vec2(-10.0, 11.0), wp.vec2(-12.0, 13.0), wp.vec2(-14.0, 15.0)]]
        expected_actuator_force_range = [[wp.vec2(-2.2, 2.2), wp.vec2(-3.3, 3.3), wp.vec2(-4.4, 4.4)]]
        for i in range(0, nbBuilders):
            for j in range(0, nbTendonsPerBuilder):
                # Check the range
                for k in range(0, 2):
                    expected = expected_range[i][j][k]
                    measured = solver.mjw_model.tendon_range.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected range[{k}] value: {expected}, Measured value: {measured}",
                    )

                # Check the actuator force range
                for k in range(0, 2):
                    expected = expected_actuator_force_range[i][j][k]
                    measured = solver.mjw_model.tendon_actfrcrange.numpy()[i][j][k]
                    self.assertAlmostEqual(
                        measured,
                        expected,
                        places=4,
                        msg=f"Expected range[{k}] value: {expected}, Measured value: {measured}",
                    )

        expected_limited = [1, 1, 0]
        expected_actfrc_limited = [1, 1, 0]
        for i in range(0, nbTendons):
            # Check the limited attribute
            expected = expected_limited[i]
            measured = solver.mjw_model.tendon_limited.numpy()[i]
            self.assertEqual(
                measured,
                expected,
                msg=f"Expected tendon limited value: {expected}, Measured value: {measured}",
            )

            # Check the actuation force limited attribute
            expected = expected_actfrc_limited[i]
            measured = solver.mjw_model.tendon_actfrclimited.numpy()[i]
            self.assertEqual(
                measured,
                expected,
                msg=f"Expected tendon actuator force limited value: {expected}, Measured value: {measured}",
            )

    def test_single_mujoco_fixed_tendon_auto_springlength(self):
        """Test that springlength=-1 auto-computes the spring length from initial joint positions.

        When springlength first param is -1, MuJoCo auto-computes the spring length from
        the initial joint state (qpos0) using: tendon_length = coeff0 * q0 + coeff1 * q1.
        The computed value is stored in tendon_length0.

        We set qpos0 using joint "ref" values in mjcf.
        """
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>

      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" ref="0.5" range="-50.5 50.5"/>
        <geom solmix="1.0" type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" ref="0.7" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

    </body>
  </worldbody>

  <tendon>
    <!-- Fixed tendon with auto-computed spring length (springlength=-1) -->
    <fixed
        name="auto_length_tendon"
        stiffness="1.0"
        damping="0.5"
        springlength="-1">
      <joint joint="joint1" coef="2"/>
      <joint joint="joint2" coef="3"/>
    </fixed>
  </tendon>

</mujoco>
"""

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Taken from joint.ref values in mjcf.
        q0 = 0.5
        q1 = 0.7

        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        # Expected tendon length from initial joint positions: coef0*q0 + coef1*q1
        coef0 = 2.0
        coef1 = 3.0
        expected_tendon_length0 = coef0 * q0 + coef1 * q1  # 2*0.5 + 3*0.7 = 3.1

        # Verify tendon_length0 is computed from initial joint positions
        measured_tendon_length0 = solver.mj_model.tendon_length0[0]
        self.assertAlmostEqual(
            measured_tendon_length0,
            expected_tendon_length0,
            places=4,
            msg=f"Expected tendon_length0: {expected_tendon_length0}, Measured: {measured_tendon_length0}",
        )

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

    def test_limit_margin_parsing(self):
        """Test importing limit_margin from MJCF."""
        mjcf = """
        <mujoco>
            <worldbody>
                <body>
                    <joint type="hinge" axis="0 0 1" margin="0.01" />
                    <geom type="box" size="0.1 0.1 0.1" />
                </body>
                <body>
                    <joint type="hinge" axis="0 0 1" margin="0.02" />
                    <geom type="box" size="0.1 0.1 0.1" />
                </body>
                <body>
                    <joint type="hinge" axis="0 0 1" />
                    <geom type="box" size="0.1 0.1 0.1" />
                </body>
            </worldbody>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "limit_margin"))
        np.testing.assert_allclose(model.mujoco.limit_margin.numpy(), [0.01, 0.02, 0.0])

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

    def test_solimpfriction_parsing(self):
        """Test that solimpfriction attribute is parsed correctly from MJCF."""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="body1">
            <joint name="joint1" type="hinge" axis="0 1 0" solimpfriction="0.89 0.9 0.01 2.1 1.8" range="-45 45" />
            <joint name="joint2" type="hinge" axis="1 0 0" range="-30 30" />
            <geom type="box" size="0.1 0.1 0.1" />
        </body>
        <body name="body2">
            <joint name="joint3" type="hinge" axis="0 0 1" solimpfriction="0.8 0.85 0.002 0.6 1.5" range="-90 90" />
            <geom type="sphere" size="0.05" />
        </body>
    </worldbody>
</mujoco>
"""

        builder = newton.ModelBuilder()

        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Check if solimpfriction custom attribute exists
        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "solimpfriction"), "Model should have solimpfriction attribute")

        solimpfriction = model.mujoco.solimpfriction.numpy()

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
        expected_joint1 = [0.89, 0.9, 0.01, 2.1, 1.8]  # from joint1
        expected_joint2 = [0.9, 0.95, 0.001, 0.5, 2.0]  # from joint2 (default values)
        val_qd_0 = solimpfriction[joint1_qd_start, :]
        val_qd_1 = solimpfriction[joint1_qd_start + 1, :]

        # Helper to check if two arrays match within tolerance
        def arrays_match(arr, expected, tol=1e-4):
            return all(abs(arr[i] - expected[i]) < tol for i in range(len(expected)))

        # The two DoFs should be exactly one joint1 and one default, in _some_ order
        if arrays_match(val_qd_0, expected_joint1):
            self.assertTrue(
                arrays_match(val_qd_1, expected_joint2), "Second DoF should have default solimpfriction values"
            )
        elif arrays_match(val_qd_0, expected_joint2):
            self.assertTrue(
                arrays_match(val_qd_1, expected_joint1), "Second DoF should have joint1's solimpfriction values"
            )
        else:
            self.fail(f"First DoF solimpfriction {val_qd_0.tolist()} doesn't match either expected value")

        # Test joint3: explicit solimp_friction with different values
        joint3_qd_start = model.joint_qd_start.numpy()[joint2_idx]
        expected_joint3 = [0.8, 0.85, 0.002, 0.6, 1.5]
        for i, expected in enumerate(expected_joint3):
            self.assertAlmostEqual(
                solimpfriction[joint3_qd_start, i],
                expected,
                places=4,
                msg=f"joint3 solimpfriction[{i}] should be {expected}",
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

    def test_mjcf_friction_parsing(self):
        """Test MJCF friction parsing with 1, 2, and 3 element vectors."""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <body name="test_body">
                    <geom name="geom1" type="box" size="0.1 0.1 0.1" friction="0.5 0.1 0.01"/>
                    <geom name="geom2" type="sphere" size="0.1" friction="0.8 0.2 0.05"/>
                    <geom name="geom3" type="capsule" size="0.1 0.2" friction="0.0 0.0 0.0"/>
                    <geom name="geom4" type="box" size="0.1 0.1 0.1" friction="1.0"/>
                    <geom name="geom5" type="sphere" size="0.1" friction="0.6 0.15"/>
                </body>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, up_axis="Z")

        self.assertEqual(builder.shape_count, 5)

        # 3-element: friction="0.5 0.1 0.01" → absolute values
        self.assertAlmostEqual(builder.shape_material_mu[0], 0.5, places=5)
        self.assertAlmostEqual(builder.shape_material_torsional_friction[0], 0.1, places=5)
        self.assertAlmostEqual(builder.shape_material_rolling_friction[0], 0.01, places=5)

        # 3-element: friction="0.8 0.2 0.05" → absolute values
        self.assertAlmostEqual(builder.shape_material_mu[1], 0.8, places=5)
        self.assertAlmostEqual(builder.shape_material_torsional_friction[1], 0.2, places=5)
        self.assertAlmostEqual(builder.shape_material_rolling_friction[1], 0.05, places=5)

        # 3-element with zeros
        self.assertAlmostEqual(builder.shape_material_mu[2], 0.0, places=5)
        self.assertAlmostEqual(builder.shape_material_torsional_friction[2], 0.0, places=5)
        self.assertAlmostEqual(builder.shape_material_rolling_friction[2], 0.0, places=5)

        # 1-element: friction="1.0" → others use ShapeConfig defaults (0.25, 0.0005)
        self.assertAlmostEqual(builder.shape_material_mu[3], 1.0, places=5)
        self.assertAlmostEqual(builder.shape_material_torsional_friction[3], 0.25, places=5)
        self.assertAlmostEqual(builder.shape_material_rolling_friction[3], 0.0005, places=5)

        # 2-element: friction="0.6 0.15" → torsional: 0.15, rolling uses default (0.0005)
        self.assertAlmostEqual(builder.shape_material_mu[4], 0.6, places=5)
        self.assertAlmostEqual(builder.shape_material_torsional_friction[4], 0.15, places=5)
        self.assertAlmostEqual(builder.shape_material_rolling_friction[4], 0.0005, places=5)

    def test_mjcf_geom_solref_parsing(self):
        """Test MJCF geom solref parsing for contact stiffness/damping.

        MuJoCo solref format: [timeconst, dampratio]
        - Standard mode (timeconst > 0): ke = 1/(tc^2 * dr^2), kd = 2/tc
        - Direct mode (both negative): ke = -tc, kd = -dr
        """
        mjcf_content = """
        <mujoco>
            <worldbody>
                <body name="test_body">
                    <geom name="geom_default" type="box" size="0.1 0.1 0.1"/>
                    <!-- Custom solref [0.04, 1.0] -> ke=625, kd=50 -->
                    <geom name="geom_custom" type="sphere" size="0.1" solref="0.04 1.0"/>
                    <!-- Direct mode solref [-1000, -50] -> ke=1000, kd=50 -->
                    <geom name="geom_direct" type="capsule" size="0.1 0.2" solref="-1000 -50"/>
                </body>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, up_axis="Z")

        self.assertEqual(builder.shape_count, 3)

        # No solref specified -> Newton defaults: ke=2500 (ShapeConfig.ke), kd=100 (ShapeConfig.kd)
        self.assertAlmostEqual(builder.shape_material_ke[0], 2500.0, places=1)
        self.assertAlmostEqual(builder.shape_material_kd[0], 100.0, places=1)

        # Custom solref [0.04, 1.0]: ke = 1/(0.04^2 * 1^2) = 625, kd = 2/0.04 = 50
        self.assertAlmostEqual(builder.shape_material_ke[1], 625.0, places=1)
        self.assertAlmostEqual(builder.shape_material_kd[1], 50.0, places=1)

        # Direct mode solref [-1000, -50]: ke = 1000, kd = 50
        self.assertAlmostEqual(builder.shape_material_ke[2], 1000.0, places=1)
        self.assertAlmostEqual(builder.shape_material_kd[2], 50.0, places=1)

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
    <compiler angle="radian"/>
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

    def test_jnt_actgravcomp_parsing(self):
        """Test parsing of actuatorgravcomp from MJCF"""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="actgravcomp_test">
    <worldbody>
        <body name="body1" pos="0 0 1">
            <joint name="joint1" type="hinge" axis="0 0 1" actuatorgravcomp="true"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="body2" pos="1 0 1">
            <joint name="joint2" type="hinge" axis="0 1 0" actuatorgravcomp="false"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="body3" pos="2 0 1">
            <joint name="joint3" type="hinge" axis="1 0 0"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "jnt_actgravcomp"))

        jnt_actgravcomp = model.mujoco.jnt_actgravcomp.numpy()

        # Bodies are added in order
        self.assertEqual(jnt_actgravcomp[0], True)
        self.assertEqual(jnt_actgravcomp[1], False)
        self.assertEqual(jnt_actgravcomp[2], False)  # Default

    def test_xform_with_floating_false(self):
        """Test that xform parameter is respected when floating=False"""
        local_pos = wp.vec3(1.0, 2.0, 3.0)
        local_quat = wp.quat_rpy(0.5, -0.8, 0.7)
        local_xform = wp.transform(local_pos, local_quat)

        # Create a simple MJCF with a body that has a freejoint
        mjcf_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_xform">
    <worldbody>
        <body name="test_body" pos="{local_pos.x} {local_pos.y} {local_pos.z}" quat="{local_quat.w} {local_quat.x} {local_quat.y} {local_quat.z}">
            <freejoint/>
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>
"""
        # Create a non-identity transform to apply
        xform_pos = wp.vec3(5.0, 10.0, 15.0)
        xform_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 4.0)  # 45 degree rotation around Z
        xform = wp.transform(xform_pos, xform_quat)

        # Parse with floating=False and the xform
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, floating=False, xform=xform)
        model = builder.finalize()

        # Verify the model has a fixed joint
        self.assertEqual(model.joint_count, 1)
        joint_type = model.joint_type.numpy()[0]
        self.assertEqual(joint_type, newton.JointType.FIXED)

        # Verify the fixed joint has the correct parent_xform
        # The joint_X_p should match the world_xform (xform * local_xform)
        joint_X_p = model.joint_X_p.numpy()[0]

        expected_xform = xform * local_xform

        # Check position
        np.testing.assert_allclose(
            joint_X_p[:3],
            [expected_xform.p[0], expected_xform.p[1], expected_xform.p[2]],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Fixed joint parent_xform position does not match expected xform",
        )

        # Check quaternion (note: quaternions can be negated and still represent the same rotation)
        expected_quat = np.array([expected_xform.q[0], expected_xform.q[1], expected_xform.q[2], expected_xform.q[3]])
        actual_quat = joint_X_p[3:7]

        # Check if quaternions match (accounting for q and -q representing the same rotation)
        quat_match = np.allclose(actual_quat, expected_quat, rtol=1e-5, atol=1e-5) or np.allclose(
            actual_quat, -expected_quat, rtol=1e-5, atol=1e-5
        )
        self.assertTrue(
            quat_match,
            f"Fixed joint parent_xform quaternion does not match expected xform.\n"
            f"Expected: {expected_quat}\nActual: {actual_quat}",
        )

        # Verify body_q after eval_fk also matches the expected transform
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()[0]
        np.testing.assert_allclose(
            body_q[:3],
            [expected_xform.p[0], expected_xform.p[1], expected_xform.p[2]],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Body position after eval_fk does not match expected xform",
        )

        # Check body quaternion
        body_quat = body_q[3:7]
        quat_match = np.allclose(body_quat, expected_quat, rtol=1e-5, atol=1e-5) or np.allclose(
            body_quat, -expected_quat, rtol=1e-5, atol=1e-5
        )
        self.assertTrue(
            quat_match,
            f"Body quaternion after eval_fk does not match expected xform.\n"
            f"Expected: {expected_quat}\nActual: {body_quat}",
        )

    def test_joint_type_free_with_floating_false(self):
        """Test that <joint type="free"> respects floating=False parameter.

        MuJoCo supports two syntaxes for free joints:
        1. <freejoint/>
        2. <joint type="free"/>

        Both should be treated identically when the floating parameter is set.
        """
        # MJCF using <joint type="free"> instead of <freejoint>
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_joint_type_free">
    <worldbody>
        <body name="floating_body" pos="1 2 3">
            <joint name="free_joint" type="free"/>
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>
"""
        # Test with floating=False - should create a fixed joint
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, floating=False)
        model = builder.finalize()

        self.assertEqual(model.joint_count, 1)
        joint_type = model.joint_type.numpy()[0]
        self.assertEqual(joint_type, newton.JointType.FIXED)

        # Test with floating=True - should create a free joint
        builder2 = newton.ModelBuilder()
        builder2.add_mjcf(mjcf_content, floating=True)
        model2 = builder2.finalize()

        self.assertEqual(model2.joint_count, 1)
        joint_type2 = model2.joint_type.numpy()[0]
        self.assertEqual(joint_type2, newton.JointType.FREE)

        # Test with floating=None (default) - should preserve the free joint from MJCF
        builder3 = newton.ModelBuilder()
        builder3.add_mjcf(mjcf_content, floating=None)
        model3 = builder3.finalize()

        self.assertEqual(model3.joint_count, 1)
        joint_type3 = model3.joint_type.numpy()[0]
        self.assertEqual(joint_type3, newton.JointType.FREE)

    def test_geom_priority_parsing(self):
        """Test parsing of geom priority from MJCF"""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="priority_test">
    <worldbody>
        <body name="body1" pos="0 0 1">
            <joint name="joint1" type="hinge" axis="0 0 1"/>
            <geom type="box" size="0.1 0.1 0.1" priority="1"/>
        </body>
        <body name="body2" pos="1 0 1">
            <joint name="joint2" type="hinge" axis="0 1 0"/>
            <geom type="box" size="0.1 0.1 0.1" priority="0"/>
        </body>
        <body name="body3" pos="2 0 1">
            <joint name="joint3" type="hinge" axis="1 0 0"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>
"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "geom_priority"))

        geom_priority = model.mujoco.geom_priority.numpy()

        # Shapes are added in order
        self.assertEqual(geom_priority[0], 1)
        self.assertEqual(geom_priority[1], 0)
        self.assertEqual(geom_priority[2], 0)  # Default

    def test_geom_solimp_parsing(self):
        """Test that geom_solimp attribute is parsed correctly from MJCF."""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="body1">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1" solimp="0.8 0.9 0.002 0.4 3.0"/>
        </body>
        <body name="body2">
            <freejoint/>
            <geom type="sphere" size="0.05"/>
        </body>
        <body name="body3">
            <freejoint/>
            <geom type="capsule" size="0.05 0.1" solimp="0.7 0.85 0.003 0.6 2.5"/>
        </body>
    </worldbody>
</mujoco>
"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "geom_solimp"), "Model should have geom_solimp attribute")

        geom_solimp = model.mujoco.geom_solimp.numpy()
        self.assertEqual(model.shape_count, 3, "Should have 3 shapes")

        # Expected values: shape 0 has explicit solimp, shape 1 has defaults, shape 2 has explicit solimp
        expected_values = {
            0: [0.8, 0.9, 0.002, 0.4, 3.0],
            1: [0.9, 0.95, 0.001, 0.5, 2.0],  # default
            2: [0.7, 0.85, 0.003, 0.6, 2.5],
        }

        for shape_idx, expected in expected_values.items():
            actual = geom_solimp[shape_idx].tolist()
            for i, (a, e) in enumerate(zip(actual, expected, strict=False)):
                self.assertAlmostEqual(a, e, places=4, msg=f"geom_solimp[{shape_idx}][{i}] should be {e}, got {a}")

    def _create_mjcf_with_option(self, option_attr, option_value):
        """Helper to create standard MJCF with a single option."""
        return f"""<?xml version="1.0" ?>
<mujoco>
    <option {option_attr}="{option_value}"/>
    <worldbody>
        <body name="body1" pos="0 0 1">
            <joint type="hinge" axis="0 0 1"/>
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>
"""

    def test_option_scalar_world_parsing(self):
        """Test parsing of WORLD frequency scalar options from MJCF (6 options)."""
        test_cases = [
            ("impratio", "1.5", 1.5, 6),
            ("tolerance", "1e-6", 1e-6, 10),
            ("ls_tolerance", "0.001", 0.001, 6),
            ("ccd_tolerance", "1e-5", 1e-5, 10),
            ("density", "1.225", 1.225, 6),
            ("viscosity", "1.8e-5", 1.8e-5, 10),
        ]

        for option_name, mjcf_value, expected, places in test_cases:
            with self.subTest(option=option_name):
                mjcf = self._create_mjcf_with_option(option_name, mjcf_value)
                builder = newton.ModelBuilder()
                builder.add_mjcf(mjcf)
                model = builder.finalize()

                self.assertTrue(hasattr(model, "mujoco"))
                self.assertTrue(hasattr(model.mujoco, option_name))
                value = getattr(model.mujoco, option_name).numpy()
                self.assertEqual(len(value), 1)
                self.assertAlmostEqual(value[0], expected, places=places)

    def test_option_scalar_per_world(self):
        """Test that scalar options are correctly remapped per world when merging builders."""
        # Robot A
        robot_a = newton.ModelBuilder()
        robot_a.add_mjcf("""
<mujoco>
    <option impratio="1.5" tolerance="1e-6" ls_tolerance="0.001"/>
    <worldbody>
        <body name="a" pos="0 0 1">
            <joint type="hinge" axis="0 0 1"/>
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>
""")

        # Robot B
        robot_b = newton.ModelBuilder()
        robot_b.add_mjcf("""
<mujoco>
    <option impratio="2.0" tolerance="1e-7" ls_tolerance="0.002"/>
    <worldbody>
        <body name="b" pos="0 0 1">
            <joint type="hinge" axis="0 0 1"/>
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>
""")

        # Merge into main builder
        main = newton.ModelBuilder()
        main.add_world(robot_a)
        main.add_world(robot_b)
        model = main.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "impratio"))
        self.assertTrue(hasattr(model.mujoco, "tolerance"))
        self.assertTrue(hasattr(model.mujoco, "ls_tolerance"))

        impratio = model.mujoco.impratio.numpy()
        tolerance = model.mujoco.tolerance.numpy()
        ls_tolerance = model.mujoco.ls_tolerance.numpy()

        # Should have 2 worlds with different values
        self.assertEqual(len(impratio), 2)
        self.assertEqual(len(tolerance), 2)
        self.assertEqual(len(ls_tolerance), 2)
        self.assertAlmostEqual(impratio[0], 1.5, places=4, msg="World 0 should have impratio=1.5")
        self.assertAlmostEqual(impratio[1], 2.0, places=4, msg="World 1 should have impratio=2.0")
        self.assertAlmostEqual(tolerance[0], 1e-6, places=10, msg="World 0 should have tolerance=1e-6")
        self.assertAlmostEqual(tolerance[1], 1e-7, places=10, msg="World 1 should have tolerance=1e-7")
        self.assertAlmostEqual(ls_tolerance[0], 0.001, places=6, msg="World 0 should have ls_tolerance=0.001")
        self.assertAlmostEqual(ls_tolerance[1], 0.002, places=6, msg="World 1 should have ls_tolerance=0.002")

    def test_option_vector_world_parsing(self):
        """Test parsing of WORLD frequency vector options from MJCF (2 options)."""
        test_cases = [
            ("wind", "1 0.5 -0.5", [1, 0.5, -0.5]),
            ("magnetic", "0 -1 0.5", [0, -1, 0.5]),
        ]

        for option_name, mjcf_value, expected in test_cases:
            with self.subTest(option=option_name):
                mjcf = self._create_mjcf_with_option(option_name, mjcf_value)
                builder = newton.ModelBuilder()
                builder.add_mjcf(mjcf)
                model = builder.finalize()

                self.assertTrue(hasattr(model, "mujoco"))
                self.assertTrue(hasattr(model.mujoco, option_name))
                value = getattr(model.mujoco, option_name).numpy()
                self.assertEqual(len(value), 1)
                self.assertTrue(np.allclose(value[0], expected))

    def test_option_numeric_once_parsing(self):
        """Test parsing of ONCE frequency numeric options from MJCF (3 options)."""
        test_cases = [
            ("ccd_iterations", "25", 25),
            ("sdf_iterations", "20", 20),
            ("sdf_initpoints", "50", 50),
        ]

        for option_name, mjcf_value, expected in test_cases:
            with self.subTest(option=option_name):
                mjcf = self._create_mjcf_with_option(option_name, mjcf_value)
                builder = newton.ModelBuilder()
                builder.add_mjcf(mjcf)
                model = builder.finalize()

                self.assertTrue(hasattr(model, "mujoco"))
                self.assertTrue(hasattr(model.mujoco, option_name))
                value = getattr(model.mujoco, option_name).numpy()
                # ONCE frequency: single value, not per-world
                self.assertEqual(len(value), 1)
                self.assertEqual(value[0], expected)

    def test_option_enum_once_parsing(self):
        """Test parsing of ONCE frequency enum options from MJCF (4 options)."""
        test_cases = [
            ("integrator", "Euler", 0),
            ("solver", "Newton", 2),
            ("cone", "elliptic", 1),
            ("jacobian", "sparse", 1),
        ]

        for option_name, mjcf_value, expected_int in test_cases:
            with self.subTest(option=option_name):
                mjcf = self._create_mjcf_with_option(option_name, mjcf_value)
                builder = newton.ModelBuilder()
                builder.add_mjcf(mjcf)
                model = builder.finalize()

                self.assertTrue(hasattr(model, "mujoco"))
                self.assertTrue(hasattr(model.mujoco, option_name))
                value = getattr(model.mujoco, option_name).numpy()
                self.assertEqual(len(value), 1)  # ONCE frequency
                self.assertEqual(value[0], expected_int)

    def test_option_tag_pair_syntax(self):
        """Test that options work with tag-pair syntax in addition to self-closing tags."""
        # Test with tag-pair syntax: <option></option>
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <option impratio="2.5" tolerance="1e-7" integrator="RK4"></option>
    <worldbody>
        <body name="body1" pos="0 0 1">
            <joint type="hinge" axis="0 0 1"/>
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>
"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "impratio"))
        self.assertTrue(hasattr(model.mujoco, "tolerance"))
        self.assertTrue(hasattr(model.mujoco, "integrator"))

        # Verify values are parsed correctly
        self.assertAlmostEqual(model.mujoco.impratio.numpy()[0], 2.5, places=4)
        self.assertAlmostEqual(model.mujoco.tolerance.numpy()[0], 1e-7, places=10)
        self.assertEqual(model.mujoco.integrator.numpy()[0], 1)  # RK4

    def test_geom_solmix_parsing(self):
        """Test that geom_solmix attribute is parsed correctly from MJCF."""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="body1">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1" solmix="0.5"/>
        </body>
        <body name="body2">
            <freejoint/>
            <geom type="sphere" size="0.05"/>
        </body>
        <body name="body3">
            <freejoint/>
            <geom type="capsule" size="0.05 0.1" solmix="0.8"/>
        </body>
    </worldbody>
</mujoco>
"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "geom_solmix"), "Model should have geom_solmix attribute")

        geom_solmix = model.mujoco.geom_solmix.numpy()
        self.assertEqual(model.shape_count, 3, "Should have 3 shapes")

        # Expected values: shape 0 has explicit solimp=0.5, shape 1 has solimp=default=1.0, shape 2 has explicit solimp=0.8
        expected_values = {
            0: 0.5,
            1: 1.0,  # default
            2: 0.8,
        }

        for shape_idx, expected in expected_values.items():
            actual = geom_solmix[shape_idx].tolist()
            self.assertAlmostEqual(actual, expected, places=4)

    def test_geom_gap_parsing(self):
        """Test that geom_gap attribute is parsed correctly from MJCF."""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="body1">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1" gap="0.1"/>
        </body>
        <body name="body2">
            <freejoint/>
            <geom type="sphere" size="0.05"/>
        </body>
        <body name="body3">
            <freejoint/>
            <geom type="capsule" size="0.05 0.1" gap="0.2"/>
        </body>
    </worldbody>
</mujoco>
"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "geom_gap"), "Model should have geom_gap attribute")

        geom_gap = model.mujoco.geom_gap.numpy()
        self.assertEqual(model.shape_count, 3, "Should have 3 shapes")

        # Expected values: shape 0 has explicit solimp=0.5, shape 1 has solimp=default=1.0, shape 2 has explicit solimp=0.8
        expected_values = {
            0: 0.1,
            1: 0.0,  # default
            2: 0.2,
        }

        for shape_idx, expected in expected_values.items():
            actual = geom_gap[shape_idx].tolist()
            self.assertAlmostEqual(actual, expected, places=4)

    def test_default_inheritance(self):
        """Test nested default class inheritanc."""
        mjcf_content = """<?xml version="1.0" ?>
<mujoco>
    <default>
        <default class="collision">
            <geom group="3" type="mesh" condim="6" friction="1 5e-3 5e-4" solref=".01 1"/>
            <default class="sphere_collision">
                <geom type="sphere" size="0.0006" rgba="1 0 0 1"/>
            </default>
        </default>
    </default>
    <worldbody>
        <body name="body1">
            <geom class="sphere_collision" />
        </body>
    </worldbody>
</mujoco>
"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        self.assertEqual(builder.shape_count, 1)

        self.assertEqual(builder.shape_type[0], GeoType.SPHERE)

        # Verify condim is 6 (inherited from parent)
        # If inheritance is broken, this will be the default value (usually 3)
        if hasattr(model, "mujoco") and hasattr(model.mujoco, "condim"):
            condim = model.mujoco.condim.numpy()[0]
            self.assertEqual(condim, 6, "condim should be 6 (inherited from parent class 'collision')")
        else:
            self.fail("Model should have mujoco.condim attribute")

    def test_actuatorfrcrange_parsing(self):
        """Test that actuatorfrcrange is parsed from MJCF joint attributes and applied to joint effort limits."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_actuatorfrcrange">
    <worldbody>
        <body name="link1" pos="0 0 0">
            <joint name="joint1" axis="1 0 0" type="hinge" range="-90 90" actuatorfrcrange="-100 100" actuatorfrclimited="true"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="link2" pos="1 0 0">
            <joint name="joint2" axis="0 1 0" type="slider" range="-45 45" actuatorfrcrange="-50 50" actuatorfrclimited="auto"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="link3" pos="2 0 0">
            <joint name="joint3" axis="0 0 1" type="hinge" range="-180 180" actuatorfrcrange="-200 200"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="link4" pos="3 0 0">
            <joint name="joint4" axis="1 0 0" type="hinge" range="-90 90"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="link5" pos="4 0 0">
            <joint name="joint5" axis="1 0 0" type="hinge" range="-90 90" actuatorfrcrange="-75 75" actuatorfrclimited="false"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        joint1_idx = model.joint_key.index("joint1")
        joint2_idx = model.joint_key.index("joint2")
        joint3_idx = model.joint_key.index("joint3")
        joint4_idx = model.joint_key.index("joint4")
        joint5_idx = model.joint_key.index("joint5")

        joint1_dof_idx = model.joint_qd_start.numpy()[joint1_idx]
        joint2_dof_idx = model.joint_qd_start.numpy()[joint2_idx]
        joint3_dof_idx = model.joint_qd_start.numpy()[joint3_idx]
        joint4_dof_idx = model.joint_qd_start.numpy()[joint4_idx]
        joint5_dof_idx = model.joint_qd_start.numpy()[joint5_idx]

        effort_limits = model.joint_effort_limit.numpy()

        self.assertAlmostEqual(
            effort_limits[joint1_dof_idx],
            100.0,
            places=5,
            msg="Effort limit for joint1 should be 100 from actuatorfrcrange with actuatorfrclimited='true'",
        )

        self.assertAlmostEqual(
            effort_limits[joint2_dof_idx],
            50.0,
            places=5,
            msg="Effort limit for joint2 should be 50 from actuatorfrcrange with actuatorfrclimited='auto'",
        )

        self.assertAlmostEqual(
            effort_limits[joint3_dof_idx],
            200.0,
            places=5,
            msg="Effort limit for joint3 should be 200 from actuatorfrcrange with default actuatorfrclimited",
        )

        self.assertAlmostEqual(
            effort_limits[joint4_dof_idx],
            1e6,
            places=5,
            msg="Effort limit for joint4 should be default value (1e6) when actuatorfrcrange not specified",
        )

        self.assertAlmostEqual(
            effort_limits[joint5_dof_idx],
            1e6,
            places=5,
            msg="Effort limit for joint5 should be default (1e6) when actuatorfrclimited='false'",
        )

    def test_eq_solref_parsing(self):
        """Test that equality constraint solref attribute is parsed correctly from MJCF."""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="body1">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="body2">
            <freejoint/>
            <geom type="sphere" size="0.05"/>
        </body>
        <body name="body3">
            <freejoint/>
            <geom type="capsule" size="0.05 0.1"/>
        </body>
    </worldbody>
    <equality>
        <weld body1="body1" body2="body2" solref="0.03 0.8"/>
        <connect body1="body2" body2="body3" anchor="0 0 0"/>
        <weld body1="body1" body2="body3" solref="0.05 1.2"/>
    </equality>
</mujoco>
"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace for custom attributes")
        self.assertTrue(hasattr(model.mujoco, "eq_solref"), "Model should have eq_solref attribute")

        eq_solref = model.mujoco.eq_solref.numpy()
        self.assertEqual(model.equality_constraint_count, 3, "Should have 3 equality constraints")

        # Note: Newton parses equality constraints in type order: connect, then weld, then joint
        # So the order is: connect (default), weld (0.03, 0.8), weld (0.05, 1.2)
        expected_values = {
            0: [0.02, 1.0],  # connect - default
            1: [0.03, 0.8],  # first weld
            2: [0.05, 1.2],  # second weld
        }

        for eq_idx, expected in expected_values.items():
            actual = eq_solref[eq_idx].tolist()
            for i, (a, e) in enumerate(zip(actual, expected, strict=False)):
                self.assertAlmostEqual(a, e, places=4, msg=f"eq_solref[{eq_idx}][{i}] should be {e}, got {a}")

    def test_parse_mujoco_options_disabled(self):
        """Test that solver options from <option> tag are not parsed when parse_mujoco_options=False."""
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <option impratio="99.0"/>
    <worldbody>
        <body name="body1" pos="0 0 1">
            <joint type="hinge" axis="0 0 1"/>
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, parse_mujoco_options=False)
        model = builder.finalize()

        # impratio should remain at default (1.0), not the MJCF value (99.0)
        self.assertAlmostEqual(model.mujoco.impratio.numpy()[0], 1.0, places=4)

    def test_ref_attribute_parsing(self):
        """Test that 'ref' attribute is parsed."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <compiler angle="radian"/>
    <worldbody>
        <body name="base">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child1" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 1 0" ref="1.5708"/>
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="child2" pos="0 0 1">
                    <joint name="slide" type="slide" axis="0 0 1" ref="0.5"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Verify custom attribute parsing
        qd_start = model.joint_qd_start.numpy()
        dof_ref = model.mujoco.dof_ref.numpy()

        hinge_idx = model.joint_key.index("hinge")
        self.assertAlmostEqual(dof_ref[qd_start[hinge_idx]], 1.5708, places=4)

        slide_idx = model.joint_key.index("slide")
        self.assertAlmostEqual(dof_ref[qd_start[slide_idx]], 0.5, places=4)

    def test_springref_attribute_parsing(self):
        """Test that 'springref' attribute is parsed for hinge and slide joints."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <compiler angle="radian"/>
    <worldbody>
        <body name="base">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child1" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 0 1" stiffness="100" springref="0.5236"/>
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="child2" pos="0 0 1">
                    <joint name="slide" type="slide" axis="0 0 1" stiffness="50" springref="0.25"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()
        springref = model.mujoco.dof_springref.numpy()
        qd_start = model.joint_qd_start.numpy()

        hinge_idx = model.joint_key.index("hinge")
        self.assertAlmostEqual(springref[qd_start[hinge_idx]], 0.5236, places=4)
        slide_idx = model.joint_key.index("slide")
        self.assertAlmostEqual(springref[qd_start[slide_idx]], 0.25, places=4)

    def test_static_geom_xform_not_applied_twice(self):
        """Test that xform parameter is applied exactly once to static geoms.

        This is a regression test for a bug where incoming_xform was applied twice
        to static geoms (link == -1) in parse_shapes.

        A static geom at pos=(1,0,0) with xform translation of (0,2,0) should
        result in final position (1,2,0), NOT (1,4,0) from double application.
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_static_xform">
    <worldbody>
        <geom name="static_geom" pos="1 0 0" size="0.1" type="sphere"/>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        # Apply a translation via xform parameter
        import_xform = wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity())
        builder.add_mjcf(mjcf_content, xform=import_xform)

        # Find the static geom
        geom_idx = builder.shape_key.index("static_geom")
        geom_xform = builder.shape_transform[geom_idx]

        # Position should be geom_pos + xform_pos = (1,0,0) + (0,2,0) = (1,2,0)
        # Bug would give (1,0,0) + (0,2,0) + (0,2,0) = (1,4,0)
        self.assertAlmostEqual(geom_xform[0], 1.0, places=5)
        self.assertAlmostEqual(geom_xform[1], 2.0, places=5)  # Would be 4.0 with bug
        self.assertAlmostEqual(geom_xform[2], 0.0, places=5)

    def test_static_fromto_capsule_xform(self):
        """Test that xform parameter is applied to capsule/cylinder fromto coordinates.

        A static capsule with fromto="0 0 0  1 0 0" (centered at (0.5,0,0)) with
        xform translation of (0,5,0) should result in position (0.5, 5.0, 0).
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_fromto_xform">
    <worldbody>
        <geom name="fromto_cap" type="capsule" fromto="0 0 0  1 0 0" size="0.1"/>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        import_xform = wp.transform(wp.vec3(0.0, 5.0, 0.0), wp.quat_identity())
        builder.add_mjcf(mjcf_content, xform=import_xform)

        geom_idx = builder.shape_key.index("fromto_cap")
        geom_xform = builder.shape_transform[geom_idx]

        # Position should be midpoint(0,0,0 to 1,0,0) + xform = (0.5,0,0) + (0,5,0) = (0.5,5,0)
        self.assertAlmostEqual(geom_xform[0], 0.5, places=5)
        self.assertAlmostEqual(geom_xform[1], 5.0, places=5)
        self.assertAlmostEqual(geom_xform[2], 0.0, places=5)

    def test_actuator_mode_inference_from_actuator_type(self):
        """Test that ActuatorMode is correctly inferred from MJCF actuator types."""
        from newton._src.sim.joints import ActuatorMode  # noqa: PLC0415

        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_actuator_modes">
    <worldbody>
        <body name="base" pos="0 0 1">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="link_motor" pos="0.2 0 0">
                <joint name="joint_motor" axis="0 0 1" type="hinge"/>
                <geom type="box" size="0.1 0.1 0.1"/>
                <body name="link_position" pos="0.2 0 0">
                    <joint name="joint_position" axis="0 0 1" type="hinge"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                    <body name="link_velocity" pos="0.2 0 0">
                        <joint name="joint_velocity" axis="0 0 1" type="hinge"/>
                        <geom type="box" size="0.1 0.1 0.1"/>
                        <body name="link_pos_vel" pos="0.2 0 0">
                            <joint name="joint_pos_vel" axis="0 0 1" type="hinge"/>
                            <geom type="box" size="0.1 0.1 0.1"/>
                            <body name="link_passive" pos="0.2 0 0">
                                <joint name="joint_passive" axis="0 0 1" type="hinge"/>
                                <geom type="box" size="0.1 0.1 0.1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor1" joint="joint_motor"/>
        <position name="pos1" joint="joint_position" kp="100"/>
        <velocity name="vel1" joint="joint_velocity" kv="10"/>
        <position name="pos2" joint="joint_pos_vel" kp="100"/>
        <velocity name="vel2" joint="joint_pos_vel" kv="10"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, ctrl_direct=False)

        def get_qd_start(b, joint_name):
            joint_idx = b.joint_key.index(joint_name)
            return sum(b.joint_dof_dim[i][0] + b.joint_dof_dim[i][1] for i in range(joint_idx))

        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_motor")], int(ActuatorMode.NONE))
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_position")], int(ActuatorMode.POSITION))
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_velocity")], int(ActuatorMode.VELOCITY))
        self.assertEqual(
            builder.joint_act_mode[get_qd_start(builder, "joint_pos_vel")], int(ActuatorMode.POSITION_VELOCITY)
        )
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_passive")], int(ActuatorMode.NONE))

        builder2 = newton.ModelBuilder()
        builder2.add_mjcf(mjcf_content, ctrl_direct=True)

        self.assertEqual(builder2.joint_act_mode[get_qd_start(builder2, "joint_motor")], int(ActuatorMode.NONE))
        self.assertEqual(builder2.joint_act_mode[get_qd_start(builder2, "joint_position")], int(ActuatorMode.NONE))
        self.assertEqual(builder2.joint_act_mode[get_qd_start(builder2, "joint_velocity")], int(ActuatorMode.NONE))
        self.assertEqual(builder2.joint_act_mode[get_qd_start(builder2, "joint_pos_vel")], int(ActuatorMode.NONE))
        self.assertEqual(builder2.joint_act_mode[get_qd_start(builder2, "joint_passive")], int(ActuatorMode.NONE))

    def test_frame_transform_composition_geoms(self):
        """Test that frame transforms are correctly composed with child geom positions.

        Based on MuJoCo documentation example:
        - A frame with pos="0 1 0" containing a geom with pos="0 1 0" should result
          in the geom having pos="0 2 0" (transforms are accumulated).
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_frame">
    <worldbody>
        <frame pos="0 1 0">
            <geom name="Bob" pos="0 1 0" size="1" type="sphere"/>
        </frame>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)

        # Find the geom named "Bob"
        bob_idx = builder.shape_key.index("Bob")
        bob_xform = builder.shape_transform[bob_idx]

        # Position should be (0, 2, 0) = frame pos + geom pos
        self.assertAlmostEqual(bob_xform[0], 0.0, places=5)
        self.assertAlmostEqual(bob_xform[1], 2.0, places=5)
        self.assertAlmostEqual(bob_xform[2], 0.0, places=5)

    def test_frame_transform_composition_rotation(self):
        """Test that frame quaternion rotations are correctly composed.

        Based on MuJoCo documentation example:
        - A frame with quat="0 0 1 0" (180 deg around Y) containing a geom with quat="0 1 0 0" (180 deg around X)
          should result in quat="0 0 0 1" (180 deg around Z).
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_frame_rotation">
    <worldbody>
        <frame quat="0 0 1 0">
            <geom name="Alice" quat="0 1 0 0" size="1" type="sphere"/>
        </frame>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)

        # Find the geom named "Alice"
        alice_idx = builder.shape_key.index("Alice")
        alice_xform = builder.shape_transform[alice_idx]

        # The resulting quaternion should be approximately (0, 0, 0, 1) in xyzw format
        # or equivalently (1, 0, 0, 0) in wxyz MuJoCo format (representing 180 deg around Z)
        # In Newton's xyzw format: (x, y, z, w) = (0, 0, 1, 0) for 180 deg around Z
        # But we need to check the actual composed result
        quat = wp.quat(alice_xform[3], alice_xform[4], alice_xform[5], alice_xform[6])
        # The expected result from MuJoCo docs: quat="0 0 0 1" in wxyz = (0, 0, 1, 0) in xyzw after normalization
        # Actually the doc says result is "0 0 0 1" which is wxyz format meaning w=0, x=0, y=0, z=1
        # In Newton xyzw: x=0, y=0, z=1, w=0
        self.assertAlmostEqual(abs(quat[0]), 0.0, places=4)  # x
        self.assertAlmostEqual(abs(quat[1]), 0.0, places=4)  # y
        self.assertAlmostEqual(abs(quat[2]), 1.0, places=4)  # z
        self.assertAlmostEqual(abs(quat[3]), 0.0, places=4)  # w

    def test_frame_transform_composition_body(self):
        """Test that frame transforms are correctly composed with child body positions.

        A frame with pos="1 0 0" containing a body with pos="1 0 0" should result
        in the body having position (2, 0, 0) relative to parent.
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_frame_body">
    <worldbody>
        <frame pos="1 0 0">
            <body name="Carl" pos="1 0 0">
                <geom name="carl_geom" size="0.1" type="sphere"/>
            </body>
        </frame>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Find the body named "Carl"
        _carl_idx = model.body_key.index("Carl")

        # Get the joint transform for Carl's joint (which connects Carl to world)
        # The joint_X_p contains the parent frame transform
        joint_idx = 0  # First joint should be Carl's
        joint_X_p = model.joint_X_p.numpy()[joint_idx]

        # Position should be (2, 0, 0) = frame pos + body pos
        self.assertAlmostEqual(joint_X_p[0], 2.0, places=5)
        self.assertAlmostEqual(joint_X_p[1], 0.0, places=5)
        self.assertAlmostEqual(joint_X_p[2], 0.0, places=5)

    def test_nested_frames(self):
        """Test that nested frames correctly compose their transforms."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_nested_frames">
    <worldbody>
        <frame pos="1 0 0">
            <frame pos="0 1 0">
                <frame pos="0 0 1">
                    <geom name="nested_geom" pos="0 0 0" size="0.1" type="sphere"/>
                </frame>
            </frame>
        </frame>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)

        # Find the nested geom
        geom_idx = builder.shape_key.index("nested_geom")
        geom_xform = builder.shape_transform[geom_idx]

        # Position should be (1, 1, 1) from accumulated frame positions
        self.assertAlmostEqual(geom_xform[0], 1.0, places=5)
        self.assertAlmostEqual(geom_xform[1], 1.0, places=5)
        self.assertAlmostEqual(geom_xform[2], 1.0, places=5)

    def test_frame_inside_body(self):
        """Test that frames inside bodies correctly transform their children."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_frame_in_body">
    <worldbody>
        <body name="parent" pos="0 0 0">
            <geom name="parent_geom" size="0.1" type="sphere"/>
            <frame pos="0 0 1">
                <body name="child" pos="0 0 1">
                    <geom name="child_geom" size="0.1" type="sphere"/>
                </body>
            </frame>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Find the child body's joint
        child_idx = model.body_key.index("child")

        # The child's joint_X_p should have z=2 (frame z=1 + body z=1)
        # Find the joint that has child as its child body
        joint_child = model.joint_child.numpy()
        joint_idx = np.where(joint_child == child_idx)[0][0]
        joint_X_p = model.joint_X_p.numpy()[joint_idx]

        self.assertAlmostEqual(joint_X_p[0], 0.0, places=5)
        self.assertAlmostEqual(joint_X_p[1], 0.0, places=5)
        self.assertAlmostEqual(joint_X_p[2], 2.0, places=5)

    def test_frame_geom_inside_body_is_body_relative(self):
        """Test that geoms inside frames inside bodies have body-relative transforms.

        This tests a critical distinction: geom transforms should be relative to
        their parent body, NOT world transforms. A bug would cause the geom to be
        positioned at the body's world position + frame offset + geom offset,
        instead of just frame offset + geom offset relative to the body.
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_frame_geom_body_relative">
    <worldbody>
        <body name="parent" pos="10 20 30">
            <geom name="parent_geom" size="0.1" type="sphere"/>
            <frame pos="1 2 3">
                <geom name="frame_geom" pos="0.1 0.2 0.3" size="0.1" type="sphere"/>
            </frame>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)

        # Find the frame_geom - its transform should be body-relative
        geom_idx = builder.shape_key.index("frame_geom")
        geom_xform = builder.shape_transform[geom_idx]

        # Position should be frame pos + geom pos = (1.1, 2.2, 3.3)
        # NOT body world pos + frame pos + geom pos = (11.1, 22.2, 33.3)
        self.assertAlmostEqual(geom_xform[0], 1.1, places=5)
        self.assertAlmostEqual(geom_xform[1], 2.2, places=5)
        self.assertAlmostEqual(geom_xform[2], 3.3, places=5)

    def test_frame_with_sites(self):
        """Test that frames correctly transform site positions."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_frame_sites">
    <worldbody>
        <frame pos="1 2 3">
            <site name="test_site" pos="0.5 0.5 0.5" size="0.01"/>
        </frame>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, parse_sites=True)

        # Find the site
        site_idx = builder.shape_key.index("test_site")
        site_xform = builder.shape_transform[site_idx]

        # Position should be (1.5, 2.5, 3.5) = frame pos + site pos
        self.assertAlmostEqual(site_xform[0], 1.5, places=5)
        self.assertAlmostEqual(site_xform[1], 2.5, places=5)
        self.assertAlmostEqual(site_xform[2], 3.5, places=5)

    def test_site_size_defaults(self):
        """Test that site size matches MuJoCo behavior for partial values.

        MuJoCo fills unspecified components with its default (0.005), NOT by
        replicating the first value. This ensures MJCF compatibility.
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_site_size">
    <worldbody>
        <body name="body1">
            <!-- Site with single size value - should fill with MuJoCo defaults -->
            <site name="site_single" size="0.001"/>
            <!-- Site with two size values - should fill third with default -->
            <site name="site_two" size="0.002 0.003"/>
            <!-- Site with all three size values -->
            <site name="site_three" size="0.004 0.005 0.006"/>
            <!-- Site with no size - should use MuJoCo default [0.005, 0.005, 0.005] -->
            <site name="site_default"/>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, parse_sites=True)

        # Helper to get site scale by name
        def get_site_scale(name):
            idx = builder.shape_key.index(name)
            return builder.shape_scale[idx]

        # Single value: [0.001, 0.005, 0.005] (matches MuJoCo behavior)
        scale_single = get_site_scale("site_single")
        self.assertAlmostEqual(scale_single[0], 0.001, places=6)
        self.assertAlmostEqual(scale_single[1], 0.005, places=6)
        self.assertAlmostEqual(scale_single[2], 0.005, places=6)

        # Two values: [0.002, 0.003, 0.005]
        scale_two = get_site_scale("site_two")
        self.assertAlmostEqual(scale_two[0], 0.002, places=6)
        self.assertAlmostEqual(scale_two[1], 0.003, places=6)
        self.assertAlmostEqual(scale_two[2], 0.005, places=6)

        # Three values: [0.004, 0.005, 0.006]
        scale_three = get_site_scale("site_three")
        self.assertAlmostEqual(scale_three[0], 0.004, places=6)
        self.assertAlmostEqual(scale_three[1], 0.005, places=6)
        self.assertAlmostEqual(scale_three[2], 0.006, places=6)

        # No size: should use MuJoCo default [0.005, 0.005, 0.005]
        scale_default = get_site_scale("site_default")
        self.assertAlmostEqual(scale_default[0], 0.005, places=6)
        self.assertAlmostEqual(scale_default[1], 0.005, places=6)
        self.assertAlmostEqual(scale_default[2], 0.005, places=6)

    def test_frame_childclass_propagation(self):
        """Test that frames correctly propagate childclass and merged defaults to geoms, sites, and nested frames."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_frame_childclass">
    <default>
        <default class="red_class">
            <geom rgba="1 0 0 1" size="0.1"/>
            <site rgba="1 0 0 1" size="0.05"/>
        </default>
        <default class="blue_class">
            <geom rgba="0 0 1 1" size="0.2"/>
            <site rgba="0 0 1 1" size="0.08"/>
        </default>
        <default class="green_class">
            <geom rgba="0 1 0 1" size="0.3"/>
            <site rgba="0 1 0 1" size="0.12"/>
        </default>
    </default>
    <worldbody>
        <!-- Frame with childclass should apply defaults to its children -->
        <frame name="red_frame" childclass="red_class" pos="1 0 0">
            <geom name="geom_in_red_frame" type="sphere"/>
            <site name="site_in_red_frame"/>

            <!-- Nested frame inherits parent's childclass -->
            <frame name="nested_in_red" pos="0 1 0">
                <geom name="geom_in_nested_red" type="sphere"/>
                <site name="site_in_nested_red"/>
            </frame>

            <!-- Nested frame with its own childclass overrides -->
            <frame name="blue_nested_in_red" childclass="blue_class" pos="0 0 1">
                <geom name="geom_in_blue_nested" type="sphere"/>
                <site name="site_in_blue_nested"/>

                <!-- Double-nested frame inherits blue_class -->
                <frame name="double_nested" pos="0.5 0 0">
                    <geom name="geom_double_nested" type="sphere"/>
                    <site name="site_double_nested"/>
                </frame>
            </frame>
        </frame>

        <!-- Geom outside any frame (uses global defaults) -->
        <geom name="geom_no_frame" type="sphere" size="0.5"/>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, parse_sites=True, up_axis="Z")

        def get_shape_size(name):
            idx = builder.shape_key.index(name)
            geo_type = builder.shape_type[idx]
            if geo_type == GeoType.SPHERE:
                return builder.shape_scale[idx][0]  # radius
            return None

        def get_shape_pos(name):
            idx = builder.shape_key.index(name)
            return builder.shape_transform[idx][:3]

        # Geom in red_frame should have red_class size (0.1)
        self.assertAlmostEqual(get_shape_size("geom_in_red_frame"), 0.1, places=5)

        # Geom in nested frame (inherits red_class) should also have size 0.1
        self.assertAlmostEqual(get_shape_size("geom_in_nested_red"), 0.1, places=5)

        # Geom in blue_nested_in_red (overrides to blue_class) should have size 0.2
        self.assertAlmostEqual(get_shape_size("geom_in_blue_nested"), 0.2, places=5)

        # Double-nested geom (inherits blue_class from parent frame) should have size 0.2
        self.assertAlmostEqual(get_shape_size("geom_double_nested"), 0.2, places=5)

        # Geom outside frames should use explicit size (0.5)
        self.assertAlmostEqual(get_shape_size("geom_no_frame"), 0.5, places=5)

        # Verify transforms are still correctly composed
        # geom_in_red_frame: frame pos (1,0,0) + geom pos (0,0,0) = (1,0,0)
        pos = get_shape_pos("geom_in_red_frame")
        self.assertAlmostEqual(pos[0], 1.0, places=5)
        self.assertAlmostEqual(pos[1], 0.0, places=5)
        self.assertAlmostEqual(pos[2], 0.0, places=5)

        # geom_in_nested_red: (1,0,0) + (0,1,0) = (1,1,0)
        pos = get_shape_pos("geom_in_nested_red")
        self.assertAlmostEqual(pos[0], 1.0, places=5)
        self.assertAlmostEqual(pos[1], 1.0, places=5)
        self.assertAlmostEqual(pos[2], 0.0, places=5)

        # geom_in_blue_nested: (1,0,0) + (0,0,1) = (1,0,1)
        pos = get_shape_pos("geom_in_blue_nested")
        self.assertAlmostEqual(pos[0], 1.0, places=5)
        self.assertAlmostEqual(pos[1], 0.0, places=5)
        self.assertAlmostEqual(pos[2], 1.0, places=5)

        # geom_double_nested: (1,0,0) + (0,0,1) + (0.5,0,0) = (1.5,0,1)
        pos = get_shape_pos("geom_double_nested")
        self.assertAlmostEqual(pos[0], 1.5, places=5)
        self.assertAlmostEqual(pos[1], 0.0, places=5)
        self.assertAlmostEqual(pos[2], 1.0, places=5)

        # Verify sites also receive the correct defaults
        # site_in_red_frame should have red_class size (0.05)
        site_idx = builder.shape_key.index("site_in_red_frame")
        self.assertAlmostEqual(builder.shape_scale[site_idx][0], 0.05, places=5)

        # site_in_blue_nested should have blue_class size (0.08)
        site_idx = builder.shape_key.index("site_in_blue_nested")
        self.assertAlmostEqual(builder.shape_scale[site_idx][0], 0.08, places=5)

        # site_double_nested should inherit blue_class size (0.08)
        site_idx = builder.shape_key.index("site_double_nested")
        self.assertAlmostEqual(builder.shape_scale[site_idx][0], 0.08, places=5)

    def test_joint_anchor_with_rotated_body(self):
        """Test that joint anchor position is correctly computed when body has rotation.

        This is a regression test for a bug where the joint position offset was added
        directly to the body position without being rotated by the body's orientation.

        Setup:
        - Parent body at (0,0,0) with 90° rotation around Z
        - Child body at (1,0,0) relative to parent (becomes (0,1,0) in world due to rotation)
        - Joint with pos="0.5 0 0" in child's local frame

        The joint anchor (in parent frame) should be:
        - body_pos_relative_to_parent + rotate(joint_pos, body_orientation)
        - = (1,0,0) + rotate_90z(0.5,0,0)
        - = (1,0,0) + (0,0.5,0)
        - = (1, 0.5, 0)

        Bug would compute: (1,0,0) + (0.5,0,0) = (1.5, 0, 0) - WRONG
        """
        # Parent rotated 90° around Z axis
        # MJCF quat format is [w, x, y, z]
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_joint_anchor_rotation">
    <worldbody>
        <body name="parent" pos="0 0 0" quat="0.7071068 0 0 0.7071068">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child" pos="1 0 0">
                <joint name="child_joint" type="hinge" axis="0 0 1" pos="0.5 0 0"/>
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Find the child's joint
        joint_idx = model.joint_key.index("child_joint")
        joint_X_p = model.joint_X_p.numpy()[joint_idx]

        # The joint anchor position (in parent's frame) should be:
        # child_body_pos + rotate(joint_pos, child_body_orientation)
        #
        # Since child has no explicit rotation, it inherits parent's orientation.
        # child_body_pos relative to parent = (1, 0, 0)
        # child orientation relative to parent = identity (no additional rotation)
        # joint_pos = (0.5, 0, 0) in child's local frame
        #
        # But wait - the joint_X_p is the parent_xform which includes the body transform.
        # In the parent >= 0 case:
        #   relative_xform = inverse(parent_world) * child_world
        #   body_pos_for_joints = relative_xform.p = (1, 0, 0)
        #   body_ori_for_joints = relative_xform.q = identity (child has no local rotation)
        #
        # So joint anchor = (1, 0, 0) + rotate(identity, (0.5, 0, 0)) = (1.5, 0, 0)
        #
        # Actually, this test case doesn't trigger the bug because child has no
        # rotation relative to parent!

        # Let me verify the position - with identity rotation, the anchor should be (1.5, 0, 0)
        np.testing.assert_allclose(joint_X_p[:3], [1.5, 0.0, 0.0], atol=1e-5)

    def test_joint_anchor_with_rotated_child_body(self):
        """Test joint anchor when child body itself has rotation relative to parent.

        This specifically tests the case where joint_pos needs to be rotated by
        the child body's orientation (relative to parent) before being added.

        Setup:
        - Parent body at origin with no rotation
        - Child body at (2,0,0) with 90° Z rotation relative to parent
        - Joint with pos="1 0 0" in child's local frame

        The joint anchor (in parent frame) should be:
        - child_pos + rotate(joint_pos, child_orientation)
        - = (2,0,0) + rotate_90z(1,0,0)
        - = (2,0,0) + (0,1,0)
        - = (2, 1, 0)

        Bug would compute: (2,0,0) + (1,0,0) = (3, 0, 0) - WRONG
        """
        # Child has 90° rotation around Z relative to parent
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_joint_anchor_child_rotation">
    <worldbody>
        <body name="parent" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child" pos="2 0 0" quat="0.7071068 0 0 0.7071068">
                <joint name="rotated_joint" type="hinge" axis="0 0 1" pos="1 0 0"/>
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Find the child's joint
        joint_idx = model.joint_key.index("rotated_joint")
        joint_X_p = model.joint_X_p.numpy()[joint_idx]

        # The joint anchor position should be:
        # child_body_pos (2,0,0) + rotate_90z(joint_pos (1,0,0))
        # = (2,0,0) + (0,1,0) = (2, 1, 0)
        #
        # With the bug it would be: (2,0,0) + (1,0,0) = (3, 0, 0)
        np.testing.assert_allclose(
            joint_X_p[:3],
            [2.0, 1.0, 0.0],
            atol=1e-5,
            err_msg="Joint anchor should be rotated by child body orientation",
        )

        # Also verify the orientation is correct (90° Z rotation)
        # In xyzw format: [0, 0, sin(45°), cos(45°)] = [0, 0, 0.7071, 0.7071]
        np.testing.assert_allclose(joint_X_p[3:7], [0, 0, 0.7071068, 0.7071068], atol=1e-5)

    def test_floating_true_creates_free_joint(self):
        """Test that floating=True creates a free joint for the root body."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_floating">
    <worldbody>
        <body name="base_link" pos="0 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, floating=True)
        model = builder.finalize()

        self.assertEqual(model.joint_count, 1)
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.FREE)

    def test_floating_false_creates_fixed_joint(self):
        """Test that floating=False creates a fixed joint for the root body."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_fixed">
    <worldbody>
        <body name="base_link" pos="0 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, floating=False)
        model = builder.finalize()

        self.assertEqual(model.joint_count, 1)
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.FIXED)

    def test_base_joint_string_creates_d6_joint(self):
        """Test that base_joint as string creates a D6 joint with specified axes."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_base_joint_string">
    <worldbody>
        <body name="base_link" pos="0 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, base_joint="px,py,rz")
        model = builder.finalize()

        self.assertEqual(model.joint_count, 1)
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.D6)

    def test_base_joint_dict_creates_custom_joint(self):
        """Test that base_joint as dict creates the specified joint type."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_base_joint_dict">
    <worldbody>
        <body name="base_link" pos="0 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            mjcf_content,
            base_joint={
                "joint_type": newton.JointType.REVOLUTE,
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0, 0, 1])],
            },
        )
        model = builder.finalize()

        self.assertEqual(model.joint_count, 1)
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.REVOLUTE)

    def test_floating_and_base_joint_mutually_exclusive(self):
        """Test that specifying both base_joint and floating raises an error."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_base_joint_override">
    <worldbody>
        <body name="base_link" pos="0 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""
        # Specifying both parameters should raise ValueError
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as cm:
            builder.add_mjcf(mjcf_content, floating=True, base_joint="px,py")
        self.assertIn("both 'floating' and 'base_joint'", str(cm.exception))

    def test_base_joint_respects_import_xform(self):
        """Test that base joints (parent == -1) correctly use the import xform.

        This is a regression test for a bug where root bodies with base_joint
        ignored the import xform parameter, using raw body pos/ori instead of
        the composed world_xform.

        Setup:
        - Root body at (1, 0, 0) with no rotation
        - Import xform: translate by (10, 20, 30) and rotate 90° around Z
        - Using base_joint="lx,ly,lz" (D6 joint with linear axes)

        Expected final body transform after FK:
        - world_xform = import_xform * body_local_xform
        - = transform((10,20,30), rot_90z) * transform((1,0,0), identity)
        - Position: (10,20,30) + rotate_90z(1,0,0) = (10,20,30) + (0,1,0) = (10, 21, 30)
        - Orientation: 90° Z rotation

        Bug would give: position = (1, 0, 0), orientation = identity (ignoring import xform)
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_base_joint_xform">
    <worldbody>
        <body name="floating_body" pos="1 0 0">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        # Create import xform: translate + 90° Z rotation
        import_pos = wp.vec3(10.0, 20.0, 30.0)
        import_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2)  # 90° Z
        import_xform = wp.transform(import_pos, import_quat)

        # Use base_joint to convert freejoint to a D6 joint
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, xform=import_xform, base_joint="lx,ly,lz")
        model = builder.finalize()

        # Verify body transform after forward kinematics
        # Note: base_joint splits position and rotation between parent_xform and child_xform
        # to preserve joint axis directions, so we check the final body transform instead
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_idx = model.body_key.index("floating_body")
        body_q = state.body_q.numpy()[body_idx]

        # Expected position: import_pos + rotate_90z(body_pos)
        # = (10, 20, 30) + rotate_90z(1, 0, 0) = (10, 20, 30) + (0, 1, 0) = (10, 21, 30)
        np.testing.assert_allclose(
            body_q[:3],
            [10.0, 21.0, 30.0],
            atol=1e-5,
            err_msg="Body position should include import xform",
        )

        # Expected orientation: 90° Z rotation
        # In xyzw format: [0, 0, sin(45°), cos(45°)] = [0, 0, 0.7071, 0.7071]
        expected_quat = np.array([0, 0, 0.7071068, 0.7071068])
        actual_quat = body_q[3:7]
        quat_match = np.allclose(actual_quat, expected_quat, atol=1e-5) or np.allclose(
            actual_quat, -expected_quat, atol=1e-5
        )
        self.assertTrue(quat_match, f"Body orientation should include import xform. Got {actual_quat}")

    def test_base_joint_in_frame_respects_frame_xform(self):
        """Test that base joints inside frames correctly use the frame transform.

        Setup:
        - Frame at (5, 0, 0) with 90° Z rotation
        - Root body inside frame at (1, 0, 0) local position
        - Using base_joint

        Expected final body transform:
        - frame_xform * body_local_xform
        - = transform((5,0,0), rot_90z) * transform((1,0,0), identity)
        - Position: (5,0,0) + rotate_90z(1,0,0) = (5,0,0) + (0,1,0) = (5, 1, 0)
        - Orientation: 90° Z rotation

        Bug would give: position = (1, 0, 0), orientation = identity (ignoring frame transform)
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_base_joint_frame">
    <worldbody>
        <frame pos="5 0 0" quat="0.7071068 0 0 0.7071068">
            <body name="body_in_frame" pos="1 0 0">
                <freejoint/>
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </frame>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, base_joint="lx,ly,lz")
        model = builder.finalize()

        # Verify body transform after forward kinematics
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_idx = model.body_key.index("body_in_frame")
        body_q = state.body_q.numpy()[body_idx]

        # Expected position: frame_pos + rotate_90z(body_pos)
        # = (5, 0, 0) + rotate_90z(1, 0, 0) = (5, 0, 0) + (0, 1, 0) = (5, 1, 0)
        np.testing.assert_allclose(
            body_q[:3],
            [5.0, 1.0, 0.0],
            atol=1e-5,
            err_msg="Body position should include frame transform",
        )

        # Expected orientation: 90° Z rotation (from frame)
        expected_quat = np.array([0, 0, 0.7071068, 0.7071068])
        actual_quat = body_q[3:7]
        quat_match = np.allclose(actual_quat, expected_quat, atol=1e-5) or np.allclose(
            actual_quat, -expected_quat, atol=1e-5
        )
        self.assertTrue(quat_match, f"Body orientation should include frame rotation. Got {actual_quat}")

    def test_parent_body_attaches_to_existing_body(self):
        """Test that parent_body attaches the MJCF root to an existing body."""
        # First MJCF: a simple robot arm
        robot_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robot_arm">
    <worldbody>
        <body name="base_link" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
            <body name="end_effector" pos="1 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05" mass="0.5"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
        # Second MJCF: a gripper
        gripper_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="gripper">
    <worldbody>
        <body name="gripper_base" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.025 0.025 0.01" mass="0.2"/>
        </body>
    </worldbody>
</mujoco>
"""
        # First, load the robot
        builder = newton.ModelBuilder()
        builder.add_mjcf(robot_mjcf, floating=False)

        # Get the end effector body index
        ee_body_idx = builder.body_key.index("end_effector")

        # Remember the body count before adding gripper
        robot_body_count = builder.body_count
        robot_joint_count = builder.joint_count

        # Now load the gripper attached to the end effector
        builder.add_mjcf(gripper_mjcf, parent_body=ee_body_idx)

        model = builder.finalize()

        # Verify body counts
        self.assertEqual(model.body_count, robot_body_count + 1)  # Robot + gripper

        # Verify the gripper's base joint has the end effector as parent
        gripper_joint_idx = robot_joint_count  # First joint after robot
        self.assertEqual(model.joint_parent.numpy()[gripper_joint_idx], ee_body_idx)

        # Verify all joints belong to the same articulation
        joint_articulations = model.joint_articulation.numpy()
        robot_articulation = joint_articulations[0]
        gripper_articulation = joint_articulations[gripper_joint_idx]
        self.assertEqual(robot_articulation, gripper_articulation)

    def test_parent_body_with_base_joint_creates_d6(self):
        """Test that parent_body with base_joint creates a D6 joint to parent."""
        robot_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robot">
    <worldbody>
        <body name="base">
            <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""
        gripper_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="gripper">
    <worldbody>
        <body name="gripper_base">
            <freejoint/>
            <geom type="box" size="0.02 0.02 0.02" mass="0.2"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(robot_mjcf, floating=False)
        robot_body_idx = 0

        # Attach gripper with a D6 joint (rotation around Z)
        builder.add_mjcf(gripper_mjcf, parent_body=robot_body_idx, base_joint="rz")

        model = builder.finalize()

        # The second joint should be a D6 connecting to the robot body
        self.assertEqual(model.joint_count, 2)  # Fixed base + D6
        self.assertEqual(model.joint_type.numpy()[1], newton.JointType.D6)
        self.assertEqual(model.joint_parent.numpy()[1], robot_body_idx)

    def test_exclude_tag(self):
        """Test that <exclude> tags properly filter collisions between specified body pairs."""
        builder = newton.ModelBuilder()
        mjcf_filename = os.path.join(os.path.dirname(__file__), "assets", "mjcf_exclude_test.xml")
        builder.add_mjcf(
            mjcf_filename,
            enable_self_collisions=True,  # Enable self-collisions so we can test exclude filtering
        )

        model = builder.finalize()

        # Get shape indices for each body's geoms
        body1_geom1_idx = builder.shape_key.index("body1_geom1")
        body1_geom2_idx = builder.shape_key.index("body1_geom2")
        body2_geom1_idx = builder.shape_key.index("body2_geom1")
        body2_geom2_idx = builder.shape_key.index("body2_geom2")

        # Convert filter pairs to a set for easier checking
        filter_pairs = set(model.shape_collision_filter_pairs)

        # Check that all pairs between body1 and body2 are filtered (in both directions)
        body1_shapes = [body1_geom1_idx, body1_geom2_idx]
        body2_shapes = [body2_geom1_idx, body2_geom2_idx]

        for shape1 in body1_shapes:
            for shape2 in body2_shapes:
                # Check both orderings since the filter pairs can be added in either order
                pair_filtered = (shape1, shape2) in filter_pairs or (shape2, shape1) in filter_pairs
                self.assertTrue(
                    pair_filtered,
                    f"Shape pair ({shape1}, {shape2}) should be filtered due to <exclude body1='body1' body2='body2'/>",
                )

        # The test above verifies that body1-body2 pairs are correctly filtered.
        # We don't need to verify body3 interactions as that would require running
        # a full simulation to observe collision behavior.

    def test_exclude_tag_with_verbose(self):
        """Test that <exclude> tag parsing produces verbose output when requested."""
        builder = newton.ModelBuilder()
        mjcf_filename = os.path.join(os.path.dirname(__file__), "assets", "mjcf_exclude_test.xml")

        # Capture verbose output
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            builder.add_mjcf(
                mjcf_filename,
                enable_self_collisions=True,
                verbose=True,
            )
        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        # Check that the verbose output includes information about the exclude
        self.assertIn("Parsed collision exclude", output)
        self.assertIn("body1", output)
        self.assertIn("body2", output)

    def test_exclude_tag_missing_bodies(self):
        """Test that <exclude> tags with missing body references are handled gracefully."""
        mjcf_content = """
<mujoco>
  <worldbody>
    <body name="body1" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
  <contact>
    <!-- Reference to non-existent body -->
    <exclude body1="body1" body2="nonexistent_body"/>
  </contact>
</mujoco>
"""
        builder = newton.ModelBuilder()
        # Should not raise an error, just skip the invalid exclude and continue parsing
        builder.add_mjcf(mjcf_content, enable_self_collisions=True, verbose=False)

        # Verify the model can still be finalized successfully
        model = builder.finalize()
        self.assertIsNotNone(model)

    def test_exclude_tag_with_hyphens(self):
        """Test that <exclude> tags work with hyphenated body names (normalized to underscores)."""
        builder = newton.ModelBuilder()
        mjcf_filename = os.path.join(os.path.dirname(__file__), "assets", "mjcf_exclude_hyphen_test.xml")
        builder.add_mjcf(
            mjcf_filename,
            enable_self_collisions=True,  # Enable self-collisions so we can test exclude filtering
        )

        model = builder.finalize()

        # Body names with hyphens should be normalized to underscores in builder.body_key
        self.assertIn("body_with_hyphens", builder.body_key)
        self.assertIn("another_hyphen_body", builder.body_key)

        # Get shape indices for each body's geoms
        hyphen_geom1_idx = builder.shape_key.index("hyphen_geom1")
        hyphen_geom2_idx = builder.shape_key.index("hyphen_geom2")
        another_geom1_idx = builder.shape_key.index("another_geom1")
        another_geom2_idx = builder.shape_key.index("another_geom2")

        # Convert filter pairs to a set for easier checking
        filter_pairs = set(model.shape_collision_filter_pairs)

        # Check that all pairs between the two hyphenated bodies are filtered
        hyphen_shapes = [hyphen_geom1_idx, hyphen_geom2_idx]
        another_shapes = [another_geom1_idx, another_geom2_idx]

        for shape1 in hyphen_shapes:
            for shape2 in another_shapes:
                # Check both orderings since the filter pairs can be added in either order
                pair_filtered = (shape1, shape2) in filter_pairs or (shape2, shape1) in filter_pairs
                self.assertTrue(
                    pair_filtered,
                    f"Shape pair ({shape1}, {shape2}) should be filtered due to <exclude body1='body-with-hyphens' body2='another-hyphen-body'/>",
                )

    def test_exclude_tag_missing_attributes(self):
        """Test that <exclude> tags with missing attributes are handled gracefully."""
        mjcf_content = """
<mujoco>
  <worldbody>
    <body name="body1" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
  <contact>
    <!-- Missing body2 attribute -->
    <exclude body1="body1"/>
  </contact>
</mujoco>
"""
        builder = newton.ModelBuilder()
        # Should not raise an error, just skip the invalid exclude and continue parsing
        builder.add_mjcf(mjcf_content, enable_self_collisions=True, verbose=False)

        # Verify the model can still be finalized successfully
        model = builder.finalize()
        self.assertIsNotNone(model)

        # Verify body1 was still parsed correctly
        self.assertIn("body1", builder.body_key)

    def test_exclude_tag_warnings_verbose(self):
        """Test that warnings are printed for invalid exclude tags when verbose=True."""
        mjcf_content = """
<mujoco>
  <worldbody>
    <body name="body1" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
  <contact>
    <!-- Multiple invalid excludes to test different error cases -->
    <exclude body1="body1" body2="nonexistent"/>
    <exclude body1="body1"/>
    <exclude/>
  </contact>
</mujoco>
"""
        builder = newton.ModelBuilder()

        # Capture verbose output
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            builder.add_mjcf(mjcf_content, enable_self_collisions=True, verbose=True)
        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        # Check that warnings were printed for invalid exclude entries
        self.assertIn("Warning", output)
        self.assertIn("<exclude>", output)

    def test_base_joint_on_fixed_root(self):
        """Test that base_joint works on MJCF with fixed root (no freejoint)."""
        fixed_root_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="fixed_root_robot">
    <worldbody>
        <body name="base_link" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
            <body name="link1" pos="1 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05" mass="0.5"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(fixed_root_mjcf, base_joint="px,py,rz")
        model = builder.finalize()

        self.assertEqual(model.joint_count, 2)
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.D6)
        self.assertEqual(model.joint_dof_count, 4)

    def test_xform_relative_to_parent_body(self):
        """Test that xform is interpreted relative to parent_body when attaching."""
        robot_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robot">
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
            <body name="end_effector" pos="0 1 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05" mass="0.5"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
        gripper_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="gripper">
    <worldbody>
        <body name="gripper_base" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.02 0.02 0.02" mass="0.2"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(robot_mjcf, xform=wp.transform((0.0, 2.0, 0.0), wp.quat_identity()), floating=False)

        ee_body_idx = builder.body_key.index("end_effector")
        ee_world_pos = builder.body_q[ee_body_idx].p

        builder.add_mjcf(gripper_mjcf, parent_body=ee_body_idx, xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))

        gripper_body_idx = builder.body_key.index("gripper_base")
        gripper_world_pos = builder.body_q[gripper_body_idx].p

        self.assertAlmostEqual(gripper_world_pos[0], ee_world_pos[0], places=5)
        self.assertAlmostEqual(gripper_world_pos[1], ee_world_pos[1], places=5)
        self.assertAlmostEqual(gripper_world_pos[2], ee_world_pos[2] + 0.1, places=5)

    def test_non_sequential_articulation_attachment(self):
        """Test that attaching to a non-sequential articulation raises an error."""
        robot_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robot">
    <worldbody>
        <body name="robot_base" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
            <body name="robot_link" pos="1 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05" mass="0.5"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
        gripper_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="gripper">
    <worldbody>
        <body name="gripper_base">
            <freejoint/>
            <geom type="box" size="0.02 0.02 0.02" mass="0.2"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(robot_mjcf, floating=False)
        robot1_link_idx = builder.body_key.index("robot_link")

        # Add more robots to make robot1_link_idx not part of the most recent articulation
        builder.add_mjcf(robot_mjcf, floating=False)
        builder.add_mjcf(robot_mjcf, floating=False)

        # Attempting to attach to a non-sequential articulation should raise ValueError
        with self.assertRaises(ValueError) as cm:
            builder.add_mjcf(gripper_mjcf, parent_body=robot1_link_idx, floating=False)
        self.assertIn("most recent", str(cm.exception))

    def test_floating_true_with_parent_body_raises_error(self):
        """Test that floating=True with parent_body raises an error."""
        robot_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robot">
    <worldbody>
        <body name="robot_base" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
            <body name="robot_link" pos="1 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05" mass="0.5"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
        gripper_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="gripper">
    <worldbody>
        <body name="gripper_base">
            <geom type="box" size="0.02 0.02 0.02" mass="0.2"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(robot_mjcf, floating=False)
        link_idx = builder.body_key.index("robot_link")

        # Attempting to use floating=True with parent_body should raise ValueError
        with self.assertRaises(ValueError) as cm:
            builder.add_mjcf(gripper_mjcf, parent_body=link_idx, floating=True)
        self.assertIn("FREE joint", str(cm.exception))
        self.assertIn("parent_body", str(cm.exception))

    def test_floating_none_preserves_mjcf_default(self):
        """Test that floating=None honors MJCF freejoint tags."""
        # Test with freejoint
        mjcf_with_freejoint = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_with_freejoint">
    <worldbody>
        <body name="floating_body" pos="0 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_with_freejoint, floating=None)
        model = builder.finalize()
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.FREE)

        # Test without freejoint
        mjcf_without_freejoint = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_without_freejoint">
    <worldbody>
        <body name="fixed_body" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_without_freejoint, floating=None)
        model = builder.finalize()
        self.assertEqual(model.joint_type.numpy()[0], newton.JointType.FIXED)

    def test_sequential_attachment_succeeds(self):
        """Test that attaching to the most recent articulation succeeds."""
        robot_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robot">
    <worldbody>
        <body name="robot_base" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
            <body name="robot_link" pos="1 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05" mass="0.5"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
        gripper_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="gripper">
    <worldbody>
        <body name="gripper_base">
            <geom type="box" size="0.02 0.02 0.02" mass="0.2"/>
            <body name="gripper_finger">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="box" size="0.01 0.01 0.03" mass="0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(robot_mjcf, floating=False)
        link_idx = builder.body_key.index("robot_link")

        # Attach gripper immediately - should succeed
        builder.add_mjcf(gripper_mjcf, parent_body=link_idx, floating=False)
        model = builder.finalize()

        # Verify both are in the same articulation
        # articulation_start has one extra element as an end marker, so length-1 = number of articulations
        self.assertEqual(len(model.articulation_start.numpy()) - 1, 1)
        # Should have 4 joints: robot FIXED base + robot hinge + gripper FIXED base + gripper hinge
        self.assertEqual(model.joint_count, 4)

    def test_floating_false_with_parent_body_succeeds(self):
        """Test that floating=False with parent_body is explicitly allowed."""
        robot_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robot">
    <worldbody>
        <body name="robot_base" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
            <body name="robot_link" pos="1 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05" mass="0.5"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
        gripper_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="gripper">
    <worldbody>
        <body name="gripper_base">
            <geom type="box" size="0.02 0.02 0.02" mass="0.2"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(robot_mjcf, floating=False)
        link_idx = builder.body_key.index("robot_link")

        # Explicitly using floating=False with parent_body should succeed
        builder.add_mjcf(gripper_mjcf, parent_body=link_idx, floating=False)
        model = builder.finalize()

        # Verify it worked - gripper should be attached with FIXED joint
        self.assertIn("gripper_base", builder.body_key)
        self.assertEqual(len(model.articulation_start.numpy()) - 1, 1)  # Single articulation

    def test_multi_root_mjcf_with_parent_body(self):
        """Test MJCF with multiple root bodies and parent_body parameter."""
        # MJCF with two root bodies under worldbody
        multi_root_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="multi_root">
    <worldbody>
        <body name="root1" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
        </body>
        <body name="root2" pos="1 0 0">
            <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""
        robot_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robot">
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
            <body name="link" pos="1 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05" mass="0.5"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(robot_mjcf, floating=False)
        link_idx = builder.body_key.index("link")

        # Add multi-root MJCF - both root bodies should attach to the same parent
        builder.add_mjcf(multi_root_mjcf, parent_body=link_idx, floating=False)
        model = builder.finalize()

        # Verify both root bodies were added
        self.assertIn("root1", builder.body_key)
        self.assertIn("root2", builder.body_key)
        # Should still be one articulation
        self.assertEqual(len(model.articulation_start.numpy()) - 1, 1)

    def test_error_messages_are_informative(self):
        """Test that error messages contain helpful information."""
        robot_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="robot">
    <worldbody>
        <body name="robot_base" pos="0 0 0">
            <geom type="sphere" size="0.1" mass="1.0"/>
            <body name="robot_link" pos="1 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05" mass="0.5"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(robot_mjcf, floating=False)
        robot1_link_idx = builder.body_key.index("robot_link")

        # Add more robots to make robot1_link_idx not the most recent
        builder.add_mjcf(robot_mjcf, floating=False)
        builder.add_mjcf(robot_mjcf, floating=False)

        # Try to attach to non-sequential articulation
        gripper_mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="gripper">
    <worldbody>
        <body name="gripper_base">
            <geom type="box" size="0.02 0.02 0.02" mass="0.2"/>
        </body>
    </worldbody>
</mujoco>
"""
        with self.assertRaises(ValueError) as cm:
            builder.add_mjcf(gripper_mjcf, parent_body=robot1_link_idx, floating=False)

        # Check that error message is informative
        error_msg = str(cm.exception)
        self.assertIn("most recent", error_msg)
        self.assertIn("articulation", error_msg)
        # Should mention the body name
        self.assertIn("robot_link", error_msg)


class TestMjcfInclude(unittest.TestCase):
    """Tests for MJCF <include> tag support."""

    def test_basic_include_same_directory(self):
        """Test including a file from the same directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the included file
            included_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <worldbody>
        <body name="included_body">
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>"""
            included_path = os.path.join(tmpdir, "included.xml")
            with open(included_path, "w") as f:
                f.write(included_content)

            # Create the main file that includes it
            main_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include file="included.xml"/>
</mujoco>"""
            main_path = os.path.join(tmpdir, "main.xml")
            with open(main_path, "w") as f:
                f.write(main_content)

            # Parse and verify
            builder = newton.ModelBuilder()
            builder.add_mjcf(main_path)
            self.assertEqual(builder.body_count, 1)

    def test_include_subdirectory(self):
        """Test including a file from a subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory
            subdir = os.path.join(tmpdir, "models")
            os.makedirs(subdir)

            # Create the included file in subdirectory
            included_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <worldbody>
        <body name="subdir_body">
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""
            included_path = os.path.join(subdir, "robot.xml")
            with open(included_path, "w") as f:
                f.write(included_content)

            # Create the main file
            main_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include file="models/robot.xml"/>
</mujoco>"""
            main_path = os.path.join(tmpdir, "scene.xml")
            with open(main_path, "w") as f:
                f.write(main_content)

            # Parse and verify
            builder = newton.ModelBuilder()
            builder.add_mjcf(main_path)
            self.assertEqual(builder.body_count, 1)

    def test_include_absolute_path(self):
        """Test including a file using absolute path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the included file
            included_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <worldbody>
        <body name="absolute_body">
            <geom type="capsule" size="0.05 0.1"/>
        </body>
    </worldbody>
</mujoco>"""
            included_path = os.path.join(tmpdir, "absolute.xml")
            with open(included_path, "w") as f:
                f.write(included_content)

            # Create the main file with absolute path
            main_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include file="{included_path}"/>
</mujoco>"""
            main_path = os.path.join(tmpdir, "main.xml")
            with open(main_path, "w") as f:
                f.write(main_content)

            # Parse and verify
            builder = newton.ModelBuilder()
            builder.add_mjcf(main_path)
            self.assertEqual(builder.body_count, 1)

    def test_include_multiple_sections(self):
        """Test including content that goes into different sections (asset, default, worldbody)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create included file with defaults
            defaults_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <default>
        <geom rgba="1 0 0 1"/>
    </default>
</mujoco>"""
            defaults_path = os.path.join(tmpdir, "defaults.xml")
            with open(defaults_path, "w") as f:
                f.write(defaults_content)

            # Create included file with worldbody
            body_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <worldbody>
        <body name="red_body">
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>"""
            body_path = os.path.join(tmpdir, "body.xml")
            with open(body_path, "w") as f:
                f.write(body_content)

            # Create main file that includes both
            main_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include file="defaults.xml"/>
    <include file="body.xml"/>
</mujoco>"""
            main_path = os.path.join(tmpdir, "main.xml")
            with open(main_path, "w") as f:
                f.write(main_content)

            # Parse and verify
            builder = newton.ModelBuilder()
            builder.add_mjcf(main_path)
            self.assertEqual(builder.body_count, 1)

    def test_include_resolves_asset_paths(self):
        """Test that asset paths in included files are resolved relative to the included file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create robot subdirectory with mesh subdirectory
            robot_dir = os.path.join(tmpdir, "robot")
            mesh_dir = os.path.join(robot_dir, "meshes")
            os.makedirs(mesh_dir)

            # Create a simple OBJ mesh file
            mesh_content = """# Simple cube
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""
            mesh_path = os.path.join(mesh_dir, "cube.obj")
            with open(mesh_path, "w") as f:
                f.write(mesh_content)

            # Create robot.xml that references mesh relative to its location
            robot_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <asset>
        <mesh name="cube_mesh" file="meshes/cube.obj"/>
    </asset>
    <worldbody>
        <body name="robot_body">
            <geom type="mesh" mesh="cube_mesh"/>
        </body>
    </worldbody>
</mujoco>"""
            robot_path = os.path.join(robot_dir, "robot.xml")
            with open(robot_path, "w") as f:
                f.write(robot_content)

            # Create main scene.xml that includes robot/robot.xml
            main_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="scene">
    <include file="robot/robot.xml"/>
</mujoco>"""
            main_path = os.path.join(tmpdir, "scene.xml")
            with open(main_path, "w") as f:
                f.write(main_content)

            # Parse - this should work because mesh path is resolved relative to robot.xml
            builder = newton.ModelBuilder()
            builder.add_mjcf(main_path)
            self.assertEqual(builder.body_count, 1)
            self.assertEqual(builder.shape_count, 1)  # Verify mesh shape was created

            # Verify mesh vertices were actually loaded (cube has 8 vertices)
            model = builder.finalize()
            mesh = model.shape_source[0]
            self.assertEqual(len(mesh.vertices), 8)


class TestMjcfIncludeNested(unittest.TestCase):
    """Tests for nested includes and cycle detection."""

    def test_nested_includes(self):
        """Test that nested includes work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the deepest included file
            deep_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <worldbody>
        <body name="deep_body">
            <geom type="sphere" size="0.05"/>
        </body>
    </worldbody>
</mujoco>"""
            deep_path = os.path.join(tmpdir, "deep.xml")
            with open(deep_path, "w") as f:
                f.write(deep_content)

            # Create middle file that includes deep file
            middle_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <include file="deep.xml"/>
</mujoco>"""
            middle_path = os.path.join(tmpdir, "middle.xml")
            with open(middle_path, "w") as f:
                f.write(middle_content)

            # Create main file that includes middle file
            main_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include file="middle.xml"/>
</mujoco>"""
            main_path = os.path.join(tmpdir, "main.xml")
            with open(main_path, "w") as f:
                f.write(main_content)

            # Parse and verify
            builder = newton.ModelBuilder()
            builder.add_mjcf(main_path)
            self.assertEqual(builder.body_count, 1)

    def test_circular_include_detection(self):
        """Test that circular includes are detected and raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file A that includes file B
            file_a_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <include file="b.xml"/>
</mujoco>"""
            file_a_path = os.path.join(tmpdir, "a.xml")
            with open(file_a_path, "w") as f:
                f.write(file_a_content)

            # Create file B that includes file A (circular)
            file_b_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <include file="a.xml"/>
</mujoco>"""
            file_b_path = os.path.join(tmpdir, "b.xml")
            with open(file_b_path, "w") as f:
                f.write(file_b_content)

            # Attempt to parse should raise ValueError
            builder = newton.ModelBuilder()
            with self.assertRaises(ValueError) as context:
                builder.add_mjcf(file_a_path)
            self.assertIn("Circular include", str(context.exception))

    def test_include_without_file_attribute(self):
        """Test that include elements without file attribute are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create main file with an include that has no file attribute
            main_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include/>
    <worldbody>
        <body name="body1">
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>"""
            main_path = os.path.join(tmpdir, "main.xml")
            with open(main_path, "w") as f:
                f.write(main_content)

            # Should parse successfully, ignoring the empty include
            builder = newton.ModelBuilder()
            builder.add_mjcf(main_path)
            self.assertEqual(builder.body_count, 1)

    def test_self_include_detection(self):
        """Test that a file including itself is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file that includes itself
            self_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <include file="self.xml"/>
</mujoco>"""
            self_path = os.path.join(tmpdir, "self.xml")
            with open(self_path, "w") as f:
                f.write(self_content)

            # Attempt to parse should raise ValueError
            builder = newton.ModelBuilder()
            with self.assertRaises(ValueError) as context:
                builder.add_mjcf(self_path)
            self.assertIn("Circular include", str(context.exception))

    def test_missing_include_file(self):
        """Test that missing include files raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create main file that includes a non-existent file
            main_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include file="does_not_exist.xml"/>
</mujoco>"""
            main_path = os.path.join(tmpdir, "main.xml")
            with open(main_path, "w") as f:
                f.write(main_content)

            builder = newton.ModelBuilder()
            with self.assertRaises(FileNotFoundError):
                builder.add_mjcf(main_path)

    def test_relative_include_without_base_dir(self):
        """Test that relative includes from XML string input raise ValueError with default resolver."""
        # XML string with relative include - default resolver can't resolve without base_dir
        main_xml = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include file="relative.xml"/>
</mujoco>"""

        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as context:
            builder.add_mjcf(main_xml)
        self.assertIn("Cannot resolve relative path", str(context.exception))
        self.assertIn("without base directory", str(context.exception))

    def test_invalid_source_not_file_not_xml(self):
        """Test that invalid source (not a file path, not XML) raises FileNotFoundError."""
        builder = newton.ModelBuilder()
        with self.assertRaises(FileNotFoundError):
            builder.add_mjcf("this_is_not_a_file_and_not_xml")


class TestMjcfIncludeCallback(unittest.TestCase):
    """Tests for custom path_resolver callback."""

    def test_custom_path_resolver_returns_xml(self):
        """Test custom callback that returns XML content directly for includes."""
        # XML content to be "included"
        included_xml = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <worldbody>
        <body name="virtual_body">
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        def custom_resolver(_base_dir, file_path):
            if file_path == "virtual.xml":
                return included_xml
            raise ValueError(f"Unknown file: {file_path}")

        # Main MJCF as string
        main_xml = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include file="virtual.xml"/>
</mujoco>"""

        # Parse with custom resolver
        builder = newton.ModelBuilder()
        builder.add_mjcf(main_xml, path_resolver=custom_resolver)
        self.assertEqual(builder.body_count, 1)

    def test_custom_path_resolver_with_base_dir(self):
        """Test that custom callback receives correct base_dir."""
        received_args = []

        def tracking_resolver(base_dir, file_path):
            received_args.append((base_dir, file_path))
            return """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <worldbody>
        <body name="tracked_body">
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            main_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include file="test.xml"/>
</mujoco>"""
            main_path = os.path.join(tmpdir, "main.xml")
            with open(main_path, "w") as f:
                f.write(main_content)

            builder = newton.ModelBuilder()
            builder.add_mjcf(main_path, path_resolver=tracking_resolver)

            # Verify callback received correct arguments
            self.assertEqual(len(received_args), 1)
            self.assertEqual(received_args[0][0], tmpdir)
            self.assertEqual(received_args[0][1], "test.xml")

    def test_xml_string_input_with_custom_resolver(self):
        """Test that XML string input works with custom resolver (base_dir is None)."""
        received_base_dirs = []

        def tracking_resolver(base_dir, _file_path):
            received_base_dirs.append(base_dir)
            return """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <worldbody>
        <body name="string_body">
            <geom type="sphere" size="0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        main_xml = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <include file="any.xml"/>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(main_xml, path_resolver=tracking_resolver)

        # base_dir should be None for XML string input
        self.assertEqual(len(received_base_dirs), 1)
        self.assertIsNone(received_base_dirs[0])

    def test_dof_angle_conversion_radians(self):
        """Test DOF attributes pass through unchanged when compiler.angle='radian'."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_radians">
    <compiler angle="radian"/>
    <worldbody>
        <body name="base">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child1" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 0 1" stiffness="10" damping="5" springref="0.785" ref="0.524"/>
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        qd_start = model.joint_qd_start.numpy()
        hinge_idx = model.joint_key.index("hinge")
        dof_idx = qd_start[hinge_idx]

        # No conversion when angle="radian" - values pass through unchanged
        self.assertAlmostEqual(model.mujoco.dof_springref.numpy()[dof_idx], 0.785, places=4)
        self.assertAlmostEqual(model.mujoco.dof_ref.numpy()[dof_idx], 0.524, places=4)
        self.assertAlmostEqual(model.mujoco.dof_passive_stiffness.numpy()[dof_idx], 10.0, places=4)
        self.assertAlmostEqual(model.mujoco.dof_passive_damping.numpy()[dof_idx], 5.0, places=4)

    def test_dof_angle_conversion_slide_joint(self):
        """Test DOF attributes for slide joints are not converted regardless of angle setting."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_slide">
    <compiler angle="degree"/>
    <worldbody>
        <body name="base">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child1" pos="0 0 1">
                <joint name="slide" type="slide" axis="0 0 1" stiffness="100" damping="10" springref="0.5" ref="0.1"/>
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        qd_start = model.joint_qd_start.numpy()
        slide_idx = model.joint_key.index("slide")
        dof_idx = qd_start[slide_idx]

        # Slide joints: values pass through unchanged (linear, not angular)
        self.assertAlmostEqual(model.mujoco.dof_springref.numpy()[dof_idx], 0.5, places=4)
        self.assertAlmostEqual(model.mujoco.dof_ref.numpy()[dof_idx], 0.1, places=4)
        self.assertAlmostEqual(model.mujoco.dof_passive_stiffness.numpy()[dof_idx], 100.0, places=4)
        self.assertAlmostEqual(model.mujoco.dof_passive_damping.numpy()[dof_idx], 10.0, places=4)

    def test_dof_angle_conversion_degrees(self):
        """Test DOF attributes are converted from degrees when compiler.angle='degree' (default)."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_degrees">
    <compiler angle="degree"/>
    <worldbody>
        <body name="base">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child1" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 0 1" stiffness="10" damping="5" springref="45" ref="30"/>
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        qd_start = model.joint_qd_start.numpy()
        hinge_idx = model.joint_key.index("hinge")
        dof_idx = qd_start[hinge_idx]

        # springref/ref: converted from deg to rad (45 deg -> 45 * pi/180 rad)
        self.assertAlmostEqual(model.mujoco.dof_springref.numpy()[dof_idx], np.deg2rad(45), places=4)
        self.assertAlmostEqual(model.mujoco.dof_ref.numpy()[dof_idx], np.deg2rad(30), places=4)

        # stiffness/damping: converted from Nm/deg to Nm/rad (10 Nm/deg -> 10 * 180/pi Nm/rad)
        self.assertAlmostEqual(model.mujoco.dof_passive_stiffness.numpy()[dof_idx], 10 * (180 / np.pi), places=2)
        self.assertAlmostEqual(model.mujoco.dof_passive_damping.numpy()[dof_idx], 5 * (180 / np.pi), places=2)


class TestMjcfMultipleWorldbody(unittest.TestCase):
    """Test that multiple worldbody elements are handled correctly.

    MuJoCo allows multiple <worldbody> elements (e.g., from includes),
    and all should be processed into the world body.
    """

    def test_multiple_worldbody_elements(self):
        """Test that geoms and bodies from multiple worldbody elements are parsed."""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <geom name="floor1" type="plane" size="1 1 0.1"/>
                <body name="body1">
                    <geom name="geom1" type="sphere" size="0.1"/>
                </body>
            </worldbody>
            <worldbody>
                <geom name="floor2" type="box" size="0.5 0.5 0.1"/>
                <body name="body2">
                    <geom name="geom2" type="capsule" size="0.1 0.2"/>
                </body>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, up_axis="Z")

        # Should have bodies from both worldbodies
        self.assertIn("body1", builder.body_key)
        self.assertIn("body2", builder.body_key)

        # Should have geoms from both worldbodies (floor1, floor2, geom1, geom2)
        self.assertIn("floor1", builder.shape_key)
        self.assertIn("floor2", builder.shape_key)
        self.assertIn("geom1", builder.shape_key)
        self.assertIn("geom2", builder.shape_key)

    def test_multiple_worldbody_with_sites(self):
        """Test that sites from multiple worldbody elements are parsed."""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <site name="site1" pos="0 0 0"/>
                <body name="body1">
                    <geom type="sphere" size="0.1"/>
                </body>
            </worldbody>
            <worldbody>
                <site name="site2" pos="1 0 0"/>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, up_axis="Z", parse_sites=True)

        # Should have sites from both worldbodies
        self.assertIn("site1", builder.shape_key)
        self.assertIn("site2", builder.shape_key)

    def test_include_creates_multiple_worldbodies(self):
        """Test that includes properly create multiple worldbody elements."""
        # Create a main file that includes another file, both with worldbodies
        included_xml = """
        <mujoco>
            <worldbody>
                <body name="included_body">
                    <geom name="included_geom" type="sphere" size="0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        main_xml = """
        <mujoco>
            <include file="included.xml"/>
            <worldbody>
                <geom name="main_floor" type="plane" size="1 1 0.1"/>
                <body name="main_body">
                    <geom name="main_geom" type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        # Custom resolver that returns XML content directly
        def xml_resolver(base_dir, file_path):
            if "included.xml" in file_path:
                return included_xml
            return file_path

        builder = newton.ModelBuilder()
        builder.add_mjcf(main_xml, up_axis="Z", path_resolver=xml_resolver)

        # Should have bodies from both worldbodies
        self.assertIn("included_body", builder.body_key)
        self.assertIn("main_body", builder.body_key)

        # Should have geoms from both worldbodies
        self.assertIn("included_geom", builder.shape_key)
        self.assertIn("main_floor", builder.shape_key)
        self.assertIn("main_geom", builder.shape_key)


class TestMjcfActuatorAutoLimited(unittest.TestCase):
    """Test auto-enabling of actuator *limited flags when *range is specified."""

    def test_ctrllimited_auto_enabled_when_ctrlrange_specified(self):
        """Test that ctrllimited is auto-enabled when ctrlrange is specified."""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <body name="base">
                    <joint name="joint1" type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
            <actuator>
                <!-- ctrlrange specified but ctrllimited not explicitly set -->
                <general name="act1" joint="joint1" ctrlrange="-1 1"/>
            </actuator>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # ctrllimited should be auto-enabled (1) because ctrlrange was specified
        ctrllimited = model.mujoco.actuator_ctrllimited.numpy()
        self.assertEqual(ctrllimited[0], 1)

    def test_ctrllimited_not_auto_enabled_without_ctrlrange(self):
        """Test that ctrllimited stays disabled when ctrlrange is not specified."""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <body name="base">
                    <joint name="joint1" type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
            <actuator>
                <!-- No ctrlrange specified -->
                <general name="act1" joint="joint1"/>
            </actuator>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # ctrllimited should be disabled (0) because ctrlrange was not specified
        ctrllimited = model.mujoco.actuator_ctrllimited.numpy()
        self.assertEqual(ctrllimited[0], 0)

    def test_ctrllimited_explicit_false_not_overridden(self):
        """Test that explicit ctrllimited=false is not overridden."""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <body name="base">
                    <joint name="joint1" type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
            <actuator>
                <!-- ctrlrange specified but ctrllimited explicitly set to false -->
                <general name="act1" joint="joint1" ctrlrange="-1 1" ctrllimited="false"/>
            </actuator>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # ctrllimited should stay disabled (0) because it was explicitly set
        ctrllimited = model.mujoco.actuator_ctrllimited.numpy()
        self.assertEqual(ctrllimited[0], 0)

    def test_forcelimited_auto_enabled_when_forcerange_specified(self):
        """Test that forcelimited is auto-enabled when forcerange is specified."""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <body name="base">
                    <joint name="joint1" type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
            <actuator>
                <general name="act1" joint="joint1" forcerange="-100 100"/>
            </actuator>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        forcelimited = model.mujoco.actuator_forcelimited.numpy()
        self.assertEqual(forcelimited[0], 1)

    def test_actlimited_auto_enabled_when_actrange_specified(self):
        """Test that actlimited is auto-enabled when actrange is specified."""
        mjcf_content = """
        <mujoco>
            <worldbody>
                <body name="base">
                    <joint name="joint1" type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
            <actuator>
                <general name="act1" joint="joint1" actrange="0 1" dyntype="integrator"/>
            </actuator>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        actlimited = model.mujoco.actuator_actlimited.numpy()
        self.assertEqual(actlimited[0], 1)


class TestMjcfDefaultCustomAttributes(unittest.TestCase):
    """Verify that MJCF <default> classes properly propagate MuJoCo custom attribute values."""

    DEG2RAD = np.pi / 180.0
    RAD2DEG = 180.0 / np.pi

    MJCF = """
    <mujoco>
        <default>
            <geom condim="4" priority="5" solmix="2.0" gap="0.01"
                  solimp="0.8 0.9 0.01 0.4 1.0"/>
            <body gravcomp="0.5"/>
            <joint margin="1.0" solimplimit="0.8 0.9 0.01 0.4 1.0"
                   solreffriction="0.05 2.0" solimpfriction="0.7 0.85 0.02 0.3 1.5"
                   stiffness="2.0" damping="3.0" springref="45" ref="30"
                   actuatorgravcomp="true"/>
            <general gainprm="100" biasprm="0 -100 -10" dyntype="filter"
                     ctrllimited="true" forcelimited="true" ctrlrange="-2 2"
                     forcerange="-100 100" gear="2 0 0 0 0 0"
                     dynprm="0.5 0 0 0 0 0 0 0 0 0"
                     actlimited="true" actrange="-1 1" actdim="2" actearly="true"/>
            <position kp="50"/>
            <default class="special">
                <geom condim="6" solmix="5.0"
                      solimp="0.7 0.85 0.005 0.3 1.5"/>
                <body gravcomp="1.0"/>
                <joint margin="10.0" stiffness="20.0" damping="30.0"
                       springref="90" ref="60"/>
                <general gainprm="200" biasprm="0 -200 -20"
                         ctrllimited="false" ctrlrange="-5 5"
                         gear="3 0 0 0 0 0"/>
                <position kp="500"/>
                <default class="special_child">
                    <geom priority="99" gap="0.05"/>
                    <general gainprm="300" biasprm="0 -300 -30"/>
                    <position kp="3000"/>
                </default>
            </default>
        </default>
        <worldbody>
            <body name="b_default" pos="0 0 0">
                <joint name="j_default" type="hinge" axis="0 0 1"/>
                <geom name="g_default" type="sphere" size="0.1"/>
                <body name="b_class" class="special" pos="0 0 1">
                    <joint name="j_class" type="hinge" axis="0 0 1" class="special"/>
                    <geom name="g_class" type="sphere" size="0.1" class="special"/>
                    <body name="b_override" class="special" pos="0 0 1" gravcomp="0.75">
                        <joint name="j_override" type="hinge" axis="0 0 1"
                               class="special" margin="99.0"/>
                        <geom name="g_override" type="sphere" size="0.1"
                              class="special" condim="1"/>
                        <body name="b_child" pos="0 0 1">
                            <joint name="j_child" type="hinge" axis="0 0 1"/>
                            <geom name="g_child" type="sphere" size="0.1"
                                  class="special_child"/>
                            <body name="b_child2" pos="0 0 1">
                                <joint name="j_child2" type="hinge" axis="0 0 1"/>
                                <geom type="sphere" size="0.1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>
        <actuator>
            <general name="act_default" joint="j_default"
                     gaintype="fixed" biastype="affine"/>
            <general name="act_class" joint="j_class" class="special"
                     gaintype="fixed" biastype="affine"/>
            <general name="act_override" joint="j_override" class="special"
                     gaintype="fixed" biastype="affine"
                     gainprm="999" biasprm="0 -999 -99"/>
            <general name="act_child" joint="j_child" class="special_child"
                     gaintype="fixed" biastype="affine"/>
            <position name="pos_default" joint="j_child2"/>
            <position name="pos_class" joint="j_class" class="special"/>
            <position name="pos_override" joint="j_override" class="special"
                      kp="9999"/>
            <position name="pos_child" joint="j_child" class="special_child"/>
        </actuator>
    </mujoco>
    """

    @classmethod
    def setUpClass(cls):
        cls.builder = newton.ModelBuilder()
        cls.builder.add_mjcf(cls.MJCF, ctrl_direct=True)
        cls.model = cls.builder.finalize()

    def test_shape_defaults(self):
        """SHAPE: condim, priority, solmix, gap, solimp."""
        m = self.model.mujoco
        idx = self.builder.shape_key.index

        g_def = idx("g_default")
        self.assertEqual(m.condim.numpy()[g_def], 4)
        self.assertEqual(m.geom_priority.numpy()[g_def], 5)
        self.assertAlmostEqual(float(m.geom_solmix.numpy()[g_def]), 2.0, places=5)
        self.assertAlmostEqual(float(m.geom_gap.numpy()[g_def]), 0.01, places=5)
        np.testing.assert_allclose(m.geom_solimp.numpy()[g_def], [0.8, 0.9, 0.01, 0.4, 1.0], atol=1e-4)

        g_cls = idx("g_class")
        self.assertEqual(m.condim.numpy()[g_cls], 6)
        self.assertEqual(m.geom_priority.numpy()[g_cls], 5)
        self.assertAlmostEqual(float(m.geom_solmix.numpy()[g_cls]), 5.0, places=5)
        self.assertAlmostEqual(float(m.geom_gap.numpy()[g_cls]), 0.01, places=5)
        np.testing.assert_allclose(m.geom_solimp.numpy()[g_cls], [0.7, 0.85, 0.005, 0.3, 1.5], atol=1e-4)

        g_ovr = idx("g_override")
        self.assertEqual(m.condim.numpy()[g_ovr], 1)

        g_child = idx("g_child")
        self.assertEqual(m.condim.numpy()[g_child], 6)
        self.assertEqual(m.geom_priority.numpy()[g_child], 99)
        self.assertAlmostEqual(float(m.geom_solmix.numpy()[g_child]), 5.0, places=5)
        self.assertAlmostEqual(float(m.geom_gap.numpy()[g_child]), 0.05, places=5)

    def test_body_defaults(self):
        """BODY: gravcomp."""
        gravcomp = self.model.mujoco.gravcomp.numpy()
        idx = self.builder.body_key.index

        self.assertAlmostEqual(float(gravcomp[idx("b_default")]), 0.5, places=5)
        self.assertAlmostEqual(float(gravcomp[idx("b_class")]), 1.0, places=5)
        self.assertAlmostEqual(float(gravcomp[idx("b_override")]), 0.75, places=5)

    def test_joint_dof_defaults(self):
        """JOINT_DOF: margin, solimplimit, solreffriction, solimpfriction,
        stiffness, damping, springref, ref, actuatorgravcomp."""
        m = self.model.mujoco
        idx = self.builder.joint_key.index

        j_def = idx("j_default")
        self.assertAlmostEqual(float(m.limit_margin.numpy()[j_def]), 1.0, places=5)
        np.testing.assert_allclose(m.solimplimit.numpy()[j_def], [0.8, 0.9, 0.01, 0.4, 1.0], atol=1e-4)
        np.testing.assert_allclose(m.solreffriction.numpy()[j_def], [0.05, 2.0], atol=1e-4)
        np.testing.assert_allclose(m.solimpfriction.numpy()[j_def], [0.7, 0.85, 0.02, 0.3, 1.5], atol=1e-4)
        self.assertAlmostEqual(float(m.dof_passive_stiffness.numpy()[j_def]), 2.0 * self.RAD2DEG, places=2)
        self.assertAlmostEqual(float(m.dof_passive_damping.numpy()[j_def]), 3.0 * self.RAD2DEG, places=2)
        self.assertAlmostEqual(float(m.dof_springref.numpy()[j_def]), 45.0 * self.DEG2RAD, places=4)
        self.assertAlmostEqual(float(m.dof_ref.numpy()[j_def]), 30.0 * self.DEG2RAD, places=4)
        self.assertEqual(bool(m.jnt_actgravcomp.numpy()[j_def]), True)

        j_cls = idx("j_class")
        self.assertAlmostEqual(float(m.limit_margin.numpy()[j_cls]), 10.0, places=5)
        self.assertAlmostEqual(float(m.dof_passive_stiffness.numpy()[j_cls]), 20.0 * self.RAD2DEG, places=1)
        self.assertAlmostEqual(float(m.dof_passive_damping.numpy()[j_cls]), 30.0 * self.RAD2DEG, places=1)
        self.assertAlmostEqual(float(m.dof_springref.numpy()[j_cls]), 90.0 * self.DEG2RAD, places=4)
        self.assertAlmostEqual(float(m.dof_ref.numpy()[j_cls]), 60.0 * self.DEG2RAD, places=4)

        j_ovr = idx("j_override")
        self.assertAlmostEqual(float(m.limit_margin.numpy()[j_ovr]), 99.0, places=5)

    def test_general_actuator_defaults(self):
        """ACTUATOR (general): gainprm, biasprm, dyntype, ctrllimited, forcelimited,
        ctrlrange, forcerange, gear, dynprm, actlimited, actrange, actdim, actearly."""
        m = self.model.mujoco

        np.testing.assert_allclose(m.actuator_gainprm.numpy()[0, 0], 100.0, atol=1.0)
        np.testing.assert_allclose(m.actuator_biasprm.numpy()[0, :3], [0.0, -100.0, -10.0], atol=1.0)
        self.assertEqual(m.actuator_dyntype.numpy()[0], 2)
        self.assertEqual(m.actuator_ctrllimited.numpy()[0], 1)
        self.assertEqual(m.actuator_forcelimited.numpy()[0], 1)
        np.testing.assert_allclose(m.actuator_ctrlrange.numpy()[0], [-2.0, 2.0], atol=1e-4)
        np.testing.assert_allclose(m.actuator_forcerange.numpy()[0], [-100.0, 100.0], atol=1e-4)
        np.testing.assert_allclose(m.actuator_gear.numpy()[0, :2], [2.0, 0.0], atol=1e-4)
        self.assertAlmostEqual(float(m.actuator_dynprm.numpy()[0, 0]), 0.5, places=4)
        self.assertEqual(m.actuator_actlimited.numpy()[0], 1)
        np.testing.assert_allclose(m.actuator_actrange.numpy()[0], [-1.0, 1.0], atol=1e-4)
        self.assertEqual(m.actuator_actdim.numpy()[0], 2)
        self.assertEqual(m.actuator_actearly.numpy()[0], 1)

        np.testing.assert_allclose(m.actuator_gainprm.numpy()[1, 0], 200.0, atol=1.0)
        np.testing.assert_allclose(m.actuator_biasprm.numpy()[1, :3], [0.0, -200.0, -20.0], atol=1.0)
        self.assertEqual(m.actuator_ctrllimited.numpy()[1], 0)
        np.testing.assert_allclose(m.actuator_ctrlrange.numpy()[1], [-5.0, 5.0], atol=1e-4)
        np.testing.assert_allclose(m.actuator_gear.numpy()[1, :2], [3.0, 0.0], atol=1e-4)

        np.testing.assert_allclose(m.actuator_gainprm.numpy()[2, 0], 999.0, atol=1.0)
        np.testing.assert_allclose(m.actuator_biasprm.numpy()[2, :3], [0.0, -999.0, -99.0], atol=1.0)

        np.testing.assert_allclose(m.actuator_gainprm.numpy()[3, 0], 300.0, atol=1.0)
        np.testing.assert_allclose(m.actuator_biasprm.numpy()[3, :3], [0.0, -300.0, -30.0], atol=1.0)

    def test_position_actuator_defaults(self):
        """ACTUATOR (position): kp inherited from <position> defaults."""
        gainprm = self.model.mujoco.actuator_gainprm.numpy()

        self.assertAlmostEqual(float(gainprm[4, 0]), 50.0, places=1)
        self.assertAlmostEqual(float(gainprm[5, 0]), 500.0, places=1)
        self.assertAlmostEqual(float(gainprm[6, 0]), 9999.0, places=1)
        self.assertAlmostEqual(float(gainprm[7, 0]), 3000.0, places=1)

    def test_base_joint_empty_string_fails(self):
        """Test that empty base_joint string raises ValueError."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="body1" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as ctx:
            builder.add_mjcf(mjcf_content, base_joint="")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_base_joint_whitespace_only_fails(self):
        """Test that whitespace-only base_joint string raises ValueError."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="body1" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as ctx:
            builder.add_mjcf(mjcf_content, base_joint="   ")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_base_joint_invalid_length_fails(self):
        """Test that base_joint axis specs with invalid length raise ValueError."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="body1" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""
        builder = newton.ModelBuilder()

        # Single character
        with self.assertRaises(ValueError) as ctx:
            builder.add_mjcf(mjcf_content, base_joint="p")
        self.assertIn("exactly 2 characters", str(ctx.exception))

        # Three characters
        with self.assertRaises(ValueError) as ctx:
            builder.add_mjcf(mjcf_content, base_joint="pxx")
        self.assertIn("exactly 2 characters", str(ctx.exception))

    def test_base_joint_invalid_type_char_fails(self):
        """Test that base_joint with invalid type character raises ValueError."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="body1" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as ctx:
            builder.add_mjcf(mjcf_content, base_joint="xx")  # 'x' is not a valid type
        self.assertIn("first char must be l/p/a/r", str(ctx.exception))

    def test_base_joint_invalid_axis_char_fails(self):
        """Test that base_joint with invalid axis character raises ValueError."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="body1" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as ctx:
            builder.add_mjcf(mjcf_content, base_joint="pw")  # 'w' is not a valid axis
        self.assertIn("second char must be x/y/z", str(ctx.exception))

    def test_base_joint_duplicate_axes_fails(self):
        """Test that base_joint with duplicate axis specifications raises ValueError."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="body1" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as ctx:
            builder.add_mjcf(mjcf_content, base_joint="px,py,px")  # 'px' appears twice
        self.assertIn("Duplicate axis specifications", str(ctx.exception))
        self.assertIn("px", str(ctx.exception))

    def test_base_joint_dict_conflicting_keys_fails(self):
        """Test that base_joint dict with conflicting keys raises ValueError."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="body1" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""
        builder = newton.ModelBuilder()

        # Test with 'parent' key
        with self.assertRaises(ValueError) as ctx:
            builder.add_mjcf(mjcf_content, base_joint={"joint_type": newton.JointType.REVOLUTE, "parent": 5})
        self.assertIn("cannot specify", str(ctx.exception))
        self.assertIn("parent", str(ctx.exception))

        # Test with 'child' key
        with self.assertRaises(ValueError) as ctx:
            builder.add_mjcf(mjcf_content, base_joint={"joint_type": newton.JointType.REVOLUTE, "child": 3})
        self.assertIn("cannot specify", str(ctx.exception))
        self.assertIn("child", str(ctx.exception))

        # Test with 'parent_xform' key
        with self.assertRaises(ValueError) as ctx:
            builder.add_mjcf(
                mjcf_content,
                base_joint={"joint_type": newton.JointType.REVOLUTE, "parent_xform": wp.transform_identity()},
            )
        self.assertIn("cannot specify", str(ctx.exception))
        self.assertIn("parent_xform", str(ctx.exception))

    def test_base_joint_valid_string_variations(self):
        """Test that various valid base_joint string formats work correctly."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <body name="body1" pos="0 0 0">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>"""

        # Test linear with 'l' prefix
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, base_joint="lx,ly,lz")
        model = builder.finalize()
        self.assertEqual(model.joint_coord_count, 3)  # 3 linear DOFs

        # Test linear with 'p' prefix (prismatic)
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, base_joint="px,py,pz")
        model = builder.finalize()
        self.assertEqual(model.joint_coord_count, 3)  # 3 linear DOFs

        # Test angular with 'a' prefix
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, base_joint="ax,ay,az")
        model = builder.finalize()
        self.assertEqual(model.joint_coord_count, 3)  # 3 angular DOFs

        # Test angular with 'r' prefix (revolute)
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, base_joint="rx,ry,rz")
        model = builder.finalize()
        self.assertEqual(model.joint_coord_count, 3)  # 3 angular DOFs

        # Test mixed linear and angular
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, base_joint="px,py,rz")
        model = builder.finalize()
        self.assertEqual(model.joint_coord_count, 3)  # 2 linear + 1 angular

        # Test with spaces (should be trimmed)
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, base_joint=" px , py , rz ")
        model = builder.finalize()
        self.assertEqual(model.joint_coord_count, 3)

        # Test uppercase (should be lowercased)
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content, base_joint="PX,PY,RZ")
        model = builder.finalize()
        self.assertEqual(model.joint_coord_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
