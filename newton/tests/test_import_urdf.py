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
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry.types import GeoType
from newton.tests.unittest_utils import assert_np_equal

try:
    from resolve_robotics_uri_py import resolve_robotics_uri
except ImportError:
    resolve_robotics_uri = None

MESH_URDF = """
<robot name="mesh_test">
    <link name="base_link">
        <visual>
            <geometry>
                <mesh filename="{filename}" scale="1.0 1.0 1.0"/>
            </geometry>
            <origin xyz="1.0 2.0 3.0" rpy="0 0 0"/>
        </visual>
    </link>
</robot>
"""

MESH_OBJ = """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
v 0.0 1.0 1.0

# Front face
f 1 2 3
f 1 3 4
# Back face
f 5 7 6
f 5 8 7
# Right face
f 2 6 7
f 2 7 3
# Left face
f 1 4 8
f 1 8 5
# Top face
f 4 3 7
f 4 7 8
# Bottom face
f 1 5 6
f 1 6 2
"""

INERTIAL_URDF = """
<robot name="inertial_test">
    <link name="base_link">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="1.0" ixy="0.0" ixz="0.0"
                     iyy="1.0" iyz="0.0"
                     izz="1.0"/>
        </inertial>
        <visual>
            <geometry>
                <capsule radius="0.5" length="1.0"/>
            </geometry>
            <origin xyz="1.0 2.0 3.0" rpy="1.5707963 0 0"/>
        </visual>
    </link>
</robot>
"""

SPHERE_URDF = """
<robot name="sphere_test">
    <link name="base_link">
        <visual>
            <geometry>
                <sphere radius="0.5"/>
            </geometry>
            <origin xyz="1.0 2.0 3.0" rpy="0 0 0"/>
        </visual>
    </link>
</robot>
"""

SELF_COLLISION_URDF = """
<robot name="self_collision_test">
    <link name="base_link">
        <collision>
            <geometry><sphere radius="0.5"/></geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </collision>
    </link>
    <link name="far_link">
        <collision>
            <geometry><sphere radius="0.5"/></geometry>
            <origin xyz="1.0 0 0" rpy="0 0 0"/>
        </collision>
    </link>
</robot>
"""

JOINT_URDF = """
<robot name="joint_test">
<link name="base_link"/>
<link name="child_link"/>
<joint name="test_joint" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0 1.0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.23" upper="3.45"/>
</joint>
</robot>
"""

JOINT_TREE_URDF = """
<robot name="joint_tree_test">
<!-- Mixed ordering of links -->
<link name="grandchild_link_1b"/>
<link name="base_link"/>
<link name="child_link_1"/>
<link name="grandchild_link_2b"/>
<link name="grandchild_link_1a"/>
<link name="grandchild_link_2a"/>
<link name="child_link_2"/>

<!-- Level 1: Two joints from base_link -->
<joint name="joint_2" type="revolute">
<parent link="base_link"/>
<child link="child_link_2"/>
<origin xyz="1.0 0 0" rpy="0 0 0"/>
<axis xyz="0 0 1"/>
<limit lower="-1.23" upper="3.45"/>
</joint>

<joint name="joint_1" type="revolute">
<parent link="base_link"/>
<child link="child_link_1"/>
<origin xyz="0 1.0 0" rpy="0 0 0"/>
<axis xyz="0 0 1"/>
<limit lower="-1.23" upper="3.45"/>
</joint>

<!-- Level 2: Two joints from child_link_1 -->
<joint name="joint_1a" type="revolute">
<parent link="child_link_1"/>
<child link="grandchild_link_1a"/>
<origin xyz="0 0.5 0" rpy="0 0 0"/>
<axis xyz="0 0 1"/>
<limit lower="-1.23" upper="3.45"/>
</joint>

<joint name="joint_1b" type="revolute">
<parent link="child_link_1"/>
<child link="grandchild_link_1b"/>
<origin xyz="0.5 0 0" rpy="0 0 0"/>
<axis xyz="0 0 1"/>
<limit lower="-1.23" upper="3.45"/>
</joint>

<!-- Level 2: Two joints from child_link_2 -->
<joint name="joint_2b" type="revolute">
<parent link="child_link_2"/>
<child link="grandchild_link_2b"/>
<origin xyz="0.5 0 0" rpy="0 0 0"/>
<axis xyz="0 0 1"/>
<limit lower="-1.23" upper="3.45"/>
</joint>

<joint name="joint_2a" type="revolute">
<parent link="child_link_2"/>
<child link="grandchild_link_2a"/>
<origin xyz="0 0.5 0" rpy="0 0 0"/>
<axis xyz="0 0 1"/>
<limit lower="-1.23" upper="3.45"/>
</joint>
</robot>
"""


class TestImportUrdf(unittest.TestCase):
    @staticmethod
    def parse_urdf(urdf: str, builder: newton.ModelBuilder, res_dir: dict[str, str] | None = None, **kwargs):
        """Parse the specified URDF file from a directory of files.
        urdf: URDF file to parse
        res_dir: dict[str, str]: (filename, content): extra resources files to include in the directory"""

        # Default to up_axis="Y" if not specified in kwargs
        if "up_axis" not in kwargs:
            kwargs["up_axis"] = "Y"

        if not res_dir:
            builder.add_urdf(urdf, **kwargs)
            return

        urdf_filename = "robot.urdf"
        # Create a temporary directory to store files
        res_dir = res_dir or {}
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write all files to the temporary directory
            for filename, content in {urdf_filename: urdf, **res_dir}.items():
                file_path = Path(temp_dir) / filename
                with open(file_path, "w") as f:
                    f.write(content)

            # Parse the URDF file
            urdf_path = Path(temp_dir) / urdf_filename
            builder.add_urdf(str(urdf_path), **kwargs)

    def test_sphere_urdf(self):
        # load a urdf containing a sphere with r=0.5 and pos=(1.0,2.0,3.0)
        builder = newton.ModelBuilder()
        self.parse_urdf(SPHERE_URDF, builder)

        assert builder.shape_count == 1
        assert builder.shape_type[0] == newton.GeoType.SPHERE
        assert builder.shape_scale[0][0] == 0.5
        assert_np_equal(builder.shape_transform[0][:], np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))

    def test_mesh_urdf(self):
        # load a urdf containing a cube mesh with 8 verts and 12 faces
        for mesh_src in ("file", "http"):
            with self.subTest(mesh_src=mesh_src):
                builder = newton.ModelBuilder()
                if mesh_src == "file":
                    self.parse_urdf(MESH_URDF.format(filename="cube.obj"), builder, {"cube.obj": MESH_OBJ})
                else:

                    def mock_mesh_download(dst, url: str):
                        dst.write(MESH_OBJ.encode("utf-8"))

                    with patch("newton._src.utils.import_urdf._download_file", side_effect=mock_mesh_download):
                        self.parse_urdf(MESH_URDF.format(filename="http://example.com/cube.obj"), builder)

                assert builder.shape_count == 1
                assert builder.shape_type[0] == newton.GeoType.MESH
                assert_np_equal(builder.shape_scale[0], np.array([1.0, 1.0, 1.0]))
                assert_np_equal(builder.shape_transform[0][:], np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))
                assert builder.shape_source[0].vertices.shape[0] == 8
                assert builder.shape_source[0].indices.shape[0] == 3 * 12

    def test_inertial_params_urdf(self):
        builder = newton.ModelBuilder()
        self.parse_urdf(INERTIAL_URDF, builder, ignore_inertial_definitions=False)

        assert builder.shape_type[0] == newton.GeoType.CAPSULE
        assert builder.shape_scale[0][0] == 0.5
        assert builder.shape_scale[0][1] == 0.5  # half height
        assert_np_equal(
            np.array(builder.shape_transform[0][:]), np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]), tol=1e-6
        )

        # Check inertial parameters
        assert_np_equal(builder.body_mass[0], np.array([1.0]))
        assert_np_equal(builder.body_inertia[0], np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
        assert_np_equal(builder.body_com[0], np.array([0.0, 0.0, 0.0]))

    def test_self_collision_filtering_parameterized(self):
        for self_collisions in [False, True]:
            with self.subTest(enable_self_collisions=self_collisions):
                builder = newton.ModelBuilder()
                self.parse_urdf(SELF_COLLISION_URDF, builder, enable_self_collisions=self_collisions)

                assert builder.shape_count == 2

                # Check if collision filtering is applied correctly based on self_collisions setting
                filter_pair = (0, 1)
                if self_collisions:
                    self.assertNotIn(filter_pair, builder.shape_collision_filter_pairs)
                else:
                    self.assertIn(filter_pair, builder.shape_collision_filter_pairs)

    def test_revolute_joint_urdf(self):
        # Test a simple revolute joint with axis and limits
        builder = newton.ModelBuilder()
        self.parse_urdf(JOINT_URDF, builder)

        # Check joint was created with correct properties
        assert builder.joint_count == 2  # base joint + revolute
        assert builder.joint_type[-1] == newton.JointType.REVOLUTE

        assert_np_equal(builder.joint_limit_lower[-1], np.array([-1.23]))
        assert_np_equal(builder.joint_limit_upper[-1], np.array([3.45]))
        assert_np_equal(builder.joint_axis[-1], np.array([0.0, 0.0, 1.0]))

    def test_cartpole_urdf(self):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 123.0
        builder.default_shape_cfg.kd = 456.0
        builder.default_shape_cfg.mu = 789.0
        builder.default_joint_cfg.armature = 42.0
        urdf_filename = newton.examples.get_asset("cartpole.urdf")
        builder.add_urdf(
            urdf_filename,
            floating=False,
        )
        self.assertTrue(all(np.array(builder.shape_material_ke) == 123.0))
        self.assertTrue(all(np.array(builder.shape_material_kd) == 456.0))
        self.assertTrue(all(np.array(builder.shape_material_mu) == 789.0))
        self.assertTrue(all(np.array(builder.joint_armature) == 42.0))
        assert builder.body_count == 4

    def test_cylinder_shapes_preserved(self):
        """Test that cylinder geometries are properly imported as cylinders, not capsules."""
        # Create URDF content with cylinder geometry
        urdf_content = """
<robot name="cylinder_test">
    <link name="base_link">
        <collision>
            <geometry>
                <cylinder radius="0.5" length="2.0"/>
            </geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </collision>
        <visual>
            <geometry>
                <cylinder radius="0.5" length="2.0"/>
            </geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </visual>
    </link>
    <link name="second_link">
        <collision>
            <geometry>
                <capsule radius="0.3" height="1.0"/>
            </geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </collision>
    </link>
</robot>
"""

        builder = newton.ModelBuilder()
        builder.add_urdf(urdf_content)

        # Check shape types
        shape_types = list(builder.shape_type)

        # First shape should be cylinder (collision)
        self.assertEqual(shape_types[0], GeoType.CYLINDER)

        # Second shape should be cylinder (visual)
        self.assertEqual(shape_types[1], GeoType.CYLINDER)

        # Third shape should be capsule
        self.assertEqual(shape_types[2], GeoType.CAPSULE)

        # Check cylinder properties - radius and half_height
        # shape_scale stores (radius, half_height, 0) for cylinders
        shape_scale = builder.shape_scale[0]
        self.assertAlmostEqual(shape_scale[0], 0.5)  # radius
        self.assertAlmostEqual(shape_scale[1], 1.0)  # half_height (length/2)

    def test_joint_ordering_original(self):
        builder = newton.ModelBuilder()
        self.parse_urdf(JOINT_TREE_URDF, builder, bodies_follow_joint_ordering=False, joint_ordering=None)
        assert builder.body_count == 7
        assert builder.joint_count == 7
        assert builder.body_key == [
            "grandchild_link_1b",
            "base_link",
            "child_link_1",
            "grandchild_link_2b",
            "grandchild_link_1a",
            "grandchild_link_2a",
            "child_link_2",
        ]
        assert builder.joint_key == ["fixed_base", "joint_2", "joint_1", "joint_1a", "joint_1b", "joint_2b", "joint_2a"]

    def test_joint_ordering_dfs(self):
        builder = newton.ModelBuilder()
        self.parse_urdf(JOINT_TREE_URDF, builder, bodies_follow_joint_ordering=False, joint_ordering="dfs")
        assert builder.body_count == 7
        assert builder.joint_count == 7
        assert builder.body_key == [
            "grandchild_link_1b",
            "base_link",
            "child_link_1",
            "grandchild_link_2b",
            "grandchild_link_1a",
            "grandchild_link_2a",
            "child_link_2",
        ]
        assert builder.joint_key == ["fixed_base", "joint_2", "joint_2b", "joint_2a", "joint_1", "joint_1a", "joint_1b"]

    def test_joint_ordering_bfs(self):
        builder = newton.ModelBuilder()
        self.parse_urdf(JOINT_TREE_URDF, builder, bodies_follow_joint_ordering=False, joint_ordering="bfs")
        assert builder.body_count == 7
        assert builder.joint_count == 7
        assert builder.body_key == [
            "grandchild_link_1b",
            "base_link",
            "child_link_1",
            "grandchild_link_2b",
            "grandchild_link_1a",
            "grandchild_link_2a",
            "child_link_2",
        ]
        assert builder.joint_key == ["fixed_base", "joint_2", "joint_1", "joint_2b", "joint_2a", "joint_1a", "joint_1b"]

    def test_joint_body_ordering_original(self):
        builder = newton.ModelBuilder()
        self.parse_urdf(JOINT_TREE_URDF, builder, bodies_follow_joint_ordering=True, joint_ordering=None)
        assert builder.body_count == 7
        assert builder.joint_count == 7
        assert builder.body_key == [
            "base_link",
            "child_link_2",
            "child_link_1",
            "grandchild_link_1a",
            "grandchild_link_1b",
            "grandchild_link_2b",
            "grandchild_link_2a",
        ]
        assert builder.joint_key == ["fixed_base", "joint_2", "joint_1", "joint_1a", "joint_1b", "joint_2b", "joint_2a"]

    def test_joint_body_ordering_dfs(self):
        builder = newton.ModelBuilder()
        self.parse_urdf(JOINT_TREE_URDF, builder, bodies_follow_joint_ordering=True, joint_ordering="dfs")
        assert builder.body_count == 7
        assert builder.joint_count == 7
        assert builder.body_key == [
            "base_link",
            "child_link_2",
            "grandchild_link_2b",
            "grandchild_link_2a",
            "child_link_1",
            "grandchild_link_1a",
            "grandchild_link_1b",
        ]
        assert builder.joint_key == ["fixed_base", "joint_2", "joint_2b", "joint_2a", "joint_1", "joint_1a", "joint_1b"]

    def test_joint_body_ordering_bfs(self):
        builder = newton.ModelBuilder()
        self.parse_urdf(JOINT_TREE_URDF, builder, bodies_follow_joint_ordering=True, joint_ordering="bfs")
        assert builder.body_count == 7
        assert builder.joint_count == 7
        assert builder.body_key == [
            "base_link",
            "child_link_2",
            "child_link_1",
            "grandchild_link_2b",
            "grandchild_link_2a",
            "grandchild_link_1a",
            "grandchild_link_1b",
        ]
        assert builder.joint_key == ["fixed_base", "joint_2", "joint_1", "joint_2b", "joint_2a", "joint_1a", "joint_1b"]

    def test_xform_with_floating_false(self):
        """Test that xform parameter is respected when floating=False"""

        # Create a simple URDF with a link (no position/orientation in URDF for root link)
        urdf_content = """<?xml version="1.0" encoding="utf-8"?>
<robot name="test_xform">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <geometry>
                <sphere radius="0.1"/>
            </geometry>
        </visual>
    </link>
</robot>
"""
        # Create a non-identity transform to apply
        xform_pos = wp.vec3(5.0, 10.0, 15.0)
        xform_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 4.0)  # 45 degree rotation around Z
        xform = wp.transform(xform_pos, xform_quat)

        # Parse with floating=False and the xform
        # Use up_axis="Z" to match builder default and avoid axis transformation
        builder = newton.ModelBuilder()
        self.parse_urdf(urdf_content, builder, floating=False, xform=xform, up_axis="Z")
        model = builder.finalize()

        # Verify the model has a fixed joint
        self.assertEqual(model.joint_count, 1)
        joint_type = model.joint_type.numpy()[0]
        self.assertEqual(joint_type, newton.JointType.FIXED)

        # Verify the fixed joint has the correct parent_xform
        # In URDF, the xform is applied directly to the root body (no local transform)
        joint_X_p = model.joint_X_p.numpy()[0]

        # Expected transform is just xform (URDF root links don't have position/orientation)
        expected_xform = xform

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

    def test_floating_true_creates_free_joint(self):
        """Test that floating=True creates a free joint for the root body."""
        urdf_content = """<?xml version="1.0" encoding="utf-8"?>
<robot name="test_floating">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <geometry><sphere radius="0.1"/></geometry>
        </visual>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        self.parse_urdf(urdf_content, builder, floating=True, up_axis="Z")
        model = builder.finalize()

        # Verify the model has a free joint
        self.assertEqual(model.joint_count, 1)
        joint_type = model.joint_type.numpy()[0]
        self.assertEqual(joint_type, newton.JointType.FREE)
        self.assertEqual(builder.joint_key[0], "floating_base")

    def test_floating_false_creates_fixed_joint(self):
        """Test that floating=False creates a fixed joint for the root body."""
        urdf_content = """<?xml version="1.0" encoding="utf-8"?>
<robot name="test_fixed">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <geometry><sphere radius="0.1"/></geometry>
        </visual>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        self.parse_urdf(urdf_content, builder, floating=False, up_axis="Z")
        model = builder.finalize()

        # Verify the model has a fixed joint
        self.assertEqual(model.joint_count, 1)
        joint_type = model.joint_type.numpy()[0]
        self.assertEqual(joint_type, newton.JointType.FIXED)

    def test_base_joint_dict_creates_d6_joint(self):
        """Test that base_joint dict with linear and angular axes creates a D6 joint."""
        urdf_content = """<?xml version="1.0" encoding="utf-8"?>
<robot name="test_base_joint_dict">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <geometry><sphere radius="0.1"/></geometry>
        </visual>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        self.parse_urdf(
            urdf_content,
            builder,
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                ],
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0])],
            },
            up_axis="Z",
        )
        model = builder.finalize()

        # Verify the model has a D6 joint
        self.assertEqual(model.joint_count, 1)
        joint_type = model.joint_type.numpy()[0]
        self.assertEqual(joint_type, newton.JointType.D6)
        self.assertEqual(builder.joint_key[0], "base_joint")

    def test_base_joint_dict_creates_custom_joint(self):
        """Test that base_joint dict with JointType.REVOLUTE creates a revolute joint with custom axis."""
        urdf_content = """<?xml version="1.0" encoding="utf-8"?>
<robot name="test_base_joint_dict">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <geometry><sphere radius="0.1"/></geometry>
        </visual>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        self.parse_urdf(
            urdf_content,
            builder,
            base_joint={
                "joint_type": newton.JointType.REVOLUTE,
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0, 0, 1])],
            },
            up_axis="Z",
        )
        model = builder.finalize()

        # Verify the model has a revolute joint
        self.assertEqual(model.joint_count, 1)
        joint_type = model.joint_type.numpy()[0]
        self.assertEqual(joint_type, newton.JointType.REVOLUTE)
        self.assertEqual(builder.joint_key[0], "base_joint")

    def test_floating_and_base_joint_mutually_exclusive(self):
        """Test that specifying both base_joint and floating raises an error."""
        urdf_content = """<?xml version="1.0" encoding="utf-8"?>
<robot name="test_base_joint_override">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <geometry><sphere radius="0.1"/></geometry>
        </visual>
    </link>
</robot>
"""
        # Specifying both parameters should raise ValueError
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as cm:
            self.parse_urdf(
                urdf_content,
                builder,
                floating=True,
                base_joint={
                    "joint_type": newton.JointType.D6,
                    "linear_axes": [
                        newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                        newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                    ],
                },
                up_axis="Z",
            )
        self.assertIn("both 'floating' and 'base_joint'", str(cm.exception))

    def test_base_joint_dict_with_conflicting_parent(self):
        """Test that base_joint dict with 'parent' key raises an error."""
        urdf_content = """<?xml version="1.0" encoding="utf-8"?>
<robot name="test_base_joint_dict_parent">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <geometry><sphere radius="0.1"/></geometry>
        </visual>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        # Test with 'parent' key in base_joint dict
        with self.assertRaises(ValueError) as cm:
            self.parse_urdf(
                urdf_content,
                builder,
                base_joint={"joint_type": newton.JointType.REVOLUTE, "parent": 0},
                up_axis="Z",
            )
        self.assertIn("base_joint dict cannot specify", str(cm.exception))
        self.assertIn("parent", str(cm.exception))

        # Test with 'child' key
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as cm:
            self.parse_urdf(
                urdf_content,
                builder,
                base_joint={"joint_type": newton.JointType.REVOLUTE, "child": 0},
                up_axis="Z",
            )
        self.assertIn("base_joint dict cannot specify", str(cm.exception))
        self.assertIn("child", str(cm.exception))

        # Test with 'parent_xform' key
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as cm:
            self.parse_urdf(
                urdf_content,
                builder,
                base_joint={"joint_type": newton.JointType.REVOLUTE, "parent_xform": wp.transform_identity()},
                up_axis="Z",
            )
        self.assertIn("base_joint dict cannot specify", str(cm.exception))
        self.assertIn("parent_xform", str(cm.exception))

    def test_base_joint_respects_import_xform(self):
        """Test that base joints (parent == -1) correctly use the import xform.

            This is a regression test for a bug where root bodies with base_joint
            ignored the import xform parameter, using raw body pos/ori instead of
            the composed world_xform.

            Setup:
            - Root body at origin with no rotation
            - Import xform: translate by (10, 20, 30) and rotate 90° around Z
            - Using base_joint={
            "joint_type": newton.JointType.D6,
            "linear_axes": [
                newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0])
            ],
        } (D6 joint with linear axes)

            Expected final body transform after FK:
            - world_xform = import_xform * body_local_xform
            - Position should reflect import position
            - Orientation should reflect import rotation
        """
        urdf_content = """<?xml version="1.0"?>
<robot name="test_base_joint_xform">
    <link name="floating_body">
        <inertial>
            <origin xyz="1 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <origin xyz="1 0 0"/>
            <geometry><box size="0.2 0.2 0.2"/></geometry>
        </visual>
    </link>
</robot>
"""
        # Create import xform: translate + 90° Z rotation
        import_pos = wp.vec3(10.0, 20.0, 30.0)
        import_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2)  # 90° Z
        import_xform = wp.transform(import_pos, import_quat)

        # Use base_joint to create a D6 joint
        builder = newton.ModelBuilder()
        self.parse_urdf(
            urdf_content,
            builder,
            xform=import_xform,
            base_joint={
                "joint_type": newton.JointType.D6,
                "linear_axes": [
                    newton.ModelBuilder.JointDofConfig(axis=[1.0, 0.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 1.0, 0.0]),
                    newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0]),
                ],
            },
            up_axis="Z",
        )
        model = builder.finalize()

        # Verify body transform after forward kinematics
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_idx = model.body_key.index("floating_body")
        body_q = state.body_q.numpy()[body_idx]

        # Expected position: import_pos (URDF body is at origin, inertial offset doesn't affect body pos)
        # = (10, 20, 30)
        np.testing.assert_allclose(
            body_q[:3],
            [10.0, 20.0, 30.0],
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

    def test_parent_body_attaches_to_existing_body(self):
        """Test that parent_body attaches the URDF root to an existing body."""
        # First URDF: a simple robot arm
        robot_urdf = """<?xml version="1.0" encoding="utf-8"?>
<robot name="robot_arm">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual>
            <geometry><sphere radius="0.1"/></geometry>
        </visual>
    </link>
    <link name="end_effector">
        <inertial>
            <mass value="0.5"/>
            <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
        </inertial>
        <visual>
            <geometry><sphere radius="0.05"/></geometry>
        </visual>
    </link>
    <joint name="arm_joint" type="revolute">
        <parent link="base_link"/>
        <child link="end_effector"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14"/>
    </joint>
</robot>
"""
        # Second URDF: a gripper
        gripper_urdf = """<?xml version="1.0" encoding="utf-8"?>
<robot name="gripper">
    <link name="gripper_base">
        <inertial>
            <mass value="0.2"/>
            <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
        </inertial>
        <visual>
            <geometry><box size="0.05 0.05 0.02"/></geometry>
        </visual>
    </link>
</robot>
"""
        # First, load the robot
        builder = newton.ModelBuilder()
        self.parse_urdf(robot_urdf, builder, floating=False, up_axis="Z")

        # Get the end effector body index
        ee_body_idx = builder.body_key.index("end_effector")

        # Remember the body count before adding gripper
        robot_body_count = builder.body_count
        robot_joint_count = builder.joint_count

        # Now load the gripper attached to the end effector
        self.parse_urdf(gripper_urdf, builder, parent_body=ee_body_idx, up_axis="Z")

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
        robot_urdf = """<?xml version="1.0" encoding="utf-8"?>
<robot name="robot">
    <link name="base">
        <inertial><mass value="1.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    </link>
</robot>
"""
        gripper_urdf = """<?xml version="1.0" encoding="utf-8"?>
<robot name="gripper">
    <link name="gripper_base">
        <inertial><mass value="0.2"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        self.parse_urdf(robot_urdf, builder, floating=False, up_axis="Z")
        robot_body_idx = 0

        # Attach gripper with a D6 joint (rotation around Z)
        self.parse_urdf(
            gripper_urdf,
            builder,
            parent_body=robot_body_idx,
            base_joint={
                "joint_type": newton.JointType.D6,
                "angular_axes": [newton.ModelBuilder.JointDofConfig(axis=[0.0, 0.0, 1.0])],
            },
            up_axis="Z",
        )

        model = builder.finalize()

        # The second joint should be a D6 connecting to the robot body
        self.assertEqual(model.joint_count, 2)  # Fixed base + D6
        self.assertEqual(model.joint_type.numpy()[1], newton.JointType.D6)
        self.assertEqual(model.joint_parent.numpy()[1], robot_body_idx)

    def test_parent_body_creates_joint_to_parent(self):
        """Test that parent_body creates a joint connecting to the parent body."""
        robot_urdf = """<?xml version="1.0"?>
<robot name="robot">
    <link name="base_link">
        <inertial><mass value="1.0"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
        <visual><geometry><sphere radius="0.1"/></geometry></visual>
    </link>
    <link name="end_effector">
        <inertial><mass value="0.5"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
        <visual><geometry><sphere radius="0.05"/></geometry></visual>
    </link>
    <joint name="joint1" type="revolute">
        <parent link="base_link"/><child link="end_effector"/>
        <origin xyz="0 1 0"/><axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>
</robot>
"""
        gripper_urdf = """<?xml version="1.0"?>
<robot name="gripper">
    <link name="gripper_base">
        <inertial><mass value="0.2"/>
            <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
        <visual><geometry><box size="0.04 0.04 0.04"/></geometry></visual>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        self.parse_urdf(robot_urdf, builder, floating=False)

        ee_body_idx = builder.body_key.index("end_effector")
        initial_joint_count = builder.joint_count

        self.parse_urdf(gripper_urdf, builder, parent_body=ee_body_idx)

        # Verify a new joint was created connecting to the parent
        self.assertEqual(builder.joint_count, initial_joint_count + 1)
        self.assertEqual(builder.joint_parent[initial_joint_count], ee_body_idx)

        # Both should be in the same articulation
        model = builder.finalize()
        joint_articulation = model.joint_articulation.numpy()
        self.assertEqual(joint_articulation[0], joint_articulation[initial_joint_count])

    def test_non_sequential_articulation_attachment(self):
        """Test that attaching to a non-sequential articulation raises an error."""
        robot_urdf = """<?xml version="1.0"?>
<robot name="robot">
    <link name="base_link">
        <inertial><mass value="1.0"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
        <visual><geometry><sphere radius="0.1"/></geometry></visual>
    </link>
    <link name="link1">
        <inertial><mass value="0.5"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
        <visual><geometry><sphere radius="0.05"/></geometry></visual>
    </link>
    <joint name="joint1" type="revolute">
        <parent link="base_link"/><child link="link1"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>
</robot>
"""
        gripper_urdf = """<?xml version="1.0"?>
<robot name="gripper">
    <link name="gripper_base">
        <inertial><mass value="0.2"/>
            <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
        <visual><geometry><box size="0.04 0.04 0.04"/></geometry></visual>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        self.parse_urdf(robot_urdf, builder, floating=False)
        robot1_link_idx = builder.body_key.index("link1")

        # Add more robots to make robot1_link_idx not part of the most recent articulation
        self.parse_urdf(robot_urdf, builder, floating=False)
        self.parse_urdf(robot_urdf, builder, floating=False)

        # Attempting to attach to a non-sequential articulation should raise ValueError
        with self.assertRaises(ValueError) as cm:
            self.parse_urdf(gripper_urdf, builder, parent_body=robot1_link_idx, floating=False)
        self.assertIn("most recent", str(cm.exception))

    def test_floating_false_with_parent_body_succeeds(self):
        """Test that floating=False with parent_body is explicitly allowed."""
        robot_urdf = """<?xml version="1.0"?>
<robot name="robot">
    <link name="base_link">
        <inertial><mass value="1.0"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
        <visual><geometry><sphere radius="0.1"/></geometry></visual>
    </link>
    <link name="link1">
        <inertial><mass value="0.5"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
        <visual><geometry><sphere radius="0.05"/></geometry></visual>
    </link>
    <joint name="joint1" type="revolute">
        <parent link="base_link"/><child link="link1"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>
</robot>
"""
        gripper_urdf = """<?xml version="1.0"?>
<robot name="gripper">
    <link name="gripper_base">
        <inertial><mass value="0.2"/>
            <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
        <visual><geometry><box size="0.04 0.04 0.04"/></geometry></visual>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        self.parse_urdf(robot_urdf, builder, floating=False)
        link_idx = builder.body_key.index("link1")

        # Explicitly using floating=False with parent_body should succeed
        self.parse_urdf(gripper_urdf, builder, parent_body=link_idx, floating=False)
        model = builder.finalize()

        # Verify it worked - gripper should be attached
        self.assertIn("gripper_base", builder.body_key)
        self.assertEqual(len(model.articulation_start.numpy()) - 1, 1)  # Single articulation

    def test_floating_true_with_parent_body_raises_error(self):
        """Test that floating=True with parent_body raises an error."""
        robot_urdf = """<?xml version="1.0"?>
<robot name="robot">
    <link name="base_link">
        <inertial><mass value="1.0"/>
            <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
        <visual><geometry><sphere radius="0.1"/></geometry></visual>
    </link>
</robot>
"""
        gripper_urdf = """<?xml version="1.0"?>
<robot name="gripper">
    <link name="gripper_base">
        <inertial><mass value="0.2"/>
            <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
        <visual><geometry><box size="0.04 0.04 0.04"/></geometry></visual>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        self.parse_urdf(robot_urdf, builder, floating=False)
        base_idx = builder.body_key.index("base_link")

        # floating=True with parent_body should raise ValueError
        with self.assertRaises(ValueError) as cm:
            self.parse_urdf(gripper_urdf, builder, parent_body=base_idx, floating=True)
        self.assertIn("FREE joint", str(cm.exception))
        self.assertIn("parent", str(cm.exception))

    def test_parent_body_not_in_articulation_raises_error(self):
        """Test that attaching to a body not in any articulation raises an error."""
        builder = newton.ModelBuilder()

        # Create a standalone body (not in any articulation)
        standalone_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(
            body=standalone_body,
            radius=0.1,
        )

        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
    <link name="base_link">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
    </link>
</robot>
"""

        # Attempting to attach to standalone body should raise ValueError
        with self.assertRaises(ValueError) as cm:
            self.parse_urdf(urdf_content, builder, parent_body=standalone_body, floating=False)

        self.assertIn("not part of any articulation", str(cm.exception))

    def test_three_level_hierarchical_composition(self):
        """Test attaching multiple levels: arm → gripper → sensor."""
        arm_urdf = """<?xml version="1.0"?>
<robot name="arm">
    <link name="arm_base">
        <inertial><mass value="1.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    </link>
    <link name="arm_link">
        <inertial><mass value="0.5"/><inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/></inertial>
    </link>
    <link name="end_effector">
        <inertial><mass value="0.2"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
    </link>
    <joint name="joint1" type="revolute">
        <parent link="arm_base"/><child link="arm_link"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    </joint>
    <joint name="joint2" type="revolute">
        <parent link="arm_link"/><child link="end_effector"/>
        <origin xyz="0.5 0 0"/><axis xyz="0 0 1"/>
    </joint>
</robot>
"""
        gripper_urdf = """<?xml version="1.0"?>
<robot name="gripper">
    <link name="gripper_base">
        <inertial><mass value="0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    </link>
    <link name="gripper_finger">
        <inertial><mass value="0.05"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
    </link>
    <joint name="gripper_joint" type="revolute">
        <parent link="gripper_base"/><child link="gripper_finger"/>
        <origin xyz="0.05 0 0"/><axis xyz="0 1 0"/>
    </joint>
</robot>
"""
        sensor_urdf = """<?xml version="1.0"?>
<robot name="sensor">
    <link name="sensor_mount">
        <inertial><mass value="0.01"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    </link>
</robot>
"""

        builder = newton.ModelBuilder()

        # Level 1: Add arm
        self.parse_urdf(arm_urdf, builder, floating=False)
        ee_idx = builder.body_key.index("end_effector")

        # Level 2: Attach gripper to end effector
        self.parse_urdf(gripper_urdf, builder, parent_body=ee_idx, floating=False)
        finger_idx = builder.body_key.index("gripper_finger")

        # Level 3: Attach sensor to gripper finger
        self.parse_urdf(sensor_urdf, builder, parent_body=finger_idx, floating=False)

        model = builder.finalize()

        # All should be in ONE articulation
        self.assertEqual(len(model.articulation_start.numpy()) - 1, 1)

        # Verify joint count: arm (3) + gripper (2) + sensor (1) = 6
        self.assertEqual(model.joint_count, 6)

    def test_xform_relative_to_parent_body(self):
        """Test that xform is interpreted relative to parent_body when attaching."""
        robot_urdf = """<?xml version="1.0"?>
<robot name="robot">
    <link name="base">
        <inertial><mass value="1.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
        <visual><geometry><sphere radius="0.1"/></geometry></visual>
    </link>
    <link name="end_effector">
        <inertial><mass value="0.5"/><inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/></inertial>
        <visual><geometry><sphere radius="0.05"/></geometry></visual>
    </link>
    <joint name="joint1" type="revolute">
        <parent link="base"/><child link="end_effector"/>
        <origin xyz="0 1 0"/><axis xyz="0 0 1"/>
    </joint>
</robot>
"""
        gripper_urdf = """<?xml version="1.0"?>
<robot name="gripper">
    <link name="gripper_base">
        <inertial><mass value="0.2"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
        <visual><geometry><box size="0.02 0.02 0.02"/></geometry></visual>
    </link>
</robot>
"""
        builder = newton.ModelBuilder()
        self.parse_urdf(
            robot_urdf, builder, xform=wp.transform((0.0, 2.0, 0.0), wp.quat_identity()), floating=False, up_axis="Z"
        )

        ee_body_idx = builder.body_key.index("end_effector")

        # xform is in world coordinates, offset by +0.1 in Z (vertical up)
        self.parse_urdf(
            gripper_urdf,
            builder,
            parent_body=ee_body_idx,
            xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()),
            up_axis="Z",
        )

        gripper_body_idx = builder.body_key.index("gripper_base")

        # Finalize and compute forward kinematics to get world-space positions
        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q = state.body_q.numpy()
        ee_world_pos = body_q[ee_body_idx, :3]  # Extract x, y, z
        gripper_world_pos = body_q[gripper_body_idx, :3]  # Extract x, y, z

        # Verify gripper is offset by +0.1 in Z (world up direction)
        self.assertAlmostEqual(gripper_world_pos[0], ee_world_pos[0], places=5)
        self.assertAlmostEqual(gripper_world_pos[1], ee_world_pos[1], places=5)
        self.assertAlmostEqual(gripper_world_pos[2], ee_world_pos[2] + 0.1, places=5)

    def test_many_independent_articulations(self):
        """Test creating many (5) independent articulations and verifying indexing."""
        robot_urdf = """<?xml version="1.0"?>
<robot name="robot">
    <link name="base">
        <inertial><mass value="1.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    </link>
    <link name="link">
        <inertial><mass value="0.5"/><inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/></inertial>
    </link>
    <joint name="joint1" type="revolute">
        <parent link="base"/><child link="link"/>
        <origin xyz="0.5 0 0"/><axis xyz="0 0 1"/>
    </joint>
</robot>
"""

        builder = newton.ModelBuilder()

        # Add 5 independent robots
        for i in range(5):
            self.parse_urdf(
                robot_urdf,
                builder,
                xform=wp.transform(wp.vec3(float(i * 2), 0.0, 0.0), wp.quat_identity()),
                floating=False,
            )

        model = builder.finalize()

        # Should have 5 articulations
        self.assertEqual(len(model.articulation_start.numpy()) - 1, 5)

        # Each articulation has 2 joints (FIXED base + revolute)
        self.assertEqual(model.joint_count, 10)

    def test_base_joint_dict_conflicting_keys_fails(self):
        """Test that base_joint dict with conflicting keys raises ValueError."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test">
    <link name="body1">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
        </inertial>
        <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
    </link>
</robot>"""
        builder = newton.ModelBuilder()

        # Test with 'parent' key
        with self.assertRaises(ValueError) as ctx:
            self.parse_urdf(
                urdf_content, builder, base_joint={"joint_type": newton.JointType.REVOLUTE, "parent": 5}, up_axis="Z"
            )
        self.assertIn("cannot specify", str(ctx.exception))
        self.assertIn("parent", str(ctx.exception))

        # Test with 'child' key
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as ctx:
            self.parse_urdf(
                urdf_content, builder, base_joint={"joint_type": newton.JointType.REVOLUTE, "child": 3}, up_axis="Z"
            )
        self.assertIn("cannot specify", str(ctx.exception))
        self.assertIn("child", str(ctx.exception))

        # Test with 'parent_xform' key
        builder = newton.ModelBuilder()
        with self.assertRaises(ValueError) as ctx:
            self.parse_urdf(
                urdf_content,
                builder,
                base_joint={"joint_type": newton.JointType.REVOLUTE, "parent_xform": wp.transform_identity()},
                up_axis="Z",
            )
        self.assertIn("cannot specify", str(ctx.exception))
        self.assertIn("parent_xform", str(ctx.exception))


class TestUrdfUriResolution(unittest.TestCase):
    """Tests for URDF URI resolution functionality."""

    SIMPLE_URDF = '<robot name="r"><link name="base"><visual><geometry>{geo}</geometry></visual></link></robot>'
    MESH_GEO = '<mesh filename="{filename}"/>'
    SPHERE_GEO = '<sphere radius="0.5"/>'

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_package(self, name="my_robot", with_mesh=True):
        pkg = self.base_path / name
        (pkg / "urdf").mkdir(parents=True)
        if with_mesh:
            (pkg / "meshes").mkdir(parents=True)
            (pkg / "meshes" / "link.obj").write_text(MESH_OBJ)
        return pkg

    def test_package_uri_mesh_resolution(self):
        """Test package:// URI in mesh filename works with library and fallback."""
        pkg = self._create_package("my_robot")
        urdf = self.SIMPLE_URDF.format(geo=self.MESH_GEO.format(filename="package://my_robot/meshes/link.obj"))
        (pkg / "urdf" / "robot.urdf").write_text(urdf)

        with patch.dict(os.environ, {"ROS_PACKAGE_PATH": str(self.base_path)}):
            builder = newton.ModelBuilder()
            builder.add_urdf(str(pkg / "urdf" / "robot.urdf"), up_axis="Z")
            self.assertEqual(builder.shape_count, 1)
            self.assertEqual(builder.shape_type[0], GeoType.MESH)

    def test_package_uri_fallback_without_library(self):
        """Test package:// URI fallback when library is not available."""
        pkg = self._create_package("my_robot")
        urdf = self.SIMPLE_URDF.format(geo=self.MESH_GEO.format(filename="package://my_robot/meshes/link.obj"))
        (pkg / "urdf" / "robot.urdf").write_text(urdf)

        with patch("newton._src.utils.import_urdf.resolve_robotics_uri", None):
            builder = newton.ModelBuilder()
            builder.add_urdf(str(pkg / "urdf" / "robot.urdf"), up_axis="Z")
            self.assertEqual(builder.shape_count, 1)
            self.assertEqual(builder.shape_type[0], GeoType.MESH)

    @unittest.skipUnless(resolve_robotics_uri, "resolve-robotics-uri-py not installed")
    def test_source_uri_resolution(self):
        """Test package:// URI in source parameter works."""
        pkg = self._create_package("my_robot", with_mesh=False)
        urdf = self.SIMPLE_URDF.format(geo=self.SPHERE_GEO)
        (pkg / "urdf" / "robot.urdf").write_text(urdf)

        with patch.dict(os.environ, {"ROS_PACKAGE_PATH": str(self.base_path)}):
            builder = newton.ModelBuilder()
            builder.add_urdf("package://my_robot/urdf/robot.urdf", up_axis="Z")
            self.assertEqual(builder.body_count, 1)

    def test_uri_requires_library_or_warns(self):
        """Test that missing library raises/warns appropriately."""
        with patch("newton._src.utils.import_urdf.resolve_robotics_uri", None):
            builder = newton.ModelBuilder()

            # Source URI requires library - raises ImportError
            with self.assertRaises(ImportError) as cm:
                builder.add_urdf("package://pkg/robot.urdf", up_axis="Z")
            self.assertIn("resolve-robotics-uri-py", str(cm.exception))

            # model:// mesh URI warns
            urdf = self.SIMPLE_URDF.format(geo=self.MESH_GEO.format(filename="model://m/mesh.obj"))
            with self.assertWarns(UserWarning) as cm:
                builder.add_urdf(urdf, up_axis="Z")
            self.assertIn("resolve-robotics-uri-py", str(cm.warning))

    def test_unresolved_package_warning(self):
        """Test warning when package cannot be found."""
        urdf = self.SIMPLE_URDF.format(geo=self.MESH_GEO.format(filename="package://nonexistent/mesh.obj"))
        (self.base_path / "robot.urdf").write_text(urdf)

        builder = newton.ModelBuilder()
        with self.assertWarns(UserWarning) as cm:
            builder.add_urdf(str(self.base_path / "robot.urdf"), up_axis="Z")
        self.assertIn("could not resolve", str(cm.warning).lower())
        self.assertEqual(builder.shape_count, 0)

    @unittest.skipUnless(resolve_robotics_uri, "resolve-robotics-uri-py not installed")
    def test_automatic_vs_manual_resolution(self):
        """Test automatic resolution matches manual workaround from original ticket."""
        pkg = self._create_package("pkg")
        mesh_path = str(pkg / "meshes" / "link.obj")

        urdf_with_pkg_uri = """<robot name="r"><link name="base">
            <visual><geometry><mesh filename="package://pkg/meshes/link.obj"/></geometry></visual>
            <collision><geometry><mesh filename="package://pkg/meshes/link.obj"/></geometry></collision>
        </link></robot>"""
        (pkg / "urdf" / "robot.urdf").write_text(urdf_with_pkg_uri)

        urdf_resolved = f"""<robot name="r"><link name="base">
            <visual><geometry><mesh filename="{mesh_path}"/></geometry></visual>
            <collision><geometry><mesh filename="{mesh_path}"/></geometry></collision>
        </link></robot>"""

        with patch.dict(os.environ, {"ROS_PACKAGE_PATH": str(self.base_path)}):
            builder_manual = newton.ModelBuilder()
            builder_manual.add_urdf(urdf_resolved, up_axis="Z")

            builder_auto = newton.ModelBuilder()
            builder_auto.add_urdf("package://pkg/urdf/robot.urdf", up_axis="Z")

            self.assertEqual(builder_manual.shape_count, builder_auto.shape_count)
            self.assertEqual(builder_auto.shape_count, 2)


MIMIC_URDF = """
<robot name="mimic_test">
    <link name="base_link"/>
    <link name="leader_link"/>
    <link name="follower_link"/>

    <joint name="leader_joint" type="revolute">
        <parent link="base_link"/>
        <child link="leader_link"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57"/>
    </joint>

    <joint name="follower_joint" type="revolute">
        <parent link="base_link"/>
        <child link="follower_link"/>
        <origin xyz="1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14"/>
        <mimic joint="leader_joint" multiplier="2.0" offset="0.5"/>
    </joint>
</robot>
"""


class TestMimicConstraints(unittest.TestCase):
    """Tests for URDF mimic joint parsing."""

    def test_mimic_constraint_basic(self):
        """Test that mimic constraints are created from URDF mimic tags."""
        builder = newton.ModelBuilder()
        builder.add_urdf(MIMIC_URDF)
        model = builder.finalize()

        # Should have 1 mimic constraint
        self.assertEqual(model.constraint_mimic_count, 1)

        # Check the constraint values
        joint0 = model.constraint_mimic_joint0.numpy()[0]
        joint1 = model.constraint_mimic_joint1.numpy()[0]
        coef0 = model.constraint_mimic_coef0.numpy()[0]
        coef1 = model.constraint_mimic_coef1.numpy()[0]
        enabled = model.constraint_mimic_enabled.numpy()[0]

        # Find joint indices by name
        leader_idx = model.joint_key.index("leader_joint")
        follower_idx = model.joint_key.index("follower_joint")

        self.assertEqual(joint0, follower_idx)  # follower joint (joint0)
        self.assertEqual(joint1, leader_idx)  # leader joint (joint1)
        self.assertAlmostEqual(coef0, 0.5, places=5)
        self.assertAlmostEqual(coef1, 2.0, places=5)
        self.assertTrue(enabled)

    def test_mimic_constraint_default_values(self):
        """Test mimic constraints with default coef1 and coef0."""
        urdf = """
        <robot name="mimic_defaults">
            <link name="base"/>
            <link name="l1"/>
            <link name="l2"/>
            <joint name="j1" type="revolute">
                <parent link="base"/><child link="l1"/>
                <axis xyz="0 0 1"/><limit lower="-1" upper="1"/>
            </joint>
            <joint name="j2" type="revolute">
                <parent link="base"/><child link="l2"/>
                <axis xyz="0 0 1"/><limit lower="-1" upper="1"/>
                <mimic joint="j1"/>
            </joint>
        </robot>
        """
        builder = newton.ModelBuilder()
        builder.add_urdf(urdf)
        model = builder.finalize()

        self.assertEqual(model.constraint_mimic_count, 1)
        coef0 = model.constraint_mimic_coef0.numpy()[0]
        coef1 = model.constraint_mimic_coef1.numpy()[0]

        # Default values from URDF spec
        self.assertAlmostEqual(coef0, 0.0, places=5)
        self.assertAlmostEqual(coef1, 1.0, places=5)

    def test_mimic_joint_skipped_child_does_not_mismatch(self):
        """Regression test: skipped joints must not be included in name->index mapping."""

        class _SkippingLinkBuilder(newton.ModelBuilder):
            def add_link(self, *args, key=None, **kwargs):
                # Simulate a link filtered out by importer-side selection logic.
                if key == "skipped_link":
                    return -1
                return super().add_link(*args, key=key, **kwargs)

        urdf = """
        <robot name="mimic_skipped_child">
            <link name="base"/>
            <link name="leader_link"/>
            <link name="skipped_link"/>
            <link name="tail_link"/>
            <joint name="leader_joint" type="revolute">
                <parent link="base"/><child link="leader_link"/>
                <axis xyz="0 0 1"/><limit lower="-1" upper="1"/>
            </joint>
            <joint name="skipped_joint" type="revolute">
                <parent link="base"/><child link="skipped_link"/>
                <axis xyz="0 0 1"/><limit lower="-1" upper="1"/>
                <mimic joint="leader_joint"/>
            </joint>
            <joint name="tail_joint" type="revolute">
                <parent link="base"/><child link="tail_link"/>
                <axis xyz="0 0 1"/><limit lower="-1" upper="1"/>
            </joint>
        </robot>
        """

        builder = _SkippingLinkBuilder()
        with self.assertWarnsRegex(UserWarning, "was not created, skipping mimic constraint"):
            builder.add_urdf(urdf, joint_ordering=None, ensure_nonstatic_links=False)

        # No mimic constraint should be created because the follower joint was skipped.
        self.assertEqual(len(builder.constraint_mimic_joint0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
