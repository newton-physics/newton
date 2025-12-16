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


if __name__ == "__main__":
    unittest.main(verbosity=2)
