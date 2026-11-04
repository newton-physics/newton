# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the independent native MJCF builder entry point."""

import builtins
import importlib.util
import os
import struct
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET
import zlib
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco import import_mjcf_native
from newton._src.utils.texture import normalize_texture
from newton.solvers import SolverMuJoCo, SolverXPBD
from newton.tests import test_import_mjcf
from newton.tests.test_mujoco_general_actuators import MJCF_SLIDERCRANK_ACTUATOR

MJCF_BALL_ACTUATORS = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="ball">
    <option gravity="0 0 0"/>
    <worldbody>
        <body name="link" pos="0 0 0">
            <joint name="bj" type="ball"/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
        </body>
    </worldbody>
    <actuator>
        <position name="p" joint="bj" kp="100" forcerange="-7 7" forcelimited="true" ctrlrange="-2 2" ctrllimited="true"/>
        <velocity name="v" joint="bj" kv="10" forcerange="-3 3" forcelimited="true" ctrlrange="-5 5" ctrllimited="true"/>
    </actuator>
</mujoco>
"""


def _create_png(path):
    """Write a red 1x1 RGB PNG without image-library dependencies."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    raw = zlib.compress(b"\x00\xff\x00\x00")
    idat_crc = zlib.crc32(b"IDAT" + raw)
    idat = struct.pack(">I", len(raw)) + b"IDAT" + raw + struct.pack(">I", idat_crc)
    iend_crc = zlib.crc32(b"IEND")
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    with open(path, "wb") as f:
        f.write(sig + ihdr + idat + iend)


def _assert_mesh_geometry(test, builder, vertices, faces=None, *, normals=None, uvs=None):
    test.assertEqual(builder.shape_count, 1)
    test.assertEqual(builder.shape_type[0], newton.GeoType.MESH)
    mesh = builder.shape_source[0]
    pose = builder.shape_transform[0]
    if builder.shape_body[0] >= 0:
        pose = builder.body_q[builder.shape_body[0]] * pose
    scale = np.asarray(builder.shape_scale[0])
    actual = np.array([wp.transform_point(pose, wp.vec3(p)) for p in np.asarray(mesh.vertices) * scale])
    distances = np.linalg.norm(actual[:, None] - np.asarray(vertices)[None, :], axis=-1)
    np.testing.assert_allclose(distances.min(axis=1), 0.0, atol=1e-5)
    vertex_ids = distances.argmin(axis=1)
    test.assertEqual(set(vertex_ids), set(range(len(vertices))))
    indices = np.asarray(mesh.indices)
    test.assertGreater(len(indices), 0)
    test.assertEqual(len(indices) % 3, 0)
    test.assertTrue(np.all((indices >= 0) & (indices < len(actual))))
    triangles = vertex_ids[indices].reshape(-1, 3)
    test.assertTrue(all(len(set(triangle)) == 3 for triangle in triangles))
    if faces is not None:
        # Cyclic permutations preserve winding; reversing a triangle does not.
        def canonical(triangle):
            return min(tuple(np.roll(triangle, offset)) for offset in range(3))

        test.assertCountEqual([canonical(t) for t in triangles], [canonical(t) for t in faces])
    if normals is not None:
        test.assertIsNotNone(mesh.normals)
        test.assertEqual(np.shape(mesh.normals), np.shape(mesh.vertices))
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
        actual_normals = (np.asarray(mesh.normals) / scale) @ np.asarray(wp.quat_to_matrix(pose.q)).reshape(3, 3).T
        actual_normals /= np.linalg.norm(actual_normals, axis=1, keepdims=True)
        np.testing.assert_allclose(actual_normals, np.asarray(normals)[vertex_ids], atol=1e-5)
    if uvs is not None:
        test.assertIsNotNone(mesh.uvs)
        np.testing.assert_allclose(mesh.uvs, np.asarray(uvs)[vertex_ids], atol=1e-6)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is required")
class TestImportMjcfNative(unittest.TestCase):
    def assert_roundtrip(self, xml, fields, **kwargs):
        """Compare native parameters after converting through the builder."""
        import mujoco

        reference = mujoco.MjModel.from_xml_string(xml)
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(xml, ctrl_direct=True, **kwargs)
        expected_warning = (
            self.assertWarnsRegex(newton.exceptions.NewtonWarning, "does not support actuator armature")
            if np.any(reference.actuator_armature)
            else nullcontext()
        )
        with expected_warning:
            solver = SolverMuJoCo(builder.finalize(), disable_contacts=True, kinematic_switchable_roots=False)
        for field in fields:
            with self.subTest(field=field):
                np.testing.assert_allclose(
                    getattr(solver.mj_model, field), getattr(reference, field), atol=3e-6, rtol=3e-5
                )
        return builder, solver, reference

    def test_independent_entry_points(self):
        """Keep legacy imports usable without importing the MuJoCo package."""
        original = builtins.__import__

        def without_mujoco(name, *args, **kwargs):
            if name == "mujoco" or name.startswith("mujoco."):
                raise ImportError("native dependency blocked")
            return original(name, *args, **kwargs)

        xml = '<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>'
        with patch("builtins.__import__", without_mujoco):
            legacy = newton.ModelBuilder()
            legacy.add_mjcf(xml)
            self.assertEqual(legacy.body_count, 1)
            with self.assertRaises(ImportError):
                newton.ModelBuilder().add_mjcf_native(xml)
        with patch("newton._src.utils.import_mjcf.parse_mjcf", side_effect=AssertionError("legacy parser called")):
            native = newton.ModelBuilder()
            native.add_mjcf_native(xml)
            self.assertEqual(native.body_count, 1)

    def test_slidercrank_nan_length(self):
        """Preserve MuJoCo's NaN warning behavior and legacy rejection."""
        import mujoco

        xml = MJCF_SLIDERCRANK_ACTUATOR.replace('cranklength="0.08"', 'cranklength="nan"')
        with self.assertRaisesRegex(ValueError, "cranklength must be positive"):
            newton.ModelBuilder().add_mjcf(xml, ctrl_direct=True)

        messages = []
        previous_warning = mujoco.get_mju_user_warning()
        mujoco.set_mju_user_warning(messages.append)
        try:
            reference = mujoco.MjModel.from_xml_string(xml)
            self.assertTrue(any("NaN" in message for message in messages))
            messages.clear()
            builder = newton.ModelBuilder()
            builder.add_mjcf_native(xml, ctrl_direct=True)
            self.assertTrue(any("NaN" in message for message in messages))
        finally:
            mujoco.set_mju_user_warning(previous_warning)

        model = builder.finalize()
        self.assertTrue(np.isnan(reference.actuator_cranklength[0]))
        self.assertTrue(np.isnan(model.mujoco.actuator_cranklength.numpy()[0]))

    def test_inline_mesh_preserves_texcoords_for_textures(self):
        """Preserve per-vertex UVs on a textured inline mesh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            texture_path = os.path.join(tmpdir, "texture.png")
            _create_png(texture_path)
            mjcf = f"""
<mujoco>
    <asset>
        <mesh name="tetra"
              vertex="0 0 0  1 0 0  0 1 0  0 0 1"
              face="0 1 2  0 3 1  0 2 3  1 3 2"
              texcoord="0 0  1 0  0 1  1 1"/>
        <texture name="texture" type="2d" file="{texture_path}"/>
        <material name="textured" texture="texture"/>
    </asset>
    <worldbody>
        <geom type="mesh" mesh="tetra" material="textured" mass="1"/>
    </worldbody>
</mujoco>
"""
            builder = newton.ModelBuilder()
            builder.add_mjcf_native(mjcf)

            _assert_mesh_geometry(
                self,
                builder,
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
                uvs=[[0, 1], [1, 1], [0, 0], [1, 0]],
            )
            texture = builder.shape_source[0].texture
            if isinstance(texture, str):
                self.assertEqual(texture, texture_path)
            else:
                np.testing.assert_array_equal(texture, [[[255, 0, 0]]])

    def test_native_mesh_materials(self):
        """Preserve material defaults, unset values, and RGB precedence over RGBA."""
        xml = """<mujoco><default><default class="surface">
            <material roughness=".2" metallic=".8"><layer role="rgb" texture="red"/></material>
            </default></default><asset>
            <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"
                face="0 1 2 0 3 1 0 2 3 1 3 2" texcoord="0 0 1 0 0 1 1 1"/>
            <texture name="red" type="2d" builtin="flat" width="1" height="1" rgb1="1 0 0"/>
            <texture name="blue" type="2d" builtin="flat" width="1" height="1" rgb1="0 0 1"/>
            <material name="legacy" texture="red"/>
            <material name="rgba" roughness="0" metallic="0"><layer role="rgba" texture="blue"/></material>
            <material name="mixed" roughness=".2" metallic=".8">
                <layer role="rgb" texture="red"/><layer role="rgba" texture="blue"/></material>
            <material name="inherited" class="surface"><layer role="rgba" texture="blue"/></material>
            <material name="unset" roughness="-1" metallic="-1"/>
            </asset><worldbody>
            <geom type="mesh" mesh="tetra" material="legacy"/>
            <geom type="mesh" mesh="tetra" material="rgba"/>
            <geom type="mesh" mesh="tetra" material="mixed"/>
            <geom type="mesh" mesh="tetra" material="inherited"/>
            <geom type="mesh" mesh="tetra" material="unset"/>
            </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(xml)
        for mesh, roughness, metallic, color in zip(
            builder.shape_source,
            (None, 0, 0.2, 0.2, None),
            (None, 0, 0.8, 0.8, None),
            ((255, 0, 0), (0, 0, 255), (255, 0, 0), (255, 0, 0), None),
            strict=True,
        ):
            with self.subTest(roughness=roughness, metallic=metallic, color=color):
                if roughness is None:
                    self.assertIsNone(mesh.roughness)
                    self.assertIsNone(mesh.metallic)
                else:
                    self.assertAlmostEqual(mesh.roughness, roughness)
                    self.assertAlmostEqual(mesh.metallic, metallic)
                if color is None:
                    self.assertIsNone(mesh.texture)
                else:
                    np.testing.assert_array_equal(mesh.texture, [[color]])

    def test_inline_mesh_applies_reference_pose_to_vertices_and_normals(self):
        """Apply the MJCF reference pose and scale to inline mesh data."""
        mjcf = """
<mujoco>
    <asset>
        <mesh name="tetra"
              vertex="1 0 0  2 0 0  1 1 0  1 0 1"
              face="0 2 1  0 1 3  0 3 2  1 2 3"
              normal="1 1 0  1 0 1  0 1 1  1 2 3"
              refpos="1 0 0"
              refquat="0.7071067811865476 0 0 0.7071067811865476"
              scale="2 3 4"/>
    </asset>
    <worldbody>
        <geom type="mesh" mesh="tetra" mass="1"/>
    </worldbody>
</mujoco>
"""
        expected_normals = np.array(
            [[0.5, -1.0 / 3.0, 0.0], [0.0, -1.0 / 3.0, 0.25], [0.5, 0.0, 0.25], [1.0, -1.0 / 3.0, 0.75]]
        )
        expected_normals /= np.linalg.norm(expected_normals, axis=1, keepdims=True)
        for shared, generate_faces in ((False, False), (True, False), (True, True)):
            with self.subTest(shared=shared, generate_faces=generate_faces):
                xml = mjcf
                normals = expected_normals
                if shared:
                    xml = xml.replace('normal="1 1 0  1 0 1  0 1 1  1 2 3"', 'normal="1 2 3"')
                    normals = np.tile(expected_normals[3], (4, 1))
                if generate_faces:
                    xml = xml.replace('face="0 2 1  0 1 3  0 3 2  1 2 3"', "")
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(xml)

                _assert_mesh_geometry(
                    self,
                    builder,
                    [[0.0, 0.0, 0.0], [0.0, -3.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
                    [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
                    normals=normals,
                )

    def test_free_ball_and_compound_joints(self):
        """Import free, ball, and co-located scalar joint groups."""
        for joints in (
            "<freejoint/>",
            '<joint type="ball" damping=".3" armature=".1"/>',
            '<joint name="slide" type="slide"/><joint name="hinge" axis="0 1 0"/>',
        ):
            with self.subTest(joints=joints):
                xml = f'<mujoco><worldbody><body pos=".1 .2 .3">{joints}<geom type="box" size=".1 .2 .3" mass="1"/></body></worldbody></mujoco>'
                self.assert_roundtrip(xml, ("jnt_type", "dof_armature", "dof_damping", "body_mass", "qpos0"))

    def test_compound_joints_share_first_anchor(self):
        """Match legacy anchors and kinematics even when scalar joints have different positions."""
        for second_type in ("hinge", "slide"):
            with self.subTest(second_type=second_type):
                xml = f'''<mujoco><worldbody><body name="compound" pos=".1 .2 .3" euler="20 30 40">
                    <joint name="first" axis="1 0 0" pos=".2 .3 .4"/>
                    <joint name="second" type="{second_type}" axis="0 1 0" pos="-.3 .1 .2"/>
                    <geom type="box" size=".1 .2 .3" mass="1"/>
                    </body></worldbody></mujoco>'''
                builders = []
                poses = []
                for importer in ("add_mjcf", "add_mjcf_native"):
                    builder = newton.ModelBuilder()
                    getattr(builder, importer)(xml)
                    self.assertEqual(builder.joint_count, 1)
                    self.assertEqual(builder.joint_type[0], newton.JointType.D6)
                    np.testing.assert_allclose(builder.joint_X_c[0].p, [0.2, 0.3, 0.4], atol=1e-6)
                    builders.append(builder)
                    model = builder.finalize()
                    state = model.state()
                    state.joint_q.assign([0.25, -0.4])
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    poses.append(state.body_q.numpy())
                for field in ("joint_X_p", "joint_X_c", "joint_axis"):
                    np.testing.assert_allclose(getattr(builders[0], field), getattr(builders[1], field), atol=1e-6)
                np.testing.assert_allclose(poses[0], poses[1], atol=1e-6)

    def test_reject_mixed_ball_joints_before_mutation(self):
        """Reject mixed ball/scalar joints without changing an existing builder."""
        import mujoco

        for joints in (
            '<joint name="ball" type="ball"/><joint name="slide" type="slide"/>',
            '<joint name="slide" type="slide"/><joint name="ball" type="ball"/>',
        ):
            with self.subTest(joints=joints):
                xml = f'<mujoco><worldbody><body name="mixed">{joints}<geom size=".1"/></body></worldbody></mujoco>'
                self.assertEqual(mujoco.MjModel.from_xml_string(xml).nv, 4)
                builder = newton.ModelBuilder()
                builder.add_body(mass=1)
                before = (builder.body_count, builder.joint_count, builder.shape_count, set(builder.custom_attributes))
                with self.assertRaisesRegex(NotImplementedError, "ball.*mixed"):
                    builder.add_mjcf_native(xml)
                self.assertEqual(
                    (builder.body_count, builder.joint_count, builder.shape_count, set(builder.custom_attributes)),
                    before,
                )

    def test_reject_incomplete_joint_conversion(self):
        """Detect missing joints and aliased DOFs in a converter's output."""
        xml = """<mujoco><worldbody><body name="compound"><joint name="x" type="slide" axis="1 0 0"/>
            <joint name="y" type="slide" axis="0 1 0"/><geom size=".1"/></body></worldbody></mujoco>"""
        convert = import_mjcf_native._add_joints
        for missing in (True, False):
            with self.subTest(missing=missing):

                def incomplete(*args, missing=missing):
                    joint, mapping = convert(*args)
                    if missing:
                        mapping.pop(1)
                    else:
                        mapping[1] = mapping[0]
                    return joint, mapping

                with patch.object(import_mjcf_native, "_add_joints", side_effect=incomplete):
                    with self.assertRaisesRegex(RuntimeError, "joint.*compound"):
                        newton.ModelBuilder().add_mjcf_native(xml)

    def test_reject_incomplete_actuator_conversion(self):
        """Detect an actuator row silently dropped during conversion."""
        xml = """<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
            <actuator><motor joint="j"/></actuator></mujoco>"""
        builder = newton.ModelBuilder()
        with patch.object(builder, "add_custom_values", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "actuator"):
                builder.add_mjcf_native(xml)

    def test_user_data_omission_warnings(self):
        """Warn when site or tendon user data is omitted."""
        for elements in (
            '<worldbody><site user="1 2"/></worldbody>',
            '<worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>'
            '<tendon><fixed user="1 2"><joint joint="j" coef="1"/></fixed></tendon>',
        ):
            with self.subTest(elements=elements):
                with self.assertWarnsRegex(UserWarning, "omits MuJoCo user data"):
                    newton.ModelBuilder().add_mjcf_native(f"<mujoco>{elements}</mujoco>")

    def test_structural_accounting_with_filters_and_root_override(self):
        """Account for deliberate filtering and replacement of root joints."""
        xml = """<mujoco><worldbody><body name="root"><freejoint name="root_joint"/><geom size=".1"/>
            <body name="kept"><joint name="kept_joint"/><geom size=".1"/></body>
            <body name="removed"><joint name="removed_joint"/><geom size=".1"/></body></body></worldbody>
            <actuator><motor joint="kept_joint"/><motor joint="removed_joint"/></actuator></mujoco>"""
        builder = newton.ModelBuilder()
        for _ in range(2):
            with self.assertWarnsRegex(UserWarning, "Actuator.*excluded element"):
                builder.add_mjcf_native(xml, floating=False, ignore_names=("removed",))
        self.assertEqual(builder.joint_dof_count, 2)
        self.assertEqual(builder._custom_frequency_counts["mujoco:actuator"], 2)
        solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
        self.assertEqual((solver.mj_model.nv, solver.mj_model.nu), (2, 2))

    def test_mesh_geom_type_defaults(self):
        """Match MuJoCo's fitted sphere default and explicit mesh/sphere geometry."""
        import mujoco

        with tempfile.TemporaryDirectory() as directory:
            mesh_path = Path(directory) / "box.stl"
            test_import_mjcf.TestImportMjcfGeometry._write_box_stl(mesh_path, hx=1.0, hy=1.0, hz=1.0)
            for geom_type in ("", 'type="mesh"', 'type="sphere"'):
                with self.subTest(geom_type=geom_type):
                    xml = f"""<mujoco><asset><mesh name="box" file="{mesh_path}"/></asset>
                        <worldbody><body name="b">
                            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                            <geom name="g" mesh="box" {geom_type}/>
                        </body></worldbody></mujoco>"""
                    reference = mujoco.MjModel.from_xml_string(xml)
                    data = mujoco.MjData(reference)
                    mujoco.mj_forward(reference, data)
                    builder = newton.ModelBuilder()
                    builder.add_mjcf_native(xml)
                    self.assertEqual(builder.shape_count, 1)
                    if geom_type == 'type="mesh"':
                        self.assertEqual(reference.geom_type[0], mujoco.mjtGeom.mjGEOM_MESH)
                        vertices = reference.mesh_vert @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0]
                        _assert_mesh_geometry(self, builder, vertices, reference.mesh_face)
                    else:
                        self.assertEqual(reference.geom_type[0], mujoco.mjtGeom.mjGEOM_SPHERE)
                        self.assertEqual(builder.shape_type[0], newton.GeoType.SPHERE)
                        self.assertAlmostEqual(builder.shape_scale[0][0], reference.geom_size[0, 0], places=6)
                        pose = builder.body_q[builder.shape_body[0]] * builder.shape_transform[0]
                        np.testing.assert_allclose(pose.p, data.geom_xpos[0], atol=1e-6)

    def test_mesh_and_heightfield(self):
        """Read native mesh buffers and normalized heightfield samples."""
        import mujoco

        xml = """<mujoco><asset><mesh name="tetra" scale="2 3 4" refpos=".1 .2 .3"
            vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
            <hfield name="h" nrow="2" ncol="3" size="1 2 .5 .1" elevation="0 1 2 3 4 5"/></asset>
            <worldbody><geom type="hfield" hfield="h"/><body><freejoint/>
            <geom type="mesh" mesh="tetra" mass="2"/></body></worldbody></mujoco>"""
        reference = mujoco.MjModel.from_xml_string(xml)
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(xml)
        self.assertEqual(builder.shape_count, 2)
        solver = SolverMuJoCo(builder.finalize())
        np.testing.assert_allclose(solver.mj_model.hfield_size, reference.hfield_size)
        np.testing.assert_allclose(builder.body_mass, [2])
        mesh = builder.shape_source[1]
        np.testing.assert_allclose(mesh.vertices, reference.mesh_vert[reference.mesh_face.reshape(-1)], atol=1e-6)
        np.testing.assert_allclose(builder.shape_transform[1].p, reference.geom_pos[1], atol=1e-6)
        np.testing.assert_allclose(np.asarray(builder.shape_source[0].data).reshape(-1), reference.hfield_data)
        builder.custom_attributes["mujoco:hfield_base"].values[0] = 0.25
        builder.shape_scale[0] = wp.vec3(2, 3, 4)
        edited = SolverMuJoCo(builder.finalize())
        np.testing.assert_allclose(edited.mj_model.hfield_size[0], [2, 6, 2, 1])

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "USD is required")
    def test_concave_mesh_usd_collision_parity(self):
        """Preserve concavity, inertia, and contact response across mesh coordinate frames."""
        if not wp.is_cuda_available():
            self.skipTest("Mesh SDF construction requires CUDA")

        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        polygon = [(0, 0), (2, 0), (2, 0.5), (0.5, 0.5), (0.5, 2), (0, 2)]
        vertices = np.array([(x, y, z) for z in (-0.3, 0.3) for x, y in polygon])
        cap = np.array([(0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5)])
        faces = [*cap[:, ::-1], *(cap + 6)]
        for i in range(6):
            j = (i + 1) % 6
            faces.extend(((i, j, j + 6), (i, j + 6, i + 6)))
        faces = np.asarray(faces)
        source_pose = wp.transform((3, -2, 1), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.4))
        vertices = np.array([wp.transform_point(source_pose, wp.vec3(p)) for p in vertices])
        shape_pose = wp.transform((0.1, 0.2, -0.1), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.2))
        body_pose = wp.transform((0.2, -0.3, 0.4), wp.quat_identity())
        world_pose = body_pose * shape_pose * source_pose
        world_vertices = np.array([wp.transform_point(body_pose * shape_pose, wp.vec3(p)) for p in vertices])
        center = (3.2, -1.6, 1.1)

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
        body = UsdGeom.Xform.Define(stage, "/body")
        body.AddTranslateOp().Set(Gf.Vec3d(*body_pose.p))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(2.0)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*center))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0, 1.3, 1.7))
        usd_mesh = UsdGeom.Mesh.Define(stage, "/body/mesh")
        usd_mesh.CreatePointsAttr(vertices.tolist())
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        usd_mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        usd_mesh.AddTranslateOp().Set(Gf.Vec3d(*shape_pose.p))
        usd_mesh.AddOrientOp().Set(Gf.Quatf(shape_pose.q[3], *shape_pose.q[:3]))
        UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(usd_mesh.GetPrim()).CreateApproximationAttr("none")
        xml = f"""<mujoco><asset><mesh name="l"
            vertex="{" ".join(map(str, vertices.reshape(-1)))}"
            face="{" ".join(map(str, faces.reshape(-1)))}"/></asset><worldbody>
            <body pos=".2 -.3 .4"><freejoint/>
            <inertial pos="{" ".join(map(str, center))}" mass="2" diaginertia="1 1.3 1.7"/>
            <geom type="mesh" mesh="l" pos=".1 .2 -.1"
            quat="{" ".join(map(str, [shape_pose.q[3], *shape_pose.q[:3]]))}"/>
            </body></worldbody></mujoco>"""

        for sdf_options in (
            {},
            {"max_resolution": 64, "margin": 0.02},
            {
                "target_voxel_size": 0.04,
                "narrow_band_range": (-0.08, 0.12),
                "texture_format": "float32",
                "margin": 0.03,
            },
        ):
            reference_velocity = None
            for importer in ("add_usd", "add_mjcf", "add_mjcf_native"):
                with self.subTest(sdf_options=sdf_options, importer=importer), wp.ScopedDevice("cuda:0"):
                    builder = newton.ModelBuilder(gravity=(0, 0, 0))
                    getattr(builder, importer)(stage if importer == "add_usd" else xml)
                    _assert_mesh_geometry(self, builder, world_vertices, faces)
                    self.assertEqual(builder.body_count, 1)
                    np.testing.assert_allclose(builder.body_mass, [2])
                    np.testing.assert_allclose(builder.body_com, [center])
                    np.testing.assert_allclose(
                        np.array(builder.body_inertia).reshape(-1, 3, 3), [np.diag([1, 1.3, 1.7])]
                    )
                    self.assertIsNone(builder.shape_source[0].sdf)
                    if sdf_options:
                        builder.shape_source[0].build_sdf(**sdf_options)
                    builder.shape_margin[0] = 0
                    builder.shape_gap[0] = 0.02
                    for name, value in (("ke", 1e4), ("kd", 100), ("mu", 0), ("mu_torsional", 0), ("mu_rolling", 0)):
                        getattr(builder, f"shape_material_{name}")[0] = value
                    cfg = newton.ModelBuilder.ShapeConfig(
                        density=0, margin=0, gap=0.02, mu=0, mu_torsional=0, mu_rolling=0
                    )
                    probe = newton.Mesh.create_box(0.1)
                    if sdf_options:
                        probe.build_sdf(max_resolution=32, margin=0.02)
                    # The first probe lies inside the hull but outside the concave surface.
                    for point in ((0.8, 0.8, 0), (1.2, 0.58, 0)):
                        builder.add_shape_mesh(
                            body=-1,
                            mesh=probe,
                            xform=wp.transform(wp.transform_point(world_pose, wp.vec3(point)), world_pose.q),
                            cfg=cfg,
                        )
                    model = builder.finalize()
                    self.assertEqual(model._shape_sdf_index.numpy()[0] >= 0, bool(sdf_options))
                    state, output = model.state(), model.state()
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    solver = SolverXPBD(model, iterations=4)
                    state.body_f.assign(np.array([[2, 4, 6, 1, 2.6, 5.1]], dtype=np.float32))
                    solver.step(state, output, model.control(), None, 0.01)
                    np.testing.assert_allclose(
                        output.body_qd.numpy(), [[0.01, 0.02, 0.03, 0.01, 0.02, 0.03]], atol=2e-5
                    )
                    state.clear_forces()
                    pipeline = newton.CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    self.assertGreater(count, 0)
                    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
                    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
                    points0 = contacts.rigid_contact_point0.numpy()[:count]
                    points1 = contacts.rigid_contact_point1.numpy()[:count]
                    normals = contacts.rigid_contact_normal.numpy()[:count]
                    expected_normal = np.array(wp.transform_vector(world_pose, wp.vec3(0, 1, 0)))
                    tolerance = 0.001 if sdf_options else 1e-5
                    for a, b, p0, p1, normal in zip(shape0, shape1, points0, points1, normals, strict=True):
                        self.assertEqual({a, b}, {0, 2})
                        world0 = np.array(wp.transform_point(body_pose, wp.vec3(p0))) if a == 0 else p0
                        world1 = np.array(wp.transform_point(body_pose, wp.vec3(p1))) if b == 0 else p1
                        self.assertAlmostEqual(float(np.dot(world1 - world0, normal)), -0.02, delta=tolerance)
                        alignment = np.dot(normal if a == 0 else -normal, expected_normal)
                        self.assertGreater(alignment, np.cos(np.deg2rad(2.0 if sdf_options else 0.1)))
                    solver.step(state, output, model.control(), contacts, 0.01)
                    velocity = output.body_qd.numpy()
                    if reference_velocity is None:
                        reference_velocity = velocity
                    else:
                        # Contact reduction and SDF voxelization depend on the mesh frame.
                        for start in (0, 3):
                            reference = reference_velocity[:, start : start + 3]
                            difference = velocity[:, start : start + 3] - reference
                            self.assertLess(np.linalg.norm(difference), 0.05 * np.linalg.norm(reference))

    def test_planar_visual_mesh(self):
        xml = """<mujoco><asset><mesh name="flat" inertia="shell"
            vertex="0 0 0 1 0 0 0 1 0 1 1 0" face="0 1 2 1 3 2"
            scale="2 3 4" refpos="1 2 3" refquat="0.7071067811865476 0 0 0.7071067811865476"/>
            </asset><worldbody><body><inertial pos="0 0 0" mass="2" diaginertia="1 1 1"/>
            <geom type="mesh" mesh="flat" contype="0" conaffinity="0"/></body></worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(xml)
        _assert_mesh_geometry(
            self, builder, [[-4, 3, -12], [-4, 0, -12], [-2, 3, -12], [-2, 0, -12]], [[0, 1, 2], [1, 3, 2]]
        )
        np.testing.assert_allclose(builder.body_mass, [2])
        np.testing.assert_allclose(np.asarray(builder.body_inertia).reshape(-1, 3, 3), [np.eye(3)])
        self.assertFalse(builder.shape_flags[0] & newton.ShapeFlags.COLLIDE_SHAPES)

    def test_sites_tendons_and_transmissions(self):
        """Preserve fixed/spatial tendon paths and tendon/site actuator targets."""
        xml = """<mujoco><worldbody><site name="anchor" pos="0 0 1"/>
            <body name="b"><joint name="j" type="slide" axis="0 0 1" range="-.2 .2"/>
            <geom name="g" size=".1" mass="1"/><site name="tip"/></body></worldbody>
            <tendon><fixed name="fixed" stiffness="3" damping=".2" range="-.3 .3"><joint joint="j" coef="2"/></fixed>
            <spatial name="spatial" stiffness="2"><site site="anchor"/><site site="tip"/></spatial></tendon>
            <actuator><motor tendon="fixed" gear="2"/><motor tendon="spatial"/>
            <general site="tip" refsite="anchor" gear="0 0 1 0 0 0"/></actuator></mujoco>"""
        self.assert_roundtrip(
            xml,
            (
                "tendon_stiffness",
                "tendon_damping",
                "tendon_lengthspring",
                "tendon_range",
                "wrap_type",
                "wrap_prm",
                "actuator_trntype",
                "actuator_gear",
                "site_pos",
            ),
        )

    def test_muscle_and_body_actuators(self):
        """Preserve muscle placeholders and body transmission parameters."""
        xml = """<mujoco><worldbody><body name="b"><joint name="j" type="slide" range="0 .3"/>
            <geom size=".1" mass="1"/></body></worldbody><actuator>
            <muscle joint="j" lengthrange=".1 .4"/><adhesion body="b" gain="3" ctrlrange="0 1"/>
            </actuator></mujoco>"""
        self.assert_roundtrip(
            xml, ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_trntype", "actuator_lengthrange")
        )

    def test_composed_muscle_placeholders(self):
        """Resolve muscle lengths and force scaling after attachment."""
        import mujoco

        compiler = '<compiler><lengthrange uselimit="true"/></compiler>'
        arm = '<joint name="j" type="slide" range=".1 .5"/><geom size=".1" mass="1"/>'
        hand = '<body name="hand"><geom size=".1" mass="2"/></body>'
        actuator = '<actuator><muscle joint="j"/></actuator>'
        reference = mujoco.MjModel.from_xml_string(
            f"<mujoco>{compiler}<worldbody><body>{arm}{hand}</body></worldbody>{actuator}</mujoco>"
        )
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(f"<mujoco>{compiler}<worldbody><body>{arm}</body></worldbody>{actuator}</mujoco>")
        builder.add_mjcf_native(
            f"<mujoco><worldbody>{hand}</worldbody></mujoco>", parent_body=0, parse_mujoco_options=False
        )
        solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
        for field in ("actuator_lengthrange", "actuator_gainprm", "actuator_acc0"):
            np.testing.assert_allclose(getattr(solver.mj_model, field), getattr(reference, field), rtol=1e-5, atol=1e-6)
        actual, expected = mujoco.MjData(solver.mj_model), mujoco.MjData(reference)
        for model, data in ((solver.mj_model, actual), (reference, expected)):
            data.qpos[:] = 0.2
            data.qvel[:] = 0.1
            data.act[:] = 0.5
            mujoco.mj_forward(model, data)
        np.testing.assert_allclose(actual.qfrc_actuator, expected.qfrc_actuator, rtol=1e-5)

    def test_contact_pairs_masks_and_excludes(self):
        """Preserve asymmetric masks, explicit pair parameters and exclusions."""
        xml = """<mujoco><worldbody><body name="a"><freejoint/><geom name="a" size=".1" contype="1" conaffinity="2"/></body>
            <body name="b" pos="1 0 0"><freejoint/><geom name="b" size=".1" contype="2" conaffinity="1"/></body>
            </worldbody><contact><pair geom1="a" geom2="b" friction=".2 .3 .01 .02 .03" solref=".01 1"/>
            <exclude body1="a" body2="b"/></contact></mujoco>"""
        self.assert_roundtrip(xml, ("pair_friction", "pair_solref", "pair_solimp", "geom_contype", "geom_conaffinity"))

    def test_equalities(self):
        """Preserve body and joint equalities with and without Newton conversion."""
        xml = """<mujoco><worldbody><body name="a"><joint name="a"/><geom size=".1"/></body>
            <body name="b" pos="1 0 0"><joint name="b"/><geom size=".1"/></body>
            <body name="c" pos="2 0 0"><joint/><geom size=".1"/></body></worldbody>
            <equality><connect body1="a" body2="b" anchor=".1 0 0"/>
            <weld body1="a" body2="c"/><joint joint1="a" joint2="b" polycoef=".1 2 0 0 0"/></equality></mujoco>"""
        for convert in (False, True):
            with self.subTest(convert=convert):
                self.assert_roundtrip(
                    xml, ("eq_type", "eq_data", "eq_solref", "eq_solimp"), convert_mjc_equality_constraints=convert
                )

    def test_filter_scale_options_and_attachment(self):
        """Apply builder filtering, scale, placement, and option settings."""
        xml = """<mujoco><option iterations="17" tolerance=".0002"/><default><default class="visual"/></default>
            <worldbody><body name="root"><freejoint/><geom type="box" size=".1 .2 .3" mass="2"/>
            <geom class="visual" size=".1" mass="1"/><site name="s" pos=".1 0 0"/>
            <body name="ignored"><geom size=".1"/></body></body></worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(
            xml,
            scale=2,
            parse_visuals=False,
            ignore_names=("ignored",),
            floating=False,
            xform=wp.transform((1, 2, 3), wp.quat_identity()),
        )
        self.assertEqual((builder.body_count, builder.shape_count), (1, 2))
        self.assertEqual(builder.joint_type[0], newton.JointType.FIXED)
        np.testing.assert_allclose(builder.body_q[0].p, [1, 2, 3])
        np.testing.assert_allclose(builder.shape_scale[0], [0.2, 0.4, 0.6])
        self.assertAlmostEqual(builder.body_mass[0], 3)
        self.assertEqual(builder.custom_attributes["mujoco:iterations"].values[0], 17)
        self.assertAlmostEqual(builder.custom_attributes["mujoco:tolerance"].values[0], 0.0002)

    def test_native_control_routing(self):
        """Map representable servos to joint targets and preserve other actuators."""
        import mujoco

        xml = """<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
            <actuator><position joint="j" kp="10" kv="2"/><velocity joint="j" kv="3"/>
            <position joint="j" kp="4" dampratio=".7"/><motor joint="j" gear="3"/></actuator></mujoco>"""
        for kind, dofs in (("hinge", 1), ("ball", 3)):
            with self.subTest(kind=kind):
                source = xml.replace('<joint name="j"/>', f'<joint name="j" type="{kind}"/>')
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(source)
                self.assertEqual(builder.joint_target_mode, [newton.JointTargetMode.POSITION] * dofs)
                self.assertEqual(builder.joint_target_ke, [10] * dofs)
                self.assertEqual(builder.joint_target_kd, [2] * dofs)
                self.assertEqual(builder.custom_attributes["mujoco:ctrl_source"].values, [0, 1, 1, 1])
                solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                reference = mujoco.MjModel.from_xml_string(source)
                indices = [0] * dofs + [1, 2, 3]
                for field in ("actuator_biasprm", "actuator_gainprm"):
                    np.testing.assert_allclose(
                        getattr(solver.mj_model, field), getattr(reference, field)[indices], rtol=3e-5
                    )

        for damping in ('kv="2"', 'dampratio=".7"'):
            for position_first in (False, True):
                with self.subTest(damping=damping, position_first=position_first):
                    position = f'<position joint="j" kp="10" {damping}/>'
                    velocity = '<velocity joint="j" kv="3"/>'
                    actuators = position + velocity if position_first else velocity + position
                    source = f'<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody><actuator>{actuators}</actuator></mujoco>'
                    builder = newton.ModelBuilder()
                    builder.add_mjcf_native(source)
                    self.assertEqual(builder.custom_attributes["mujoco:ctrl_source"].values, [0, 1])
                    expected_mode = (
                        newton.JointTargetMode.POSITION if position_first else newton.JointTargetMode.VELOCITY
                    )
                    self.assertEqual(builder.joint_target_mode, [expected_mode])

    def test_general_affine_gain(self):
        """Preserve affine gain terms even when the bias resembles a position servo."""
        xml = """<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
            <actuator><general joint="j" gaintype="affine" gainprm="10 2 3"
            biastype="affine" biasprm="0 -10 -2"/></actuator></mujoco>"""
        self.assert_roundtrip(xml, ("actuator_gaintype", "actuator_gainprm", "actuator_biasprm"))

    def test_contact_material_conversion(self):
        """Populate Newton contact gains while retaining raw MuJoCo parameters."""
        import mujoco

        for defaults, authored, expected in (
            ("", "", (2500.0, 100.0)),
            ("", 'solref=".04 .5"', (2500.0, 50.0)),
            ('<default><geom solref=".05 2"/></default>', "", (100.0, 40.0)),
            ("", 'solref="-1200 -35"', (1200.0, 35.0)),
        ):
            with self.subTest(defaults=defaults, authored=authored):
                xml = f'<mujoco>{defaults}<worldbody><body><joint/><geom size=".1" {authored}/></body></worldbody></mujoco>'
                builder = newton.ModelBuilder()
                builder.default_shape_cfg.ke = 123.0
                builder.default_shape_cfg.kd = 456.0
                builder.add_mjcf_native(xml)
                model = builder.finalize()
                np.testing.assert_allclose(model.shape_material_ke.numpy(), [expected[0]])
                np.testing.assert_allclose(model.shape_material_kd.numpy(), [expected[1]])
                reference = mujoco.MjModel.from_xml_string(xml)
                solver = SolverMuJoCo(model)
                np.testing.assert_allclose(solver.mj_model.geom_solref, reference.geom_solref)

    def test_imported_opacity_clamping(self):
        """Clamp geometry and material opacity using Newton's importer convention."""
        for source in ("geom", "material"):
            for opacity, expected, message in (
                (11, 1, "Clamping opacity"),
                (-2, 0, "Clamping opacity"),
                ("nan", 1, "Ignoring non-finite opacity"),
            ):
                with self.subTest(source=source, opacity=opacity):
                    rgba = f'rgba=".2 .3 .4 {opacity}"'
                    asset = f'<asset><material name="m" {rgba}/></asset>' if source == "material" else ""
                    attributes = 'material="m"' if source == "material" else rgba
                    xml = f'<mujoco>{asset}<worldbody><geom size=".1" {attributes}/></worldbody></mujoco>'
                    builder = newton.ModelBuilder()
                    with self.assertWarnsRegex(UserWarning, message):
                        builder.add_mjcf_native(xml)
                    self.assertEqual(builder.shape_opacity, [expected])
                    np.testing.assert_allclose(builder.shape_color, [[0.2, 0.3, 0.4]])

    def test_mesh_reference_pose_validation(self):
        """Reject invalid mesh reference poses before modifying an existing builder."""
        for attribute in (
            'refpos="nan 0 0"',
            'refpos="inf 0 0"',
            'refquat="nan 0 0 1"',
            'refquat="inf 0 0 1"',
            'refquat="0 0 0 0"',
        ):
            with self.subTest(attribute=attribute):
                xml = f"""<mujoco><asset><mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1" {attribute}/></asset>
                    <worldbody><body><geom type="mesh" mesh="tetra"/></body></worldbody></mujoco>"""
                builder = newton.ModelBuilder()
                builder.add_body(mass=1)
                before = (builder.body_count, builder.joint_count, builder.shape_count, set(builder.custom_attributes))
                with self.assertRaisesRegex(ValueError, "ref(pos|quat).*finite|zero quaternion"):
                    builder.add_mjcf_native(xml)
                self.assertEqual(
                    (builder.body_count, builder.joint_count, builder.shape_count, set(builder.custom_attributes)),
                    before,
                )

    def test_builtin_texture_fallback(self):
        """Tint primitive fallback colors while retaining mesh texture images."""
        for builtin, mean in (
            ("flat", [0.2, 0.3, 0.4]),
            ("checker", [0.15, 0.25, 0.35]),
            ("gradient", [0.15, 0.25, 0.35]),
        ):
            with self.subTest(builtin=builtin):
                xml = f'''<mujoco><asset>
                    <texture name="t" type="2d" builtin="{builtin}" rgb1=".2 .3 .4" rgb2=".1 .2 .3" width="8" height="8"/>
                    <material name="m" texture="t" rgba="1 1 .5 .7"/>
                    <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
                    </asset><worldbody><geom name="plain" size=".1" material="m"/>
                    <geom name="override" size=".1" material="m" rgba="0 1 0 .4"/>
                    <geom name="textured" type="mesh" mesh="tetra" material="m"/>
                    </worldbody></mujoco>'''
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(xml)
                np.testing.assert_allclose(builder.shape_color[0], np.array(mean) * [1, 1, 0.5])
                np.testing.assert_allclose(builder.shape_color[1], [0, 1, 0])
                np.testing.assert_allclose(builder.shape_color[2], [1, 1, 0.5])
                np.testing.assert_allclose(builder.shape_opacity, [0.7, 0.4, 0.7])
                self.assertEqual(builder.shape_source[2].texture.shape, (8, 8, 3))

    def test_nonlinear_equality_conversion_warning(self):
        """Warn only when generic mimic conversion approximates a nonlinear equality."""
        import mujoco

        for convert in (False, True):
            for nonlinear in (False, True):
                with self.subTest(convert=convert, nonlinear=nonlinear):
                    polycoef = ".1 2 .3 0 0" if nonlinear else ".1 2 0 0 0"
                    xml = f'''<mujoco><worldbody><body><joint name="a"/><geom size=".1"/>
                        <body><joint name="b"/><geom size=".1"/></body></body></worldbody>
                        <equality><joint name="couple" joint1="a" joint2="b" polycoef="{polycoef}"/></equality></mujoco>'''
                    builder = newton.ModelBuilder()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        builder.add_mjcf_native(xml, convert_mjc_equality_constraints=convert)
                    messages = [w for w in caught if "higher-order polycoef" in str(w.message)]
                    self.assertEqual(len(messages), int(convert and nonlinear))
                    solver = SolverMuJoCo(builder.finalize(), disable_contacts=True, kinematic_switchable_roots=False)
                    reference = mujoco.MjModel.from_xml_string(xml)
                    np.testing.assert_allclose(solver.mj_model.eq_data, reference.eq_data)

    def test_ball_actuator_torques(self):
        """Use Newton ball targets by default and preserve authored transmissions in direct mode."""
        import mujoco

        cases = (
            ((0, 1, 0), 0.2, (0, 1, -0.5), (0, 0)),
            ((1, 0, 0), 0.0, (1, 0, 0), (0, 0)),
            ((1, 0, 0), 0.05, (0.1, 0, 0), (0.02, 0.2)),
            ((0, 0, 1), 0.05, (0.2, -0.3, 0.1), (0.02, 0.2)),
            ((1, 0, 0), 0.2, (0, 0, 0), (10, -10)),
        )
        for direct, gear in product((False, True), ("1 0 0", "0 2 0")):
            with self.subTest(direct=direct, gear=gear):
                xml = MJCF_BALL_ACTUATORS.replace('joint="bj"', f'joint="bj" gear="{gear}"')
                root = ET.fromstring(xml)
                if not direct:
                    actuators = root.find("actuator")
                    originals = list(actuators)
                    actuators.clear()
                    for axis in range(3):
                        for actuator in originals:
                            attrs = dict(actuator.attrib)
                            attrs["name"] += f"_{axis}"
                            attrs["gear"] = " ".join(str(int(i == axis)) for i in range(3))
                            ET.SubElement(actuators, actuator.tag, attrs)
                reference = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(xml, ctrl_direct=direct)
                model = builder.finalize()
                solver = SolverMuJoCo(model, disable_contacts=True)
                self.assertEqual(solver.mj_model.nu, 2 if direct else 6)
                source = SolverMuJoCo.CtrlSource.CTRL_DIRECT if direct else SolverMuJoCo.CtrlSource.JOINT_TARGET
                np.testing.assert_array_equal(model.mujoco.ctrl_source.numpy(), source)
                for field in ("actuator_gear", "actuator_ctrlrange", "actuator_forcerange"):
                    np.testing.assert_allclose(getattr(solver.mj_model, field), getattr(reference, field))

                control = model.control()
                observables = solver.observables(kinds={solver.ObservableKind.QFRC_ACTUATOR})
                for axis, angle, velocity, controls in cases:
                    with self.subTest(axis=axis, angle=angle, velocity=velocity, controls=controls):
                        q = wp.quat_from_axis_angle(wp.vec3(*axis), angle)
                        expected = mujoco.MjData(reference)
                        expected.qpos[:] = np.array(q)[[3, 0, 1, 2]]
                        expected.qvel[:] = velocity
                        if direct:
                            expected.ctrl[:] = controls
                            control.mujoco.ctrl.assign(controls)
                        else:
                            target = wp.quat_from_axis_angle(wp.vec3(*axis), float(controls[0]))
                            angular_target = np.empty(3)
                            mujoco.mju_quat2Vel(angular_target, np.array(target)[[3, 0, 1, 2]].astype(float), 1.0)
                            velocity_target = wp.vec3(controls[1], -controls[1], 2 * controls[1])
                            expected.ctrl[0::2] = angular_target
                            expected.ctrl[1::2] = wp.quat_rotate(wp.quat_inverse(q), velocity_target)
                            control.joint_target_q.assign(np.array(target))
                            control.joint_target_qd.assign(np.array(velocity_target))
                        mujoco.mj_forward(reference, expected)

                        state = model.state()
                        state.joint_q.assign(np.array(q))
                        # Newton ball velocities use the parent-joint frame; MuJoCo uses the child frame.
                        state.joint_qd.assign(np.array(wp.quat_rotate(q, wp.vec3(*velocity))))
                        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                        state_out = model.state()
                        solver.step(state, state_out, control, None, dt=0.001, observables=observables)
                        # Observables rotate the last force into the post-step Newton frame.
                        output_rotation = wp.quat(*state_out.joint_q.numpy())
                        np.testing.assert_allclose(
                            observables.qfrc_actuator.numpy().reshape(-1),
                            np.array(wp.quat_rotate(output_rotation, wp.vec3(*expected.qfrc_actuator))),
                            atol=3e-6,
                        )

    def test_example_assets(self):
        """Match native parameters and forces for bundled articulated assets."""
        import mujoco

        assets = Path(newton.__file__).parent / "examples" / "assets"
        for name in ("nv_ant.xml", "nv_humanoid.xml"):
            with self.subTest(asset=name):
                source = assets / name
                spec = mujoco.MjSpec.from_file(str(source))
                for sensor in list(spec.sensors):
                    spec.delete(sensor)
                reference = spec.compile()
                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(UserWarning, "Native MJCF import omits"):
                    builder.add_mjcf_native(source, ctrl_direct=True)
                solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                for field in ("jnt_type", "jnt_range", "dof_damping", "dof_armature", "body_mass", "actuator_gear"):
                    np.testing.assert_allclose(
                        getattr(solver.mj_model, field), getattr(reference, field), atol=3e-6, rtol=3e-5
                    )
                actual, expected = mujoco.MjData(solver.mj_model), mujoco.MjData(reference)
                for native, data in ((solver.mj_model, actual), (reference, expected)):
                    data.qvel[:] = np.linspace(-0.2, 0.3, native.nv)
                    mujoco.mj_integratePos(native, data.qpos, data.qvel, 0.1)
                    data.ctrl[:] = np.linspace(-0.3, 0.4, native.nu)
                    mujoco.mj_forward(native, data)
                for field in ("xpos", "qfrc_actuator", "qfrc_passive", "qfrc_bias"):
                    np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=3e-5, rtol=5e-5)

    def test_scaled_tendon_and_slider_ranges(self):
        """Scale spatial lengths while preserving automatic-length sentinels."""
        xml = """<mujoco><worldbody><site name="anchor" pos="0 0 1"/>
            <body><joint name="j" type="slide" range="-.1 .2"/><geom size=".1"/><site name="tip"/></body>
            </worldbody><tendon><spatial springlength=".2 .3" range=".1 2" margin=".01">
            <site site="anchor"/><site site="tip"/></spatial></tendon>
            <actuator><position joint="j" kp="10" ctrlrange="-.1 .2"/></actuator></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(xml, scale=2, ctrl_direct=True)
        solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
        np.testing.assert_allclose(solver.mj_model.jnt_range, [[-0.2, 0.4]])
        np.testing.assert_allclose(solver.mj_model.actuator_ctrlrange, [[-0.2, 0.4]])
        np.testing.assert_allclose(solver.mj_model.tendon_lengthspring, [[0.4, 0.6]])
        np.testing.assert_allclose(solver.mj_model.tendon_range, [[0.2, 4]])

    def test_builder_copy_across_worlds(self):
        """Remap all actuator target types and retain deferred parameters across copies."""
        xml = """<mujoco><worldbody><site name="anchor" pos="0 0 1"/>
            <body name="b"><joint name="j"/><geom size=".1" mass="1"/><site name="tip"/></body></worldbody>
            <tendon><fixed name="t" limited="true" range="-1 2"><joint joint="j" coef="1"/></fixed></tendon>
            <actuator><position joint="j" kp="10" dampratio=".7"/>
            <motor jointinparent="j"/><position tendon="t" kp="2" inheritrange=".8"/>
            <general site="tip" refsite="anchor"/><adhesion body="b" gain="1" ctrlrange="0 1"/>
            <general cranksite="tip" slidersite="anchor" cranklength="2"/></actuator></mujoco>"""
        asset = newton.ModelBuilder()
        asset.add_mjcf_native(xml)
        builder = newton.ModelBuilder()
        for _ in range(2):
            builder.begin_world()
            builder.add_builder(asset)
            builder.end_world()
        model = builder.finalize()
        solver = SolverMuJoCo(model, disable_contacts=True)
        self.assertEqual(solver.mjw_data.nworld, 2)
        self.assertEqual(builder.custom_attributes["mujoco:actuator_world"].values, [0] * 6 + [1] * 6)
        np.testing.assert_array_equal(
            model.mujoco.actuator_trnid.numpy(),
            [[0, -1], [0, -1], [0, -1], [2, 1], [0, -1], [2, 1], [1, -1], [1, -1], [1, -1], [5, 4], [1, -1], [5, 4]],
        )
        np.testing.assert_array_equal(
            solver.mj_model.actuator_trnid,
            [[0, -1], [0, -1], [0, -1], [1, 0], [1, -1], [1, 0]],
        )
        np.testing.assert_allclose(model.mujoco.actuator_ctrlrange.numpy()[[2, 8]], [[-0.7, 1.7]] * 2)
        self.assertAlmostEqual(builder.custom_attributes["mujoco:actuator_biasprm"].values[6][2], 0.7, places=6)

    def test_native_option_flags(self):
        """Preserve native flags and contact override parameters."""
        import mujoco

        xml = """<mujoco><option actuatorgroupdisable="2" o_margin=".01" o_solref=".03 1" o_friction=".2 .3 .4 .5 .6">
            <flag gravity="disable" warmstart="disable" energy="enable"/></option>
            <worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
            <actuator><motor joint="j" group="2"/></actuator></mujoco>"""
        reference = mujoco.MjModel.from_xml_string(xml)
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(xml)
        solver = SolverMuJoCo(builder.finalize())
        self.assertEqual(solver.mj_model.opt.enableflags, reference.opt.enableflags)
        self.assertEqual(solver.mj_model.opt.disableflags & reference.opt.disableflags, reference.opt.disableflags)
        for field in ("disableactuator", "o_margin", "o_solref", "o_solimp", "o_friction"):
            np.testing.assert_allclose(getattr(solver.mj_model.opt, field), getattr(reference.opt, field))
        omitted = newton.ModelBuilder()
        omitted.add_mjcf_native(xml, parse_mujoco_options=False)
        self.assertEqual(SolverMuJoCo(omitted.finalize()).mj_model.opt.enableflags, 0)

    def test_native_option_flags_unsupported_by_solver(self):
        """Preserve valid flags while exposing the solver's Warp dependency in both modes."""
        import mujoco

        for name in ("override", "fwdinv", "diagexact", "midphase", "autoreset", "ipc"):
            with self.subTest(flag=name):
                value = "disable" if name in ("midphase", "autoreset") else "enable"
                options = 'integrator="discrete" solver="CG"' if name == "ipc" else ""
                xml = f"""<mujoco><option {options}><flag {name}="{value}" multiccd="disable"/></option>
                    <worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>"""
                reference = mujoco.MjModel.from_xml_string(xml)
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(xml)
                model = builder.finalize()
                self.assertEqual(model.mujoco.native_option_enableflags.numpy()[0], reference.opt.enableflags)
                self.assertEqual(model.mujoco.native_option_disableflags.numpy()[0], reference.opt.disableflags)
                for use_cpu in (False, True):
                    with self.subTest(use_mujoco_cpu=use_cpu):
                        with self.assertRaisesRegex(NotImplementedError, name.upper()):
                            SolverMuJoCo(model, use_mujoco_cpu=use_cpu)

    def test_native_geometry_parameters(self):
        """Preserve fluid forces, surface motion, and adhesion."""
        xml = """<mujoco><option density="1.2" viscosity=".01" wind="1 2 3"/>
            <worldbody><body><freejoint/><geom type="box" size=".1 .2 .3" mass="1"
            fluidshape="ellipsoid" fluidcoef=".2 .3 .4 .5 .6" surfacevel="1 2 3 0 0 1" adhesion=".4"/>
            </body></worldbody></mujoco>"""
        for moving in (True, False):
            with self.subTest(moving=moving):
                self.assert_roundtrip(
                    xml if moving else xml.replace("<freejoint/>", ""),
                    ("geom_fluid", "geom_surfacevel", "geom_adhesion"),
                )

    def test_native_actuator_history_and_dcmotor(self):
        """Preserve actuator history, passive properties, and DC-motor inputs."""
        for actuator in (
            '<general joint="j" delay=".02" nsample="3" interp="linear" group="2"/>',
            '<dcmotor joint="j" motorconst=".2" resistance="2"/>',
            '<general joint="j" damping=".7 .2 .1" armature=".02"/>',
            '<dcmotor joint="j" motorconst=".2" resistance="2" input="pos" controller="5 1 .2 10 2 3" damping=".007 .002 .001"/>',
            '<dcmotor joint="j" motorconst=".2" resistance="2" input="vel" controller="0 0 .2 10 0 3"/>',
        ):
            with self.subTest(actuator=actuator):
                xml = f'<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody><actuator>{actuator}</actuator></mujoco>'
                self.assert_roundtrip(
                    xml,
                    (
                        "actuator_delay",
                        "actuator_history",
                        "actuator_historyadr",
                        "actuator_group",
                        "actuator_gainprm",
                        "actuator_biasprm",
                        "actuator_dynprm",
                        "actuator_dyntype",
                        "actuator_ctrlspec",
                        "actuator_damping",
                        "actuator_dampingpoly",
                        "actuator_armature",
                    ),
                )

    def test_native_joint_target_history(self):
        """Match delayed commands after interpolation, buffer wrap, and ball expansion."""
        import mujoco

        ball = '<position joint="ball" kp="4" delay=".0015" nsample="4" interp="cubic"/>'
        xml = f"""<mujoco><compiler angle="radian"/><option gravity="0 0 0" timestep=".001" integrator="Euler"/>
            <worldbody><body><joint name="j" ref=".2"/>
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>
            <body><joint name="ball" type="ball"/>
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body></worldbody>
            <actuator><velocity joint="j" kv="3" delay=".00325" nsample="4" interp="zoh"/>
            <position joint="j" kp="10" delay=".0025" nsample="4" interp="linear"/>{ball}</actuator></mujoco>"""
        expanded = xml.replace(
            ball, "".join(ball.replace("/>", f' gear="{gear}"/>') for gear in ("1 0 0", "0 1 0", "0 0 1"))
        )
        reference = mujoco.MjModel.from_xml_string(expanded)
        for direct, cpu in product((False, True), (False, True)):
            with self.subTest(direct=direct, cpu=cpu):
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(expanded if direct else xml, ctrl_direct=direct)
                model = builder.finalize()
                np.testing.assert_array_equal(model.mujoco.ctrl_source.numpy(), int(direct))
                solver = SolverMuJoCo(model, use_mujoco_cpu=cpu, disable_contacts=True)
                order = [0, 1, 2, 3, 4] if direct else [1, 0, 2, 3, 4]
                for field in ("actuator_delay", "actuator_history"):
                    np.testing.assert_allclose(getattr(solver.mj_model, field), getattr(reference, field)[order])
                expected = mujoco.MjData(reference)
                expected.qvel[:] = [0.4, 0.1, -0.2, 0.3]
                state, state_out = model.state(), model.state()
                state.joint_qd.assign(expected.qvel)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                control = model.control()
                observables = solver.observables(kinds={solver.ObservableKind.QFRC_ACTUATOR})
                for step in range(16):
                    position, velocity = (0.1, -0.3) if step % 5 < 2 else (-0.2, 0.5)
                    angular = np.array([0.1, -0.2, 0.3]) * (1 if step % 3 else -1)
                    target = wp.quat_from_axis_angle(
                        wp.vec3(*angular / np.linalg.norm(angular)), float(np.linalg.norm(angular))
                    )
                    expected.ctrl[:] = [velocity, position + 0.2, *angular]
                    if direct:
                        control.mujoco.ctrl.assign(expected.ctrl)
                    else:
                        control.joint_target_q.assign([position, *target])
                        control.joint_target_qd.assign([velocity, 0, 0, 0])
                    mujoco.mj_step(reference, expected)
                    solver.step(state, state_out, control, None, 0.001, observables=observables)
                    output = state_out.joint_q.numpy()
                    np.testing.assert_allclose(output[0] + 0.2, expected.qpos[0], atol=2e-6)
                    np.testing.assert_allclose(output[1:][[3, 0, 1, 2]], expected.qpos[1:], atol=2e-6)
                    forces = [
                        expected.qfrc_actuator[0],
                        *wp.quat_rotate(wp.quat(*output[1:]), wp.vec3(*expected.qfrc_actuator[1:])),
                    ]
                    np.testing.assert_allclose(
                        observables.qfrc_actuator.numpy().reshape(-1), forces, rtol=2e-5, atol=2e-6
                    )
                    state, state_out = state_out, state

    def test_native_dcmotor_control_ranges(self):
        """Scale linear position and velocity controls while retaining voltage units."""
        for signature, code, factor in (("pos", 1, 2), ("vel", 2, 2), ("voltage", 8, 1)):
            with self.subTest(signature=signature):
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(
                    f'''<mujoco><worldbody><body><joint name="j" type="slide"/><geom size=".1"/></body></worldbody>
                    <actuator><dcmotor joint="j" motorconst=".2" resistance="2" input="{signature}"
                    ctrlrange="-1 3"/></actuator></mujoco>''',
                    scale=2,
                )
                self.assertEqual(builder.custom_attributes["mujoco:actuator_ctrlspec"].values[0], code)
                np.testing.assert_allclose(
                    builder.custom_attributes["mujoco:actuator_ctrlrange"].values[0], np.array([-1, 3]) * factor
                )
        for signature in ("ff", "none", "pos vel"):
            with self.subTest(signature=signature):
                builder = newton.ModelBuilder()
                with self.assertRaisesRegex(NotImplementedError, "one DC-motor control input"):
                    builder.add_mjcf_native(
                        f'''<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
                        <actuator><dcmotor joint="j" motorconst=".2" resistance="2" input="{signature}"/></actuator></mujoco>'''
                    )
                self.assertEqual((builder.body_count, builder.joint_count, builder.shape_count), (0, 0, 0))

    def test_tendon_equalities(self):
        """Retain tendon coupling through builder copies and property updates."""
        import mujoco

        xml = """<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
            <tendon><fixed name="t1"><joint joint="j" coef="1"/></fixed>
            <fixed name="t2"><joint joint="j" coef="2"/></fixed></tendon>
            <equality><tendon tendon1="t1" tendon2="t2" polycoef=".1 .2 .3 .4 .5"/>
            <tendon tendon1="t2" polycoef=".2 .3 0 0 0"/></equality></mujoco>"""
        reference = mujoco.MjModel.from_xml_string(xml)
        asset = newton.ModelBuilder()
        asset.add_mjcf_native(xml)
        builder = newton.ModelBuilder()
        for _ in range(2):
            builder.begin_world()
            builder.add_builder(asset)
            builder.add_builder(asset)
            builder.end_world()
        solver = SolverMuJoCo(builder.finalize(), disable_contacts=True, kinematic_switchable_roots=False)
        for field in ("eq_type", "eq_data"):
            np.testing.assert_allclose(getattr(solver.mj_model, field), np.concatenate([getattr(reference, field)] * 2))
        np.testing.assert_array_equal(solver.mj_model.eq_obj1id, [0, 1, 2, 3])
        np.testing.assert_array_equal(solver.mj_model.eq_obj2id, [1, -1, 3, -1])
        solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy(), np.tile(reference.eq_data, (2, 2, 1)))

    def test_native_joint_parameters(self):
        """Preserve ball limits, free-joint passive forces, and asymmetric effort limits."""
        for joint in (
            '<joint type="ball" limited="true" range="0 35" stiffness="2" damping=".3" armature=".1" frictionloss=".2" margin=".01" solreflimit=".01 1" solreffriction=".03 1"/>',
            '<joint type="free" stiffness="2" damping=".3" armature=".1" frictionloss=".2"/>',
            '<joint actuatorfrcrange="-1 2" stiffness="3" damping=".2"/>',
            '<joint springdamper=".2 .7" armature=".1"/>',
            '<joint stiffness="3 .2 .1" damping=".2 .3 .4"/>',
        ):
            with self.subTest(joint=joint):
                xml = f'<mujoco><worldbody><body>{joint}<geom type="box" size=".1 .2 .3" mass="1"/></body></worldbody></mujoco>'
                _, solver, reference = self.assert_roundtrip(
                    xml,
                    (
                        "jnt_limited",
                        "jnt_range",
                        "jnt_stiffness",
                        "jnt_stiffnesspoly",
                        "dof_dampingpoly",
                        "jnt_actfrclimited",
                        "jnt_actfrcrange",
                        "dof_damping",
                        "dof_armature",
                        "dof_frictionloss",
                        "jnt_solref",
                        "dof_solref",
                    ),
                )
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                np.testing.assert_allclose(
                    solver.mjw_model.jnt_actfrcrange.numpy().reshape(-1, 2), reference.jnt_actfrcrange
                )
                if reference.jnt_limited[0]:
                    np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy().reshape(-1, 2), reference.jnt_range)

    def test_composed_joint_springdamper(self):
        """Resolve spring and actuator damping against the complete assembled inertia."""
        import mujoco

        joint = '<joint name="j" axis="0 1 0" springdamper=".2 .7"/>'
        geom = '<geom type="capsule" size=".05 .5" mass="0"/><body name="arm_mass"><geom type="capsule" size=".05 .5" mass="1"/></body>'
        hand = '<body name="hand" pos="0 0 1"><geom size=".1" mass="2"/></body>'
        actuator = '<actuator><position joint="j" kp="10" dampratio=".8"/></actuator>'
        arm = f"<mujoco><worldbody><body>{joint}{geom}</body></worldbody>{actuator}</mujoco>"
        assembled = f"<mujoco><worldbody><body>{joint}{geom}{hand}</body></worldbody>{actuator}</mujoco>"
        reference = mujoco.MjModel.from_xml_string(assembled)
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(arm)
        with self.assertWarnsRegex(newton.exceptions.NewtonWarning, "zero or negative mass"):
            builder.add_mjcf_native(f"<mujoco><worldbody>{hand}</worldbody></mujoco>", parent_body=0)
        self.assertEqual(builder.custom_attributes["mujoco:ctrl_source"].values, [SolverMuJoCo.CtrlSource.JOINT_TARGET])
        np.testing.assert_array_equal(builder.joint_target_kd, [0])
        model = builder.finalize()
        solver = SolverMuJoCo(model, disable_contacts=True)
        np.testing.assert_allclose(model.joint_target_kd.numpy(), -reference.actuator_biasprm[:, 2], rtol=3e-5)
        for field in ("jnt_stiffness", "dof_damping", "actuator_biasprm"):
            np.testing.assert_allclose(getattr(solver.mj_model, field), getattr(reference, field), rtol=3e-5)
            np.testing.assert_allclose(
                getattr(solver.mjw_model, field).numpy().reshape(getattr(reference, field).shape),
                getattr(reference, field),
                rtol=3e-5,
            )

    def test_site_equalities(self):
        """Preserve both site references and frames through copying and conversion."""
        import mujoco

        for kind in ("connect", "weld"):
            for convert in (False, True):
                with self.subTest(kind=kind, convert=convert):
                    xml = f"""<mujoco><worldbody><body name="a"><freejoint/><geom size=".1"/>
                        <site name="a" pos=".2 0 0" euler="0 0 10"/></body>
                        <body name="b" pos="1 0 0"><freejoint/><geom size=".1"/>
                        <site name="b" pos="-.3 0 0" euler="0 0 20"/></body></worldbody>
                        <equality><{kind} site1="a" site2="b"/></equality></mujoco>"""
                    reference = mujoco.MjModel.from_xml_string(xml)
                    asset = newton.ModelBuilder()
                    asset.add_mjcf_native(xml, convert_mjc_equality_constraints=convert)
                    builder = newton.ModelBuilder()
                    builder.add_builder(asset)
                    builder.add_builder(asset)
                    solver = SolverMuJoCo(builder.finalize(), disable_contacts=True, kinematic_switchable_roots=False)
                    for field in ("eq_type", "eq_objtype", "eq_data"):
                        np.testing.assert_allclose(
                            getattr(solver.mj_model, field), np.concatenate([getattr(reference, field)] * 2), atol=1e-6
                        )
                    np.testing.assert_array_equal(solver.mj_model.eq_obj1id, [0, 2])
                    np.testing.assert_array_equal(solver.mj_model.eq_obj2id, [1, 3])
                    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
                    np.testing.assert_allclose(
                        solver.mjw_model.eq_data.numpy()[0], np.tile(reference.eq_data, (2, 1)), atol=1e-6
                    )

    def test_compound_joint_references_and_nonlinear_tendons(self):
        """Preserve compound axes and edited scalar references through composition."""
        xml = """<mujoco><worldbody><body><joint name="x" axis="1 0 0"/><joint name="y" axis="0 1 0"/>
            <geom type="box" size=".1 .2 .3"/></body><body><joint name="z"/><geom size=".1"/></body>
            <body><joint name="w"/><geom size=".1"/></body></worldbody>
            <tendon><fixed stiffness="2 3 4" damping=".1 .2 .3"><joint joint="y" coef="2"/><joint joint="z" coef="-1"/></fixed></tendon>
            <equality><joint joint1="y" joint2="z" polycoef=".1 2 0 0 0"/>
            <joint joint1="z" joint2="y"/><joint joint1="z"/></equality></mujoco>"""
        asset, _, reference = self.assert_roundtrip(
            xml,
            (
                "wrap_objid",
                "wrap_prm",
                "tendon_stiffness",
                "tendon_stiffnesspoly",
                "tendon_dampingpoly",
                "eq_obj1id",
                "eq_obj2id",
                "eq_data",
            ),
            convert_mjc_equality_constraints=False,
        )
        asset.custom_attributes["mujoco:tendon_joint"].values[1] = 2
        asset.custom_attributes["mujoco:equality_constraint_joint2"].values[0] = 2
        asset.custom_attributes["mujoco:equality_constraint_joint1"].values[1:3] = [2, 2]
        builder = newton.ModelBuilder()
        builder.add_builder(asset)
        builder.add_builder(asset)
        solver = SolverMuJoCo(builder.finalize(), disable_contacts=True, kinematic_switchable_roots=False)
        np.testing.assert_array_equal(solver.mj_model.wrap_objid, [1, 3, 5, 7])
        np.testing.assert_array_equal(solver.mj_model.eq_obj1id, [1, 3, 3, 5, 7, 7])
        np.testing.assert_array_equal(solver.mj_model.eq_obj2id, [3, 1, -1, 7, 5, -1])
        for field in ("tendon_stiffness", "tendon_stiffnesspoly", "tendon_dampingpoly", "eq_data"):
            np.testing.assert_allclose(
                getattr(solver.mj_model, field), np.concatenate([getattr(reference, field)] * 2), atol=1e-6
            )

    def test_native_compiler_optimizations(self):
        """Import the topology and frames produced by native compiler optimizations."""
        for option, joint in (('fusestatic="true"', "<joint/>"), ('alignfree="true"', "<freejoint/>")):
            with self.subTest(option=option):
                xml = f'<mujoco><compiler {option}/><worldbody><body><geom size=".1"/></body><body>{joint}<geom type="box" size=".1 .2 .3" pos=".2 .3 .4" euler="10 20 30"/></body></worldbody></mujoco>'
                self.assert_roundtrip(xml, ("body_mass", "body_pos", "body_ipos", "geom_pos", "qpos0"))

    def test_native_rejects_malformed_legacy_inputs(self):
        """Reject malformed sources that the legacy parser skips or tolerates."""
        import mujoco

        world = """<worldbody><body name="b"><joint name="j"/><geom name="g" size=".1"/>
            <site name="tip"/></body><body name="other"><geom name="other_g" size=".1"/>
            <site name="base"/></body></worldbody>"""
        cases = {
            "generated geom label": world + '<contact><pair geom1="worldbody/b/g" geom2="other_g"/></contact>',
            "unknown equality": world + "<equality><unknown/></equality>",
            "missing tendon": world + '<equality><tendon tendon1="absent"/></equality>',
            "exclude missing body": world + '<contact><exclude body1="b" body2="absent"/></contact>',
            "exclude missing attribute": world + '<contact><exclude body1="b"/></contact>',
            "unknown flag": world + '<option><flag mysteryflag="enable"/></option>',
            "missing include attribute": "<include/>",
            "undefined default class": world.replace('name="b"', 'name="b" childclass="absent"'),
            "duplicate body name": world.replace('name="other"', 'name="b"'),
            "conflicting position damping": world + '<actuator><position joint="j" kv="3" dampratio=".9"/></actuator>',
            "activation without dynamics": world + '<actuator><general joint="j" actrange="-1 1"/></actuator>',
            "parent transmission inherited range": world
            + '<actuator><position jointinparent="j" inheritrange="1"/></actuator>',
            "missing crank site": world
            + '<actuator><general cranksite="absent" slidersite="base" cranklength="1"/></actuator>',
            "missing slider site": world + '<actuator><general cranksite="tip" cranklength="1"/></actuator>',
            "missing transmission": world + '<actuator><general cranklength="1"/></actuator>',
            "invalid crank length": world
            + '<actuator><general cranksite="tip" slidersite="base" cranklength="-1"/></actuator>',
            "zero length site": world.replace(
                '<site name="tip"/>', '<site type="capsule" size=".1" fromto="0 0 0 0 0 0"/>'
            ),
        }
        for attribute in (
            'diaginertia="1 2"',
            'diaginertia="1 2 3 4"',
            'fullinertia="1 2 3 0 0"',
            'fullinertia="1 2 3 0 0 0 0"',
            'fullinertia="2 3 4 .1 .2 .3" quat="1 0 0 0"',
        ):
            cases[attribute] = f'<worldbody><body><inertial pos="0 0 0" mass="1" {attribute}/></body></worldbody>'
        for name, content in cases.items():
            with self.subTest(case=name):
                xml = f"<mujoco>{content}</mujoco>"
                with self.assertRaises(ValueError):
                    mujoco.MjSpec.from_string(xml).compile()
                builder = newton.ModelBuilder()
                with self.assertRaises(ValueError):
                    builder.add_mjcf_native(xml)
                self.assertEqual((builder.body_count, builder.joint_count, builder.shape_count), (0, 0, 0))

    def test_native_requires_compilable_sources(self):
        """Keep standalone mass and collision-hull requirements explicit."""
        import mujoco

        sources = (
            "<mujoco><worldbody><body><joint/></body></worldbody></mujoco>",
            '<mujoco><asset><mesh name="m" vertex="0 0 0 1 0 0 0 1 0"/></asset>'
            '<worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>',
            '<mujoco><asset><mesh name="m" vertex="0 0 0 1 0 0 0 1 0 1 1 0" face="0 1 2 1 3 2"/></asset>'
            '<worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>',
        )
        for xml in sources:
            with self.subTest(xml=xml):
                with self.assertRaises(ValueError):
                    mujoco.MjSpec.from_string(xml).compile()
                with self.assertRaises(ValueError):
                    newton.ModelBuilder().add_mjcf_native(xml)

    def test_native_small_primitive_mass(self):
        """Retain native mass below the legacy reference-volume cutoff."""
        import mujoco

        xml = """<mujoco><worldbody><body><freejoint/>
            <geom type="sphere" size=".00005" mass=".6"/>
            </body></worldbody></mujoco>"""
        reference = mujoco.MjModel.from_xml_string(xml)
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(xml)
        np.testing.assert_allclose(builder.body_mass, [0.6], rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(builder.body_inertia[0]).reshape(3, 3), np.diag(reference.body_inertia[1]), rtol=1e-6, atol=0
        )

    def test_native_scaled_inherited_slide_range(self):
        """Resolve scaled control and activation limits with a nonzero slide reference."""
        for actuator, field in (("position", "ctrl"), ("intvelocity", "act")):
            with self.subTest(actuator=actuator):
                xml = f"""<mujoco><worldbody><body><joint name="j" type="slide" range="-.3 .5" ref=".1"/>
                    <geom size=".1" mass="1"/></body></worldbody>
                    <actuator><{actuator} joint="j" kp="20" inheritrange="1"/></actuator></mujoco>"""
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(xml, scale=2)
                model = builder.finalize()
                source = (
                    SolverMuJoCo.CtrlSource.JOINT_TARGET
                    if actuator == "position"
                    else SolverMuJoCo.CtrlSource.CTRL_DIRECT
                )
                np.testing.assert_array_equal(model.mujoco.ctrl_source.numpy(), source)
                np.testing.assert_array_equal(getattr(model.mujoco, f"actuator_{field}range").numpy(), [[0, 0]])
                solver = SolverMuJoCo(model, disable_contacts=True)
                np.testing.assert_allclose(getattr(solver.mj_model, f"actuator_{field}range"), [[-0.6, 1]])
                np.testing.assert_allclose(getattr(model.mujoco, f"actuator_{field}range").numpy(), [[-0.6, 1]])
                np.testing.assert_array_equal(getattr(solver.mj_model, f"actuator_{field}limited"), [True])

    def test_native_compiled_primitive_fitting(self):
        """Retain fitted primitive poses and sizes from the native compiler."""
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "box.stl"
            test_import_mjcf.TestImportMjcfGeometry._write_box_stl(mesh, hx=1, hy=0.5, hz=2)
            for kind in ("box", "ellipsoid"):
                for scale, fitscale in (("1 1 1", 1), ("2 3 4", 0.7)):
                    with self.subTest(kind=kind, scale=scale, fitscale=fitscale):
                        xml = f'''<mujoco><compiler fitaabb="true"/>
                            <default><mesh scale="{scale}"/></default>
                            <asset><mesh name="m" file="{mesh}"/></asset><worldbody><body>
                            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                            <geom type="{kind}" mesh="m" fitscale="{fitscale}"/>
                            </body></worldbody></mujoco>'''
                        self.assert_roundtrip(xml, ("geom_size", "geom_pos", "geom_quat"))

    def test_native_texture_orientation(self):
        """Match MuJoCo texture samples through Newton's viewer conventions."""
        import mujoco
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asymmetric.png"
            Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], np.uint8)).save(path)
            xml = f'''<mujoco><asset><mesh name="m" vertex="0 0 0 1 0 0 0 1 0 0 0 1"
                face="0 2 1 0 1 3 0 3 2 1 2 3" texcoord=".1 .2 .6 .3 .2 .8 .7 .9"/>
                <texture name="t" type="2d" file="{path}"/><material name="mat" texture="t"/>
                </asset><worldbody><geom type="mesh" mesh="m" material="mat"/></worldbody></mujoco>'''
            reference = mujoco.MjModel.from_xml_string(xml)
            builder = newton.ModelBuilder()
            builder.add_mjcf_native(xml)
            mesh = builder.shape_source[0]
            uv = reference.mesh_texcoord[reference.mesh_facetexcoord.reshape(-1)]
            reference_image = reference.tex_data.reshape(2, 2, 3)
            expected = reference_image[(uv[:, 1] * 2).astype(int), (uv[:, 0] * 2).astype(int)]
            for flip_image in (False, True):
                with self.subTest(flip_image=flip_image):
                    image = normalize_texture(mesh.texture, flip_vertical=flip_image)
                    v = mesh.uvs[:, 1] if flip_image else 1 - mesh.uvs[:, 1]
                    actual = image[(v * 2).astype(int), (mesh.uvs[:, 0] * 2).astype(int), :3]
                    np.testing.assert_array_equal(actual, expected)

    def test_native_binary_mesh_geometry(self):
        """Preserve MSH surfaces and attributes across compiled vertex duplication."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32)
        uvs = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            for has_faces, has_normals, has_uvs in product((False, True), repeat=3):
                with self.subTest(faces=has_faces, normals=has_normals, uvs=has_uvs):
                    path = Path(directory) / f"mesh_{has_faces}_{has_normals}_{has_uvs}.msh"
                    with path.open("wb") as output:
                        output.write(np.array([4, 4 * has_normals, 4 * has_uvs, 4 * has_faces], np.int32).tobytes())
                        output.write(vertices.tobytes())
                        if has_normals:
                            output.write(normals.tobytes())
                        if has_uvs:
                            output.write(uvs.tobytes())
                        if has_faces:
                            output.write(faces.tobytes())
                    xml = f'''<mujoco><asset><mesh name="m" file="{path}"/></asset>
                        <worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>'''
                    builder = newton.ModelBuilder()
                    builder.add_mjcf_native(xml)
                    expected_uvs = uvs.copy()
                    expected_uvs[:, 1] = 1 - expected_uvs[:, 1]
                    _assert_mesh_geometry(
                        self,
                        builder,
                        vertices,
                        faces,
                        normals=normals if has_normals else None,
                        uvs=expected_uvs if has_uvs else None,
                    )

    def test_source_errors_preserve_context(self):
        """Report file and parse errors without suggesting an importer switch."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parts").mkdir()
            (root / "main.xml").write_text('<mujoco><include file="missing.xml"/></mujoco>')
            (root / "nested.xml").write_text('<mujoco><include file="parts/part.xml"/></mujoco>')
            (root / "parts/part.xml").write_text('<mujoco><include file="missing_nested.xml"/></mujoco>')
            cases = (
                (root / "missing.xml", FileNotFoundError, None, "missing.xml"),
                (root / "main.xml", ValueError, ValueError, "missing.xml"),
                (root / "nested.xml", ValueError, ValueError, "missing_nested.xml"),
                (f'<mujoco><include file="{root / "missing.xml"}"/></mujoco>', ValueError, ValueError, "missing.xml"),
                ("<mujoco><worldbody>", ValueError, ET.ParseError, None),
                ('<mujoco><worldbody unexpected="1"/></mujoco>', ValueError, ValueError, "unexpected"),
            )
            for source, error_type, cause_type, detail in cases:
                with self.subTest(source=source):
                    builder = newton.ModelBuilder()
                    builder.add_body(mass=1)
                    before = (
                        builder.body_count,
                        builder.joint_count,
                        builder.shape_count,
                        set(builder.custom_attributes),
                    )
                    with self.assertRaises(error_type) as context:
                        builder.add_mjcf_native(source)
                    error = context.exception
                    self.assertNotIn("Use add_mjcf()", str(error))
                    if detail is not None:
                        self.assertIn(detail, str(error))
                    if cause_type is not None:
                        self.assertIsInstance(error.__cause__, cause_type)
                        self.assertIn("Native MJCF parsing failed", str(error))
                    self.assertEqual(
                        (builder.body_count, builder.joint_count, builder.shape_count, set(builder.custom_attributes)),
                        before,
                    )

    def test_custom_path_resolver(self):
        source = '<mujoco><worldbody><include file="part.xml"/></worldbody></mujoco>'
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(
            source, path_resolver=lambda base, name: '<mujoco><body><geom size=".1"/></body></mujoco>'
        )
        self.assertEqual(builder.body_count, 1)

        def resolve(base_dir, filename):
            return str((Path(base_dir or ".") / filename).resolve())

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "geom.xml").write_text('<mujoco><geom size=".1"/></mujoco>')
            for name in ("left.xml", "right.xml"):
                (root / name).write_text('<mujoco><include file="geom.xml"/></mujoco>')
            source = root / "main.xml"
            for left, right in (("geom.xml", "geom.xml"), ("left.xml", "right.xml")):
                with self.subTest(left=left, right=right):
                    source.write_text(
                        f'<mujoco><worldbody><body name="left"><include file="{left}"/></body>'
                        f'<body name="right"><include file="{right}"/></body></worldbody></mujoco>'
                    )
                    if left == right:
                        with self.assertRaisesRegex(ValueError, "already included"):
                            newton.ModelBuilder().add_mjcf_native(source)
                    builder = newton.ModelBuilder()
                    builder.add_mjcf_native(source, path_resolver=resolve)
                    self.assertEqual(builder.body_count, 2)
                    self.assertEqual(builder.shape_count, 2)
                    self.assertEqual(builder.shape_body, [0, 1])
                    np.testing.assert_allclose(np.asarray(builder.shape_scale)[:, 0], [0.1, 0.1])

            (root / "left.xml").write_text('<mujoco><include file="right.xml"/></mujoco>')
            (root / "right.xml").write_text('<mujoco><include file="left.xml"/></mujoco>')
            builder = newton.ModelBuilder()
            with self.assertRaisesRegex(ValueError, "Circular include detected"):
                builder.add_mjcf_native(source, path_resolver=resolve)
            self.assertEqual(builder.body_count, 0)
            self.assertEqual(builder.shape_count, 0)


if __name__ == "__main__":
    unittest.main()
