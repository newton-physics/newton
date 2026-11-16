# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical and transactional contracts of the opt-in authoring scene."""

import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

from newton import BodyFlags, CollisionPipeline, JointType, Mesh, ModelBuilder, ShapeFlags, eval_ik
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import get_cuda_test_devices
from newton.viewer import SceneEditor, ViewerNull


class TestSceneAuthoring(unittest.TestCase):
    def make_editor(self):
        """Create a real CPU scene without a graphical or network dependency."""
        editor = SceneEditor(ViewerNull(), device="cpu")
        self.addCleanup(editor.close)
        return editor

    @staticmethod
    def write_jointed_mesh_asset(path):
        """Write a closed native mesh and fixed joint for preparation/control tests."""
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/Assembly")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        for name, height in (("Base", 0.0), ("Child", 0.2)):
            if name == "Base":
                body = UsdGeom.Cube.Define(stage, f"/Assembly/{name}")
                body.CreateSizeAttr(0.1)
            else:
                body = UsdGeom.Mesh.Define(stage, f"/Assembly/{name}")
                body.CreatePointsAttr(
                    [
                        (-0.05, -0.05, -0.05),
                        (0.05, -0.05, -0.05),
                        (0.05, 0.05, -0.05),
                        (-0.05, 0.05, -0.05),
                        (-0.05, -0.05, 0.05),
                        (0.05, -0.05, 0.05),
                        (0.05, 0.05, 0.05),
                        (-0.05, 0.05, 0.05),
                    ]
                )
                body.CreateFaceVertexCountsAttr([4] * 6)
                body.CreateFaceVertexIndicesAttr(
                    [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5]
                )
                body.CreateSubdivisionSchemeAttr("none")
            body.AddTranslateOp().Set((0, 0, height))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1)
        joint = UsdPhysics.FixedJoint.Define(stage, "/Assembly/Weld")
        joint.CreateBody0Rel().SetTargets(["/Assembly/Base"])
        joint.CreateBody1Rel().SetTargets(["/Assembly/Child"])
        joint.CreateLocalPos0Attr((0, 0, 0.2))
        stage.GetRootLayer().Save()

    def test_fast_edits_preserve_storage_and_gesture_undo(self):
        """Batch poses into both states and reset coordinates without replacing storage."""
        editor = self.make_editor()
        object_ids = tuple(editor.add_primitive("box", position=(2 * index, 0, 2)) for index in range(3))
        editor.commit()
        editor.step()
        model = editor.model
        storage = tuple(
            (source, source.body_q, source.joint_q, source.joint_qd)
            for source in (editor.state_0, editor.state_1, model)
        )

        def assert_poses(expected):
            self.assertIs(editor.model, model)
            self.assertEqual(editor.rebuild_count, 1)
            self.assertIs(editor.state_0, storage[0][0])
            self.assertIs(editor.state_1, storage[1][0])
            for source, body_q, joint_q, joint_qd in storage:
                self.assertIs(source.body_q, body_q)
                self.assertIs(source.joint_q, joint_q)
                self.assertIs(source.joint_qd, joint_qd)
                poses = source.body_q.numpy()
                np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-6)
                np.testing.assert_allclose(poses, expected, atol=1e-6)
                np.testing.assert_allclose(source.joint_q.numpy(), expected.reshape(-1), atol=1e-6)
                np.testing.assert_allclose(source.joint_qd.numpy(), 0, atol=1e-6)

        editor.begin_edit()
        rotations = tuple(
            tuple(wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.4, 0.5)), 0.7 * (index + 1)))
            for index in range(3)
        )
        for sample in range(48):
            expected = []
            for index, object_id in enumerate(object_ids):
                position = (2 * index + sample * 0.01, 0, 2)
                rotation = rotations[index]
                expected.append((*position, *rotation))
                editor.update_object(
                    object_id, position=position, rotation=rotation, color=(0.2, 0.4, 0.6), friction=0.8
                )
            editor.before_frame()
            if sample == 0:
                assert_poses(np.asarray(expected))
        editor.end_edit()
        expected = np.asarray(expected)
        assert_poses(expected)
        np.testing.assert_allclose(editor.get_pose(object_ids[0])[0], (0.47, 0, 2), atol=1e-6)
        np.testing.assert_allclose(editor.model.shape_color.numpy()[1], (0.2, 0.4, 0.6))
        self.assertAlmostEqual(editor.model.shape_material_mu.numpy()[1], 0.8, places=6)
        editor.undo()
        editor.before_frame()
        assert_poses(np.asarray([(2 * index, 0, 2, 0, 0, 0, 1) for index in range(3)]))
        editor.redo()
        editor.before_frame()
        assert_poses(expected)
        editor.reset_scene()
        assert_poses(expected)

    def test_batched_topology_preserves_unedited_physical_state(self):
        """Add and remove objects once while retaining an existing falling body's state."""
        editor = self.make_editor()
        removed = editor.add_primitive("box", position=(2, 0, 3))
        first = editor.add_primitive("sphere", position=(0, 0, 3), size=(0.5, 0.5, 0.5))
        editor.commit()
        for _ in range(4):
            editor.step()
        index = editor.body_indices(first)[0]
        before_q = editor.state_0.body_q.numpy()[index].copy()
        before_qd = editor.state_0.body_qd.numpy()[index].copy()
        editor.control.joint_f[6:7].fill_(0.7)
        editor.begin_edit()
        editor.add_primitive("cylinder", position=(3, 0, 2))
        editor.add_primitive("capsule", position=(4, 0, 2), size=(0.5, 0.5, 1.5))
        editor.remove_object(removed)
        editor.before_frame()
        self.assertEqual(editor.rebuild_count, 1)
        editor.end_edit()
        editor.before_frame()
        self.assertEqual(editor.rebuild_count, 2)
        self.assertEqual(editor.model.body_count, 3)
        index = editor.body_indices(first)[0]
        np.testing.assert_array_equal(editor.state_0.body_q.numpy()[index], before_q)
        np.testing.assert_array_equal(editor.state_0.body_qd.numpy()[index], before_qd)
        self.assertAlmostEqual(editor.control.joint_f.numpy()[0], 0.7, places=6)
        editor.step()
        self.assertLess(editor.get_pose(first)[0][2], before_q[2])

    def test_authored_objects_collide_in_one_physical_scene(self):
        """Settle on an authored support and insert overlapping drops without launching its occupant."""
        editor = self.make_editor()
        support = editor.add_primitive("box", position=(0, 0, 0.5), dynamic=False)
        sphere = editor.add_primitive("sphere", position=(0, 0, 2), size=(0.5, 0.5, 0.5))
        editor.add_primitive("box", position=(4, 0, 10), size=(2, 2, 0.5), dynamic=False)
        editor.commit()
        for _ in range(120):
            editor.step()
        height = editor.get_pose(sphere)[0][2]
        self.assertGreater(height, 1.20)
        self.assertLess(height, 1.30)
        np.testing.assert_allclose(editor.get_pose(support)[0], (0, 0, 0.5), atol=1e-6)
        self.assertEqual(editor.rebuild_count, 1)
        index = editor.body_indices(sphere)[0]
        editor.state_0.body_qd[index : index + 1].fill_(wp.spatial_vector(10, 0, 0, 0, 0, 0))
        for _ in range(12):
            editor.step()
        for state in (editor.state_0, editor.state_1):
            state.body_qd[index : index + 1].zero_()
        live = editor.get_pose(sphere)[0]
        self.assertGreater(live[0], 1.5)
        height = live[2]
        before_q, before_v = editor.state_0.body_q.numpy().copy(), editor.state_0.body_qd.numpy().copy()
        first = editor.add_primitive("box", position=(live[0], live[1], 0.5))
        second = editor.add_primitive("box", position=(live[0], live[1], 0.5))
        reset_drop = editor.add_primitive("box", position=(0, 0, 0.5))
        overhead_drop = editor.add_primitive("box", position=(4, 0, 2))
        for object_id in (first, second, reset_drop, overhead_drop):
            editor.place_object(object_id)
        editor.commit()
        np.testing.assert_array_equal(editor.state_0.body_q.numpy()[: len(before_q)], before_q)
        np.testing.assert_array_equal(editor.state_0.body_qd.numpy()[: len(before_v)], before_v)
        self.assertGreater(editor.get_object(first).position[2], height + 0.7)
        self.assertLess(editor.get_object(first).position[2], height + 0.8)
        self.assertGreater(editor.get_object(second).position[2] - editor.get_object(first).position[2], 1.0)
        self.assertGreater(editor.get_object(reset_drop).position[2], 2.7)
        self.assertGreater(editor.get_object(overhead_drop).position[2], 10.75)
        self.assertLess(editor.get_object(overhead_drop).position[2], 10.8)
        self.assertEqual(editor.rebuild_count, 2)
        editor.step()
        self.assertLess(np.linalg.norm(editor.state_0.body_qd.numpy()[:, :3], axis=1).max(), 1.0)
        self.assertTrue(np.isfinite(editor.state_0.body_q.numpy()).all())
        resolved = {obj.id: obj.position for obj in editor.objects}
        for removed in (overhead_drop, reset_drop, second):
            first_body = editor.body_indices(first)[0]
            first_pose = editor.state_0.body_q.numpy()[first_body].copy()
            first_velocity = editor.state_0.body_qd.numpy()[first_body].copy()
            editor.undo()
            editor.before_frame()
            self.assertNotIn(removed, {obj.id for obj in editor.objects})
            self.assertEqual(editor.get_object(first).position, resolved[first])
            first_body = editor.body_indices(first)[0]
            np.testing.assert_array_equal(editor.state_0.body_q.numpy()[first_body], first_pose)
            np.testing.assert_array_equal(editor.state_0.body_qd.numpy()[first_body], first_velocity)
        for restored_id in (second, reset_drop, overhead_drop):
            editor.redo()
            editor.before_frame()
            self.assertEqual(editor.get_object(restored_id).position, resolved[restored_id])
            self.assertEqual(editor.get_object(first).position, resolved[first])
        editor.reset_scene()
        for _ in range(60):
            editor.step()
            self.assertLess(np.linalg.norm(editor.state_0.body_qd.numpy()[:, :3], axis=1).max(), 10.0)
        if importlib.util.find_spec("pxr"):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "safe-reset.usda"
                editor.save_usd(output)
                restored = self.make_editor()
                restored.load_usd(output)
                restored.commit()
                for _ in range(60):
                    restored.step()
                    self.assertLess(np.linalg.norm(restored.state_0.body_qd.numpy()[:, :3], axis=1).max(), 10.0)

    def test_structural_edits_preserve_live_pose_and_authored_reset(self):
        """Resize and change mass of a settled prop at its live location, then reset it."""
        editor = self.make_editor()
        object_id = editor.add_primitive("box", position=(0, 0, 4))
        editor.commit()
        editor.update_object(object_id, position=(0, 0, 3))
        editor.before_frame()
        for _ in range(90):
            editor.step()
        position, rotation = editor.get_pose(object_id)
        self.assertLess(position[2], 0.6)
        editor.begin_edit()
        for index in range(10):
            editor.update_object(object_id, size=(1 + index * 0.02, 1, 1), mass=2)
            editor.before_frame()
        self.assertEqual(editor.rebuild_count, 1)
        editor.end_edit()
        editor.before_frame()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], position, atol=1e-6)
        np.testing.assert_allclose(editor.get_pose(object_id)[1], rotation, atol=1e-6)
        self.assertAlmostEqual(editor.model.body_mass.numpy()[0], 2, places=6)
        editor.update_object(object_id, dynamic=False)
        editor.before_frame()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], position, atol=1e-6)
        editor.update_object(object_id, position=(2, 0, 3), mass=3)
        editor.before_frame()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], (2, 0, 3), atol=1e-6)
        editor.reset_scene()
        np.testing.assert_allclose(editor.get_pose(object_id)[0], (2, 0, 3), atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_failed_usd_transaction_keeps_working_runtime(self):
        """Reject an unsupported USD without losing the existing simulation."""
        editor = self.make_editor()
        object_id = editor.add_primitive("box", position=(0, 0, 2))
        editor.commit()
        model, state = editor.model, editor.state_0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsupported.usda"
            path.write_text(
                '#usda 1.0\n( metersPerUnit = 1 )\ndef Points "Unsupported" (prepend apiSchemas = ["NewtonParticleAPI"]) {}\n'
            )
            valid = Path(directory) / "native.usda"
            valid.write_text(
                '#usda 1.0\n( metersPerUnit = 1 )\ndef Cube "Body" '
                '(prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]) {\n'
                " double size = 0.1\n float physics:mass = 2\n}\n"
            )
            prepared = editor.add_usd(valid, position=(2, 0, 2))
            invalid = editor.add_usd(path)
            pending_objects = editor.objects
            with self.assertRaisesRegex(ValueError, "rigid USD"):
                editor.commit()
            self.assertEqual(editor.objects, pending_objects)
            self.assertIs(editor.model, model)
            self.assertIs(editor.state_0, state)
            self.assertEqual(editor.rebuild_count, 1)
            editor.step()
            self.assertLess(editor.get_pose(object_id)[0][2], 2)
            editor.remove_object(invalid)
            editor.remove_object(prepared)
            editor.commit()
            self.assertEqual(editor.model.body_count, 1)
            # A discarded failed preparation must not block unrelated valid edits.
            obsolete = editor.add_usd(path)
            editor.remove_object(obsolete)
            editor.add_primitive("sphere", position=(2, 0, 2))
            deadline = time.monotonic() + 3
            while editor.busy and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(editor.busy)
            editor.before_frame()
            self.assertIsNone(editor.error)
            self.assertEqual(editor.model.body_count, 2)

            malformed = Path(directory) / "malformed.usda"
            malformed.write_text('#usda 1.0\ndef Cube "Broken" { invalid syntax }\n')
            failed = editor.add_usd(malformed)
            model = editor.model
            with self.assertLogs("newton", level="ERROR"):
                deadline = time.monotonic() + 30
                while editor.error is None and time.monotonic() < deadline:
                    editor.before_frame()
                    time.sleep(0.001)
            self.assertRegex(editor.error, "(?i)(parse|syntax|expected)")
            self.assertNotIn("Pickling", editor.error)
            self.assertIs(editor.model, model)

            # A failed pending asset must not restart while an existing prop is edited.
            for sample in range(3):
                position = (0.1 * (sample + 1), 0, 2)
                editor.update_object(object_id, position=position)
                editor.before_frame()
                self.assertFalse(editor._preparing)
                self.assertFalse(editor.busy)
                self.assertIs(editor.model, model)
                np.testing.assert_allclose(editor.get_pose(object_id)[0], position, atol=1e-6)
            self.assertTrue(editor.pending)
            with mock.patch.object(editor, "_commit_signature", wraps=editor._commit_signature) as signature:
                for _ in range(3):
                    editor.before_frame()
                self.assertEqual(signature.call_count, 0)
            editor.request_commit()
            editor.before_frame()
            self.assertTrue(editor._preparing)
            with self.assertRaises(ValueError):
                next(iter(editor._preparing.values())).result(timeout=30)
            with self.assertLogs("newton", level="ERROR"):
                editor.before_frame()
            self.assertFalse(editor._preparing)
            self.assertFalse(editor.busy)
            self.assertIs(editor.model, model)

            malformed.write_text('#usda 1.0\n( metersPerUnit = 1 )\ndef Cube "Body" {\n double size = 0.1\n}\n')
            editor.request_commit()
            deadline = time.monotonic() + 30
            while editor.pending and time.monotonic() < deadline:
                editor.before_frame()
                time.sleep(0.001)
            self.assertFalse(editor.pending, editor.error)
            self.assertIsNone(editor.error)
            self.assertEqual(editor.model.body_count, 3)
            editor.remove_object(failed)
            editor.before_frame()

            malformed.write_text('#usda 1.0\n( metersPerUnit = 1 )\ndef Cube "Body" {\n double size = 0.1\n}\n')
            imported = editor.add_usd(malformed)
            next(iter(editor._preparing.values())).result(timeout=30)
            malformed.write_text('#usda 1.0\n( metersPerUnit = 1 )\ndef Cube "Body" {\n double size = 0.2\n}\n')
            model = editor.model
            with self.assertLogs("newton", level="ERROR"):
                editor.before_frame()
            self.assertIn("changed after preparation", editor.error)
            self.assertIs(editor.model, model)
            editor.request_commit()
            deadline = time.monotonic() + 30
            while editor.pending and time.monotonic() < deadline:
                editor.before_frame()
                time.sleep(0.001)
            self.assertFalse(editor.pending, editor.error)
            self.assertIsNone(editor.error)
            self.assertEqual(editor.model.body_count, 3)

            editor.preview_usd(malformed, position=(0, 0, 2), scale=(3, 3, 3))
            next(iter(editor._preparing.values())).result(timeout=30)
            malformed.write_text('#usda 1.0\n( metersPerUnit = 1 )\ndef Cube "Body" {\n double size = 0.3\n}\n')
            editor.update_object(imported, size=(3, 3, 3))
            editor.commit()
            editor.clear_preview()
            editor.before_frame()
            self.assertIsNone(editor.error)

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_static_usd_plane_supports_dynamic_objects(self):
        """Keep an imported zero-volume static collider physical and editable."""
        from pxr import Usd, UsdGeom, UsdPhysics

        editor = self.make_editor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plane.usda"
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            plane = UsdGeom.Plane.Define(stage, "/Plane")
            plane.CreateAxisAttr(UsdGeom.Tokens.z)
            UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
            stage.GetRootLayer().Save()
            plane_id = editor.add_usd(path, position=(0, 0, 1))
            sphere = editor.add_primitive("sphere", position=(0, 0, 2), size=(0.5, 0.5, 0.5))
            editor.commit()
            self.assertFalse(editor.get_object(plane_id).dynamic)
            for _ in range(90):
                editor.step()
            self.assertGreater(editor.get_pose(sphere)[0][2], 1.20)
            self.assertLess(editor.get_pose(sphere)[0][2], 1.30)
            np.testing.assert_allclose(editor.get_pose(plane_id)[0], (0, 0, 1), atol=1e-6)
            # A minute tilt must use this footprint's plane equation, rather
            # than the collision half-space's million-metre lateral extent.
            editor.update_object(plane_id, rotation=tuple(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.001)))
            editor.before_frame()
            placed = editor.add_primitive("box", position=(3, 0, 0.5))
            editor.place_object(placed)
            editor.commit()
            self.assertGreater(editor.get_object(placed).position[2], 1.5)
            self.assertLess(editor.get_object(placed).position[2], 1.6)

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_safe_usd_placement_uses_bounds_and_rolls_back(self):
        """Place offset assets as a batch and retain unresolved authoring data on failed publication."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        editor = self.make_editor()
        editor.commit()
        original_model = editor.model
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "offset.usda"
            stage = Usd.Stage.CreateNew(str(source))
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            cube = UsdGeom.Cube.Define(stage, "/Asset/Cube")
            cube.CreateSizeAttr(0.4)
            cube.AddTranslateOp().Set(Gf.Vec3d(0.4, -0.3, -0.6))
            stage.GetRootLayer().Save()
            with (
                mock.patch.object(editor.viewer, "log_lines", wraps=editor.viewer.log_lines) as lines,
                mock.patch.object(
                    ModelBuilder, "finalize", autospec=True, side_effect=ModelBuilder.finalize
                ) as finalize,
            ):
                for x in (2.0, 2.1, 2.2):
                    deadline = time.monotonic() + 5
                    pivot = None
                    while pivot is None and time.monotonic() < deadline:
                        pivot = editor.preview_usd(source, position=(x, -1, 0), scale=(2, 0.5, 3), placement=True)
                        editor.before_frame()
                        self.assertFalse(editor.busy)
                        self.assertEqual(editor.objects, ())
                        self.assertIs(editor.model, original_model)
                        self.assertEqual(editor.rebuild_count, 1)
                        if pivot is None:
                            time.sleep(0.005)
                    self.assertIsNotNone(pivot, editor.error)
                    editor.render()
                    vertices = np.concatenate([array.numpy() for array in lines.call_args.args[1:3]])
                    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
                    np.testing.assert_allclose(upper - lower, (0.8, 0.2, 1.2), atol=1e-6)
                    np.testing.assert_allclose((lower + upper) / 2 - pivot, (0.8, -0.15, -1.8), atol=1e-6)
                    self.assertGreaterEqual(lower[2], 0)
                    self.assertLess(lower[2], 0.03)
                finalize.assert_not_called()
                editor.clear_preview()
            plane_path = Path(directory) / "plane.usda"
            plane_stage = Usd.Stage.CreateNew(str(plane_path))
            UsdGeom.SetStageMetersPerUnit(plane_stage, 1)
            UsdGeom.SetStageUpAxis(plane_stage, UsdGeom.Tokens.z)
            plane = UsdGeom.Plane.Define(plane_stage, "/Plane")
            UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
            plane_stage.GetRootLayer().Save()
            first = editor.add_usd(source, position=(0, 0, 0))
            second = editor.add_usd(source, position=(0, 0, 0))
            invalid = editor.add_usd(plane_path)
            for object_id in (first, second, invalid):
                editor.place_object(object_id)
            with self.assertRaisesRegex(ValueError, "bounded geometry"):
                editor.commit()
            self.assertIs(editor.model, original_model)
            self.assertEqual(editor.get_object(first).position, (0, 0, 0))
            self.assertEqual(editor.get_object(second).position, (0, 0, 0))
            editor.remove_object(invalid)
            explicit = editor.add_primitive("box", position=(4, 0, 0.123))
            editor.commit()
            self.assertGreater(editor.get_object(first).position[2], 0.8)
            self.assertGreater(editor.get_object(second).position[2] - editor.get_object(first).position[2], 0.4)
            self.assertEqual(editor.get_object(explicit).position, (4, 0, 0.123))
            self.assertEqual(editor.rebuild_count, 2)
            editor.step()
            for object_id in (first, second):
                index = editor.body_indices(object_id)[0]
                self.assertLess(np.linalg.norm(editor.state_0.body_qd.numpy()[index, :3]), 1.0)
            output = Path(directory) / "placed.usda"
            editor.save_usd(output)
            restored = self.make_editor()
            restored.load_usd(output)
            restored.commit()
            self.assertEqual(restored.objects, editor.objects)
            editor.reset_scene()
            for object_id in (first, second):
                np.testing.assert_allclose(
                    editor.get_pose(object_id)[0], editor.get_object(object_id).position, atol=1e-6
                )

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_usd_roundtrip_and_plain_geometry_get_real_physics(self):
        """Import a bare USD cube as a physical prop and reopen its authored scene."""
        editor = self.make_editor()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cube.usda"
            source.write_text(
                '#usda 1.0\n(\n metersPerUnit = 1\n upAxis = "Z"\n)\ndef Cube "Cube" {\n double size = 0.4\n}\n'
            )
            object_id = editor.add_usd(source, position=(0, 0, 2))
            editor.commit()
            self.assertEqual(editor.model.body_count, 1)
            editor.step()
            self.assertLess(editor.get_pose(object_id)[0][2], 2)
            for _ in range(90):
                editor.step()
            self.assertGreater(editor.get_pose(object_id)[0][2], 0.15)
            self.assertLess(editor.get_pose(object_id)[0][2], 0.30)
            output = Path(directory) / "scene.usda"
            editor.save_usd(output)
            restored = self.make_editor()
            restored.load_usd(output)
            restored.commit()
            self.assertEqual(restored.objects, editor.objects)
            self.assertEqual(restored.model.body_count, 1)
            np.testing.assert_allclose(restored.get_pose(object_id)[0], (0, 0, 2), atol=1e-6)

            model = editor.model
            with mock.patch.object(editor.viewer, "log_lines", wraps=editor.viewer.log_lines) as lines:
                editor.begin_edit()
                for index in range(16):
                    scale = (1 + index * 0.01,) * 3
                    editor.update_object(object_id, size=scale)
                    pivot = editor.preview_usd(source, position=editor.get_pose(object_id)[0], scale=scale)
                    self.assertFalse(editor.busy, "USD resize previews must not start intermediate preparation")
                    self.assertIsNotNone(pivot)
                    editor.before_frame()
                    editor.render()
                    vertices = np.concatenate([array.numpy() for array in lines.call_args.args[1:3]])
                    np.testing.assert_allclose(
                        vertices.max(axis=0) - vertices.min(axis=0), np.asarray(scale) * 0.4, atol=1e-6
                    )
                    self.assertIs(editor.model, model)
                    self.assertEqual(editor.rebuild_count, 1)
                editor.end_edit()
                editor.commit()
            self.assertEqual(editor.rebuild_count, 2)
            np.testing.assert_allclose(editor.model.shape_scale.numpy()[1], np.asarray(scale) * 0.2, atol=1e-6)

            # Undo snapshots created while metadata loads retain authored mass semantics.
            from pxr import Usd, UsdGeom, UsdPhysics

            geometry_source = Path(directory) / "native.usda"
            stage = Usd.Stage.CreateNew(str(geometry_source))
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            cube = UsdGeom.Cube.Define(stage, "/Body")
            cube.CreateSizeAttr(0.4)
            cube.AddTranslateOp().Set((0, 0, 0.4))
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(2)
            stage.GetRootLayer().Save()
            native_source = Path(directory) / "referenced.usda"
            root = Usd.Stage.CreateNew(str(native_source))
            UsdGeom.SetStageMetersPerUnit(root, 1)
            UsdGeom.SetStageUpAxis(root, UsdGeom.Tokens.z)
            root.DefinePrim("/Body").GetReferences().AddReference(geometry_source.name, "/Body")
            root.GetRootLayer().Save()
            native = self.make_editor()
            native_id = native.add_usd(native_source, position=(0, 0, 2))
            native.update_object(native_id, position=(0, 0, 3))
            native.commit()
            native.undo()
            native.before_frame()
            self.assertEqual(native.get_object(native_id).mass, 2)
            self.assertAlmostEqual(native.model.body_mass.numpy()[0], 2, places=6)
            self.assertEqual(native.rebuild_count, 1)
            placement = native.get_pose(native_id)[0]
            native.update_object(native_id, size=(2, 2, 2))
            native.commit()
            np.testing.assert_allclose(native.get_pose(native_id)[0], placement, atol=1e-6)
            explicit = self.make_editor()
            explicit_id = explicit.add_usd(native_source)
            explicit.update_object(explicit_id, mass=1)
            explicit.commit()
            self.assertEqual(explicit.get_object(explicit_id).mass, 1)
            self.assertAlmostEqual(explicit.model.body_mass.numpy()[0], 1, places=6)

            # A changed referenced layer must invalidate an unchanged root's template.
            root_stat = native_source.stat()
            cube.GetSizeAttr().Set(0.8)
            UsdPhysics.MassAPI(cube.GetPrim()).GetMassAttr().Set(5)
            stage.GetRootLayer().Save()
            self.assertEqual(
                (native_source.stat().st_mtime_ns, native_source.stat().st_size),
                (root_stat.st_mtime_ns, root_stat.st_size),
            )
            model, revision = native.model, native.revision
            pivot = None
            deadline = time.monotonic() + 30
            while pivot is None and time.monotonic() < deadline:
                pivot = native.preview_usd(native_source, position=(0, 0, 3), scale=(2, 2, 2))
                native.before_frame()
                time.sleep(0.001)
            self.assertIsNotNone(pivot, native.error)
            np.testing.assert_allclose(native._preview[2], (1.6, 1.6, 1.6), atol=1e-6)
            native.clear_preview()
            self.assertIs(native.model, model)
            self.assertEqual(native.revision, revision)
            self.assertEqual(native.get_object(native_id).mass, 2)
            self.assertAlmostEqual(native.model.body_mass.numpy()[0], 2, places=6)
            added = native.add_usd(native_source, position=(3, 0, 2))
            deadline = time.monotonic() + 30
            while native.pending and time.monotonic() < deadline:
                native.before_frame()
                if native.error:
                    break
                time.sleep(0.001)
            self.assertIsNone(native.error)
            self.assertFalse(native.pending)
            body = native.body_indices(added)[0]
            shape = np.flatnonzero(native.model.shape_body.numpy() == body)[0]
            np.testing.assert_allclose(native.model.shape_scale.numpy()[shape], (0.4, 0.4, 0.4), atol=1e-6)
            self.assertEqual(native.get_object(added).mass, 5)
            self.assertAlmostEqual(native.model.body_mass.numpy()[body], 5, places=6)

            reopened = Path(directory) / "reopened.usda"
            native.save_usd(reopened)
            cube.GetSizeAttr().Set(1.2)
            UsdPhysics.MassAPI(cube.GetPrim()).GetMassAttr().Set(7)
            stage.GetRootLayer().Save()
            model, builds = native.model, native.rebuild_count
            native.load_usd(reopened)
            self.assertTrue(native.pending)
            self.assertIs(native.model, model)
            deadline = time.monotonic() + 30
            while native.pending and time.monotonic() < deadline:
                native.before_frame()
                if native.error:
                    break
                time.sleep(0.001)
            self.assertIsNone(native.error)
            self.assertFalse(native.pending)
            self.assertIsNot(native.model, model)
            self.assertEqual(native.rebuild_count, builds + 1)
            body = native.body_indices(added)[0]
            shape = np.flatnonzero(native.model.shape_body.numpy() == body)[0]
            np.testing.assert_allclose(native.model.shape_scale.numpy()[shape], (0.6, 0.6, 0.6), atol=1e-6)
            self.assertEqual(native.get_object(added).mass, 7)
            self.assertAlmostEqual(native.model.body_mass.numpy()[body], 7, places=6)

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_topology_changes_preserve_fixed_joint_asset_controls(self):
        """Preserve compatible controls and initialize changed or replaced joints from native defaults."""
        from pxr import Usd, UsdPhysics

        editor = self.make_editor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.usda"
            self.write_jointed_mesh_asset(path)
            source_stage = Usd.Stage.Open(str(path))
            UsdPhysics.RigidBodyAPI(source_stage.GetPrimAtPath("/Assembly/Base")).CreateKinematicEnabledAttr(True)
            source_stage.GetRootLayer().Save()
            source_contents = path.read_bytes()
            leading = editor.add_primitive("box", position=(-3, 0, 1))
            cuda = get_cuda_test_devices()
            owner_device = cuda[0] if cuda else wp.get_device("cpu")
            with wp.ScopedDevice(owner_device):
                values = np.arange(16, dtype=np.float32)
                readback = wp.array(values, device=owner_device)
                asset = editor.add_usd(path, position=(0, 0, 2))
                deadline = time.monotonic() + 30
                while editor.pending and time.monotonic() < deadline:
                    np.testing.assert_array_equal(readback.numpy(), values)
                    self.assertEqual(wp.get_device(), owner_device)
                    editor.before_frame()
                    self.assertEqual(wp.get_device(), owner_device)
                    if editor.error:
                        break
                    time.sleep(0.001)
                self.assertIsNone(editor.error)
                self.assertFalse(editor.pending)
            mesh = next(source for source in editor.model.shape_source if isinstance(source, Mesh))
            np.testing.assert_allclose(np.ptp(mesh.vertices, axis=0), (0.1, 0.1, 0.1), atol=1e-6)
            self.assertEqual(mesh.indices.size, 36)
            bodies = list(editor.body_indices(asset))
            np.testing.assert_array_equal(
                editor.model.body_flags.numpy()[bodies], [BodyFlags.KINEMATIC, BodyFlags.DYNAMIC]
            )
            self.assertTrue(editor.get_object(asset).dynamic)
            model = editor.model
            editor.update_object(asset, dynamic=True)
            editor.before_frame()
            self.assertIsNot(editor.model, model)
            np.testing.assert_array_equal(
                editor.model.body_flags.numpy()[list(editor.body_indices(asset))], BodyFlags.DYNAMIC
            )
            old_pose = editor.state_0.body_q.numpy()[list(editor.body_indices(asset))].copy()
            expected = {}
            for field in ("joint_f", "joint_act", "joint_target_q", "joint_target_qd"):
                values = getattr(editor.control, field, None)
                if values is not None:
                    data = np.arange(values.size, dtype=np.float32) / 100 + 0.1
                    values.assign(data)
                    expected[field] = data[7 if field == "joint_target_q" else 6 :].copy()
            editor.remove_object(leading)
            trailing = editor.add_primitive("box", position=(3, 0, 1))
            editor.commit()
            np.testing.assert_array_equal(editor.state_0.body_q.numpy()[list(editor.body_indices(asset))], old_pose)
            for field, data in expected.items():
                actual = getattr(editor.control, field).numpy()
                np.testing.assert_array_equal(actual[: len(data)], data)
                np.testing.assert_array_equal(
                    actual[len(data) :], getattr(editor.model.control(), field).numpy()[len(data) :]
                )
            editor.remove_object(trailing)
            editor.commit()
            for field, data in expected.items():
                np.testing.assert_array_equal(getattr(editor.control, field).numpy(), data)
            self.assertEqual(editor.rebuild_count, 4)
            for _ in range(60):
                editor.step()
            self.assertTrue(np.isfinite(editor.state_0.body_q.numpy()).all())
            saved = Path(directory) / "scene.usda"
            editor.save_usd(saved)
            exported = ModelBuilder()
            exported.add_usd(str(saved), collapse_fixed_joints=False, load_visual_shapes=False)
            np.testing.assert_array_equal(exported.body_flags, BodyFlags.DYNAMIC)
            self.assertEqual(path.read_bytes(), source_contents)
            for schema, joint_type in (
                (UsdPhysics.RevoluteJoint, JointType.REVOLUTE),
                (UsdPhysics.PrismaticJoint, JointType.PRISMATIC),
            ):
                previous = editor.model
                joint = next(
                    index for index, label in enumerate(previous.joint_label) if label.endswith("/Assembly/Weld")
                )
                preserved = {}
                for field in expected:
                    values = getattr(editor.control, field)
                    starts = previous.joint_q_start if field == "joint_target_q" else previous.joint_qd_start
                    first, end = (int(value) for value in starts.numpy()[joint : joint + 2])
                    if end > first:
                        values[first:end].fill_(0.7)
                    preserved[field] = values.numpy()[:first].copy()
                schema.Define(source_stage, "/Assembly/Weld")
                source_stage.GetRootLayer().Save()
                editor.load_usd(saved)
                editor.commit()
                self.assertIsNot(editor.model, previous)
                joint = next(
                    index for index, label in enumerate(editor.model.joint_label) if label.endswith("/Assembly/Weld")
                )
                self.assertEqual(editor.model.joint_type.numpy()[joint], joint_type)
                defaults = editor.model.control()
                for field, data in preserved.items():
                    actual = getattr(editor.control, field).numpy()
                    np.testing.assert_array_equal(actual[: len(data)], data)
                    np.testing.assert_array_equal(actual[len(data) :], getattr(defaults, field).numpy()[len(data) :])
            # A different source with identical joint labels must not inherit the old asset's controls.
            replacement = Path(directory) / "replacement.usda"
            replacement.write_bytes(path.read_bytes())
            for field in expected:
                getattr(editor.control, field).fill_(0.9)
            document = Usd.Stage.Open(str(saved))
            prim = document.GetPrimAtPath("/World/Object_0")
            definition = json.loads(prim.GetCustomDataByKey("newtonSceneAuthoring"))
            prim.SetCustomDataByKey("newtonSceneAuthoring", json.dumps({**definition, "source": str(replacement)}))
            document.GetRootLayer().Save()
            editor.load_usd(saved)
            editor.commit()
            defaults = editor.model.control()
            for field in expected:
                np.testing.assert_array_equal(getattr(editor.control, field).numpy(), getattr(defaults, field).numpy())

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "Requires the USD importer dependency")
    def test_preparation_worker_failure_recovery_and_close(self):
        """Retain the running scene after a real worker exit, recover, and terminate owned workers on close."""
        editor = self.make_editor()
        editor.add_primitive("box", position=(3, 0, 1))
        editor.commit()
        model = editor.model
        with tempfile.TemporaryDirectory() as directory:
            failed_path = Path(directory) / "interrupted.usda"
            self.write_jointed_mesh_asset(failed_path)
            failed = editor.add_usd(failed_path)
            processes = list(editor._executor._processes.values())
            self.assertTrue(processes)
            self.assertTrue(editor.busy)
            for process in processes:
                process.terminate()
                process.join(5)
                self.assertFalse(process.is_alive())
            with self.assertLogs("newton", level="ERROR"):
                deadline = time.monotonic() + 10
                while editor.error is None and time.monotonic() < deadline:
                    editor.before_frame()
                    time.sleep(0.005)
            self.assertIsNotNone(editor.error)
            self.assertIs(editor.model, model)
            editor.step()
            self.assertTrue(np.isfinite(editor.state_0.body_q.numpy()).all())
            editor.remove_object(failed)
            recovered_path = Path(directory) / "recovered.usda"
            self.write_jointed_mesh_asset(recovered_path)
            recovered = editor.add_usd(recovered_path)
            deadline = time.monotonic() + 30
            while editor.pending and time.monotonic() < deadline:
                editor.before_frame()
                if editor.error:
                    break
                time.sleep(0.005)
            self.assertIsNone(editor.error)
            self.assertFalse(editor.pending)
            self.assertEqual(len(editor.body_indices(recovered)), 2)
            self.assertEqual(editor.model.body_count, 3)
            editor.preview_usd(recovered_path, position=(0, 0, 0), scale=(2, 2, 2))
            self.assertTrue(editor._preparing)
            processes = list(editor._executor._processes.values())
            manager = editor._executor._executor_manager_thread
            self.assertTrue(any(process.is_alive() for process in processes))
            editor.close()
            self.assertFalse(manager.is_alive())
            for process in processes:
                self.assertFalse(process.is_alive())

    def test_contact_capacity_recovers_before_solving(self):
        """Grow an undersized native contact pipeline while retaining the physical runtime."""
        editor = self.make_editor()
        editor.add_primitive("box", position=(0, 0, 0.5))
        editor.commit()
        model, solver, control = editor.model, editor.solver, editor.control
        states = {id(editor.state_0), id(editor.state_1)}
        poses = {id(editor.state_0.body_q), id(editor.state_1.body_q)}
        undersized = CollisionPipeline(model, rigid_contact_max=1)
        editor._runtime.pipeline = editor.collision_pipeline = undersized
        editor._runtime.contacts = editor.contacts = undersized.contacts()
        editor.viewer.set_collision_pipeline(undersized)
        with self.assertLogs("newton", level="WARNING") as messages:
            editor.step()
        self.assertTrue(any("Expanded rigid contact capacity" in line for line in messages.output))
        self.assertGreater(editor.contacts.rigid_contact_max, 1)
        self.assertGreaterEqual(int(editor.contacts.rigid_contact_count.numpy()[0]), 4)
        self.assertLessEqual(int(editor.contacts.rigid_contact_count.numpy()[0]), editor.contacts.rigid_contact_max)
        self.assertEqual(model.rigid_contact_max, editor.contacts.rigid_contact_max)
        self.assertEqual(
            model._get_contact_capacity(), (editor.contacts.rigid_contact_max, editor.contacts.soft_contact_max)
        )
        self.assertIs(editor.collision_pipeline, editor._runtime.pipeline)
        self.assertIs(editor.contacts, editor._runtime.contacts)
        self.assertIs(editor.model, model)
        self.assertIs(editor.solver, solver)
        self.assertIs(editor.control, control)
        self.assertEqual({id(editor.state_0), id(editor.state_1)}, states)
        self.assertEqual({id(editor.state_0.body_q), id(editor.state_1.body_q)}, poses)
        self.assertTrue(np.isfinite(editor.state_0.body_q.numpy()).all())
        self.assertEqual(editor.rebuild_count, 1)

    @unittest.skipUnless(
        importlib.util.find_spec("mujoco") and importlib.util.find_spec("mujoco_warp"), "Requires MuJoCo"
    )
    def test_mujoco_cpu_edits_and_native_contacts(self):
        """Use lazy native CPU physics, preserving authored edits and exported contacts."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        editor = SceneEditor(ViewerNull(), device="cpu", solver="mujoco")
        self.addCleanup(editor.close)
        editor.commit()
        editor.step()
        self.assertIsNone(editor.solver)
        support = editor.add_primitive("box", position=(0, 0, 0.25), size=(2, 2, 0.5), dynamic=False)
        box = editor.add_primitive("box", position=(0, 0, 0.74), size=(0.5, 0.5, 0.5))
        editor.commit()
        self.assertTrue(editor.model.device.is_cpu)
        self.assertIsNone(editor.solver)
        model, control = editor.model, editor.control
        states = {id(editor.state_0), id(editor.state_1)}
        builds = editor.rebuild_count
        with mock.patch.object(editor, "_collide", side_effect=AssertionError("CPU physics uses native contacts")):
            editor.step()
            solver = editor.solver
            self.assertTrue(solver.use_mujoco_cpu)
            self.assertIsNone(editor._runtime.observables)
            self.assertFalse(editor._runtime.solver_graphs)
            self.assertGreater(solver.mj_data.ncon, 0)
            self.assertEqual(int(editor.contacts.rigid_contact_count.numpy()[0]), solver.mj_data.ncon)

            rotation = tuple(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.4))
            editor.update_object(box, position=(0.2, 0, 2.2), rotation=rotation, friction=0.82)
            editor.update_object(support, position=(0.1, 0, 0.3))
            editor.before_frame()
            body = editor.body_indices(box)[0]
            shape = np.flatnonzero(model.shape_body.numpy() == body)[0]
            geoms = solver.mjc_geom_to_newton_shape.numpy().reshape(-1) == shape
            self.assertTrue(geoms.any())
            np.testing.assert_allclose(solver.mj_model.geom_friction[geoms, 0], 0.82)
            solver.mj_data.qacc_warmstart[:] = 123
            editor.reset_scene()
            np.testing.assert_array_equal(solver.mj_data.qacc_warmstart, 0)
            np.testing.assert_allclose(editor.get_pose(box)[0], (0.2, 0, 2.2), atol=1e-6)
            editor.step()
            np.testing.assert_allclose(editor.get_pose(box)[0][:2], (0.2, 0), atol=1e-6)
            self.assertLess(editor.get_pose(box)[0][2], 2.2)
            np.testing.assert_allclose(editor.get_pose(support)[0], (0.1, 0, 0.3), atol=1e-6)
            self.assertIs(editor.model, model)
            self.assertIs(editor.solver, solver)
            self.assertIs(editor.control, control)
            self.assertEqual({id(editor.state_0), id(editor.state_1)}, states)
            self.assertEqual(editor.rebuild_count, builds)

            # Contact display storage may grow without replacing native physics.
            pipeline = CollisionPipeline(model, rigid_contact_max=1, publish_capacity=False)
            editor._runtime.pipeline = pipeline
            editor._runtime.contacts = pipeline.contacts()
            editor._publish(editor._runtime)
            editor.update_object(box, position=(0.1, 0, 0.79))
            editor.before_frame()
            editor.step()
            self.assertIs(editor.solver, solver)
            self.assertGreater(editor.contacts.rigid_contact_max, 1)
            self.assertEqual(model.rigid_contact_max, editor.contacts.rigid_contact_max)
            self.assertGreater(int(editor.contacts.rigid_contact_count.numpy()[0]), 0)
            np.testing.assert_array_equal(editor._runtime.contacts.contact_counters.numpy()[2:], 0)

        before = editor.state_0.body_q.numpy().copy()
        editor.add_primitive("sphere", position=(3, 0, 2), size=(0.5, 0.5, 0.5))
        editor.before_frame()
        self.assertIsNone(editor.solver)
        np.testing.assert_array_equal(editor.state_0.body_q.numpy()[: len(before)], before)
        editor.undo()
        editor.before_frame()
        np.testing.assert_array_equal(editor.state_0.body_q.numpy(), before)

        if importlib.util.find_spec("pxr"):
            from pxr import Usd, UsdGeom, UsdPhysics

            path = Path(directory.name) / "platform.usda"
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            cube = UsdGeom.Cube.Define(stage, "/Platform")
            cube.CreateSizeAttr(1)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            stage.GetRootLayer().Save()
            platform = editor.add_usd(path, position=(0, 0, 0.5))
            neighbor = editor.add_usd(path, position=(-6, 0, 0.5))
            for object_id in (platform, neighbor):
                editor.update_object(object_id, dynamic=False)
            editor.commit()
            editor.step()
            solver = editor.solver
            editor.update_object(platform, position=(3, 0, 0.5))
            editor.update_object(box, position=(3, 0, 1.24))
            editor.before_frame()
            for _ in range(40):
                editor.step()
            self.assertIs(editor.solver, solver)
            self.assertGreater(solver.mj_data.ncon, 0)
            self.assertAlmostEqual(editor.get_pose(box)[0][2], 1.25, delta=0.03)

    @unittest.skipUnless(importlib.util.find_spec("mujoco_warp"), "Requires MuJoCo Warp")
    def test_mujoco_edits_preserve_state_and_handle_capacity(self):
        """Keep CUDA authoring edits coherent with MuJoCo and stop before publishing incomplete physics."""
        devices = get_cuda_test_devices()[:1]
        if not devices:
            self.skipTest("Requires a CUDA device")
        plane_path = None
        if importlib.util.find_spec("pxr"):
            from pxr import Usd, UsdGeom

            directory = tempfile.TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            plane_path = Path(directory.name) / "plane.usda"
            stage = Usd.Stage.CreateNew(str(plane_path))
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.Plane.Define(stage, "/Surface")
            stage.GetRootLayer().Save()
        editor = SceneEditor(ViewerNull(), device=devices[0], solver="mujoco")
        self.addCleanup(editor.close)
        editor.commit()
        self.assertIsNone(editor.solver)
        editor.step()
        support = editor.add_primitive("box", position=(0, 0, 0.25), size=(2, 2, 0.5), dynamic=False)
        box = editor.add_primitive("box", position=(0, 0, 0.74), size=(0.5, 0.5, 0.5))
        plane = editor.add_usd(plane_path, position=(3, 0, -3)) if plane_path else None
        if plane is not None:
            editor.update_object(plane, dynamic=False)
        editor.commit()
        builds = editor.rebuild_count
        self.assertEqual(editor.solver_type, SceneEditor.Solver.MUJOCO)
        self.assertIsNone(editor.solver)
        self.assertFalse(editor._runtime.solver_graphs)
        np.testing.assert_array_equal(editor.contacts.contact_counters.numpy(), 0)
        model, control = editor.model, editor.control
        states = {id(editor.state_0), id(editor.state_1)}
        arrays = {editor.state_0.body_q.ptr, editor.state_1.body_q.ptr}
        rotation = tuple(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.4))
        editor.update_object(box, position=(0.2, 0, 2.2), rotation=rotation, friction=0.82)
        if plane is not None:
            editor.update_object(plane, position=(4, 0, -2), friction=0.6)
        editor.before_frame()
        self.assertIsNone(editor.solver)
        self.assertFalse(editor._runtime.solver_graphs)
        # Compare the first captured frame against four real eager native substeps.
        reference = [model.state(), model.state()]
        for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
            wp.copy(getattr(reference[0], field), getattr(editor.state_0, field))
        reference_solver, reference_observables = editor._create_solver(model, editor.contacts.rigid_contact_max)
        reference_pipeline = CollisionPipeline(model)
        reference_contacts = reference_pipeline.contacts()

        def compare_frame():
            for _ in range(editor.sim_substeps):
                reference[0].clear_forces()
                reference_pipeline.collide(reference[0], reference_contacts)
                reference_solver.step(
                    reference[0],
                    reference[1],
                    editor.control,
                    reference_contacts,
                    editor.frame_dt / editor.sim_substeps,
                    observables=reference_observables,
                )
                reference.reverse()
            editor.step()
            for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    getattr(editor.state_0, field).numpy(), getattr(reference[0], field).numpy(), atol=1e-6
                )
            np.testing.assert_array_equal(editor._runtime.observables.overflow.numpy(), 0)
            self.assertEqual(len(editor._runtime.solver_graphs), 2)

        compare_frame()
        solver = editor.solver
        self.assertIsInstance(solver, SolverMuJoCo)
        graphs = {id(entry[0]) for entry in editor._runtime.solver_graphs.values()}
        body = editor.body_indices(box)[0]
        shape = np.flatnonzero(model.shape_body.numpy() == body)[0]
        geoms = solver.mjc_geom_to_newton_shape.numpy() == shape
        self.assertTrue(geoms.any())
        np.testing.assert_allclose(solver.mjw_model.geom_friction.numpy()[..., 0][geoms], 0.82)
        if plane is not None:
            shape = next(i for i, label in enumerate(model.shape_label) if label.startswith(f"objects/{plane}/"))
            self.assertEqual(model.shape_body.numpy()[shape], -1)
            self.assertTrue(model.shape_flags.numpy()[shape] & int(ShapeFlags.COLLIDE_SHAPES))
            geoms = solver.mjc_geom_to_newton_shape.numpy() == shape
            np.testing.assert_allclose(model.shape_transform.numpy()[shape, :3], (4, 0, -2))
            np.testing.assert_allclose(solver.mjw_model.geom_pos.numpy()[geoms], ((4, 0, -2),))
        self.assertIs(editor.model, model)
        self.assertIs(editor.solver, solver)
        self.assertIs(editor.control, control)
        self.assertEqual({id(editor.state_0), id(editor.state_1)}, states)
        self.assertEqual({editor.state_0.body_q.ptr, editor.state_1.body_q.ptr}, arrays)
        self.assertEqual(editor.rebuild_count, builds)
        np.testing.assert_allclose(editor.get_pose(box)[0][:2], (0.2, 0), atol=1e-6)
        self.assertGreater(editor.get_pose(box)[0][2], 2.19)
        np.testing.assert_allclose(editor.get_pose(box)[1], rotation, atol=1e-6)
        np.testing.assert_allclose(editor.get_pose(support)[0], (0, 0, 0.25), atol=1e-6)

        # Replay both State directions across a real zero-to-contact transition.
        editor.update_object(box, position=(0.2, 0, 0.83))
        editor.before_frame()
        for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
            wp.copy(getattr(reference[0], field), getattr(editor.state_0, field))
        reference_solver.reset(reference[0], flags=0)
        self.assertEqual(int(editor.contacts.rigid_contact_count.numpy()[0]), 0)
        for _ in range(12):
            compare_frame()
        self.assertGreater(int(editor.contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual({id(entry[0]) for entry in editor._runtime.solver_graphs.values()}, graphs)

        # Rebinding storage on the same owners must redirect captured reads and writes.
        editor._runtime.state_1.body_q = wp.clone(editor._runtime.state_1.body_q)
        compare_frame()
        joint = int(np.flatnonzero(model.joint_child.numpy() == body)[0])
        force = control.joint_f.numpy().copy()
        force[int(model.joint_qd_start.numpy()[joint])] = 20
        control.joint_f = wp.array(force, dtype=control.joint_f.dtype, device=editor.device)
        compare_frame()
        self.assertGreater(float(reference[0].body_qd.numpy()[body, 0]), 0.01)
        editor._runtime.observables.overflow = wp.clone(editor._runtime.observables.overflow)
        editor._runtime.observables.overflow.fill_(1)
        compare_frame()
        control.joint_f.zero_()
        editor.update_object(box, position=(0.2, 0, 2.2))
        editor.before_frame()
        editor.step()

        # A placed addition and its Undo each rebuild once without rewinding the falling box.
        before_q = editor.state_0.body_q.numpy()[body].copy()
        before_qd = editor.state_0.body_qd.numpy()[body].copy()
        editor.begin_edit()
        added = editor.add_primitive("box", position=(0.2, 0, 2.2), size=(0.5, 0.5, 0.5))
        editor.place_object(added)
        editor.end_edit()
        editor.before_frame()
        self.assertEqual(editor.rebuild_count, builds + 1)
        self.assertGreater(editor.get_pose(added)[0][2], 2.69)
        np.testing.assert_array_equal(editor.state_0.body_q.numpy()[editor.body_indices(box)[0]], before_q)
        np.testing.assert_array_equal(editor.state_0.body_qd.numpy()[editor.body_indices(box)[0]], before_qd)
        editor.undo()
        editor.before_frame()
        self.assertEqual(editor.rebuild_count, builds + 2)
        np.testing.assert_array_equal(editor.state_0.body_q.numpy()[editor.body_indices(box)[0]], before_q)
        np.testing.assert_array_equal(editor.state_0.body_qd.numpy()[editor.body_indices(box)[0]], before_qd)
        self.assertIsNone(editor.solver)
        self.assertFalse(editor._runtime.solver_graphs)
        editor.step()
        solver = editor.solver
        graphs = {id(entry[0]) for entry in editor._runtime.solver_graphs.values()}
        solver.mjw_data.qacc_warmstart.fill_(123)
        editor.reset_scene()
        self.assertIs(editor.solver, solver)
        np.testing.assert_array_equal(solver.mjw_data.qacc_warmstart.numpy(), 0)
        np.testing.assert_allclose(editor.get_pose(box)[0], (0.2, 0, 2.2), atol=1e-6)
        np.testing.assert_array_equal(editor.state_0.body_qd.numpy(), 0)
        editor.step()
        self.assertGreater(editor.get_pose(box)[0][2], 2.19)
        self.assertEqual({id(entry[0]) for entry in editor._runtime.solver_graphs.values()}, graphs)

        # Both native contact buffers must grow while physical storage remains owned and stable.
        editor.update_object(box, position=(0, 0, 0.74), rotation=(0, 0, 0, 1))
        editor.before_frame()
        model, control = editor.model, editor.control
        states = {id(editor.state_0), id(editor.state_1)}
        arrays = {editor.state_0.body_q.ptr, editor.state_1.body_q.ptr}
        pipeline = CollisionPipeline(model, rigid_contact_max=1)
        editor._runtime.pipeline = editor.collision_pipeline = pipeline
        editor._runtime.contacts = editor.contacts = pipeline.contacts()
        editor.viewer.set_collision_pipeline(pipeline)
        limited = SolverMuJoCo(
            model, use_mujoco_contacts=False, nconmax=1, strict_capacity=True, kinematic_switchable_roots=False
        )
        self.assertEqual(limited.get_max_contact_count(), 1)
        editor._runtime.solver = editor.solver = limited
        editor._runtime.observables = limited.observables({limited.ObservableKind.OVERFLOW})
        editor.viewer.set_solver(limited)
        editor.step()
        self.assertIsNot(editor.solver, limited)
        self.assertGreaterEqual(editor.contacts.rigid_contact_max, 4)
        self.assertGreaterEqual(editor.solver.get_max_contact_count(), editor.contacts.rigid_contact_max)
        np.testing.assert_array_equal(editor._runtime.observables.overflow.numpy(), 0)
        self.assertIs(editor.model, model)
        self.assertIs(editor.control, control)
        self.assertEqual({id(editor.state_0), id(editor.state_1)}, states)
        self.assertEqual({editor.state_0.body_q.ptr, editor.state_1.body_q.ptr}, arrays)
        self.assertEqual(editor.rebuild_count, builds + 2)

        # Compile six weld rows without contacts, then exceed that real constraint capacity.
        editor.update_object(box, position=(0, 0, 2.2))
        editor.before_frame()
        limited = SolverMuJoCo(
            model,
            use_mujoco_contacts=False,
            nconmax=editor.contacts.rigid_contact_max,
            njmax=6,
            strict_capacity=True,
            kinematic_switchable_roots=False,
        )
        self.assertEqual(limited.mjw_data.njmax, 6)
        editor._runtime.solver = editor.solver = limited
        editor._runtime.observables = limited.observables({limited.ObservableKind.OVERFLOW})
        editor.viewer.set_solver(limited)
        editor.update_object(box, position=(0, 0, 0.74))
        editor.before_frame()
        incoming = editor.state_0
        before = [
            array.numpy().copy() for array in (incoming.body_q, incoming.body_qd, incoming.joint_q, incoming.joint_qd)
        ]
        with self.assertLogs("newton", level="ERROR"), self.assertRaises(RuntimeError):
            editor.step()
        self.assertIn("overflow", editor.error.lower())
        self.assertTrue(editor._runtime.observables.overflow.numpy().any())
        self.assertIs(editor.state_0, incoming)
        self.assertIs(editor._runtime.state_0, incoming)
        for array, expected in zip(
            (incoming.body_q, incoming.body_qd, incoming.joint_q, incoming.joint_qd), before, strict=True
        ):
            np.testing.assert_array_equal(array.numpy(), expected)
        editor.begin_edit()
        for obj in editor.objects:
            editor.remove_object(obj.id)
        editor.end_edit()
        editor.before_frame()
        self.assertEqual(editor.model.body_count, 0)
        self.assertIsNone(editor.solver)
        self.assertEqual(editor.solver_type, SceneEditor.Solver.MUJOCO)
        editor.step()
        sphere = editor.add_primitive("sphere", position=(0, 0, 0.25), size=(0.5, 0.5, 0.5))
        editor.before_frame()
        self.assertIsNone(editor.solver)
        body = editor.body_indices(sphere)[0]
        editor.state_0.body_qd.assign(np.array([[1, 0, 0, 0, 4, 0]], dtype=np.float32))
        eval_ik(editor.model, editor.state_0, editor.state_0.joint_q, editor.state_0.joint_qd)
        for _ in range(240):
            editor.step()
        self.assertIsInstance(editor.solver, SolverMuJoCo)
        self.assertTrue(np.isfinite(editor.state_0.body_q.numpy()).all())
        self.assertLess(editor.state_0.body_qd.numpy()[body, 0], 0.7)
        self.assertGreater(editor.state_0.body_qd.numpy()[body, 0], 0)
        np.testing.assert_array_equal(editor._runtime.observables.overflow.numpy(), 0)

    def test_empty_scene_and_invalid_values(self):
        """Keep an empty ground usable and reject invalid edits before changing the document."""
        editor = self.make_editor()
        editor.commit()
        editor.step()
        editor.reset_scene()
        self.assertEqual(editor.model.body_count, 0)
        for kind, size in (
            ("sphere", (1, 2, 1)),
            ("sphere", (1, 1.000001, 1)),
            ("capsule", (1, 1, 0.5)),
            ("box", (-1, 1, 1)),
        ):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                editor.add_primitive(kind, size=size)
        self.assertFalse(editor.objects)
        self.assertEqual(editor.rebuild_count, 1)
        normalized = editor.add_primitive("box")
        with self.assertRaises(ValueError):
            editor.update_object(normalized, rotation=(0, 0, 0, 0))
        editor.update_object(normalized, rotation=(0, 0, 1.0e300, 1.0e300))
        np.testing.assert_allclose(editor.get_object(normalized).rotation, (0, 0, np.sqrt(0.5), np.sqrt(0.5)))


if __name__ == "__main__":
    unittest.main()
