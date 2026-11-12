# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test authored MuJoCo collision masks imported from USD."""

import unittest

import numpy as np

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE

if USD_AVAILABLE:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdCollisionMasks(unittest.TestCase):
    def _stage(self, body_masks):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        for body_index, masks in enumerate(body_masks):
            body = UsdGeom.Xform.Define(stage, f"/body{body_index}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            for shape_index, mask in enumerate(masks):
                sphere = UsdGeom.Sphere.Define(stage, f"/body{body_index}/shape{shape_index}")
                sphere.CreateRadiusAttr(0.1)
                prim = sphere.GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim)
                if mask is not None:
                    for name, value in zip(("mjc:contype", "mjc:conaffinity"), mask, strict=True):
                        prim.CreateAttribute(name, Sdf.ValueTypeNames.Int, custom=True).Set(value)
        return stage

    def _builder(self):
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        return builder

    def _solver(self, model):
        return SolverMuJoCo(model, use_mujoco_contacts=True, use_mujoco_cpu=True)

    def _masks_allow(self, solver, shape_a, shape_b):
        shape_to_geom = {
            int(shape): geom for geom, shape in enumerate(solver.mjc_geom_to_newton_shape.numpy()[0]) if shape >= 0
        }
        geom_a, geom_b = shape_to_geom[shape_a], shape_to_geom[shape_b]
        contype, conaffinity = solver.mj_model.geom_contype, solver.mj_model.geom_conaffinity
        return bool(
            (int(contype[geom_a]) & int(conaffinity[geom_b])) or (int(contype[geom_b]) & int(conaffinity[geom_a]))
        )

    def test_forward_authored_masks(self):
        """Forward raw asymmetric masks, including zero and signed high bits."""
        for masks in (((0, 1), (1, 0), (1, 0)), ((-2147483648, 0), (0, -2147483648), (-1, 0))):
            with self.subTest(masks=masks):
                builder = self._builder()
                result = builder.add_usd(self._stage([[mask] for mask in masks]), load_visual_shapes=False)
                model = builder.finalize(device="cpu")
                domains = model.mujoco.collision_mask_domain.numpy()
                self.assertTrue(np.all(domains >= 0))
                self.assertEqual(len(np.unique(domains)), 1)
                shapes = [result["path_shape_map"][f"/body{i}/shape0"] for i in range(3)]
                expected_pairs = {tuple(sorted((shapes[0], shapes[1]))), tuple(sorted((shapes[1], shapes[2])))}
                if masks[0][0] == 0:
                    expected_pairs = {tuple(sorted((shapes[0], shapes[1]))), tuple(sorted((shapes[0], shapes[2])))}
                self.assertEqual(set(map(tuple, model.shape_contact_pairs.numpy())), expected_pairs)
                solver = self._solver(model)
                mapping = solver.mjc_geom_to_newton_shape.numpy()[0]
                for geom, shape in enumerate(mapping):
                    if shape >= 0:
                        index = shapes.index(int(shape))
                        self.assertEqual(int(solver.mj_model.geom_contype[geom]), masks[index][0])
                        self.assertEqual(int(solver.mj_model.geom_conaffinity[geom]), masks[index][1])

    def test_filtered_pairs_narrow_masks(self):
        """Keep USD group, explicit pair, and mask exclusions during solver fallback."""
        stage = self._stage([[(1, 0), (2, 0)], [(0, 1), (0, 3)]])
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/body0/shape0")).CreateFilteredPairsRel().AddTarget(
            Sdf.Path("/body1/shape0")
        )
        group_a = UsdPhysics.CollisionGroup.Define(stage, "/groupA")
        group_b = UsdPhysics.CollisionGroup.Define(stage, "/groupB")
        group_a.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(Sdf.Path("/body0/shape1"))
        group_b.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(Sdf.Path("/body1/shape1"))
        group_a.CreateFilteredGroupsRel().AddTarget(group_b.GetPath())
        builder = self._builder()
        result = builder.add_usd(stage, load_visual_shapes=False)
        shapes = result["path_shape_map"]
        model = builder.finalize(device="cpu")
        expected = {
            tuple(sorted((shapes["/body0/shape0"], shapes["/body1/shape1"]))),
        }
        self.assertEqual(set(map(tuple, model.shape_contact_pairs.numpy())), expected)
        solver = self._solver(model)
        for a in range(2):
            for b in range(2):
                shape_a, shape_b = shapes[f"/body0/shape{a}"], shapes[f"/body1/shape{b}"]
                self.assertEqual(self._masks_allow(solver, shape_a, shape_b), a == 0 and b == 1)

    def test_separate_import_domains(self):
        """Keep repeated USD imports independent when compiling collision masks."""
        builder = self._builder()
        stage = self._stage([[(0, 1)], [(1, 0)], [(1, 0)]])
        builder.add_usd(stage, load_visual_shapes=False)
        builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize(device="cpu")
        domains = model.mujoco.collision_mask_domain.numpy()
        self.assertEqual(len(np.unique(domains)), 2)
        solver = self._solver(model)
        expected = set(map(tuple, model.shape_contact_pairs.numpy()))
        for a in range(model.shape_count):
            for b in range(a + 1, model.shape_count):
                self.assertEqual(self._masks_allow(solver, a, b), (a, b) in expected)

    def test_unauthored_masks_have_no_domain(self):
        """Leave unauthored shapes outside preserved collision mask domains."""
        builder = self._builder()
        result = builder.add_usd(self._stage([[(0, 1)], [None]]), load_visual_shapes=False)
        model = builder.finalize(device="cpu")
        shapes = result["path_shape_map"]
        domains = model.mujoco.collision_mask_domain.numpy()
        self.assertGreaterEqual(int(domains[shapes["/body0/shape0"]]), 0)
        self.assertEqual(int(domains[shapes["/body1/shape0"]]), -1)
        solver = self._solver(model)
        self.assertTrue(self._masks_allow(solver, shapes["/body0/shape0"], shapes["/body1/shape0"]))


if __name__ == "__main__":
    unittest.main()
