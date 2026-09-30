# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test USD scale extraction across affine transforms and authored scale ops."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton.usd as usd
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdTransforms(unittest.TestCase):
    def test_scale_does_not_decompose_rotation(self):
        """Extract scale without constructing and decomposing an unused rotation."""
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Xform.Define(stage, "/Transform")
        prim.AddScaleOp().Set(Gf.Vec3f(2.0, 3.0, 4.0))
        with mock.patch.object(wp, "transform_decompose", wraps=wp.transform_decompose) as decompose:
            np.testing.assert_array_equal(usd.get_scale(prim.GetPrim()), (2.0, 3.0, 4.0))
        decompose.assert_not_called()

    def test_scale_matches_transform_decomposition(self):
        """Preserve scale magnitudes for rotated, mirrored, sheared, and singular matrices."""
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Xform.Define(stage, "/Transform")
        op = prim.AddTransformOp()
        rng = np.random.default_rng(17)
        matrices = [np.eye(4), np.diag([-2.0, 3.0, 0.0, 1.0])]
        for _ in range(32):
            matrix = np.eye(4)
            matrix[:3, :3] = rng.normal(size=(3, 3))
            matrix[3, :3] = rng.normal(size=3)
            matrices.append(matrix)

        for matrix in matrices:
            with self.subTest(matrix=matrix):
                op.Set(Gf.Matrix4d(*matrix.flatten()))
                expected = wp.transform_decompose(wp.mat44(matrix.astype(np.float32).T))[2]
                cache = UsdGeom.XformCache()
                for local in (False, True):
                    np.testing.assert_array_equal(
                        usd.get_scale(prim.GetPrim(), local=local, xform_cache=cache), expected
                    )

    def test_scale_preserves_authored_signs_and_reset_stack(self):
        """Respect inherited negative scales, inverse ops, and reset transform stacks."""
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/Root")
        root.AddScaleOp().Set(Gf.Vec3f(-2.0, 3.0, 4.0))
        child = UsdGeom.Xform.Define(stage, "/Root/Child")
        child.AddScaleOp().Set(Gf.Vec3f(0.5, -2.0, 1.5))
        cancelled = child.AddScaleOp(opSuffix="cancelled")
        cancelled.Set(Gf.Vec3f(-4.0, 2.0, 0.5))
        child.AddScaleOp(opSuffix="cancelled", isInverseOp=True)
        reset = UsdGeom.Xform.Define(stage, "/Root/Child/Reset")
        reset.SetResetXformStack(True)
        reset.AddScaleOp().Set(Gf.Vec3f(5.0, -6.0, 7.0))
        leaf = UsdGeom.Xform.Define(stage, "/Root/Child/Reset/Leaf")
        leaf.AddScaleOp().Set(Gf.Vec3f(-0.5, 2.0, 0.0))
        cache = UsdGeom.XformCache()

        for xform, local, expected in (
            (child, True, (0.5, -2.0, 1.5)),
            (child, False, (-1.0, -6.0, 6.0)),
            (reset, False, (5.0, -6.0, 7.0)),
            (leaf, False, (-2.5, -12.0, 0.0)),
        ):
            with self.subTest(path=xform.GetPath(), local=local):
                np.testing.assert_array_equal(usd.get_scale(xform.GetPrim(), local=local), expected)
                np.testing.assert_array_equal(usd.get_scale(xform.GetPrim(), local=local, xform_cache=cache), expected)

    def test_scale_preserves_signs_with_nonuniform_rotated_ancestors(self):
        """Keep authored signs when rotated nonuniform ancestors introduce shear."""
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/Root")
        root.AddScaleOp().Set(Gf.Vec3f(-2.0, 3.0, 4.0))
        child = UsdGeom.Xform.Define(stage, "/Root/Child")
        child.AddRotateXYZOp().Set(Gf.Vec3f(19.0, 37.0, -23.0))
        child.AddScaleOp().Set(Gf.Vec3f(0.5, -2.0, 1.5))
        prim = child.GetPrim()
        matrix = np.array(child.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float32)
        magnitudes = wp.transform_decompose(wp.mat44(matrix.T))[2]
        expected = np.asarray(magnitudes) * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(usd.get_scale(prim, local=False), expected)
        np.testing.assert_array_equal(usd.get_scale(prim, local=False, xform_cache=UsdGeom.XformCache()), expected)


if __name__ == "__main__":
    unittest.main()
