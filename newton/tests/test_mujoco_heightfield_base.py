# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify that MJCF heightfield conversion preserves the collision base."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

try:
    import mujoco
except ImportError:
    mujoco = None


def _scene(base: float) -> str:
    """Build a self-contained MJCF scene with an authored collision base."""
    return f"""<mujoco>
        <asset><hfield name="terrain" nrow="3" ncol="3" size="1 2 0.5 {base}"
            elevation="0 0 0 0 1 0 0 0 0"/></asset>
        <worldbody>
            <geom name="ground" type="hfield" hfield="terrain"/>
            <body name="box" pos="0 0 1"><freejoint/>
                <geom type="box" size="0.04 0.025 0.02" mass="1"/>
            </body>
        </worldbody>
    </mujoco>"""


class TestMJCFHeightfieldSize(unittest.TestCase):
    def test_reject_invalid_mjcf_size(self):
        """Reject missing, malformed, and nonpositive authored heightfield sizes."""
        sizes = (
            None,
            "",
            "1 2 0.5",
            "1 2 0.5 0.1 2",
            "1 2 0.5 0",
            "1 2 0.5 -0.1",
            "0 2 0.5 0.1",
            "1 -2 0.5 0.1",
            "1 2 0 0.1",
        )
        for size in sizes:
            with self.subTest(size=size):
                attribute = "" if size is None else f'size="{size}"'
                xml = _scene(0.1).replace('size="1 2 0.5 0.1"', attribute)
                if mujoco is not None:
                    with self.assertRaises(ValueError):
                        mujoco.MjModel.from_xml_string(xml)
                with self.assertRaisesRegex(ValueError, "heightfield.*terrain.*size"):
                    newton.ModelBuilder().add_mjcf(xml)

    def test_reject_nonfinite_mjcf_size(self):
        """Reject NaN and infinity in every authored heightfield size component."""
        for value in ("nan", "inf", "-inf"):
            for component in range(4):
                with self.subTest(value=value, component=component):
                    size = ["1", "2", "0.5", "0.1"]
                    size[component] = value
                    xml = _scene(0.1).replace('size="1 2 0.5 0.1"', f'size="{" ".join(size)}"')
                    with self.assertRaisesRegex(ValueError, "heightfield.*terrain.*size"):
                        newton.ModelBuilder().add_mjcf(xml)


@unittest.skipIf(mujoco is None, "MuJoCo is not installed")
class TestMuJoCoHeightfieldBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Skip solver tests when either optional MuJoCo dependency is missing."""
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_mjcf_roundtrip_preserves_base(self):
        """Preserve authored base depths, including values below the old epsilon."""
        for base in (0.1, 0.35, 1.0e-6):
            with self.subTest(base=base):
                xml = _scene(base)
                source = mujoco.MjModel.from_xml_string(xml)
                builder = newton.ModelBuilder()
                builder.add_mjcf(xml)
                solver = SolverMuJoCo(builder.finalize(device="cpu"), use_mujoco_cpu=True)
                np.testing.assert_allclose(solver.mj_model.hfield_size, source.hfield_size, rtol=1.0e-6, atol=0)
                np.testing.assert_array_equal(solver.mj_model.hfield_data, source.hfield_data)
                ground_id = int(np.flatnonzero(solver.mj_model.geom_type == int(mujoco.mjtGeom.mjGEOM_HFIELD))[0])
                source_ground = source.geom("ground").id
                np.testing.assert_allclose(
                    solver.mj_model.geom_aabb[ground_id], source.geom_aabb[source_ground], rtol=1.0e-6, atol=1.0e-7
                )
                np.testing.assert_allclose(
                    solver.mj_model.geom_rbound[ground_id], source.geom_rbound[source_ground], rtol=1.0e-6
                )

    def test_mjcf_base_honors_import_and_shape_scale(self):
        """Scale imported base depth once with each applicable vertical scale."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(_scene(0.1), scale=2.0)
        ground = builder.shape_type.index(newton.GeoType.HFIELD)
        builder.shape_scale[ground] = wp.vec3(2.0, 3.0, 4.0)
        solver = SolverMuJoCo(builder.finalize(device="cpu"), use_mujoco_cpu=True)
        np.testing.assert_allclose(solver.mj_model.hfield_size[0], [4.0, 12.0, 4.0, 0.8], rtol=1.0e-6)

    def test_mjcf_bases_remain_distinct_across_builder_copies(self):
        """Keep each imported base attached to its shape when composing builders."""
        builder = newton.ModelBuilder()
        for i, base in enumerate((0.1, 0.35)):
            child = newton.ModelBuilder()
            child.add_mjcf(_scene(base))
            builder.add_builder(child, xform=wp.transform((4.0 * i, 0.0, 0.0), wp.quat_identity()), label_prefix=str(i))
        # Finalization warns for any two heightfields, even separated static ones.
        with self.assertWarnsRegex(UserWarning, "^Heightfield-vs-heightfield collision is not supported;"):
            model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        np.testing.assert_allclose(np.sort(solver.mj_model.hfield_size[:, 3]), [0.1, 0.35], rtol=1.0e-6)

    def test_native_heightfield_keeps_default_base(self):
        """Keep the existing default for native fields without an authored MJCF base."""
        for register in (False, True):
            with self.subTest(register=register):
                builder = newton.ModelBuilder()
                if register:
                    SolverMuJoCo.register_custom_attributes(builder)
                builder.add_shape_heightfield(
                    heightfield=newton.Heightfield(np.zeros((2, 2)), 2, 2), scale=(1.0, 1.0, 2.0)
                )
                body = builder.add_link()
                joint = builder.add_joint_free(body)
                builder.add_articulation([joint])
                builder.add_shape_sphere(body, radius=0.1)
                solver = SolverMuJoCo(builder.finalize(device="cpu"), use_mujoco_cpu=True)
                self.assertAlmostEqual(solver.mj_model.hfield_size[0, 3], 1.0e-4, delta=1.0e-10)

    @unittest.skipUnless(wp.is_cuda_available(), "MuJoCo Warp heightfield check requires CUDA")
    def test_gpu_cloned_worlds_preserve_base_and_bounds(self):
        """Preserve the authored collision base and bounds in the GPU model."""
        child = newton.ModelBuilder()
        child.add_mjcf(_scene(0.1))
        builder = newton.ModelBuilder()
        for _ in range(3):
            builder.add_world(child)
        # The builder also warns when the heightfields belong to separate worlds.
        with self.assertWarnsRegex(UserWarning, "^Heightfield-vs-heightfield collision is not supported;"):
            model = builder.finalize(device="cuda:0")
        solver = SolverMuJoCo(model, use_mujoco_cpu=False, separate_worlds=True)
        source = mujoco.MjModel.from_xml_string(_scene(0.1))
        np.testing.assert_allclose(solver.mjw_model.hfield_size.numpy()[0], source.hfield_size[0], rtol=1.0e-6)
        ground = int(np.flatnonzero(solver.mj_model.geom_type == int(mujoco.mjtGeom.mjGEOM_HFIELD))[0])
        bounds = solver.mjw_model.geom_aabb.numpy().reshape(-1, solver.mj_model.ngeom, 2, 3)[:, ground]
        expected_bounds = source.geom_aabb[source.geom("ground").id].reshape(2, 3)
        for actual in bounds:
            np.testing.assert_allclose(actual, expected_bounds, rtol=1.0e-6, atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
