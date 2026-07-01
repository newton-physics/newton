# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end import of the bundled mixed-deformable scene (cables + cloth + volumes).

The fixture mirrors the shape of Isaac Lab generated assets -- each deformable is a
``PhysicsDeformableBodyAPI`` Xform with a simulation-geometry child and a bound family
material -- without external references, so the happy path stays covered by a
repository-owned regression.
"""

import os
import unittest

import newton
from newton.tests.unittest_utils import USD_AVAILABLE

_ASSET = os.path.join(os.path.dirname(__file__), "assets", "deformables_mixed.usda")


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableMixed(unittest.TestCase):
    """Mixed cable/cloth/volume scene imports, groups, and finalizes in one pass."""

    def test_mixed_scene_imports_and_finalizes(self):
        builder = newton.ModelBuilder()
        builder.add_usd(_ASSET)
        model = builder.finalize()

        self.assertEqual(model.cable_count, 2)
        self.assertEqual(model.cloth_count, 1)
        self.assertEqual(model.soft_count, 2)

        # Each cable: 3 segments -> 3 capsule bodies wrapped in its own articulation.
        for path in ("/World/CableA/sim", "/World/CableB/sim"):
            b0, b1 = model.cable_body_range(model.cable_index(path))
            self.assertEqual(b1 - b0, 3)
            self.assertIn(f"{path}_articulation", model.articulation_label)
        self.assertEqual(model.body_count, 6)
        self.assertEqual(model.articulation_count, 2)

        # Cloth: 4 particles / 2 triangles; volumes: 4 particles / 1 tet each, disjoint.
        cloth = model.cloth_index("/World/Cloth/sim")
        p0, p1 = model.cloth_particle_range(cloth)
        self.assertEqual(p1 - p0, 4)
        self.assertEqual(model.cloth_tri_range(cloth), (0, 2))
        ranges = [model.soft_particle_range(model.soft_index(f"/World/Soft{s}/sim")) for s in ("A", "B")]
        self.assertEqual([end - start for start, end in ranges], [4, 4])
        self.assertNotEqual(ranges[0], ranges[1])
        self.assertEqual(model.particle_count, 12)
        self.assertEqual(model.tet_count, 2)


if __name__ == "__main__":
    unittest.main()
