# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check portable collision-fidelity inputs without constructing a solver."""

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


class TestCollisionFidelityAssets(unittest.TestCase):
    """Preserve the catalog and the source geometry used by the scenes."""

    def test_catalog_is_lightweight_and_preserves_all_cases(self):
        """Listing cases must neither initialize Warp nor download assets."""
        code = (
            "import sys; sys.path.insert(0, 'asv'); from collision_fidelity.catalog import load_cases; "
            "c=load_cases(); assert len(c)==45; "
            "assert c['r2_franka_duck']['support']['status']=='unsupported'; "
            "assert 'warp' not in sys.modules; assert 'newton' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_native_jacket_geometry(self):
        """The shipped jacket preserves every native vertex and triangle."""
        from collision_fidelity.jacket_asset import load_native_jacket  # noqa: PLC0415

        mesh = load_native_jacket()
        self.assertEqual(mesh.vertices.shape, (33703, 3))
        self.assertEqual(mesh.triangles.shape, (66930, 3))
        digest = hashlib.sha256(mesh.vertices.tobytes() + mesh.triangles.tobytes()).hexdigest()
        self.assertEqual(digest, "5aaf165379f3b3bfab613c05719ea98da3e73dc16252873a852db5d33294b8e8")

    def test_procedural_geometry_matches_manifest(self):
        """Every original procedural asset retains its coarse simulation mesh."""
        from collision_fidelity import assets  # noqa: PLC0415

        manifest = json.loads((Path(assets.__file__).with_name("assets") / "manifest.json").read_text())
        expected = manifest["procedural_geometry_sha256"]
        self.assertEqual(set(expected), set(assets.CLOTH_NAMES + assets.SOLID_NAMES))
        for name, digest in expected.items():
            factory = assets.cloth_mesh if name in assets.CLOTH_NAMES else assets.solid_mesh
            with self.subTest(asset=name):
                self.assertEqual(factory(name, resolution="coarse").metadata["geometry_sha256"], digest)


if __name__ == "__main__":
    unittest.main()
