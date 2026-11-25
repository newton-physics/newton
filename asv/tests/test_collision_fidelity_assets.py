# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check portable collision-fidelity inputs without constructing a solver."""

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

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
        subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], check=True)

    def test_native_jacket_geometry(self):
        """The shipped jacket preserves every native vertex and triangle."""
        from collision_fidelity.jacket_asset import load_native_jacket  # noqa: PLC0415

        mesh = load_native_jacket()
        self.assertEqual(mesh.vertices.shape, (33703, 3))
        self.assertEqual(mesh.triangles.shape, (66930, 3))
        digest = hashlib.sha256(mesh.vertices.tobytes() + mesh.triangles.tobytes()).hexdigest()
        self.assertEqual(digest, "5aaf165379f3b3bfab613c05719ea98da3e73dc16252873a852db5d33294b8e8")

    def test_procedural_asset_mechanisms(self):
        """Keep scale, connected surfaces, garment openings and usable landmarks."""
        from collision_fidelity import assets  # noqa: PLC0415

        manifest = json.loads((Path(assets.__file__).with_name("assets") / "manifest.json").read_text())
        # Hashes describe an archived generation, not portable acceptance limits.
        self.assertEqual(set(manifest["procedural_geometry_sha256"]), set(assets.CLOTH_NAMES + assets.SOLID_NAMES))
        # Approximate dimensions [m] check input scale, not a physical pass limit.
        # Boundary loops preserve the mechanism independently of tessellation.
        expected = {
            "patch": ((0.1, 0.0, 0.1), 1),
            "towel": ((0.4, 0.0, 0.6), 1),
            "sheet": ((1.0, 0.0, 1.0), 1),
            "trousers": ((0.4, 0.05, 0.85), 3),
            "dress": ((0.76, 0.07, 1.0), 4),
            "tshirt": ((0.98, 0.07, 0.68), 4),
            "skirt": ((0.68, 0.09, 0.55), 2),
            "tote": ((0.3, 0.1, 0.50), 3),
            "pouch": ((0.15, 0.018, 0.2), 1),
            "gusset_bag": ((0.25, 0.12, 0.45), 3),
            "block": ((0.06, 0.04, 0.03), 0),
            "pad": ((0.12, 0.08, 0.01), 0),
            "sphere": ((0.05, 0.05, 0.05), 0),
            "capsule": ((0.035, 0.035, 0.08), 0),
            "torus": ((0.08, 0.08, 0.02), 0),
            "branched": ((0.12, 0.12, 0.024), 0),
        }
        self.assertEqual(set(expected), set(assets.CLOTH_NAMES + assets.SOLID_NAMES))
        for name, (dimensions, loop_count) in expected.items():
            factory = assets.cloth_mesh if name in assets.CLOTH_NAMES else assets.solid_mesh
            with self.subTest(asset=name):
                mesh = factory(name, resolution="coarse")
                self.assertTrue(assets.validate_asset(mesh)["valid"])
                v, t = mesh.vertices, mesh.triangles
                np.testing.assert_allclose(np.ptp(v, axis=0), dimensions, rtol=0.01, atol=1e-6)
                self.assertAlmostEqual(float(v[:, 2].min()), 0.0, places=6)
                edges, counts = np.unique(
                    np.sort(np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1),
                    axis=0,
                    return_counts=True,
                )
                adjacency = {}
                for a, b in edges[counts == 1]:
                    adjacency.setdefault(a, set()).add(b)
                    adjacency.setdefault(b, set()).add(a)
                self.assertTrue(all(len(neighbors) == 2 for neighbors in adjacency.values()))
                pending, loops = set(adjacency), []
                while pending:
                    stack, loop = [pending.pop()], set()
                    while stack:
                        node = stack.pop()
                        if node not in loop:
                            loop.add(node)
                            pending.discard(node)
                            stack.extend(adjacency[node] - loop)
                    loops.append(loop)
                self.assertEqual(len(loops), loop_count)
                # Count only surface vertices: solid meshes also have interior nodes.
                euler = len(np.unique(t)) - len(edges) + len(t)
                self.assertEqual(euler, 0 if name == "torus" else 2 - loop_count)
                for label in ("top", "bottom", "left", "right"):
                    ids = mesh.landmarks[label]
                    self.assertTrue(ids)
                    self.assertTrue(all(0 <= i < len(v) for i in ids))
                self.assertGreater(v[mesh.landmarks["top"], 2].mean(), v[mesh.landmarks["bottom"], 2].mean())
                self.assertLess(v[mesh.landmarks["left"], 0].max(), v[mesh.landmarks["right"], 0].min())
                if "seam" in mesh.landmarks:
                    centers_y = v[t, 1].mean(axis=1)
                    front = set(np.unique(t[centers_y < -1e-7]))
                    back = set(np.unique(t[centers_y > 1e-7]))
                    self.assertTrue(front and back)
                    self.assertEqual(front & back, set(mesh.landmarks["seam"]))
                    np.testing.assert_allclose(v[mesh.landmarks["seam"], 1], 0, atol=1e-7)
                if name in ("tote", "gusset_bag", "pouch"):
                    self.assertTrue(mesh.landmarks["rim"])
                    self.assertTrue(mesh.landmarks["bottom"])
                    handle = v[mesh.landmarks["handle_top"]]
                    if name == "pouch":
                        self.assertEqual(len(handle), 0)
                    else:
                        self.assertTrue(np.any(handle[:, 1] < 0) and np.any(handle[:, 1] > 0))
                        self.assertGreater(handle[:, 2].min(), v[mesh.landmarks["rim"], 2].max())

    def test_reject_invalid_element_indices(self):
        """Reject malformed indices before NumPy can wrap or dereference them."""
        from collision_fidelity.assets import solid_mesh, validate_asset  # noqa: PLC0415

        for field, width in (("triangles", 3), ("tetrahedra", 4)):
            for defect in ("negative", "out_of_range", "noninteger", "wrong_width"):
                with self.subTest(field=field, defect=defect):
                    mesh = solid_mesh("block", resolution="coarse")
                    indices = getattr(mesh, field).copy()
                    if defect == "negative":
                        indices[0, 0] = -1
                    elif defect == "out_of_range":
                        indices[0, 0] = len(mesh.vertices)
                    elif defect == "noninteger":
                        indices = indices.astype(float)
                    else:
                        indices = indices[:, : width - 1]
                    setattr(mesh, field, indices)
                    with self.assertRaisesRegex(ValueError, field):
                        validate_asset(mesh)


if __name__ == "__main__":
    unittest.main()
