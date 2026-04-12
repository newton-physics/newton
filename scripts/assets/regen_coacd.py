"""Regenerate CoACD decomposition for the dish rack STL.

Tighter threshold (0.03) + uncapped hull count for cleaner peg separation.
Writes to scripts/assets/dish-rack-2.snapshot.2/coacd/.
"""

import json
import shutil
from pathlib import Path

import coacd
import numpy as np
import trimesh

THRESHOLD = 0.03
MAX_CONVEX_HULL = -1

ASSET_ROOT = Path(__file__).resolve().parent / "dish-rack-2.snapshot.2"
STL_PATH = ASSET_ROOT / "Escurridor platos.STL"
OUT_DIR = ASSET_ROOT / "coacd"


def main() -> None:
    raw = trimesh.load(str(STL_PATH), process=False, force="mesh")
    verts = np.asarray(raw.vertices, dtype=np.float64)
    faces = np.asarray(raw.faces, dtype=np.int32)
    print(f"source mesh: {verts.shape[0]} verts, {faces.shape[0]} faces")

    mesh = coacd.Mesh(verts, faces)
    print(f"running CoACD: threshold={THRESHOLD}, max_convex_hull={MAX_CONVEX_HULL}")
    parts = coacd.run_coacd(
        mesh,
        threshold=THRESHOLD,
        max_convex_hull=MAX_CONVEX_HULL,
    )
    print(f"CoACD produced {len(parts)} parts")

    if OUT_DIR.exists():
        for p in OUT_DIR.glob("part_*.obj"):
            p.unlink()
        manifest = OUT_DIR / "manifest.json"
        if manifest.exists():
            manifest.unlink()
    else:
        OUT_DIR.mkdir(parents=True)

    for i, (pv, pf) in enumerate(parts):
        path = OUT_DIR / f"part_{i:03d}.obj"
        with path.open("w") as fh:
            for v in pv:
                fh.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for f in pf:
                fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "n_parts": len(parts),
                "source": str(STL_PATH.relative_to(ASSET_ROOT.parents[1])),
                "threshold": THRESHOLD,
                "max_convex_hull": MAX_CONVEX_HULL,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(parts)} parts to {OUT_DIR}")


if __name__ == "__main__":
    main()
