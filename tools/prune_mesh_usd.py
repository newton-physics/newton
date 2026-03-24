#!/usr/bin/env python3
"""Prune a USD mesh (and any 3D Gaussian Splat point clouds) by removing all
triangles / splat particles outside a given radius.

All positions are transformed to world space before the radius test, so the
center and radius are always in world-space units regardless of which prim has
local-space transforms.

Usage:
    uv run tools/prune_mesh_usd.py <usd_file> <prim_path> <radius> [--center X Y Z] [--output <out_file>]

Example:
    uv run tools/prune_mesh_usd.py scene.usd /World/mesh_ 10.0
    uv run tools/prune_mesh_usd.py scene.usd /World/mesh_ 10.0 --center 1.0 2.0 0.0 --output pruned.usd
"""

import argparse
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, Vt


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _local_to_world(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    """Return the 4×4 local-to-world matrix for *prim_path* as a numpy array.

    USD uses row-vector convention: ``world = local_row_vec @ M``.
    """
    cache = UsdGeom.XformCache()
    prim = stage.GetPrimAtPath(prim_path)
    return np.array(cache.GetLocalToWorldTransform(prim))  # (4, 4)


def _to_world(points: np.ndarray, L2W: np.ndarray) -> np.ndarray:
    """Transform an (N, 3) array of points from local space to world space."""
    n = len(points)
    pts_h = np.column_stack([points, np.ones(n)])  # (N, 4) row vectors
    world_h = pts_h @ L2W                          # USD: row * M
    return world_h[:, :3]


# ---------------------------------------------------------------------------
# Mesh pruning
# ---------------------------------------------------------------------------

def _prune_mesh(stage: Usd.Stage, prim_path: str, center: np.ndarray, radius: float) -> None:
    """Prune the UsdGeomMesh at *prim_path* in-place on *stage*."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim not found: {prim_path}")

    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise ValueError(f"Prim at {prim_path} is not a UsdGeomMesh")

    points = np.array(mesh.GetPointsAttr().Get())          # (N, 3)
    face_counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())

    if not np.all(face_counts == 3):
        raise ValueError("Only triangle meshes (faceVertexCounts == 3) are supported.")

    normals_attr = mesh.GetNormalsAttr()
    normals = np.array(normals_attr.Get()) if normals_attr.HasValue() else None
    normals_interp = mesh.GetNormalsInterpolation()

    # Radius test in world space
    L2W = _local_to_world(stage, prim_path)
    points_world = _to_world(points, L2W)
    inside = np.linalg.norm(points_world - center, axis=1) <= radius

    tri_v = face_indices.reshape(-1, 3)
    keep_face = inside[tri_v[:, 0]] & inside[tri_v[:, 1]] & inside[tri_v[:, 2]]

    n_original_faces = len(keep_face)
    n_kept_faces = int(keep_face.sum())
    n_original_verts = len(points)

    kept_tri_v = tri_v[keep_face]
    used_verts = np.unique(kept_tri_v)
    old_to_new = np.full(n_original_verts, -1, dtype=np.int32)
    old_to_new[used_verts] = np.arange(len(used_verts), dtype=np.int32)

    new_points = points[used_verts]
    new_indices = old_to_new[kept_tri_v].reshape(-1)
    new_face_counts = np.full(n_kept_faces, 3, dtype=np.int32)

    new_normals = None
    if normals is not None:
        if normals_interp == UsdGeom.Tokens.faceVarying:
            face_norm_idx = np.repeat(np.where(keep_face)[0], 3) * 3
            face_norm_idx += np.tile([0, 1, 2], n_kept_faces)
            new_normals = normals[face_norm_idx]
        elif normals_interp == UsdGeom.Tokens.vertex:
            new_normals = normals[used_verts]
        else:
            print(f"  Warning: unsupported normals interpolation '{normals_interp}', dropping normals.")

    print(
        f"  Mesh {prim_path}: {n_original_faces} → {n_kept_faces} faces, "
        f"{n_original_verts} → {len(new_points)} vertices"
    )

    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new_points.astype(np.float32)))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(new_face_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(new_indices))

    if new_normals is not None:
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(new_normals.astype(np.float32)))

    mesh.GetExtentAttr().Set(
        Vt.Vec3fArray.FromNumpy(
            np.stack([new_points.min(axis=0), new_points.max(axis=0)]).astype(np.float32)
        )
    )


# ---------------------------------------------------------------------------
# Point cloud (3D Gaussian Splat) pruning
# ---------------------------------------------------------------------------

def _prune_particle_field(stage: Usd.Stage, prim_path: str, center: np.ndarray, radius: float) -> None:
    """Prune a ParticleField3DGaussianSplat prim in-place on *stage*."""
    prim = stage.GetPrimAtPath(prim_path)

    positions = np.array(prim.GetAttribute("positions").Get())  # (N, 3)
    n_original = len(positions)

    # Radius test in world space
    L2W = _local_to_world(stage, prim_path)
    positions_world = _to_world(positions, L2W)
    mask = np.linalg.norm(positions_world - center, axis=1) <= radius
    n_kept = int(mask.sum())

    print(f"  Point cloud {prim_path}: {n_original} → {n_kept} particles")

    _PER_PARTICLE = {
        "opacities":    (Vt.FloatArray,  np.float32),
        "orientations": (Vt.QuatfArray,  np.float32),
        "scales":       (Vt.Vec3fArray,  np.float32),
    }
    for attr_name, (vt_type, dtype) in _PER_PARTICLE.items():
        attr = prim.GetAttribute(attr_name)
        if attr.HasValue():
            data = np.array(attr.Get())
            attr.Set(vt_type.FromNumpy(data[mask].astype(dtype)))

    # positions
    new_positions = positions[mask]
    prim.GetAttribute("positions").Set(Vt.Vec3fArray.FromNumpy(new_positions.astype(np.float32)))

    # SH coefficients: float3[] of length N * sh_count — reshape to filter
    sh_attr = prim.GetAttribute("radiance:sphericalHarmonicsCoefficients")
    if sh_attr.HasValue():
        sh_degree = prim.GetAttribute("radiance:sphericalHarmonicsDegree").Get() or 3
        sh_count = (sh_degree + 1) ** 2
        sh = np.array(sh_attr.Get())              # (N * sh_count, 3)
        sh = sh.reshape(n_original, sh_count, 3)  # (N, sh_count, 3)
        sh = sh[mask].reshape(-1, 3)              # (N', sh_count, 3) → (N'*sh_count, 3)
        sh_attr.Set(Vt.Vec3fArray.FromNumpy(sh.astype(np.float32)))

    # Update extent
    extent_attr = prim.GetAttribute("extent")
    if extent_attr.HasValue() and len(new_positions) > 0:
        extent_attr.Set(
            Vt.Vec3fArray.FromNumpy(
                np.stack([new_positions.min(axis=0), new_positions.max(axis=0)]).astype(np.float32)
            )
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prune(
    usd_path: str,
    mesh_prim_path: str,
    radius: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    output_path: str | None = None,
) -> str:
    """Prune the mesh and any 3D Gaussian Splat point clouds by radius.

    All positions are compared against *center* and *radius* in world space.

    Args:
        usd_path: Path to the input USD file.
        mesh_prim_path: Prim path of the UsdGeomMesh to prune.
        radius: Keep radius in world-space units.
        center: XYZ world-space center of the keep sphere.
        output_path: Where to write the pruned USD. Defaults to
            ``<stem>_pruned<suffix>`` next to the input file.

    Returns:
        The path of the written output file.
    """
    stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
    center_np = np.array(center, dtype=np.float64)

    print(f"Pruning (world-space center={center}, radius={radius}):")

    # Prune the mesh
    _prune_mesh(stage, mesh_prim_path, center_np, radius)

    # Prune any 3D Gaussian Splat particle fields
    for prim in stage.Traverse():
        if prim.GetTypeName() == "ParticleField3DGaussianSplat":
            _prune_particle_field(stage, str(prim.GetPath()), center_np, radius)

    if output_path is None:
        p = Path(usd_path)
        output_path = str(p.parent / (p.stem + "_pruned.usd"))

    # Export only the root layer so payload references stay as arcs (not inlined)
    stage.GetRootLayer().Export(output_path)
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prune a USD mesh (and point clouds) to a sphere of given radius.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("usd_file", help="Input USD file path")
    parser.add_argument("prim_path", help="Prim path of the UsdGeomMesh")
    parser.add_argument("radius", type=float, help="Keep radius in world-space units")
    parser.add_argument(
        "--center", nargs=3, type=float, default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"), help="World-space center of the keep sphere (default: 0 0 0)"
    )
    parser.add_argument("--output", default=None, help="Output USD file path")
    args = parser.parse_args()

    prune(
        usd_path=args.usd_file,
        mesh_prim_path=args.prim_path,
        radius=args.radius,
        center=tuple(args.center),
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
