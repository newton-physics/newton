# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare and validate triangle meshes for Newton hydroelastic SDF workflows.

Example:
    uv run python scripts/prepare_hydro_mesh.py foot.obj --output foot_hydro.obj
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np

import newton


def _load_trimesh_module():
    try:
        return importlib.import_module("trimesh")
    except ImportError as exc:
        raise RuntimeError(
            "This script requires trimesh. Install example dependencies (for example: `uv sync --extra examples`)."
        ) from exc


def _load_as_single_mesh(mesh_file: Path) -> Any:
    trimesh = _load_trimesh_module()
    tri = trimesh.load(str(mesh_file), force="mesh", process=False)
    if hasattr(tri, "geometry"):
        geometries = []
        for geom in tri.geometry.values():
            geom_vertices = np.asarray(getattr(geom, "vertices", []))
            geom_faces = np.asarray(getattr(geom, "faces", []))
            if geom_vertices.size == 0 or geom_faces.size == 0:
                continue
            geometries.append(geom)
        if not geometries:
            raise ValueError(f"Mesh file '{mesh_file}' did not contain any triangle geometry.")
        tri = trimesh.util.concatenate(tuple(geometries))
    return tri


def _collect_mesh_stats(tri_mesh: Any, vertices: np.ndarray) -> dict[str, Any]:
    extents = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    components = len(tri_mesh.split(only_watertight=False))
    return {
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": int(np.asarray(tri_mesh.faces).shape[0]),
        "components": int(components),
        "watertight": bool(tri_mesh.is_watertight),
        "winding_consistent": bool(tri_mesh.is_winding_consistent),
        "is_volume": bool(tri_mesh.is_volume),
        "extents": extents.astype(np.float32),
    }


def _hydro_readiness_issues(stats: dict[str, Any], *, allow_multiple_components: bool) -> list[str]:
    issues: list[str] = []
    if not stats["watertight"]:
        issues.append("mesh is not watertight")
    if not stats["winding_consistent"]:
        issues.append("mesh winding is inconsistent")
    if not stats["is_volume"]:
        issues.append("mesh does not represent a closed volume")
    if (not allow_multiple_components) and stats["components"] != 1:
        issues.append(f"mesh has {stats['components']} disconnected components (expected 1)")
    return issues


def _print_stats(label: str, stats: dict[str, Any]):
    extents = stats["extents"]
    print(
        f"{label}: "
        f"{stats['vertex_count']} vertices, {stats['triangle_count']} triangles, "
        f"components={stats['components']}, watertight={stats['watertight']}, "
        f"winding_consistent={stats['winding_consistent']}, is_volume={stats['is_volume']}, "
        f"extents=[{extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f}] m"
    )


def _apply_mesh_fixes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    solidify_thickness: float,
    remesh_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    out_vertices = np.asarray(vertices, dtype=np.float32)
    out_faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)

    if solidify_thickness > 0.0:
        solid_faces, solid_vertices = newton.utils.solidify_mesh(out_faces, out_vertices, solidify_thickness)
        out_vertices = np.asarray(solid_vertices, dtype=np.float32)
        out_faces = np.asarray(solid_faces, dtype=np.int32).reshape(-1, 3)

    if remesh_method != "none":
        mesh = newton.Mesh(out_vertices, out_faces.reshape(-1), compute_inertia=False)
        remeshed = newton.utils.remesh_mesh(
            mesh,
            method=remesh_method,
            recompute_inertia=False,
            inplace=False,
        )
        out_vertices = np.asarray(remeshed.vertices, dtype=np.float32)
        out_faces = np.asarray(remeshed.indices, dtype=np.int32).reshape(-1, 3)

    return out_vertices, out_faces


def _run_sdf_build_check(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sdf_resolution: int,
    narrow_band: float,
    sdf_margin: float,
):
    wp = importlib.import_module("warp")

    if not wp.get_device().is_cuda:
        raise RuntimeError("SDF build check requires a CUDA device.")

    mesh = newton.Mesh(vertices, faces.reshape(-1), compute_inertia=False)
    mesh.build_sdf(
        max_resolution=int(sdf_resolution),
        narrow_band_range=(-float(narrow_band), float(narrow_band)),
        margin=float(sdf_margin),
    )


def _write_obj(mesh_file: Path, vertices: np.ndarray, faces: np.ndarray):
    trimesh = _load_trimesh_module()
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tri.export(str(mesh_file), file_type="obj")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_file", type=str, help="Input mesh path (.obj/.stl/etc., trimesh-supported).")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output OBJ path. When omitted, no mesh file is written.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform vertex scale factor applied after optional repairs.",
    )
    parser.add_argument(
        "--center-origin",
        action="store_true",
        help="Recenter vertices around their AABB center after optional repairs.",
    )
    parser.add_argument(
        "--solidify-thickness",
        type=float,
        default=0.0,
        help="Extrude the surface by this thickness [m] to create a volume shell before validation.",
    )
    parser.add_argument(
        "--remesh-method",
        type=str,
        default="none",
        choices=["none", "poisson", "ftetwild", "quadratic", "convex_hull", "alphashape"],
        help="Optional remeshing method applied before final validation.",
    )
    parser.add_argument(
        "--allow-multiple-components",
        action="store_true",
        help="Do not fail validation when the mesh has more than one connected component.",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Exit successfully even if hydro-readiness checks fail.",
    )
    parser.add_argument(
        "--check-sdf-build",
        action="store_true",
        help="Build an SDF with Newton as a final runtime check (requires CUDA).",
    )
    parser.add_argument(
        "--sdf-resolution",
        type=int,
        default=128,
        help="SDF max resolution used by --check-sdf-build.",
    )
    parser.add_argument(
        "--narrow-band",
        type=float,
        default=0.01,
        help="SDF narrow band half-width [m] used by --check-sdf-build.",
    )
    parser.add_argument(
        "--sdf-margin",
        type=float,
        default=0.01,
        help="SDF AABB padding [m] used by --check-sdf-build.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    mesh_file = Path(args.mesh_file).expanduser().resolve()
    if not mesh_file.exists():
        raise FileNotFoundError(f"Input mesh file not found: {mesh_file}")

    tri_input = _load_as_single_mesh(mesh_file)
    input_vertices = np.asarray(tri_input.vertices, dtype=np.float32)
    input_faces = np.asarray(tri_input.faces, dtype=np.int32).reshape(-1, 3)
    if input_vertices.size == 0 or input_faces.size == 0:
        raise ValueError(f"Input mesh '{mesh_file}' has no vertices or faces.")

    input_stats = _collect_mesh_stats(tri_input, input_vertices)
    _print_stats("Input", input_stats)

    vertices, faces = _apply_mesh_fixes(
        input_vertices,
        input_faces,
        solidify_thickness=float(args.solidify_thickness),
        remesh_method=str(args.remesh_method),
    )

    if args.center_origin:
        center = 0.5 * (np.min(vertices, axis=0) + np.max(vertices, axis=0))
        vertices = vertices - center.astype(np.float32)
    if float(args.scale) != 1.0:
        vertices = vertices * float(args.scale)

    trimesh = _load_trimesh_module()
    tri_output = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    output_stats = _collect_mesh_stats(tri_output, vertices)
    _print_stats("Prepared", output_stats)

    issues = _hydro_readiness_issues(
        output_stats,
        allow_multiple_components=bool(args.allow_multiple_components),
    )
    if issues:
        print("Hydro readiness: FAIL")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Hydro readiness: PASS")

    if args.check_sdf_build:
        _run_sdf_build_check(
            vertices,
            faces,
            sdf_resolution=int(args.sdf_resolution),
            narrow_band=float(args.narrow_band),
            sdf_margin=float(args.sdf_margin),
        )
        print("SDF build check: PASS")

    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_obj(output_path, vertices, faces)
        print(f"Wrote OBJ: {output_path}")

    if issues and (not args.allow_invalid):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
