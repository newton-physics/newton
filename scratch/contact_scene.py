# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the digital-Instron scene colored by the FITTED contact pressure field.

Unlike the workflow ``surface-scene`` step (which colors the heatmap by raw
geometric compression depth), this drives the surface heatmap with the actual
per-cell vertical pressure produced by the learned material via
``evaluate_contact_field``. It animates the indenter pressing into the midsole,
colors the contact surface by pressure, marks the center of pressure, and writes
PNG snapshots (loading / peak / unloading) so progress is visible even headless.

Usage:
    uv run --extra examples python3 scratch/contact_scene.py
    uv run --extra examples python3 scratch/contact_scene.py --trial fullfoot_185ms --viewer gl
    uv run --extra examples python3 scratch/contact_scene.py --viewer null   # snapshots only
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton.viewer
from projects.digital_instron_v2.foundation import (
    FoundationMaterial,
    evaluate_contact_field,
)
from projects.digital_instron_v2.geometry import build_baked_midsole_geometry
from projects.digital_instron_v2.manifest import load_manifest
from projects.digital_instron_v2.workflow import (
    _autodiff_batches,
    _baked_quadrature,
    _compression_heat_texture,
    _initial_material,
    _load_midsole_mesh,
    _mesh_vertices_z_up,
    _plane_mesh,
    _surface_map_mesh,
    _with_baked_cell_area,
    bake_indenter_maps,
)

MANIFEST = "DigitalInstron/manifest_v2.json"
DEFAULT_MATERIAL = "DigitalInstron/processed/v2_cache/digital_instron_v2_foundation_material.json"
DEVICE = "cuda:0"


def _load_material(path: str, manifest) -> FoundationMaterial:
    p = Path(path)
    if not p.exists():
        print(f"[contact_scene] material artifact {p} not found; using initial material")
        return _initial_material(manifest)
    data = json.loads(p.read_text())
    mat = data.get("material", data)
    kwargs = set(inspect.signature(FoundationMaterial.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in mat.items() if k in kwargs}
    return FoundationMaterial(**filtered)


def _pressure_grid(baked_geometry, sample_uv_m: np.ndarray, cell_pressure_pa: np.ndarray) -> np.ndarray:
    """Scatter per-cell pressure back onto the HxW surface-map grid."""
    h, w = baked_geometry.top_map.shape
    u = (sample_uv_m[:, 0] - baked_geometry.mins_uv[0]) / (baked_geometry.maxs_uv[0] - baked_geometry.mins_uv[0])
    v = (sample_uv_m[:, 1] - baked_geometry.mins_uv[1]) / (baked_geometry.maxs_uv[1] - baked_geometry.mins_uv[1])
    px = np.clip(np.rint(np.clip(u, 0.0, 1.0) * (w - 1.0)).astype(np.int32), 0, w - 1)
    py = np.clip(np.rint(np.clip(v, 0.0, 1.0) * (h - 1.0)).astype(np.int32), 0, h - 1)
    grid = np.zeros((h, w), dtype=np.float64)
    grid[py, px] = np.asarray(cell_pressure_pa, dtype=np.float64)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    parser.add_argument("--trial", default=None, help="Trial name; default = first included trial")
    parser.add_argument("--viewer", choices=("gl", "null"), default="gl")
    parser.add_argument("--max-frames", type=int, default=400)
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST)
    output_dir = Path("scratch")
    output_dir.mkdir(exist_ok=True)
    vertices, faces = _load_midsole_mesh(manifest, output_dir)
    material = _load_material(args.material, manifest)

    thickness_axis = manifest.grid.get("force_thickness_axis")
    spacing = float(manifest.grid.get("baked_spacing_m", 0.002))
    geom = build_baked_midsole_geometry(vertices, faces, spacing_m=spacing, thickness_axis=thickness_axis)
    sample_uv_m, _xy, cell_area = _baked_quadrature(geom)
    batches = _autodiff_batches(manifest, None, vertices, use_baked=True)
    batches = _with_baked_cell_area(batches, cell_area, geom.spacing_m)

    if args.trial is not None:
        batch = next(b for b in batches if b.name == args.trial)
    else:
        included = {t.name for t in manifest.trials if t.include_in_fit}
        batch = next(b for b in batches if b.name in included)
    trial = next(t for t in manifest.trials if t.name == batch.name)
    ind_map, ind_valid = bake_indenter_maps(geom, trial, manifest, vertices, spacing)

    # Evaluate the fitted contact pressure field for the whole trial.
    field = evaluate_contact_field(
        sample_uv_m,
        geom,
        ind_map,
        ind_valid,
        batch,
        material=material,
        use_equilibrium=True,
        use_subcell_coverage=True,
        device=DEVICE,
    )
    disp = np.asarray(batch.displacement_m, dtype=np.float64)
    time_s = np.asarray(batch.time_s, dtype=np.float64)
    frame_count = min(int(args.max_frames), len(disp))
    p_vmax_pa = max(float(np.max(field.cell_pressure_pa)), 1.0)
    ipk = int(np.argmax(field.net_force_n))
    print(f"[contact_scene] trial={batch.name} frames={frame_count} peak_frame={ipk} "
          f"peak_Fz={field.net_force_n[ipk]:.1f} N peak_pressure={p_vmax_pa / 1e3:.1f} kPa")

    # Static meshes.
    midsole_points = _mesh_vertices_z_up(vertices, geom)
    midsole_indices = np.asarray(faces, dtype=np.int32).reshape(-1)
    top_points, top_indices, top_uvs = _surface_map_mesh(geom, geom.top_map + 0.0008)
    valid_midsole = (
        np.ones_like(geom.bottom_map, dtype=bool)
        if geom.valid_map is None
        else np.asarray(geom.valid_map, dtype=np.float64) > 0.5
    )
    min_bottom_z = float(np.min(geom.bottom_map[valid_midsole]))
    ground_points, ground_indices, ground_uvs = _plane_mesh(geom, min_bottom_z, padding_m=0.03)

    viewer = (
        newton.viewer.ViewerNull(num_frames=frame_count)
        if args.viewer == "null"
        else newton.viewer.ViewerGL()
    )
    if hasattr(viewer, "set_camera"):
        extents = np.asarray(geom.frame.extents_m, dtype=np.float64)
        span = max(float(np.max(extents)), 0.20)
        viewer.set_camera(wp.vec3(0.55 * span, -1.75 * span, 0.85 * span), pitch=-26.0, yaw=-18.0)
    dev = viewer.device

    midsole_points_wp = wp.array(midsole_points, dtype=wp.vec3, device=dev)
    midsole_indices_wp = wp.array(midsole_indices, dtype=wp.int32, device=dev)
    top_indices_wp = wp.array(top_indices, dtype=wp.int32, device=dev)
    top_uvs_wp = wp.array(top_uvs, dtype=wp.vec2, device=dev)
    ground_points_wp = wp.array(ground_points, dtype=wp.vec3, device=dev)
    ground_indices_wp = wp.array(ground_indices, dtype=wp.int32, device=dev)
    ground_uvs_wp = wp.array(ground_uvs, dtype=wp.vec2, device=dev)

    # Snapshot targets: a loading frame, the peak, an unloading frame.
    snap_frames = {
        max(ipk // 2, 0): "loading",
        ipk: "peak",
        min(ipk + (frame_count - ipk) // 2, frame_count - 1): "unloading",
    }
    captured: dict[str, str] = {}

    def render_frame(fi: int) -> None:
        d = float(disp[fi])
        sim_time = float(time_s[fi]) if fi < len(time_s) else fi / 60.0
        press_grid = _pressure_grid(geom, sample_uv_m, field.cell_pressure_pa[fi])
        heat = np.flipud(_compression_heat_texture(press_grid, ind_valid, p_vmax_pa))
        # Foot/indenter surface descending by current displacement.
        foot_points, foot_indices, foot_uvs = _surface_map_mesh(geom, ind_map - max(d, 0.0), ind_valid)
        active = bool(np.any(field.cell_force_n[fi] > 0.0))

        viewer.begin_frame(sim_time)
        viewer.log_mesh(
            "/digital_instron/midsole",
            midsole_points_wp,
            midsole_indices_wp,
            color=(0.56, 0.58, 0.62),
            roughness=0.72,
            backface_culling=False,
        )
        viewer.log_mesh(
            "/digital_instron/pressure_heatmap",
            wp.array(top_points, dtype=wp.vec3, device=dev),
            top_indices_wp,
            uvs=top_uvs_wp,
            texture=heat,
            roughness=0.5,
            hidden=not active,
            backface_culling=False,
        )
        if len(foot_indices) > 0:
            viewer.log_mesh(
                "/digital_instron/foot",
                wp.array(foot_points, dtype=wp.vec3, device=dev),
                wp.array(foot_indices, dtype=wp.int32, device=dev),
                uvs=wp.array(foot_uvs, dtype=wp.vec2, device=dev),
                color=(0.30, 0.32, 0.38),
                roughness=0.4,
                hidden=not active,
                backface_culling=False,
            )
        viewer.log_mesh(
            "/digital_instron/ground",
            ground_points_wp,
            ground_indices_wp,
            uvs=ground_uvs_wp,
            color=(0.22, 0.22, 0.24),
            roughness=0.85,
            backface_culling=False,
        )
        viewer.end_frame()

    # Capture snapshots first (one render pass each) so headless runs still produce output.
    if hasattr(viewer, "get_frame"):
        from PIL import Image

        for fi, label in snap_frames.items():
            render_frame(fi)
            frame_img = viewer.get_frame(render_ui=False).numpy()
            out = output_dir / f"contact_scene_{batch.name}_{label}.png"
            Image.fromarray(frame_img).save(out)
            captured[label] = str(out)
            print(f"[contact_scene] saved {out}")

    # Live animation loop (interactive window). Ctrl-C or close to exit.
    frame_index = 0
    while viewer.is_running():
        render_frame(frame_index)
        frame_index = (frame_index + 1) % frame_count

    viewer.close()
    if captured:
        print(json.dumps({"trial": batch.name, "snapshots": captured}, indent=2))


if __name__ == "__main__":
    main()
