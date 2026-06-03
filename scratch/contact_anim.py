# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Animate the FITTED contact pressure field over a full Instron cycle.

GL rendering is unreliable on headless/SSH boxes (CUDA/GL interop unavailable),
so this renders a robust matplotlib animation instead. It evaluates the learned
material's per-cell pressure via ``evaluate_contact_field`` and writes an animated
GIF showing:

  * LEFT  - contact pressure heatmap (per-cell) with the center of pressure marked,
  * RIGHT - force-vs-displacement loop with a moving marker at the current frame.

Both panels animate together so you can see the pressure field build and release
as the indenter presses in and lifts off.

Usage:
    uv run --extra examples python3 scratch/contact_anim.py
    uv run --extra examples python3 scratch/contact_anim.py --trial fullfoot_185ms
    uv run --extra examples python3 scratch/contact_anim.py --stride 4 --fps 25
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from projects.digital_instron_v2.foundation import (
    FoundationMaterial,
    evaluate_contact_field,
)
from projects.digital_instron_v2.geometry import build_baked_midsole_geometry
from projects.digital_instron_v2.manifest import load_manifest
from projects.digital_instron_v2.workflow import (
    _autodiff_batches,
    _baked_quadrature,
    _initial_material,
    _load_midsole_mesh,
    _with_baked_cell_area,
    bake_indenter_maps,
)

MANIFEST = "DigitalInstron/manifest_v2.json"
DEFAULT_MATERIAL = "DigitalInstron/processed/v2_cache/digital_instron_v2_foundation_material.json"
DEVICE = "cuda:0"


def _load_material(path: str, manifest) -> FoundationMaterial:
    p = Path(path)
    if not p.exists():
        print(f"[contact_anim] material artifact {p} not found; using initial material")
        return _initial_material(manifest)
    data = json.loads(p.read_text())
    mat = data.get("material", data)
    kwargs = set(inspect.signature(FoundationMaterial.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in mat.items() if k in kwargs}
    return FoundationMaterial(**filtered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    parser.add_argument("--trial", default=None, help="Trial name; default = every trial in the manifest")
    parser.add_argument("--stride", type=int, default=3, help="Use every Nth frame to keep the GIF small")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

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
        selected = [b for b in batches if b.name == args.trial]
    else:
        selected = list(batches)
    if not selected:
        raise SystemExit(f"no matching trial(s) for --trial {args.trial!r}")

    saved = []
    for batch in selected:
        out = _render_trial(
            batch,
            manifest,
            geom,
            sample_uv_m,
            vertices,
            spacing,
            material,
            output_dir,
            stride=max(int(args.stride), 1),
            fps=int(args.fps),
        )
        saved.append(str(out))
    print(f"[contact_anim] rendered {len(saved)} animation(s):")
    for s in saved:
        print(f"  {s}")


def _render_trial(
    batch,
    manifest,
    geom,
    sample_uv_m,
    vertices,
    spacing,
    material,
    output_dir,
    *,
    stride: int,
    fps: int,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    trial = next(t for t in manifest.trials if t.name == batch.name)
    ind_map, ind_valid = bake_indenter_maps(geom, trial, manifest, vertices, spacing)

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
    meas = np.asarray(batch.measured_force_n, dtype=np.float64)
    xy = field.cell_xy_m * 1e3  # mm
    press_kpa = field.cell_pressure_pa / 1e3  # (frames, cells)
    net = field.net_force_n
    cop = field.cop_xy_m * 1e3  # mm
    p_vmax = max(float(np.max(press_kpa)), 1.0)

    frames = list(range(0, len(disp), stride))
    ipk = int(np.argmax(net))
    if ipk not in frames:
        frames.append(ipk)
        frames.sort()
    print(f"[contact_anim] trial={batch.name} total_frames={len(disp)} anim_frames={len(frames)} "
          f"peak_frame={ipk} peak_Fz={net[ipk]:.1f} N peak_pressure={p_vmax:.1f} kPa")

    fig, (ax_p, ax_f) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Pressure heatmap panel (square cells).
    sc = ax_p.scatter(xy[:, 0], xy[:, 1], c=press_kpa[0], s=16, cmap="inferno", marker="s", vmin=0.0, vmax=p_vmax)
    cop_dot = ax_p.scatter([], [], c="cyan", s=160, marker="+", linewidths=2.5, label="CoP", zorder=5)
    ax_p.set_aspect("equal")
    ax_p.set_xlabel("x [mm]")
    ax_p.set_ylabel("y [mm]")
    ax_p.legend(loc="upper right")
    fig.colorbar(sc, ax=ax_p, label="contact pressure [kPa]")

    # Force-displacement panel.
    ax_f.plot(disp * 1e3, meas, "k-", lw=1.5, label="measured")
    ax_f.plot(disp * 1e3, net, "r-", lw=1.5, alpha=0.8, label="predicted (fit)")
    moving_dot, = ax_f.plot([], [], "ro", ms=9, zorder=5)
    ax_f.set_xlabel("displacement [mm]")
    ax_f.set_ylabel("force [N]")
    ax_f.legend(loc="upper left")
    ax_f.grid(alpha=0.3)

    def update(fi: int):
        sc.set_array(press_kpa[fi])
        if net[fi] > 1e-9 and np.isfinite(cop[fi, 0]):
            cop_dot.set_offsets(cop[fi : fi + 1, :])
        else:
            cop_dot.set_offsets(np.empty((0, 2)))
        moving_dot.set_data([disp[fi] * 1e3], [net[fi]])
        ax_p.set_title(f"{batch.name}: pressure  (Fz={net[fi]:6.0f} N, d={disp[fi] * 1e3:5.2f} mm)")
        ax_f.set_title(f"frame {fi}/{len(disp) - 1}")
        return sc, cop_dot, moving_dot

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = output_dir / f"contact_anim_{batch.name}.gif"
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[contact_anim] saved {out}")
    return out


if __name__ == "__main__":
    main()
