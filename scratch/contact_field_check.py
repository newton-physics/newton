# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the general contact-field evaluator and render the pressure field.

Rebuilds the production baked geometry / indenter maps / batches, then calls the
manifest-free ``evaluate_contact_field`` evaluator. It checks that the summed
per-cell vertical force matches ``evaluate_foundation_baked_batch`` (the fitting
path), reports the center of pressure (CoP) and wrench at the peak frame, and
writes a hydroelastic-style pressure-field heatmap PNG per trial.

Usage:
    uv run --extra examples python3 scratch/contact_field_check.py
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
    evaluate_foundation_baked_batch,
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
        print(f"[contact_field_check] material artifact {p} not found; using initial material")
        return _initial_material(manifest)
    data = json.loads(p.read_text())
    mat = data.get("material", data)
    kwargs = set(inspect.signature(FoundationMaterial.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in mat.items() if k in kwargs}
    return FoundationMaterial(**filtered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = load_manifest(MANIFEST)
    output_dir = Path("scratch")
    output_dir.mkdir(exist_ok=True)
    vertices, faces = _load_midsole_mesh(manifest, output_dir)
    material = _load_material(args.material, manifest)

    thickness_axis = manifest.grid.get("force_thickness_axis")
    spacing = float(manifest.grid.get("baked_spacing_m", 0.002))
    geom = build_baked_midsole_geometry(vertices, faces, spacing_m=spacing, thickness_axis=thickness_axis)
    baked_uv_m, _xy, cell_area = _baked_quadrature(geom)
    batches = _autodiff_batches(manifest, None, vertices, use_baked=True)
    batches = _with_baked_cell_area(batches, cell_area, geom.spacing_m)

    for batch in batches:
        trial = next(t for t in manifest.trials if t.name == batch.name)
        ind_map, ind_valid = bake_indenter_maps(geom, trial, manifest, vertices, spacing)

        field = evaluate_contact_field(
            baked_uv_m,
            geom,
            ind_map,
            ind_valid,
            batch,
            material=material,
            use_equilibrium=True,
            use_subcell_coverage=True,
            device=DEVICE,
        )
        ref = evaluate_foundation_baked_batch(
            baked_uv_m,
            geom,
            ind_map,
            ind_valid,
            batch,
            material=material,
            use_equilibrium=True,
            use_subcell_coverage=True,
            device=DEVICE,
        )
        ref_force = np.asarray(ref.predicted_force_n, dtype=np.float64)
        max_abs_err = float(np.max(np.abs(field.net_force_n - ref_force)))
        rel = max_abs_err / max(float(np.max(np.abs(ref_force))), 1e-9)

        ipk = int(np.argmax(field.net_force_n))
        cop = field.cop_xy_m[ipk]
        wrench = field.wrench[ipk]
        peak_pressure_kpa = float(np.max(field.cell_pressure_pa[ipk])) / 1e3

        print(f"\n=== {batch.name} ===")
        print(f"  cells                {field.cell_force_n.shape[1]}")
        print(f"  net vs fitting-path  max_abs_err={max_abs_err:.3e} N  rel={rel:.2e}")
        print(f"  peak frame           {ipk}  Fz={field.net_force_n[ipk]:.2f} N")
        print(f"  CoP at peak [mm]     ({cop[0] * 1e3:.2f}, {cop[1] * 1e3:.2f})")
        print(f"  wrench [Fz,Mx,My]    ({wrench[0]:.2f} N, {wrench[1]:.3f} N·m, {wrench[2]:.3f} N·m)")
        print(f"  peak cell pressure   {peak_pressure_kpa:.1f} kPa")

        # Pressure-field heatmap at the peak frame (hydroelastic-style).
        xy = field.cell_xy_m
        press_kpa = field.cell_pressure_pa[ipk] / 1e3
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            xy[:, 0] * 1e3,
            xy[:, 1] * 1e3,
            c=press_kpa,
            s=14,
            cmap="inferno",
            marker="s",
        )
        ax.scatter([cop[0] * 1e3], [cop[1] * 1e3], c="cyan", s=120, marker="+", linewidths=2.5, label="CoP")
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"{batch.name}: contact pressure @ peak (Fz={field.net_force_n[ipk]:.0f} N)")
        ax.legend(loc="upper right")
        fig.colorbar(sc, ax=ax, label="pressure [kPa]")
        fig.tight_layout()
        out = output_dir / f"contact_field_{batch.name}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
