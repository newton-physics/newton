# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnose how well the fitted material matches the FULL hysteresis loop.

Loads the fitted material artifact, rebuilds the baked geometry / indenter maps
/ batches at the production spacing, forward-evaluates the predicted force trace,
and reports loading-branch RMSE, unloading-branch RMSE, peak error and loop-area
error separately (not just peak). Writes an F-vs-displacement loop + F-vs-time
overlay PNG per trial to scratch/.

Usage:
    uv run --extra examples python3 scratch/curve_check.py
    uv run --extra examples python3 scratch/curve_check.py --material path/to/material.json
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from projects.digital_instron_v2.foundation import (
    FoundationMaterial,
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
        print(f"[curve_check] material artifact {p} not found; using initial material")
        return _initial_material(manifest)
    data = json.loads(p.read_text())
    mat = data.get("material", data)
    kwargs = set(inspect.signature(FoundationMaterial.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in mat.items() if k in kwargs}
    return FoundationMaterial(**filtered)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default=DEFAULT_MATERIAL)
    parser.add_argument("--no-equilibrium", dest="use_equilibrium", action="store_false")
    parser.add_argument("--no-subcell-coverage", dest="use_coverage", action="store_false")
    parser.set_defaults(use_equilibrium=True, use_coverage=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = load_manifest(MANIFEST)
    output_dir = Path("scratch")
    output_dir.mkdir(exist_ok=True)
    vertices, faces = _load_midsole_mesh(manifest, output_dir)
    material = _load_material(args.material, manifest)
    print(f"[curve_check] material: {material.__dict__}")

    thickness_axis = manifest.grid.get("force_thickness_axis")
    spacing = float(manifest.grid.get("baked_spacing_m", 0.002))
    geom = build_baked_midsole_geometry(vertices, faces, spacing_m=spacing, thickness_axis=thickness_axis)
    baked_uv_m, _xy, cell_area = _baked_quadrature(geom)
    batches = _autodiff_batches(manifest, None, vertices, use_baked=True)
    batches = _with_baked_cell_area(batches, cell_area, geom.spacing_m)

    summary = {}
    for batch in batches:
        trial = next(t for t in manifest.trials if t.name == batch.name)
        ind_map, ind_valid = bake_indenter_maps(geom, trial, manifest, vertices, spacing)
        res = evaluate_foundation_baked_batch(
            baked_uv_m,
            geom,
            ind_map,
            ind_valid,
            batch,
            material=material,
            use_equilibrium=args.use_equilibrium,
            use_subcell_coverage=args.use_coverage,
            device=DEVICE,
        )
        pred = np.asarray(res.predicted_force_n, dtype=np.float64)
        meas = np.asarray(batch.measured_force_n, dtype=np.float64)
        disp = np.asarray(batch.displacement_m, dtype=np.float64)
        t = np.asarray(batch.time_s, dtype=np.float64)

        # Split loading vs unloading at peak displacement.
        ipk = int(np.argmax(disp))
        load = slice(0, ipk + 1)
        unload = slice(ipk, len(disp))

        peak_meas = float(np.max(meas))
        peak_pred = float(np.max(pred))
        loop_meas = float(np.trapezoid(meas, disp))
        loop_pred = float(np.trapezoid(pred, disp))

        rec = {
            "peak_meas_N": peak_meas,
            "peak_pred_N": peak_pred,
            "peak_err_pct": 100.0 * abs(peak_pred - peak_meas) / max(peak_meas, 1e-9),
            "loading_rmse_N": _rmse(pred[load], meas[load]),
            "unloading_rmse_N": _rmse(pred[unload], meas[unload]),
            "trace_rmse_N": _rmse(pred, meas),
            "loading_rmse_pct": 100.0 * _rmse(pred[load], meas[load]) / max(peak_meas, 1e-9),
            "unloading_rmse_pct": 100.0 * _rmse(pred[unload], meas[unload]) / max(peak_meas, 1e-9),
            "loop_area_meas_J": loop_meas,
            "loop_area_pred_J": loop_pred,
            "loop_area_err_pct": 100.0 * abs(loop_pred - loop_meas) / max(abs(loop_meas), 1e-12),
        }
        summary[batch.name] = rec
        print(f"\n=== {batch.name} ===")
        for k, v in rec.items():
            print(f"  {k:20s} {v:12.4f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(disp * 1e3, meas, "k-", lw=2, label="measured")
        ax1.plot(disp * 1e3, pred, "r--", lw=1.5, label="predicted")
        ax1.set_xlabel("displacement [mm]")
        ax1.set_ylabel("force [N]")
        ax1.set_title(f"{batch.name}: F-x loop")
        ax1.legend()
        ax2.plot(t, meas, "k-", lw=2, label="measured")
        ax2.plot(t, pred, "r--", lw=1.5, label="predicted")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("force [N]")
        ax2.set_title(f"{batch.name}: F-t trace")
        ax2.legend()
        fig.tight_layout()
        out = output_dir / f"curve_check_{batch.name}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  saved {out}")

    (output_dir / "curve_check_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
