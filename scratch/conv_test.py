# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure grid-resolution dependence of the baked foundation force.

Forward-evaluates the baked foundation replay at several grid spacings using a
FIXED material, and reports the peak predicted force per trial. A resolution-
independent contact model should show negligible spread across spacings.

Usage:
    uv run --extra examples python scratch/conv_test.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from projects.digital_instron_v2.geometry import build_baked_midsole_geometry
from projects.digital_instron_v2.foundation import evaluate_foundation_baked_batch
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
DEVICE = "cuda:0"
SPACINGS_M = [0.004, 0.003, 0.002, 0.0015, 0.001]


def main() -> None:
    manifest = load_manifest(MANIFEST)
    output_dir = Path("scratch")
    output_dir.mkdir(exist_ok=True)
    vertices, faces = _load_midsole_mesh(manifest, output_dir)
    material = _initial_material(manifest)
    thickness_axis = manifest.grid.get("force_thickness_axis")

    # (coverage_mode, trial_name) -> list of (spacing_m, ncells, peak_force_N)
    records: dict[tuple[bool, str], list[tuple[float, int, float]]] = {}

    for use_coverage in (False, True):
        for spacing in SPACINGS_M:
            baked_geometry = build_baked_midsole_geometry(
                vertices, faces, spacing_m=spacing, thickness_axis=thickness_axis
            )
            baked_uv_m, _xy, baked_cell_area_m2 = _baked_quadrature(baked_geometry)
            batches = _autodiff_batches(manifest, None, vertices, use_baked=True)
            batches = _with_baked_cell_area(batches, baked_cell_area_m2, baked_geometry.spacing_m)
            ncells = len(baked_uv_m)

            for batch in batches:
                trial = next(t for t in manifest.trials if t.name == batch.name)
                ind_map, ind_valid_map = bake_indenter_maps(
                    baked_geometry, trial, manifest, vertices, spacing
                )
                res = evaluate_foundation_baked_batch(
                    baked_uv_m,
                    baked_geometry,
                    ind_map,
                    ind_valid_map,
                    batch,
                    material=material,
                    use_equilibrium=True,
                    use_subcell_coverage=use_coverage,
                    device=DEVICE,
                )
                peak = float(np.max(np.asarray(res.predicted_force_n)))
                records.setdefault((use_coverage, batch.name), []).append((spacing, ncells, peak))
                tag = "coverage" if use_coverage else "baseline"
                print(f"  [{tag:8s}] {batch.name:24s} spacing={spacing*1e3:5.2f}mm cells={ncells:6d} peak={peak:10.2f} N")

    print("\n=== Resolution convergence summary (peak-force spread across spacings) ===")
    for use_coverage in (False, True):
        tag = "coverage" if use_coverage else "baseline"
        for (cov, name), rows in records.items():
            if cov != use_coverage:
                continue
            peaks = np.array([r[2] for r in rows])
            spread = (peaks.max() - peaks.min()) / max(peaks.mean(), 1e-9)
            print(f"  [{tag:8s}] {name:24s} spread = {spread*100:5.2f}%  (mean {peaks.mean():.1f} N)")


if __name__ == "__main__":
    main()
