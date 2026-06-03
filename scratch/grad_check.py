# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify baked-foundation autodiff gradients against central finite differences.

Builds the baked geometry + indenter maps once at the production spacing, then
compares the analytic gradient returned by ``foundation_baked_batch_loss_gradient``
against a central finite-difference of the loss for each free material parameter.

Usage:
    uv run --extra examples python scratch/grad_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from projects.digital_instron_v2.foundation import (
    _array_to_material,
    _material_to_array,
    foundation_baked_batch_loss_gradient,
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
DEVICE = "cuda:0"

PARAM_NAMES = [
    "stiffness_pa",
    "ogden_alpha",
    "lock_strain",
    "damping_pa_s",
    "damping_power",
    "prony_stiffness",
    "prony_damping",
    "pasternak(off)",
    "spatial_slope",
]
# Parameters that are actively optimised (pasternak[7] is frozen at 0).
FREE_PARAMS = [0, 1, 2, 3, 4, 5, 6, 8]


def _loss(xy_m, geom, ind_map, ind_valid, batch, x, material0, use_equilibrium, use_coverage):
    material = _array_to_material(x, material0)
    res = foundation_baked_batch_loss_gradient(
        xy_m,
        geom,
        ind_map,
        ind_valid,
        batch,
        material=material,
        loop_weight=0.0,
        use_equilibrium=use_equilibrium,
        use_subcell_coverage=use_coverage,
        device=DEVICE,
    )
    return float(res.loss), np.asarray(res.gradient, dtype=np.float64)


def main() -> None:
    manifest = load_manifest(MANIFEST)
    output_dir = Path("scratch")
    output_dir.mkdir(exist_ok=True)
    vertices, faces = _load_midsole_mesh(manifest, output_dir)
    material0 = _initial_material(manifest)
    thickness_axis = manifest.grid.get("force_thickness_axis")
    spacing = float(manifest.grid.get("baked_spacing_m", 0.002))

    geom = build_baked_midsole_geometry(vertices, faces, spacing_m=spacing, thickness_axis=thickness_axis)
    baked_uv_m, _xy, cell_area = _baked_quadrature(geom)
    batches = _autodiff_batches(manifest, None, vertices, use_baked=True)
    batches = _with_baked_cell_area(batches, cell_area, geom.spacing_m)

    indenter_maps = {}
    for batch in batches:
        trial = next(t for t in manifest.trials if t.name == batch.name)
        indenter_maps[batch.name] = bake_indenter_maps(geom, trial, manifest, vertices, spacing)

    x0 = _material_to_array(material0, include_state=True).astype(np.float64)

    for use_coverage in (True,):
        for use_equilibrium in (True,):
            for batch in batches:
                ind_map, ind_valid = indenter_maps[batch.name]
                _l0, g_analytic = _loss(
                    baked_uv_m, geom, ind_map, ind_valid, batch, x0, material0, use_equilibrium, use_coverage
                )
                print(
                    f"\n=== {batch.name}  equilibrium={use_equilibrium} coverage={use_coverage}  loss={_l0:.6f} ==="
                )
                print(f"{'param':16s} {'analytic':>14s} {'finite-diff':>14s} {'rel-err':>10s}")
                for i in FREE_PARAMS:
                    # Relative central step in physical units (parameters span orders of magnitude).
                    h = max(abs(x0[i]) * 1e-4, 1e-6)
                    xp = x0.copy()
                    xm = x0.copy()
                    xp[i] += h
                    xm[i] -= h
                    lp, _ = _loss(
                        baked_uv_m, geom, ind_map, ind_valid, batch, xp, material0, use_equilibrium, use_coverage
                    )
                    lm, _ = _loss(
                        baked_uv_m, geom, ind_map, ind_valid, batch, xm, material0, use_equilibrium, use_coverage
                    )
                    fd = (lp - lm) / (2.0 * h)
                    a = g_analytic[i]
                    denom = max(abs(a), abs(fd), 1e-12)
                    rel = abs(a - fd) / denom
                    flag = "  <-- MISMATCH" if rel > 1e-2 and denom > 1e-9 else ""
                    print(f"{PARAM_NAMES[i]:16s} {a:14.6e} {fd:14.6e} {rel:10.3e}{flag}")


if __name__ == "__main__":
    main()
