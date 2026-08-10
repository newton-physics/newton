# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render Phase-1 calibration-backend comparison figures.

Fits the shared shoe material on the train cycles with three conditions and
scores each on the held-out cycles, then saves four figures under
``DigitalInstron/figures/backends/``:

* ``scipy`` -- the shipped derivative-free least-squares fit (authoritative).
* ``diff (cold)`` -- the differentiable Adam fit started from the manifest seed.
* ``diff (warm)`` -- the differentiable Adam fit started from the scipy optimum.

The comparison shows that exact gradients do not beat the derivative-free fit:
warm-started diff reproduces scipy, while a cold diff descent settles in a worse
basin (a higher train loss and a much larger held-out hysteresis error), so the
residual floor is a data/model identifiability ceiling, not an optimizer limit.

Run (uses the GPU differentiable backend; a few minutes)::

    uv run -m projects.digital_instron_v2.plot_backends --manifest DigitalInstron/manifest_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import phase1
from .core import Material, fit_material, predict
from .geometry import build_column_grid, load_mesh
from .inverse_id import fit_material_to_trials
from .workflow import prepare_trials

_COLORS = {"measured": "#111111", "scipy": "#d1495b", "diff (cold)": "#e9a000", "diff (warm)": "#2b8cbe"}
_ORDER = ("scipy", "diff (cold)", "diff (warm)")
_FIXTURES = (("rearfoot_140ms", "Rearfoot punch (140 ms)"), ("fullfoot_185ms", "Fullfoot last (185 ms)"))
_METRICS = (("peak_force_error", "peak err"), ("force_rmse_relative", "RMSE"), ("hysteresis_error", "hyst err"))


def _baseline(x: np.ndarray) -> np.ndarray:
    """Inactive-baseline correction: subtract the minimum, matching the metric convention."""
    x = np.asarray(x, np.float64)
    return x - float(np.min(x))


def _collect(manifest_path: Path, *, evaluations: int, iterations: int, learning_rate: float) -> dict[str, Any]:
    """Fit the three backend conditions on train cycles and score them on held-out cycles."""
    config = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    midsole = load_mesh(base / config["midsole_mesh"], 0.001)
    grid = build_column_grid(midsole, config["grid"]["coarse_spacing_m"])
    split = phase1.generate_split_traces(manifest_path)
    train, _, _ = prepare_trials(base, config, grid, midsole, trace_paths=split["train"])
    validate, _, _ = prepare_trials(base, config, grid, midsole, trace_paths=split["validate"])
    seed = Material(*config["fit"].values())

    history: list[dict[str, float]] = []
    mat_scipy = fit_material(train, seed, evaluations, history)
    res_cold = fit_material_to_trials(train, seed, iterations=iterations, learning_rate=learning_rate)
    res_warm = fit_material_to_trials(train, mat_scipy, iterations=iterations, learning_rate=learning_rate)

    materials = {"scipy": mat_scipy, "diff (cold)": res_cold.material, "diff (warm)": res_warm.material}
    curves: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for name, _label in _FIXTURES:
        trial = next(t for t in validate if t.name == name)
        curves[name] = {
            "disp_mm": np.asarray(trial.displacement_m, np.float64) * 1000.0,
            "measured": _baseline(trial.force_n),
            "sim": {tag: _baseline(predict(trial, m)) for tag, m in materials.items()},
        }
        metrics[name] = {tag: phase1._trace_metrics(trial, m) for tag, m in materials.items()}
    return {
        "materials": materials,
        "curves": curves,
        "metrics": metrics,
        "loss": {
            "diff (cold)": np.asarray(res_cold.loss_history, np.float64),
            "diff (warm)": np.asarray(res_warm.loss_history, np.float64),
        },
        "scipy_train_loss": float(res_warm.loss_history[0]),
    }


def render_figures(
    manifest_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    evaluations: int = 100,
    iterations: int = 400,
    learning_rate: float = 0.03,
) -> list[Path]:
    """Render and save the four calibration-backend figures; return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(manifest_path).resolve()
    out = Path(out_dir) if out_dir is not None else path.parent / "figures" / "backends"
    out.mkdir(parents=True, exist_ok=True)
    d = _collect(path, evaluations=evaluations, iterations=iterations, learning_rate=learning_rate)
    saved: list[Path] = []

    # 1. Held-out force-displacement loops per backend.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, label) in zip(axes, _FIXTURES, strict=True):
        c = d["curves"][name]
        ax.plot(c["disp_mm"], c["measured"], color=_COLORS["measured"], lw=2.4, label="measured (held-out)")
        for tag in _ORDER:
            hy = d["metrics"][name][tag]["hysteresis_error"] * 100.0
            ls = "--" if tag == "diff (warm)" else "-"
            ax.plot(c["disp_mm"], c["sim"][tag], color=_COLORS[tag], lw=1.7, ls=ls, label=f"{tag} (hyst {hy:.0f}%)")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("indenter displacement [mm]")
        ax.set_ylabel("force [N]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Calibration backends on held-out cycles: force-displacement loops", fontsize=12, y=1.02)
    fig.tight_layout()
    saved.append(_save(plt, fig, out / "fig1_backend_heldout_loops.png"))

    # 2. Held-out validation metrics.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    xb = np.arange(len(_METRICS))
    width = 0.26
    for ax, (name, label) in zip(axes, _FIXTURES, strict=True):
        for k, tag in enumerate(_ORDER):
            vals = [d["metrics"][name][tag][key] * 100.0 for key, _ in _METRICS]
            ax.bar(xb + (k - 1) * width, vals, width, color=_COLORS[tag], label=tag)
        ax.axhline(10.0, color="#111111", ls=":", lw=1.2, label="10% gate")
        ax.set_xticks(xb)
        ax.set_xticklabels([lab for _, lab in _METRICS])
        ax.set_ylabel("held-out error [%]")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("Calibration backends: held-out validation metrics", fontsize=12, y=1.02)
    fig.tight_layout()
    saved.append(_save(plt, fig, out / "fig2_backend_metrics.png"))

    # 3. Fitted parameters.
    params = [
        ("instantaneous_shear_modulus_pa", "G_inst [kPa]", 1.0e-3),
        ("hyperfoam_exponent", "alpha [-]", 1.0),
        ("equilibrium_fraction", "eq_fraction [-]", 1.0),
        ("pasternak_n_per_m", "pasternak [N/m]", 1.0),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (key, label, scale) in zip(axes, params, strict=True):
        vals = [getattr(d["materials"][tag], key) * scale for tag in _ORDER]
        ax.bar(_ORDER, vals, color=[_COLORS[t] for t in _ORDER])
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelrotation=20)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Calibration backends: fitted shared material parameters", fontsize=12, y=1.03)
    fig.tight_layout()
    saved.append(_save(plt, fig, out / "fig3_backend_parameters.png"))

    # 4. Differentiable Adam convergence.
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag in ("diff (cold)", "diff (warm)"):
        ax.semilogy(d["loss"][tag], color=_COLORS[tag], lw=1.8, label=f"{tag} Adam loss")
    ax.axhline(d["scipy_train_loss"], color=_COLORS["scipy"], ls="--", lw=1.6, label="scipy optimum (train loss)")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("peak-normalized train force loss")
    ax.set_title(
        "Differentiable fit convergence: warm start sits at the scipy optimum,\ncold start settles in a worse basin",
        fontsize=11,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    saved.append(_save(plt, fig, out / "fig4_backend_convergence.png"))
    return saved


def _save(plt, fig, path: Path) -> Path:
    """Save a figure to ``path`` and close it."""
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Render the calibration-backend comparison figures from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--evaluations", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args()
    paths = render_figures(
        args.manifest,
        args.out_dir,
        evaluations=args.evaluations,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )
    print("saved:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
