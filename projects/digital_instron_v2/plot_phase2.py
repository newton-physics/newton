# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render Phase-2 dynamic-replay validation figures.

Runs the held-out dynamic replay for both fixtures in both drive modes and saves
four figures under ``DigitalInstron/figures/phase2/``:

1. Force-displacement hysteresis loops (measured vs simulated, kinematic drive).
2. Force vs cycle phase (pointwise agreement).
3. Dynamics diagnostics: COP trajectory, active contact area, max compression,
   and wrench continuity over the cycle.
4. Drive comparison: kinematic crosshead vs closed-loop PD servo, with servo
   tracking.

Run::

    uv run -m projects.digital_instron_v2.plot_phase2 --manifest DigitalInstron/manifest_v2.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from . import phase1, phase2
from .phase2 import DynamicReplayConfig

_MEASURED, _KINEMATIC, _SERVO = "#111111", "#d1495b", "#2b8cbe"

_FIXTURES = (
    ("rearfoot_140ms", "rearfoot_punch", "Rearfoot punch (140 ms)"),
    ("fullfoot_185ms", "fullfoot_last", "Fullfoot last (185 ms)"),
)


def _collect(manifest_path: Path, substeps: int, warmup_cycles: int) -> dict[str, Any]:
    """Fit the train-only material and run the kinematic and servo replays per fixture."""
    material, _, split = phase1.fit_train_material(manifest_path)
    data: dict[str, Any] = {}
    for name, fixture, label in _FIXTURES:
        trace = phase2._load_validation_trace(split, name)
        kin_cfg = DynamicReplayConfig(drive="kinematic", substeps=substeps, warmup_cycles=warmup_cycles)
        srv_cfg = DynamicReplayConfig(drive="servo", substeps=substeps, warmup_cycles=warmup_cycles)
        kin = phase2.run_dynamic_replay(manifest_path, fixture, material, trace, kin_cfg)
        srv = phase2.run_dynamic_replay(manifest_path, fixture, material, trace, srv_cfg)
        sim_kin = phase2._resample_force(kin, trace["phase"])
        sim_kin = sim_kin - sim_kin.min()
        sim_srv = phase2._resample_force(srv, trace["phase"])
        sim_srv = sim_srv - sim_srv.min()
        data[name] = {
            "label": label,
            "trace": trace,
            "kin": kin,
            "srv": srv,
            "sim_kin": sim_kin,
            "sim_srv": sim_srv,
            "mk": phase2.validate_trace_metrics(trace["force_n"], sim_kin, trace["displacement_m"]),
            "ms": phase2.validate_trace_metrics(trace["force_n"], sim_srv, trace["displacement_m"]),
        }
    return data


def render_figures(
    manifest_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    substeps: int = 32,
    warmup_cycles: int = 3,
) -> list[Path]:
    """Render and save the four Phase-2 validation figures; return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(manifest_path).resolve()
    out = Path(out_dir) if out_dir is not None else path.parent / "figures" / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    data = _collect(path, substeps, warmup_cycles)
    saved: list[Path] = []

    # 1. Hysteresis loops.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, _fixture, _label) in zip(axes, _FIXTURES, strict=True):
        d = data[name]
        disp = d["trace"]["displacement_m"] * 1000.0
        ax.plot(disp, d["trace"]["force_n"], color=_MEASURED, lw=2.2, label="measured (held-out 99-100)")
        ax.plot(disp, d["sim_kin"], color=_KINEMATIC, lw=1.8, ls="--", label="simulated (MuJoCo, kinematic)")
        mk = d["mk"]
        ax.set_title(
            f"{d['label']}\npeak err {mk.peak_force_error * 100:.1f}%  |  "
            f"RMSE {mk.force_rmse_relative * 100:.1f}%  |  hyst err {mk.hysteresis_error * 100:.1f}%",
            fontsize=10,
        )
        ax.set_xlabel("indenter displacement [mm]")
        ax.set_ylabel("force [N]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Phase-2 dynamic replay through SolverMuJoCo: force-displacement hysteresis loops", fontsize=12, y=1.02
    )
    fig.tight_layout()
    saved.append(_save(fig, out / "fig1_hysteresis_loops.png"))

    # 2. Force vs cycle phase.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (name, _fixture, _label) in zip(axes, _FIXTURES, strict=True):
        d = data[name]
        ph = d["trace"]["phase"]
        ax.plot(ph, d["trace"]["force_n"], color=_MEASURED, lw=2.2, label="measured")
        ax.plot(ph, d["sim_kin"], color=_KINEMATIC, lw=1.6, ls="--", label="simulated (kinematic)")
        ax.fill_between(ph, d["trace"]["force_n"], d["sim_kin"], color=_KINEMATIC, alpha=0.12)
        ax.set_title(d["label"], fontsize=10)
        ax.set_xlabel("cycle phase [0-1]")
        ax.set_ylabel("force [N]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Phase-2 force vs cycle phase (pointwise agreement)", fontsize=12, y=1.02)
    fig.tight_layout()
    saved.append(_save(fig, out / "fig2_force_vs_phase.png"))

    # 3. Dynamics diagnostics.
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for row, (name, _fixture, label) in enumerate(_FIXTURES):
        d = data[name]
        k = d["kin"]
        ph = k.phase
        area = k.active_columns * k.column_area_m2 * 1.0e4
        peak = int(np.argmax(k.force_n))
        a0, a1, a2, a3 = axes[row]
        a0.plot(k.cop_x_m * 1000.0, k.cop_y_m * 1000.0, color=_KINEMATIC, lw=1.5)
        a0.scatter(
            [k.cop_x_m[peak] * 1000.0], [k.cop_y_m[peak] * 1000.0], color=_MEASURED, zorder=5, label="peak-force COP"
        )
        a0.set_title(f"{label}\nCOP trajectory")
        a0.set_xlabel("COP x [mm]")
        a0.set_ylabel("COP y [mm]")
        a0.grid(alpha=0.3)
        a0.legend(fontsize=8)
        a0.axis("equal")
        a1.plot(ph, area, color="#2a9d8f", lw=1.8)
        a1.set_title("active contact area")
        a1.set_xlabel("phase")
        a1.set_ylabel("area [cm^2]")
        a1.grid(alpha=0.3)
        a2.plot(ph, k.max_compression_m * 1000.0, color="#e76f51", lw=1.8)
        a2.set_title("max column compression")
        a2.set_xlabel("phase")
        a2.set_ylabel("compression [mm]")
        a2.grid(alpha=0.3)
        jump = np.abs(np.diff(k.force_n))
        a3.plot(ph[1:], jump, color="#6a4c93", lw=1.2)
        a3.axhline(
            0.25 * float(np.max(d["trace"]["force_n"])), color=_MEASURED, ls=":", label="25% peak (continuity gate)"
        )
        a3.set_title(f"wrench continuity\nmax dF/frame = {jump.max():.0f} N")
        a3.set_xlabel("phase")
        a3.set_ylabel("|dFz| per frame [N]")
        a3.grid(alpha=0.3)
        a3.legend(fontsize=8)
    fig.suptitle(
        "Phase-2 dynamics diagnostics (kinematic drive): COP, active area, compression, wrench continuity",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    saved.append(_save(fig, out / "fig3_dynamics_diagnostics.png"))

    # 4. Drive comparison.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, (name, _fixture, label) in enumerate(_FIXTURES):
        d = data[name]
        disp = d["trace"]["displacement_m"] * 1000.0
        srv = d["srv"]
        left = axes[row, 0]
        left.plot(disp, d["trace"]["force_n"], color=_MEASURED, lw=2.2, label="measured")
        left.plot(
            disp,
            d["sim_kin"],
            color=_KINEMATIC,
            lw=1.6,
            ls="--",
            label=f"kinematic (hyst {d['mk'].hysteresis_error * 100:.0f}%)",
        )
        left.plot(
            disp,
            d["sim_srv"],
            color=_SERVO,
            lw=1.6,
            ls="-.",
            label=f"servo PD (hyst {d['ms'].hysteresis_error * 100:.0f}%)",
        )
        left.set_title(f"{label}: loops by drive mode")
        left.set_xlabel("displacement [mm]")
        left.set_ylabel("force [N]")
        left.grid(alpha=0.3)
        left.legend(fontsize=8)
        right = axes[row, 1]
        track = float(np.max(np.abs(srv.commanded_depth_m - srv.achieved_depth_m))) * 1000.0
        right.plot(srv.phase, srv.commanded_depth_m * 1000.0, color=_MEASURED, lw=2.0, label="commanded depth")
        right.plot(
            srv.phase, srv.achieved_depth_m * 1000.0, color=_SERVO, lw=1.4, ls="--", label="servo achieved depth"
        )
        right.set_title(f"servo tracking (max err {track:.2f} mm)")
        right.set_xlabel("phase")
        right.set_ylabel("depth [mm]")
        right.grid(alpha=0.3)
        right.legend(fontsize=8)
    fig.suptitle(
        "Phase-2 drive comparison: kinematic crosshead (faithful) vs closed-loop PD servo (stress test)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    saved.append(_save(fig, out / "fig4_kinematic_vs_servo.png"))
    return saved


def _save(fig, path: Path):
    """Save a figure to ``path`` and close it."""
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Render the Phase-2 validation figures from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument("--warmup-cycles", type=int, default=3)
    args = parser.parse_args()
    paths = render_figures(args.manifest, args.out_dir, substeps=args.substeps, warmup_cycles=args.warmup_cycles)
    print("saved:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
