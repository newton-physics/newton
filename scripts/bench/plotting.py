# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared plotting utilities for the benchmark platform.

Enforces log-log axes (per CLAUDE.md convention), consistent styling,
IQR bands, and power-law exponent annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scripts.bench.infra import power_law_exponent


@dataclass
class PlotStyle:
    color: str
    marker: str
    linestyle: str
    label: str


# Consistent style registry for stepping modes.
STYLES: dict[str, PlotStyle] = {
    "graph": PlotStyle("#1f77b4", "o", "-", "CENIC adaptive"),
    "fixed": PlotStyle("#ff7f0e", "D", "-", "Fixed-step (dt=10 ms)"),
    "manual": PlotStyle("#d62728", "s", "--", "Manual (fixed K, no graph)"),
}


@dataclass
class SeriesData:
    medians: list[float]
    p25: list[float] | None = None
    p75: list[float] | None = None


def log_log_plot(
    ax: Axes,
    ns: list[int],
    series: dict[str, SeriesData],
    ylabel: str,
    title: str,
    show_exponents: bool = True,
    show_iqr: bool = True,
) -> dict[str, float]:
    """Standard log-log scaling plot. Returns {mode: exponent}."""
    exponents = {}
    for mode, sd in series.items():
        style = STYLES.get(mode)
        if style is None:
            continue
        exp = power_law_exponent(ns, sd.medians)
        exponents[mode] = exp
        label = f'{style.label}  $N^{{{exp:.2f}}}$' if show_exponents else style.label
        ax.plot(
            ns, sd.medians,
            color=style.color, marker=style.marker, ls=style.linestyle,
            lw=2, ms=5, label=label,
        )
        if show_iqr and sd.p25 is not None and sd.p75 is not None:
            ax.fill_between(ns, sd.p25, sd.p75, color=style.color, alpha=0.10)

    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    return exponents


def save_fig(fig: Figure, path: str | Path, dpi: int = 150) -> None:
    """tight_layout + savefig + close."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  saved -> {path}", flush=True)
