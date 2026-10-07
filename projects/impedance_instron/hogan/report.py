# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Write a static HTML report that explains one Hogan impedance run step by step.

Example::

    uv run python -m projects.impedance_instron.hogan.report \\
        --run runs/hogan_time --compare runs/hogan_touchdown --baseline runs/hogan_before
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import math
from pathlib import Path

import numpy as np

from ..cartesian import data as reference_data
from ..cartesian import profile as profile_data
from ..cartesian.shoe import Shoe
from .mechanics import COORDINATE_NAMES, RestOfBody, chain_from_profile
from .plan import CONTACT_THRESHOLD_N, build_plan

MEASURED = "#3d4b5a"
RUN_COLORS = ("#1261a0", "#d94801", "#2b8a3e", "#6a3d9a")
BASELINE = "#c08a00"
ROTATIONAL = ("pelvis", "hip", "knee", "ankle")
COORDINATE_INFO = {
    "hip_x": ("Hip center, forward", "m", "N/m", "N·s/m"),
    "hip_z": ("Hip center, up", "m", "N/m", "N·s/m"),
    "pelvis": ("Lumped body tilt from upright (\u2212 = forward lean)", "rad", "N·m/rad", "N·m·s/rad"),
    "hip": ("Thigh relative to pelvis (+ = flexion)", "rad", "N·m/rad", "N·m·s/rad"),
    "knee": ("Shank relative to thigh (\u2212 = flexion)", "rad", "N·m/rad", "N·m·s/rad"),
    "ankle": ("Foot relative to shank (+ = dorsiflexion, 0 = foot ⟂ shank)", "rad", "N·m/rad", "N·m·s/rad"),
}


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    raw = (hi - lo) / max(count - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw * 0.999)
    start = math.ceil(lo / step - 1e-9) * step
    return [round(start + i * step, 10) for i in range(int((hi - start) / step + 1e-9) + 1)]


def _plot(series, *, xlabel: str, ylabel: str, band=None, xlim=None) -> str:
    """Return an SVG line plot; each series is ``(label, x, y, color, dash)``."""
    left, right, top, bottom = 78.0, 700.0, 66.0, 270.0
    xs = np.concatenate([np.asarray(s[1], dtype=float) for s in series])
    ys = np.concatenate([np.asarray(s[2], dtype=float) for s in series])
    ys = ys[np.isfinite(ys)]
    xlim = xlim or (float(np.nanmin(xs)), float(np.nanmax(xs)))
    lo, hi = (float(ys.min()), float(ys.max())) if len(ys) else (0.0, 1.0)
    pad = max(0.06 * (hi - lo), 1e-6 + 1e-3 * max(abs(lo), abs(hi)))
    lo, hi = lo - pad, hi + pad

    def sx(x):
        return left + (np.asarray(x) - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)

    def sy(y):
        return bottom - (np.asarray(y) - lo) / (hi - lo) * (bottom - top)

    parts = []
    if band is not None:
        x0, x1 = sx(max(band[0], xlim[0])), sx(min(band[1], xlim[1]))
        parts.append(f'<rect x="{x0:.1f}" y="{top}" width="{x1 - x0:.1f}" height="{bottom - top}" fill="#f1f6fb"/>')
    for value in _nice_ticks(*xlim):
        px = sx(value)
        parts.append(
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{bottom}" stroke="#e5eaf0"/>'
            f'<text x="{px:.1f}" y="{bottom + 18}" text-anchor="middle">{value:g}</text>'
        )
    for value in _nice_ticks(lo, hi):
        py = sy(value)
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}" stroke="#e5eaf0"/>'
            f'<text x="{left - 8}" y="{py + 4:.1f}" text-anchor="end">{value:g}</text>'
        )
    if lo < 0.0 < hi:
        parts.append(f'<line x1="{left}" y1="{sy(0.0):.1f}" x2="{right}" y2="{sy(0.0):.1f}" stroke="#9aa8b8"/>')
    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9aa8b8"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#9aa8b8"/>'
    )
    legend_x, legend_y = left, 22
    for label, x_raw, y_raw, color, dash in series:
        x, y = np.asarray(x_raw, dtype=float), np.asarray(y_raw, dtype=float)
        keep = np.isfinite(y) & (x >= xlim[0]) & (x <= xlim[1])
        stride = max(1, int(np.count_nonzero(keep) // 900))
        x, y = x[keep][::stride], y[keep][::stride]
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(sx(x), sy(y), strict=True))
        style = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"{style}/>')
        entry = 44 + 7.6 * len(label)
        if legend_x + entry > right + 20 and legend_x > left:
            legend_x, legend_y = left, legend_y + 20
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 26}" y2="{legend_y}" stroke="{color}" stroke-width="3"{style}/>'
            f'<text x="{legend_x + 32}" y="{legend_y + 5}">{html.escape(label)}</text>'
        )
        legend_x += entry
    return (
        f'<svg class="plot" viewBox="0 0 720 324" role="img" aria-label="{html.escape(ylabel)} versus {html.escape(xlabel)}">'
        '<g font-family="system-ui,sans-serif" font-size="15" fill="#526174">'
        + "".join(parts)
        + f'<text x="389" y="316" text-anchor="middle">{html.escape(xlabel)}</text>'
        f'<text x="18" y="{(top + bottom) / 2:.0f}" transform="rotate(-90 18 {(top + bottom) / 2:.0f})" '
        f'text-anchor="middle">{html.escape(ylabel)}</text></g></svg>'
    )


def _figure(title: str, svg: str, caption: str = "") -> str:
    note = f"<span>{caption}</span>" if caption else ""
    return f"<figure><figcaption><strong>{title}</strong>{note}</figcaption>{svg}</figure>"


def _hull(points: np.ndarray) -> np.ndarray:
    ordered = sorted(set(map(tuple, np.round(points, 6))))
    lower, upper = [], []
    for chain, sequence in ((lower, ordered), (upper, reversed(ordered))):
        for point in sequence:
            while len(chain) >= 2:
                a, b = np.subtract(chain[-1], chain[-2]), np.subtract(point, chain[-1])
                if a[0] * b[1] - a[1] * b[0] > 0:
                    break
                chain.pop()
            chain.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def _table(header: list[str], rows: list[list[str]], caption: str = "") -> str:
    head = "".join(f'<th scope="col">{h}</th>' for h in header)
    body = "".join(
        "<tr>" + f'<th scope="row">{r[0]}</th>' + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>" for r in rows
    )
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="table-scroll"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


class _Scene:
    """Draw sagittal stick figures with the registered shoe outline and ground forces."""

    def __init__(self, chain, shoe: Shoe, rest_com_local_m):
        self.chain = chain
        self.shoe = shoe
        self.last = _hull(shoe.last_vertices_local_m[:, (0, 2)])
        self.midsole = _hull(np.vstack((shoe.anchor_local_m, shoe.attachment_local_m))[:, (0, 2)])
        self.rest_com = np.asarray(rest_com_local_m, dtype=float)

    def _shoe_points(self, local: np.ndarray, ankle: np.ndarray, q) -> np.ndarray:
        angle = self.chain.angle(q, 3) - self.shoe.static_pitch_rad
        c, s = math.cos(angle), math.sin(angle)
        return local @ np.array([[c, s], [-s, c]]) + ankle

    def pose(self, q, to_px, *, simulated: bool, labels: bool = False) -> str:
        hip, knee, ankle, _ = self.chain.kinematics(q)
        pelvis = float(q[2])
        rotation = np.array([[math.cos(pelvis), -math.sin(pelvis)], [math.sin(pelvis), math.cos(pelvis)]])
        com = hip + rotation @ self.rest_com

        def poly(points):
            return " ".join(f"{a:.1f},{b:.1f}" for a, b in (to_px(p) for p in points))

        sole = poly(self._shoe_points(self.midsole, ankle, q))
        out = []
        if simulated:
            last = poly(self._shoe_points(self.last, ankle, q))
            out.append(f'<polygon points="{last}" fill="#e7b270" fill-opacity=".55" stroke="#b5803c"/>')
            out.append(f'<polygon points="{sole}" fill="#5c9f93" fill-opacity=".55" stroke="#3e7a6f"/>')
            stroke, width, dash = "#1261a0", 4.5, ""
        else:
            out.append(f'<polygon points="{sole}" fill="none" stroke="#8a97a6" stroke-dasharray="4 3"/>')
            stroke, width, dash = "#8a97a6", 2.2, ' stroke-dasharray="6 4"'
        out.append(
            f'<polyline points="{poly([com, hip])}" fill="none" stroke="{stroke}" stroke-width="{width}"{dash}/>'
        )
        out.append(
            f'<polyline points="{poly([hip, knee, ankle])}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round"{dash}/>'
        )
        c = to_px(com)
        out.append(
            f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{9 if simulated else 7}" '
            f'fill="{"#1261a0" if simulated else "none"}" stroke="{stroke}"/>'
        )
        for joint in (hip, knee, ankle):
            p = to_px(joint)
            out.append(
                f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3.5" fill="white" stroke="{stroke}" stroke-width="2"/>'
            )
        if labels:
            mass = self.chain.masses_kg
            for point, text, dx in (
                (com, f"Lumped rest of body {mass[0]:.1f} kg", 14),
                (hip, "Hip center (x, z free)", 10),
                (0.5 * (hip + knee), f"Thigh {mass[1]:.1f} kg", 10),
                (knee, "Knee", 10),
                (0.5 * (knee + ankle), f"Shank {mass[2]:.1f} kg", 10),
                (ankle, f"Ankle; foot + shoe carrier {mass[3]:.1f} kg", 14),
            ):
                p = to_px(point)
                out.append(f'<text x="{p[0] + dx:.1f}" y="{p[1] + 4:.1f}" fill="#172b40">{html.escape(text)}</text>')
        return "".join(out)

    @staticmethod
    def arrow(origin, force, to_px, color: str, scale_m_per_n: float = 2.0e-4) -> str:
        a = to_px(origin)
        b = to_px(np.asarray(origin) + scale_m_per_n * np.asarray(force))
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        head = [(b[0] - 9 * math.cos(angle + s), b[1] - 9 * math.sin(angle + s)) for s in (-0.45, 0.45)]
        return (
            f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{color}" stroke-width="2.5"/>'
            f'<polygon points="{b[0]:.1f},{b[1]:.1f} {head[0][0]:.1f},{head[0][1]:.1f} {head[1][0]:.1f},{head[1][1]:.1f}" fill="{color}"/>'
        )


def _simulated_cop(chain, trace) -> np.ndarray:
    """Return the simulated center of pressure on the ground [m]; NaN out of contact."""
    cop = np.full(len(trace["time_s"]), np.nan)
    for k, (q, force, moment) in enumerate(
        zip(trace["state"], trace["grf_n"], trace["ankle_contact_moment_nm"], strict=True)
    ):
        if force[1] > CONTACT_THRESHOLD_N:
            ankle = chain.kinematics(q)[2]
            cop[k] = ankle[0] + (moment - ankle[1] * force[0]) / force[1]
    return cop


def _snapshots(scene: _Scene, reference, trace, times, *, label: str) -> str:
    """Draw a whole-chain row and a zoomed foot row at each time."""
    width = 260.0
    # (row, panel height [px], scale [px/m], ground line [px], arrow scale [m/N])
    rows = (("body", 380.0, 215.0, 350.0, 2.0e-4), ("foot", 250.0, 700.0, 220.0, 1.0e-4))
    cop_sim = _simulated_cop(scene.chain, trace)
    panels = []
    top = 0.0
    for row, height, scale, ground, force_scale in rows:
        for i, t in enumerate(times):
            k = int(np.argmin(np.abs(trace["time_s"] - t)))
            q, q_ref = trace["state"][k], trace["reference_state"][k]
            hip_ref, _, ankle_ref, _ = scene.chain.kinematics(q_ref)
            cx = 0.5 * (hip_ref[0] + ankle_ref[0]) if row == "body" else ankle_ref[0] + 0.03

            def to_px(p, cx=cx, scale=scale, ground=ground):
                return (width / 2 + (p[0] - cx) * scale, ground - p[1] * scale)

            measured = np.array(
                [np.interp(t, reference["grf_time_s"], reference["grf_target_n"][:, j]) for j in range(2)]
            )
            measured_cop = float(np.interp(t, reference["grf_time_s"], reference["cop_target_m"]))
            parts = [
                f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" fill="none" stroke="#eef2f6"/>',
                f'<line x1="0" y1="{ground}" x2="{width}" y2="{ground}" stroke="#7b9d86" stroke-width="2"/>',
                scene.pose(q_ref, to_px, simulated=False),
                scene.pose(q, to_px, simulated=True),
            ]
            if measured[1] > CONTACT_THRESHOLD_N:
                parts.append(_Scene.arrow([measured_cop, 0.0], measured, to_px, "#7a5230", force_scale))
            if np.isfinite(cop_sim[k]):
                parts.append(_Scene.arrow([cop_sim[k], 0.0], trace["grf_n"][k], to_px, "#d94801", force_scale))
            if row == "body":
                parts.append(
                    f'<text x="{width / 2}" y="22" text-anchor="middle" font-weight="650" fill="#172b40">{t * 1000:.0f} ms</text>'
                    f'<text x="{width / 2}" y="42" text-anchor="middle">Fz meas. {measured[1]:.0f} N · sim. {trace["grf_n"][k][1]:.0f} N</text>'
                )
            else:
                pitch = math.degrees(scene.chain.angle(q, 3) - scene.chain.angle(q_ref, 3))
                parts.append(
                    f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle">Shoe pitch sim &minus; ref {pitch:+.1f}°</text>'
                )
            panels.append(
                f'<svg x="{i * width}" y="{top}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
                + "".join(parts)
                + "</svg>"
            )
        top += height
    total = width * len(times)
    return (
        f'<svg class="scene" viewBox="0 0 {total:.0f} {top:.0f}" role="img" aria-label="{html.escape(label)}">'
        '<g font-family="system-ui,sans-serif" font-size="14" fill="#526174">' + "".join(panels) + "</g></svg>"
    )


def _model_figure(scene: _Scene, reference, trace, t: float) -> str:
    width, height, scale, ground = 470.0, 540.0, 390.0, 505.0
    k = int(np.argmin(np.abs(trace["time_s"] - t)))
    q = trace["state"][k]
    hip, _, ankle, _ = scene.chain.kinematics(q)
    cx = 0.5 * (hip[0] + ankle[0]) + 0.12

    def to_px(p):
        return (width / 2 + (p[0] - cx) * scale, ground - p[1] * scale)

    cop = _simulated_cop(scene.chain, trace)[k]
    parts = [
        f'<line x1="10" y1="{ground}" x2="{width - 10}" y2="{ground}" stroke="#7b9d86" stroke-width="2"/>',
        scene.pose(q, to_px, simulated=True, labels=True),
    ]
    if np.isfinite(cop):
        parts.append(_Scene.arrow([cop, 0.0], trace["grf_n"][k], to_px, "#d94801", 1.5e-4))
        tip = to_px(np.array([cop, 0.0]) + 1.5e-4 * trace["grf_n"][k])
        parts.append(
            f'<text x="{tip[0] - 8:.1f}" y="{tip[1]:.1f}" text-anchor="end" fill="#d94801">Shoe ground force at COP</text>'
        )
    parts.append(
        f'<text x="{width - 14}" y="{ground + 20}" text-anchor="end" fill="#3e7a6f">Ground z = 0 · column-bed midsole (green) + rigid last (gold)</text>'
    )
    return (
        f'<svg class="model" viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Annotated model">'
        '<g font-family="system-ui,sans-serif" font-size="13" fill="#526174">' + "".join(parts) + "</g></svg>"
    )


def _load(directory: Path) -> tuple[dict, list[tuple[dict, dict]]]:
    summary = json.loads((directory / "summary.json").read_text())
    runs = []
    for run in summary["runs"]:
        with np.load(directory / f"{run['name']}.npz") as trace:
            runs.append((run, {k: trace[k] for k in trace.files}))
    return summary, runs


def _run_label(run: dict) -> str:
    label = f"Sim, {run['phase']} phase"
    if run.get("modulus_scale", 1.0) != 1.0:
        label += f", modulus \u00d7{run['modulus_scale']:g}"
    if run.get("adaptation", "fixed") != "fixed":
        label += f", {run['adaptation']} gains"
    return label


def write_report(
    run_dir: Path,
    *,
    compare: list[Path] = (),
    baseline: Path | None = None,
    mount_m=None,
    output: Path | None = None,
) -> Path:
    """Render the explanatory report for a Hogan run directory.

    Args:
        run_dir: Output of ``python -m projects.impedance_instron.hogan``; its plan is rebuilt.
        compare: Further run directories on the same reference, e.g. another phase mode.
        baseline: An earlier run shown as "before" in the force comparison.
        mount_m: Ankle location in the intrinsic shoe frame [m]; needed only when the
            summary predates recording it.
        output: HTML destination. Defaults to ``run_dir / "report.html"``.

    Returns:
        Path of the written report.
    """
    summary, runs = _load(run_dir)
    reference = reference_data.load(Path(summary["reference"]))
    profile = profile_data.load(Path(summary["profile"]))
    rest = summary["rest_of_body"]
    chain = chain_from_profile(reference, profile, RestOfBody(tuple(rest["com_local_m"]), rest["radius_of_gyration_m"]))
    mount = summary.get("shoe_mount_m", mount_m)
    if mount is None:
        raise ValueError("The summary does not record the shoe mount; pass mount_m")
    pitch = summary.get(
        "shoe_static_pitch_rad", float(reference.get("shoe_static_pitch_rad", reference["static_pitch_rad"]))
    )
    shoe = Shoe(summary["shoe_artifact"], mount, pitch)
    scene = _Scene(chain, shoe, rest["com_local_m"])
    diagnostics = summary["plan"]
    offset = diagnostics["height_offset_m"]
    dt = runs[0][0]["dt_s"]
    cutoff = diagnostics.get("kinematic_cutoff_hz")
    pelvis_cutoff = diagnostics.get("pelvis_cutoff_hz")

    # Rebuild the inverse-dynamics plan one modeling step at a time.
    steps = [
        ("Marker hip, raw pelvis tilt, FK ankle", {"hip_source": "markers", "pelvis_cutoff_hz": None, "leg_ik": False}),
        ("+ hip from GRF-integrated COM", {"hip_source": "grf_com", "pelvis_cutoff_hz": None, "leg_ik": False}),
        (
            f"+ pelvis tilt low-passed at {pelvis_cutoff:g} Hz",
            {"hip_source": "grf_com", "pelvis_cutoff_hz": pelvis_cutoff, "leg_ik": False},
        ),
        (
            "+ leg IK onto measured ankle (used)",
            {"hip_source": "grf_com", "pelvis_cutoff_hz": pelvis_cutoff, "leg_ik": True},
        ),
    ]
    plans = [
        build_plan(reference, chain, dt_s=dt, height_offset_m=offset, cutoff_hz=cutoff, **options)
        for _, options in steps
    ]
    plan = plans[-1]
    clock_ms = plan.time_s * 1000.0
    ankle_measured = np.column_stack(
        [np.interp(plan.time_s, reference["time_s"], reference["ankle_target_m"][:, j]) for j in range(2)]
    ) + np.array([0.0, offset])
    weight = chain.total_mass_kg * 9.81
    step_rows = []
    for (name, _), p in zip(steps, plans, strict=True):
        d = p.diagnostics
        ankle = np.array([chain.kinematics(row)[2] for row in p.q])
        error = np.max(np.abs(ankle - ankle_measured), axis=0) * 1000.0
        torque = d["joint_torque_peak_nm"]
        step_rows.append(
            [
                html.escape(name),
                f"{d['residual_force_rms_n'][0] / weight:.2f} / {d['residual_force_rms_n'][1] / weight:.2f}",
                f"{d['residual_moment_rms_nm']:.0f} ({d['residual_moment_peak_nm']:.0f})",
                f"{error[0]:.0f} / {error[1]:.0f}",
                f"{torque['hip']:.0f} / {torque['knee']:.0f} / {torque['ankle']:.0f}",
            ]
        )

    grf_t = reference["grf_time_s"]
    grf = reference["grf_target_n"]
    loaded = np.flatnonzero(grf[:, 1] > CONTACT_THRESHOLD_N)
    contact = (float(grf_t[loaded[0]]) * 1000.0, float(grf_t[loaded[-1]]) * 1000.0)
    peak_index = int(np.argmax(grf[:, 1]))

    all_runs = [
        (_run_label(run), run, trace, RUN_COLORS[i % len(RUN_COLORS)], "") for i, (run, trace) in enumerate(runs)
    ]
    for directory in compare:
        _, extra = _load(directory)
        for run, trace in extra:
            all_runs.append((_run_label(run), run, trace, RUN_COLORS[len(all_runs) % len(RUN_COLORS)], "7 4"))
    before = None
    if baseline is not None:
        _, base_runs = _load(baseline)
        before = ("Before (marker hip, FK ankle)", base_runs[0][0], base_runs[0][1], BASELINE, "2 3")
    main_label, main_run, main_trace, _, _ = all_runs[0]
    trace_ms = main_trace["time_s"] * 1000.0

    # Results table.
    result_rows = [
        [
            "Measured",
            "—",
            f"{grf[peak_index, 1]:.0f}",
            "—",
            f"{contact[0]:.1f}",
            f"{contact[1]:.1f}",
            "—",
            "—",
            "—",
        ]
    ]
    for label, run, _, _, _ in [*all_runs, *([before] if before else [])]:
        track = run["tracking_rmse"]
        result_rows.append(
            [
                html.escape(label),
                run["phase"],
                f"{run['peak_grf_n'][1]:.0f}",
                f"{run['grf_rmse_n'][1]:.0f} / {run['grf_rmse_n'][0]:.0f}",
                f"{1000 * run['touchdown_s']:.1f}" if run.get("touchdown_s") is not None else "—",
                f"{1000 * run['toeoff_s']:.1f}" if run.get("toeoff_s") is not None else "—",
                f"{1000 * track['hip_x']:.1f} / {1000 * track['hip_z']:.1f}",
                " / ".join(f"{math.degrees(track[n]):.1f}" for n in ("hip", "knee", "ankle")),
                f"{100 * run['maximum_compression_fraction']:.0f}%",
            ]
        )

    def run_series(transform, *, include_before: bool = True):
        series = []
        for label, _, trace, color, dash in all_runs:
            series.append((label, trace["time_s"] * 1000.0, transform(trace), color, dash))
        if include_before and before:
            series.append((before[0], before[2]["time_s"] * 1000.0, transform(before[2]), before[3], before[4]))
        return series

    xlim = (0.0, float(clock_ms[-1]))
    band = contact
    grf_plots = [
        _figure(
            "Vertical ground force",
            _plot(
                [("Measured", grf_t * 1000, grf[:, 1], MEASURED, ""), *run_series(lambda tr: tr["grf_n"][:, 1])],
                xlabel="Time [ms]",
                ylabel="Fz [N]",
                band=band,
                xlim=xlim,
            ),
            "Measured force plate vs force produced by the simulated shoe.",
        ),
        _figure(
            "Fore-aft ground force",
            _plot(
                [("Measured", grf_t * 1000, grf[:, 0], MEASURED, ""), *run_series(lambda tr: tr["grf_n"][:, 0])],
                xlabel="Time [ms]",
                ylabel="Fx [N] (+ forward)",
                band=band,
                xlim=xlim,
            ),
            "Negative = braking, positive = propulsion.",
        ),
    ]
    ankle_ref = np.array([chain.kinematics(q)[2] for q in main_trace["reference_state"]])
    measured_cop = np.interp(main_trace["time_s"], grf_t, reference["cop_target_m"])
    # The COP is ill-conditioned at light load.
    cop_threshold = 150.0
    measured_loaded = np.interp(main_trace["time_s"], grf_t, grf[:, 1]) > cop_threshold

    def cop_relative(trace):
        cop = _simulated_cop(chain, trace)
        cop[trace["grf_n"][:, 1] <= cop_threshold] = np.nan
        return 1000 * (cop - np.array([chain.kinematics(q)[2][0] for q in trace["state"]]))

    cop_plot = _figure(
        "Center of pressure relative to the ankle",
        _plot(
            [
                (
                    "Measured COP \u2212 reference ankle",
                    trace_ms,
                    np.where(measured_loaded, 1000 * (measured_cop - ankle_ref[:, 0]), np.nan),
                    MEASURED,
                    "",
                ),
                *[
                    (label, tr["time_s"] * 1000, cop_relative(tr), color, dash)
                    for label, _, tr, color, dash in all_runs
                ],
            ],
            xlabel="Time [ms]",
            ylabel="COP \u2212 ankle x [mm]",
            band=band,
            xlim=xlim,
        ),
        f"Shown where Fz &gt; {cop_threshold:.0f} N. The measured COP moves from the heel to well ahead of the ankle. "
        "The treadmill force plate's COP is unreliable in early and late stance, so differences there are accepted; "
        "compare the curves in mid-stance.",
    )

    def degrees(values):
        return np.degrees(values)

    tracking_plots = []
    for j, name in enumerate(COORDINATE_NAMES):
        unit = "mm" if j < 2 else "deg"
        scale = 1000.0 if j < 2 else math.degrees(1.0)
        if j < 2:
            series = [
                (label, tr["time_s"] * 1000, scale * (tr["state"][:, j] - tr["reference_state"][:, j]), color, dash)
                for label, _, tr, color, dash in all_runs
            ]
            title = f"{COORDINATE_INFO[name][0]}: simulated \u2212 reference"
        else:
            # Show the pelvis as tilt from upright.
            shift = 90.0 if name == "pelvis" else 0.0
            series = [("Reference", trace_ms, scale * main_trace["reference_state"][:, j] - shift, MEASURED, "")]
            series += [
                (label, tr["time_s"] * 1000, scale * tr["state"][:, j] - shift, color, dash)
                for label, _, tr, color, dash in all_runs
            ]
            title = COORDINATE_INFO[name][0]
        tracking_plots.append(
            _figure(title, _plot(series, xlabel="Time [ms]", ylabel=f"{name} [{unit}]", band=band, xlim=xlim))
        )
    foot_ref = degrees(np.array([chain.angle(q, 3) for q in main_trace["reference_state"]]) - pitch)
    tracking_plots.append(
        _figure(
            "Shoe pitch relative to the static (level) shoe, + toe-up",
            _plot(
                [("Reference", trace_ms, foot_ref, MEASURED, "")]
                + [
                    (
                        label,
                        tr["time_s"] * 1000,
                        degrees(np.array([chain.angle(q, 3) for q in tr["state"]]) - pitch),
                        color,
                        dash,
                    )
                    for label, _, tr, color, dash in all_runs
                ],
                xlabel="Time [ms]",
                ylabel="Shoe pitch [deg]",
                band=band,
                xlim=xlim,
            ),
            "0° is the static standing pose, where the shoe sole is level.",
        )
    )

    torque_plots = []
    feedback_rows = []
    feedback = main_trace["load"] - main_trace["feedforward"]
    for j, name in enumerate(COORDINATE_NAMES):
        unit = "N" if j < 2 else "N·m"
        ff_rms = float(np.sqrt(np.mean(np.square(main_trace["feedforward"][:, j]))))
        fb_rms = float(np.sqrt(np.mean(np.square(feedback[:, j]))))
        if j < 2:
            note = f"External support force on the lumped body, {100 * fb_rms / weight:.0f}% body weight RMS"
            ratio = "—"
        else:
            note = "External residual moment on the lumped body" if name == "pelvis" else "Joint torque"
            ratio = f"{fb_rms / max(ff_rms, 1e-9):.2f}"
        feedback_rows.append([name, f"{ff_rms:.0f} {unit}", f"{fb_rms:.0f} {unit}", ratio, note])
        if j >= 3:
            torque_plots.append(
                _figure(
                    f"{name.capitalize()} torque ({main_label})",
                    _plot(
                        [
                            ("Feedforward (ID)", trace_ms, main_trace["feedforward"][:, j], MEASURED, ""),
                            ("Impedance feedback", trace_ms, feedback[:, j], "#d94801", ""),
                            ("Total applied", trace_ms, main_trace["load"][:, j], "#1261a0", ""),
                        ],
                        xlabel="Time [ms]",
                        ylabel=f"{name} torque [N·m]",
                        band=band,
                        xlim=xlim,
                    ),
                )
            )
    compression_plot = _figure(
        "Peak column compression",
        _plot(
            [
                (label, tr["time_s"] * 1000, 100 * tr["compression_fraction"], color, dash)
                for label, _, tr, color, dash in all_runs
            ],
            xlabel="Time [ms]",
            ylabel="Max compression / rest length [%]",
            band=band,
            xlim=xlim,
        ),
        "Largest compression of any driven midsole column. The rollout stops above 90%.",
    )

    # Inverse-dynamics plots.
    marker_plan, raw_pelvis_plan, fk_plan = plans[0], plans[1], plans[2]
    id_plots = [
        _figure(
            "Hip reference: GRF-consistent minus marker hip",
            _plot(
                [
                    ("Forward", clock_ms, 1000 * (plan.q[:, 0] - marker_plan.q[:, 0]), "#1261a0", ""),
                    ("Up", clock_ms, 1000 * (plan.q[:, 1] - marker_plan.q[:, 1]), "#d94801", ""),
                ],
                xlabel="Time [ms]",
                ylabel="Hip shift [mm]",
                band=band,
                xlim=xlim,
            ),
            "How far the hip had to move so the whole-body COM follows the measured force.",
        ),
        _figure(
            "Lumped body (pelvis) tilt",
            _plot(
                [
                    ("Measured pelvis", clock_ms, degrees(raw_pelvis_plan.q[:, 2]) - 90.0, "#8a97a6", ""),
                    (f"Low-passed {pelvis_cutoff:g} Hz (used)", clock_ms, degrees(plan.q[:, 2]) - 90.0, "#1261a0", ""),
                ],
                xlabel="Time [ms]",
                ylabel="Tilt from upright [deg]",
                band=band,
                xlim=xlim,
            ),
            "Negative = forward lean. The trunk does not follow the fast pelvis tilt at impact.",
        ),
        _figure(
            "Ankle position error vs measured ankle center",
            _plot(
                [
                    (
                        "FK forward",
                        clock_ms,
                        1000 * (np.array([chain.kinematics(q)[2][0] for q in fk_plan.q]) - ankle_measured[:, 0]),
                        "#8a97a6",
                        "",
                    ),
                    (
                        "FK up",
                        clock_ms,
                        1000 * (np.array([chain.kinematics(q)[2][1] for q in fk_plan.q]) - ankle_measured[:, 1]),
                        "#8a97a6",
                        "6 4",
                    ),
                    (
                        "IK forward (used)",
                        clock_ms,
                        1000 * (np.array([chain.kinematics(q)[2][0] for q in plan.q]) - ankle_measured[:, 0]),
                        "#1261a0",
                        "",
                    ),
                    (
                        "IK up (used)",
                        clock_ms,
                        1000 * (np.array([chain.kinematics(q)[2][1] for q in plan.q]) - ankle_measured[:, 1]),
                        "#1261a0",
                        "6 4",
                    ),
                ],
                xlabel="Time [ms]",
                ylabel="Ankle error [mm]",
                band=band,
                xlim=xlim,
            ),
            "FK = static segment lengths with Visual3D joint angles. The FK ankle runs low and forward in stance, pushing the shoe into the ground.",
        ),
        _figure(
            "Pelvis residual force (what the lumped body cannot explain)",
            _plot(
                [
                    ("Marker hip, forward", clock_ms, marker_plan.feedforward[:, 0], "#8a97a6", ""),
                    ("Marker hip, up", clock_ms, marker_plan.feedforward[:, 1], "#8a97a6", "6 4"),
                    ("Used, forward", clock_ms, plan.feedforward[:, 0], "#1261a0", ""),
                    ("Used, up", clock_ms, plan.feedforward[:, 1], "#1261a0", "6 4"),
                ],
                xlabel="Time [ms]",
                ylabel="Residual force [N]",
                band=band,
                xlim=xlim,
            ),
        ),
        _figure(
            "Pelvis residual moment",
            _plot(
                [
                    (name, clock_ms, p.feedforward[:, 2], color, "")
                    for (name, _), p, color in zip(
                        steps, plans, ("#c9ced6", "#8a97a6", "#d94801", "#1261a0"), strict=True
                    )
                ],
                xlabel="Time [ms]",
                ylabel="Residual moment [N·m]",
                band=band,
                xlim=xlim,
            ),
            "The remainder is angular momentum of the arms and swing leg, which a rigid lumped body cannot carry.",
        ),
        _figure(
            "Inverse-dynamics joint torques (used feedforward)",
            _plot(
                [
                    ("Hip", clock_ms, plan.feedforward[:, 3], "#1261a0", ""),
                    ("Knee", clock_ms, plan.feedforward[:, 4], "#d94801", ""),
                    ("Ankle", clock_ms, plan.feedforward[:, 5], "#2b8a3e", ""),
                ],
                xlabel="Time [ms]",
                ylabel="Torque [N·m]",
                band=band,
                xlim=xlim,
            ),
            "Positive torque rotates the distal segment counter-clockwise (toward +z from +x) relative to the proximal one.",
        ),
    ]

    stiffness = summary["impedance"]["stiffness"]
    damping = summary["impedance"]["damping"]
    gain_rows = [
        [
            name,
            html.escape(COORDINATE_INFO[name][0]),
            f"{stiffness[j]:g} {COORDINATE_INFO[name][2]}",
            f"{damping[j]:.1f} {COORDINATE_INFO[name][3]}",
        ]
        for j, name in enumerate(COORDINATE_NAMES)
    ]
    masses = chain.masses_kg
    lengths = reference["lengths_m"]
    body_rows = [
        [
            "Lumped rest of body (pelvis)",
            f"{masses[0]:.1f}",
            "—",
            f"COM {1000 * rest['com_local_m'][0]:.0f} mm above hip; radius of gyration {1000 * rest['radius_of_gyration_m']:.0f} mm (provisional)",
        ],
        ["Thigh", f"{masses[1]:.2f}", f"{1000 * lengths[0]:.0f}", "Profile (static calibration)"],
        ["Shank", f"{masses[2]:.2f}", f"{1000 * lengths[1]:.0f}", "Profile (static calibration)"],
        ["Foot + shoe carrier", f"{masses[3]:.2f}", "—", "Rigid; carries the column-bed midsole"],
    ]
    registration = summary["registration"]
    seat = registration["unloaded_ankle_height_m"] - registration["static_compression_m"]
    snap_times = [
        contact[0] / 1000.0,
        contact[0] / 1000.0 + 0.3 * (contact[1] - contact[0]) / 1000.0,
        float(grf_t[peak_index]),
        contact[0] / 1000.0 + 0.85 * (contact[1] - contact[0]) / 1000.0,
    ]
    snapshots = _snapshots(scene, reference, main_trace, snap_times, label="Stance snapshots")
    model_figure = _model_figure(scene, reference, main_trace, float(grf_t[peak_index]))
    meta = json.loads(str(reference.get("metadata_json", "{}")))
    trial = Path(summary["reference"]).parent.name
    used = plan.diagnostics
    main_dir = run_dir.as_posix()

    document = _PAGE.format(
        trial=html.escape(trial),
        side=html.escape(str(meta.get("side", "unspecified"))),
        mass=float(reference["subject_mass_kg"]),
        main_label=html.escape(main_label),
        workflow_rows=_table(
            ["Quantity", "How it is obtained", "Measured or simulated?"],
            [
                [
                    "Reference motion q<sub>ref</sub>(t)",
                    "Markers → filtered joint angles; hip moved so the COM follows the measured GRF; leg IK onto the measured ankle center",
                    "Measured (processed)",
                ],
                [
                    "Feedforward τ<sub>ff</sub>(t)",
                    "Inverse dynamics of q<sub>ref</sub> with the measured GRF applied at the measured COP",
                    "Computed from measurements",
                ],
                [
                    "Pelvis residual loads",
                    "The pelvis rows of τ<sub>ff</sub>: loads an exact rest-of-body model would not need",
                    "Computed from measurements",
                ],
                [
                    "Simulated motion q(t)",
                    "Forward integration of the 6-DOF chain from the measured initial state",
                    "<b>Simulated</b>",
                ],
                [
                    "Simulated GRF",
                    "Column-bed shoe model reacting to the simulated foot pose and velocity",
                    "<b>Simulated</b>",
                ],
                [
                    "Feedback torque",
                    "K (q<sub>ref</sub> &minus; q) + D (q̇<sub>ref</sub> &minus; q̇) with fixed K, D",
                    "<b>Simulated</b>",
                ],
            ],
        ),
        body_table=_table(["Body", "Mass [kg]", "Length [mm]", "Notes"], body_rows),
        gain_table=_table(["Coordinate", "Meaning", "Stiffness K", "Damping D (critical)"], gain_rows),
        model_figure=model_figure,
        static_ankle=1000 * registration["static_ankle_height_m"],
        unloaded=1000 * registration["unloaded_ankle_height_m"],
        static_compression=1000 * registration["static_compression_m"],
        static_load=registration["static_load_n"],
        seat=1000 * seat,
        offset=1000 * offset,
        columns=len(shoe.anchor_local_m),
        shoe_id=html.escape(shoe.shoe.shoe_id),
        static_pitch=math.degrees(pitch),
        cutoff=cutoff or 0.0,
        pelvis_cutoff=pelvis_cutoff or 0.0,
        step_table=_table(
            [
                "Plan",
                "Pelvis residual force RMS [BW] fwd / up",
                "Residual moment RMS (peak) [N·m]",
                "Peak ankle error [mm] fwd / up",
                "Peak ID torque [N·m] hip / knee / ankle",
            ],
            step_rows,
            "Each row adds one step to the row above. The last row is the plan the controller tracks.",
        ),
        hip_rms=" / ".join(f"{1000 * v:.1f}" for v in used["hip_adjustment_rms_m"]),
        hip_peak=" / ".join(f"{1000 * v:.0f}" for v in used["hip_adjustment_peak_m"]),
        id_plots="".join(id_plots),
        result_table=_table(
            [
                "Run",
                "Phase",
                "Peak Fz [N]",
                "GRF RMSE [N] Fz / Fx",
                "Touchdown [ms]",
                "Toe-off [ms]",
                "Hip RMSE [mm] fwd / up",
                "Joint RMSE [deg] hip / knee / ankle",
                "Max compression",
            ],
            result_rows,
            f"Contact threshold {CONTACT_THRESHOLD_N:.0f} N. Tracking errors are against the reference at the same clock time.",
        ),
        snapshots=snapshots,
        grf_plots="".join(grf_plots),
        cop_plot=cop_plot,
        compression_plot=compression_plot,
        tracking_plots="".join(tracking_plots),
        feedback_table=_table(
            ["Coordinate", "Feedforward RMS", "Feedback RMS", "Feedback / feedforward", "What it is"],
            feedback_rows,
            f"{html.escape(main_label)}, whole rollout.",
        ),
        torque_plots="".join(torque_plots),
        main_dir=html.escape(main_dir),
        reference_path=html.escape(Path(summary["reference"]).as_posix()),
        profile_path=html.escape(Path(summary["profile"]).as_posix()),
        shoe_path=html.escape(Path(summary["shoe_artifact"]).as_posix()),
        mount=" ".join(f"{v!r}" for v in mount),
        generated=datetime.date.today().isoformat(),
        peak_ms=1000 * float(grf_t[peak_index]),
        peak_fz=float(grf[peak_index, 1]),
        main_peak=main_run["peak_grf_n"][1],
        main_rmse=main_run["grf_rmse_n"][1],
        main_td=1000 * main_run["touchdown_s"],
        td=contact[0],
        moment_rms=used["residual_moment_rms_nm"],
    )
    path = output or run_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


_CSS = """
:root{--ink:#172b40;--muted:#526174;--line:#dce3eb;--panel:#f5f7fa;--blue:#1261a0}
*{box-sizing:border-box}body{margin:0 auto;max-width:1200px;padding:3rem 2rem;color:var(--ink);background:#fff;font:16px/1.65 system-ui,sans-serif}
a{color:var(--blue)}h1,h2,h3{line-height:1.25;letter-spacing:-.02em}h1{max-width:900px;margin:.6rem 0 1rem;font-size:clamp(1.9rem,3.6vw,2.8rem)}
h2{margin:0 0 1.25rem;font-size:1.7rem}h3{margin:2rem 0 .75rem;font-size:1.15rem}p{margin:.75rem 0}
.eyebrow{color:var(--blue);font-size:.8rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase}
.subtitle{max-width:820px;color:var(--muted);font-size:1.1rem}
nav{display:flex;flex-wrap:wrap;gap:.6rem 1.6rem;padding:1.1rem 0;border-bottom:1px solid var(--line)}nav a{font-size:.9rem;font-weight:650;text-decoration:none}
section{margin-top:3rem}.status{margin:1.5rem 0 .5rem;padding:1.1rem 1.3rem;border:1px solid #edc9b8;border-radius:.65rem;background:#fff8f4}
.status strong{display:block;color:#a3331d;font-size:.85rem;letter-spacing:.015em}.status p{margin:.4rem 0 0;font-size:.95rem}
.note{color:var(--muted);font-size:.9rem}.key{margin:1.25rem 0;padding:1rem 1.25rem;border-left:4px solid var(--blue);background:#f1f6fb;border-radius:.3rem}
.table-scroll{max-width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:.55rem;margin:1rem 0}
table{width:100%;border-collapse:collapse;font-size:.9rem;font-variant-numeric:tabular-nums}caption{padding:.7rem 1rem;text-align:left;color:var(--muted)}
th,td{padding:.7rem 1rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}thead th{background:var(--panel);font-weight:650}
tbody th{font-weight:600}tbody tr:last-child>*{border-bottom:0}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1.25rem;margin:1rem 0}
figure{margin:0;padding:.9rem;border:1px solid var(--line);border-radius:.6rem;min-width:0}
figcaption{margin-bottom:.4rem;font-size:.9rem}figcaption strong{display:block}figcaption span{display:block;color:var(--muted);font-size:.85rem}
svg.plot,svg.scene,svg.model{display:block;width:100%;height:auto}svg.model{max-width:620px;margin:auto}
.workflow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.75rem;list-style:none;padding:0;margin:1.25rem 0;counter-reset:s}
.workflow li{padding:.9rem;border:1px solid var(--line);border-radius:.5rem;counter-increment:s}.workflow strong{display:block;font-size:.95rem}
.workflow strong::before{content:counter(s) " / ";color:var(--blue)}.workflow span{display:block;margin-top:.3rem;color:var(--muted);font-size:.85rem}
.equation{margin:.8rem 0;padding:.85rem 1rem;border-left:3px solid var(--blue);background:#f1f6fb;font:.95rem/1.7 ui-monospace,monospace;overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;color:var(--muted);font-size:.85rem;margin:.5rem 0}
.legend i{display:inline-block;width:22px;height:0;border-top:3px solid;vertical-align:middle;margin-right:.35rem}
pre{max-width:100%;padding:1rem;overflow-x:auto;border-radius:.4rem;background:var(--panel);font-size:.85rem}
ol.issues li{margin:.6rem 0}footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}
@media(max-width:800px){body{padding:1.5rem 1rem}.grid{grid-template-columns:1fr}.workflow{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media print{body{max-width:none;padding:0;font-size:11pt}nav{display:none}figure{break-inside:avoid}}
"""

_PAGE = (
    """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hogan impedance — {trial}</title><style>"""
    + _CSS.replace("{", "{{").replace("}", "}}")
    + """</style></head><body>
<header>
<p class="eyebrow">Impedance Instron / Hogan controller · step 1 of 4</p>
<h1>{trial}: inverse-dynamics feedforward plus fixed impedance, running in a simulated shoe</h1>
<p class="subtitle">One measured {side} stance ({mass:.0f} kg runner). This report separates what is measured, what inverse dynamics computes from the measurements, and what the forward simulation produces with the shoe model in the loop.</p>
<div class="status"><strong>FORWARD SIMULATION · FIXED GAINS · ONE STANCE · NOT VALIDATED</strong>
<p>Every "simulated" force, position, and angle below comes from integrating the equations of motion with the shoe in the loop. Measured ground force enters <em>only</em> through the inverse-dynamics feedforward torque. Gains are fixed (no learning yet) and the rest-of-body inertia is a provisional estimate.</p></div>
<nav><a href="#pipeline">01 Pipeline</a><a href="#model">02 Model</a><a href="#reference">03 Reference &amp; inverse dynamics</a><a href="#controller">04 Controller</a><a href="#results">05 Results</a><a href="#issues">06 Open issues</a><a href="#reproduce">07 Reproduce</a></nav>
</header>
<main>
<section id="pipeline"><h2>1. Pipeline: inverse dynamics first, then a forward rollout</h2>
<ol class="workflow">
<li><strong>Measure</strong><span>Markers at 200 Hz, force plate at 1000 Hz, static trial</span></li>
<li><strong>Register</strong><span>Seat the static ankle on the shoe; shoe pitch from sole markers</span></li>
<li><strong>Reference</strong><span>Filter, COM-consistent hip, leg IK onto the ankle</span></li>
<li><strong>Inverse dynamics</strong><span>Joint torques that reproduce the reference under the measured GRF</span></li>
<li><strong>Forward rollout</strong><span>Integrate the chain with the shoe; feedforward + impedance feedback</span></li>
</ol>
<p>The two dynamics steps use the same chain model but answer different questions:</p>
<div class="equation">Inverse dynamics (reference, measured force):<br>τ<sub>ff</sub>(t) = M(q<sub>ref</sub>) q̈<sub>ref</sub> + h(q<sub>ref</sub>, q̇<sub>ref</sub>) &minus; J(q<sub>ref</sub>)ᵀ F<sub>measured</sub>(t)</div>
<div class="equation">Forward rollout (state evolves, shoe force simulated):<br>M(q) q̈ + h(q, q̇) = τ<sub>ff</sub>(φ) + K (q<sub>ref</sub>(φ) &minus; q) + D (q̇<sub>ref</sub>(φ) &minus; q̇) + J(q)ᵀ F<sub>shoe</sub>(q, q̇, shoe history)</div>
<div class="key"><b>What this means:</b> if the simulated shoe produced exactly the measured force along the reference motion, then q = q<sub>ref</sub> would solve the rollout with zero feedback. Any gap between simulated and measured GRF therefore shows where the shoe model, or the rigid foot, differs from the real foot&ndash;shoe&ndash;ground interaction. The impedance feedback is the correction the controller applies for that gap.</div>
{workflow_rows}
</section>

<section id="model"><h2>2. Model</h2>
<div class="grid"><figure><figcaption><strong>Planar pelvis&ndash;leg chain at peak vertical force ({main_label})</strong><span>Six coordinates: hip x and z, pelvis angle, hip, knee, ankle. The rest of the body is lumped into the pelvis.</span></figcaption>{model_figure}</figure>
<div>{body_table}
<h3>Shoe</h3>
<p>Digital shoe <code>{shoe_id}</code>: {columns} midsole columns under a rigid last. The column bed supplies all ground force (normal foam response plus elastic-Coulomb friction). The shoe is rigidly mounted at the ankle; its pitch at rest is the measured static foot pitch ({static_pitch:.1f}°), so 0° shoe pitch means the static, level sole.</p>
<h3>Static height registration</h3>
<p>The marker ankle center and the shoe mount are placed independently, so the motion is shifted vertically to seat them together. In the static trial the ankle sits {static_ankle:.1f} mm above the ground. The unloaded shoe holds the ankle {unloaded:.1f} mm up, and {static_load:.0f} N (half body weight) compresses it {static_compression:.1f} mm, so the seated ankle sits at {seat:.1f} mm. The measured motion is raised by <b>{offset:+.1f} mm</b>. The offset stays fixed when the shoe changes.</p>
</div></div>
</section>

<section id="reference"><h2>3. Reference motion and inverse dynamics</h2>
<p>Joint angles are low-passed at {cutoff:g} Hz before differentiation. Measured GRF is not filtered. Starting from the raw marker model, three changes make the measured motion consistent with the measured force and a single lumped body:</p>
<ol>
<li><b>Hip from the measured force.</b> The measured GRF divided by body mass, minus gravity, is integrated twice to get the whole-body COM path. Initial position and velocity are fitted to the markers. The hip is then placed so the chain COM lies on that path. This removes the pelvis residual force. Shift RMS {hip_rms} mm (peak {hip_peak} mm), forward / up.</li>
<li><b>Pelvis tilt low-passed at {pelvis_cutoff:g} Hz.</b> The lumped body stands in for the trunk, which does not follow the fast pelvis tilt at impact. Rotating the whole lumped mass with the measured pelvis needs a large residual moment.</li>
<li><b>Leg IK onto the measured ankle center.</b> Static segment lengths combined with Visual3D's 3D knee angle put the ankle centimetres away from the measured malleolus center in stance. Hip and knee are re-solved so the ankle lands on the measured center; the foot's absolute angle is kept.</li>
</ol>
{step_table}
<div class="grid">{id_plots}</div>
</section>

<section id="controller"><h2>4. Controller (fixed gains)</h2>
<div class="equation">τ = τ<sub>ff</sub>(φ) + K (q<sub>ref</sub>(φ) &minus; q) + D (q̇<sub>ref</sub>(φ) &minus; q̇)</div>
<p>All six coordinates are actuated. The hip, knee, and ankle rows are joint torques. The pelvis rows (hip x, hip z, pelvis angle) are <em>external</em> residual loads on the lumped body: the inverse-dynamics residual plus feedback, standing in for the missing upper body and swing leg. Gains are constant across stance. D is set to critical damping for each coordinate's diagonal inertia at t = 0. The phase φ selects where in the reference the controller is:</p>
<ul><li><b>time</b>: φ = t. The reference plays at the measured clock.</li>
<li><b>touchdown</b>: φ is shifted at simulated touchdown so it matches measured touchdown. A shoe that changes contact timing then re-phases the reference; any hip-position jump from that shift is taken up by the stiff hip feedback.</li></ul>
{gain_table}
<p class="note">The next steps learn K(φ) and D(φ) across stances in place of these constants.</p>
</section>

<section id="results"><h2>5. Results</h2>
<p>Measured peak Fz {peak_fz:.0f} N at {peak_ms:.0f} ms. {main_label}: peak {main_peak:.0f} N, Fz RMSE {main_rmse:.0f} N, touchdown {main_td:.1f} ms vs {td:.1f} ms measured.</p>
{result_table}
<h3>5.1 Stance snapshots ({main_label})</h3>
<div class="legend"><span><i style="border-color:#1261a0"></i>Simulated leg, shoe (green midsole, gold last), lumped body</span><span><i style="border-color:#8a97a6;border-top-style:dashed"></i>Reference leg and undeformed sole</span><span><i style="border-color:#7a5230"></i>Measured GRF at measured COP</span><span><i style="border-color:#d94801"></i>Simulated shoe GRF at simulated COP</span></div>
<figure>{snapshots}<figcaption><span>Top: whole chain. Bottom: the foot zoomed in. The sole outline is drawn undeformed, so the part below the ground line is the midsole compression.</span></figcaption></figure>
<h3>5.2 Ground force</h3>
<p class="note">Shaded band: measured contact (Fz &gt; 50 N).</p>
<div class="grid">{grf_plots}{cop_plot}{compression_plot}</div>
<h3>5.3 Motion tracking</h3>
<div class="grid">{tracking_plots}</div>
<h3>5.4 How hard the controller works</h3>
<p>Feedback that is small relative to feedforward means the inverse-dynamics torques, applied with the simulated shoe, already nearly reproduce the measured motion. Feedback that stays one-signed through stance shows a systematic disagreement between the feedforward and the simulated contact. Example: τ<sub>ff</sub> is computed with the measured COP, so where the simulated COP sits closer to the ankle, the feedforward plantarflexor torque over-rotates the shoe toe-down and the ankle feedback pushes back. In early and late stance part of this comes from the treadmill force plate's COP error rather than the shoe model.</p>
{feedback_table}
<div class="grid">{torque_plots}</div>
</section>

<section id="issues"><h2>6. Open issues</h2>
<ol class="issues">
<li><b>Touchdown is late.</b> The static registration seats the shoe so that geometric contact comes after the force-plate touchdown. This is accepted as expected for a static, half-body-weight registration of a soft model.</li>
<li><b>Early/late-stance COP.</b> The treadmill force plate's COP is inaccurate at heel strike and push-off, so the measured-vs-simulated COP gap there is accepted. Because τ<sub>ff</sub> uses that COP, the ankle feedforward inherits the error and the ankle feedback absorbs it; this also drives the late-stance shoe-pitch error.</li>
<li><b>Heel-to-forefoot transition.</b> The simulated force overshoots after heel loading and then dips before the forefoot takes load, while the measured force rises smoothly.</li>
<li><b>Residual pelvis moment.</b> {moment_rms:.0f} N·m RMS remains after the pelvis low-pass. It comes from arm and swing-leg angular momentum, which the lumped body cannot represent, and is applied as feedforward.</li>
<li><b>Provisional rest-of-body inertia.</b> The lumped COM height and radius of gyration are rough adult estimates, not subject measurements.</li>
</ol>
<p><b>Next:</b> learn K(&phi;), D(&phi;) over 100 stances (10 held out), then predict across shoes.</p>
</section>

<section id="reproduce"><h2>7. Reproduce</h2>
<pre><code>python -m projects.impedance_instron.hogan --reference {reference_path} --profile {profile_path} \\
    --shoe-artifact {shoe_path} --mount {mount} \\
    --contact measured --phase time --output {main_dir}
python -m projects.impedance_instron.hogan.report --run {main_dir}</code></pre>
</section>
</main>
<footer>Generated {generated} from <code>{main_dir}</code>. Reference <code>{reference_path}</code>.</footer>
</body></html>
"""
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="Hogan run directory")
    parser.add_argument("--compare", type=Path, nargs="*", default=[], help="Other runs on the same reference")
    parser.add_argument("--baseline", type=Path, help="Earlier run shown as 'before'")
    parser.add_argument(
        "--mount", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Shoe mount [m] for older summaries"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    path = write_report(args.run, compare=args.compare, baseline=args.baseline, mount_m=args.mount, output=args.output)
    print(path)


if __name__ == "__main__":
    main()
