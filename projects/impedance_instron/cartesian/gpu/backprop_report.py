# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build self-contained HTML reports for physics-gradient fitting experiments."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


def _fmt(value: float, digits: int = 3) -> str:
    """Format a finite value for human-readable report text."""
    if not math.isfinite(float(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def _svg_lines(series: list[tuple[str, list[tuple[float, float]], str]], *, width: int = 760, height: int = 300) -> str:
    """Draw labeled line series as inline SVG with a logarithmic y axis."""
    points = [point for _, values, _ in series for point in values if point[1] > 0]
    if not points:
        return '<p class="muted">No history was recorded.</p>'
    xmin = min(x for x, _ in points)
    xmax = max(x for x, _ in points)
    ymin = min(y for _, y in points)
    ymax = max(y for _, y in points)
    lo, hi = math.log10(ymin), math.log10(ymax)
    if hi - lo < 1e-12:
        hi = lo + 1.0
    left, right, top, bottom = 64, width - 24, 22, height - 48

    def xy(x: float, y: float) -> tuple[float, float]:
        px = left + (x - xmin) / max(xmax - xmin, 1e-12) * (right - left)
        py = bottom - (math.log10(y) - lo) / (hi - lo) * (bottom - top)
        return px, py

    labels = []
    paths = []
    for name, values, color in series:
        valid = [(x, y) for x, y in values if y > 0 and math.isfinite(y)]
        if not valid:
            continue
        coordinates = " ".join(f"{xy(x, y)[0]:.1f},{xy(x, y)[1]:.1f}" for x, y in valid)
        paths.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        labels.append(f'<span class="legend"><i style="background:{color}"></i>{html.escape(name)}</span>')
    y_ticks = []
    for fraction in (0.0, 0.5, 1.0):
        value = 10 ** (lo + fraction * (hi - lo))
        y = bottom - fraction * (bottom - top)
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/><text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{_fmt(value, 2)}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Loss history">'
        + "".join(y_ticks)
        + f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axisline"/>'
        + f'<text x="{(left + right) / 2}" y="{height - 8}" text-anchor="middle" class="axis">Update</text>'
        + f'<text x="16" y="{(top + bottom) / 2}" transform="rotate(-90 16 {(top + bottom) / 2})" text-anchor="middle" class="axis">Objective loss (log scale)</text>'
        + "".join(paths)
        + '</svg><div class="legendrow">'
        + "".join(labels)
        + "</div>"
    )


def _metric_table(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Render old and new six-channel measured RMSE values."""
    old_values = old["metrics"]
    new_values = new["rmse"][0] if new["rmse"] and isinstance(new["rmse"][0], list) else new["rmse"]
    labels = [
        ("Hip x", "mm", 1000.0),
        ("Hip z", "mm", 1000.0),
        ("Joint 1", "mrad", 1000.0),
        ("Joint 2", "mrad", 1000.0),
        ("GRF x", "N", 1.0),
        ("GRF z", "N", 1.0),
    ]
    rows = []
    for index, (label, unit, scale) in enumerate(labels):
        previous = old_values[index] * scale
        current = new_values[index] * scale
        delta = (current / previous - 1.0) * 100 if previous else float("nan")
        rows.append(
            f"<tr><th>{label}</th><td>{_fmt(previous, 2)} {unit}</td><td>{_fmt(current, 2)} {unit}</td><td>{_fmt(delta, 1)}%</td></tr>"
        )
    return (
        "<table><thead><tr><th>Measured channel RMSE</th><th>Saved fit</th><th>Gradient fit</th><th>Change</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_single_stance(
    gradient_report: dict[str, Any],
    qualification: dict[str, Any],
    saved_fit: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    """Render results for one fully qualified stance."""
    progress = gradient_report.get("progress", [])
    grad_history = [(float(row["iteration"]), float(row["loss"])) for row in progress if row.get("loss", 0) > 0]
    old_history = [
        (float(row["iteration"]), float(row["best_loss"]))
        for row in saved_fit.get("history", [])
        if row.get("best_loss", 0) > 0
    ]
    # Both histories are plotted against their own update counts; marker styling preserves method identity.
    chart = _svg_lines(
        [
            ("Gradient L-BFGS", grad_history, "#2878b5"),
            ("Resident search", old_history, "#cf6d36"),
        ]
    )
    grad_score = qualification["native_scores"]
    old_metrics = [
        *saved_fit["metrics"]["hip_rmse_m"],
        *saved_fit["metrics"]["joint_rmse_rad"],
        *saved_fit["metrics"]["force_rmse_n"],
    ]
    metrics_html = _metric_table({"metrics": old_metrics}, grad_score)
    gradient_s = float(gradient_report["search_wall_s"])
    old_s = float(saved_fit["timers"].get("search_wall_s", saved_fit.get("wall_s", 0)))
    speedup = old_s / gradient_s if gradient_s > 0 else float("nan")
    comparison_data = comparison.get("gradient", {})
    provenance = {
        "Stance baseline": gradient_report.get("baseline"),
        "Gradient audit": gradient_report.get("gradient_audit"),
        "Native/fine qualification": qualification.get("fit_directory"),
        "Matched five-update comparison": str(
            Path(gradient_report.get("baseline", ""))
            / ".."
            / ".."
            / "physics_backprop_20260929"
            / "comparison_5_unfitted"
            / "comparison.json"
        ),
        "Gradient starting point": gradient_report.get("start"),
        "Gradient fit status": gradient_report.get("status"),
        "Qualification accepted": qualification.get("accepted"),
    }
    provenance_html = "".join(
        f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in provenance.items() if v is not None
    )
    qual_refine = qualification.get("refinement", {})
    pass_text = (
        "Accepted by native and fine measured-tolerance checks."
        if qualification.get("accepted")
        else "Qualification did not accept this controller."
    )
    return _document(
        title="Physics backpropagation · single stance",
        subtitle="FR3_2 saved-fit stance · measured six-channel objective · right-foot shoe contact",
        summary=[
            ("Starting measured loss", _fmt(gradient_report["initial_scores"]["loss"][0], 6)),
            ("Final measured loss", _fmt(grad_score["loss"][0], 6)),
            ("Search time", f"{gradient_s:.1f} s"),
            ("Saved-fit search", f"{old_s:.1f} s"),
            ("Search speed ratio", f"{speedup:.2f}x"),
        ],
        body=(
            "<section><h2>Fit progress</h2><p>Each curve uses its own update count. The plot shows the best accepted loss at each update on a logarithmic axis; it does not imply equal work per update.</p>"
            + chart
            + "</section>"
            "<section><h2>Measured tracking accuracy</h2><p>RMSE values compare the qualified gradient controller against the previously saved fit on the same recorded stance and measured objective. Hip values are in millimeters, joint angles in milliradians, and ground reaction force in newtons.</p>"
            + metrics_html
            + "</section>"
            '<section><h2>Runtime and validation</h2><div class="cards">'
            + f"<article><b>{_fmt(float(gradient_report['setup_wall_s']), 1)} s</b><span>gradient engine setup and capture</span></article>"
            + f"<article><b>{_fmt(float(gradient_report['total_wall_s']), 1)} s</b><span>gradient fit total wall time</span></article>"
            + f"<article><b>{_fmt(float(qual_refine.get('maximum_hip_position_difference_m', 0)) * 1000, 3)} mm</b><span>native vs fine maximum hip difference</span></article>"
            + f"<article><b>{_fmt(float(qual_refine.get('maximum_grf_difference_n', 0)), 2)} N</b><span>native vs fine maximum GRF difference</span></article>"
            + '</div><p class="callout">'
            + html.escape(pass_text)
            + " Qualification result: native pass = "
            + str(qualification.get("native_within_measured_tolerances"))
            + "; fine pass = "
            + str(qualification.get("fine_within_measured_tolerances"))
            + ".</p>"
            + "</section>"
            + "<details><summary>Method, comparison scope, and provenance</summary><p>The gradient fit started from the original unfitted controller, used the full stance rollout, and optimized the production measured objective with normalized L-BFGS, spline-bound projection, and production Armijo backtracking. The old fit used the resident population search. The search-time ratio is specific to this stance and these runs.</p>"
            + f"<p>The separate matched five-update comparison recorded gradient loss {_fmt(float(comparison_data.get('final_loss', float('nan'))), 4)} after {_fmt(float(comparison_data.get('search_wall_s', 0)), 2)} s; resident loss {_fmt(float(comparison.get('resident', {}).get('loss', float('nan'))), 4)}. Five updates are an implementation check, not evidence of matched convergence.</p>"
            + "<dl>"
            + provenance_html
            + "</dl><p>Gradient quality is qualified numerically against the fine timestep rollout. Contact branch changes remain nonsmooth; the measured result applies to this stance and does not establish multi-stance generalization.</p></details>"
        ),
    )


def render_report(data: dict[str, Any]) -> str:
    """Render an aggregated multi-stance report using the documented generic schema.

    Args:
        data: Report values with ``title``, ``train_count``, ``eval_count``,
            ``initial_loss``, ``final_loss``, ``search_wall_s``, optional
            ``setup_wall_s``, ``history``, ``metrics``, and ``provenance``.

    Each metric row should contain ``split``, ``stance_id``, and the six
    measured RMSE channels (``hip_rmse_m``, ``joint_rmse_rad``,
    ``force_rmse_n``). If no metric rows are present, the report says so.
    """
    history = [
        (float(row.get("iteration", i + 1)), float(row["loss"]))
        for i, row in enumerate(data.get("history", []))
        if row.get("loss") is not None and float(row["loss"]) > 0
    ]
    rows = data.get("metrics", [])
    sections = []
    for split in ("train", "eval"):
        current = [row for row in rows if row.get("split") == split]
        if current:
            sections.append(_aggregate_table(split, current))
    if not sections:
        sections.append('<p class="muted">Per-stance RMSE values were not present in the aggregate report.</p>')
    provenance = data.get("provenance", {})
    provenance_html = "".join(
        f"<dt>{html.escape(str(k))}</dt><dd>{html.escape(str(v))}</dd>" for k, v in provenance.items()
    )
    train_loss = data.get(
        "train_mean_loss", data.get("mean_loss", {}).get("train") if isinstance(data.get("mean_loss"), dict) else None
    )
    eval_loss = data.get(
        "eval_mean_loss", data.get("mean_loss", {}).get("eval") if isinstance(data.get("mean_loss"), dict) else None
    )
    counts = _acceptance_counts(rows)
    acceptance = data.get("acceptance", {})
    accepted = acceptance.get("native_all6_pass_count", counts["accepted"])
    eligible = acceptance.get("native_all6_total", counts["eligible"])
    failed_rows = [row for row in rows if row.get("complete") is False or row.get("failure")]
    trial_sections = _per_trial_table(rows)
    comparison = data.get("previous_shared_run")
    comparison_html = ""
    if comparison:
        comparison_html = (
            "<section><h2>Earlier shared-controller experiment</h2><table><thead><tr><th>Method</th><th>Train mean loss</th><th>Eval mean loss</th><th>Train all-six</th><th>Eval all-six</th><th>Search time</th></tr></thead><tbody>"
            + f"<tr><th>Common residual / population search</th><td>{_fmt(comparison['train_mean_loss'], 4)}</td><td>{_fmt(comparison['eval_mean_loss'], 4)}</td><td>{comparison['train_pass_count']}/100</td><td>{comparison['eval_pass_count']}/10</td><td>{_fmt(comparison['search_wall_s'], 1)} s</td></tr>"
            + f"<tr><th>Conditioned network / physics gradients</th><td>{_fmt(train_loss, 4)}</td><td>{_fmt(eval_loss, 4)}</td><td>{data['final_training']['native_all6_pass_count']}/100</td><td>{data['final_evaluation']['native_all6_pass_count']}/10</td><td>{_fmt(data['search_wall_s'], 1)} s</td></tr>"
            + "</tbody></table><p>Both experiments start from unfitted reference controllers on the same dataset. Controller capacity and optimization strategy both changed; this comparison measures the combined approach rather than isolating an optimizer speedup. Neither run establishes time to the best achievable fit.</p></section>"
        )
    independent = data.get("independent_qualification")
    independent_html = ""
    if independent:
        independent_html = (
            "<section><h2>Frozen-controller production check</h2>"
            + f"<p>Separate native production rollouts complete {independent['native_complete_count']}/110 stances. Agreement with batch losses: {str(independent['native_batch_parity_passed']).lower()}; maximum absolute loss difference {_fmt(independent['maximum_native_loss_difference'], 12)}.</p>"
            + f"<p>At half timestep, {independent['fine_eval_complete_count']}/10 held-out rollouts complete and {independent['fine_eval_all6_pass_count']}/10 pass all six thresholds. Held-out mean loss: {_fmt(independent['fine_eval_mean_loss'], 5)}. Training stances were not checked at half timestep. No controller adaptation occurs in this experiment.</p></section>"
        )
    return _document(
        title=str(data.get("title", "Physics backpropagation · multi-stance")),
        subtitle="Shared physics-gradient fitting · held-out stance metrics remain separate from training metrics",
        summary=[
            ("Training stances", str(data.get("train_count", "—"))),
            ("Evaluation stances", str(data.get("eval_count", "—"))),
            ("Initial loss", _fmt(data.get("initial_loss", float("nan")), 4)),
            ("Final loss", _fmt(data.get("final_loss", float("nan")), 4)),
            ("Search time", f"{_fmt(data.get('search_wall_s', float('nan')), 1)} s"),
            ("Native all-6 pass", f"{accepted}/{eligible}" if eligible else "—"),
            ("Failed/incomplete", str(len(failed_rows))),
        ],
        body=(
            "<section><h2>Training progress</h2>"
            + _svg_lines([("Training objective", history, "#2878b5")])
            + f"<p>Mean train loss: {_fmt(train_loss, 5)} · Mean held-out eval loss: {_fmt(eval_loss, 5)}. Evaluation stances were excluded from optimizer updates: {html.escape(str(data.get('eval_excluded_from_updates', True))).lower()}.</p></section>"
            + "<section><h2>Experiment scope</h2><p>FR3_1 and FR3_2 only, with right-foot contacts bounded by preceding and following right-hip height peaks. A reference-conditioned network predicts the 72 spline coefficients. Full-rollout physics derivatives train the network through the spline-bound projection; each proposed update is checked with an actual forward rollout.</p>"
            + f"<p>Completed {data.get('iterations', '—')} updates over {data.get('epochs', '—')} epochs; selected update {data.get('selected_iteration', '—')} using training loss only. Accepted updates: {data.get('accepted_updates', '—')}. "
            + html.escape(str(data.get("qualification", "No timestep-refinement qualification was supplied.")))
            + "</p><p>Contact and bound transitions remain nonsmooth. The gradient qualification covers the recorded probes, rather than establishing exact derivatives at every fitted contact state.</p></section>"
            + "<section><h2>Stance accuracy</h2>"
            + "".join(sections)
            + "</section>"
            + ("<section><h2>Per-trial summary</h2>" + trial_sections + "</section>" if trial_sections else "")
            + comparison_html
            + independent_html
            + "<details><summary>Setup and provenance</summary>"
            + f"<p>Engine setup and capture: {_fmt(data.get('setup_wall_s', float('nan')), 2)} s. "
            + html.escape(str(data.get("method", "Method details were not supplied.")))
            + "</p>"
            + f"<p>Native acceptance requires all six hip, joint, and GRF RMSE channels to pass the recorded limits. Passing {accepted} of {eligible} eligible stances. {len(failed_rows)} rows are incomplete or marked as failures.</p>"
            + "<dl>"
            + provenance_html
            + "</dl></details>"
        ),
    )


def _aggregate_table(split: str, rows: list[dict[str, Any]]) -> str:
    """Render metric summaries plus every stance's recorded values."""
    output = [
        f"<h3>{html.escape(split.title())}: {len(rows)} stances</h3><table><thead><tr><th>Trial / stance</th><th>Hip RMSE initial → final (m, x/z)</th><th>Joint RMSE initial → final (rad, 1/2)</th><th>GRF RMSE initial → final (N, x/z)</th><th>All-six pass</th><th>Status</th></tr></thead><tbody>"
    ]
    for row in rows:
        values = _row_channels(row)
        initial = _row_channels(row, initial=True)
        display = []
        for before, after, units in zip(initial, values, (("m", "m"), ("rad", "rad"), ("N", "N")), strict=True):
            if not after:
                display.append("—")
                continue
            current_text = " / ".join(f"{_fmt(value, 4)} {unit}" for value, unit in zip(after, units, strict=True))
            initial_text = (
                " / ".join(f"{_fmt(value, 4)} {unit}" for value, unit in zip(before, units, strict=True))
                if before
                else "—"
            )
            display.append(f"{initial_text} → {current_text}")
        passed = row.get("native_all6_pass", row.get("all6_pass", "—"))
        if isinstance(passed, bool):
            passed = "Pass" if passed else "Fail"
        status = (
            "Failed"
            if row.get("failure") or row.get("complete") is False
            else ("Complete" if row.get("complete") is True else "Recorded")
        )
        output.append(
            f"<tr><th>{html.escape(str(row.get('trial', '')))} / {html.escape(str(row.get('stance_id', 'unknown')))}</th><td>{display[0]}</td><td>{display[1]}</td><td>{display[2]}</td><td>{html.escape(str(passed))}</td><td>{html.escape(status)}</td></tr>"
        )
    output.append("</tbody></table>")
    output.append(
        '<p class="muted">Within each stance, x/z and joint channels are reported separately; no pooled sum hides a weak channel.</p>'
    )
    return "".join(output)


def _row_channels(row: dict[str, Any], *, initial: bool = False) -> tuple[list[float], list[float], list[float]]:
    """Extract the six measured channels from either explicit or flat schema."""
    prefix = "initial_" if initial else ""
    values = row.get(f"{prefix}nativechannelrmse", row.get(f"{prefix}native_channel_rmse", row.get(f"{prefix}rmse")))
    if values is not None and len(values) >= 6:
        return list(values[:2]), list(values[2:4]), list(values[4:6])
    return (
        list(row.get(f"{prefix}hip_rmse_m") or []),
        list(row.get(f"{prefix}joint_rmse_rad") or []),
        list(row.get(f"{prefix}force_rmse_n") or []),
    )


def _acceptance_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count explicit all-channel pass flags for complete stance rows."""
    eligible = [row for row in rows if row.get("complete", True) is not False and not row.get("failure")]
    accepted = sum(bool(row.get("native_all6_pass", row.get("all6_pass", False))) for row in eligible)
    return {"accepted": accepted, "eligible": len(eligible)}


def _per_trial_table(rows: list[dict[str, Any]]) -> str:
    """Show mean measured loss and all-six acceptance by trial when supplied."""
    trials = sorted({str(row["trial"]) for row in rows if row.get("trial") is not None})
    if not trials:
        return ""
    result = [
        "<table><thead><tr><th>Trial / split</th><th>Stances</th><th>Mean loss</th><th>Native all-six pass</th></tr></thead><tbody>"
    ]
    for trial, split in ((trial, split) for trial in trials for split in ("train", "eval")):
        selected = [row for row in rows if str(row.get("trial")) == trial and row.get("split") == split]
        losses = [
            float(row["loss"]) for row in selected if row.get("loss") is not None and math.isfinite(float(row["loss"]))
        ]
        counts = _acceptance_counts(selected)
        result.append(
            f"<tr><th>{html.escape(trial)} / {split}</th><td>{len(selected)}</td><td>{_fmt(sum(losses) / len(losses), 5) if losses else '—'}</td><td>{counts['accepted']}/{counts['eligible']}</td></tr>"
        )
    result.append("</tbody></table>")
    return "".join(result)


def _document(*, title: str, subtitle: str, summary: list[tuple[str, str]], body: str) -> str:
    """Wrap report sections in a standalone responsive HTML document."""
    cards = "".join(
        f"<article><b>{html.escape(value)}</b><span>{html.escape(label)}</span></article>" for label, value in summary
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{color-scheme:light;--ink:#172433;--muted:#607082;--line:#dbe3eb;--paper:#fff;--back:#f3f6f9;--blue:#2878b5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--back);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1120px;margin:auto;padding:36px 24px 64px}}header{{padding:12px 0 24px}}h1{{font-size:clamp(28px,4vw,42px);line-height:1.1;margin:0 0 10px}}header p,.muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:18px 0 26px}}article,section,details{{background:var(--paper);border:1px solid var(--line);border-radius:12px}}article{{padding:16px;display:flex;flex-direction:column;gap:5px}}article b{{font-size:22px}}article span{{font-size:13px;color:var(--muted)}}section{{padding:22px;margin:16px 0}}h2{{margin:0 0 8px;font-size:22px}}h3{{margin:18px 0 8px}}p{{margin:8px 0 14px}}table{{width:100%;border-collapse:collapse;margin:14px 0 8px;font-variant-numeric:tabular-nums}}th,td{{text-align:right;padding:10px 9px;border-bottom:1px solid var(--line)}}th:first-child,td:first-child{{text-align:left}}thead{{background:#f7f9fb}}svg{{display:block;width:100%;height:auto;margin:14px 0}}.grid{{stroke:#e3e8ee;stroke-width:1}}.axisline{{stroke:#8190a0}}.axis{{fill:#607082;font-size:12px}}.legendrow{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:13px}}.legend{{display:inline-flex;align-items:center;gap:7px}}.legend i{{width:12px;height:3px;border-radius:2px}}details{{padding:17px 22px;margin:16px 0}}summary{{cursor:pointer;font-weight:650}}details p{{margin-top:12px}}dl{{display:grid;grid-template-columns:minmax(150px,240px) 1fr;gap:5px 14px;overflow-wrap:anywhere}}dt{{font-weight:600}}dd{{margin:0;color:var(--muted)}}.callout{{padding:12px 15px;background:#eef7f1;border-left:4px solid #33945b;border-radius:4px}}@media(max-width:650px){{main{{padding:22px 14px}}section{{padding:15px;overflow-x:auto}}table{{min-width:580px}}dl{{grid-template-columns:1fr}}dd{{margin-bottom:8px}}}}
</style></head><body><main><header><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></header><div class="cards">{cards}</div>{body}<footer class="muted">Generated from recorded experiment artifacts. All charts are embedded SVG; this report needs no network access.</footer></main></body></html>"""


def main() -> None:
    """Generate the recorded single-stance report or a generic aggregate report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-report", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--saved-fit", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--aggregate", type=Path, help="JSON using render_report's multi-stance schema")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.aggregate:
        document = render_report(json.loads(args.aggregate.read_text()))
    elif all((args.gradient_report, args.qualification, args.saved_fit, args.comparison)):
        values = [
            json.loads(path.read_text())
            for path in (args.gradient_report, args.qualification, args.saved_fit, args.comparison)
        ]
        document = render_single_stance(*values)
    else:
        parser.error("provide --aggregate or all four single-stance input paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)


if __name__ == "__main__":
    main()
