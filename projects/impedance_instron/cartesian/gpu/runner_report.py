# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the shared human identification and frozen material experiments."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .backprop_report import _aggregate_table, _document, _per_trial_table, _svg_lines
from .backprop_report import _fmt as _finite_fmt


def _fmt(value, digits=3):
    """Display unavailable experiment summaries without inventing numerical scores."""
    return _finite_fmt(float("nan") if value is None else value, digits)


def material_chart(evaluation, field, channel, label, unit):
    """Draw linear-axis material response curves on the same initial condition."""
    colors = ("#2878b5", "#243b53", "#cf6d36")
    series = []
    for scale, color in zip((0.8, 1.0, 1.2), colors, strict=True):
        with np.load(evaluation / f"example_{scale:.1f}.npz", allow_pickle=False) as archive:
            x = archive["time_s"][::16]
            y = archive[field][::16, channel]
        series.append((scale, x, y, color))
    ymax = max(float(y.max()) for _, _, y, _ in series)
    ymin = min(float(y.min()) for _, _, y, _ in series)
    duration = max(float(x[-1]) for _, x, _, _ in series)
    yspan = max(ymax - ymin, 1e-12)
    lines = []
    for _scale, x, y, color in series:
        coordinates = " ".join(
            f"{65 + t / duration * 660:.2f},{245 - (value - ymin) / yspan * 215:.2f}"
            for t, value in zip(x, y, strict=True)
        )
        lines.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.2"/>')
    ticks = []
    for fraction in (0, 0.5, 1):
        y = 245 - fraction * 215
        ticks.append(
            f'<line x1="65" x2="725" y1="{y}" y2="{y}" class="grid"/><text x="57" y="{y + 4}" text-anchor="end" class="axis">{_fmt(ymin + fraction * yspan, 2)}</text>'
        )
    legend = " · ".join(
        f'<span style="color:{color}">{scale:.1f}&times; modulus</span>' for scale, _, _, color in series
    )
    return (
        f'<h3>{html.escape(label)} ({unit})</h3><svg viewBox="0 0 760 295" role="img" aria-label="{html.escape(label)}">'
        + "".join(ticks + lines)
        + f'<text x="395" y="286" text-anchor="middle" class="axis">Time (s), same initial conditions and human curve</text></svg><p>{legend}</p>'
    )


def render(run, evaluation=None):
    """Combine fit distributions, full scoring coverage and causal verification."""
    data = json.loads((run / "report.json").read_text())
    final_train, final_eval = data["final_training"], data["final_evaluation"]
    initial = data["initial_training"]["mean_loss"]
    curves = [(r["iteration"], r["loss"]) for r in data["history"] if r["loss"] is not None]
    history = _svg_lines([("All 100 training mean", curves, "#2878b5")])
    metrics = data["metrics"]
    coverage = np.asarray([r["scoring_time_coverage"] for r in metrics])
    distributions = []
    labels = ("Hip x (mm)", "Hip z (mm)", "Joint1 (mrad)", "Joint2 (mrad)", "GRF x (N)", "GRF z (N)")
    for split in ("train", "eval"):
        rows = [r for r in metrics if r["split"] == split and r["complete"]]
        values = np.asarray([r["nativechannelrmse"] for r in rows]) * np.asarray([1000] * 4 + [1] * 2)
        for i, label in enumerate(labels):
            statistics = (
                (np.mean(values[:, i]), np.median(values[:, i]), np.percentile(values[:, i], 90), np.max(values[:, i]))
                if len(values)
                else [float("nan")] * 4
            )
            distributions.append(
                f"<tr><th>{split} / {label}</th>" + "".join(f"<td>{_fmt(x, 2)}</td>" for x in statistics) + "</tr>"
            )
    body = (
        "<section><h2>The learned human model</h2><p>One shared 12&times;6 equilibrium spline, one stored phase period, frozen body geometry and impedance gains. Initial horizontal position translates the curve; initial pose and velocity change the simulated response. No stance ID, desired GRF, future motion, per-stance nominal curve or measured duration enters controller inference.</p>"
        + f"<p>Common period: {_fmt(data['duration_s'], 6)} s. All 100 training stances contribute to the same 72 coefficients. No previously fitted controller was loaded. Checkpoints use complete training loss only; evaluation does not select parameters.</p></section>"
        + "<section><h2>From-scratch fit</h2>"
        + history
        + f"<p>{data['epochs']} passes, {data['iterations']} updates, {data['accepted_updates']} accepted. Selected update: {data['selected_iteration']}. Search: {_fmt(data['search_wall_s'] / 60, 2)} min; total fit: {_fmt(data['total_wall_s'] / 60, 2)} min.</p></section>"
        + "<section><h2>Scoring support</h2><p>The recorded endpoint defines available observations, never controller phase. Motion targets are interpolated only at the common support endpoint; GRF is scored on its native grid. No time normalization to the recorded stride is applied.</p>"
        + f"<p>Scoring coverage is {_fmt(coverage.min() * 100, 2)}-{_fmt(coverage.max() * 100, 2)}% of each recorded window (mean {_fmt(coverage.mean() * 100, 2)}%). Samples beyond the shared one-cycle support are excluded and are not claimed as fitted. Rollout completion and all-six flags refer to this reported scoring support.</p>"
        + "<p>Six-channel limits: 20mm hip axes, 50mrad joint axes, 100N GRF axes. A population controller is not expected to reconstruct every observed stance exactly; per-stance errors and tail statistics remain visible.</p></section>"
        + "<section><h2>Error distribution</h2><table><thead><tr><th>Split / channel</th><th>Mean</th><th>Median</th><th>90th percentile</th><th>Maximum</th></tr></thead><tbody>"
        + "".join(distributions)
        + "</tbody></table></section>"
        + "<section><h2>Trial summaries</h2>"
        + _per_trial_table(metrics)
        + "</section>"
        + "<details><summary>All 110 stance measurements</summary>"
        + _aggregate_table("train", metrics[:100])
        + _aggregate_table("eval", metrics[100:])
        + "</details>"
    )
    if evaluation:
        check = json.loads((evaluation / "report.json").read_text())
        body += (
            "<section><h2>Frozen-controller verification</h2>"
            + f"<p>Independent native production rollouts complete {check['native_complete_count']}/110; maximum batch loss difference {_fmt(check['maximum_native_loss_difference'], 12)}. Target-free held-out physics parity passes: {str(check['target_free_eval_parity_passed']).lower()}.</p>"
            + f"<p>Target-free full-period rollouts complete {check['full_period_target_free_complete_count']}/110 initial conditions, including continuation past shorter measurement windows.</p>"
            + f"<p>Half-timestep held-out completion: {check['fine_eval_complete_count']}/10; all-six passes: {check['fine_eval_all6_pass_count']}/10; mean loss: {_fmt(check['fine_eval_mean_loss'], 5)}. Training-wide half-timestep qualification was not performed.</p></section>"
        )
        material_rows = []
        for scale in (0.8, 1, 1.2):
            selected = [r for r in check["material_rows"] if r["modulus_scale"] == scale]
            valid = [r for r in selected if r["complete"]]

            def mean(field, index=None, records=tuple(valid)):
                return (
                    _fmt(np.mean([r[field] if index is None else r[field][index] for r in records]), 2)
                    if records
                    else "—"
                )

            material_rows.append(
                f"<tr><th>{scale:.1f}&times;</th><td>{len(valid)}/{len(selected)}</td><td>{mean('peak_grf_n', 1)}</td><td>{mean('grf_impulse_ns', 1)}</td><td>{mean('contact_duration_s')}</td><td>{mean('positive_actuator_work_j')}</td></tr>"
            )
        body += (
            "<section><h2>Material response with the same human controller</h2><p>Synthetic 0.8&times;/1.0&times;/1.2&times; shear-modulus sensitivity, unchanged shoe geometry, friction and relaxation. These are parameter perturbations, not calibrated alternative materials. Each rollout starts with reset shoe history and the same held-out initial conditions; it runs the full stored period.</p>"
            + f"<p>Human coefficients are identical across materials: {str(check['controller_same_across_materials']).lower()}. Completion: {check['material_complete_count']}/{check['material_rollout_count']}. Geometry remains fixed for all stances.</p>"
            + "<table><thead><tr><th>Modulus</th><th>Complete</th><th>Mean peak Fz (N)</th><th>Mean vertical impulse (N·s)</th><th>Mean contact (s)</th><th>Positive actuator work (J)</th></tr></thead><tbody>"
            + "".join(material_rows)
            + "</tbody></table>"
            + "<p>Material summary means include completed rollouts only; completion counts include every attempted rollout.</p>"
            + f"<p>Mean warmed forward physics time per rollout: {_fmt(np.mean([r['rollout_wall_s'] for r in check['material_rows']]), 3)} s, excluding setup, capture and trace reporting. The material exemplar completes at half timestep for all three variants; maximum peak vertical GRF difference: {_fmt(max(abs(r['fine_peak_grf_n'][1] - r['peak_grf_n'][1]) for r in check['material_rows'] if 'fine_complete' in r), 3)} N.</p>"
            + f"<p>Example initial conditions: {html.escape(check['material_exemplar'])}. The plotted material curves use the same human controller and initial state.</p>"
            + material_chart(evaluation, "grf_n", 1, "Vertical ground reaction force", "N")
            + material_chart(evaluation, "state", 1, "Hip height", "m")
            + "</section>"
        )
    body += (
        "<details><summary>Artifacts and interpretation</summary><p>runner_controller.npz contains the shared human curve and training-only clock/geometry, with profile and shoe metadata. runner_rollout accepts only initial pose/velocity and a frozen controller; a shoe artifact or synthetic modulus scale can be supplied independently.</p>"
        + "<p>The equilibrium plan is fixed for the rollout; impedance supplies causal state feedback. Material experiments model its immediate response without refitting. Continuous bilateral gait or adaptation to a new material has not been established. Contact transitions remain nonsmooth; derivative evidence applies to the recorded qualification probes.</p>"
        + "<p>No unit tests were added or run. Full-rollout experiments, independent production parity, numerical directional checks, and material comparisons provide the evidence.</p></details>"
    )
    return _document(
        title="One learned runner ·100 stances",
        subtitle="Initial conditions → shared equilibrium trajectory → leg/shoe physics; material varies while the human remains fixed",
        summary=[
            ("Human coefficients", "72 shared"),
            ("Initial training loss", _fmt(initial, 4)),
            ("Training mean loss", _fmt(final_train["mean_loss"], 4)),
            (
                "Held-out mean loss",
                _fmt(final_eval["mean_loss"] if final_eval["mean_loss"] is not None else float("nan"), 4),
            ),
            ("Train all-six", f"{final_train['native_all6_pass_count']}/100"),
            ("Held-out all-six", f"{final_eval['native_all6_pass_count']}/10"),
        ],
        body=body,
    )


def main():
    """Build the standalone shared-runner experiment HTML report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.run, args.evaluation))


if __name__ == "__main__":
    main()
