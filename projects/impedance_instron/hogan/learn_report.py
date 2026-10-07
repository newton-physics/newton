# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Write an HTML report for a schedule learned by :mod:`.learn`.

Example::

    uv run python -m projects.impedance_instron.hogan.learn_report --run runs/hogan_learn
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
from pathlib import Path

import numpy as np

from ..cartesian import data as reference_data
from ..cartesian import profile as profile_data
from ..cartesian.shoe import Shoe
from .batch import reference_timing, schedule_impedance
from .learn import _load_plan
from .mechanics import RestOfBody, chain_from_profile
from .report import _CSS, BASELINE, COORDINATE_INFO, MEASURED, ROTATIONAL, RUN_COLORS, _figure, _plot, _table
from .rollout import Config, simulate

LEARNED_COLOR = RUN_COLORS[0]


def _pick_learned(summary: dict) -> str:
    """Choose the learned schedule by training loss only, never by held-out loss."""
    train = summary["splits"]["train"]
    return min(("best", "final_mean"), key=lambda label: train[label]["aggregate"].get("mean_loss", np.inf))


def _fmt(value, digits=1, scale=1.0) -> str:
    return "&ndash;" if value is None or not np.isfinite(value) else f"{value * scale:.{digits}f}"


def _aggregate_table(summary: dict, learned: str) -> str:
    header = [
        "Split / schedule",
        "Completed",
        "Loss",
        "Fz RMSE [N]",
        "Fx RMSE [N]",
        "|Peak Fz error| [N]",
        "Hip z RMSE [mm]",
        "Pelvis [mrad]",
        "Hip [mrad]",
        "Knee [mrad]",
        "Ankle [mrad]",
    ]
    rows = []
    for split, results in summary["splits"].items():
        for label, name in (("baseline", "baseline"), (learned, "learned")):
            agg = results[label]["aggregate"]
            rmse = agg.get("tracking_rmse", {})
            grf = agg.get("grf_rmse_n", [np.nan, np.nan])
            rows.append(
                [
                    f"{split} / {name}",
                    f"{agg['completed']}/{agg['stances']}",
                    _fmt(agg.get("mean_loss", np.nan), 2),
                    _fmt(grf[1], 0),
                    _fmt(grf[0], 0),
                    _fmt(agg.get("peak_fz_error_n", np.nan), 0),
                    _fmt(rmse.get("hip_z", np.nan), 1, 1e3),
                    *(_fmt(rmse.get(name_, np.nan), 0, 1e3) for name_ in ROTATIONAL),
                ]
            )
    return _table(header, rows, "Means over stances. Loss is the search objective (lower is better).")


def _eval_table(summary: dict, learned: str) -> str:
    split = summary["splits"].get("eval")
    if split is None:
        return ""
    rows = []
    for base, new in zip(split["baseline"]["stances"], split[learned]["stances"], strict=True):
        rows.append(
            [
                html.escape(base["id"]),
                f"{base['loss']:.2f} &rarr; {new['loss']:.2f}",
                f"{base['grf_rmse_n'][1]:.0f} &rarr; {new['grf_rmse_n'][1]:.0f}",
                f"{base['reference_peak_fz_n']:.0f}",
                f"{base['peak_grf_n'][1]:.0f} &rarr; {new['peak_grf_n'][1]:.0f}",
                f"{1e3 * base['reference_contact_duration_s']:.0f}",
                f"{1e3 * base['contact_duration_s']:.0f} &rarr; {1e3 * new['contact_duration_s']:.0f}",
                f"{1e3 * base['tracking_rmse']['ankle']:.0f} &rarr; {1e3 * new['tracking_rmse']['ankle']:.0f}",
                new["status"],
            ]
        )
    return _table(
        [
            "Held-out stance",
            "Loss",
            "Fz RMSE [N]",
            "Measured peak Fz [N]",
            "Simulated peak Fz [N]",
            "Measured contact [ms]",
            "Simulated contact [ms]",
            "Ankle RMSE [mrad]",
            "Status",
        ],
        rows,
        "Baseline &rarr; learned. None of these stances was used in the search.",
    )


def _gain_plots(schedule, labels: list[str], learned: str) -> str:
    knots = schedule["knots_phase"]
    phi = np.linspace(knots[0], knots[-1], 301)
    figures = []
    index = {label: i for i, label in enumerate(labels)}
    for name in ROTATIONAL:
        channel = 2 + ROTATIONAL.index(name)
        _, _, k_unit, d_unit = COORDINATE_INFO[name]
        for table, unit, title in (("stiffness", k_unit, "K"), ("damping", d_unit, "D")):
            values = schedule[table]
            series = [
                ("baseline", phi, np.interp(phi, knots, values[index["baseline"], :, channel]), BASELINE, "6 4"),
                ("learned", phi, np.interp(phi, knots, values[index[learned], :, channel]), LEARNED_COLOR, ""),
            ]
            figures.append(
                _figure(
                    f"{name.capitalize()} {title}(&phi;)",
                    _plot(series, xlabel="gait phase phi", ylabel=f"{title} [{unit}]", band=(1.0, 2.0)),
                )
            )
    return "".join(figures)


def _loss_plot(summary: dict, split: str, learned: str) -> str:
    base = np.array([row["loss"] for row in summary["splits"][split]["baseline"]["stances"]])
    new = np.array([row["loss"] for row in summary["splits"][split][learned]["stances"]])
    order = np.argsort(base)
    x = np.arange(len(base))
    return _plot(
        [("baseline", x, base[order], BASELINE, "6 4"), ("learned", x, new[order], LEARNED_COLOR, "")],
        xlabel=f"{split} stance (sorted by baseline loss)",
        ylabel="loss",
    )


def _example(run: Path, summary: dict, schedule, labels: list[str], learned: str, stance: dict):
    """Re-run one held-out stance on the CPU with both schedules and plot it against the measurement."""
    dataset = Path(summary["dataset"])
    manifest = json.loads((dataset / "manifest.json").read_text())
    member = next(m for m in manifest["members"] if m["id"] == stance["id"])
    reference = reference_data.load(dataset / member["reference"])
    profile = profile_data.load(dataset / manifest["shared_assets"]["profile.json"]["file"])
    chain = chain_from_profile(reference, profile, RestOfBody())
    plan = _load_plan(run / "plans" / f"{stance['id']}.npz")
    shoe = Shoe(
        summary["shoe_artifact"],
        summary["shoe_mount_m"],
        summary["shoe_static_pitch_rad"],
        friction_model=summary["friction_model"],
    )
    config = Config(phase=summary["phase"])
    timing = reference_timing(plan, config.contact_threshold_n)
    traces = {}
    for label in ("baseline", learned):
        i = labels.index(label)
        impedance = schedule_impedance(
            schedule["knots_phase"], schedule["stiffness"][i], schedule["damping"][i], timing
        )
        traces[label], _ = simulate(plan, chain, impedance, shoe, config=config)
    band = (1e3 * timing[0], 1e3 * timing[1])
    figures = []
    for axis, name in ((1, "Vertical"), (0, "Horizontal")):
        series = [("measured", 1e3 * plan.time_s, plan.grf_n[:, axis], MEASURED, "")]
        for label, color, dash in (("baseline", BASELINE, "6 4"), (learned, LEARNED_COLOR, "")):
            trace = traces[label]
            series.append(
                ("learned" if label == learned else label, 1e3 * trace["time_s"], trace["grf_n"][:, axis], color, dash)
            )
        figures.append(_figure(f"{name} GRF", _plot(series, xlabel="time [ms]", ylabel="force [N]", band=band)))
    for coordinate in ("hip_z", "knee", "ankle"):
        c = list(COORDINATE_INFO).index(coordinate)
        title, unit, _, _ = COORDINATE_INFO[coordinate]
        series = [("reference", 1e3 * plan.time_s, plan.q[:, c], MEASURED, "")]
        for label, color, dash in (("baseline", BASELINE, "6 4"), (learned, LEARNED_COLOR, "")):
            trace = traces[label]
            series.append(
                ("learned" if label == learned else label, 1e3 * trace["time_s"], trace["state"][:, c], color, dash)
            )
        figures.append(_figure(title, _plot(series, xlabel="time [ms]", ylabel=f"{coordinate} [{unit}]", band=band)))
    return "".join(figures)


def write_report(run: Path) -> Path:
    """Write ``report.html`` in a :mod:`.learn` output directory and return its path."""
    summary = json.loads((run / "summary.json").read_text())
    with np.load(run / "schedule.npz", allow_pickle=False) as archive:
        schedule = {name: archive[name] for name in archive.files}
    labels = [str(label) for label in schedule["labels"]]
    learned = _pick_learned(summary)
    history = summary["history"]
    generation = np.array([row["generation"] for row in history])
    convergence = _plot(
        [
            ("mean candidate", generation, [row["mean_score"] for row in history], LEARNED_COLOR, ""),
            ("best sample", generation, [row["best_score"] for row in history], RUN_COLORS[2], ""),
            ("median sample", generation, [row["median_score"] for row in history], BASELINE, "6 4"),
        ],
        xlabel="generation",
        ylabel="training score",
    )
    example = ""
    eval_rows = summary["splits"].get("eval", {}).get(learned, {}).get("stances", [])
    if eval_rows:
        order = np.argsort([row["loss"] for row in eval_rows])
        stance = eval_rows[int(order[len(order) // 2])]
        example = (
            f"<h3>Held-out example: {html.escape(stance['id'])} (median learned loss)</h3>"
            '<p class="note">CPU re-run with the learned schedule mapped to this stance&rsquo;s clock. '
            "Shaded band: measured contact.</p>"
            f'<div class="grid">{_example(run, summary, schedule, labels, learned, stance)}</div>'
        )
    train_agg = summary["splits"]["train"]
    eval_agg = summary["splits"].get("eval", {})

    def change(split, key):
        if not split:
            return "&ndash;"
        a = split["baseline"]["aggregate"].get(key, np.nan)
        b = split[learned]["aggregate"].get(key, np.nan)
        return f"{a:.2f} &rarr; {b:.2f}"

    search = summary["search"]
    knots = ", ".join(f"{k:g}" for k in summary["knots_phase"])
    document = _PAGE.format(
        css=_CSS,
        stances=train_agg["baseline"]["aggregate"]["stances"],
        held_out=eval_agg.get("baseline", {}).get("aggregate", {}).get("stances", 0),
        train_change=change(train_agg, "mean_loss"),
        eval_change=change(eval_agg, "mean_loss"),
        aggregate_table=_aggregate_table(summary, learned),
        eval_table=_eval_table(summary, learned),
        convergence=_figure("Search convergence", convergence, "Training score includes the small log-gain penalty."),
        train_loss=_figure("Training stances", _loss_plot(summary, "train", learned)),
        eval_loss=_figure("Held-out stances", _loss_plot(summary, "eval", learned)) if eval_rows else "",
        gain_plots=_gain_plots(schedule, labels, learned),
        example=example,
        learned=learned.replace("_", " "),
        knots=knots,
        knot_count=len(summary["knots_phase"]),
        population=search["population"],
        generations=search["generations"],
        elite=search["elite_count"],
        phase=summary["phase"],
        scales=summary["loss_scales"],
        run=html.escape(str(run)),
        dataset=html.escape(summary["dataset"]),
        mount=" ".join(f"{v:g}" for v in summary["shoe_mount_m"]),
        wall=summary["wall_s"] / 60.0,
        generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    path = run / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hogan impedance: learned K(phi), D(phi)</title><style>{css}</style></head><body>
<header>
<p class="eyebrow">Impedance Instron / Hogan controller · step 3 of 4</p>
<h1>One impedance schedule learned over {stances} measured stances</h1>
<p class="subtitle">A single stiffness and damping schedule K(&phi;), D(&phi;) on a normalized gait phase, shared by every stance, searched on the GPU with the shoe in the loop and checked on {held_out} held-out stances.</p>
<div class="status"><strong>FORWARD SIMULATION · ONE SHOE · ONE RUNNER · NOT VALIDATED</strong>
<p>Training loss {train_change}; held-out loss {eval_change} (baseline &rarr; learned). Held-out stances never enter the search, and the learned schedule ({learned}) is selected on training loss alone.</p></div>
<nav><a href="#results">01 Results</a><a href="#schedule">02 Learned schedule</a><a href="#method">03 Method</a><a href="#reproduce">04 Reproduce</a></nav>
</header>
<main>
<section id="results"><h2>1. Results</h2>
{aggregate_table}
<div class="grid">{train_loss}{eval_loss}</div>
{eval_table}
{example}
</section>
<section id="schedule"><h2>2. Learned schedule</h2>
<p>Gait phase &phi;: 0&ndash;1 from the window start to measured touchdown, 1&ndash;2 over measured contact (shaded), 2&ndash;3 from toe-off to the window end. Gains are piecewise linear between {knot_count} knots at &phi; = {knots}. Hip x and z gains stay at zero, so the body is carried by the shoe alone.</p>
<div class="grid">{gain_plots}</div>
</section>
<section id="method"><h2>3. Method</h2>
<div class="equation">&tau; = &tau;<sub>ff</sub>(t) + K(&phi;) (q<sub>ref</sub>(t) &minus; q) + D(&phi;) (q̇<sub>ref</sub>(t) &minus; q̇)</div>
<p>Each stance keeps its own reference, inverse-dynamics feedforward, and static height registration (step 1). The pelvis, hip, knee, and ankle K and D are searched as log offsets about the step-1 baseline (K = 500, 500, 300, 300; D critically damped on the mean initial inertia). Reference phase: <b>{phase}</b>.</p>
<p>Per-stance loss: mean over the four angles of (RMSE / {scales[joint_rad]:g} rad)&sup2; + mean over hip x, z of (RMSE / {scales[hip_m]:g} m)&sup2; + mean over Fx, Fz of (RMSE / {scales[force_n]:g} N)&sup2;, plus {scales[failure_penalty]:g} (scaled up by the unfinished fraction) for a failed rollout. The search minimizes the mean over training stances.</p>
<p>Search: cross-entropy method, {population} candidates per generation (the current mean plus {population} &minus; 1 samples), {elite} elites, {generations} generations. Every candidate runs all training stances in one batched CUDA rollout: one world per (stance, candidate) with the batched column-bed shoe and the chain dynamics in double precision. The batched rollout reproduces the CPU rollout to five significant digits on FR3_2.</p>
{convergence}
</section>
<section id="reproduce"><h2>4. Reproduce</h2>
<pre><code>python -m projects.impedance_instron.hogan.learn --dataset {dataset} \\
    --mount {mount} --population {population} --generations {generations} --output {run}
python -m projects.impedance_instron.hogan.learn_report --run {run}</code></pre>
<p class="note">Search and evaluation took {wall:.0f} min on one GPU.</p>
</section>
</main>
<footer>Generated {generated} from <code>{run}</code>.</footer>
</body></html>
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    print(write_report(args.run))


if __name__ == "__main__":
    main()
