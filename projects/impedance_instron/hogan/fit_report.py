# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Write an HTML report for a generative runner fitted by :mod:`.identify`.

Example::

    uv run python -m projects.impedance_instron.hogan.fit_report --run outputs/impedance_instron/generative_fit_lm
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
from pathlib import Path

import numpy as np

from ..cartesian import data as reference_data
from .identify import load_trials, predict_many
from .report import _CSS, BASELINE, COORDINATE_INFO, MEASURED, RUN_COLORS, _figure, _plot, _Scene, _snapshots, _table
from .runner import RolloutConfig, Runner

FITTED = RUN_COLORS[0]
JOINT_COLORS = RUN_COLORS[:3]
JOINTS = ("hip", "knee", "ankle")
CONTACT_N = 50.0


def _rows(summary: dict, split: str, label: str) -> list[dict]:
    return summary["splits"][split][label]["trials"]


def _mean_loss(summary: dict, split: str, label: str) -> float:
    return float(np.mean([row["loss"] for row in _rows(summary, split, label)]))


def _aggregate_table(summary: dict) -> str:
    header = [
        "Split / model",
        "Completed",
        "Loss",
        "Hip RMSE [mm] fwd / up",
        "Pelvis [deg]",
        "Hip [deg]",
        "Knee [deg]",
        "Ankle [deg]",
        "GRF RMSE [N] Fx / Fz",
        "|Peak Fz error| [N]",
        "Contact error [ms]",
    ]
    rows = []
    for split in summary["splits"]:
        for label, name in (("baseline", "seed"), ("learned", "fitted")):
            data = _rows(summary, split, label)
            tracking = np.array([row["tracking_rmse"] for row in data])
            grf = np.array([row["grf_rmse_n"] for row in data])
            rows.append(
                [
                    f"{split} / {name}",
                    f"{sum(row['status'] == 'completed' for row in data)}/{len(data)}",
                    f"{np.mean([row['loss'] for row in data]):.2f}",
                    f"{1e3 * tracking[:, 0].mean():.0f} / {1e3 * tracking[:, 1].mean():.0f}",
                    *(f"{np.rad2deg(tracking[:, c].mean()):.1f}" for c in range(2, 6)),
                    f"{grf[:, 0].mean():.0f} / {grf[:, 1].mean():.0f}",
                    f"{np.mean([abs(row['peak_fz_error_n']) for row in data]):.0f}",
                    f"{1e3 * np.mean([row['contact_duration_error_s'] for row in data]):+.0f}",
                ]
            )
    return _table(
        header,
        rows,
        "Means over stances of free predictions. Contact error is simulated minus measured duration "
        "(&minus; = early toe-off). Loss is <code>identify.score</code> (lower is better).",
    )


def _stance_table(summary: dict, split: str) -> str:
    rows = []
    for base, new in zip(_rows(summary, split, "baseline"), _rows(summary, split, "learned"), strict=True):
        measured_peak = new["peak_grf_n"][1] - new["peak_fz_error_n"]
        measured_contact = new["contact_duration_s"] - new["contact_duration_error_s"]
        rows.append(
            [
                html.escape(new["id"]),
                f"{base['loss']:.1f} &rarr; {new['loss']:.1f}",
                f"{1e3 * new['tracking_rmse'][0]:.0f} / {1e3 * new['tracking_rmse'][1]:.0f}",
                f"{np.rad2deg(np.mean(new['tracking_rmse'][2:])):.1f}",
                f"{base['grf_rmse_n'][1]:.0f} &rarr; {new['grf_rmse_n'][1]:.0f}",
                f"{measured_peak:.0f} / {new['peak_grf_n'][1]:.0f}",
                f"{1e3 * measured_contact:.0f} / {1e3 * new['contact_duration_s']:.0f}",
                new["status"],
            ]
        )
    return _table(
        [
            "Held-out stance",
            "Loss seed &rarr; fitted",
            "Hip RMSE [mm] fwd / up",
            "Mean angle RMSE [deg]",
            "Fz RMSE [N] seed &rarr; fitted",
            "Peak Fz [N] meas. / sim.",
            "Contact [ms] meas. / sim.",
            "Status",
        ],
        rows,
        "None of these stances was used in the fit.",
    )


def _loss_plot(summary: dict, split: str) -> str:
    base = np.array([row["loss"] for row in _rows(summary, split, "baseline")])
    new = np.array([row["loss"] for row in _rows(summary, split, "learned")])
    order = np.argsort(base)
    x = np.arange(len(base))
    return _plot(
        [("seed", x, base[order], BASELINE, "6 4"), ("fitted", x, new[order], FITTED, "")],
        xlabel=f"{split} stance (sorted by seed loss)",
        ylabel="loss",
    )


def _convergence(summary: dict) -> str:
    history = summary.get("history", [])
    if not history:
        return ""
    if "iteration" not in history[0]:
        x = [row["generation"] for row in history]
        series = [("best score", x, [row["best_score"] for row in history], FITTED, "")]
        return _figure("Search convergence", _plot(series, xlabel="generation", ylabel="training score"))
    x = np.arange(len(history) + 1)
    cost = [summary["initial_cost"], *(row["cost"] for row in history)]
    motion = [summary["initial_motion"], *(row["motion"] for row in history)]
    hip = 1e3 * np.array([m["hip_rmse_m"] for m in motion])
    grf = np.array([m["grf_rmse_n"] for m in motion])
    joints = np.rad2deg([m["joint_rmse_rad"] for m in motion])
    peak = np.abs([m["peak_fz_error_n"] for m in motion])
    figures = [
        _figure(
            "LM objective",
            _plot([("cost", x, cost, FITTED, "")], xlabel="iteration", ylabel="training cost"),
            "Mean training objective after each accepted or rejected step.",
        ),
        _figure(
            "Hip position error",
            _plot(
                [("forward", x, hip[:, 0], RUN_COLORS[0], ""), ("up", x, hip[:, 1], RUN_COLORS[1], "")],
                xlabel="iteration",
                ylabel="RMSE [mm]",
            ),
        ),
        _figure(
            "Ground force error",
            _plot(
                [
                    ("Fx RMSE", x, grf[:, 0], RUN_COLORS[0], ""),
                    ("Fz RMSE", x, grf[:, 1], RUN_COLORS[1], ""),
                    ("|peak Fz error|", x, peak, RUN_COLORS[2], "6 4"),
                ],
                xlabel="iteration",
                ylabel="force [N]",
            ),
        ),
        _figure(
            "Joint angle error",
            _plot([("mean angle RMSE", x, joints, RUN_COLORS[0], "")], xlabel="iteration", ylabel="RMSE [deg]"),
        ),
    ]
    return f'<div class="grid">{"".join(figures)}</div>'


def _inside(trial) -> np.ndarray:
    return (trial.force_time_s >= 0) & (trial.force_time_s <= trial.duration_s)


def _contact(trial) -> tuple[float, float, float] | None:
    """Return measured touchdown, toe-off, and peak-Fz times [s] in the prediction window."""
    inside = _inside(trial)
    clock, fz = trial.force_time_s[inside], trial.grf_n[inside, 1]
    loaded = np.flatnonzero(fz > CONTACT_N)
    if not len(loaded):
        return None
    return float(clock[loaded[0]]), float(clock[loaded[-1]]), float(clock[np.argmax(fz)])


def _view(trace: dict, trial) -> dict:
    """Align a trace with its force samples and attach the measured pose for drawing."""
    n = len(trace["grf_n"])
    time = trace["time_s"][:n]
    return {
        "time_s": time,
        "state": trace["state"][:n],
        "grf_n": trace["grf_n"],
        "ankle_contact_moment_nm": trace["ankle_contact_moment_nm"],
        "reference_state": np.column_stack([np.interp(time, trial.time_s, trial.q[:, c]) for c in range(6)]),
    }


def _example(trial, fitted: dict, seed: dict, row: dict, title: str, why: str) -> str:
    contact = _contact(trial)
    cop = reference_data.load(Path(trial.provenance["reference"]))["cop_target_m"]
    reference = {"grf_time_s": trial.force_time_s, "grf_target_n": trial.grf_n, "cop_target_m": cop}
    view = _view(fitted, trial)
    parts = [
        f"<h3>{html.escape(title)}: {html.escape(trial.id)}</h3>",
        f'<p class="note">{why} Loss {row["loss"]:.2f}; hip RMSE {1e3 * row["tracking_rmse"][0]:.0f} / '
        f"{1e3 * row['tracking_rmse'][1]:.0f} mm; Fz RMSE {row['grf_rmse_n'][1]:.0f} N; "
        f"contact {1e3 * row['contact_duration_error_s']:+.0f} ms. Solid blue: fitted model. "
        "Dashed grey: measured pose. Arrows: measured (brown) and simulated (orange) ground force.</p>",
    ]
    band = None
    if contact is not None:
        touchdown, toeoff, peak = contact
        band = (1e3 * touchdown, 1e3 * toeoff)
        times = [touchdown, touchdown + 0.3 * (toeoff - touchdown), peak, touchdown + 0.85 * (toeoff - touchdown)]
        scene = _Scene(trial.chain, trial.shoe, trial.provenance["rest_of_body"]["com_local_m"])
        parts.append(_snapshots(scene, reference, view, times, label=f"{trial.id} stance snapshots"))
    inside = _inside(trial)
    figures = []
    for axis, name in ((1, "Vertical"), (0, "Horizontal")):
        series = [("measured", 1e3 * trial.force_time_s[inside], trial.grf_n[inside, axis], MEASURED, "")]
        for label, trace, color, dash in (("seed", seed, BASELINE, "6 4"), ("fitted", fitted, FITTED, "")):
            n = len(trace["grf_n"])
            series.append((label, 1e3 * trace["time_s"][:n], trace["grf_n"][:, axis], color, dash))
        figures.append(_figure(f"{name} GRF", _plot(series, xlabel="time [ms]", ylabel="force [N]", band=band)))
    for c, name in enumerate(COORDINATE_INFO):
        title_, unit, _, _ = COORDINATE_INFO[name]
        scale, unit = (1.0, unit) if unit == "m" else (np.rad2deg(1.0), "deg")
        # Remove belt-speed travel so forward errors of a few centimeters stay visible.
        drift = trial.task.speed_m_s if c == 0 else 0.0
        if c == 0:
            title_ += ", minus belt-speed travel"
        series = [("measured", 1e3 * trial.time_s, scale * (trial.q[:, c] - drift * trial.time_s), MEASURED, "")]
        for label, trace, color, dash in (("seed", seed, BASELINE, "6 4"), ("fitted", fitted, FITTED, "")):
            values = scale * (trace["state"][:, c] - drift * trace["time_s"])
            series.append((label, 1e3 * trace["time_s"], values, color, dash))
        figures.append(_figure(title_, _plot(series, xlabel="time [ms]", ylabel=f"{name} [{unit}]", band=band)))
    parts.append(f'<div class="grid">{"".join(figures)}</div>')
    return "".join(parts)


def _impedance(trial, fitted: dict) -> str:
    n = len(fitted["grf_n"])
    time = 1e3 * fitted["time_s"][:n]
    contact = _contact(trial)
    band = (1e3 * contact[0], 1e3 * contact[1]) if contact else None
    figures = []
    for key, title, ylabel, scale in (
        ("equilibrium_rad", "Equilibrium angle q<sub>eq</sub>", "angle [deg]", np.rad2deg(1.0)),
        ("stiffness_nm_rad", "Stiffness K", "K [N·m/rad]", 1.0),
        ("damping_nms_rad", "Damping D", "D [N·m·s/rad]", 1.0),
    ):
        series = [(joint, time, scale * fitted[key][:, j], JOINT_COLORS[j], "") for j, joint in enumerate(JOINTS)]
        figures.append(_figure(title, _plot(series, xlabel="time [ms]", ylabel=ylabel, band=band)))
    torque = fitted["load"][:, 3:]
    series = [(joint, time, torque[:, j], JOINT_COLORS[j], "") for j, joint in enumerate(JOINTS)]
    figures.append(_figure("Joint torque", _plot(series, xlabel="time [ms]", ylabel="torque [N·m]", band=band)))
    return f'<div class="grid">{"".join(figures)}</div>'


def write_report(run: Path, *, device: str = "cuda:0") -> Path:
    """Write ``report.html`` in an ``identify fit`` output directory and return its path.

    Args:
        run: Output directory of ``python -m projects.impedance_instron.hogan.identify fit``.
        device: Device for re-running the seed model on the example stances.

    Returns:
        Path of the written report.
    """
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    command = summary["command"]
    trials = load_trials(
        command["dataset"],
        mount_m=command["mount"],
        pitch_rad=command["pitch"],
        speed_m_s=command["speed"],
        height_offset_m=command["height_offset"],
        friction_model=command["friction_model"],
        limit_per_split=command["limit_per_split"],
    )
    if [t.id for t in trials] != [entry["id"] for entry in summary["trials"]]:
        raise ValueError("Dataset trials no longer match the fit summary")
    by_id = {trial.id: (trial, entry) for trial, entry in zip(trials, summary["trials"], strict=True)}
    split = "eval" if "eval" in summary["splits"] else "train"
    rows = _rows(summary, split, "learned")
    order = np.argsort([row["loss"] for row in rows])
    picks = [
        (rows[int(order[0])], "Best motion", f"Lowest {split} loss of {len(rows)} stances."),
        (rows[int(order[len(order) // 2])], "Typical motion", f"Median {split} loss of {len(rows)} stances."),
    ]
    if picks[1][0]["id"] == picks[0][0]["id"]:
        picks = picks[:1]
    chosen = [by_id[row["id"]][0] for row, _, _ in picks]
    config = RolloutConfig(**summary["rollout"])
    seed_traces = [
        trace
        for trace, _ in predict_many([Runner.from_dict(summary["initial_model"])], chosen, config, device=device)[0]
    ]
    examples, impedance = [], ""
    for (row, title, why), trial, seed in zip(picks, chosen, seed_traces, strict=True):
        with np.load(run / by_id[trial.id][1]["trace"]) as archive:
            fitted = {key: archive[key] for key in archive.files}
        examples.append(_example(trial, fitted, seed, row, title, why))
        if not impedance:
            impedance = _impedance(trial, fitted)

    train_count = len(_rows(summary, "train", "learned"))
    eval_count = len(_rows(summary, "eval", "learned")) if "eval" in summary["splits"] else 0
    flight = sum("plate force" in entry["initialization"] for entry in summary["trials"])
    search = summary["search"]
    lm = summary.get("method") == "levenberg_marquardt"
    optimizer = (
        f"Levenberg&ndash;Marquardt, {len(summary['history'])} iterations, forward-difference Jacobian "
        f"over {summary['parameters']} parameters, damping ladder {search['ladder']}, "
        f"{search['chunk']} candidates per batched GPU rollout, {summary['rollouts']} candidate evaluations"
        if lm
        else f"cross-entropy method, population {search['population']}, {search['generations']} generations"
    )
    flags = [
        f"--{key.replace('_', '-')} {' '.join(map(str, value)) if isinstance(value, list) else value}"
        for key, value in command.items()
        if key in ("dataset", "mount", "speed", "compression_limit", "method", "iterations", "chunk", "device")
        and value is not None
    ]
    reproduce = "python -m projects.impedance_instron.hogan.identify fit \\\n    " + " \\\n    ".join(
        [*flags, f"--output {command['output']}"]
    )
    reproduce += f"\npython -m projects.impedance_instron.hogan.fit_report --run {run}"

    def change(split_name):
        if split_name not in summary["splits"]:
            return "&ndash;"
        return (
            f"{_mean_loss(summary, split_name, 'baseline'):.2f} &rarr; {_mean_loss(summary, split_name, 'learned'):.2f}"
        )

    document = _PAGE.format(
        css=_CSS,
        train=train_count,
        held_out=eval_count,
        mass=chosen[0].chain.total_mass_kg,
        speed=chosen[0].task.speed_m_s,
        train_change=change("train"),
        eval_change=change("eval"),
        aggregate_table=_aggregate_table(summary),
        train_loss=_figure("Training stances", _loss_plot(summary, "train")) if train_count > 1 else "",
        eval_loss=_figure("Held-out stances", _loss_plot(summary, "eval")) if eval_count > 1 else "",
        stance_table=_stance_table(summary, "eval") if eval_count else "",
        examples="".join(examples),
        impedance_id=html.escape(chosen[0].id),
        impedance=impedance,
        flight=flight,
        total=len(summary["trials"]),
        optimizer=optimizer,
        objective=html.escape(summary.get("objective", "identify.score")),
        convergence=_convergence(summary),
        wall=summary.get("wall_s", 0.0) / 60.0,
        reproduce=html.escape(reproduce),
        run=html.escape(run.as_posix()),
        generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    path = run / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generative runner fit</title><style>{css}</style></head><body>
<header>
<p class="eyebrow">Impedance Instron / generative runner</p>
<h1>One generative runner fitted to {train} measured stances</h1>
<p class="subtitle">A single shared, impedance-controlled runner ({mass:.0f} kg, {speed:g} m/s belt) predicts every stance freely from its flight start, with the digital shoe in the loop. No measured force, reference trajectory, or feedforward enters the rollout. Checked on {held_out} held-out stances.</p>
<div class="status"><strong>FREE PREDICTION · ONE SHOE · ONE RUNNER · NOT VALIDATED</strong>
<p>Training loss {train_change}; held-out loss {eval_change} (seed &rarr; fitted). Held-out stances never enter the fit.</p></div>
<nav><a href="#results">01 Results</a><a href="#motion">02 Motion</a><a href="#impedance">03 Learned impedance</a><a href="#method">04 Method</a><a href="#reproduce">05 Reproduce</a></nav>
</header>
<main>
<section id="results"><h2>1. Results</h2>
{aggregate_table}
<div class="grid">{train_loss}{eval_loss}</div>
{stance_table}
</section>
<section id="motion"><h2>2. Motion</h2>
<p>Stances are picked after the fit; the choice does not affect the model. Snapshots are at measured touchdown, 30% of contact, measured peak Fz, and 85% of contact. Shaded band: measured contact (Fz &gt; 50 N).</p>
{examples}
</section>
<section id="impedance"><h2>3. Learned impedance on {impedance_id}</h2>
<p>The runner&rsquo;s equilibrium, stiffness, and damping for hip, knee, and ankle as generated during this prediction. They follow from the shared weights, the internal oscillator phase, and the simulated load and posture; nothing here is fitted per stance.</p>
{impedance}
</section>
<section id="method"><h2>4. Method</h2>
<div class="equation">&tau; = K(s) (q<sub>eq</sub>(s) &minus; q) &minus; D(s) q̇,&nbsp;&nbsp; s = (oscillator phase, filtered simulated load, posture, speed)</div>
<p>Planar chain: hip x and z, lumped rest-of-body tilt, hip, knee, and ankle, with the column-bed digital shoe on the foot. Only the hip, knee, and ankle are actuated; torques pass through a first-order response and slew bound. The weights are shared by all stances; there are no stance-specific coefficients.</p>
<p><b>Initialization.</b> Positions come from the three-frame flight prefix, and joint rates from backward quadratic differentiation. The hip velocity is set so the model&rsquo;s whole-body center of mass moves at the flight velocity integrated from the preceding stride&rsquo;s treadmill plate force ({flight}/{total} stances; the rest have no preceding stride and use the three-frame hip estimate). The swinging leg alone carries enough momentum to shift the COM by about 0.3 m/s, so matching the hip instead biases the start.</p>
<p><b>Loss</b> per stance: mean over hip x, z of (RMSE / 0.02 m)&sup2; + mean over the four angles of (RMSE / 0.05 rad)&sup2; + mean over Fx, Fz of (RMSE / 100 N)&sup2; + (peak Fz error / 100 N)&sup2; + mean over axes of (impulse error / 20 N·s)&sup2; + (contact error / 0.02 s)&sup2; + 0.01 &times; normalized effort, plus a penalty for an unfinished rollout. Optimizer objective: {objective}.</p>
<p><b>Fit.</b> {optimizer}. Selection uses training stances only.</p>
{convergence}
</section>
<section id="reproduce"><h2>5. Reproduce</h2>
<pre><code>{reproduce}</code></pre>
<p class="note">The fit took {wall:.0f} min on one GPU.</p>
</section>
</main>
<footer>Generated {generated} from <code>{run}</code>.</footer>
</body></html>
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Device for the seed comparison rollouts")
    args = parser.parse_args(argv)
    print(write_report(args.run, device=args.device))


if __name__ == "__main__":
    main()
