# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Write a local static video review page from the same run's recorded results."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from urllib.parse import quote

from .runner import summarize_results

_METRICS = (
    ("Jitter (mm)", "stability", "jitter_m", "jitter_mm", 1000.0),
    ("RMS speed (m/s)", "stability", "rms_speed_m_s", "rms_speed_m_s", 1.0),
    (
        "Rigid-soft penetration: fixed supports (mm)",
        "penetration",
        "rigid_soft_penetration_m",
        "rigid_soft_penetration_mm",
        1000.0,
    ),
)

_CSS = """
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#192438;font:16px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:24px}h1{font-size:1.8rem}h2{font-size:1.25rem}
h1,h2,p,code{overflow-wrap:anywhere}article{margin:24px 0;padding:20px;background:white;border:1px solid #d7dfeb;border-radius:8px}
.integration{background:#eaf1ff;padding:10px;font-variant-numeric:tabular-nums}.muted{color:#526177}
video{display:block;width:100%;max-width:960px;height:auto;background:#111;margin:12px auto}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:12px 0;font-size:.9rem}
th,td{padding:8px;text-align:left;border-bottom:1px solid #dce2eb;vertical-align:top}th{background:#eaf0f7}
details{padding:12px 0;border-top:1px solid #dce2eb}summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem;background:#f4f6fa;padding:12px}
a{color:#125cac}@media(max-width:600px){main{padding:12px}article{padding:12px}th,td{padding:6px}h1{font-size:1.5rem}}
"""


def _escape(value):
    """Escape plain data for HTML text and quoted attributes."""
    return html.escape(str(value), quote=True)


def _number(value, multiplier=1.0):
    """Keep missing values distinct from zero and retain small nonzero values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "Missing"
    return f"{value * multiplier:.6g}"


def _label(value):
    """Display schema names without inventing an outcome."""
    return str(value).replace("_", " ").capitalize()


def _table(headers, rows, table_id=None):
    """Render a small escaped table with a local horizontal scroll boundary."""
    identifier = f' id="{_escape(table_id)}"' if table_id else ""
    head = "".join(f'<th scope="col">{_escape(value)}</th>' for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>" for row in rows)
    return (
        f'<div class="table-wrap"><table{identifier}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def _local_file(out_dir, name, suffix):
    """Accept only an existing nonempty file inside this result directory."""
    if not isinstance(name, str) or not name or Path(name).name != name or "\\" in name:
        return None
    path = out_dir / name
    if path.suffix != suffix or path.resolve().parent != out_dir.resolve() or not path.is_file():
        return None
    return quote(name, safe="") if path.stat().st_size else None


def _recording(out_dir, recording):
    """Expose completed or partial recordings without inferring visual success."""
    status = recording.get("status", "unavailable")
    count = recording.get("frame_count")
    source = _local_file(out_dir, recording.get("path"), ".mp4")
    usable = status in ("completed", "partial") and isinstance(count, int) and count > 0 and source
    title = "Partial recording" if status == "partial" else f"Recording: {_label(status)}"
    message = f"<p><strong>{_escape(title)}</strong>"
    if count is not None:
        message += f" · {_escape(_number(count))} frames · {_escape(_number(recording.get('fps')))} fps"
    message += "</p>"
    if recording.get("error"):
        message += f"<p>{_escape(recording['error'])}</p>"
    if usable:
        message += (
            '<p class="muted">Not visually reviewed. Recording completion does not establish task success.</p>'
            f'<video controls preload="metadata" playsinline src="{_escape(source)}"></video>'
        )
    else:
        message += '<p class="muted">No usable recording available.</p>'
    return message


def _card(out_dir, result, case):
    """Show measured values and timing while folding detailed input metadata."""
    case_id = result["case_id"]
    execution = result["execution"]
    provenance = result.get("provenance") or {}
    inputs = result.get("inputs") or {}
    physics = inputs.get("physics_parameters") or {}
    protocol = result.get("protocol") or {}

    def setting(key, *aliases):
        for source in (provenance, physics, protocol, case):
            for name in (key, *aliases):
                if source.get(name) is not None:
                    return source[name]
        return None

    integration = (
        f"Frame dt {_number(setting('frame_dt_s'))} s · Physical dt {_number(setting('dt_s', 'substep_dt_s'))} s · "
        f"Substeps {_number(setting('substeps'))} · Iterations {_number(setting('iterations'))}"
    )
    measurement = result["measurements"]
    rows = [
        [label, _number(measurement.get(section, {}).get(key), scale)] for label, section, key, _, scale in _METRICS
    ]
    stability = measurement.get("stability", {})
    criteria = result["criteria"]
    states = [[_label(name), _label(row.get("status", "incomplete"))] for name, row in criteria.items()]
    if not states:
        states = [["Overall", "Unsupported" if execution["status"] == "unsupported" else "Incomplete"]]
    href = _local_file(out_dir, f"{case_id}.json", ".json")
    link = f'<a href="{_escape(href)}">Result JSON</a>' if href else "Result JSON unavailable."
    details = {
        "case": case,
        "inputs": inputs,
        "provenance": provenance,
        "protocol": protocol,
        "criteria": criteria,
        "recording": result.get("recording") or {},
        "measurement_scope": {key: measurement.get(key) for key in ("sampling", "finite", "penetration")},
    }
    return f"""<article>
<h2>{_escape(case.get("title", case_id))}</h2><p><code>{_escape(case_id)}</code> · {link}</p>
<p class="integration">{_escape(integration)}</p>
<p><strong>Execution: {_escape(_label(execution["status"]))}</strong> · Observed {_number(execution.get("time_s"))} s · {_number(execution.get("steps"))} physical steps.</p>
<p>{_escape(execution.get("reason") or "")}</p>
{_table(["Criterion", "Recorded outcome"], states)}
{_table(["Measurement", "Raw recorded value"], rows)}
<p>Complete stationary windows: {_number(stability.get("complete_window_count"))} / {_number(stability.get("declared_complete_window_count"))} declared. Values from incomplete runs describe only their observed prefix.</p>
{_recording(out_dir, result.get("recording") or {})}
<details><summary>Description, material, contact, mesh and provenance</summary>
<p>{_escape(case.get("description") or inputs.get("description") or result.get("description") or "No description recorded.")}</p>
<pre>{_escape(json.dumps(details, indent=2, ensure_ascii=False))}</pre></details>
</article>"""


def write_review(out_dir: Path, results: list[dict], cases: dict) -> Path:
    """Write index.html beside selected results and their local recordings.

    Read only the supplied records and existing local filenames. This function
    does not run physics, decode videos, infer visual outcomes or fetch assets.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = []
    for result in results:
        measurements = result.get("measurements") or {}
        normalized.append(
            {
                **result,
                "execution": {"status": "unknown", **(result.get("execution") or {})},
                "criteria": result.get("criteria") or {},
                "measurements": {
                    **measurements,
                    **{key: measurements.get(key) or {} for key in ("stability", "penetration", "finite")},
                },
            }
        )
    summary = summarize_results(normalized)
    criteria = summary["criteria"]
    counts = [
        [
            _label(name),
            *(
                row.get(key, 0)
                for key in ("fail", "evaluated", "applicable", "incomplete", "not_applicable", "unsupported")
            ),
            _number(row.get("failure_rate"), 100.0),
            _number(row.get("unevaluated_fraction"), 100.0),
        ]
        for name, row in criteria.items()
    ]
    means = []
    for label, _, _, key, _ in _METRICS:
        row = summary["means"].get(key, {})
        means.append([label, _number(row.get("value")), _number(row.get("count"))])
    execution = " · ".join(f"{_label(key)}: {value}" for key, value in summary["execution"].items())
    cards = "".join(_card(out_dir, result, cases.get(result["case_id"], {})) for result in normalized)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Collision-fidelity video review</title>
<style>{_CSS}</style></head><body><main><h1>Collision-fidelity video review</h1>
<p>{len(results)} selected cases. {_escape(execution)}</p>
<p>Execution, recorded criteria and visual review are separate. Incomplete is neither Fail nor Pass. No outcome is inferred from video availability.</p>
{_table(["Criterion", "Failed", "Evaluated", "Applicable", "Incomplete", "N/A", "Unsupported", "Failure rate (%)", "Unevaluated fraction (%)"], counts, "aggregate-criteria")}
<p>Failure rate is Fail / (Pass + Fail). Unevaluated fraction is Incomplete / (Pass + Fail + Incomplete). Empty denominators are Missing. Unsupported and not-applicable cases are excluded from those denominators.</p>
{_table(["Equal-case mean of recorded maxima", "Value", "Cases with complete coverage"], means, "aggregate-measurements")}
<p>Means use only the runner's complete-measurement cohorts. Per-case values below retain observed prefixes and their window coverage. Lower motion alone does not establish retention or task success.</p>
<details><summary>Measurement definitions and review limits</summary>
<p>Jitter is point-displacement RMS after removing fitted linear drift. RMS speed uses point velocities in the declared frame. Each value is the maximum over all measured points and complete 0.5 s stationary windows. Trailing partial windows are excluded.</p>
<p>Rigid-soft penetration samples particle-shell overlap with declared fixed supports. It does not cover every contact interface or prove absence of crossings. Missing measurements and unset acceptance limits remain explicit. Video phase labels name commands, not observed outcomes.</p></details>
{cards}</main></body></html>"""
    path = out_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path
