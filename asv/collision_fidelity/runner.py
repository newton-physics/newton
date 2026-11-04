# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run collision-fidelity scenes with the installed, unmodified VBD solver."""

from __future__ import annotations

import math
import subprocess
import time
import traceback
from pathlib import Path

from .protocol import Timeline


def summarize_results(results: list[dict]) -> dict:
    """Aggregate declared outcomes and means of covered per-case maxima."""
    summary = {"case_count": len(results), "execution": {}, "criteria": {}, "means": {}}
    for result in results:
        status = result["execution"]["status"]
        summary["execution"][status] = summary["execution"].get(status, 0) + 1
    names = {name for result in results for name in result.get("criteria", {})}
    for name in sorted(names):
        counts = dict.fromkeys(("pass", "fail", "incomplete", "not_applicable", "unsupported"), 0)
        for result in results:
            state = result.get("criteria", {}).get(name, {}).get("status", "incomplete")
            if result["execution"]["status"] == "unsupported":
                state = "unsupported"
            counts[state if state in counts else "incomplete"] += 1
        evaluated = counts["pass"] + counts["fail"]
        applicable = evaluated + counts["incomplete"]
        summary["criteria"][name] = {
            **counts,
            "evaluated": evaluated,
            "applicable": applicable,
            "pass_rate": counts["pass"] / evaluated if evaluated else None,
            "evaluation_coverage": evaluated / applicable if applicable else None,
        }
    for field, destination, multiplier in (
        ("jitter_m", "jitter_mm", 1000.0),
        ("rms_speed_m_s", "rms_speed_m_s", 1.0),
        ("rigid_soft_penetration_m", "rigid_soft_penetration_mm", 1000.0),
    ):
        values = []
        cases = []
        for result in results:
            measurement = result.get("measurements", {})
            section = measurement.get("penetration" if field == "rigid_soft_penetration_m" else "stability", {})
            value = section.get(field)
            if field == "rigid_soft_penetration_m":
                covered = (
                    result["execution"].get("full_timeline_completed", False)
                    and measurement.get("finite", {}).get("all_samples_finite") is True
                )
            else:
                required = section.get("declared_complete_window_count", 0)
                covered = required > 0 and section.get("complete_window_count") == required
            if covered and value is not None and math.isfinite(value):
                values.append(value * multiplier)
                cases.append(result["case_id"])
        summary["means"][destination] = {
            "value": sum(values) / len(values) if values else None,
            "count": len(values),
            "cases": cases,
            "aggregation": "Equal-case mean of per-case maxima; complete required measurement coverage only",
        }
    return summary


def _static_supports(model):
    """Describe exposed fixed planes and boxes for particle-shell sampling."""
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415

    if not model.shape_count:
        return []
    bodies = model.shape_body.numpy()
    types = model.shape_type.numpy()
    poses = model.shape_transform.numpy()
    scales = model.shape_scale.numpy()
    flags = model.shape_flags.numpy()
    margins = model.shape_margin.numpy()
    surfaces = []
    for index, body in enumerate(bodies):
        if body != -1 or not int(flags[index]) & int(newton.ShapeFlags.COLLIDE_PARTICLES):
            continue
        pose = poses[index]
        common = {"shape_index": index, "shape_margin_m": float(margins[index])}
        if int(types[index]) == int(newton.GeoType.PLANE):
            normal = np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(0, 0, 1)), dtype=float)
            surfaces.append(
                {**common, "kind": "plane", "normal": normal.tolist(), "offset_m": float(normal @ pose[:3])}
            )
        elif int(types[index]) == int(newton.GeoType.BOX):
            surfaces.append(
                {
                    **common,
                    "kind": "box",
                    "center_m": pose[:3].tolist(),
                    "half_extents_m": scales[index].tolist(),
                    "rotation_xyzw": pose[3:].tolist(),
                }
            )
    return surfaces


def _revision(module):
    """Record a source checkout revision when the installed package has one."""
    root = Path(module.__file__).resolve().parent.parent
    if not (root / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _sample_capacity(solver, record):
    """Check exposed final contact counts, without intercepting detection."""
    contacts = getattr(solver, "contacts", None)
    if contacts is None:
        return False
    observed = False
    overflow = False
    for kind in ("rigid", "soft"):
        count = getattr(contacts, f"{kind}_contact_count", None)
        capacity = getattr(contacts, f"{kind}_contact_max", None)
        if count is None or capacity is None:
            continue
        value = int(count.numpy()[0])
        observed = True
        record[f"maximum_{kind}_count"] = max(record.get(f"maximum_{kind}_count", 0), value)
        record[f"{kind}_capacity"] = int(capacity)
        overflow |= value > capacity
    if observed:
        record["samples"] += 1
        record["status"] = "overflow" if overflow else record.get("status", "observed")
        if record["status"] == "unavailable":
            record["status"] = "observed"
    return overflow


def run_case(
    case_id: str,
    *,
    profile: str = "full",
    device: str | None = None,
    duration: float | None = None,
    usd_path: str | Path | None = None,
) -> dict:
    """Execute one case and return measurements, coverage and explicit outcomes.

    A smoke run advances at most one physical second. It keeps the full
    measurement schedule, so an unobserved hold remains incomplete.
    """
    from .catalog import load_cases  # noqa: PLC0415
    from .version import semantic_version  # noqa: PLC0415

    cases = load_cases()
    if case_id not in cases:
        raise ValueError(f"Unknown case {case_id!r}")
    if profile not in ("smoke", "full"):
        raise ValueError("Profile must be smoke or full")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Duration must be finite and positive")
    case = cases[case_id]
    timeline = Timeline(case)
    result = {
        "schema_version": 1,
        "case_id": case_id,
        "profile": profile,
        "protocol": timeline.protocol,
        "execution": {"status": "unsupported", "completed": False, "steps": 0, "planned_steps": 0, "time_s": 0.0},
        "measurements": {},
        "criteria": {},
        "provenance": {
            "input_version": case.get("input_version"),
            "benchmark_version": semantic_version(case_id),
            "solver": "Standard VBD",
        },
    }
    support = case.get("support", {})
    if support.get("status") == "unsupported":
        result["execution"]["reason"] = support["reason"]
        result["criteria"]["overall"] = {"status": "unsupported", "applicable": False, "reason": support["reason"]}
        return result

    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415

    from .metrics import FidelityObserver  # noqa: PLC0415
    from .scenes import build_scene  # noqa: PLC0415
    from .tasks import criterion_results, sample_task  # noqa: PLC0415

    started = time.perf_counter()
    observer = None
    viewer = None
    frames = []
    execution = result["execution"]
    execution["status"] = "error"
    capacity = {
        "status": "unavailable",
        "samples": 0,
        "scope": "Final rigid/soft contact counts after each step; internal detection peaks and self-contact capacity are unavailable.",
    }
    result["capacity"] = capacity
    try:
        selected_device = wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
        with wp.ScopedDevice(selected_device):
            scene = build_scene(case_id, device=selected_device)
            if scene.substeps != timeline.substeps or not math.isclose(
                scene.frame_dt, timeline.frame_dt, abs_tol=1e-12
            ):
                raise ValueError("Scene integration settings differ from the case manifest")
            if scene.solver.iterations != case["iterations"]:
                raise ValueError("Scene solver iterations differ from the case manifest")
            if not math.isclose(scene.duration, case["duration_s"], abs_tol=1e-8):
                raise ValueError("Scene duration differs from the case manifest")
            result["provenance"].update(
                newton_version=getattr(newton, "__version__", None),
                newton_revision=_revision(newton),
                warp_version=wp.__version__,
                device=str(selected_device),
                particles=int(scene.model.particle_count),
                bodies=int(scene.model.body_count),
                iterations=int(scene.solver.iterations),
                frame_dt_s=timeline.frame_dt,
                substeps=timeline.substeps,
                dt_s=timeline.dt,
                solver_settings=getattr(scene.model, "collision_fidelity_solver_options", None),
                contact_policy=getattr(scene.model, "collision_benchmark_contact_policy", None),
            )
            result["inputs"] = scene.metrics()
            timeline.protocol["penetration_surfaces"] = _static_supports(scene.model)
            if hasattr(scene, "cable_bodies"):
                timeline.protocol["cable_body_indices"] = [int(i) for i in scene.cable_bodies]
            observer = FidelityObserver(scene.model, timeline.protocol)
            observer.frame(scene.state, 0.0)
            frames.append(sample_task(scene, 0.0, protocol=timeline.protocol))
            stop = min(timeline.duration, 1.0) if profile == "smoke" else timeline.duration
            if duration is not None:
                stop = min(stop, duration)
            planned_steps = min(timeline.steps, math.ceil(stop / timeline.dt - 1e-9))
            execution.update(planned_steps=planned_steps, full_timeline_steps=timeline.steps)
            if usd_path is not None:
                from newton.viewer import ViewerUSD  # noqa: PLC0415

                viewer = ViewerUSD(str(usd_path), fps=round(1.0 / timeline.frame_dt), num_frames=None)
                viewer.set_model(scene.model)
                viewer.begin_frame(0.0)
                viewer.log_state(scene.state)
                viewer.end_frame()
            for index in range(planned_steps):
                physical_start = index * timeline.dt
                state = scene.step_substep(timeline.command_time(physical_start))
                end = (index + 1) * timeline.dt
                execution.update(steps=index + 1, time_s=end)
                if _sample_capacity(scene.solver, capacity):
                    execution.update(status="invalid", reason="Exposed contact count exceeds capacity")
                    break
                observer.observe(state, end, timeline.dt)
                if not observer.all_samples_finite:
                    execution.update(status="nonfinite", reason="Nonfinite state observed")
                    break
                if (index + 1) % timeline.substeps == 0 or index + 1 == planned_steps:
                    frames.append(sample_task(scene, end, protocol=timeline.protocol))
                    if viewer is not None:
                        viewer.begin_frame(end)
                        viewer.log_state(state)
                        viewer.end_frame()
            else:
                execution.update(status="completed", completed=True)
            execution["full_timeline_completed"] = execution["completed"] and execution["steps"] == timeline.steps
    except NotImplementedError as error:
        execution.update(status="unsupported", reason=str(error))
    except Exception as error:
        execution.update(status="error", reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if viewer is not None:
            viewer.close()
        result["wall_time_s"] = time.perf_counter() - started
    if observer is not None:
        result["measurements"] = observer.result()
    result["task_observations"] = frames
    result["criteria"] = criterion_results(
        frames, timeline.protocol, execution=execution, measurements=result["measurements"]
    )
    return result
