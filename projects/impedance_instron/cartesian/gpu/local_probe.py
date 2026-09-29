# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Probe one qualified 12x4 controller near its saved local neighborhood.

The probe keeps the frozen measured objective, native timestep, fixed gains,
and strict canonical spline bounds unchanged. It samples a finite set of local
directions around one saved fitted controller. Finite bounded probes can show
that some nearby directions are worse or better, but they cannot prove local or
global optimality.

Example:
    uv run --no-sync -m projects.impedance_instron.cartesian.gpu.local_probe         --run-directory outputs/impedance_instron/gain_sweeps/S001/runs/k0.50_d0.50_reference_s17_n50         --output outputs/impedance_instron/gain_sweeps/local_probe_example
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import warp as wp

from ..fit import FitConfig
from ..run import Config
from ..trajectory import Spline
from .__main__ import _validation
from .benchmark import _plain, execution_identity, load_frozen
from .constraints import SplineConstraints
from .engine import Engine
from .provenance import source_snapshot, validate_sources

wp.set_module_options({"enable_backward": False, "fuse_fp": False})

_CONTROL_COUNT = 12
_CHANNEL_COUNT = 4
_WORLD_COUNT = 128
_AXIS_SCALE_FRACTIONS = (0.01, 0.02)
_RANDOM_DIRECTION_COUNT = 32
_FAILURE_REASONS = {
    1: "Nonfinite leg state or velocity",
    2: "Nonfinite actuator load",
    4: "Joint range exceeded",
    8: "Hip height screen exceeded; leg-ground collision is not modeled",
    16: "Numerical speed screen exceeded",
    32: "Hip force screen exceeded",
    64: "Nonfinite or invalid shoe wrench",
    128: "Nonfinite shoe compression",
    256: "Driven shoe compression screen exceeded",
    512: "Shoe supplied tensile ground normal force",
    1024: "Ground force screen exceeded",
    2048: "Nonfinite integrated state or velocity",
}
_CHANNEL_NAMES = ("hip_x_m", "hip_z_m", "knee_rad", "ankle_rad")
_RANGE_NAMES = ("knee_rad", "ankle_rad")


@wp.kernel
def _populate_probe_batch(
    batch_start: int,
    total_probes: int,
    baseline: wp.array2d[wp.float64],
    box_span: wp.array[wp.float64],
    kind_code: wp.array[int],
    coord_index: wp.array[int],
    random_index: wp.array[int],
    sign_code: wp.array[int],
    scale_fraction: wp.array[wp.float64],
    random_relative_direction: wp.array3d[wp.float64],
    coefficients: wp.array3d[wp.float64],
):
    """Populate one fixed-size 128-slot batch directly on the GPU."""
    world = wp.tid()
    probe_index = batch_start + world
    if probe_index >= total_probes:
        for row in range(_CONTROL_COUNT):
            for channel in range(_CHANNEL_COUNT):
                coefficients[world, row, channel] = baseline[row, channel]
        return

    kind = kind_code[probe_index]
    signed_scale = wp.float64(sign_code[probe_index]) * scale_fraction[probe_index]
    coordinate = coord_index[probe_index]
    random_id = random_index[probe_index]
    coordinate_row = coordinate // _CHANNEL_COUNT
    coordinate_channel = coordinate - coordinate_row * _CHANNEL_COUNT

    for row in range(_CONTROL_COUNT):
        for channel in range(_CHANNEL_COUNT):
            value = baseline[row, channel]
            if kind == 1 and row == coordinate_row and channel == coordinate_channel:
                value += signed_scale * box_span[channel]
            elif kind == 2:
                value += signed_scale * box_span[channel] * random_relative_direction[random_id, row, channel]
            coefficients[world, row, channel] = value


@wp.kernel
def _copy_probe_candidate(
    world: int,
    source: wp.array3d[wp.float64],
    destination: wp.array2d[wp.float64],
):
    """Copy one selected candidate from the resident batch into a retained buffer."""
    row, channel = wp.tid()
    destination[row, channel] = source[world, row, channel]


def _write_json(path: Path, value: Any) -> None:
    """Atomically replace one completed JSON artifact."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_npz(path: Path, **values: Any) -> None:
    """Atomically replace one completed compressed array artifact."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _decode_failure(code: int) -> list[str]:
    """Expand the saved engine bit mask into stable text reasons."""
    return [text for bit, text in _FAILURE_REASONS.items() if bit & code]


def _range_metadata(mask: int) -> dict[str, bool]:
    """Report which diagnostic joint ranges were exceeded."""
    return {name: bool(mask & (1 << index)) for index, name in enumerate(_RANGE_NAMES)}


@wp.kernel
def _random_relative_directions(
    seed: int,
    directions: wp.array3d[wp.float64],
):
    """Generate and normalize smooth relative directions without host proposals."""
    index = wp.tid()
    state = wp.rand_init(seed, index)
    peak = wp.float64(0.0)
    for channel in range(_CHANNEL_COUNT):
        a = wp.float64(wp.randn(state))
        b = wp.float64(wp.randn(state))
        c = wp.float64(wp.randn(state))
        d = wp.float64(wp.randn(state))
        for row in range(directions.shape[1]):
            u = wp.float64(row) / wp.float64(_CONTROL_COUNT - 1)
            value = a + b * (u - wp.float64(0.5))
            value += c * wp.sin(wp.float64(3.141592653589793) * u)
            value += d * wp.sin(wp.float64(6.283185307179586) * u)
            directions[index, row, channel] = value
            peak = wp.max(peak, wp.abs(value))
    for row in range(directions.shape[1]):
        for channel in range(_CHANNEL_COUNT):
            if peak > wp.float64(0.0):
                directions[index, row, channel] /= peak


@wp.kernel
def _mask_padding(batch_start: int, total_probes: int, bounded: wp.array[int]):
    """Keep valid baseline copies in padding slots out of the integration graph."""
    world = wp.tid()
    if batch_start + world >= total_probes:
        bounded[world] = 0


def _probe_specs(random_direction_count: int) -> list[dict[str, Any]]:
    """Enumerate baseline, axis, and smooth random probes in a fixed order."""
    specs: list[dict[str, Any]] = [
        {
            "probe_index": 0,
            "kind": "baseline",
            "kind_code": 0,
            "coord_index": -1,
            "control_index": None,
            "channel_index": None,
            "channel_name": None,
            "random_direction_index": None,
            "sign": 0,
            "scale_fraction": 0.0,
            "scale_percent": 0.0,
        }
    ]
    for coord in range(_CONTROL_COUNT * _CHANNEL_COUNT):
        row, channel = divmod(coord, _CHANNEL_COUNT)
        for scale_fraction in _AXIS_SCALE_FRACTIONS:
            for sign in (1, -1):
                specs.append(
                    {
                        "probe_index": len(specs),
                        "kind": "axis",
                        "kind_code": 1,
                        "coord_index": coord,
                        "control_index": row,
                        "channel_index": channel,
                        "channel_name": _CHANNEL_NAMES[channel],
                        "random_direction_index": None,
                        "sign": sign,
                        "scale_fraction": float(scale_fraction),
                        "scale_percent": float(100.0 * scale_fraction),
                    }
                )
    for random_index in range(random_direction_count):
        for scale_fraction in _AXIS_SCALE_FRACTIONS:
            for sign in (1, -1):
                specs.append(
                    {
                        "probe_index": len(specs),
                        "kind": "random",
                        "kind_code": 2,
                        "coord_index": -1,
                        "control_index": None,
                        "channel_index": None,
                        "channel_name": None,
                        "random_direction_index": random_index,
                        "sign": sign,
                        "scale_fraction": float(scale_fraction),
                        "scale_percent": float(100.0 * scale_fraction),
                    }
                )
    return specs


def _identity_from_summary(summary: dict[str, Any], reference_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the warm-start identity format used by GPU optimization."""
    shoe = summary["shoe"]
    return {
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        "simulation_config": summary["simulation_config"],
        "shoe": {
            "artifact_sha256": shoe["sha256"],
            "mount_m": shoe["mount_m"],
            "static_pitch_rad": shoe["static_pitch_rad"],
            "friction": shoe["friction"],
            "device": "cuda:0",
        },
    }


def _verify_starting_controller_identity(
    starting_controller: Path,
    baseline_directory: Path,
    baseline_summary: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Require warm-start compatibility with the qualified baseline bundle."""
    with np.load(starting_controller, allow_pickle=False) as archive:
        duration_s = float(archive["duration_s"])
        identity = json.loads(str(archive["identity_json"]))
    expected = _identity_from_summary(baseline_summary, baseline_directory / "reference.npz", profile)
    if identity != expected:
        raise ValueError("Starting controller identity differs from the qualified baseline inputs")
    with np.load(baseline_directory / "reference.npz", allow_pickle=False) as archive:
        expected_duration_s = float(archive["time_s"][-1])
    if not np.isclose(duration_s, expected_duration_s, rtol=0.0, atol=1.0e-12):
        raise ValueError("Starting controller duration differs from the qualified baseline duration")
    return identity


def _named_metrics(values: np.ndarray) -> dict[str, list[float | None]]:
    """Group six objective channels into the standard three physical blocks."""
    return {
        "hip": [float(item) if np.isfinite(item) else None for item in values[0:2]],
        "joint": [float(item) if np.isfinite(item) else None for item in values[2:4]],
        "force": [float(item) if np.isfinite(item) else None for item in values[4:6]],
    }


def _result_record(
    spec: dict[str, Any],
    *,
    batch_index: int,
    slot_index: int,
    bounded: bool,
    integrated_steps: int,
    failure_code: int,
    failure_step: int,
    range_step: int,
    range_mask: int,
    loss: float,
    rmse: np.ndarray,
    maximum_error: np.ndarray,
    costs: np.ndarray,
    time_s: np.ndarray,
) -> dict[str, Any]:
    """Convert one resident slot into a serializable per-probe record."""
    completed = bool(
        bounded
        and failure_code == 0
        and integrated_steps == len(time_s) - 1
        and np.isfinite(loss)
        and loss >= 0.0
        and np.isfinite(rmse).all()
        and np.isfinite(maximum_error).all()
        and np.isfinite(costs).all()
    )
    if not bounded:
        status = "bound_rejected"
    elif completed:
        status = "completed"
    else:
        status = "failed"
    failure_time_s = None
    if failure_code != 0 and 0 <= failure_step < len(time_s):
        failure_time_s = float(time_s[failure_step])
    range_time_s = None
    if range_mask != 0 and 0 <= range_step < len(time_s):
        range_time_s = float(time_s[range_step])
    return {
        **spec,
        "batch_index": batch_index,
        "slot_index": slot_index,
        "attempted": True,
        "bounded": bool(bounded),
        "evaluated": bool(bounded),
        "completed": completed,
        "status": status,
        "loss": float(loss) if np.isfinite(loss) else None,
        "rmse": [float(value) if np.isfinite(value) else None for value in rmse],
        "maximum_error": [float(value) if np.isfinite(value) else None for value in maximum_error],
        "rmse_by_block": _named_metrics(rmse),
        "maximum_error_by_block": _named_metrics(maximum_error),
        "objective_components": [float(value) if np.isfinite(value) else None for value in costs],
        "integrated_steps": int(integrated_steps),
        "failure_code": int(failure_code) if bounded else None,
        "failure_reasons": _decode_failure(int(failure_code)) if bounded and failure_code != 0 else None,
        "failure_step": int(failure_step) if bounded and failure_step >= 0 else None,
        "failure_time_s": failure_time_s,
        "joint_range_step": int(range_step) if bounded and range_step >= 0 else None,
        "joint_range_time_s": range_time_s,
        "joint_range_mask": int(range_mask) if bounded and range_mask != 0 else None,
        "joint_range_exceeded": _range_metadata(int(range_mask)) if bounded and range_mask != 0 else None,
    }


def probe_local_neighborhood(
    run_directory: Path,
    output: Path,
    *,
    seed: int = 17,
    random_direction_count: int = _RANDOM_DIRECTION_COUNT,
) -> dict[str, Any]:
    """Probe one qualified fitted controller without changing the physics path."""
    started = perf_counter()
    run_directory = Path(run_directory)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(random_direction_count, bool)
        or not isinstance(random_direction_count, int)
        or random_direction_count < 1
    ):
        raise ValueError("random_direction_count must be a positive integer")

    baseline_directory = run_directory / "baseline"
    fit_directory = run_directory / "fit"
    single_validation = run_directory / "single" / "benchmark.json"
    mixed_validation = run_directory / "mixed" / "benchmark.json"
    contact_evidence = run_directory / "contact" / "report.json"
    starting_controller = fit_directory / "equilibrium.npz"
    if not fit_directory.exists():
        raise FileNotFoundError(f"Missing fitted controller directory: {fit_directory}")

    reference, profile, coefficients, fit_summary = load_frozen(fit_directory)
    if coefficients.shape != (_CONTROL_COUNT, _CHANNEL_COUNT):
        raise ValueError("The fitted controller must have shape (12, 4)")
    if not fit_summary.get("complete", False):
        raise ValueError("The fitted controller summary is incomplete")
    if fit_summary.get("run", {}).get("status") != "completed":
        raise ValueError("The fitted controller did not finish a complete native rollout")
    if fit_summary.get("control_count") != _CONTROL_COUNT:
        raise ValueError("The fitted controller summary must retain exactly 12 controls")
    if fit_summary.get("loss") is None or not np.isfinite(float(fit_summary["loss"])):
        raise ValueError("The fitted controller summary must contain a finite saved loss")

    baseline_summary = json.loads((baseline_directory / "summary.json").read_text())
    validate_sources(baseline_summary)
    controller_identity = _verify_starting_controller_identity(
        starting_controller, baseline_directory, baseline_summary, profile
    )
    single_report = _validation(
        single_validation,
        baseline_directory,
        mixed=False,
        expected_controls=_CONTROL_COUNT,
        contact_evidence=contact_evidence,
    )
    mixed_report = _validation(
        mixed_validation,
        baseline_directory,
        mixed=True,
        expected_controls=_CONTROL_COUNT,
    )
    if mixed_report.get("batch_size") != _WORLD_COUNT:
        raise ValueError("The mixed-world qualification must use exactly 128 worlds")

    config = Config(**fit_summary["simulation_config"])
    settings = FitConfig(**fit_summary["fit_config"])
    if settings.control_count != _CONTROL_COUNT:
        raise ValueError("The fitted controller must keep a 12-point measured objective")

    lower = np.asarray(profile["equilibrium_lower"], dtype=np.float64)
    upper = np.asarray(profile["equilibrium_upper"], dtype=np.float64)
    box_span = upper - lower
    if box_span.shape != (_CHANNEL_COUNT,) or not np.isfinite(box_span).all() or np.any(box_span <= 0.0):
        raise ValueError("Profile equilibrium bounds must define four finite positive box spans")

    probe_specs = _probe_specs(random_direction_count)
    total_probes = len(probe_specs)
    random_direction_metadata = {
        "seed": seed,
        "count": random_direction_count,
        "generation": "Warp GPU randn weights, constant/linear/sine1/sine2 basis, global relative L-infinity normalization",
        "shape": [random_direction_count, _CONTROL_COUNT, _CHANNEL_COUNT],
        "directions_copied_to_host": False,
    }
    probe_source_hash = _sha(Path(__file__))

    source_before = source_snapshot()
    output.mkdir(parents=True)
    (output / "reference.npz").write_bytes((fit_directory / "reference.npz").read_bytes())
    (output / "profile.json").write_bytes((fit_directory / "profile.json").read_bytes())

    identity = _identity_from_summary(fit_summary, output / "reference.npz", profile)
    optimization_inputs = {
        "schema": "cartesian_gpu_local_probe_inputs_1",
        "probe_source_sha256": probe_source_hash,
        "command": sys.argv,
        "run_directory": str(run_directory.resolve()),
        "baseline_directory": str(baseline_directory.resolve()),
        "fit_directory": str(fit_directory.resolve()),
        "output_directory": str(output.resolve()),
        "starting_controller": str(starting_controller.resolve()),
        "starting_controller_sha256": _sha(starting_controller),
        "single_validation": str(single_validation.resolve()),
        "mixed_validation": str(mixed_validation.resolve()),
        "contact_rounding_evidence": str(contact_evidence.resolve()),
        "single_validation_sha256": _sha(single_validation),
        "mixed_validation_sha256": _sha(mixed_validation),
        "contact_rounding_evidence_sha256": _sha(contact_evidence),
        "simulation_config": asdict(config),
        "fit_config": asdict(settings),
        "execution_identity": execution_identity(),
        "source_sha256": source_before,
        "source_audit": validate_sources(fit_summary),
        "identity": identity,
        "controller_identity_matches_baseline": controller_identity == identity,
        "probe_plan": {
            "world_count": _WORLD_COUNT,
            "axis_scale_percent": [100.0 * value for value in _AXIS_SCALE_FRACTIONS],
            "random_direction_count": random_direction_count,
            "total_probes": total_probes,
            "baseline_probes": 1,
            "axis_probes": _CONTROL_COUNT * _CHANNEL_COUNT * 2 * len(_AXIS_SCALE_FRACTIONS),
            "random_probes": random_direction_count * 2 * len(_AXIS_SCALE_FRACTIONS),
            "box_span": box_span.tolist(),
            "fixed_gains": {
                key: profile[key]
                for key in (
                    "hip_stiffness_n_m",
                    "joint_stiffness_nm_rad",
                    "hip_damping_ns_m",
                    "joint_damping_nms_rad",
                )
            },
        },
    }
    _write_json(output / "optimization_inputs.json", optimization_inputs)

    engine_started = perf_counter()
    engine = Engine(
        reference,
        profile,
        fit_summary["shoe"]["path"],
        fit_summary["shoe"]["mount_m"],
        fit_summary["shoe"]["static_pitch_rad"],
        config=config,
        settings=settings,
        world_count=_WORLD_COUNT,
        friction_model=fit_summary["shoe"]["friction_model"],
    )
    if engine.shoe.metadata != fit_summary["shoe"]:
        raise ValueError("GPU shoe metadata changed from the fitted controller bundle")
    repeated_seed = np.repeat(coefficients[None], _WORLD_COUNT, axis=0)
    engine.capture(repeated_seed)
    capture_warmup_steps = int(engine.integrated.numpy().sum())
    capture_wall_s = perf_counter() - engine_started

    engine.foundation.enabled.fill_(1)
    baseline_started = perf_counter()
    engine.evaluate_device()
    baseline_loss_all = engine.objective.loss.numpy()
    baseline_rmse_all = engine.objective.rmse.numpy()
    baseline_maximum_error_all = engine.objective.maximum_error.numpy()
    baseline_costs_all = engine.objective.costs.numpy()
    baseline_integrated_all = engine.integrated.numpy()
    baseline_failure_all = engine.failure.numpy()
    baseline_failure_step_all = engine.failure_step.numpy()
    baseline_range_step_all = engine.range_step.numpy()
    baseline_range_mask_all = engine.range_mask.numpy()
    baseline_wall_s = perf_counter() - baseline_started
    if not np.allclose(baseline_loss_all, baseline_loss_all[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("Repeated baseline worlds did not rescore identically")
    baseline_loss = float(baseline_loss_all[0])
    if not np.isclose(baseline_loss, float(fit_summary["loss"]), rtol=0.0, atol=1.0e-12):
        raise ValueError("Baseline rescoring differs from the saved fitted loss")
    if baseline_failure_all[0] != 0 or baseline_integrated_all[0] != engine.steps:
        raise ValueError("Saved fitted controller no longer completes the qualified native rollout")
    baseline_record = _result_record(
        probe_specs[0],
        batch_index=0,
        slot_index=0,
        bounded=True,
        integrated_steps=int(baseline_integrated_all[0]),
        failure_code=int(baseline_failure_all[0]),
        failure_step=int(baseline_failure_step_all[0]),
        range_step=int(baseline_range_step_all[0]),
        range_mask=int(baseline_range_mask_all[0]),
        loss=baseline_loss,
        rmse=baseline_rmse_all[0],
        maximum_error=baseline_maximum_error_all[0],
        costs=baseline_costs_all[0],
        time_s=engine.time_s,
    )

    device = engine.device
    constraints = SplineConstraints(
        float(reference["time_s"][-1]),
        profile["equilibrium_lower"],
        profile["equilibrium_upper"],
        profile["equilibrium_rate_limit"],
        profile["equilibrium_acceleration_limit"],
        device=device,
    )
    baseline_wp = wp.array(np.ascontiguousarray(coefficients), dtype=wp.float64, device=device)
    box_span_wp = wp.array(np.ascontiguousarray(box_span), dtype=wp.float64, device=device)
    random_direction_wp = wp.empty(
        (random_direction_count, _CONTROL_COUNT, _CHANNEL_COUNT), dtype=wp.float64, device=device
    )
    wp.launch(
        _random_relative_directions, dim=random_direction_count, inputs=[seed, random_direction_wp], device=device
    )
    kind_code_wp = wp.array(
        np.array([spec["kind_code"] for spec in probe_specs], dtype=np.int32), dtype=wp.int32, device=device
    )
    coord_index_wp = wp.array(
        np.array([spec["coord_index"] for spec in probe_specs], dtype=np.int32), dtype=wp.int32, device=device
    )
    random_index_wp = wp.array(
        np.array(
            [-1 if spec["random_direction_index"] is None else spec["random_direction_index"] for spec in probe_specs],
            dtype=np.int32,
        ),
        dtype=wp.int32,
        device=device,
    )
    sign_code_wp = wp.array(
        np.array([spec["sign"] for spec in probe_specs], dtype=np.int32), dtype=wp.int32, device=device
    )
    scale_fraction_wp = wp.array(
        np.array([spec["scale_fraction"] for spec in probe_specs], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    bounded_wp = wp.zeros(_WORLD_COUNT, dtype=wp.int32, device=device)
    best_coefficients_wp = wp.array(np.ascontiguousarray(coefficients), dtype=wp.float64, device=device)

    batch_count = (total_probes + _WORLD_COUNT - 1) // _WORLD_COUNT
    probe_records: list[dict[str, Any] | None] = [None] * total_probes
    probe_records[0] = baseline_record
    best_loss = baseline_loss
    best_probe_index = 0
    best_record = baseline_record
    counts = {
        "batches": batch_count,
        "slots": batch_count * _WORLD_COUNT,
        "attempted_probes": total_probes,
        "baseline_probes": 1,
        "axis_probes": _CONTROL_COUNT * _CHANNEL_COUNT * 2 * len(_AXIS_SCALE_FRACTIONS),
        "random_probes": random_direction_count * 2 * len(_AXIS_SCALE_FRACTIONS),
        "bounded_probes": 0,
        "evaluated_probes": 0,
        "completed_probes": 0,
        "failed_probes": 0,
        "bound_rejections": 0,
        "padding_slots": 0,
        "actually_integrated_steps": 0,
        "unique_measured_probes": None,
        "unique_controllers_tracked": False,
    }
    batch_wall_s: list[float] = []

    probe_started = perf_counter()
    for batch_index in range(batch_count):
        batch_start = batch_index * _WORLD_COUNT
        world_slots = np.arange(batch_start, min(batch_start + _WORLD_COUNT, total_probes), dtype=np.int64)
        padding_slots = _WORLD_COUNT - len(world_slots)
        counts["padding_slots"] += int(padding_slots)
        wp.launch(
            _populate_probe_batch,
            dim=_WORLD_COUNT,
            inputs=[
                batch_start,
                total_probes,
                baseline_wp,
                box_span_wp,
                kind_code_wp,
                coord_index_wp,
                random_index_wp,
                sign_code_wp,
                scale_fraction_wp,
                random_direction_wp,
                engine.coefficients,
            ],
            device=device,
            record_tape=False,
        )
        constraints.evaluate_control_bounds_device(engine.coefficients, bounded_wp, device=device)
        wp.launch(_mask_padding, dim=_WORLD_COUNT, inputs=[batch_start, total_probes, bounded_wp], device=device)
        wp.copy(engine.foundation.enabled, bounded_wp)
        batch_started = perf_counter()
        engine.evaluate_device()
        loss = engine.objective.loss.numpy()
        rmse = engine.objective.rmse.numpy()
        maximum_error = engine.objective.maximum_error.numpy()
        costs_array = engine.objective.costs.numpy()
        integrated = engine.integrated.numpy()
        failure = engine.failure.numpy()
        failure_step = engine.failure_step.numpy()
        range_step = engine.range_step.numpy()
        range_mask = engine.range_mask.numpy()
        bounded = bounded_wp.numpy().astype(bool, copy=False)
        batch_wall_s.append(perf_counter() - batch_started)
        if np.any(integrated[~bounded] != 0):
            raise RuntimeError("Disabled local-probe slots integrated despite the foundation.enabled mask")

        counts["bounded_probes"] += int(bounded[world_slots - batch_start].sum())
        counts["evaluated_probes"] += int(bounded[world_slots - batch_start].sum())
        counts["bound_rejections"] += int((~bounded[world_slots - batch_start]).sum())
        counts["actually_integrated_steps"] += int(integrated.sum())

        for slot_index, probe_index in enumerate(world_slots):
            spec = probe_specs[int(probe_index)]
            record = _result_record(
                spec,
                batch_index=batch_index,
                slot_index=slot_index,
                bounded=bool(bounded[slot_index]),
                integrated_steps=int(integrated[slot_index]),
                failure_code=int(failure[slot_index]),
                failure_step=int(failure_step[slot_index]),
                range_step=int(range_step[slot_index]),
                range_mask=int(range_mask[slot_index]),
                loss=float(loss[slot_index]),
                rmse=rmse[slot_index],
                maximum_error=maximum_error[slot_index],
                costs=costs_array[slot_index],
                time_s=engine.time_s,
            )
            probe_records[int(probe_index)] = record
            if record["completed"]:
                counts["completed_probes"] += 1
                if record["loss"] is not None and float(record["loss"]) < best_loss - 1.0e-12:
                    best_loss = float(record["loss"])
                    best_probe_index = int(probe_index)
                    best_record = record
                    wp.launch(
                        _copy_probe_candidate,
                        dim=(_CONTROL_COUNT, _CHANNEL_COUNT),
                        inputs=[slot_index, engine.coefficients, best_coefficients_wp],
                        device=device,
                        record_tape=False,
                    )
            elif record["bounded"]:
                counts["failed_probes"] += 1

    total_probe_wall_s = perf_counter() - probe_started
    if any(record is None for record in probe_records):
        raise RuntimeError("One or more attempted probes were not recorded")
    if source_snapshot() != source_before or _sha(Path(__file__)) != probe_source_hash:
        raise ValueError("GPU runtime or probe sources changed during local probing")

    best_coefficients = best_coefficients_wp.numpy()
    if best_coefficients.shape != (_CONTROL_COUNT, _CHANNEL_COUNT):
        raise RuntimeError("Saved best controller has an unexpected shape")
    if not Spline(engine.duration, best_coefficients).bounds(
        lower, upper, profile["equilibrium_rate_limit"], profile["equilibrium_acceleration_limit"]
    ):
        raise RuntimeError("Winning probe violates canonical bounds")

    summary = {
        "schema": "cartesian_gpu_local_probe_1",
        "status": "completed",
        "complete": True,
        "accepted": False,
        "global_optimality_claimed": False,
        "local_optimality_claimed": False,
        "qualification": (
            "Finite bounded probes preserve the qualified measured objective and native timestep, "
            "but they cannot prove local or global optimality."
        ),
        "simulation_config": asdict(config),
        "fit_config": asdict(settings),
        "control_count": _CONTROL_COUNT,
        "shoe": fit_summary["shoe"],
        "source_sha256": source_before,
        "original_source_sha256": fit_summary.get("original_source_sha256", fit_summary["source_sha256"]),
        "source_audit": validate_sources(fit_summary),
        "execution_identity": execution_identity(),
        "identity": identity,
        "run_directory": str(run_directory.resolve()),
        "baseline": str(baseline_directory.resolve()),
        "fit_directory": str(fit_directory.resolve()),
        "starting_controller": str(starting_controller.resolve()),
        "starting_controller_sha256": _sha(starting_controller),
        "gpu_validation": {
            "single": str(single_validation.resolve()),
            "mixed": str(mixed_validation.resolve()),
            "contact": str(contact_evidence.resolve()),
            "single_sha256": _sha(single_validation),
            "mixed_sha256": _sha(mixed_validation),
            "contact_sha256": _sha(contact_evidence),
            "single_same_timestep_parity_passed": single_report.get("same_timestep_parity_passed"),
            "mixed_passed": mixed_report.get("passed"),
        },
        "loss": best_record["loss"],
        "initial_loss": baseline_loss,
        "wall_s": total_probe_wall_s,
        "loss_improvement": None if best_record["loss"] is None else baseline_loss - float(best_record["loss"]),
        "metrics": {
            "rmse": best_record["rmse"],
            "maximum_error": best_record["maximum_error"],
            "rmse_by_block": best_record["rmse_by_block"],
            "maximum_error_by_block": best_record["maximum_error_by_block"],
        },
        "objective_components": best_record["objective_components"],
        "run": {
            "status": best_record["status"],
            "integrated_steps": best_record["integrated_steps"],
            "failure": best_record["failure_reasons"],
            "joint_range_step": best_record["joint_range_step"],
            "joint_range_mask": best_record["joint_range_mask"],
            "joint_range_exceeded": best_record["joint_range_exceeded"],
        },
        "best_probe": best_record,
        "counts": {
            **counts,
            "failed_or_incomplete_probes": counts["failed_probes"],
            "evaluated_probes_equal_bounded_probes": True,
            "unique_measured_probe_policy": "Not coefficient-level deduplicated; completed probe slots are not unique-controller counts.",
            "baseline_rescore_slots": _WORLD_COUNT,
            "baseline_rescore_integrated_world_steps": int(baseline_integrated_all.sum()),
            "capture_warmup_integrated_world_steps": capture_warmup_steps,
            "total_integrated_world_steps": counts["actually_integrated_steps"]
            + int(baseline_integrated_all.sum())
            + capture_warmup_steps,
        },
        "timings_s": {
            "engine_initialization_and_capture": capture_wall_s,
            "baseline_rescore": baseline_wall_s,
            "probe_batches_total": total_probe_wall_s,
            "probe_batches": batch_wall_s,
            "total": capture_wall_s + baseline_wall_s + total_probe_wall_s,
        },
        "probe_plan": {
            "axis_scale_percent": [100.0 * value for value in _AXIS_SCALE_FRACTIONS],
            "random_direction_count": random_direction_count,
            "total_probes": total_probes,
            "world_count": _WORLD_COUNT,
            "box_span": box_span.tolist(),
        },
        "controller_identity_matches_baseline": controller_identity == identity,
    }

    summary["total_wall_s"] = perf_counter() - started
    summary["timings_s"]["total"] = summary["total_wall_s"]
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "probes.json",
        {
            "schema": "cartesian_gpu_local_probe_records_1",
            "seed": seed,
            "axis_scale_percent": [100.0 * value for value in _AXIS_SCALE_FRACTIONS],
            "box_span": box_span.tolist(),
            "random_direction_library": random_direction_metadata,
            "probe_records": probe_records,
        },
    )
    _write_npz(
        output / "probe_arrays.npz",
        baseline_coefficients=coefficients,
        best_coefficients=best_coefficients,
        random_seed=seed,
        loss=np.array(
            [np.nan if record["loss"] is None else record["loss"] for record in probe_records], dtype=np.float64
        ),
        bounded=np.array([record["bounded"] for record in probe_records], dtype=np.bool_),
        completed=np.array([record["completed"] for record in probe_records], dtype=np.bool_),
        integrated_steps=np.array([record["integrated_steps"] for record in probe_records], dtype=np.int32),
        failure_code=np.array(
            [-1 if record["failure_code"] is None else record["failure_code"] for record in probe_records],
            dtype=np.int32,
        ),
        kind_code=np.array([record["kind_code"] for record in probe_records], dtype=np.int32),
        coord_index=np.array([record["coord_index"] for record in probe_records], dtype=np.int32),
        random_direction_index=np.array(
            [
                -1 if record["random_direction_index"] is None else record["random_direction_index"]
                for record in probe_records
            ],
            dtype=np.int32,
        ),
        sign=np.array([record["sign"] for record in probe_records], dtype=np.int32),
        scale_fraction=np.array([record["scale_fraction"] for record in probe_records], dtype=np.float64),
    )
    _write_npz(
        output / "equilibrium.npz",
        duration_s=float(reference["time_s"][-1]),
        coefficients=best_coefficients,
        identity_json=json.dumps(identity, sort_keys=True),
    )
    readme_lines = [
        "# Local controller probe",
        "",
        f"- Run directory: `{run_directory.resolve()}`",
        f"- Starting fitted loss: `{baseline_loss:.15g}`",
        f"- Best measured loss: `{best_loss:.15g}`",
        f"- Best probe index: `{best_probe_index}` ({best_record['kind']})",
        f"- Attempted probes: `{counts['attempted_probes']}`",
        f"- Bounded/evaluated probes: `{counts['bounded_probes']}`",
        f"- Completed probes: `{counts['completed_probes']}`",
        f"- Failed bounded probes: `{counts['failed_probes']}`",
        f"- Bound rejections: `{counts['bound_rejections']}`",
        f"- Padding slots: `{counts['padding_slots']}`",
        f"- Actually integrated steps: `{counts['actually_integrated_steps']}`",
        f"- Probe wall time [s]: `{total_probe_wall_s:.6f}`",
        "",
        "`equilibrium.npz` keeps the best completed bounded probe with warm-start identity metadata.",
        "Use it only as a warm start for later bounded polishing if needed.",
        "",
        "Finite bounded probes can diagnose some local sensitivity.",
        "They cannot prove local optimality or global optimality.",
    ]
    (output / "README.md").write_text("\n".join(readme_lines) + "\n")
    print(json.dumps(_plain(summary), indent=2, allow_nan=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for the local probe CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--random-direction-count", type=int, default=_RANDOM_DIRECTION_COUNT)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the reusable frozen local-neighborhood probe."""
    args = build_parser().parse_args(argv)
    probe_local_neighborhood(
        args.run_directory,
        args.output,
        seed=args.seed,
        random_direction_count=args.random_direction_count,
    )


if __name__ == "__main__":
    main()
