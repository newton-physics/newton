# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnose CUDA contact arithmetic at saved CPU simulation states, without integration.

Run from the repository root with the native uv environment. Only saved simulated
states drive this diagnostic. Recorded reference motion is never replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from projects.digital_shoe import runtime
from projects.impedance_instron.cartesian.gpu import engine
from projects.impedance_instron.cartesian.gpu.mechanics import Params, Vec5
from projects.impedance_instron.cartesian.mechanics import Body

from ..shoe import Shoe

# This option belongs only to this diagnostic's copy/store kernels.
wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.kernel
def stage(
    clock: wp.array[int],
    saved_q: wp.array[wp.transform],
    saved_qd: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    body_q[0] = saved_q[clock[0]]
    body_qd[0] = saved_qd[clock[0]]


@wp.kernel
def store(
    clock: wp.array[int],
    body_f: wp.array[wp.spatial_vector],
    compression: wp.array[float],
    wrench_history: wp.array[wp.spatial_vector],
    compression_history: wp.array2d[float],
):
    c = wp.tid()
    s = clock[0]
    compression_history[s, c] = compression[c]
    if c == 0:
        wrench_history[s] = body_f[0]


@wp.kernel
def tick(clock: wp.array[int]):
    clock[0] = clock[0] + 1


def sha256(path):
    """Identify each input and source without modifying it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path):
    """Load an identified saved archive without executing a simulation."""
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def exact_cpu_poses(body, trace, static_pitch):
    """Reproduce Body.point/angle and the exact float32 Shoe.apply construction."""
    q = np.empty((len(trace["state"]), 7), dtype=np.float32)
    qd = np.empty((len(q), 6), dtype=np.float32)
    angular_jacobian = body.angular_jacobian(2)
    ankle_local = np.zeros(2)
    for i, (state, velocity) in enumerate(zip(trace["state"], trace["velocity"], strict=True)):
        ankle, jacobian, _ = body.point(state, 2, ankle_local)
        velocity_m_s = jacobian @ velocity
        pitch_rad = body.angle(state, 2)
        angular_velocity_rad_s = float(angular_jacobian @ velocity)
        angle = pitch_rad - static_pitch
        position = np.array([[ankle[0], 0.0, ankle[1], 0.0, -np.sin(angle / 2), 0.0, np.cos(angle / 2)]])
        spatial_velocity = np.array([[velocity_m_s[0], 0.0, velocity_m_s[1], 0.0, -angular_velocity_rad_s, 0.0]])
        q[i] = position.astype(np.float32)[0]
        qd[i] = spatial_velocity.astype(np.float32)[0]
    return q, qd


def prepare_saved_states(reference, profile, summary, trace, device):
    """Evaluate the actual engine _prepare kernel on independent saved states only.

    Each saved timestep is an independent slot. The controller uses one exact
    saved equilibrium row per slot through a unit basis. Pose staging is the
    unmodified _prepare code. No Engine instance or _advance launch exists here.
    """
    n = len(trace["state"])
    params = Params()
    params.lengths = wp.vec2d(*reference["lengths_m"])
    params.masses = wp.vec3d(*profile["masses_kg"])
    params.com0 = wp.vec2d(*profile["com_local_m"][0])
    params.com1 = wp.vec2d(*profile["com_local_m"][1])
    params.com2 = wp.vec2d(*profile["com_local_m"][2])
    params.inertias = wp.vec3d(*profile["inertias_kg_m2"])
    cfg = engine._Settings()
    cfg.stiffness = wp.vec4d(*profile["hip_stiffness_n_m"], *profile["joint_stiffness_nm_rad"])
    cfg.damping = wp.vec4d(*profile["hip_damping_ns_m"], *profile["joint_damping_nms_rad"])
    cfg.lower = wp.vec2d(*profile["joint_lower_rad"])
    cfg.upper = wp.vec2d(*profile["joint_upper_rad"])
    config = summary["simulation_config"]
    cfg.dt = summary["run"]["actual_dt_s"]
    cfg.gravity = config["gravity_m_s2"]
    cfg.pitch = summary["shoe"]["static_pitch_rad"]
    cfg.hip_floor = config["minimum_hip_height_m"]
    cfg.max_speed = config["maximum_speed"]
    cfg.max_force = config["maximum_force_n"]
    cfg.compression_limit = config["compression_limit"]
    cfg.joint_diagnostic = int(config["joint_limits_diagnostic"])
    cfg.steps, cfg.controls = 1, 1
    clock = wp.zeros(1, dtype=int, device=device)
    basis = wp.ones((1, 1), dtype=wp.float64, device=device)
    coefficients = wp.array(trace["equilibrium"][:, None, :], dtype=wp.float64, device=device)
    states = wp.array(trace["state"][None, :, :], dtype=Vec5, device=device)
    velocities = wp.array(trace["velocity"][None, :, :], dtype=Vec5, device=device)
    eq = wp.empty((1, n), dtype=wp.vec4d, device=device)
    actuator = wp.empty_like(eq)
    q = wp.empty(n, dtype=wp.transform, device=device)
    qd = wp.empty(n, dtype=wp.spatial_vector, device=device)
    failure = wp.zeros(n, dtype=int, device=device)
    failure_step = wp.full(n, -1, dtype=int, device=device)
    range_step = wp.full(n, -1, dtype=int, device=device)
    range_mask = wp.zeros(n, dtype=int, device=device)
    wp.launch(
        engine._prepare,
        dim=n,
        inputs=[
            params,
            cfg,
            clock,
            basis,
            coefficients,
            states,
            velocities,
            eq,
            actuator,
            q,
            qd,
            failure,
            failure_step,
            range_step,
            range_mask,
        ],
        device=device,
        block_dim=32,
    )
    flags = failure.numpy()
    if np.any(flags):
        raise RuntimeError(f"Saved state _prepare screen failed: {np.unique(flags)}")
    return q.numpy(), qd.numpy()


def stats(actual, expected, times):
    """Describe every saved row, including exact first difference and largest error."""
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    flat = np.abs(delta).reshape(len(delta), -1)
    rows = np.max(flat, axis=1)
    changed = np.flatnonzero(rows != 0)
    peak = int(np.argmax(rows))
    return {
        "max_absolute_error": float(np.max(rows)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_row": peak,
        "max_time_s": float(times[peak]),
        "max_row_actual": np.asarray(actual[peak]).tolist(),
        "max_row_expected": np.asarray(expected[peak]).tolist(),
        "first_different_row": int(changed[0]) if len(changed) else None,
        "first_different_time_s": float(times[changed[0]]) if len(changed) else None,
        "different_rows": len(changed),
        "per_component_max_absolute_error": np.max(flat, axis=0).tolist(),
    }


def compare(actual, cpu, times):
    """Compare all force, moment, and compression summaries available in the CPU trace."""
    keys = (
        "grf_n",
        "ankle_contact_moment_nm",
        "compression_fraction",
        "driven_compression_fraction",
        "passive_compression_fraction",
        "passive_cap_column_count",
    )
    result = {key: stats(actual[key], cpu[key], times) for key in keys}
    force_error = np.linalg.norm(actual["grf_n"] - cpu["grf_n"], axis=1)
    peak = int(np.argmax(force_error))
    result["force_vector_n"] = {
        "max_absolute_error": float(force_error[peak]),
        "max_row": peak,
        "rmse": float(np.sqrt(np.mean(force_error * force_error))),
    }
    return result


def replay(shoe, q, qd, dt, chunk_steps):
    """Replay one contact history with all poses uploaded once and no step readbacks."""
    n = len(q)
    device = shoe.device
    saved_q = wp.array(q, dtype=wp.transform, device=device)
    saved_qd = wp.array(qd, dtype=wp.spatial_vector, device=device)
    clock = wp.zeros(1, dtype=int, device=device)
    wrenches = wp.empty(n, dtype=wp.spatial_vector, device=device)
    compression = wp.empty((n, shoe.foundation.column_count), dtype=float, device=device)
    # Refresh the constant host-to-device material coefficients before capture.
    shoe.foundation._refresh_surround_constants(dt)
    wp.load_module(module=runtime, device=device)
    wp.load_module(module=__name__, device=device)

    def step():
        wp.launch(stage, dim=1, inputs=[clock, saved_q, saved_qd, shoe.state.body_q, shoe.state.body_qd], device=device)
        shoe.foundation.apply(shoe.state, dt, clear_body_force=True)
        wp.launch(
            store,
            dim=shoe.foundation.column_count,
            inputs=[clock, shoe.state.body_f, shoe.foundation.compression, wrenches, compression],
            device=device,
        )
        wp.launch(tick, dim=1, inputs=[clock], device=device)

    started = perf_counter()
    with wp.ScopedCapture(device=device) as capture:
        for _ in range(min(chunk_steps, n)):
            step()
    tail_graph = None
    tail = n % chunk_steps if n >= chunk_steps else 0
    if tail:
        with wp.ScopedCapture(device=device) as tail_capture:
            for _ in range(tail):
                step()
        tail_graph = tail_capture.graph
    capture_s = perf_counter() - started
    # One explicit history reset before this replay. Shoe also resets at construction.
    shoe.foundation.reset()
    clock.zero_()
    started = perf_counter()
    chunks = max(1, n // chunk_steps)
    for _ in range(chunks):
        wp.capture_launch(capture.graph)
    if tail_graph is not None:
        wp.capture_launch(tail_graph)
    # These are reporting-boundary copies after all contact steps are queued.
    result = {"wrench_newton": wrenches.numpy(), "compression_m": compression.numpy()}
    result["clock"] = int(clock.numpy()[0])
    result["capture_wall_s"] = capture_s
    result["replay_and_readback_wall_s"] = perf_counter() - started
    result["chunk_launches"] = chunks
    result["tail_steps"] = tail
    if result["clock"] != n:
        raise RuntimeError("Resident contact clock does not match the saved trace length")
    return result


def main(argv=None):
    """Write reproducible contact-only evidence without fitting or integrating the leg."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--full-gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    summary = json.loads((args.baseline / "summary.json").read_text())
    profile = json.loads((args.baseline / "profile.json").read_text())
    reference_archive = archive(args.baseline / "reference.npz")
    # Only geometry constants come from reference.npz; no recorded motion is used.
    reference = {key: reference_archive[key] for key in ("lengths_m", "endpoint_local_m")}
    cpu = archive(args.baseline / "trace.npz")
    full = archive(args.full_gpu / "trace.npz")
    previous = json.loads((args.full_gpu / "benchmark.json").read_text())
    if not np.array_equal(cpu["time_s"], full["time_s"]):
        raise ValueError("Saved CPU/GPU traces have different time grids")
    source_identity_path = args.full_gpu.parent / "physical_source_identity.json"
    source_identity = json.loads(source_identity_path.read_text())
    for path, identity in source_identity.items():
        if sha256(path) != identity["baseline"]:
            raise ValueError(f"Physical source changed from frozen CPU identity: {path}")
    shoe_info = summary["shoe"]
    if sha256(shoe_info["path"]) != shoe_info["sha256"]:
        raise ValueError("Shoe artifact identity changed")
    body = Body(
        reference["lengths_m"],
        reference["endpoint_local_m"],
        profile["masses_kg"],
        profile["com_local_m"],
        profile["inertias_kg_m2"],
    )
    started = perf_counter()
    q, qd = exact_cpu_poses(body, cpu, shoe_info["static_pitch_rad"])
    full_q, full_qd = exact_cpu_poses(body, full, shoe_info["static_pitch_rad"])
    pose_precompute_s = perf_counter() - started
    shoe = Shoe(
        shoe_info["path"],
        shoe_info["mount_m"],
        shoe_info["static_pitch_rad"],
        device="cuda:0",
        friction_model=shoe_info["friction_model"],
    )
    if shoe.metadata != shoe_info:
        raise ValueError("CUDA Shoe metadata differs from frozen CPU Shoe")
    runtime_options = wp.get_module_options(module=runtime)
    if runtime_options["fuse_fp"] is not True:
        raise ValueError("This diagnostic requires the unchanged default runtime fuse_fp=True")
    prepared_q, prepared_qd = prepare_saved_states(reference, profile, summary, cpu, shoe.device)
    print(
        json.dumps(
            {"phase": "contact_replay", "saved_simulation_rows": len(q), "runtime_fuse_fp": runtime_options["fuse_fp"]}
        ),
        flush=True,
    )
    result = replay(shoe, q, qd, summary["run"]["actual_dt_s"], 32)
    driven = shoe.foundation.driven.numpy().astype(bool)
    fraction = result["compression_m"] / shoe.shoe.column_bed.rest_length_m
    driven_max = np.max(fraction[:, driven], axis=1)
    passive_max = np.max(fraction[:, ~driven], axis=1)
    actual = {
        "grf_n": result["wrench_newton"][:, [0, 2]].astype(float),
        "ankle_contact_moment_nm": -result["wrench_newton"][:, 4].astype(float),
        "compression_fraction": np.maximum(driven_max, passive_max),
        "driven_compression_fraction": driven_max,
        "passive_compression_fraction": passive_max,
        "passive_cap_column_count": np.count_nonzero(
            fraction[:, ~driven] >= float(shoe.foundation.surround.max_strain) - 1.0e-6, axis=1
        ),
    }
    times = cpu["time_s"]
    contact_error = compare(actual, cpu, times)
    full_error = compare(full, cpu, times)
    errors_ratio = {
        key: contact_error[key]["max_absolute_error"] / full_error[key]["max_absolute_error"]
        for key in ("force_vector_n", "ankle_contact_moment_nm", "compression_fraction")
        if full_error[key]["max_absolute_error"] != 0
    }
    checks = {
        key: stats(prepared, exact, times)
        for key, prepared, exact in (("body_q", prepared_q, q), ("body_qd", prepared_qd, qd))
    }
    state_pose_error = {
        key: stats(saved_gpu, saved_cpu, times)
        for key, saved_gpu, saved_cpu in (("body_q", full_q, q), ("body_qd", full_qd, qd))
    }
    important_rows = sorted(
        set(
            [512]
            + [contact_error[key]["max_row"] for key in ("force_vector_n", "ankle_contact_moment_nm")]
            + [full_error[key]["max_row"] for key in ("force_vector_n", "ankle_contact_moment_nm")]
        )
    )
    row_details = {}
    for row in important_rows:
        row_details[str(row)] = {
            "time_s": float(times[row]),
            "cpu_grf_n": cpu["grf_n"][row].tolist(),
            "replay_grf_n": actual["grf_n"][row].tolist(),
            "full_gpu_grf_n": full["grf_n"][row].tolist(),
            "cpu_moment_nm": float(cpu["ankle_contact_moment_nm"][row]),
            "replay_moment_nm": float(actual["ankle_contact_moment_nm"][row]),
            "full_gpu_moment_nm": float(full["ankle_contact_moment_nm"][row]),
            "cpu_pose": q[row].tolist(),
            "full_gpu_state_cpu_pose": full_q[row].tolist(),
            "engine_prepare_pose": prepared_q[row].tolist(),
        }
    report = {
        "schema": "cartesian_saved_simulation_contact_replay_1",
        "scope": "Contact-only replay of saved CPU simulation states, not prescribed recorded reference motion",
        "leg_integration_executed": False,
        "optimizer_started": False,
        "tests_executed": False,
        "limits_changed": False,
        "full_rollout_parity_fixed": False,
        "device": str(shoe.device),
        "gpu_name": shoe.device.name,
        "warp_version": wp.__version__,
        "runtime_fuse_fp": runtime_options["fuse_fp"],
        "engine_fuse_fp": wp.get_module_options(module=engine)["fuse_fp"],
        "saved_rows": len(q),
        "actual_dt_s": summary["run"]["actual_dt_s"],
        "column_count": shoe.foundation.column_count,
        "chunk_steps": 32,
        "poses_uploaded_once": True,
        "per_step_readbacks": 0,
        "history_resets_per_replay": 1,
        "initialization_note": "Shoe constructor also resets its fresh foundation; no warm contact steps run",
        "resident_final_clock": result["clock"],
        "chunk_launches": result["chunk_launches"],
        "tail_steps": result["tail_steps"],
        "pose_precompute_wall_s": pose_precompute_s,
        "capture_wall_s": result["capture_wall_s"],
        "replay_and_readback_wall_s": result["replay_and_readback_wall_s"],
        "compression_comparison_scope": "All saved per-step overall/driven/passive maxima and cap counts; "
        "the CPU trace has no per-column compression history. Raw CUDA column histories are saved.",
        "exact_saved_cpu_pose_contact_error": contact_error,
        "saved_full_gpu_error": full_error,
        "contact_only_to_full_gpu_max_error_ratio": errors_ratio,
        "engine_prepare_at_saved_cpu_states_vs_exact_cpu_float32": checks,
        "saved_full_gpu_states_vs_saved_cpu_states_cpu_pose_construction": state_pose_error,
        "diagnostic_rows": row_details,
        "unchanged_provisional_limits_from_saved_benchmark": previous["same_timestep_limits"],
        "saved_full_gpu_parity_passed": previous["same_timestep_parity_passed"],
        "inputs_sha256": {
            str(path.resolve()): sha256(path)
            for path in (
                args.baseline / "trace.npz",
                args.baseline / "summary.json",
                args.baseline / "profile.json",
                args.baseline / "reference.npz",
                args.full_gpu / "trace.npz",
                args.full_gpu / "benchmark.json",
                Path(shoe_info["path"]),
            )
        },
        "sources_sha256": {path: sha256(path) for path in source_identity},
        "diagnostic_sha256": sha256(__file__),
        "engine_source_sha256": sha256(engine.__file__),
    }
    np.savez_compressed(
        args.output / "histories.npz",
        time_s=times,
        exact_cpu_body_q=q,
        exact_cpu_body_qd=qd,
        engine_prepare_body_q=prepared_q,
        engine_prepare_body_qd=prepared_qd,
        saved_full_gpu_state_cpu_body_q=full_q,
        saved_full_gpu_state_cpu_body_qd=full_qd,
        wrench_newton=result["wrench_newton"],
        compression_m=result["compression_m"],
        **actual,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ratios": errors_ratio,
                "contact_errors": {key: value["max_absolute_error"] for key, value in contact_error.items()},
                "prepare_errors": {key: value["max_absolute_error"] for key, value in checks.items()},
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
