# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark rolling loaded contacts with identical settings across revisions.

Run, for example::

    uv run --extra dev python -m newton.tests.benchmarks.fpgs_contact_rolling \
        --kind foot --worlds 1024 --objects 1 --reduced --iterations 64

Requires CUDA. Timings include external-load and physical-telemetry kernels in
addition to collision and solver work. Every graph replay advances the scene;
these are fixed-budget measurements, not a matched-quality speedup claim.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests.test_contact_reduction_body_pairs import _cylinder_foot

DT = 1.0 / 240.0


@wp.kernel
def load_bodies(mass: wp.array[float], force: wp.array[wp.spatial_vector]):
    body = wp.tid()
    # An additional downward load and a sub-Coulomb horizontal load.
    force[body] = wp.spatial_vector(0.1 * 9.81 * mass[body], 0.0, -9.81 * mass[body], 0.0, 0.0, 0.0)


@wp.kernel
def measure(
    pose: wp.array[wp.transform],
    velocity: wp.array[wp.spatial_vector],
    initial: wp.array[wp.transform],
    metrics: wp.array[float],
):
    body = wp.tid()
    position = wp.transform_get_translation(pose[body])
    origin = wp.transform_get_translation(initial[body])
    linear = wp.spatial_top(velocity[body])
    angular = wp.spatial_bottom(velocity[body])
    if (
        not wp.isfinite(wp.length(position))
        or not wp.isfinite(wp.length(linear))
        or not wp.isfinite(wp.length(angular))
    ):
        wp.atomic_max(metrics, 5, 1.0)
    else:
        wp.atomic_max(metrics, 0, origin[2] - position[2])
        wp.atomic_max(metrics, 1, position[2] - origin[2])
        wp.atomic_max(metrics, 2, wp.length(wp.vec2(position[0] - origin[0], position[1] - origin[1])))
        wp.atomic_max(metrics, 3, wp.length(linear))
        wp.atomic_max(metrics, 4, wp.length(angular))


def run(worlds, objects, kind, warm, reduced, iterations, friction):
    """Advance one loaded scene and return timing and physical measurements."""
    template = newton.ModelBuilder()
    template.default_shape_cfg.mu = 0.6
    for obj in range(objects):
        if kind == "foot":
            _cylinder_foot(template, wp.vec3(0.0, obj * 0.3, 0.015))
        else:
            body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1 + obj * 0.201), wp.quat_identity()))
            template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds, spacing=(1.0, 0.0, 0.0))
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    initial = wp.clone(state_in.body_q)
    metrics = wp.zeros(6, dtype=wp.float32, device=model.device)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_warmstart=warm,
        pgs_iterations=iterations,
        friction_mode=friction,
        mf_max_constraints=max(128, objects * 128),
    )
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching="latest" if warm else "disabled",
        reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=reduced),
    )
    contacts, control = pipeline.contacts(), model.control()

    def step(record=False):
        nonlocal state_in, state_out
        wp.launch(load_bodies, model.body_count, inputs=[model.body_mass, state_in.body_f], device=model.device)
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, DT)
        if record:
            wp.launch(
                measure,
                model.body_count,
                inputs=[state_out.body_q, state_out.body_qd, initial, metrics],
                device=model.device,
            )
        state_in, state_out = state_out, state_in

    for _ in range(100):
        step()
    wp.synchronize()
    # One writer graph owns this stateful pipeline. No eager solve follows it.
    with wp.ScopedCapture() as capture:
        solver.seed_double_buffer_events()
        step(True)
        step(True)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(40):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / 80)
    if hasattr(solver, "check_constraint_capacity"):
        solver.check_constraint_capacity()
    observed = metrics.numpy().tolist()
    counts = solver.mf_constraint_count.numpy()
    impulses = solver.mf_impulses.numpy()
    row_type = solver.mf_row_type.numpy()
    active = np.arange(impulses.shape[1])[None, :] < counts[:, None]
    normal = np.where(active & (row_type == 0), impulses, 0.0).sum(axis=1)
    expected = 2 * 9.81 * model.body_mass.numpy().reshape(worlds, objects).sum(axis=1) * DT
    result = {
        "worlds": worlds,
        "objects_per_world": objects,
        "kind": kind,
        "warm": warm,
        "reduced": reduced,
        "iterations": iterations,
        "friction": friction,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "metrics": dict(
            zip(
                ("max_sink_m", "max_rise_m", "max_drift_m", "max_linear_speed", "max_angular_speed", "nonfinite"),
                observed,
                strict=True,
            )
        ),
        "support_relative_error": float(np.max(np.abs(normal / expected - 1))) if kind == "foot" else None,
        "final_contacts": int(contacts.rigid_contact_count.numpy()[0]),
        "max_rows": int(counts.max()),
        "overflow": bool(solver.constraint_overflow.numpy().any()) if hasattr(solver, "constraint_overflow") else None,
        "warp": wp.__version__,
    }
    if reduced:
        result["reduction"] = pipeline.body_pair_reduction_stats()
    if warm:
        count = result["final_contacts"]
        result["final_matched_fraction"] = (
            float(np.mean(contacts.rigid_contact_match_index.numpy()[:count] >= 0)) if count else 0.0
        )
    print(json.dumps(result), flush=True)
    return result


def main():
    """Run repeatable, explicitly configured baseline comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("foot", "stack"), default="stack")
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--objects", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.worlds, args.objects, args.iterations, args.repeats) < 1:
        parser.error("worlds, objects, iterations and repeats must be positive")
    if not wp.is_cuda_available():
        parser.error("this benchmark requires CUDA")
    root = Path(newton.__file__).resolve().parents[1]
    sources = (
        "newton/_src/solvers/feather_pgs/solver_feather_pgs.py",
        "newton/_src/solvers/feather_pgs/kernels.py",
        "newton/_src/geometry/contact_match.py",
        "newton/_src/geometry/contact_reduction_body_pairs.py",
        "newton/_src/geometry/contact_reduction_global.py",
        "newton/_src/sim/collide.py",
        "newton/_src/sim/contacts.py",
    )
    report = {
        "source_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        "source_root": str(root),
        "device": wp.get_cuda_device().name,
        "warp": wp.__version__,
        "dt": DT,
        "warmup_steps": 100,
        "timed_steps": 400,
        "includes_load_and_quality_telemetry": True,
        "results": [],
    }
    for repeat in range(args.repeats):
        result = run(args.worlds, args.objects, args.kind, args.warm_start, args.reduced, args.iterations, "current")
        result["repeat"] = repeat
        report["results"].append(result)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.output:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
