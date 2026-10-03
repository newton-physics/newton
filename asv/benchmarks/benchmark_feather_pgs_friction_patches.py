# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare captured FeatherPGS collision and solve cost across friction implementations.

Run this same script against the PR's base and feature revision::

    uv run python asv/benchmarks/benchmark_feather_pgs_friction_patches.py --worlds 64 --tiles 4

Each one-kilogram body is assembled from ``tiles ** 2`` touching boxes. Its
footprint, total mass, and applied load stay fixed as contact density increases.
Both revisions use contact matching for a like-for-like comparison; persistent
patch anchors themselves do not require it. Compilation and warmup are excluded.

Use ``--frozen-contacts`` to generate contacts once and restore the same input
poses and velocities before every solve. Compare ``--beta 0`` with ``--beta 0.2``
on the same revision to isolate patch construction plus solving from collision
and trajectory differences. The JSON contact hash must match across that pair.
State restoration costs are included in both runs. This mode measures solver
cost on fixed contacts, not stability or end-to-end collision performance.

Use ``--profile-stages`` to collect a separate uncaptured CUDA activity timeline
after the throughput measurement, with warm starts independently controlled by
``--warmstart``. The timeline includes prepare/build/link/seed/finish/store kernel
names and copies. Native patch radix sort is measured separately with CUDA
events because it is absent from Warp's kernel activity list. Event spans include
any stream gaps within the native call; neither measure is host wall-clock cost.
``--tiles 13 --worlds 1`` is a synthetic 169-shape stress case, not a decomposed
gripper asset or evidence of any gripper's performance.
"""

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from unittest import mock

import numpy as np
import warp as wp

import newton


@wp.kernel
def _apply_load(body_f: wp.array[wp.spatial_vector]):
    i = wp.tid()
    body_f[i] = wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.03)


@contextmanager
def _time_patch_sort(patches):
    """Record CUDA event spans for native patch sorts outside the throughput measurement."""
    original = wp.utils.radix_sort_pairs
    spans = []

    def timed_sort(keys, values, count, *args, **kwargs):
        if not patches.view.enabled or keys.ptr != patches.current.keys.ptr:
            return original(keys, values, count, *args, **kwargs)
        start = wp.Event(device=keys.device, enable_timing=True)
        end = wp.Event(device=keys.device, enable_timing=True)
        with wp.ScopedDevice(keys.device):
            wp.record_event(start)
            result = original(keys, values, count, *args, **kwargs)
            wp.record_event(end)
        spans.append((start, end))
        return result

    with mock.patch.object(wp.utils, "radix_sort_pairs", timed_sort):
        yield spans


def run(
    worlds: int,
    tiles: int,
    beta: float | None,
    steps: int,
    *,
    warmstart: bool = False,
    profile_stages: bool = False,
    frozen_contacts: bool = False,
):
    """Measure an even number of captured steps and report row counts and motion."""
    if worlds < 1 or tiles < 1 or tiles > 16 or steps < 2 or steps % 2:
        raise ValueError("Require worlds >= 1, 1 <= tiles <= 16, and an even steps >= 2")
    template = newton.ModelBuilder()
    template.add_ground_plane()
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity()))
    h = 0.1 / tiles
    for x in range(tiles):
        for y in range(tiles):
            template.add_shape_box(
                body,
                hx=h,
                hy=h,
                hz=0.1,
                xform=wp.transform(wp.vec3(-0.1 + h + 2 * h * x, -0.1 + h + 2 * h * y, 0), wp.quat_identity()),
                cfg=newton.ModelBuilder.ShapeConfig(density=125.0, mu=0.5),
            )
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    model = builder.finalize(device="cuda:0")
    model.rigid_contact_max = worlds * tiles * tiles * 4
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=32,
        pgs_beta=0.05,
        pgs_warmstart=warmstart,
        mf_warmstart=warmstart,
        **({"friction_anchor_beta": beta} if beta is not None else {}),
        dense_max_constraints=32,
        mf_max_constraints=max(256, 12 * tiles * tiles),
        warn_constraint_overflow=True,
    )
    pipeline = newton.CollisionPipeline(
        model, rigid_contact_max=model.rigid_contact_max, broad_phase="nxn", contact_matching="latest"
    )
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()

    frozen_inputs = []
    contact_hash = None
    if frozen_contacts:
        pipeline.collide(s0, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        columns = []
        for name in ("shape0", "shape1", "point0", "point1", "normal", "offset0", "offset1", "margin0", "margin1"):
            values = getattr(contacts, "rigid_contact_" + name).numpy()[:count]
            columns.append(values.reshape(count, -1))
        packet = np.concatenate(columns, axis=1)
        order = np.lexsort(tuple(packet[:, i] for i in range(packet.shape[1])))
        contact_hash = hashlib.sha256(packet[order].tobytes()).hexdigest()
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            value = getattr(s0, name)
            if value is not None and value.size:
                frozen_inputs.append((name, wp.clone(value)))

    def step(source, dest):
        for name, snapshot in frozen_inputs:
            wp.copy(getattr(source, name), snapshot)
        source.clear_forces()
        wp.launch(_apply_load, dim=model.body_count, inputs=[source.body_f], device=model.device)
        if not frozen_contacts:
            pipeline.collide(source, contacts)
        solver.step(source, dest, control, contacts, 0.005)

    for _ in range(10):
        step(s0, s1)
        step(s1, s0)
    with wp.ScopedCapture(device=model.device) as capture:
        solver.seed_double_buffer_events()
        step(s0, s1)
        step(s1, s0)
    before = s0.body_q.numpy().copy()
    start = time.perf_counter()
    for _ in range(steps // 2):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    elapsed = time.perf_counter() - start
    after = s0.body_q.numpy()
    types = solver.mf_row_type.numpy()
    counts = solver.mf_constraint_count.numpy()
    normal_rows = sum(int(np.count_nonzero(types[w, : counts[w]] == 0)) for w in range(worlds))
    friction_rows = sum(int(np.count_nonzero(types[w, : counts[w]] == 2)) for w in range(worlds))
    result = {
        "newton": newton.__file__,
        "warp": wp.__version__,
        "worlds": worlds,
        "tiles": tiles,
        "beta": solver.friction_anchor_beta,
        "steps": steps,
        "warmstart": warmstart,
        "frozen_contacts": frozen_contacts,
        "contact_input_sha256": contact_hash,
        "milliseconds_per_step": elapsed * 1000 / steps,
        "contacts": int(contacts.rigid_contact_count.numpy()[0]),
        "normal_rows": normal_rows,
        "friction_rows": friction_rows,
        "max_translation": float(np.linalg.norm(after[:, :3] - before[:, :3], axis=1).max()),
        "max_quaternion_change": float(np.linalg.norm(after[:, 3:] - before[:, 3:], axis=1).max()),
        "finite": bool(np.isfinite(after).all()),
    }
    assert result["finite"]
    assert normal_rows > 0
    if profile_stages:
        # Keep instrumentation and its synchronization out of the captured
        # throughput measurement. Two steps exercise both state buffers.
        with _time_patch_sort(solver._friction_patches) as sort_spans:
            with wp.ScopedTimer("contact_stages", cuda_filter=wp.TIMING_ALL, print=False) as timer:
                step(s0, s1)
                step(s1, s0)
        result["patch_sort_cuda_event_milliseconds_per_step"] = (
            sum(wp.get_event_elapsed_time(start, end) for start, end in sort_spans) / 2
        )
        activity = {}
        for timing in timer.timing_results:
            entry = activity.setdefault(timing.name, {"calls_per_step": 0.0, "milliseconds_per_step": 0.0})
            entry["calls_per_step"] += 0.5
            entry["milliseconds_per_step"] += timing.elapsed * 0.5
        result["uncaptured_cuda_activity"] = activity
    print("RESULT " + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument(
        "--beta", type=float, default=None, help="Override the default patch gain; zero disables patches"
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmstart", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--frozen-contacts", action="store_true")
    args = parser.parse_args()
    run(
        args.worlds,
        args.tiles,
        args.beta,
        args.steps,
        warmstart=args.warmstart,
        profile_stages=args.profile_stages,
        frozen_contacts=args.frozen_contacts,
    )
