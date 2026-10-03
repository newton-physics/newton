# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Time friction-patch construction on immutable contacts, without collision or solving.

Use ``--input snapshot.npz`` to replay captured construction inputs, or omit it
for generated fragmented contact tables (not a simulated scene). Run against
each checkout using PYTHONPATH.
CUDA graph event timings include preparation, sorting, and patch construction;
``build_with_reset_ms`` measures the grouping/history kernel plus input reset.
Compilation and warmup are excluded. ``--output`` also saves patch outputs for
exact comparison across revisions. This is not a physics or end-to-end benchmark.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.friction_patches import _build, _FrictionPatchState

MODEL_DTYPES = {
    "body_world": int,
    "shape_body": int,
    "shape_collision_radius": float,
    "shape_transform": wp.transform,
    "shape_scale": wp.vec3,
    "shape_type": int,
    "shape_source_ptr": wp.uint64,
    "shape_margin": float,
    "shape_gap": float,
    "shape_is_solid": bool,
    "shape_material_mu": float,
}
CONTACT_DTYPES = {
    "rigid_contact_count": int,
    "rigid_contact_shape0": int,
    "rigid_contact_shape1": int,
    "rigid_contact_point0": wp.vec3,
    "rigid_contact_point1": wp.vec3,
    "rigid_contact_normal": wp.vec3,
    "rigid_contact_margin0": float,
    "rigid_contact_margin1": float,
}


def snapshot(model, state, contacts):
    """Copy construction inputs; mesh assets are not needed for replay."""
    data = {"model_" + name: getattr(model, name).numpy() for name in MODEL_DTYPES}
    # Pointers are not dereferenced by construction and cannot survive replay.
    data["model_shape_source_ptr"] = np.zeros_like(data["model_shape_source_ptr"])
    data.update({"contacts_" + name: getattr(contacts, name).numpy() for name in CONTACT_DTYPES})
    data["body_q"] = state.body_q.numpy()
    return data


def generated(count, groups):
    """Interleave planar regions on one body pair, with deterministic input order."""
    rng = np.random.default_rng(1927)
    region = np.arange(count) % groups
    points = rng.uniform(-0.02, 0.02, (count, 3)).astype(np.float32)
    points[:, 2] = region * 0.01
    return {
        "model_body_world": np.array([0, 0], np.int32),
        "model_shape_body": np.array([0, 1], np.int32),
        "model_shape_collision_radius": np.full(2, 0.2, np.float32),
        "model_shape_transform": np.tile([0, 0, 0, 0, 0, 0, 1], (2, 1)).astype(np.float32),
        "model_shape_scale": np.ones((2, 3), np.float32),
        "model_shape_type": np.zeros(2, np.int32),
        "model_shape_source_ptr": np.zeros(2, np.uint64),
        "model_shape_margin": np.zeros(2, np.float32),
        "model_shape_gap": np.zeros(2, np.float32),
        "model_shape_is_solid": np.ones(2, bool),
        "model_shape_material_mu": np.full(2, 0.5, np.float32),
        "body_q": np.tile([0, 0, 0, 0, 0, 0, 1], (2, 1)).astype(np.float32),
        "contacts_rigid_contact_count": np.array([count], np.int32),
        "contacts_rigid_contact_shape0": np.zeros(count, np.int32),
        "contacts_rigid_contact_shape1": np.ones(count, np.int32),
        "contacts_rigid_contact_point0": points,
        "contacts_rigid_contact_point1": points.copy(),
        "contacts_rigid_contact_normal": np.tile([0, 0, -1], (count, 1)).astype(np.float32),
        "contacts_rigid_contact_margin0": np.zeros(count, np.float32),
        "contacts_rigid_contact_margin1": np.zeros(count, np.float32),
    }


def run(data, device, iterations, history):
    model = SimpleNamespace(device=wp.get_device(device), body_count=len(data["body_q"]))
    for name, dtype in MODEL_DTYPES.items():
        setattr(model, name, wp.array(data["model_" + name], dtype=dtype, device=device))
    contacts = SimpleNamespace()
    for name, dtype in CONTACT_DTYPES.items():
        setattr(contacts, name, wp.array(data["contacts_" + name], dtype=dtype, device=device))
    state = SimpleNamespace(body_q=wp.array(data["body_q"], dtype=wp.transform, device=device))
    capacity = len(data["contacts_rigid_contact_shape0"])
    patches = _FrictionPatchState(model, capacity, True, wp.zeros(capacity, dtype=wp.vec2, device=device))
    patches.build(model, state, contacts)
    if history:
        patches.store(state)
    patches.build(model, state, contacts)
    outputs = {
        name: getattr(patches.view, name).numpy() for name in ("weight", "next_contact", "point_a", "point_b", "phi")
    }
    outputs.update(
        {
            name: getattr(patches.current, name).numpy()
            for name in ("owner", "valid", "source", "displacement", "tangent_impulse")
        }
    )
    result = {
        "contacts": int(data["contacts_rigid_contact_count"][0]),
        "capacity": capacity,
        "anchors": int(np.count_nonzero(outputs["weight"])),
        "history": history,
    }
    if model.device.is_cuda:

        def measure(operation):
            # Each graph replay starts from a clean owner map and history usage.
            with wp.ScopedCapture(device=device) as capture:
                operation()
            for _ in range(10):
                wp.capture_launch(capture.graph)
            samples = []
            start = wp.Event(device=device, enable_timing=True)
            end = wp.Event(device=device, enable_timing=True)
            for _ in range(5):
                wp.record_event(start)
                for _ in range(iterations):
                    wp.capture_launch(capture.graph)
                wp.record_event(end)
                wp.synchronize_event(end)
                samples.append(wp.get_event_elapsed_time(start, end) / iterations)
            return samples

        result["construction_ms"] = measure(lambda: patches.build(model, state, contacts))
        # _build mutates owner/next_contact. Restore prepared inputs each replay;
        # report the reset-inclusive span explicitly instead of subtracting noise.
        patches.build(model, state, contacts)

        def build_only():
            patches.current.owner.fill_(-1)
            patches.view.next_contact.fill_(-1)
            wp.launch(
                _build,
                dim=capacity,
                inputs=[
                    contacts.rigid_contact_count,
                    state.body_q,
                    patches.previous_q,
                    model.shape_transform,
                    model.shape_collision_radius,
                    patches.current,
                    patches.previous,
                    patches.view,
                ],
                device=device,
            )

        result["build_with_reset_ms"] = measure(build_only)
    return result, outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contacts", type=int, default=1024)
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cold", action="store_true")
    args = parser.parse_args()
    if min(args.contacts, args.groups, args.iterations) < 1 or args.groups > args.contacts:
        parser.error("Require positive counts and groups <= contacts")
    data = dict(np.load(args.input)) if args.input else generated(args.contacts, args.groups)
    digest = hashlib.sha256()
    for name in sorted(data):
        digest.update(name.encode())
        digest.update(data[name].tobytes())
    result, outputs = run(data, args.device, args.iterations, not args.cold)
    result["input_sha256"] = digest.hexdigest()
    result["device"] = str(wp.get_device(args.device))
    result["source_sha256"] = hashlib.sha256(Path(inspect.getfile(_FrictionPatchState)).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez(args.output.with_suffix(".npz"), **outputs)
    print(json.dumps(result))
