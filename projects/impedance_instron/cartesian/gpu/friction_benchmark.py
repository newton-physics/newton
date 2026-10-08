# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Paired, warmed CUDA-graph timing of the same configured foundation/friction law."""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from projects.digital_shoe.friction_parameter_adapter import FrictionParameterAdapter
from projects.digital_shoe.runtime import FoundationConfig, ShoeMaterial, SurroundConfig
from projects.impedance_instron.cartesian.gpu.foundation import FoundationFused

_DT = 0.000125
_RTOL = 2e-6


def _build(worlds, device, law, fused):
    """Build an asset-free 35x26 bed and retain pristine device-side reset copies."""
    x, y = np.indices((35, 26)).reshape(2, -1)
    anchors = np.column_stack(((x - 17) * 0.005, (y - 12.5) * 0.005, np.full(910, -0.02)))
    neighbors = np.column_stack(
        (
            np.where(x > 0, x * 26 + y - 26, -1),
            np.where(x < 34, x * 26 + y + 26, -1),
            np.where(y > 0, x * 26 + y - 1, -1),
            np.where(y < 25, x * 26 + y + 1, -1),
        )
    )
    material = ShoeMaterial(74000.0, 0.22, 0.7, 0.0, instantaneous_shear_modulus_2_pa=18000.0, hyperfoam_exponent_2=3.1)
    config = FoundationConfig(ground_height_m=0.0, friction_model=law, friction_stiffness=1e4, friction=10.0, mu=0.8)
    surround = SurroundConfig(driven=(abs(x - 17) <= 12) & (abs(y - 12.5) <= 8), sweeps=8, carrier_bond=True)
    state = SimpleNamespace(
        body_q=wp.array(np.tile([0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 1.0], (worlds, 1)), dtype=wp.transform),
        body_qd=wp.array(np.tile([0.4, -0.08, 0.0, 0.3, -0.2, 0.7], (worlds, 1)), dtype=wp.spatial_vector),
        body_f=wp.zeros(worlds, dtype=wp.spatial_vector),
    )
    geometry = anchors, np.zeros(910), 0.02 + 0.002 * np.cos(x / 34 * np.pi), np.full(910, 0.005**2), neighbors, 0.005
    com = wp.zeros(worlds, dtype=wp.vec3)
    foundation = FoundationFused(
        *geometry, material, np.arange(worlds), com, config, device, surround, world_count=worlds
    )
    foundation.fused_apply = fused
    adapter = foundation.friction_solver
    if type(adapter) is not FrictionParameterAdapter or not adapter.is_default:
        raise RuntimeError("The configured default friction adapter must be active")
    np.testing.assert_array_equal(
        adapter.settings.numpy()[:, 0], {"maxwell": 7, "column_maxwell": 8, "elastic_coulomb": 9}[law]
    )
    foundation._refresh_surround_constants(_DT)
    arrays = {
        f"{prefix}.{name}": arr
        for prefix, obj in (("foundation", foundation), ("friction", adapter), ("state", state))
        for name, arr in vars(obj).items()
        if isinstance(arr, wp.array)
    }
    return foundation, state, arrays, {name: wp.clone(arr) for name, arr in arrays.items()}


def _reset(entry):
    """Restore all arrays, not just the histories cleared by the production reset."""
    for name, arr in entry[2].items():
        wp.copy(arr, entry[3][name])


def _parity(pair):
    """Reject nonfinite/inactive results and compare every resident array and full wrench."""
    errors = {}
    for name, arr in pair[0][2].items():
        reference, actual = arr.numpy(), pair[1][2][name].numpy()
        if reference.dtype.fields or reference.dtype.kind in "iu":
            np.testing.assert_array_equal(actual, reference, err_msg=name)
        else:
            if not (np.isfinite(reference).all() and np.isfinite(actual).all()):
                raise RuntimeError(f"Nonfinite array: {name}")
            atol = 1e-9 if "tangent" in name or name.startswith("friction.") else 1e-7
            np.testing.assert_allclose(actual, reference, rtol=_RTOL, atol=atol, err_msg=name)
            error = float(np.max(np.abs(actual - reference)))
            if error:
                errors[name] = error
    wrench = pair[1][1].body_f.numpy()
    normal, tangent = wrench[:, 2], np.linalg.norm(wrench[:, :2], axis=1)
    if not (np.all(normal > 0.0) and np.all(tangent > 0.0)):
        raise RuntimeError("Every world must have active normal and tangential forces")
    return {
        "passed": True,
        "arrays_verified": sorted(pair[0][2]),
        "nonzero_max_abs_errors": errors,
        "tolerances": {"rtol": _RTOL, "atol_history": 1e-9, "atol_other": 1e-7},
        "min_normal_n": float(normal.min()),
        "min_tangential_n": float(tangent.min()),
    }


def _benchmark(worlds, device, args):
    """Warm both step-only graphs, then alternate reset-and-time pairs."""
    pair = [_build(worlds, device, args.friction_model, fused) for fused in (False, True)]
    for name, arr in pair[0][3].items():
        np.testing.assert_array_equal(arr.numpy(), pair[1][3][name].numpy(), err_msg=f"initial {name}")
    graphs, launches = [], []
    for entry in pair:
        foundation, state = entry[:2]
        foundation.apply(state, _DT, clear_body_force=True)
        with (
            patch.object(wp, "launch", wraps=wp.launch) as plain,
            patch.object(wp, "launch_tiled", wraps=wp.launch_tiled) as tiled,
        ):
            foundation.apply(state, _DT, clear_body_force=True)
        tiled_keys = [(c.args[0] if c.args else c.kwargs["kernel"]).key for c in tiled.call_args_list]
        plain_keys = [(c.args[0] if c.args else c.kwargs["kernel"]).key for c in plain.call_args_list]
        # Some Warp versions route tiled launches through the public launch alias.
        keys = tiled_keys + [key for key in plain_keys if key not in tiled_keys]
        expected = 3 if foundation.fused_apply else 10
        fused_keys = ["_surround_fused", "_apply_contact_world", "_reduce_contact_world"]
        if len(keys) != expected or (foundation.fused_apply and keys != fused_keys):
            raise RuntimeError(f"Unexpected warmed kernel launches: {keys}")
        launches.append({"fused_apply": foundation.fused_apply, "count": len(keys), "keys": keys})
        _reset(entry)
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(args.steps):
                foundation.apply(state, _DT, clear_body_force=True)
        graphs.append(capture.graph)
    for _ in range(3):
        for entry, graph in zip(pair, graphs, strict=True):
            _reset(entry)
            wp.capture_launch(graph)
    parity = _parity(pair)
    start, end = (wp.Event(device=device, enable_timing=True) for _ in range(2))
    elapsed = np.zeros((args.repeats, 2))
    orders = []
    for repeat in range(args.repeats):
        order = (0, 1) if repeat % 2 == 0 else (1, 0)
        orders.append([bool(i) for i in order])
        for index in order:
            _reset(pair[index])
            wp.synchronize_device(device)
            wp.record_event(start)
            wp.capture_launch(graphs[index])
            wp.record_event(end)
            wp.synchronize_event(end)
            elapsed[repeat, index] = wp.get_event_elapsed_time(start, end, synchronize=False)
        parity = _parity(pair)
    if not (np.isfinite(elapsed).all() and np.all(elapsed > 0.0)):
        raise RuntimeError("Invalid CUDA event timings")
    return {
        "worlds": worlds,
        "warm_launches": launches,
        "parity": parity,
        "order_fused_apply": orders,
        "timings_ms_partial_full": elapsed.tolist(),
        "batch_step_us_partial_full": (elapsed * 1000 / args.steps).tolist(),
        "median_batch_step_us_partial_full": (np.median(elapsed, axis=0) * 1000 / args.steps).tolist(),
        "median_paired_speedup": float(np.median(elapsed[:, 0] / elapsed[:, 1])),
        "world_steps_per_second_partial_full": (worlds * args.steps * 1000 / np.median(elapsed, axis=0)).tolist(),
    }


def _hashes():
    """Fingerprint this benchmark, foundation, and all imported shoe/friction dependencies."""
    paths = {Path(__file__).resolve(), Path(sys.modules[FoundationFused.__module__].__file__).resolve()}
    paths.update(
        Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if name.startswith("projects.digital_shoe") and getattr(module, "__file__", None)
    )
    root = Path(__file__).resolve().parents[4]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def main():
    """Run the standalone benchmark and optionally create a new JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--friction-model", choices=("maxwell", "column_maxwell", "elastic_coulomb"), default="elastic_coulomb"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0", help="CUDA device alias (default: cuda:0)")
    args = parser.parse_args()
    if min(*args.worlds, args.steps, args.repeats) < 1:
        parser.error("worlds, steps and repeats must be positive")
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error(f"Refusing to overwrite {args.output}")
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        parser.error("CUDA is required for paired graph/event timing")
    hashes = _hashes()
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "warp": wp.__version__,
        "numpy": np.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(device),
        "gpu": {"name": device.name, "arch": device.arch, "uuid": device.uuid},
        "cuda": {"driver": wp.get_cuda_driver_version(), "toolkit": wp.get_cuda_toolkit_version()},
        "source_sha256": hashes,
        "execution_options": {
            "warp_mode": wp.config.mode,
            "foundation": {
                key: wp.get_module_options(sys.modules[FoundationFused.__module__])[key]
                for key in ("enable_backward", "fuse_fp", "fast_math", "mode", "optimization_level")
            },
        },
        "friction_model": args.friction_model,
        "dt_s": _DT,
        "steps": args.steps,
        "repeats": args.repeats,
        "bed": {"columns": 910, "grid": [35, 26], "spacing_m": 0.005, "driven": 400, "passive": 510, "sweeps": 8},
        "friction_config": {"friction_stiffness": 1e4, "friction": 10.0, "mu": 0.8},
        "warm_graph_replays": 3,
        "timing_caveat": "Shared desktop GPU; clocks, power limits and other GPU activity are not controlled. "
        "Small differences are not evidence of a portable or end-to-end speedup.",
        "timing_scope": "CUDA events around one warmed steps-only graph replay on the same dedicated stream; "
        "includes foundation.apply(clear_body_force=True), surround, configured default friction and diagnostics. "
        "Excludes compilation, capture, full-array reset, host reads/parity, controller diagnostics and rigid integration. "
        "Fixed loaded pose (z=0.015 m, anchors z=-0.02 m); prescribed linear velocity [0.4,-0.08,0] m/s "
        "and angular velocity [0.3,-0.2,0.7] rad/s. Independent pristine state restored before each replay.",
    }
    with wp.ScopedDevice(device), wp.ScopedStream(wp.Stream(device)):
        report["results"] = [_benchmark(worlds, device, args) for worlds in args.worlds]
    if hashes != _hashes():
        raise RuntimeError("Source files changed during the benchmark; discard timings and rerun")
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
