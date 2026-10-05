# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Asset-free arithmetic benchmark; excludes assembly, velocity scatter and collisions.

Use --source to select the Newton checkout. CPU mode validates outputs only.
CUDA timings measure independent pairs, not serial PGS or whole-scene speed.

    uv run python asv/benchmarks/benchmark_feather_pgs_friction_pair.py --source .

Use --pairs 64, 4096 (default), or 65536 to compare small and large batches.
Compilation, warmup, reference checks and transfers are excluded from CUDA
measurements. Each launch reads immutable, seeded inputs, so sliding cannot
converge into sticking during timing. Alternating old/new graph batches reduce
order bias. JSON includes all samples, input/source hashes and reference error.
The baseline reproduces two scalar tangent visits with projection after each;
the corrected path imports the production Warp function. Both use precomputed
positive-definite blocks (condition numbers 1 to 8), unit friction radius and
unit relaxation. Native CUDA generation, block assembly and velocity scatter
are excluded. Amortized ns/pair measures throughput, not serial pair latency.
No claim about complete solver or scene speed follows from this benchmark.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("friction-pair.json"))
    args = parser.parse_args()
    if min(args.pairs, args.samples, args.launches) < 1:
        parser.error("pairs, samples and launches must be positive")
    sys.path.insert(0, str(args.source.resolve()))

from newton._src.solvers.feather_pgs.friction import friction_pair_candidate


@wp.func
def project(x: wp.vec2, radius: float):
    length = wp.length(x)
    if length > radius:
        x *= radius / length
    return x


@wp.kernel
def old_pair(blocks: wp.array[wp.vec3], residuals: wp.array[wp.vec2], old: wp.array[wp.vec2], out: wp.array[wp.vec2]):
    i = wp.tid()
    a = blocks[i]
    initial = old[i]
    x = initial
    # Algebraic equivalent of two scalar GS row visits with sibling projection.
    x[0] -= residuals[i][0] / a[0]
    x = project(x, 1.0)
    r1 = residuals[i][1] + a[1] * (x[0] - initial[0]) + a[2] * (x[1] - initial[1])
    x[1] -= r1 / a[2]
    out[i] = project(x, 1.0)


@wp.kernel
def corrected_pair(
    blocks: wp.array[wp.vec3], residuals: wp.array[wp.vec2], old: wp.array[wp.vec2], out: wp.array[wp.vec2]
):
    i = wp.tid()
    a = blocks[i]
    out[i] = project(friction_pair_candidate(a[0], a[1], a[2], residuals[i], old[i], 1.0, 1.0), 1.0)


def reference(block, residual, old):
    """Independent float64 solve in original coordinates, without eigenvectors."""
    a, c, d = block.astype(np.float64)
    matrix = np.array([[a, c], [c, d]])
    b = matrix @ old - residual
    x = np.linalg.solve(matrix, b)
    if np.linalg.norm(x) <= 1:
        return x
    lo, hi = 0.0, np.linalg.norm(b)
    for _ in range(80):
        mid = (lo + hi) / 2
        x = np.linalg.solve(matrix + mid * np.eye(2), b)
        if np.linalg.norm(x) > 1:
            lo = mid
        else:
            hi = mid
    return np.linalg.solve(matrix + hi * np.eye(2), b)


def run(args):
    """Validate and time the same independent tangent blocks in both methods."""
    started = time.perf_counter()
    wp.set_device(args.device)
    rng = np.random.default_rng(102)
    results = {}
    for case in ("sticking", "sliding", "mixed"):
        n = args.pairs
        angle = rng.uniform(-np.pi, np.pi, n)
        cosine, sine = np.cos(angle), np.sin(angle)
        eigenvalue = rng.uniform(1.0, 8.0, n)
        blocks = np.column_stack(
            (cosine**2 + eigenvalue * sine**2, (1 - eigenvalue) * cosine * sine, sine**2 + eigenvalue * cosine**2)
        ).astype(np.float32)
        old = rng.uniform(-0.15, 0.15, (n, 2)).astype(np.float32)
        directions = rng.uniform(-np.pi, np.pi, n)
        sliding = np.full(n, case == "sliding")
        if case == "mixed":
            sliding = rng.random(n) < 0.5
        target = np.column_stack((np.cos(directions), np.sin(directions))) * np.where(sliding, 2.0, 0.25)[:, None]
        delta = old - target
        residuals = np.column_stack(
            (
                blocks[:, 0] * delta[:, 0] + blocks[:, 1] * delta[:, 1],
                blocks[:, 1] * delta[:, 0] + blocks[:, 2] * delta[:, 1],
            )
        ).astype(np.float32)
        inputs = [wp.array(blocks, dtype=wp.vec3), wp.array(residuals, dtype=wp.vec2), wp.array(old, dtype=wp.vec2)]
        outputs = [wp.empty(n, dtype=wp.vec2), wp.empty(n, dtype=wp.vec2)]
        kernels = [old_pair, corrected_pair]
        for kernel, output in zip(kernels, outputs, strict=True):
            for _ in range(5):
                wp.launch(kernel, dim=n, inputs=inputs, outputs=[output])
        wp.synchronize()
        values = [output.numpy() for output in outputs]
        for value in values:
            assert np.isfinite(value).all()
            assert np.linalg.norm(value, axis=1).max() <= 1.000002
        indices = np.arange(min(n, 64))
        expected = np.array([reference(blocks[i], residuals[i], old[i]) for i in indices])
        error = float(np.max(np.abs(values[1][indices] - expected)))
        assert error < 2e-5, (case, error)
        result = {
            "reference_max_error": error,
            "sliding_pairs": int(sliding.sum()),
            "input_sha256": hashlib.sha256(blocks.tobytes() + residuals.tobytes() + old.tobytes()).hexdigest(),
        }
        if wp.get_device().is_cuda:
            graphs = []
            for kernel, output in zip(kernels, outputs, strict=True):
                with wp.ScopedCapture() as capture:
                    for _ in range(args.launches):
                        wp.launch(kernel, dim=n, inputs=inputs, outputs=[output])
                graphs.append(capture.graph)
            for graph in graphs:
                for _ in range(3):
                    wp.capture_launch(graph)
            wp.synchronize()
            timings = [[], []]
            for sample in range(args.samples):
                for index in (0, 1) if sample % 2 == 0 else (1, 0):
                    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
                    wp.record_event(begin)
                    wp.capture_launch(graphs[index])
                    wp.record_event(end)
                    wp.synchronize()
                    timings[index].append(wp.get_event_elapsed_time(begin, end) / args.launches)
            for name, samples in zip(("old", "corrected"), timings, strict=True):
                result[name] = {
                    "batch_ms": samples,
                    "median_batch_ms": float(np.median(samples)),
                    "amortized_ns_per_pair": float(np.median(samples) * 1e6 / n),
                }
            result["ratio"] = result["corrected"]["median_batch_ms"] / result["old"]["median_batch_ms"]
        results[case] = result
    source = args.source / "newton/_src/solvers/feather_pgs/friction.py"
    report = {
        "device": str(wp.get_device()),
        "device_name": wp.get_device().name,
        "warp": wp.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pairs": args.pairs,
        "samples": args.samples,
        "launches": args.launches,
        "wall_seconds": time.perf_counter() - started,
        "scope": "Arithmetic throughput only; precomputed 2x2 blocks, fixed normal load, omega=1; no assembly/scatter.",
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(args)
