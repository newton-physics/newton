# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark orchestrator.

Discovers benchmark modules, runs each in a subprocess, saves
version-keyed results to scripts/bench/results/<git-hash>/.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import warp as wp


BENCHMARKS_PKG = "scripts.bench.benchmarks"
RESULTS_ROOT = Path("scripts/bench/results")

# Benchmark modules in execution order.
BENCHMARK_NAMES = ["scaling", "components", "accuracy", "timeline", "initial_conditions"]


def _git_short_hash() -> str:
    """Return 7-char git hash of HEAD, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _gpu_info() -> dict:
    """Collect GPU metadata."""
    info = {"device": "unknown", "warp": wp.__version__}
    try:
        dev = wp.get_device("cuda:0")
        info["device"] = dev.name
    except Exception:
        pass
    return info


def _run_benchmark_subprocess(
    bench_name: str,
    args: dict,
    out_dir: Path,
) -> tuple[float, dict | None]:
    """Run a benchmark in a subprocess. Returns (duration_s, data_dict)."""
    # Build CLI args for the benchmark module.
    cmd = [sys.executable, "-m", f"{BENCHMARKS_PKG}.{bench_name}"]
    cmd.extend(["--out-dir", str(out_dir)])

    if "ns" in args and bench_name in ("scaling", "components", "initial_conditions"):
        cmd.append("--ns")
        cmd.extend(str(n) for n in args["ns"])
    if "steps" in args and bench_name in ("scaling", "components", "initial_conditions"):
        cmd.extend(["--steps", str(args["steps"])])
    if "warmup" in args and bench_name in ("scaling", "components", "initial_conditions"):
        cmd.extend(["--warmup", str(args["warmup"])])
    if "trials" in args and bench_name in ("accuracy", "initial_conditions"):
        cmd.extend(["--trials", str(args["trials"])])
    if "tols" in args and bench_name == "initial_conditions":
        cmd.append("--tols")
        cmd.extend(str(t) for t in args["tols"])
    if "seed" in args and bench_name == "initial_conditions":
        cmd.extend(["--seed", str(args["seed"])])

    print(f"\n{'=' * 60}", flush=True)
    print(f"Running benchmark: {bench_name}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"{'=' * 60}", flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=False,  # Stream output to terminal.
        text=True,
        timeout=1800,  # 30 min max.
    )
    duration = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})", flush=True)
        return duration, None

    # Read the JSON data file that the benchmark wrote.
    json_path = out_dir / f"{bench_name}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return duration, data

    return duration, None


def run(
    only: str | None = None,
    skip: list[str] | None = None,
    args: dict | None = None,
) -> Path:
    """Run benchmarks, save results. Returns output directory path."""
    if args is None:
        args = {}
    if skip is None:
        skip = []

    commit = _git_short_hash()
    out_dir = RESULTS_ROOT / commit
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Determine which benchmarks to run.
    to_run = BENCHMARK_NAMES
    if only:
        to_run = [only]
    to_run = [b for b in to_run if b not in skip]

    print(f"Benchmark run: commit={commit}  benchmarks={to_run}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    # Run each benchmark.
    meta_benchmarks = {}
    for bench_name in to_run:
        duration, data = _run_benchmark_subprocess(bench_name, args, out_dir)
        status = "ok" if data is not None else "failed"
        meta_benchmarks[bench_name] = {
            "status": status,
            "duration_s": round(duration, 1),
        }

        # Generate plots from the saved JSON data.
        if data is not None:
            try:
                mod = importlib.import_module(f"{BENCHMARKS_PKG}.{bench_name}")
                mod.plot(data, plots_dir)
            except Exception as e:
                print(f"  Plot generation failed: {e}", flush=True)

    # Write meta.json.
    meta = {
        "commit": commit,
        "branch": _git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_gpu_info(),
        "args": args,
        "benchmarks": meta_benchmarks,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta saved -> {out_dir / 'meta.json'}", flush=True)

    # Summary.
    print(f"\n{'=' * 60}")
    print(f"All benchmarks complete. Results in: {out_dir}")
    for name, info in meta_benchmarks.items():
        print(f"  {name}: {info['status']} ({info['duration_s']:.1f}s)")
    print(f"{'=' * 60}", flush=True)

    return out_dir


def list_benchmarks() -> None:
    """Print available benchmarks."""
    print("Available benchmarks:")
    for name in BENCHMARK_NAMES:
        try:
            mod = importlib.import_module(f"{BENCHMARKS_PKG}.{name}")
            doc = (mod.__doc__ or "").strip().split("\n")[0]
        except ImportError:
            doc = "(import failed)"
        print(f"  {name:15s}  {doc}")
