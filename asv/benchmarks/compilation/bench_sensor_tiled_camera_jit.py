# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""JIT compile-time benchmarks for the tiled camera render megakernel.

Measures how long the first renders of each scene from
``bench_sensor_tiled_camera.py`` take to JIT-compile. Each sample runs
in a fresh subprocess that clears the Warp kernel cache immediately before the
first render call and disables the CUDA driver's PTX cache, so the measurement
always includes full codegen, NVRTC compilation, and driver JIT of the render
megakernel (scene setup and model kernels compile beforehand and are excluded).

Run directly to print per-scene compile times::

    uv run asv/benchmarks/compilation/bench_sensor_tiled_camera_jit.py
"""

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import warp as wp

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

from asv_runner.benchmarks.mark import skip_benchmark_if

_SCENES_BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "simulation" / "bench_sensor_tiled_camera.py"
_RESULT_MARKER = "TILED_CAMERA_JIT_SECONDS="

# JIT time does not depend on world count or resolution; keep the rig small so
# the child process spends its time compiling, not building or rendering.
_JIT_WORLD_COUNT = 16
_JIT_RESOLUTION = 64


def _load_scenes_module():
    spec = importlib.util.spec_from_file_location("bench_sensor_tiled_camera", _SCENES_BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _child_main(scene: str) -> None:
    """Build the scene rig, then time the first (cold-cache) render calls.

    Times each output combination the render benchmarks warm up. Every new
    combination adds a kernel to the render module, so combinations after the
    first also recompile the previously created kernels; the reported total is
    the compile time a cold-cache render benchmark run actually pays.
    """
    scenes_module = _load_scenes_module()
    # The rig constructor does not render, so the megakernel stays uncompiled.
    rig = scenes_module._TiledCameraSceneRig(scenes_module.SCENES[scene], _JIT_WORLD_COUNT, _JIT_RESOLUTION)

    # Everything up to here (model finalize, BVH build, ray setup) has compiled
    # its own kernels. Clearing the cache now forces the megakernel - the only
    # kernel still uncompiled - to JIT from scratch, including any cached-on-disk
    # build from a previous run.
    wp.clear_lto_cache()
    wp.clear_kernel_cache()

    total = 0.0
    for label, (color, depth) in (
        ("color_depth", (True, True)),
        ("color_only", (True, False)),
        ("depth_only", (False, True)),
    ):
        start = time.perf_counter()
        rig.render(color=color, depth=depth)
        wp.synchronize()
        elapsed = time.perf_counter() - start
        total += elapsed
        print(f"{_RESULT_MARKER}{label}={elapsed}")

    print(f"{_RESULT_MARKER}total={total}")


def _measure_jit_times(scene: str) -> dict[str, float]:
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--scene", scene]
    # Warp emits PTX, which the CUDA driver JIT-compiles to SASS and caches in
    # ~/.nv/ComputeCache across processes. Disable that cache so every sample
    # pays the full compile, not just Warp's NVRTC step.
    env = {**os.environ, "CUDA_CACHE_DISABLE": "1"}
    result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
    times = {}
    for line in result.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            label, _, value = line[len(_RESULT_MARKER) :].partition("=")
            times[label] = float(value)
    if "total" not in times:
        raise RuntimeError(f"Child process produced no result marker:\n{result.stdout}\n{result.stderr}")
    return times


class _JitBenchmark:
    """Shared harness; subclasses pick a scene from the scene benchmark's SCENES."""

    warmup_time = 0
    repeat = 2
    number = 1
    timeout = 600
    unit = "s"
    scene: str

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_jit_compile(self):
        """Total cold-cache compile time of all measured render output combinations [s]."""
        return _measure_jit_times(self.scene)["total"]


class JitTiledCameraQuadruped(_JitBenchmark):
    scene = "quadruped"


class JitTiledCameraFrankaCabinet(_JitBenchmark):
    scene = "franka_cabinet"


class JitTiledCameraShapes256(_JitBenchmark):
    scene = "shapes_256"


BENCHMARKS = {
    cls.scene: cls
    for cls in (JitTiledCameraQuadruped, JitTiledCameraFrankaCabinet, JitTiledCameraShapes256)
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "-s",
        "--scene",
        default=None,
        action="append",
        choices=sorted(BENCHMARKS),
        help="Scene to benchmark; may be repeated. Defaults to all scenes.",
    )
    args = parser.parse_known_args()[0]

    if args.child:
        (scene,) = args.scene
        _child_main(scene)
    else:
        from newton.utils import run_benchmark

        for name in args.scene or sorted(BENCHMARKS):
            run_benchmark(BENCHMARKS[name])
