# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test registered examples for browser switch & reset compatibility.

Run this script manually with the GL or Viser viewer.

Iterates through examples returned by ``newton.examples.get_examples()``,
attempts to instantiate each one (as the example browser would) using the selected
viewer, runs N frames of step + render, and then resets it.  Exceptions are
caught and logged so the full suite runs to completion.

Usage:
    uv run python newton/tests/test_example_browser.py                     # all examples, 1 frame
    uv run python newton/tests/test_example_browser.py "mpm_*" --frames 10 # mpm examples, 10 frames
    uv run python newton/tests/test_example_browser.py "robot_h1" "cloth_*" --frames 5
    uv run --extra notebook python newton/tests/test_example_browser.py --viewer viser "basic_*"
"""

import argparse
import fnmatch
import gc
import importlib
import sys
import time
import traceback

import warp as wp

import newton
import newton.examples
import newton.viewer

wp.init()

CUDA_REQUIRED_EXAMPLES = {
    "brick_stacking",
    "contacts_rj45_plug",
    "mujoco_sleeping",
    "nut_bolt_hydro",
    "nut_bolt_sdf",
    "robot_panda_hydro",
}


def _step_and_render(example, num_frames):
    viewer = example.viewer
    if hasattr(example, "gui") and hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(example.gui)
    for _ in range(num_frames):
        if viewer.should_step() and hasattr(example, "step"):
            example.step()
        if hasattr(example, "render"):
            example.render()


def _get_skip_reason(name, cuda_available):
    if not cuda_available and name in CUDA_REQUIRED_EXAMPLES:
        return "requires CUDA"
    return None


def main():
    parser = argparse.ArgumentParser(description="Smoke-test example browser switch & reset.")
    parser.add_argument(
        "patterns", nargs="*", default=["*"], help="Wildcard patterns to match example names (default: all)"
    )
    parser.add_argument(
        "--frames", "-n", type=int, default=1, help="Number of frames to step/render per example (default: 1)"
    )
    parser.add_argument("--viewer", choices=("gl", "viser"), default="gl", help="Viewer backend to exercise")
    cli_args = parser.parse_args()

    example_map = newton.examples.get_examples()
    create_parser = newton.examples.create_parser
    default_args = newton.examples.default_args

    selected = {
        name: module_path
        for name, module_path in sorted(example_map.items())
        if any(fnmatch.fnmatch(name, pattern) for pattern in cli_args.patterns)
    }

    if not selected:
        print(f"No examples matched patterns: {cli_args.patterns}")
        return 1

    cuda_available = wp.is_cuda_available()
    skipped = {name: reason for name in selected if (reason := _get_skip_reason(name, cuda_available)) is not None}
    matched = {name: module_path for name, module_path in selected.items() if name not in skipped}

    if skipped:
        print(f"Skipping {len(skipped)} example(s):")
        for name, reason in skipped.items():
            print(f"  {name}: {reason}")
        print()

    if not matched:
        print("No runnable examples matched.")
        return 0

    viewer = newton.viewer.ViewerViser() if cli_args.viewer == "viser" else newton.viewer.ViewerGL()

    results: list[dict] = []
    total = len(matched)

    print(f"Running {total} example(s), {cli_args.frames} frame(s) each\n", flush=True)

    for i, (name, module_path) in enumerate(matched.items(), 1):
        entry = {"name": name, "module": module_path, "switch": None, "reset": None}
        print(f"[{i}/{total}] {name} ({module_path})", flush=True)

        # --- switch (instantiate from scratch) ---
        try:
            viewer.clear_all_layers()
            mod = importlib.import_module(module_path)
            ex_parser = getattr(mod.Example, "create_parser", create_parser)()
            args = default_args(ex_parser)
            t0 = time.perf_counter()
            example = mod.Example(viewer, args)
            _step_and_render(example, cli_args.frames)
            dt = time.perf_counter() - t0
            entry["switch"] = "OK"
            print(f"  switch: OK ({dt:.2f}s)", flush=True)
        except Exception:
            entry["switch"] = traceback.format_exc()
            print(f"  switch: FAIL\n{entry['switch']}", flush=True)
            results.append(entry)
            continue

        # --- reset (re-instantiate same class) ---
        try:
            example_class = type(example)
            example = None
            viewer.clear_all_layers()
            gc.collect()
            ex_parser = getattr(example_class, "create_parser", create_parser)()
            args = default_args(ex_parser)
            t0 = time.perf_counter()
            example2 = example_class(viewer, args)
            _step_and_render(example2, cli_args.frames)
            example2 = None
            dt = time.perf_counter() - t0
            entry["reset"] = "OK"
            print(f"  reset:  OK ({dt:.2f}s)", flush=True)
        except Exception:
            entry["reset"] = traceback.format_exc()
            print(f"  reset:  FAIL\n{entry['reset']}", flush=True)

        results.append(entry)

    viewer.close()

    # --- summary ---
    switch_ok = sum(1 for r in results if r["switch"] == "OK")
    reset_ok = sum(1 for r in results if r["reset"] == "OK")
    switch_fail = [r for r in results if r["switch"] != "OK"]
    reset_fail = [r for r in results if r["reset"] not in ("OK", None)]

    print("\n" + "=" * 70, flush=True)
    print(f"RESULTS: {switch_ok}/{total} switch OK, {reset_ok}/{total} reset OK, {len(skipped)} skipped")
    print("=" * 70)

    if switch_fail:
        print(f"\n--- SWITCH FAILURES ({len(switch_fail)}) ---")
        for r in switch_fail:
            print(f"\n  {r['name']} ({r['module']}):")
            for line in r["switch"].strip().splitlines():
                print(f"    {line}")

    if reset_fail:
        print(f"\n--- RESET FAILURES ({len(reset_fail)}) ---")
        for r in reset_fail:
            print(f"\n  {r['name']} ({r['module']}):")
            for line in r["reset"].strip().splitlines():
                print(f"    {line}")

    if not switch_fail and not reset_fail:
        print("\nAll runnable examples passed!" if skipped else "\nAll examples passed!")

    return 1 if (switch_fail or reset_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
