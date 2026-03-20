# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test registered examples for browser switch & reset compatibility.

This script requires the GL viewer and must be run manually (not via pytest).
No functions use the ``test_`` prefix, so pytest will not collect any test cases.

Iterates through examples returned by ``newton.examples.get_examples()``,
attempts to instantiate each one (as the example browser would) using the GL
viewer, runs N frames of step + render, and then resets it.  Exceptions are
caught and logged so the full suite runs to completion.

Usage:
    uv run python newton/tests/test_example_browser.py                     # all examples, 1 frame
    uv run python newton/tests/test_example_browser.py "mpm_*" --frames 10 # mpm examples, 10 frames
    uv run python newton/tests/test_example_browser.py "robot_h1" "cloth_*" --frames 5
"""

import argparse
import fnmatch
import importlib
import sys
import time
import traceback

import warp as wp

import newton
import newton.examples
import newton.viewer

wp.init()

SKIP_EXAMPLES = {
    "robot_policy",  # non-standard constructor: (viewer, config, asset_directory, mjc_to_physx, physx_to_mjc)
}


def _step_and_render(example, num_frames):
    for _ in range(num_frames):
        if hasattr(example, "step"):
            example.step()
        if hasattr(example, "render"):
            example.render()


def main():
    parser = argparse.ArgumentParser(description="Smoke-test example browser switch & reset.")
    parser.add_argument(
        "patterns", nargs="*", default=["*"], help="Wildcard patterns to match example names (default: all)"
    )
    parser.add_argument(
        "--frames", "-n", type=int, default=1, help="Number of frames to step/render per example (default: 1)"
    )
    cli_args = parser.parse_args()

    example_map = newton.examples.get_examples()
    create_parser = newton.examples.create_parser
    default_args = newton.examples.default_args

    matched = {
        name: mod
        for name, mod in sorted(example_map.items())
        if name not in SKIP_EXAMPLES and any(fnmatch.fnmatch(name, p) for p in cli_args.patterns)
    }

    if not matched:
        print(f"No examples matched patterns: {cli_args.patterns}")
        return 1

    viewer = newton.viewer.ViewerGL()

    results: list[dict] = []
    total = len(matched)

    print(f"Running {total} example(s), {cli_args.frames} frame(s) each\n", flush=True)

    for i, (name, module_path) in enumerate(matched.items(), 1):
        entry = {"name": name, "module": module_path, "switch": None, "reset": None}
        print(f"[{i}/{total}] {name} ({module_path})", flush=True)

        # --- switch (instantiate from scratch) ---
        try:
            viewer.clear_model()
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
            viewer.clear_model()
            example_class = type(example)
            ex_parser = getattr(example_class, "create_parser", create_parser)()
            args = default_args(ex_parser)
            t0 = time.perf_counter()
            example2 = example_class(viewer, args)
            _step_and_render(example2, cli_args.frames)
            dt = time.perf_counter() - t0
            entry["reset"] = "OK"
            print(f"  reset:  OK ({dt:.2f}s)", flush=True)
        except Exception:
            entry["reset"] = traceback.format_exc()
            print(f"  reset:  FAIL\n{entry['reset']}", flush=True)

        results.append(entry)

    # --- summary ---
    switch_ok = sum(1 for r in results if r["switch"] == "OK")
    reset_ok = sum(1 for r in results if r["reset"] == "OK")
    switch_fail = [r for r in results if r["switch"] != "OK"]
    reset_fail = [r for r in results if r["reset"] not in ("OK", None)]

    print("\n" + "=" * 70, flush=True)
    print(f"RESULTS: {switch_ok}/{total} switch OK, {reset_ok}/{total} reset OK")
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
        print("\nAll examples passed!")

    return 1 if (switch_fail or reset_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
