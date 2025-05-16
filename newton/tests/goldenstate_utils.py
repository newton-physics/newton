# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Utility script to (re)generate golden reference ``.npz`` files
# used by :pyfile:`newton/tests/test_solvers_on_examples.py`.
###########################################################################

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

from newton.tests.test_solvers_on_examples import CASES, RUN_EXAMPLE, golden_path
from newton.tests.unittest_utils import get_test_devices


def main() -> None:
    """Generate golden reference files for every applicable test case."""
    all_devices = get_test_devices()
    gpu_devices = [d for d in all_devices if d.is_cuda]

    if not gpu_devices:
        print("[Error] No GPU device found. Golden states must be generated on GPU.")
        sys.exit(1)

    generation_device = gpu_devices[0]
    print(f"[Info] Using device '{generation_device}' for golden state generation.")

    for case in CASES:
        example = case["example"]
        solver = case["solver"]

        # MuJoCo cases compare against a live MuJoCo-native run instead of a golden file.
        if solver == "mujoco":
            print(f"[Skip] {example}/mujoco - uses native reference, no golden file")
            continue

        run_example = RUN_EXAMPLE[example]

        try:
            state = run_example(
                solver_name=solver,
                policy=case.get("policy", "none"),
                num_frames=case.get("num_frames", 100),
                num_envs=case.get("num_envs", 1),
                solver_kwargs=case.get("solver_kwargs"),
                device=generation_device,
                stage_path=None,
                enable_timers=False,
            )
        except Exception as exc:
            print(f"[Skip] {example}/{solver} on {generation_device}: {exc}")
            continue

        out_path = golden_path(
            example_name=example,
            solver_name=solver,
            policy=case.get("policy", "none"),
            num_envs=case.get("num_envs", 1),
            num_frames=case.get("num_frames", 100),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(out_path, **state)
        rel_path = out_path.relative_to(Path(__file__).parent)
        print(f"[Saved] {rel_path}")


if __name__ == "__main__":
    wp.init()
    main()
