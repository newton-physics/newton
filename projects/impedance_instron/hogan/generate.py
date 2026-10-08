# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate motion from a frozen runner and an initial-condition scenario only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..cartesian.shoe import Shoe
from .mechanics import Chain
from .runner import RolloutConfig, Runner, State, Task, simulate


def main(argv: list[str] | None = None) -> None:
    """Run a scenario containing no measured trajectory, force, or event tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="CUDA by default; cpu uses the reference solver")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    expected = {"schema", "chain", "shoe", "initial", "task", "duration_s", "config"}
    if set(scenario) != expected or scenario["schema"] != "generative_runner_scenario_1":
        raise ValueError("Unsupported scenario schema or fields; only physical/task/initial inputs are accepted")
    runner = Runner.load(args.model)
    chain = Chain(**scenario["chain"])
    spec = scenario["shoe"]
    path = (args.scenario.resolve().parent / spec["artifact"]).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError("Scenario shoe hash mismatch")
    shoe = Shoe(path, spec["mount_m"], spec["pitch_rad"], friction_model=spec["friction_model"])
    initial, task, config = State(**scenario["initial"]), Task(**scenario["task"]), RolloutConfig(**scenario["config"])
    if args.device == "cpu":
        trace, summary = simulate(runner, chain, shoe, initial, task, duration_s=scenario["duration_s"], config=config)
    else:
        from .gpu_runner import GpuBatch  # noqa: PLC0415 - optional execution backend

        batch = GpuBatch(
            [chain],
            [shoe],
            [initial],
            [task],
            [scenario["duration_s"]],
            candidates=1,
            config=config,
            device=args.device,
        )
        trace, summary = batch.evaluate([runner])[0][0]
    summary.update(
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        scenario_sha256=hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        shoe_sha256=digest,
        scenario=scenario,
        device=args.device,
    )
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "trace.npz", **trace)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"generate: {summary['status']}; no reference inputs; not validated; wrote {args.output}")


if __name__ == "__main__":
    main()
