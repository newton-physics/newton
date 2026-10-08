# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrate a cable from a portable bundle using an explicitly configured optimizer.

Run with::

    uv run --extra calibration -m newton.examples cable_calibration

Without --bundle, the example downloads its bundle from newton-assets.
Without --run-spec, the example reads run-spec.json from the bundle directory.
Use optimizer.kind="cma" for CMA-ES. Install newton[calibration]. A null
output.directory writes no files. The search logs its progress after each
iteration; the result is in result.json and the search history in history.json.
Trace masks come from visible cable shape IDs, independently of material color;
the Chamfer objective uses projected cable geometry and needs no simulation mask.
The run spec separates physical setup from fitting configuration.

An editable template is provided in newton/examples/cable/run-spec.json.
Its geometry and initial physical values are illustrative; adapt them to your
cable. It fits rest shape and bend stiffness, with damping held fixed. Output
paths are relative to the working directory and must be absent or empty.
The attachment transform is explicit: the template uses zero offset and identity
rotation (+Z cable direction). For driven recordings it is relative to the TCP;
otherwise its rotation is in the robot base frame and offset is from cable_start.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


class Example:
    """Demonstrate problem construction, explicit search, and result consumption."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.result = None

    def run(self):
        """Fit the configured problem through the public calibration API."""
        # The calibration module is private until its public PR; import it from _src for now.
        import newton.utils  # noqa: PLC0415 -- keep CLI help lightweight
        from newton._src.calibration.calibrate import calibrate  # noqa: PLC0415
        from newton._src.calibration.evidence import CableEvidenceBundle  # noqa: PLC0415
        from newton._src.calibration.optimizer import optimizer_from_spec  # noqa: PLC0415
        from newton._src.calibration.problem import CableCalibrationProblem  # noqa: PLC0415
        from newton._src.calibration.run_spec import CableRunSpec  # noqa: PLC0415

        # Show the progress lines that the search logs after each iteration.
        logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.WARNING)
        logging.getLogger("newton._src.calibration").setLevel(logging.INFO)

        # Physical setup is independent of objective, search, and solver policy.
        # This driver selects the optimizer and output explicitly.
        bundle_dir = self.args.bundle or newton.utils.download_asset("cable_calibration_demo")
        bundle = CableEvidenceBundle.load(bundle_dir)
        run_spec_path = self.args.run_spec or Path(bundle_dir) / "run-spec.json"
        with run_spec_path.open(encoding="utf-8") as stream:
            run_spec = CableRunSpec.from_dict(json.load(stream))

        optimizer = optimizer_from_spec(run_spec.optimizer)

        problem = CableCalibrationProblem.from_bundle(bundle, bundle_dir, run_spec)
        self.result = calibrate(
            problem,
            optimizer=optimizer,
            output_dir=run_spec.output.directory,
            trace_every=run_spec.output.trace_every,
        )

        print(
            f"Fit status: {self.result.status} ({self.result.diagnostics['stop_reason']}); "
            f"objective: {self.result.metrics['loss']:.6g}"
        )
        print(json.dumps(self.result.fit, indent=2))
        return self.result

    def test_final(self):
        """Check that the driver received a valid result."""
        if self.result is None:
            raise AssertionError("run the calibration example before checking its output")
        self.result.validate()


def create_parser() -> argparse.ArgumentParser:
    """Describe the headless bundle-to-fit driver."""
    parser = argparse.ArgumentParser(
        description="Fit a cable model to a prepared evidence bundle.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help="Local directory containing bundle.json. If omitted, the bundle is downloaded from newton-assets.",
    )
    parser.add_argument(
        "--run-spec",
        type=Path,
        help=(
            "CableRunSpec JSON: setup, objective, optimizer, solver, and output settings. "
            "If omitted, run-spec.json in the bundle directory is used."
        ),
    )
    return parser


if __name__ == "__main__":
    example = Example(create_parser().parse_args())
    example.run()
    example.test_final()
