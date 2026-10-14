# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrate a cable from a portable bundle using an explicitly configured optimizer.

Run with::

    uv run --extra calibration -m newton.examples cable_calibration

Without --bundle, the example downloads its bundle from newton-assets.
Without --run-spec, the example reads run-spec.json from the bundle directory.
Use optimizer.kind="cma" for CMA-ES. Install newton[calibration]; rendered traces
also need newton[calibration-trace]. A null output.directory writes no files.
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
import time
from pathlib import Path


class Example:
    """Demonstrate problem construction, explicit search, and result consumption."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.result = None

    def run(self):
        """Fit the configured problem through the public calibration API."""
        from newton.calibration import (  # noqa: PLC0415 -- keep CLI help lightweight
            CableCalibrationProblem,
            CableEvidenceBundle,
            CableRunSpec,
            calibrate,
            optimizer_from_spec,
        )

        import newton.utils  # noqa: PLC0415 -- keep CLI help lightweight

        # Physical setup is independent of objective, search, and solver policy.
        # This driver selects the optimizer and output explicitly.
        bundle_dir = self.args.bundle or newton.utils.download_asset("cable_calibration_demo")
        bundle = CableEvidenceBundle.load(bundle_dir)
        run_spec_path = self.args.run_spec or Path(bundle_dir) / "run-spec.json"
        with run_spec_path.open(encoding="utf-8") as stream:
            run_spec = CableRunSpec.from_dict(json.load(stream))

        optimizer = optimizer_from_spec(run_spec.optimizer)

        problem = CableCalibrationProblem.from_bundle(bundle, bundle_dir, run_spec)
        self._scalar_names = problem.search_space.searched_scalars()
        self._header_printed = False
        self._t0 = time.perf_counter()
        self.result = calibrate(
            problem,
            optimizer=optimizer,
            output_dir=run_spec.output.directory,
            trace_every=run_spec.output.trace_every,
            on_iteration=self.report_iteration,
        )

        print(f"Fit status: {self.result.status}; objective: {self.result.metrics['loss']:.6g}")
        print(json.dumps(self.result.fit, indent=2))
        return self.result

    def report_iteration(self, iteration, candidate, loss, diagnostics):
        """Display one row per iteration: elapsed time, objective, and each searched scalar."""
        elapsed = time.perf_counter() - self._t0
        if not self._header_printed:
            names = "".join(f" {n:>15}" for n in self._scalar_names)
            print(f"{'iter':>5} {'elapsed[s]':>10} {'objective':>12}{names}", flush=True)
            self._header_printed = True
        values = "".join(f" {getattr(candidate, n):>15.6g}" for n in self._scalar_names)
        # Only some optimizers report diagnostics (e.g. fdgrad's grad_norm/step);
        # CMA passes an empty dict, so omit it rather than print "diagnostics={}".
        extra = f"  {diagnostics}" if diagnostics else ""
        print(f"{iteration:>5} {elapsed:>10.1f} {loss:>12.6g}{values}{extra}", flush=True)

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
