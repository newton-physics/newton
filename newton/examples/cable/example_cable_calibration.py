# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrate a cable from a portable bundle using an explicitly configured optimizer.

Run with::

    uv run --extra calibration -m newton.examples cable_calibration

Without --bundle, the example downloads its bundle from newton-assets.
Install newton[calibration]. The search logs its progress after each
iteration; the result is in result.json and the search history in history.json.
Trace masks come from visible cable shape IDs, independently of material color;
the Chamfer objective uses projected cable geometry and needs no simulation mask.

The setup, the searched parameters and the solver settings are set in code
below. Their values are illustrative; adapt them to your cable. The example
fits the rest shape and the bend stiffness, with the damping held fixed and the
twist coupled to the bend. The output directory is relative to the working
directory and must be absent or empty. The attachment transform is explicit:
zero offset and identity rotation (+Z cable direction). For driven recordings
it is relative to the TCP; otherwise its rotation is in the robot base frame
and its offset is from cable_start.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

NUM_ELEMENTS = 8
OUTPUT_DIR = "synthetic-fit-output"
TRACE_EVERY = 40


class Example:
    """Demonstrate problem construction, explicit search, and result consumption."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.result = None

    def run(self):
        """Fit the configured problem through the calibration API."""
        # The calibration module is private until its public PR; import it from _src for now.
        import newton.utils  # noqa: PLC0415 -- keep CLI help lightweight
        from newton._src.calibration.cable_setup import CableSetup  # noqa: PLC0415
        from newton._src.calibration.calibrate import calibrate  # noqa: PLC0415
        from newton._src.calibration.evaluate import CableParameters  # noqa: PLC0415
        from newton._src.calibration.evidence import CableEvidenceBundle  # noqa: PLC0415
        from newton._src.calibration.optimizer import OptimizerCMA  # noqa: PLC0415
        from newton._src.calibration.problem import CableCalibrationProblem  # noqa: PLC0415
        from newton._src.calibration.search_space import CableSearchSpace  # noqa: PLC0415

        # Show the progress lines that the search logs after each iteration.
        logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.WARNING)
        logging.getLogger("newton._src.calibration").setLevel(logging.INFO)

        bundle_dir = self.args.bundle or newton.utils.download_asset("cable_calibration_demo")
        bundle = CableEvidenceBundle.load(bundle_dir)

        # The physical cable.
        setup = CableSetup(
            num_elements=NUM_ELEMENTS,
            segment_length=0.05,
            cable_radius=0.007,
            cable_mass=0.05,
            angle_parametrization="exp_map",
            attachment_transform=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            clamp_position=0.0,
            stretch_stiffness=1000.0,
        )

        # The search start: a straight cable. Fields that are not searched keep these values.
        start = CableParameters(
            angles=[(0.0, 0.0)] * NUM_ELEMENTS,
            bend_stiffness=5.0,
            twist_stiffness=5.0,
            bend_damping=0.04,
            twist_damping=0.04,
        )
        search_space = CableSearchSpace(
            start,
            search=["angles", "bend_stiffness"],
            bounds={"bend_stiffness": (0.01, 1000.0)},
            couple_twist=True,
        )

        problem = CableCalibrationProblem.from_bundle(
            bundle, bundle_dir, setup, search_space, settle_mode="dynamic", settle_frames=500, sim_iterations=20
        )
        self.result = calibrate(
            problem,
            optimizer=OptimizerCMA(seed=42, sigma0=0.1, maxiter=120, popsize=24),
            output_dir=OUTPUT_DIR,
            trace_every=TRACE_EVERY,
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
    return parser


if __name__ == "__main__":
    example = Example(create_parser().parse_args())
    example.run()
    example.test_final()
