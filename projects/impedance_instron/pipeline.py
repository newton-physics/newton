# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the twelve-point baseline, numerical qualification, GPU fit, and replay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "outputs/impedance_instron/baseline12_maxwell"


def create_parser() -> argparse.ArgumentParser:
    """Expose one fixed-size controller pipeline and its reusable stages."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Processed data: use `visual3d --help` to inspect/import exports or `prepare-visual3d --help` to prepare a reference.",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "prepare", "validate", "fit", "report"), default="all")
    parser.add_argument(
        "--from-scratch", action="store_true", help="Prepare an unfitted controller without using saved coefficients."
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--wall-seconds", type=float, default=3600.0)
    parser.add_argument("--plateau-patience", type=int, default=20)
    parser.add_argument("--plateau-rtol", type=float, default=1.0e-4)
    parser.add_argument(
        "--mesh-only", action="store_true", help="Skip verified spring export only when explicitly requested."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Execute complete stages without reusing historical numerical permissions."""
    args = create_parser().parse_args(argv)
    if args.iterations < 1 or args.plateau_patience < 1:
        raise ValueError("Iteration and plateau budgets must be positive")
    if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        raise ValueError("Wall budget must be finite and positive")
    if not math.isfinite(args.plateau_rtol) or args.plateau_rtol <= 0:
        raise ValueError("Plateau tolerance must be finite and positive")
    output = args.output.resolve()
    baseline = output / "baseline"
    if args.stage in ("all", "prepare"):
        from .cartesian.gpu.baseline import build  # noqa: PLC0415
        from .cartesian.gpu.provenance import source_snapshot  # noqa: PLC0415

        if output.exists():
            raise FileExistsError(f"Choose a new output directory: {output}")
        build(args.baseline, baseline, from_scratch=args.from_scratch)
        sources = {str(ROOT / path): {"baseline": digest} for path, digest in source_snapshot().items()}
        (output / "physical_source_identity.json").write_text(json.dumps(sources, indent=2) + "\n")
    if args.from_scratch and args.stage not in ("all", "prepare"):
        prepared = json.loads((baseline / "summary.json").read_text())
        if prepared.get("provenance", {}).get("generation") != "fresh_twelve_point_controller":
            raise ValueError("--from-scratch requires a baseline prepared with --from-scratch")
    if args.stage in ("all", "validate"):
        from .cartesian.gpu.__main__ import _validation  # noqa: PLC0415
        from .cartesian.gpu.batch_benchmark import benchmark as mixed_benchmark  # noqa: PLC0415
        from .cartesian.gpu.benchmark import benchmark  # noqa: PLC0415
        from .cartesian.gpu.contact_replay import main as contact_main  # noqa: PLC0415

        benchmark(baseline, output / "single", repeats=2)
        contact_main(
            [
                "--baseline",
                str(baseline),
                "--full-gpu",
                str(output / "single"),
                "--output",
                str(output / "contact"),
            ]
        )
        mixed_benchmark(baseline, output / "mixed", baseline / "failure_equilibrium.npz", worlds=128)
        _validation(
            output / "single/benchmark.json",
            baseline,
            mixed=False,
            expected_controls=12,
            contact_evidence=output / "contact/report.json",
        )
        _validation(output / "mixed/benchmark.json", baseline, mixed=True, expected_controls=12)
    if args.stage in ("all", "fit"):
        from .cartesian.gpu.__main__ import main as fit_main  # noqa: PLC0415

        command = [
            str(baseline),
            "--output",
            str(output / "fit"),
            "--single-validation",
            str(output / "single/benchmark.json"),
            "--batch-validation",
            str(output / "mixed/benchmark.json"),
            "--contact-rounding-evidence",
            str(output / "contact/report.json"),
            "--iterations",
            str(args.iterations),
            "--wall-seconds",
            str(args.wall_seconds),
            "--plateau-patience",
            str(args.plateau_patience),
            "--plateau-rtol",
            str(args.plateau_rtol),
        ]
        if args.mesh_only:
            command.append("--mesh-only")
        fit_main(command)
    if args.stage == "report":
        from .cartesian.__main__ import main as report_main  # noqa: PLC0415

        command = ["report", str(output / "fit")]
        if args.mesh_only:
            command.append("--mesh-only")
        report_main(command)


if __name__ == "__main__":
    main()
