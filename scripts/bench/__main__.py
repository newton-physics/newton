# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CENIC benchmark platform entry point.

Usage:
    uv run -m scripts.bench                          # run all benchmarks
    uv run -m scripts.bench --only scaling            # run one benchmark
    uv run -m scripts.bench --skip accuracy           # skip one
    uv run -m scripts.bench --list                    # list available benchmarks
    uv run -m scripts.bench --ns 1 4 16 64 256        # override N values
    uv run -m scripts.bench --steps 50 --warmup 20    # override timing params
"""

import argparse

from scripts.bench.runner import list_benchmarks, run


def main():
    parser = argparse.ArgumentParser(
        description="CENIC benchmark platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this benchmark")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        help="Skip these benchmarks")
    parser.add_argument("--list", action="store_true",
                        help="List available benchmarks and exit")

    # Benchmark parameter overrides.
    parser.add_argument("--ns", type=int, nargs="+", default=None,
                        help="Override N values for scaling/components")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override timed steps for scaling/components")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Override warmup steps for scaling/components")
    parser.add_argument("--trials", type=int, default=None,
                        help="Override trial count for accuracy")
    parser.add_argument("--epsilons", type=float, nargs="+", default=None,
                        help="Epsilon values for IC perturbation sweep")
    parser.add_argument("--epsilon-sweep-n", type=int, default=None,
                        help="Fixed N for epsilon sweep experiment")

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    bench_args = {}
    if args.ns is not None:
        bench_args["ns"] = sorted(args.ns)
    if args.steps is not None:
        bench_args["steps"] = args.steps
    if args.warmup is not None:
        bench_args["warmup"] = args.warmup
    if args.trials is not None:
        bench_args["trials"] = args.trials
    if args.epsilons is not None:
        bench_args["epsilons"] = sorted(args.epsilons)
    if args.epsilon_sweep_n is not None:
        bench_args["epsilon_sweep_n"] = args.epsilon_sweep_n

    run(only=args.only, skip=args.skip, args=bench_args)


if __name__ == "__main__":
    main()
