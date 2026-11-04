# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""List or run standalone collision-fidelity benchmarks with standard VBD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from collision_fidelity.catalog import load_cases


def main(argv=None) -> int:
    """Select cases before importing a device or building a scene."""
    cases = load_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true", help="List cases without initializing a device")
    selection.add_argument("--case", action="append", choices=sorted(cases), help="Select a case; may be repeated")
    selection.add_argument("--all", action="store_true", help="Run the complete catalog")
    parser.add_argument("--json", action="store_true", help="Print the case list as JSON")
    parser.add_argument("--solver", choices=("vbd",), default="vbd")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", help="Warp device, for example cuda:0")
    parser.add_argument("--duration", type=float, help="Stop early after this many physical seconds")
    parser.add_argument("--out", type=Path, help="Directory for one JSON result per case")
    parser.add_argument("--usd", action="store_true", help="Also record each scene as USD")
    args = parser.parse_args(argv)
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if args.list:
        rows = [
            {"case_id": key, "title": value.get("title", key), "support": value.get("support", {})}
            for key, value in cases.items()
        ]
        if args.json:
            print(json.dumps(rows, indent=2, allow_nan=False))
        else:
            for row in rows:
                print(f"{row['case_id']:38s} {row['title']} [{row['support'].get('status', 'unchecked')}]")
        return 0
    if args.out is None:
        parser.error("--out is required when running cases")

    from collision_fidelity.runner import run_case, summarize_results  # noqa: PLC0415

    args.out.mkdir(parents=True, exist_ok=True)
    selected = list(cases) if args.all else list(dict.fromkeys(args.case))
    outputs = [args.out / f"{case_id}.json" for case_id in selected]
    if any(path.exists() for path in [*outputs, args.out / "summary.json"]):
        parser.error("A result already exists in --out; choose a new directory")
    failed = False
    results = []
    for case_id, path in zip(selected, outputs, strict=True):
        result = run_case(
            case_id,
            profile=args.profile,
            device=args.device,
            duration=args.duration,
            usd_path=args.out / f"{case_id}.usd" if args.usd else None,
        )
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        results.append(result)
        (args.out / "summary.json").write_text(json.dumps(summarize_results(results), indent=2, allow_nan=False) + "\n")
        status = result["execution"]["status"]
        print(f"{case_id}: {status} — {path}", flush=True)
        failed |= status not in ("completed", "unsupported")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
