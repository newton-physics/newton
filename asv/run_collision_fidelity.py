# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""List or run standalone collision-fidelity benchmarks with standard VBD."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from pathlib import Path

from collision_fidelity.catalog import load_cases
from collision_fidelity.protocol import Timeline


def _write_video_result(case_id, result_path, options):
    """Own one viewer context in a fresh process and persist its measured result."""
    from collision_fidelity.runner import run_case  # noqa: PLC0415

    result = run_case(case_id, **options)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def _run_video_case(case_id, result_path, options):
    """Contain a failed recording process so the remaining selected cases can run."""
    process = multiprocessing.get_context("spawn").Process(
        target=_write_video_result, args=(case_id, result_path, options)
    )
    try:
        process.start()
    except OSError as error:
        reason = f"Could not launch video worker: {error}"
        exitcode = None
    else:
        try:
            process.join()
        except BaseException:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            raise
        exitcode = process.exitcode
        process.close()
        reason = f"Video worker exited with code {exitcode} without a valid result"
    if exitcode == 0 and result_path.is_file():
        try:
            return json.loads(result_path.read_text())
        except (ValueError, OSError):
            pass
    return {
        "schema_version": 1,
        "case_id": case_id,
        "profile": options["profile"],
        "execution": {"status": "error", "completed": False, "reason": reason},
        "criteria": {"overall": {"status": "incomplete", "reason": reason}},
        "measurements": {},
        "recording": {"status": "error", "path": Path(options["video_path"]).name, "error": reason},
    }


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
    parser.add_argument("--video", action="store_true", help="Record MP4 videos and write an HTML review page")
    args = parser.parse_args(argv)
    if args.json and not args.list:
        parser.error("--json is only valid with --list")
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

    selected = list(cases) if args.all else list(dict.fromkeys(args.case))
    if args.duration is not None:
        for case_id in selected:
            dt = Timeline(cases[case_id]).dt
            if args.duration < dt:
                parser.error(f"--duration must cover at least one physical substep for {case_id} ({dt:g} s)")
    outputs = [args.out / f"{case_id}.json" for case_id in selected]
    recordings = [args.out / f"{case_id}.usd" for case_id in selected] if args.usd else []
    if args.video:
        recordings += [args.out / f"{case_id}.mp4" for case_id in selected]
        recordings.append(args.out / "index.html")
    if any(path.exists() for path in [*outputs, *recordings, args.out / "summary.json"]):
        parser.error("A result already exists in --out; choose a new directory")
    if args.video and any(cases[key].get("support", {}).get("status") != "unsupported" for key in selected):
        from collision_fidelity.recording import recording_preflight  # noqa: PLC0415

        try:
            recording_preflight()
        except RuntimeError as error:
            parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    # Keep the claim after completion or interruption: one directory belongs to
    # one invocation, including the interval before its first result exists.
    try:
        (args.out / ".collision-fidelity-run").mkdir()
    except FileExistsError:
        parser.error("Another run owns --out; choose a new directory")
    failed = False
    results = []
    for case_id, path in zip(selected, outputs, strict=True):
        options = {
            "profile": args.profile,
            "device": args.device,
            "duration": args.duration,
            "usd_path": args.out / f"{case_id}.usd" if args.usd else None,
        }
        if args.video:
            options["video_path"] = args.out / f"{case_id}.mp4"
        if args.video and cases[case_id].get("support", {}).get("status") != "unsupported":
            result = _run_video_case(case_id, path, options)
        else:
            result = run_case(case_id, **options)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        results.append(result)
        (args.out / "summary.json").write_text(json.dumps(summarize_results(results), indent=2, allow_nan=False) + "\n")
        status = result["execution"]["status"]
        print(f"{case_id}: {status} — {path}", flush=True)
        failed |= status not in ("completed", "unsupported")
        if args.video:
            from collision_fidelity.review import write_review  # noqa: PLC0415

            write_review(args.out, results, cases)
            failed |= result["recording"]["status"] not in ("completed", "unsupported")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
