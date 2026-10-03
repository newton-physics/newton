# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-contact accuracy probe for articulated contact propagation rows.

This intentionally measures one fixed contact problem.  Contacts are generated
once from the initial state, then reused for every solver path and iteration
count.  That avoids trajectory divergence, contact-discovery changes, and pile
settling noise while still exercising the public ``SolverFeatherPGS.step`` path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from fpgs_articulation_row_scaling import (  # noqa: E402
    BenchCase,
    _abs_l2,
    _abs_linf,
    _planned_row_capacity,
    _prepare_case_run,
    _restore_state,
    _snapshot_state,
    _state_linf,
    scalar_hinv_jt,
)

_NEWTON_REPO = Path(__file__).resolve().parents[3]
if str(_NEWTON_REPO) not in sys.path:
    sys.path.insert(0, str(_NEWTON_REPO))

import newton  # noqa: E402
from newton.solvers import SolverFeatherPGS  # noqa: E402

PATHS = ("mf_immediate", "propagation")
CONVERGENCE_KEYS = (
    "max_delta_lambda",
    "complementarity_gap",
    "tangent_residual",
    "fb_merit",
)
NCP_KEYS = (
    "r_compl",
    "r_cone",
    "r_gap",
    "r_ds_compl",
    "r_ds_dual",
    "r_mdp_dir",
)


@dataclass
class AccuracyResult:
    label: str
    path: str
    pgs_iterations: int
    reference_iterations: int
    friction: bool
    articulations: int
    links: int
    contacts_per_articulation: int
    rigid_contacts: int
    joint_dof_count: int
    dense_rows_total: int
    mf_rows_total: int
    propagation_rows_total: int
    final_max_delta_lambda: float
    final_complementarity_gap: float
    final_tangent_residual: float
    final_fb_merit: float
    final_r_compl: float
    final_r_cone: float
    final_r_gap: float
    final_r_ds_compl: float
    final_r_ds_dual: float
    final_r_mdp_dir: float
    self_ref_joint_qd_abs_l2: float
    self_ref_joint_qd_abs_linf: float
    self_ref_body_qd_abs_l2: float
    self_ref_body_qd_abs_linf: float
    self_ref_state_linf: float
    mf_ref_joint_qd_abs_l2: float
    mf_ref_joint_qd_abs_linf: float
    mf_ref_body_qd_abs_l2: float
    mf_ref_body_qd_abs_linf: float
    mf_ref_state_linf: float
    same_iter_vs_mf_joint_qd_abs_l2: float | None = None
    same_iter_vs_mf_joint_qd_abs_linf: float | None = None
    same_iter_vs_mf_body_qd_abs_l2: float | None = None
    same_iter_vs_mf_body_qd_abs_linf: float | None = None
    same_iter_vs_mf_state_linf: float | None = None


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _case_label(case: BenchCase) -> str:
    return f"A={case.articulations},L={case.links},boxes={case.contacts_per_articulation}"


def _solver_capacities(case: BenchCase) -> tuple[int, int]:
    row_capacity = _planned_row_capacity(case)
    dense_capacity = row_capacity
    mf_capacity = 16
    return dense_capacity, mf_capacity


def _make_solver(
    model: newton.Model,
    case: BenchCase,
    path: str,
    *,
    pgs_iterations: int,
    pgs_velocity_iterations: int,
    enable_contact_friction: bool,
    contact_friction_position_iterations: int,
    pgs_debug: bool,
) -> SolverFeatherPGS:
    dense_capacity, mf_capacity = _solver_capacities(case)
    response_mode = "propagation" if path == "propagation" else "immediate"
    with scalar_hinv_jt():
        return SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response=response_mode,
            pgs_iterations=pgs_iterations,
            pgs_velocity_iterations=pgs_velocity_iterations,
            enable_contact_friction=enable_contact_friction,
            contact_friction_position_iterations=contact_friction_position_iterations,
            dense_max_constraints=dense_capacity,
            mf_max_constraints=mf_capacity,
            pgs_warmstart=False,
            mf_warmstart=False,
            pgs_debug=pgs_debug,
        )


def _run_one_step(
    model: newton.Model,
    solver: SolverFeatherPGS,
    contacts: newton.Contacts,
    initial: dict[str, np.ndarray],
    dt: float,
) -> dict[str, np.ndarray]:
    state_in = model.state()
    state_out = model.state()
    _restore_state(model, state_in, initial)
    _restore_state(model, state_out, initial)
    state_in.clear_forces()
    state_out.clear_forces()
    solver.step(state_in, state_out, model.control(), contacts, dt)
    wp.synchronize()
    return _snapshot_state(state_out)


def _count_rows(solver: SolverFeatherPGS) -> tuple[int, int, int]:
    dense = int(np.sum(solver.constraint_count.numpy()))
    mf = int(np.sum(solver.mf_constraint_count.numpy()))
    propagation_count_array = getattr(solver, "propagation_constraint_count", None)
    propagation = 0 if propagation_count_array is None else int(np.sum(propagation_count_array.numpy()))
    return dense, mf, propagation


def _last_debug_metrics(solver: SolverFeatherPGS) -> tuple[dict[str, float], dict[str, float]]:
    if not solver._pgs_convergence_log:
        return (dict.fromkeys(CONVERGENCE_KEYS, math.nan), dict.fromkeys(NCP_KEYS, math.nan))

    convergence = np.asarray(solver._pgs_convergence_log[-1], dtype=np.float64)
    if convergence.size == 0:
        conv = dict.fromkeys(CONVERGENCE_KEYS, math.nan)
    else:
        conv = {key: float(convergence[-1, idx]) for idx, key in enumerate(CONVERGENCE_KEYS)}

    if not solver._pgs_ncp_residual_log:
        ncp = dict.fromkeys(NCP_KEYS, math.nan)
    else:
        ncp_log = np.asarray(solver._pgs_ncp_residual_log[-1], dtype=np.float64)
        if ncp_log.size == 0:
            ncp = dict.fromkeys(NCP_KEYS, math.nan)
        else:
            final = np.max(ncp_log[-1], axis=0)
            ncp = {key: float(final[idx]) for idx, key in enumerate(NCP_KEYS)}
    return conv, ncp


def _state_delta(prefix: str, value: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        f"{prefix}_joint_qd_abs_l2": _abs_l2(value["joint_qd"], reference["joint_qd"]),
        f"{prefix}_joint_qd_abs_linf": _abs_linf(value["joint_qd"], reference["joint_qd"]),
        f"{prefix}_body_qd_abs_l2": _abs_l2(value["body_qd"], reference["body_qd"]),
        f"{prefix}_body_qd_abs_linf": _abs_linf(value["body_qd"], reference["body_qd"]),
        f"{prefix}_state_linf": _state_linf(value, reference),
    }


def _make_case(args: argparse.Namespace, links: int) -> BenchCase:
    return BenchCase(
        "frozen_accuracy",
        _case_label(
            BenchCase(
                "frozen_accuracy",
                "",
                "articulated_free",
                articulations=args.articulations,
                links=links,
                contacts_per_articulation=args.contacts_per_articulation,
                joint_armature=args.joint_armature,
            )
        ),
        "articulated_free",
        articulations=args.articulations,
        links=links,
        contacts_per_articulation=args.contacts_per_articulation,
        joint_armature=args.joint_armature,
    )


def _run_accuracy_case(args: argparse.Namespace, case: BenchCase) -> list[AccuracyResult]:
    model, initial, contacts = _prepare_case_run(case, args.device)

    reference_states: dict[str, dict[str, np.ndarray]] = {}
    for path in PATHS:
        solver = _make_solver(
            model,
            case,
            path,
            pgs_iterations=args.reference_iterations,
            pgs_velocity_iterations=args.pgs_velocity_iterations,
            enable_contact_friction=not args.no_friction,
            contact_friction_position_iterations=args.contact_friction_position_iterations,
            pgs_debug=False,
        )
        reference_states[path] = _run_one_step(model, solver, contacts, initial, args.dt)

    states: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    partial: dict[tuple[str, int], dict[str, Any]] = {}
    for iterations in args.iterations:
        for path in PATHS:
            solver = _make_solver(
                model,
                case,
                path,
                pgs_iterations=iterations,
                pgs_velocity_iterations=args.pgs_velocity_iterations,
                enable_contact_friction=not args.no_friction,
                contact_friction_position_iterations=args.contact_friction_position_iterations,
                pgs_debug=True,
            )
            final_state = _run_one_step(model, solver, contacts, initial, args.dt)
            dense_rows, mf_rows, propagation_rows = _count_rows(solver)
            convergence, ncp = _last_debug_metrics(solver)
            states[(path, iterations)] = final_state
            partial[(path, iterations)] = {
                "label": case.label,
                "path": path,
                "pgs_iterations": iterations,
                "reference_iterations": args.reference_iterations,
                "friction": not args.no_friction,
                "articulations": case.articulations,
                "links": case.links,
                "contacts_per_articulation": case.contacts_per_articulation,
                "rigid_contacts": int(contacts.rigid_contact_count.numpy()[0]),
                "joint_dof_count": model.joint_dof_count,
                "dense_rows_total": dense_rows,
                "mf_rows_total": mf_rows,
                "propagation_rows_total": propagation_rows,
                "final_max_delta_lambda": convergence["max_delta_lambda"],
                "final_complementarity_gap": convergence["complementarity_gap"],
                "final_tangent_residual": convergence["tangent_residual"],
                "final_fb_merit": convergence["fb_merit"],
                "final_r_compl": ncp["r_compl"],
                "final_r_cone": ncp["r_cone"],
                "final_r_gap": ncp["r_gap"],
                "final_r_ds_compl": ncp["r_ds_compl"],
                "final_r_ds_dual": ncp["r_ds_dual"],
                "final_r_mdp_dir": ncp["r_mdp_dir"],
                **_state_delta("self_ref", final_state, reference_states[path]),
                **_state_delta("mf_ref", final_state, reference_states["mf_immediate"]),
            }

    results: list[AccuracyResult] = []
    for iterations in args.iterations:
        mf_state = states[("mf_immediate", iterations)]
        for path in PATHS:
            row = dict(partial[(path, iterations)])
            if path == "propagation":
                row.update(_state_delta("same_iter_vs_mf", states[(path, iterations)], mf_state))
            results.append(AccuracyResult(**row))
    return results


def run_accuracy_probe(args: argparse.Namespace) -> list[AccuracyResult]:
    links = args.link_sweep if args.link_sweep is not None else [args.links]
    rows: list[AccuracyResult] = []
    for link_count in links:
        rows.extend(_run_accuracy_case(args, _make_case(args, link_count)))
    return rows


def write_outputs(out_dir: Path, rows: list[AccuracyResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dicts = [asdict(row) for row in rows]
    (out_dir / "accuracy_probe.json").write_text(json.dumps(dicts, indent=2), encoding="utf-8")

    with (out_dir / "accuracy_probe.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dicts[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(dicts)

    lines = [
        "# Frozen-contact articulated accuracy probe",
        "",
        "Contacts are generated once and reused for every path. Metrics therefore measure the fixed PGS problem, not long-horizon trajectory divergence.",
        "",
        "| case | path | iters | rows dense/mf/propagation | final r_compl | final r_cone | final tangent | self-ref qd Linf | MF-ref qd Linf | same-iter MF qd Linf |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.label,
                    row.path,
                    str(row.pgs_iterations),
                    f"{row.dense_rows_total}/{row.mf_rows_total}/{row.propagation_rows_total}",
                    f"{row.final_r_compl:.6g}",
                    f"{row.final_r_cone:.6g}",
                    f"{row.final_tangent_residual:.6g}",
                    f"{row.self_ref_joint_qd_abs_linf:.6g}",
                    f"{row.mf_ref_joint_qd_abs_linf:.6g}",
                    ""
                    if row.same_iter_vs_mf_joint_qd_abs_linf is None
                    else f"{row.same_iter_vs_mf_joint_qd_abs_linf:.6g}",
                ]
            )
            + " |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/fpgs_articulation_row_scaling/frozen_accuracy_probe")
    )
    parser.add_argument("--articulations", type=int, default=1)
    parser.add_argument("--links", type=int, default=64)
    parser.add_argument("--link-sweep", type=_parse_int_list, default=None)
    parser.add_argument("--contacts-per-articulation", type=int, default=64)
    parser.add_argument("--joint-armature", type=float, default=1.0)
    parser.add_argument("--iterations", type=_parse_int_list, default=[4, 8, 16, 32])
    parser.add_argument("--reference-iterations", type=int, default=64)
    parser.add_argument("--pgs-velocity-iterations", type=int, default=0)
    parser.add_argument("--contact-friction-position-iterations", type=int, default=-1)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--no-friction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("This probe is intended for CUDA devices")
    rows = run_accuracy_probe(args)
    write_outputs(args.out_dir, rows)
    print(f"Wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
