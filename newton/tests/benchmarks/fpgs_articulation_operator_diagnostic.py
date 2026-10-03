# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare D-wide and propagation articulated contact row operators.

This diagnostic is intentionally not an end-state PGS comparison. It builds the
same contact problem through ``SolverFeatherPGS.step`` with zero PGS iterations,
then compares the row-to-row Delassus operator:

    W = J H^-1 J^T

For the D-wide reference path, the tiled ``H^-1 J^T`` kernel materializes
``J_world`` and ``Y_world``, and W is computed directly from those buffers. For
the propagation path, each source row is injected as a unit body-space row
impulse, the actual propagation kernel is run once, and the resulting target row
velocities are sampled through the body-space row Jacobians.

If these operators match, later state differences are solve ordering or
underconvergence. If they do not match, the propagation math/setup is different from
the current production operator before PGS convergence enters the picture.
"""

from __future__ import annotations

import argparse
import json
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
    _planned_row_capacity,
    _prepare_case_run,
    _restore_state,
)

from newton.solvers import SolverFeatherPGS  # noqa: E402


@dataclass
class WorstColumn:
    source_row: int
    source_row_type: int
    source_body_a: int
    source_body_b: int
    source_articulation: int | None
    source_link: int | None
    target_row: int
    target_row_type: int
    target_body_a: int
    target_body_b: int
    target_articulation: int | None
    target_link: int | None
    abs_linf: float
    rel_l2: float


@dataclass
class LinkError:
    source_link: int
    sampled_rows: int
    max_abs: float
    rel_l2: float
    worst_target_link: int | None


@dataclass
class DistanceError:
    link_distance: int
    entries: int
    max_abs: float
    rms_abs: float


@dataclass
class OperatorDiagnosticResult:
    label: str
    links: int
    contacts_per_articulation: int
    world: int
    rows: int
    rigid_contacts: int
    D: int
    sampled_sources: int
    all_sources: bool
    row_alignment_complete: bool
    row_order_identity: bool
    row_metadata_match: bool
    row_type_mismatch_count: int
    row_parent_mismatch_count: int
    operator_status: str
    rel_fro: float
    abs_linf: float
    diag_rel_l2: float
    diag_abs_linf: float
    offdiag_rel_l2: float
    qd_rel_l2: float
    qd_abs_linf: float
    mass_condition_est: float | None
    local_vs_propagated_rel_fro: float
    local_vs_propagated_abs_linf: float
    worst_columns: list[WorstColumn]
    per_source_link: list[LinkError]
    per_link_distance: list[DistanceError]


def _parse_int_list(value: str) -> list[int]:
    out = []
    for item in value.split(","):
        token = item.strip()
        if token:
            out.append(int(token))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _contacts_for_links(links: list[int], value: str) -> list[int]:
    value = value.strip().lower()
    if value == "same":
        return links
    parsed = _parse_int_list(value)
    if len(parsed) == 1:
        return parsed * len(links)
    if len(parsed) != len(links):
        raise ValueError("--contacts-per-articulation must be 'same', one integer, or one integer per link count")
    return parsed


def _make_solver(
    model: Any,
    case: BenchCase,
    path: str,
    *,
    enable_contact_friction: bool,
) -> SolverFeatherPGS:
    row_capacity = _planned_row_capacity(case)
    dense_capacity = row_capacity
    mf_capacity = 16
    response_mode = "propagation" if path == "propagation" else "immediate"

    previous_overrides = SolverFeatherPGS._kernel_overrides
    if path == "mf_immediate":
        SolverFeatherPGS._kernel_overrides = {**previous_overrides, "hinv_jt_kernel": "tiled"}
    try:
        return SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response=response_mode,
            pgs_iterations=0,
            pgs_velocity_iterations=0,
            enable_contact_friction=enable_contact_friction,
            dense_max_constraints=dense_capacity,
            mf_max_constraints=mf_capacity,
            pgs_warmstart=False,
            mf_warmstart=False,
        )
    finally:
        SolverFeatherPGS._kernel_overrides = previous_overrides


def _run_zero_iteration_setup(
    model: Any,
    solver: SolverFeatherPGS,
    contacts: Any,
    initial: dict[str, np.ndarray],
    *,
    dt: float,
) -> None:
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    _restore_state(model, state_in, initial)
    _restore_state(model, state_out, initial)
    state_in.clear_forces()
    state_out.clear_forces()
    solver.step(state_in, state_out, control, contacts, dt)
    wp.synchronize()


def _body_link_maps(solver: SolverFeatherPGS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = solver.model
    body_to_art = solver.body_to_articulation.numpy().astype(np.int32, copy=False)
    body_to_link = np.full(model.body_count, -1, dtype=np.int32)
    articulation_start = model.articulation_start.numpy().astype(np.int32, copy=False)
    joint_child = model.joint_child.numpy().astype(np.int32, copy=False)
    for art in range(model.articulation_count):
        for local_link, joint in enumerate(range(int(articulation_start[art]), int(articulation_start[art + 1]))):
            child = int(joint_child[joint])
            if child >= 0:
                body_to_link[child] = local_link

    if solver.is_free_rigid is None:
        art_is_free = np.zeros(model.articulation_count, dtype=bool)
    else:
        art_is_free = solver.is_free_rigid.numpy().astype(np.int32, copy=False) != 0
    body_is_free = np.zeros(model.body_count, dtype=bool)
    for body, art in enumerate(body_to_art):
        if art >= 0 and art_is_free[int(art)]:
            body_is_free[body] = True
    return body_to_art, body_to_link, body_is_free


def _row_art_link(
    row: int,
    body_a: np.ndarray,
    body_b: np.ndarray,
    body_to_art: np.ndarray,
    body_to_link: np.ndarray,
    body_is_free: np.ndarray,
) -> tuple[int | None, int | None]:
    for body in (int(body_a[row]), int(body_b[row])):
        if body >= 0 and not body_is_free[body]:
            art = int(body_to_art[body])
            link = int(body_to_link[body])
            return art, link
    return None, None


def _select_sources(row_count: int, max_sources: int) -> np.ndarray:
    if max_sources <= 0 or max_sources >= row_count:
        return np.arange(row_count, dtype=np.int32)
    values = np.linspace(0, row_count - 1, max_sources)
    return np.unique(np.rint(values).astype(np.int32))


def _world_operator_from_jy(solver: SolverFeatherPGS, world: int) -> np.ndarray:
    counts = solver.constraint_count.numpy().astype(np.int32, copy=False)
    row_count = int(counts[world])
    width = int(solver.world_dof_count.numpy()[world])
    J = solver.J_world.numpy()[world, :row_count, :width].astype(np.float64, copy=True)
    Y = solver.Y_world.numpy()[world, :row_count, :width].astype(np.float64, copy=True)
    return J @ Y.T


def _contact_row_map(
    solver: SolverFeatherPGS,
    world: int,
    path_id: int,
    row_count: int,
    row_parent: np.ndarray,
) -> dict[tuple[int, int], int]:
    contact_world = solver.contact_world.numpy().astype(np.int32, copy=False)
    contact_slot = solver.contact_slot.numpy().astype(np.int32, copy=False)
    contact_path = solver.contact_path.numpy().astype(np.int32, copy=False)
    out: dict[tuple[int, int], int] = {}
    for contact_index in range(contact_slot.shape[0]):
        if int(contact_path[contact_index]) != path_id:
            continue
        if int(contact_world[contact_index]) != world:
            continue
        base = int(contact_slot[contact_index])
        if base < 0 or base >= row_count:
            continue
        out[(contact_index, 0)] = base
        for offset in (1, 2):
            row = base + offset
            if row < row_count and int(row_parent[row]) == base:
                out[(contact_index, offset)] = row
    return out


def _dense_rows_in_propagation_order(
    old_solver: SolverFeatherPGS,
    propagation_solver: SolverFeatherPGS,
    world: int,
    row_count: int,
) -> tuple[np.ndarray, bool]:
    old_parent = old_solver.row_parent.numpy()[world, :row_count].astype(np.int32, copy=True)
    propagation_parent = propagation_solver.propagation_row_parent.numpy()[world, :row_count].astype(
        np.int32, copy=True
    )
    dense_by_key = _contact_row_map(old_solver, world, 0, row_count, old_parent)
    propagation_by_key = _contact_row_map(propagation_solver, world, 2, row_count, propagation_parent)

    dense_rows = np.full(row_count, -1, dtype=np.int32)
    for key, propagation_row in propagation_by_key.items():
        dense_row = dense_by_key.get(key)
        if dense_row is not None and 0 <= propagation_row < row_count:
            dense_rows[propagation_row] = int(dense_row)

    complete = bool(np.all(dense_rows >= 0))
    if not complete:
        missing = np.where(dense_rows < 0)[0][:8].tolist()
        raise RuntimeError(
            f"world {world}: could not align dense rows to propagation contact rows; missing propagation rows {missing}"
        )
    return dense_rows, bool(np.array_equal(dense_rows, np.arange(row_count, dtype=np.int32)))


def _propagation_target_velocities(
    propagation_qd: np.ndarray,
    body_a: np.ndarray,
    body_b: np.ndarray,
    J_a: np.ndarray,
    J_b: np.ndarray,
) -> np.ndarray:
    rows = body_a.shape[0]
    out = np.zeros(rows, dtype=np.float64)
    for row in range(rows):
        ba = int(body_a[row])
        bb = int(body_b[row])
        if ba >= 0:
            out[row] += float(np.dot(J_a[row], propagation_qd[ba]))
        if bb >= 0:
            out[row] += float(np.dot(J_b[row], propagation_qd[bb]))
    return out


def _zero_propagation_state(solver: SolverFeatherPGS) -> None:
    solver.v_out.zero_()
    solver.propagation_body_qd.zero_()
    solver.propagation_body_impulses.zero_()
    for name in ("propagation_tree_pA", "propagation_tree_u", "propagation_tree_qdd", "propagation_tree_body_delta"):
        array = getattr(solver, name, None)
        if array is not None:
            array.zero_()


def _propagation_operator_columns(
    solver: SolverFeatherPGS,
    world: int,
    source_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    propagation_counts = solver.propagation_constraint_count.numpy().astype(np.int32, copy=False)
    row_count = int(propagation_counts[world])
    body_a = solver.propagation_body_a.numpy()[world, :row_count].astype(np.int32, copy=True)
    body_b = solver.propagation_body_b.numpy()[world, :row_count].astype(np.int32, copy=True)
    J_a = solver.propagation_J_a.numpy()[world, :row_count].astype(np.float64, copy=True)
    J_b = solver.propagation_J_b.numpy()[world, :row_count].astype(np.float64, copy=True)
    MiJt_a = solver.propagation_MiJt_a.numpy()[world, :row_count].astype(np.float64, copy=True)
    MiJt_b = solver.propagation_MiJt_b.numpy()[world, :row_count].astype(np.float64, copy=True)

    body_count = solver.model.body_count
    world_start = int(solver.world_dof_start.numpy()[world])
    world_width = int(solver.world_dof_count.numpy()[world])
    propagated = np.zeros((row_count, len(source_rows)), dtype=np.float64)
    local_only = np.zeros_like(propagated)
    qd_propagated = np.zeros((world_width, len(source_rows)), dtype=np.float64)
    for out_col, source in enumerate(source_rows):
        body_qd = np.zeros((body_count, 6), dtype=np.float32)
        body_impulses = np.zeros((body_count, 6), dtype=np.float32)
        ba = int(body_a[source])
        bb = int(body_b[source])
        if ba >= 0:
            body_qd[ba] += MiJt_a[source].astype(np.float32, copy=False)
            body_impulses[ba] += J_a[source].astype(np.float32, copy=False)
        if bb >= 0:
            body_qd[bb] += MiJt_b[source].astype(np.float32, copy=False)
            body_impulses[bb] += J_b[source].astype(np.float32, copy=False)

        local_only[:, out_col] = _propagation_target_velocities(body_qd.astype(np.float64), body_a, body_b, J_a, J_b)

        _zero_propagation_state(solver)
        solver.propagation_body_qd.assign(body_qd)
        solver.propagation_body_impulses.assign(body_impulses)
        solver._propagate_response()
        wp.synchronize()

        qd_after = solver.propagation_body_qd.numpy().astype(np.float64, copy=True)
        propagated[:, out_col] = _propagation_target_velocities(qd_after, body_a, body_b, J_a, J_b)
        qd_propagated[:, out_col] = solver.v_out.numpy()[world_start : world_start + world_width].astype(
            np.float64, copy=True
        )

    return propagated, local_only, qd_propagated


def _nonfree_world_dof_mask_and_condition(solver: SolverFeatherPGS, world: int) -> tuple[np.ndarray, float | None]:
    width = int(solver.world_dof_count.numpy()[world])
    world_start = int(solver.world_dof_start.numpy()[world])
    mask = np.zeros(width, dtype=bool)

    if solver.is_free_rigid is None:
        art_is_free = np.zeros(solver.model.articulation_count, dtype=bool)
    else:
        art_is_free = solver.is_free_rigid.numpy().astype(np.int32, copy=False) != 0

    art_to_world = solver.art_to_world.numpy().astype(np.int32, copy=False)
    art_dof_start = solver.articulation_dof_start.numpy().astype(np.int32, copy=False)
    # PR #13 renamed the per-articulation solve-size array (art_size ->
    # articulation_response_dof_count, sourced from the model plan); read the
    # plan's response_dof_count like the surviving solver-side consumers
    # (801d8dd0). Identical to the old art_size here because prescribed-response
    # elision is gated to the immediate path.
    art_size = solver._model_plan.response_dof_count.astype(np.int32, copy=False)

    condition_values: list[float] = []
    for art in range(solver.model.articulation_count):
        if int(art_to_world[art]) != world or bool(art_is_free[art]):
            continue

        local_start = int(art_dof_start[art]) - world_start
        local_end = local_start + int(art_size[art])
        if local_end <= 0 or local_start >= width:
            continue
        mask[max(local_start, 0) : min(local_end, width)] = True

        size = int(art_size[art])
        if size <= 0 or size not in solver.L_by_size:
            continue
        group_to_art = solver.group_to_art[size].numpy().astype(np.int32, copy=False)
        matches = np.where(group_to_art == art)[0]
        if matches.size == 0:
            continue
        group_idx = int(matches[0])
        L = solver.L_by_size[size].numpy()[group_idx, :size, :size].astype(np.float64, copy=True)
        H = L @ L.T
        try:
            eig = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            continue
        positive = eig[eig > 0.0]
        if positive.size:
            condition_values.append(float(positive[-1] / positive[0]))

    condition = max(condition_values) if condition_values else None
    return mask, condition


def _rel_l2(value: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference.reshape(-1)))
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm((value - reference).reshape(-1)) / denom)


def _offdiag_rel(value: np.ndarray, reference: np.ndarray, source_rows: np.ndarray) -> float:
    mask = np.ones_like(value, dtype=bool)
    for col, source in enumerate(source_rows):
        if source < mask.shape[0]:
            mask[int(source), col] = False
    ref = reference[mask]
    val = value[mask]
    denom = float(np.linalg.norm(ref))
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm(val - ref) / denom)


def _worst_columns(
    diff: np.ndarray,
    old: np.ndarray,
    source_rows: np.ndarray,
    row_type: np.ndarray,
    body_a: np.ndarray,
    body_b: np.ndarray,
    body_to_art: np.ndarray,
    body_to_link: np.ndarray,
    body_is_free: np.ndarray,
    *,
    limit: int,
) -> list[WorstColumn]:
    items: list[WorstColumn] = []
    for col, source in enumerate(source_rows):
        column_diff = diff[:, col]
        target = int(np.argmax(np.abs(column_diff)))
        col_ref = old[:, col]
        denom = float(np.linalg.norm(col_ref))
        if denom == 0.0:
            denom = 1.0
        source_art, source_link = _row_art_link(source, body_a, body_b, body_to_art, body_to_link, body_is_free)
        target_art, target_link = _row_art_link(target, body_a, body_b, body_to_art, body_to_link, body_is_free)
        items.append(
            WorstColumn(
                source_row=int(source),
                source_row_type=int(row_type[source]),
                source_body_a=int(body_a[source]),
                source_body_b=int(body_b[source]),
                source_articulation=source_art,
                source_link=source_link,
                target_row=target,
                target_row_type=int(row_type[target]),
                target_body_a=int(body_a[target]),
                target_body_b=int(body_b[target]),
                target_articulation=target_art,
                target_link=target_link,
                abs_linf=float(np.max(np.abs(column_diff))),
                rel_l2=float(np.linalg.norm(column_diff) / denom),
            )
        )
    items.sort(key=lambda item: item.abs_linf, reverse=True)
    return items[:limit]


def _link_errors(
    diff: np.ndarray,
    old: np.ndarray,
    source_rows: np.ndarray,
    body_a: np.ndarray,
    body_b: np.ndarray,
    body_to_art: np.ndarray,
    body_to_link: np.ndarray,
    body_is_free: np.ndarray,
) -> tuple[list[LinkError], list[DistanceError]]:
    by_source: dict[int, list[int]] = {}
    row_links = []
    for row in range(body_a.shape[0]):
        _art, link = _row_art_link(row, body_a, body_b, body_to_art, body_to_link, body_is_free)
        row_links.append(link)
    for col, source in enumerate(source_rows):
        link = row_links[int(source)]
        if link is None or link < 0:
            continue
        by_source.setdefault(int(link), []).append(col)

    source_errors: list[LinkError] = []
    for link, cols in sorted(by_source.items()):
        sub_diff = diff[:, cols]
        sub_old = old[:, cols]
        worst_flat = int(np.argmax(np.abs(sub_diff)))
        worst_target = worst_flat // len(cols)
        denom = float(np.linalg.norm(sub_old.reshape(-1)))
        if denom == 0.0:
            denom = 1.0
        source_errors.append(
            LinkError(
                source_link=link,
                sampled_rows=len(cols),
                max_abs=float(np.max(np.abs(sub_diff))),
                rel_l2=float(np.linalg.norm(sub_diff.reshape(-1)) / denom),
                worst_target_link=row_links[worst_target],
            )
        )

    distance_values: dict[int, list[float]] = {}
    for col, source in enumerate(source_rows):
        source_link = row_links[int(source)]
        if source_link is None or source_link < 0:
            continue
        for target, target_link in enumerate(row_links):
            if target_link is None or target_link < 0:
                continue
            distance = abs(int(target_link) - int(source_link))
            distance_values.setdefault(distance, []).append(float(abs(diff[target, col])))
    distance_errors = [
        DistanceError(
            link_distance=distance,
            entries=len(values),
            max_abs=float(np.max(values)),
            rms_abs=float(np.sqrt(np.mean(np.square(values)))),
        )
        for distance, values in sorted(distance_values.items())
        if values
    ]
    return source_errors, distance_errors


def _diagnose_case(
    case: BenchCase,
    *,
    device: str,
    dt: float,
    enable_contact_friction: bool,
    max_sources: int,
    abs_tol: float,
    rel_tol: float,
    worst_limit: int,
) -> list[OperatorDiagnosticResult]:
    model, initial, contacts = _prepare_case_run(case, device)
    old_solver = _make_solver(
        model,
        case,
        "mf_immediate",
        enable_contact_friction=enable_contact_friction,
    )
    propagation_solver = _make_solver(
        model,
        case,
        "propagation",
        enable_contact_friction=enable_contact_friction,
    )
    _run_zero_iteration_setup(model, old_solver, contacts, initial, dt=dt)
    _run_zero_iteration_setup(model, propagation_solver, contacts, initial, dt=dt)

    dense_counts = old_solver.constraint_count.numpy().astype(np.int32, copy=False)
    propagation_counts = propagation_solver.propagation_constraint_count.numpy().astype(np.int32, copy=False)
    body_to_art, body_to_link, body_is_free = _body_link_maps(propagation_solver)
    rigid_contacts = int(contacts.rigid_contact_count.numpy()[0])

    results: list[OperatorDiagnosticResult] = []
    for world in range(old_solver.world_count):
        dense_rows = int(dense_counts[world])
        propagation_rows = int(propagation_counts[world])
        if dense_rows == 0 and propagation_rows == 0:
            continue
        if dense_rows != propagation_rows:
            raise RuntimeError(
                f"{case.label} world {world}: dense rows {dense_rows} != propagation rows {propagation_rows}"
            )

        source_rows = _select_sources(dense_rows, max_sources)
        old_full = _world_operator_from_jy(old_solver, world)
        dense_rows_for_propagation, row_order_identity = _dense_rows_in_propagation_order(
            old_solver, propagation_solver, world, dense_rows
        )
        old_aligned = old_full[np.ix_(dense_rows_for_propagation, dense_rows_for_propagation)]
        old = old_aligned[:, source_rows]
        propagation, local_only, propagation_qd = _propagation_operator_columns(propagation_solver, world, source_rows)
        diff = propagation - old
        local_diff = local_only - old
        width = int(old_solver.world_dof_count.numpy()[world])
        old_Y = old_solver.Y_world.numpy()[world, :dense_rows, :width].astype(np.float64, copy=True)
        old_qd = old_Y[dense_rows_for_propagation[source_rows]].T
        nonfree_qd_mask, mass_condition = _nonfree_world_dof_mask_and_condition(old_solver, world)
        qd_diff = propagation_qd[nonfree_qd_mask] - old_qd[nonfree_qd_mask]

        row_type_old_raw = old_solver.row_type.numpy()[world, :dense_rows].astype(np.int32, copy=True)
        row_parent_old_raw = old_solver.row_parent.numpy()[world, :dense_rows].astype(np.int32, copy=True)
        old_to_propagation = np.full(dense_rows, -1, dtype=np.int32)
        for propagation_row, dense_row in enumerate(dense_rows_for_propagation):
            old_to_propagation[int(dense_row)] = propagation_row
        row_type_old = row_type_old_raw[dense_rows_for_propagation]
        row_parent_old_selected = row_parent_old_raw[dense_rows_for_propagation]
        row_parent_old = np.full_like(row_parent_old_selected, -1)
        parent_mask = row_parent_old_selected >= 0
        row_parent_old[parent_mask] = old_to_propagation[row_parent_old_selected[parent_mask]]
        row_type_propagation = propagation_solver.propagation_row_type.numpy()[world, :dense_rows].astype(
            np.int32, copy=True
        )
        row_parent_propagation = propagation_solver.propagation_row_parent.numpy()[world, :dense_rows].astype(
            np.int32, copy=True
        )
        row_type_mismatch = int(np.count_nonzero(row_type_old != row_type_propagation))
        row_parent_mismatch = int(np.count_nonzero(row_parent_old != row_parent_propagation))

        body_a = propagation_solver.propagation_body_a.numpy()[world, :dense_rows].astype(np.int32, copy=True)
        body_b = propagation_solver.propagation_body_b.numpy()[world, :dense_rows].astype(np.int32, copy=True)
        row_type = row_type_propagation
        worst = _worst_columns(
            diff,
            old,
            source_rows,
            row_type,
            body_a,
            body_b,
            body_to_art,
            body_to_link,
            body_is_free,
            limit=worst_limit,
        )
        per_source_link, per_link_distance = _link_errors(
            diff,
            old,
            source_rows,
            body_a,
            body_b,
            body_to_art,
            body_to_link,
            body_is_free,
        )

        diag_old = np.array([old_aligned[int(source), int(source)] for source in source_rows], dtype=np.float64)
        diag_propagation = np.array(
            [propagation[int(source), col] for col, source in enumerate(source_rows)], dtype=np.float64
        )
        diag_diff = diag_propagation - diag_old

        abs_linf = float(np.max(np.abs(diff))) if diff.size else 0.0
        rel_fro = _rel_l2(propagation, old)
        status = "operator_match" if abs_linf <= abs_tol and rel_fro <= rel_tol else "operator_mismatch"

        results.append(
            OperatorDiagnosticResult(
                label=case.label,
                links=case.links,
                contacts_per_articulation=case.contacts_per_articulation,
                world=world,
                rows=dense_rows,
                rigid_contacts=rigid_contacts,
                D=int(old_solver.world_dof_count.numpy()[world]),
                sampled_sources=len(source_rows),
                all_sources=len(source_rows) == dense_rows,
                row_alignment_complete=True,
                row_order_identity=row_order_identity,
                row_metadata_match=row_type_mismatch == 0 and row_parent_mismatch == 0,
                row_type_mismatch_count=row_type_mismatch,
                row_parent_mismatch_count=row_parent_mismatch,
                operator_status=status,
                rel_fro=rel_fro,
                abs_linf=abs_linf,
                diag_rel_l2=_rel_l2(diag_propagation, diag_old),
                diag_abs_linf=float(np.max(np.abs(diag_diff))) if diag_diff.size else 0.0,
                offdiag_rel_l2=_offdiag_rel(propagation, old, source_rows),
                qd_rel_l2=_rel_l2(propagation_qd[nonfree_qd_mask], old_qd[nonfree_qd_mask])
                if np.any(nonfree_qd_mask)
                else 0.0,
                qd_abs_linf=float(np.max(np.abs(qd_diff))) if qd_diff.size else 0.0,
                mass_condition_est=mass_condition,
                local_vs_propagated_rel_fro=_rel_l2(local_only, old),
                local_vs_propagated_abs_linf=float(np.max(np.abs(local_diff))) if local_diff.size else 0.0,
                worst_columns=worst,
                per_source_link=per_source_link,
                per_link_distance=per_link_distance,
            )
        )
    return results


def _write_summary(path: Path, results: list[OperatorDiagnosticResult], args: argparse.Namespace) -> None:
    any_mismatch = any(result.operator_status != "operator_match" for result in results)
    lines = [
        "# FPGS Articulation Operator Diagnostic",
        "",
        "This compares the tiled D-wide reference row operator `J_world @ Y_world.T` with the propagation operator formed by unit row impulse injection plus the actual propagation kernel.",
        "",
        f"Classification: {'operator mismatch found' if any_mismatch else 'all sampled operators match'}.",
        f"Tolerance: abs_linf <= {args.abs_tol:g} and rel_fro <= {args.rel_tol:g}.",
        f"Friction rows: {'on' if not args.no_friction else 'off'}.",
        "",
        "If this table reports `operator_match`, end-state differences at finite PGS iterations are convergence or ordering effects. If it reports `operator_mismatch`, the propagation operator differs before PGS convergence is relevant.",
        "",
        "| case | world | D | rows | contacts | sampled cols | row order | status | rel Fro | abs Linf | diag rel | diag Linf | offdiag rel | qd rel | qd Linf | cond est | local-only rel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        sampled = str(result.sampled_sources) if result.all_sources else f"{result.sampled_sources}/{result.rows}"
        row_order = "identity" if result.row_order_identity else "aligned"
        lines.append(
            f"| {result.label} | {result.world} | {result.D} | {result.rows} | {result.rigid_contacts} | {sampled} | "
            f"{row_order} | {result.operator_status} | {result.rel_fro:.6g} | {result.abs_linf:.6g} | "
            f"{result.diag_rel_l2:.6g} | {result.diag_abs_linf:.6g} | {result.offdiag_rel_l2:.6g} | "
            f"{result.qd_rel_l2:.6g} | {result.qd_abs_linf:.6g} | "
            f"{'' if result.mass_condition_est is None else f'{result.mass_condition_est:.6g}'} | "
            f"{result.local_vs_propagated_rel_fro:.6g} |"
        )

    lines.extend(["", "## Worst Columns", ""])
    for result in results:
        lines.append(f"### {result.label}, world {result.world}")
        if not result.row_metadata_match:
            lines.append(
                f"Row metadata mismatch: row_type={result.row_type_mismatch_count}, row_parent={result.row_parent_mismatch_count}."
            )
        lines.append("")
        lines.append("| source row | source link | target row | target link | abs Linf | rel L2 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for item in result.worst_columns:
            source_link = "" if item.source_link is None else str(item.source_link)
            target_link = "" if item.target_link is None else str(item.target_link)
            lines.append(
                f"| {item.source_row} | {source_link} | {item.target_row} | {target_link} | "
                f"{item.abs_linf:.6g} | {item.rel_l2:.6g} |"
            )
        lines.append("")

    lines.extend(["## Per Source Link", ""])
    for result in results:
        lines.append(f"### {result.label}, world {result.world}")
        lines.append("")
        lines.append("| source link | sampled rows | worst target link | max abs | rel L2 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for item in result.per_source_link:
            worst_target = "" if item.worst_target_link is None else str(item.worst_target_link)
            lines.append(
                f"| {item.source_link} | {item.sampled_rows} | {worst_target} | {item.max_abs:.6g} | {item.rel_l2:.6g} |"
            )
        lines.append("")

    lines.extend(["## Per Link Distance", ""])
    for result in results:
        lines.append(f"### {result.label}, world {result.world}")
        lines.append("")
        lines.append("| link distance | entries | max abs | rms abs |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for item in result.per_link_distance:
            lines.append(f"| {item.link_distance} | {item.entries} | {item.max_abs:.6g} | {item.rms_abs:.6g} |")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--links", type=_parse_int_list, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument(
        "--contacts-per-articulation",
        default="same",
        help="'same', one integer, or a comma-separated list matching --links",
    )
    parser.add_argument("--max-sources", type=int, default=0, help="0 means all source columns")
    parser.add_argument("--abs-tol", type=float, default=1.0e-4)
    parser.add_argument("--rel-tol", type=float, default=1.0e-3)
    parser.add_argument("--joint-armature", type=float, default=0.0)
    parser.add_argument("--worst-limit", type=int, default=8)
    parser.add_argument("--no-friction", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/fpgs_articulation_row_scaling/operator_diagnostic"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("This diagnostic is intended for CUDA devices")

    contacts = _contacts_for_links(args.links, args.contacts_per_articulation)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[OperatorDiagnosticResult] = []
    for links, contact_count in zip(args.links, contacts, strict=True):
        case = BenchCase(
            sweep="operator_diagnostic",
            label=f"L={links},boxes={contact_count},armature={args.joint_armature:g}",
            case_kind="articulated_free",
            articulations=1,
            links=links,
            contacts_per_articulation=contact_count,
            joint_armature=args.joint_armature,
        )
        results.extend(
            _diagnose_case(
                case,
                device=args.device,
                dt=args.dt,
                enable_contact_friction=not args.no_friction,
                max_sources=args.max_sources,
                abs_tol=args.abs_tol,
                rel_tol=args.rel_tol,
                worst_limit=args.worst_limit,
            )
        )

    rows = [asdict(result) for result in results]
    (args.out_dir / "operator_diagnostic.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_summary(args.out_dir / "summary.md", results, args)
    print(f"Wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
