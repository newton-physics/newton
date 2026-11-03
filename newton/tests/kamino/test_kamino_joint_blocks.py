# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regressions for joint-block solves and conditional Schur dispatch."""

import unittest
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.solvers.dvi import sparse
from newton._src.solvers.kamino._src.solvers.dvi.joint_blocks import (
    JointBlockSolver,
    _build_body_masks,
    _build_topology,
)


def _fixture(device, *, topology="chain"):
    """Construct two padded worlds with interleaved five/six/five-row groups."""
    rng = np.random.default_rng(94017)
    worlds, n, ld, stride = 2, 16, 20, 9
    groups = [[0, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11], [2, 12, 13, 14, 15]]
    edges = {(0, 1), (1, 2)}
    if topology != "chain":
        lengths = [5, 6] * (2 if topology == "cycle" else 7)
        starts = np.cumsum([0, *lengths])
        groups = [list(range(start, end)) for start, end in pairwise(starts)]
        n, ld = int(starts[-1]), int(starts[-1]) + 4
        edges = {(0, 1), (1, 2), (2, 3), (0, 3)}
        edges.update((i, j) for i in range(4, len(groups)) for j in range(i + 1, len(groups)))
    origin = max(64, 1 << (n - 1).bit_length())

    def ints(values):
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    def floats(values):
        return wp.array(np.asarray(values, dtype=np.float32).ravel(), dtype=wp.float32, device=device)

    mio = np.array([3, 3 + ld * ld])
    vio = np.array([2, 2 + n + 3])
    rio = np.array([5, 5 + n * stride + 3])
    nus = np.array([3, 7])
    matrix = np.full(int(mio[-1] + ld * ld + 5), 71.0, dtype=np.float32)
    dense = []
    pair_world, pair_row, pair_col = [], [], []
    for world in range(worlds):
        a = np.zeros((n, n))
        for i, rows in enumerate(groups):
            for j, cols in enumerate(groups):
                if i == j or (min(i, j), max(i, j)) in edges:
                    a[np.ix_(rows, cols)] = rng.normal(0, 0.03, (len(rows), len(cols)))
        a = (a + a.T) * 0.5 + np.eye(n) * (2 + world)
        dense.append(a)
        for row in range(n):
            matrix[mio[world] + row * ld : mio[world] + row * ld + row + 1] = a[row, : row + 1]
        for i in range(n):
            for j in range(i + 1, n):
                if a[i, j] != 0:
                    pair_world.append(world)
                    pair_row.append(i)
                    pair_col.append(j)
    info = SimpleNamespace(dim=ints([n, n]), mio=ints(mio), vio=ints(vio), maxdim=ints([ld, ld]))
    operator = SimpleNamespace(info=info, mat=floats(matrix))
    scale = rng.uniform(0.5, 2.0, int(vio[-1] + n + 5)).astype(np.float32)
    coupling = np.full(int(rio[-1] + n * stride + 5), 83.0, dtype=np.float32)
    for world, nu in enumerate(nus):
        coupling[rio[world] : rio[world] + n * nu] = rng.normal(size=n * nu)
    state = SimpleNamespace(
        bilateral_response_stride=ints([stride, stride]),
        bilateral_response_mio=ints(rio),
        bilateral_preconditioner=floats(scale),
        bilateral_coupling=floats(coupling),
        bilateral_response=wp.full(len(coupling), -123.0, dtype=wp.float32, device=device),
    )
    joints = SimpleNamespace(
        wid=ints([world for world in range(worlds) for _ in groups]),
        num_dynamic_cts=ints([1] * (worlds * len(groups))),
        num_kinematic_cts=ints([len(group) - 1 for _ in range(worlds) for group in groups]),
        dynamic_cts_offset_total_cts=ints([world * origin + group[0] for world in range(worlds) for group in groups]),
        kinematic_cts_offset_total_cts=ints([world * origin + group[1] for world in range(worlds) for group in groups]),
    )
    body_sets = ((0,), (0, 1), (1,)) if topology == "chain" else [(0,)] * len(groups)
    owners = {row: group for group, rows in enumerate(groups) for row in rows}
    joint_coords = [(row, 6 * body) for row in range(n) for body in body_sets[owners[row]]]
    block_stride = len(joint_coords) + stride
    coords = [*joint_coords, *((n + col, 0) for col in range(stride))] * worlds
    bsm = SimpleNamespace(
        nzb_coords=wp.array(coords, dtype=wp.int32, device=device),
        nzb_start=ints(np.arange(worlds) * block_stride),
        num_nzb=ints([block_stride] * worlds),
    )
    path = SimpleNamespace(
        device=device,
        size=SimpleNamespace(num_worlds=worlds, max_of_num_bodies=2),
        jacobians=SimpleNamespace(
            _J_cts=SimpleNamespace(bsm=bsm), joint_constraint_nzb_count=ints([len(joint_coords)] * worlds)
        ),
        model=SimpleNamespace(
            joints=joints,
            info=SimpleNamespace(total_cts_offset=ints([0, origin]), bodies_offset=ints([0, 2])),
        ),
        data=SimpleNamespace(bilateral_dim=info.dim, bilateral_operator=operator, state=state),
        bilateral_nzb_pairs=(ints(pair_world), ints(pair_row), ints(pair_col), *[ints(np.zeros(len(pair_world)))] * 3),
        bilateral_entry_starts=ints(np.arange(len(pair_world) + 1)),
    )
    problem = SimpleNamespace(data=SimpleNamespace(dim=ints(n + nus), njc=info.dim))
    return path, problem, dense, matrix, mio, vio, rio, nus, scale, coupling


def _packed_matrix(solver, matrix, mio, ld):
    """Pack a CPU matrix reference, including identity padding for short groups."""
    order = solver._schedule["order"].numpy()
    rows = solver._schedule["rows"].numpy()
    cols = solver._schedule["cols"].numpy()
    packed = np.zeros((len(mio), solver._blocks, 6, 6), dtype=np.float32)
    for world, offset in enumerate(mio):
        for block, (row, col) in enumerate(zip(rows, cols, strict=True)):
            for i in range(6):
                for j in range(6):
                    original_i, original_j = order[row * 6 + i], order[col * 6 + j]
                    if original_i >= 0 and original_j >= 0:
                        packed[world, block, i, j] = matrix[
                            offset + max(original_i, original_j) * ld + min(original_i, original_j)
                        ]
                    elif row == col and i == j:
                        packed[world, block, i, j] = 1.0
    return packed.ravel()


class TestKaminoJointBlocks(unittest.TestCase):
    def _device(self, conditional=False):
        """Select a CUDA device with the capabilities required by the test."""
        if not wp.get_cuda_device_count():
            self.skipTest("Requires CUDA tile solves")
        device = wp.get_cuda_devices()[0]
        if conditional and not wp.is_conditional_graph_supported():
            self.skipTest("Requires CUDA conditional graphs")
        return device

    def test_mixed_group_dense_reference(self):
        """Match dense solves and response Grams with padding and inactive worlds."""
        device = self._device()
        path, problem, matrices, matrix, mio, vio, rio, nus, scale, coupling = _fixture(device)
        solver = JointBlockSolver.create(path)
        self.assertIsNotNone(solver)
        solver._matrix.assign(_packed_matrix(solver, matrix, mio, 20))
        solver.prepare(path, problem)
        self.assertEqual(int(solver.failure.numpy()[0]), 0)
        rng = np.random.default_rng(419)
        rhs = rng.normal(size=len(scale)).astype(np.float32)
        rhs_wp = wp.array(rhs, device=device)
        out = wp.full(len(rhs), -123.0, dtype=wp.float32, device=device)
        active = wp.array([16, 0], dtype=wp.int32, device=device)
        solver.solve(rhs_wp, out, active)
        result = out.numpy()
        np.testing.assert_allclose(
            result[vio[0] : vio[0] + 16], np.linalg.solve(matrices[0], rhs[vio[0] : vio[0] + 16]), atol=2e-6, rtol=2e-6
        )
        untouched = np.ones(len(rhs), dtype=bool)
        untouched[vio[0] : vio[0] + 16] = False
        np.testing.assert_array_equal(result[untouched], -123.0)
        solver.solve(rhs_wp, out)
        for world in range(2):
            np.testing.assert_allclose(
                out.numpy()[vio[world] : vio[world] + 16],
                np.linalg.solve(matrices[world], rhs[vio[world] : vio[world] + 16]),
                atol=2e-6,
                rtol=2e-6,
            )
        # This algebraic oracle deliberately uses dense, nonphysical coupling.
        solver._body_reach = None
        solver.response(path, problem)
        response = path.data.state.bilateral_response.numpy()
        untouched = np.ones(len(response), dtype=bool)
        for world, nu in enumerate(nus):
            start = rio[world]
            y = response[start : start + 16 * nu].reshape(16, nu).astype(np.float64)
            b = coupling[start : start + 16 * nu].reshape(16, nu) * scale[vio[world] : vio[world] + 16, None]
            expected = b.T @ np.linalg.solve(matrices[world], b)
            np.testing.assert_allclose(y.T @ y, expected, atol=2e-5, rtol=3e-6)
            untouched[start : start + 16 * nu] = False
        np.testing.assert_array_equal(response[untouched], -123.0)

    def test_persistent_filled_cycle_dense_reference(self):
        """Factor filled ragged cycles exactly, including panels wider than eight warps."""
        device = self._device()
        for topology in ("cycle", "wide"):
            with self.subTest(topology=topology):
                path, problem, matrices, matrix, mio, vio, *_ = _fixture(device, topology=topology)
                metadata, reason = _build_topology(path, *path.bilateral_nzb_pairs[:3])
                self.assertIsNone(metadata)
                self.assertIn("nonchordal", reason)
                metadata, reason = _build_topology(path, *path.bilateral_nzb_pairs[:3], allow_fill=True)
                self.assertIsNotNone(metadata, reason)
                groups = 4 if topology == "cycle" else 14
                original_edges = 4 if topology == "cycle" else 49
                self.assertEqual(metadata["nb"], groups + original_edges + 1)
                with patch(
                    "newton._src.solvers.kamino._src.solvers.dvi.joint_blocks._PERSISTENT_MAX_BLOCKS",
                    metadata["nb"] - 1,
                ):
                    rejected, reason = _build_topology(path, *path.bilateral_nzb_pairs[:3], allow_fill=True)
                    self.assertIsNone(rejected)
                    self.assertIn("budget", reason)
                solver = JointBlockSolver(path, metadata, persistent=True)
                n, ld = matrices[0].shape[0], int(path.data.bilateral_operator.info.maxdim.numpy()[0])
                packed = _packed_matrix(solver, matrix, mio, ld)
                solver._matrix.assign(packed)
                solver.prepare(path, problem)
                self.assertEqual(int(solver.failure.numpy()[0]), 0)
                np.testing.assert_array_equal(solver._matrix.numpy(), packed)
                order = solver._schedule["order"].numpy()
                rows, cols = solver._schedule["rows"].numpy(), solver._schedule["cols"].numpy()
                factors = solver._factor.numpy().reshape(2, solver._blocks, 6, 6)
                active = np.flatnonzero(order >= 0)
                for world in range(2):
                    lower = np.zeros((groups * 6, groups * 6))
                    for block, (row, col) in enumerate(zip(rows, cols, strict=True)):
                        value = factors[world, block]
                        lower[row * 6 : row * 6 + 6, col * 6 : col * 6 + 6] = np.tril(value) if row == col else value
                    expected = np.eye(groups * 6)
                    expected[np.ix_(active, active)] = matrices[world][np.ix_(order[active], order[active])]
                    np.testing.assert_allclose(lower @ lower.T, expected, atol=2e-6, rtol=2e-6)
                rng = np.random.default_rng(83051)
                rhs = rng.normal(size=int(vio[-1] + n + 5)).astype(np.float32)
                out = wp.full(len(rhs), -123.0, dtype=wp.float32, device=device)
                solver.solve(wp.array(rhs, device=device), out)
                result, untouched = out.numpy(), np.ones(len(rhs), dtype=bool)
                for world, offset in enumerate(vio):
                    np.testing.assert_allclose(
                        result[offset : offset + n],
                        np.linalg.solve(matrices[world], rhs[offset : offset + n]),
                        atol=2e-6,
                        rtol=2e-6,
                    )
                    untouched[offset : offset + n] = False
                np.testing.assert_array_equal(result[untouched], -123.0)

    def test_persistent_capture_changes_and_recovers(self):
        """Refresh filled factors on replay and preserve inactive worlds through fallback recovery."""
        device = self._device(conditional=True)
        path, problem, matrices, matrix, mio, vio, *_ = _fixture(device, topology="cycle")
        metadata, _ = _build_topology(path, *path.bilateral_nzb_pairs[:3], allow_fill=True)
        solver = JointBlockSolver(path, metadata, persistent=True)
        n, ld = matrices[0].shape[0], int(path.data.bilateral_operator.info.maxdim.numpy()[0])
        rhs = wp.ones(int(vio[-1] + n + 5), dtype=wp.float32, device=device)
        out = wp.full(rhs.size, -123.0, dtype=wp.float32, device=device)
        active = wp.array([n, 0], dtype=wp.int32, device=device)
        solver._matrix.assign(_packed_matrix(solver, matrix, mio, ld))
        solver.prepare(path, problem)
        solver.solve(rhs, out, active)
        with wp.ScopedCapture(device=device) as capture:
            solver.prepare(path, problem)
            wp.capture_if(
                solver.failure, on_true=lambda: out.fill_(-777.0), on_false=lambda: solver.solve(rhs, out, active)
            )
        for case in ("changed", "factor", "valid", "zeros", "valid"):
            dense = [value.copy() for value in matrices]
            if case == "changed":
                dense = [value * 1.7 for value in dense]
            elif case == "factor":
                dense[1][0, 0] = -1.0
            elif case == "zeros":
                dense = [np.diag(np.diag(value)) for value in dense]
            current = matrix.copy()
            for world, offset in enumerate(mio):
                for row in range(n):
                    current[offset + row * ld : offset + row * ld + row + 1] = dense[world][row, : row + 1]
            solver._matrix.assign(_packed_matrix(solver, current, mio, ld))
            out.fill_(-123.0)
            wp.capture_launch(capture.graph)
            self.assertEqual(bool(solver.failure.numpy()[0]), case == "factor")
            result = out.numpy()
            if case == "factor":
                np.testing.assert_array_equal(result, -777.0)
            else:
                np.testing.assert_allclose(
                    result[vio[0] : vio[0] + n], np.linalg.solve(dense[0], np.ones(n)), atol=2e-6, rtol=2e-6
                )
                untouched = np.ones(len(result), dtype=bool)
                untouched[vio[0] : vio[0] + n] = False
                np.testing.assert_array_equal(result[untouched], -123.0)

    def test_response_reach_tracks_contact_body_changes(self):
        """Match unpruned ragged responses and erase stale groups as contacts move."""
        device = self._device()
        path, problem, matrices, matrix, mio, vio, rio, _, scale, coupling = _fixture(device)
        solver = JointBlockSolver.create(path)
        self.assertIsNotNone(solver._body_reach)
        solver._matrix.assign(_packed_matrix(solver, matrix, mio, 20))

        def ints(values):
            return wp.array(values, dtype=wp.int32, device=device)

        bsm = path.jacobians._J_cts.bsm
        coords = bsm.nzb_coords.numpy()
        starts = bsm.nzb_start.numpy()
        counts = path.jacobians.joint_constraint_nzb_count.numpy()
        contact_offsets = starts + counts
        # One three-component body-ground contact per world; remaining slots
        # deliberately retain unrelated capacity entries.
        problem.data.dim.assign([19, 19])
        for name in ("nbc", "nl", "bcio", "lio"):
            setattr(problem.data, name, ints([0, 0]))
        problem.data.cio = ints([0, 1])
        problem.delassus = SimpleNamespace(bsm=bsm)
        path.data.state.limit_indices = ints([-1])
        path.data.state.contact_indices = ints([0, 1])
        path.jacobians.bounded_constraint_nzb_offsets = wp.array([[-1, -1]], dtype=wp.vec2i, device=device)
        path.jacobians.limit_constraint_nzb_offsets = ints([-1])
        path.jacobians.contact_constraint_nzb_offsets = ints(contact_offsets)
        solver.prepare(path, problem)
        self.assertEqual(int(solver.failure.numpy()[0]), 0)
        rng = np.random.default_rng(2751)
        values = rng.normal(size=(2, 16, 3)).astype(np.float32)
        groups = [[0, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11], [2, 12, 13, 14, 15]]
        state = path.data.state
        output = state.bilateral_response
        reference = wp.empty_like(output)
        reach = solver._body_reach
        output.fill_(float("nan"))
        active = np.zeros(output.size, dtype=bool)
        for start in rio:
            active[start : start + 16 * 3] = True
        for iteration, body in enumerate((0, 1, 1, 0)):
            if iteration == 2:
                output.fill_(float("nan"))
            rows = groups[body] + groups[body + 1]
            for world in range(2):
                coords[contact_offsets[world] : contact_offsets[world] + 3, 1] = 6 * body
                rhs = np.zeros((16, 3), dtype=np.float32)
                rhs[rows] = values[world, rows]
                coupling[rio[world] : rio[world] + 16 * 3] = rhs.ravel()
            bsm.nzb_coords.assign(coords)
            state.bilateral_coupling.assign(coupling)
            solver.response(path, problem)
            actual = output.numpy()
            state.bilateral_response = reference
            solver._body_reach = None
            solver.response(path, problem)
            expected = reference.numpy()
            state.bilateral_response = output
            solver._body_reach = reach
            np.testing.assert_allclose(actual[active], expected[active], atol=2e-6, rtol=2e-6)
            self.assertTrue(np.isnan(actual[~active]).all())
            for world, start in enumerate(rio):
                y = actual[start : start + 16 * 3].reshape(16, 3).astype(np.float64)
                b = coupling[start : start + 16 * 3].reshape(16, 3) * scale[vio[world] : vio[world] + 16, None]
                np.testing.assert_allclose(y.T @ y, b.T @ np.linalg.solve(matrices[world], b), atol=2e-5, rtol=3e-6)

    def test_response_reach_rejects_unsupported_topology(self):
        """Keep unpruned response support for oversized or heterogeneous body maps."""
        device = self._device()
        path, *_ = _fixture(device)
        metadata, _ = _build_topology(path, *path.bilateral_nzb_pairs[:3])
        self.assertIsNotNone(_build_body_masks(path, metadata))
        self.assertIsNone(_build_body_masks(path, {**metadata, "ng": 65}))
        bsm = path.jacobians._J_cts.bsm
        coords = bsm.nzb_coords.numpy()
        start = int(bsm.nzb_start.numpy()[1])
        count = int(path.jacobians.joint_constraint_nzb_count.numpy()[1])
        # Both worlds retain the same joint graph, but body labels no longer
        # describe the same group incidence and cannot share reach masks.
        coords[start : start + count, 1] = 6 - coords[start : start + count, 1]
        bsm.nzb_coords.assign(coords)
        solver = JointBlockSolver.create(path)
        self.assertIsNotNone(solver)
        self.assertIsNone(solver._body_reach)

    def test_direct_assembly_matches_original(self):
        """Match original assembly after Jacobian, inertia, and compliance changes."""
        device = self._device()
        path, problem, _, _, mio, vio, *_ = _fixture(device)
        rng = np.random.default_rng(80149)
        groups = [[0, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11], [2, 12, 13, 14, 15]]
        owners = {row: group for group, rows in enumerate(groups) for row in rows}
        body_sets = ((0,), (0, 1), (1,))
        block_ids, entries = {}, []
        for world in range(2):
            for row in range(16):
                for body in body_sets[owners[row]]:
                    block_ids[world, row, body] = len(entries)
                    entries.append((world, row, body))
        pairs = [[] for _ in range(6)]
        for world in range(2):
            for row in range(16):
                for col in range(row + 1, 16):
                    for body in body_sets[owners[row]]:
                        if body in body_sets[owners[col]]:
                            values = (
                                world,
                                row,
                                col,
                                world * 2 + body,
                                block_ids[world, row, body],
                                block_ids[world, col, body],
                            )
                            for field, value in zip(pairs, values, strict=True):
                                field.append(value)
        pairs, starts = sparse.group_bilateral_pairs(pairs)

        def ints(values):
            return wp.array(values, dtype=wp.int32, device=device)

        path.bilateral_nzb_pairs = tuple(ints(values) for values in pairs)
        path.bilateral_entry_starts = ints(starts)
        path.size.max_of_num_bilateral_joint_cts = 16
        problem.data.vio = ints([0, 64])
        diagonal = wp.zeros(128, dtype=wp.float32, device=device)
        path.data.state.scratch = wp.empty_like(diagonal)
        jacobian = wp.empty(len(entries), dtype=vec6f, device=device)
        problem.delassus = SimpleNamespace(
            constraint_jacobian=SimpleNamespace(nzb_values=jacobian),
            diagonal=lambda destination: wp.copy(destination, diagonal),
        )
        inv_mass = wp.empty(4, dtype=wp.float32, device=device)
        inv_inertia = wp.empty(4, dtype=wp.mat33f, device=device)
        path.model.bodies = SimpleNamespace(inv_m_i=inv_mass)
        path.model_data = SimpleNamespace(bodies=SimpleNamespace(inv_I_i=inv_inertia))
        # This fixture contains only joint Jacobians, without reserved contact slots.
        path.jacobians._J_cts.bsm.nzb_start = ints([0, len(entries) // 2])
        solver = JointBlockSolver.create(path)
        self.assertIsNotNone(solver)
        self.assertIsNotNone(solver._assembly_template)
        self.assertIs(solver._entry_target, solver._assembly_template[-1])
        with patch(
            "newton._src.solvers.kamino._src.solvers.dvi.joint_blocks._build_assembly_template", return_value=None
        ):
            ordinary = JointBlockSolver.create(path)
        self.assertIsNone(ordinary._assembly_template)
        self.assertEqual(ordinary._entry_target.size, 2 * solver._entry_target.size)
        raw_values = rng.normal(0, 0.2, (len(entries), 6)).astype(np.float32)
        reference = path.data.bilateral_operator.mat
        for iteration in range(2):
            values = raw_values.copy()
            if iteration:
                for index, (_, row, _) in enumerate(entries):
                    if row in (0, 7):
                        values[index] = 0.0
            masses = np.array([0.5, 0.7, 0.9, 1.1], dtype=np.float32) * (iteration + 1)
            inertias = np.array([np.diag([0.2 + i * 0.1, 0.7, 1.2]) for i in range(4)], dtype=np.float32)
            inertias[:, 0, 1] = inertias[:, 1, 0] = 0.03 * (iteration + 1)
            raw_diagonal = np.zeros(128, dtype=np.float32)
            for index, (world, row, body) in enumerate(entries):
                v, angular = values[index, :3], values[index, 3:]
                bid = world * 2 + body
                raw_diagonal[world * 64 + row] += masses[bid] * (v @ v) + angular @ inertias[bid] @ angular
            # Positive passive joint compliance is part of the original diagonal.
            for world in range(2):
                raw_diagonal[world * 64 : world * 64 + 3] += np.array([0.2, 0.3, 0.4]) * (iteration + 1)
            jacobian.assign(values)
            inv_mass.assign(masses)
            inv_inertia.assign(inertias)
            diagonal.assign(raw_diagonal)
            reference.zero_()
            sparse._assemble_sparse_bilateral_block(path, problem, reference)
            expected = _packed_matrix(solver, reference.numpy(), mio, 20)
            expected_scale = path.data.state.bilateral_preconditioner.numpy()
            path.data.state.bilateral_preconditioner.fill_(-99.0)
            solver.assemble(path, problem)
            shared_result = solver._matrix.numpy()
            np.testing.assert_allclose(shared_result, expected, atol=2e-6, rtol=2e-6)
            ordinary.assemble(path, problem)
            np.testing.assert_array_equal(ordinary._matrix.numpy(), shared_result)
            for world in range(2):
                np.testing.assert_allclose(
                    path.data.state.bilateral_preconditioner.numpy()[vio[world] : vio[world] + 16],
                    expected_scale[vio[world] : vio[world] + 16],
                    atol=1e-7,
                    rtol=1e-7,
                )

        # Matching joint graphs alone do not establish identical assembly metadata.
        original_pairs = [array.numpy() for array in path.bilateral_nzb_pairs]
        original_starts = path.bilateral_entry_starts.numpy()
        original_bodies = path.model.info.bodies_offset.numpy()
        original_nzb_starts = path.jacobians._J_cts.bsm.nzb_start.numpy()
        second_world = np.flatnonzero(original_pairs[0] == 1)
        double_entry = next(
            int(start) for start, end in pairwise(original_starts) if end - start == 2 and original_pairs[0][start] == 1
        )
        for mismatch in ("pair_order", "body_index", "body_offset", "jacobian_offset"):
            with self.subTest(mismatch=mismatch):
                changed = [array.copy() for array in original_pairs]
                bodies = original_bodies.copy()
                nzb_starts = original_nzb_starts.copy()
                if mismatch == "pair_order":
                    # Reverse two body contributions to the same matrix entry.
                    for array in changed:
                        array[double_entry : double_entry + 2] = array[double_entry : double_entry + 2][::-1]
                elif mismatch == "body_index":
                    changed[3][second_world[0]] = 2 + (changed[3][second_world[0]] - 2 + 1) % 2
                elif mismatch == "body_offset":
                    bodies[1] += 1
                else:
                    nzb_starts[1] += 1
                path.bilateral_nzb_pairs = tuple(ints(array) for array in changed)
                path.model.info.bodies_offset.assign(bodies)
                path.jacobians._J_cts.bsm.nzb_start.assign(nzb_starts)
                fallback = JointBlockSolver.create(path)
                self.assertIsNotNone(fallback)
                self.assertIsNone(fallback._assembly_template)
                np.testing.assert_array_equal(fallback._schedule["order"].numpy(), solver._schedule["order"].numpy())
                reference.zero_()
                sparse._assemble_sparse_bilateral_block(path, problem, reference)
                expected = _packed_matrix(fallback, reference.numpy(), mio, 20)
                fallback.assemble(path, problem)
                np.testing.assert_allclose(fallback._matrix.numpy(), expected, atol=2e-6, rtol=2e-6)
        path.bilateral_nzb_pairs = tuple(ints(array) for array in original_pairs)
        path.model.info.bodies_offset.assign(original_bodies)
        path.jacobians._J_cts.bsm.nzb_start.assign(original_nzb_starts)
        self.assertIsNotNone(JointBlockSolver.create(path)._assembly_template)

    def test_reject_nonchordal_topology(self):
        """Reject a four-cycle instead of silently dropping symbolic fill."""
        device = self._device()
        path, *_ = _fixture(device)

        def ints(values):
            return wp.array(values, dtype=wp.int32, device=device)

        path.model.joints = SimpleNamespace(
            wid=ints([0] * 4 + [1] * 4),
            num_dynamic_cts=ints([0] * 8),
            num_kinematic_cts=ints([4] * 8),
            dynamic_cts_offset_total_cts=ints([0] * 4 + [64] * 4),
            kinematic_cts_offset_total_cts=ints([0, 4, 8, 12, 64, 68, 72, 76]),
        )
        pairs = (ints([0] * 4 + [1] * 4), ints([0, 4, 8, 0] * 2), ints([4, 8, 12, 12] * 2))
        metadata, reason = _build_topology(path, *pairs)
        self.assertIsNone(metadata)
        self.assertIn("nonchordal", reason)

    def test_captured_failure_and_recovery(self):
        """Reset fallback selection across invalid factors, dimensions, and recovery."""
        device = self._device(conditional=True)
        path, problem, matrices, matrix, mio, vio, *_ = _fixture(device)
        solver = JointBlockSolver.create(path)
        solver._matrix.assign(_packed_matrix(solver, matrix, mio, 20))
        rhs = wp.ones(40, dtype=wp.float32, device=device)
        out = wp.zeros_like(rhs)
        solver.prepare(path, problem)
        solver.solve(rhs, out)
        with wp.ScopedCapture(device=device) as capture:
            solver.prepare(path, problem)
            wp.capture_if(solver.failure, on_true=lambda: out.fill_(-777.0), on_false=lambda: solver.solve(rhs, out))
        original_dim = problem.data.dim.numpy()
        for case in ("valid", "dimension", "valid", "factor", "valid"):
            current = matrix.copy()
            dims = original_dim.copy()
            if case == "dimension":
                dims[1] = 33
            elif case == "factor":
                current[mio[1]] = -1.0
            solver._matrix.assign(_packed_matrix(solver, current, mio, 20))
            problem.data.dim.assign(dims)
            wp.capture_launch(capture.graph)
            expected_failure = case != "valid"
            self.assertEqual(bool(solver.failure.numpy()[0]), expected_failure)
            result = out.numpy()
            if expected_failure:
                np.testing.assert_array_equal(result, -777.0)
            else:
                for world in range(2):
                    np.testing.assert_allclose(
                        result[vio[world] : vio[world] + 16],
                        np.linalg.solve(matrices[world], np.ones(16)),
                        atol=2e-6,
                        rtol=2e-6,
                    )

    def test_bilateral_only_dispatch_keeps_selected_factor(self):
        """Use the selected block factor when no unilateral constraints need elimination."""
        path = SimpleNamespace(
            data=SimpleNamespace(state=SimpleNamespace(projected_mio=None)),
            size=SimpleNamespace(
                max_of_num_bilateral_joint_cts=384,
                max_of_num_bounded_joint_cts=0,
                max_of_max_limits=0,
                max_of_max_contacts=0,
                num_worlds=1,
            ),
            bilateral_solver=object(),
            device=SimpleNamespace(is_cuda=True),
            max_alternating_iterations=4,
            has_unilateral_constraints=False,
        )
        problem = object()
        for blocks in (None, object()):
            with (
                self.subTest(blocks=blocks is not None),
                patch.object(sparse, "_can_use_cooperative_articulation", return_value=False),
                patch.object(sparse, "_factor_sparse_bilateral_block") as factor,
                patch.object(sparse, "_solve_sparse_bilateral_block") as solve,
                patch.object(sparse, "_compute_sparse_solution_vectors") as finish,
            ):
                sparse._solve_sparse_with_bilateral_schur_complement(path, problem, joint_blocks=blocks)
                self.assertEqual(factor.call_count, int(blocks is None))
                solve.assert_called_once_with(path, problem, joint_blocks=blocks)
                finish.assert_called_once_with(path, problem)

    def test_dispatch_refreshes_cached_values_before_conditional(self):
        """Refresh changing Jacobian values on both branches of every graph replay."""
        device = self._device(conditional=True)
        source = wp.array([1.0], dtype=wp.float32, device=device)
        cached = wp.zeros_like(source)
        output = wp.zeros_like(source)
        flag = wp.zeros(1, dtype=wp.int32, device=device)
        operator = SimpleNamespace(_needs_update=True)

        def update():
            wp.copy(cached, source)
            operator._needs_update = False

        operator.update = update
        blocks = SimpleNamespace(failure=flag, prepare=lambda path, problem: None, assemble=lambda path, problem: None)
        path = SimpleNamespace(
            use_schur_complement=True,
            joint_block_solver=blocks,
            device=device,
            data=SimpleNamespace(
                bilateral_dim=None, bilateral_operator=SimpleNamespace(info=SimpleNamespace(dim=None), mat=None)
            ),
        )

        def continuation(path, problem, joint_blocks=None):
            if operator._needs_update:
                operator.update()
            wp.copy(output, cached)

        with (
            patch.object(sparse, "_get_sparse_delassus", return_value=operator),
            patch.object(sparse, "_assemble_sparse_bilateral_block"),
            patch.object(sparse, "_solve_sparse_with_bilateral_schur_complement", side_effect=continuation),
        ):
            with wp.ScopedCapture(device=device) as capture:
                sparse._solve_sparse_with_bilateral_direct_block(path, SimpleNamespace())
        for failed, value in ((0, 2.0), (1, 3.0), (0, 5.0)):
            flag.fill_(failed)
            source.fill_(value)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(output.numpy(), [value])


if __name__ == "__main__":
    unittest.main()
