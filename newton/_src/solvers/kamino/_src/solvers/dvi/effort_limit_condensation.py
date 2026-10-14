# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conditionally eliminate interior effort rows from a compact Schur solve.

Factor the effort block M and subtract C.T @ inv(M) @ C from the remaining
system. Recover effort impulses after projected sweeps and certify against
the original bounds and projection law before replacing the original solve.
Rejected worlds retain their original inputs for ordinary PGS.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...linalg.factorize.llt_blocked_rcm import _sync_threads, get_float32_array_offset_ptr
from .kernels import _compact_schur_fits
from .projections import (
    contact_friction_normal_load,
    project_box_update,
    project_contact_normal_update,
    project_contact_tangent_update,
)
from .sparse_kernels import make_compact_schur_pgs_kernel
from .types import DVIConfigStruct, DVIStatus

wp.set_module_options({"enable_backward": False, "enable_mathdx_solver": False, "enable_mathdx_gemm": False})


@wp.struct
class _Inputs:
    contact_indices: wp.array[wp.int32]
    nbc: wp.array[wp.int32]
    nl: wp.array[wp.int32]
    nc: wp.array[wp.int32]
    bcio: wp.array[wp.int32]
    cio: wp.array[wp.int32]
    uio: wp.array[wp.int32]
    bcgo: wp.array[wp.int32]
    lcgo: wp.array[wp.int32]
    ccgo: wp.array[wp.int32]
    vio: wp.array[wp.int32]
    mu: wp.array[wp.float32]
    lower: wp.array[wp.float32]
    upper: wp.array[wp.float32]
    P: wp.array[wp.float32]
    vb: wp.array[wp.float32]
    raw_diag: wp.array[wp.float32]
    diag: wp.array[wp.float32]
    njc: wp.array[wp.int32]
    mio: wp.array[wp.int32]
    stride: wp.array[wp.int32]
    schur: wp.array[wp.float32]
    q: wp.array[wp.float32]
    colors: wp.array[wp.int32]
    ids: wp.array[wp.int32]
    color_starts: wp.array[wp.int32]
    group_starts: wp.array[wp.int32]
    config: wp.array[DVIConfigStruct]
    status: wp.array[DVIStatus]
    impulses: wp.array[wp.float32]


# Match the existing compact-PGS argument order without a second launch signature.
_INPUT_FIELDS = {
    1: "contact_indices",
    2: "nbc",
    3: "nl",
    4: "nc",
    5: "bcio",
    7: "cio",
    8: "uio",
    9: "bcgo",
    10: "lcgo",
    11: "ccgo",
    12: "vio",
    13: "mu",
    14: "lower",
    15: "upper",
    16: "P",
    17: "vb",
    18: "raw_diag",
    19: "diag",
    20: "njc",
    21: "mio",
    22: "stride",
    23: "schur",
    24: "q",
    25: "colors",
    26: "ids",
    27: "color_starts",
    28: "group_starts",
    29: "config",
    30: "status",
    31: "impulses",
}


def _view(arguments):
    result = _Inputs()
    for index, name in _INPUT_FIELDS.items():
        setattr(result, name, arguments[index])
    return result


@wp.struct
class _Workspace:
    efforts: wp.array[wp.int32]
    effort_count: wp.array[wp.int32]
    remaining: wp.array[wp.int32]
    eligible: wp.array[wp.int32]
    failure: wp.array[wp.int32]
    fallback_nc: wp.array[wp.int32]
    factor: wp.array[wp.float32]
    response: wp.array[wp.float32]
    gradient: wp.array[wp.float32]
    delta: wp.array[wp.float32]
    candidate: wp.array[wp.float32]
    candidate_q: wp.array[wp.float32]
    schedule_capacity: wp.int32


@wp.kernel
def _gather(source: _Inputs, target: _Inputs, work: _Workspace):
    w, lane = wp.tid()
    nm = work.effort_count[w]
    nb = source.nbc[w]
    nc = source.nc[w]
    nj = source.njc[w]
    nu = nb + 3 * nc
    nr = nu - nm
    cfg = source.config[w]
    valid = nm > 0 and nm <= 16 and nc > 0 and source.nl[w] == 0 and nr > 0 and nr <= 32
    valid = valid and source.bcgo[w] == nj and source.lcgo[w] == nj + nb and source.ccgo[w] == nj + nb
    valid = valid and nu <= 48 and _compact_schur_fits(nj, nu, source.stride[w])
    valid = valid and cfg.max_alternating_iterations * cfg.inequality_sweeps_per_iteration >= 32
    if valid:
        for i in range(nm):
            if work.efforts[w * 16 + i] < 0 or work.efforts[w * 16 + i] >= nb:
                valid = False
        for i in range(nc):
            if source.contact_indices[source.cio[w] + i] < 0:
                valid = False
    if lane == 0:
        work.eligible[w] = wp.where(valid, 1, 0)
        work.failure[w] = 0
        target.nc[w] = wp.where(valid, nc, 0)
        target.nbc[w] = wp.where(valid, nb - nm, 0)
        target.lcgo[w] = 32 + wp.where(valid, nb - nm, 0)
        target.ccgo[w] = target.lcgo[w]
    if not valid:
        return
    origin = source.vio[w] + nj
    for row in range(lane, nu, 32):
        removed = int(0)
        effort = bool(False)
        for i in range(nm):
            index = work.efforts[w * 16 + i]
            if index < row:
                removed += 1
            if index == row:
                effort = True
        if not effort:
            r = row - removed
            work.remaining[w * 32 + r] = row
            target.P[w * 64 + 32 + r] = source.P[origin + row]
            target.vb[w * 64 + 32 + r] = source.vb[origin + row]
            target.raw_diag[w * 64 + 32 + r] = source.raw_diag[origin + row]
            if row < nb:
                target.lower[w * 32 + r] = source.lower[source.bcio[w] + row]
                target.upper[w * 32 + r] = source.upper[source.bcio[w] + row]
    if lane < nc:
        target.mu[w * 32 + lane] = source.mu[source.cio[w] + lane]
    if lane == 0:
        capacity = work.schedule_capacity
        base = source.uio[w] + w
        output = w * (capacity + 1)
        ncolors = source.colors[w]
        good = ncolors >= 0 and ncolors <= capacity
        groups = int(0)
        if good:
            groups = source.color_starts[base + ncolors]
            good = groups >= 0 and groups <= capacity
        kept = int(0)
        target.group_starts[output] = 0
        if good:
            for color in range(ncolors + 1):
                group = source.color_starts[base + color]
                if group < 0 or group > groups:
                    good = False
                target.color_starts[output + color] = group
            for group in range(groups):
                first = source.group_starts[base + group]
                last = source.group_starts[base + group + 1]
                if first < 0 or last < first or last > nb + nc:
                    good = False
                else:
                    for slot in range(first, last):
                        uid = source.ids[source.uio[w] + slot]
                        removed = int(0)
                        effort = bool(False)
                        for i in range(nm):
                            index = work.efforts[w * 16 + i]
                            if index < uid:
                                removed += 1
                            if index == uid:
                                effort = True
                        if uid < 0 or uid >= nb + nc:
                            good = False
                        elif not effort:
                            if kept < capacity:
                                target.ids[w * capacity + kept] = uid - removed
                                kept += 1
                            else:
                                good = False
                target.group_starts[output + group + 1] = kept
        if good and kept == nb - nm + nc:
            target.colors[w] = ncolors
        else:
            work.eligible[w] = 0
            target.nc[w] = 0


@wp.kernel
def _condense(source: _Inputs, target: _Inputs, work: _Workspace):
    w, lane = wp.tid()
    if work.eligible[w] == 0:
        return
    nm = work.effort_count[w]
    nu = source.nbc[w] + 3 * source.nc[w]
    nr = nu - nm
    origin = source.vio[w] + source.njc[w]
    offset = source.mio[w]
    for entry in range(lane, 256, 32):
        row, col = entry // 16, entry % 16
        value = wp.float32(0.0)
        if row < nm and col < nm:
            i = wp.int32(work.efforts[w * 16 + row])
            j = wp.int32(work.efforts[w * 16 + col])
            value = -source.schur[offset + j * nu + i]
        elif row == col:
            value = wp.float32(1.0)
        work.factor[w * 256 + entry] = value
    for entry in range(lane, 512, 32):
        row, col = entry // 32, entry % 32
        value = wp.float32(0.0)
        if row < nm and col < nr:
            i = wp.int32(work.efforts[w * 16 + row])
            j = wp.int32(work.remaining[w * 32 + col])
            value = -source.schur[offset + j * nu + i]
        work.response[w * 512 + entry] = value
    if lane < 16:
        value = wp.float32(0.0)
        if lane < nm:
            value = source.q[origin + work.efforts[w * 16 + lane]]
        work.gradient[w * 16 + lane] = value
    _sync_threads()
    matrix = wp.array(ptr=get_float32_array_offset_ptr(work.factor, w * 256), shape=(16, 16), dtype=wp.float32)
    response = wp.array(ptr=get_float32_array_offset_ptr(work.response, w * 512), shape=(16, 32), dtype=wp.float32)
    gradient = wp.array(ptr=get_float32_array_offset_ptr(work.gradient, w * 16), shape=(16, 1), dtype=wp.float32)
    factor = wp.tile_load(matrix, shape=(16, 16), storage="shared")
    wp.tile_cholesky_inplace(factor, fill_mode="upper")
    lower = wp.tile_transpose(factor)
    columns = wp.tile_load(response, shape=(16, 32), storage="shared")
    rhs = wp.tile_load(gradient, shape=(16, 1), storage="shared")
    wp.tile_lower_solve_inplace(lower, columns)
    wp.tile_lower_solve_inplace(lower, rhs)
    wp.tile_store(matrix, lower)
    wp.tile_store(response, columns)
    wp.tile_store(gradient, rhs)
    gram = wp.tile_zeros(shape=(32, 32), dtype=wp.float32, storage="shared")
    correction = wp.tile_zeros(shape=(32, 1), dtype=wp.float32, storage="shared")
    wp.tile_matmul(wp.tile_transpose(columns), columns, gram)
    wp.tile_matmul(wp.tile_transpose(columns), rhs, correction)
    _sync_threads()
    if lane < 16:
        pivot = work.factor[w * 256 + lane * 17]
        if not wp.isfinite(pivot) or pivot <= 0.0:
            wp.atomic_or(work.failure, w, 1)
    for entry in range(lane, nr * nr, 32):
        row, col = entry % nr, entry // nr
        i = wp.int32(work.remaining[w * 32 + row])
        j = wp.int32(work.remaining[w * 32 + col])
        value = source.schur[offset + j * nu + i] + gram[row, col]
        target.schur[w * 1024 + entry] = value
        if not wp.isfinite(value):
            wp.atomic_or(work.failure, w, 2)
    if lane < nr:
        row = work.remaining[w * 32 + lane]
        q = source.q[origin + row] - correction[lane, 0]
        diagonal = source.diag[origin + row] - gram[lane, lane]
        target.q[w * 64 + 32 + lane] = q
        target.diag[w * 64 + 32 + lane] = diagonal
        target.impulses[w * 64 + 32 + lane] = source.impulses[origin + row]
        if not wp.isfinite(q) or not wp.isfinite(diagonal):
            wp.atomic_or(work.failure, w, 2)
    _sync_threads()
    if lane == 0 and work.failure[w] != 0:
        target.nc[w] = 0


@wp.kernel
def _recover_and_certify(source: _Inputs, target: _Inputs, work: _Workspace):
    w, lane = wp.tid()
    if lane == 0:
        work.fallback_nc[w] = source.nc[w]
    if work.eligible[w] == 0 or work.failure[w] != 0:
        return
    nm = work.effort_count[w]
    nu = source.nbc[w] + 3 * source.nc[w]
    nr = nu - nm
    origin = source.vio[w] + source.njc[w]
    cfg = source.config[w]
    value = wp.float32(0.0)
    if lane < nr:
        row = work.remaining[w * 32 + lane]
        impulse = target.impulses[w * 64 + 32 + lane]
        work.candidate[w * 48 + row] = impulse
        value = impulse - source.impulses[origin + row]
    work.delta[w * 32 + lane] = value
    _sync_threads()
    matrix = wp.array(ptr=get_float32_array_offset_ptr(work.factor, w * 256), shape=(16, 16), dtype=wp.float32)
    response = wp.array(ptr=get_float32_array_offset_ptr(work.response, w * 512), shape=(16, 32), dtype=wp.float32)
    gradient = wp.array(ptr=get_float32_array_offset_ptr(work.gradient, w * 16), shape=(16, 1), dtype=wp.float32)
    delta = wp.array(ptr=get_float32_array_offset_ptr(work.delta, w * 32), shape=(32, 1), dtype=wp.float32)
    lower = wp.tile_load(matrix, shape=(16, 16))
    columns = wp.tile_load(response, shape=(16, 32))
    rhs = wp.tile_load(gradient, shape=(16, 1), storage="shared")
    change = wp.tile_load(delta, shape=(32, 1))
    wp.tile_matmul(columns, change, rhs)
    wp.tile_upper_solve_inplace(wp.tile_transpose(lower), rhs)
    if lane < nm:
        row = work.efforts[w * 16 + lane]
        impulse = source.impulses[origin + row] - rhs[lane, 0]
        work.candidate[w * 48 + row] = impulse
        lo = wp.float32(source.lower[source.bcio[w] + row])
        hi = wp.float32(source.upper[source.bcio[w] + row])
        if not wp.isfinite(impulse) or wp.isnan(lo) or wp.isnan(hi) or impulse < lo or impulse > hi:
            wp.atomic_or(work.failure, w, 4)
    _sync_threads()
    for row in range(lane, nu, 32):
        q = wp.float32(source.q[origin + row])
        for col in range(nu):
            q -= source.schur[source.mio[w] + col * nu + row] * (
                work.candidate[w * 48 + col] - source.impulses[origin + col]
            )
        work.candidate_q[w * 48 + row] = q
        if not wp.isfinite(q) or not wp.isfinite(work.candidate[w * 48 + row]):
            wp.atomic_or(work.failure, w, 2)
    _sync_threads()
    for row in range(lane, source.nbc[w], 32):
        old = work.candidate[w * 48 + row]
        diagonal = source.diag[origin + row]
        lo = wp.float32(source.lower[source.bcio[w] + row])
        hi = wp.float32(source.upper[source.bcio[w] + row])
        new = project_box_update(old, work.candidate_q[w * 48 + row], diagonal, cfg.regularization, cfg.omega, lo, hi)
        if (
            not wp.isfinite(new)
            or not wp.isfinite(diagonal)
            or wp.isnan(lo)
            or wp.isnan(hi)
            or wp.abs(new - old) > cfg.tolerance
        ):
            wp.atomic_or(work.failure, w, 8)
    if lane < nm:
        row = work.efforts[w * 16 + lane]
        scale = source.P[origin + row]
        if not wp.isfinite(scale) or wp.abs(work.candidate_q[w * 48 + row]) > cfg.tolerance * wp.abs(scale):
            wp.atomic_or(work.failure, w, 8)
    if lane < source.nc[w]:
        row = source.nbc[w] + 3 * lane
        normal = row + 2
        old_n = work.candidate[w * 48 + normal]
        new_n = project_contact_normal_update(
            old_n, work.candidate_q[w * 48 + normal], source.diag[origin + normal], cfg.regularization, cfg.omega
        )
        scale = source.P[origin + normal]
        normal_diag = wp.abs(source.raw_diag[origin + normal]) * scale * scale
        load = source.mu[source.cio[w] + lane] * contact_friction_normal_load(
            old_n, source.vb[origin + normal], scale, normal_diag, cfg.regularization, cfg.omega
        )
        old_t = wp.vec2f(work.candidate[w * 48 + row], work.candidate[w * 48 + row + 1])
        new_t = project_contact_tangent_update(
            old_t,
            wp.vec2f(work.candidate_q[w * 48 + row], work.candidate_q[w * 48 + row + 1]),
            wp.vec2f(source.diag[origin + row], source.diag[origin + row + 1]),
            -source.schur[source.mio[w] + (row + 1) * nu + row],
            cfg.regularization,
            cfg.omega,
            load,
        )
        finite = wp.isfinite(new_n) and wp.isfinite(new_t.x) and wp.isfinite(new_t.y) and wp.isfinite(load)
        finite = finite and wp.isfinite(normal_diag) and wp.isfinite(source.vb[origin + normal])
        for component in range(3):
            finite = finite and wp.isfinite(source.diag[origin + row + component])
        if (
            not finite
            or wp.abs(new_n - old_n) > cfg.tolerance
            or wp.max(wp.abs(new_t.x - old_t.x), wp.abs(new_t.y - old_t.y)) > cfg.tolerance
        ):
            wp.atomic_or(work.failure, w, 8)
    _sync_threads()
    if lane == 0 and work.failure[w] == 0:
        work.fallback_nc[w] = 0


@wp.kernel
def _publish(source: _Inputs, staged_status: wp.array[DVIStatus], work: _Workspace):
    w, lane = wp.tid()
    if work.eligible[w] == 0 or work.failure[w] != 0:
        return
    nu = source.nbc[w] + 3 * source.nc[w]
    origin = source.vio[w] + source.njc[w]
    for row in range(lane, nu, 32):
        source.impulses[origin + row] = work.candidate[w * 48 + row]
        source.q[origin + row] = work.candidate_q[w * 48 + row]
    if lane == 0:
        status = source.status[w]
        status.iterations = staged_status[w].iterations
        source.status[w] = status


class EffortLimitCondensation:
    """Own mapped candidate solves and preserve rejected worlds for ordinary PGS."""

    @classmethod
    def create(cls, path):
        """Allocate only for saturated compact solves with enough projected sweeps."""
        if (
            not path.device.is_cuda
            or path.size.num_worlds < 2048
            or not path.use_schur_complement
            or path.data.bilateral_operator is None
            or path.max_alternating_iterations * path.max_inequality_sweeps_per_iteration < 32
            or path.size.sum_of_num_effort_joint_cts == 0
        ):
            return None
        joints, info = path.model.joints, path.model.info
        worlds, counts = joints.wid.numpy(), joints.num_effort_cts.numpy()
        offsets = joints.effort_cts_offset_total_cts.numpy()
        origins = info.total_cts_offset.numpy() + info.joint_bounded_cts_group_offset.numpy()
        efforts = [[] for _ in range(path.size.num_worlds)]
        for joint, world in enumerate(worlds):
            start = int(offsets[joint] - origins[world])
            efforts[world].extend(range(start, start + int(counts[joint])))
        efforts = [sorted(rows) if 0 < len(rows) <= 16 else [] for rows in efforts]
        if not any(efforts):
            return None
        return cls(efforts, device=path.device, schedule_capacity=path.size.max_of_max_inequalities)

    def __init__(self, effort_ids: list[list[int]], *, device, schedule_capacity: int):
        """Allocate fixed-capacity scratch for each world's original bounded IDs."""
        self._device = wp.get_device(device)
        self._worlds = len(effort_ids)
        if schedule_capacity < 1:
            raise ValueError("A positive inequality schedule capacity is required.")
        ids = np.full((self._worlds, 16), -1, dtype=np.int32)
        for world, rows in enumerate(effort_ids):
            if len(rows) > 16 or rows != sorted(set(rows)) or any(row < 0 for row in rows):
                raise ValueError("Effort limit IDs must be sorted, unique bounded IDs, with at most sixteen per world.")
            ids[world, : len(rows)] = rows

        def array(values):
            return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=self._device)

        def zeros(count, dtype=wp.float32):
            return wp.zeros(count, dtype=dtype, device=self._device)

        def constant(value):
            return array(np.full(self._worlds, value, dtype=np.int32))

        work = _Workspace()
        work.efforts = array(ids.ravel())
        work.effort_count = array([len(rows) for rows in effort_ids])
        work.remaining = zeros(self._worlds * 32, wp.int32)
        work.eligible = zeros(self._worlds, wp.int32)
        work.failure = zeros(self._worlds, wp.int32)
        work.fallback_nc = zeros(self._worlds, wp.int32)
        for name, count in (
            ("factor", 256),
            ("response", 512),
            ("gradient", 16),
            ("delta", 32),
            ("candidate", 48),
            ("candidate_q", 48),
        ):
            setattr(work, name, zeros(self._worlds * count))
        work.schedule_capacity = schedule_capacity
        self._work = work
        self.eligible, self.failure, self.fallback_nc = work.eligible, work.failure, work.fallback_nc
        # A private 32-row prefix lets the existing PGS indexing operate on
        # compact scratch without changing the original problem's row layout.
        self._overrides = {
            0: zeros(1, wp.int32),
            1: zeros(self._worlds * 32, wp.int32),
            2: constant(0),
            3: constant(0),
            4: constant(0),
            5: array(np.arange(self._worlds) * 32),
            6: constant(0),
            7: array(np.arange(self._worlds) * 32),
            8: array(np.arange(self._worlds) * schedule_capacity),
            9: constant(32),
            10: constant(32),
            11: constant(32),
            12: array(np.arange(self._worlds) * 64),
            13: zeros(self._worlds * 32),
            14: zeros(self._worlds * 32),
            15: zeros(self._worlds * 32),
            20: constant(32),
            21: array(np.arange(self._worlds) * 1024),
            22: constant(32),
            23: zeros(self._worlds * 1024),
            25: constant(0),
            26: zeros(self._worlds * schedule_capacity, wp.int32),
            27: zeros(self._worlds * (schedule_capacity + 1), wp.int32),
            28: zeros(self._worlds * (schedule_capacity + 1), wp.int32),
            30: zeros(self._worlds, DVIStatus),
        }
        for index in (16, 17, 18, 19, 24, 31):
            self._overrides[index] = zeros(self._worlds * 64)

    def prepare_candidates(self, compact_inputs: list):
        """Solve and certify candidates without changing original impulses or q."""
        stage = list(compact_inputs)
        for index, value in self._overrides.items():
            stage[index] = value
        source, target = _view(compact_inputs), _view(stage)
        for kernel in (_gather, _condense):
            wp.launch(
                kernel, dim=(self._worlds, 32), inputs=[source, target, self._work], block_dim=32, device=self._device
            )
        wp.launch(
            make_compact_schur_pgs_kernel(32), dim=self._worlds * 32, inputs=stage, block_dim=32, device=self._device
        )
        wp.launch(
            _recover_and_certify,
            dim=(self._worlds, 32),
            inputs=[source, target, self._work],
            block_dim=32,
            device=self._device,
        )
        return self.fallback_nc

    def publish(self, compact_inputs: list):
        """Commit certified candidates after the original masked fallback launches."""
        wp.launch(
            _publish,
            dim=(self._worlds, 32),
            inputs=[_view(compact_inputs), self._overrides[30], self._work],
            block_dim=32,
            device=self._device,
        )
