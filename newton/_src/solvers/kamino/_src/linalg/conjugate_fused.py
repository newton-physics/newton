# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""KAMINO: single-kernel sparse Conjugate Residual solver.

Fused CR for the matrix-free Delassus operator ``A = P J M^-1 J^T P + diag(eta)``
(``J``: constraint Jacobian in 1x6 ``vec6`` blocks, <= :data:`MAX_BLOCKS_PER_ROW`
per row; ``M^-1``: block-diagonal inverse mass; ``P``: diagonal preconditioner;
``eta``: diagonal regularization). One Warp tile-block of ``block_dim`` threads solves
one world; each thread owns ``ceil(max_rows / block_dim)`` constraint rows and keeps
their CR state (``x, r, p, Ap``) in registers.

Each ``A`` apply is ``t = J^T (P v)`` (transpose gather over columns via a per-body
cursor from a one-off segmented sort), ``M^-1 t`` per body, then ``P (J M^-1 t) + eta v``
(forward gather over rows). The raw ``nzb_values`` are read directly -- no ``P J M^-1``
or ``(P J)^T`` value copy is materialized; the only auxiliary data are ``wp.int32`` index
arrays. Per-phase intermediates live in shared memory via ``wp.tile``; CR-scalar
reductions use ``tile_sum``.
"""

from __future__ import annotations

import functools

import warp as wp

from ..core.types import vec6f

wp.set_module_options({"enable_backward": False})

__all__ = [
    "MAX_BLOCKS_PER_ROW",
    "build_row_index",
    "build_transpose_index",
    "make_cursor_kernel",
    "make_fill_sort_kernel",
    "make_fused_cr_kernel",
    "make_row_index_kernel",
]


# Sum-reduce within a contiguous ``width``-lane sub-group (width must divide 32 and the
# group must be lane-aligned). Used by the cooperative per-body transpose, where ``T``
# threads share one body's block walk and combine their partials. width==1 is a no-op.
@wp.func_native(
    """
    float r = v;
    for (int o = width >> 1; o > 0; o >>= 1)
        r += __shfl_xor_sync(0xffffffffu, r, o, width);
    return r;
    """
)
def warp_subreduce_sum(v: wp.float32, width: wp.int32) -> wp.float32: ...


# Butterfly all-reduce over a full 32-lane warp (no barrier): the two-level CR reduction's intra-warp step.
@wp.func_native(
    """
    float r = v;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        r += __shfl_xor_sync(0xffffffffu, r, o, 32);
    return r;
    """
)
def warp_allreduce_sum(v: wp.float32) -> wp.float32: ...


# Maximum number of non-zero vec6 blocks per constraint row. Contacts and
# bilateral constraints touch two bodies (two blocks); limits touch one.
MAX_BLOCKS_PER_ROW: int = 2


# ---------------------------------------------------------------------------
# Matrix-preparation kernels (run once per sim step, outside the solve loop)
# ---------------------------------------------------------------------------


@functools.cache
def make_row_index_kernel(max_blocks: int):
    """Build a per-row index of Jacobian block ids (for the forward gather).

    ``row_blk[g, k]`` holds the global block id of the ``k``-th block of
    constraint row ``row`` (``g = row_offset[w] + row``), or ``-1`` for empty
    slots. This is an ``wp.int32`` index into the Jacobian's ``nzb_values`` -- no
    block values are copied.
    """

    @wp.kernel
    def row_index_kernel(
        num_nzb: wp.array[wp.int32],
        nzb_start: wp.array[wp.int32],
        nzb_coords: wp.array2d[wp.int32],
        row_offset: wp.array[wp.int32],
        # Outputs:
        row_blk: wp.array2d[wp.int32],
        slot_count: wp.array[wp.int32],
    ):
        wid, bi = wp.tid()
        if bi >= num_nzb[wid]:
            return
        gb = nzb_start[wid] + bi
        row = nzb_coords[gb, 0]
        g = row_offset[wid] + row
        slot = wp.atomic_add(slot_count, g, 1)
        # A row with more blocks than the cap would have its extras dropped here, silently
        # corrupting the solve. Hard-error in debug builds; warn once per offending row otherwise.
        assert slot < wp.static(max_blocks)
        if slot < wp.static(max_blocks):
            # Pack the body index (col//6) into the high bits so the forward reads it free from row_blk.
            bl = nzb_coords[gb, 1] // 6
            row_blk[g, slot] = gb | (bl << 24)
        elif slot == wp.static(max_blocks):
            wp.printf(
                "[kamino fused CR] WARNING: constraint row %d has more than %d Jacobian blocks; "
                "the forward gather drops the extra block(s) and the solve will be incorrect.\n",
                g,
                wp.static(max_blocks),
            )

    return row_index_kernel


@functools.cache
def make_fill_sort_kernel():
    """Fill the segmented-sort input arrays and per-world segment ends.

    For each active block, ``sort_key`` gets the block's column (body DoF start,
    a multiple of 6) and ``sort_val`` gets the global block id. ``seg_end[w]``
    is the exclusive end of world ``w``'s block segment.
    """

    @wp.kernel
    def fill_sort_kernel(
        num_nzb: wp.array[wp.int32],
        nzb_start: wp.array[wp.int32],
        nzb_coords: wp.array2d[wp.int32],
        # Outputs:
        sort_key: wp.array[wp.int32],
        sort_val: wp.array[wp.int32],
        seg_end: wp.array[wp.int32],
    ):
        wid, bi = wp.tid()
        if bi == 0:
            seg_end[wid] = nzb_start[wid] + num_nzb[wid]
        if bi >= num_nzb[wid]:
            return
        gb = nzb_start[wid] + bi
        sort_key[gb] = nzb_coords[gb, 1]
        sort_val[gb] = gb

    return fill_sort_kernel


@functools.cache
def make_cursor_kernel():
    """Build the per-body cursor into the column-sorted block list.

    After the segmented sort, blocks within a world are ordered by column (=
    ``6 * body``). ``cursor[w, body]`` holds the global index of the first sorted
    block for that body, or ``-1`` if the body has no blocks.
    """

    @wp.kernel
    def cursor_kernel(
        num_nzb: wp.array[wp.int32],
        nzb_start: wp.array[wp.int32],
        sort_key: wp.array[wp.int32],
        # Outputs:
        cursor: wp.array2d[wp.int32],
    ):
        wid, pos = wp.tid()
        if pos >= num_nzb[wid]:
            return
        idx = nzb_start[wid] + pos
        maj = sort_key[idx] // 6
        if pos == 0:
            cursor[wid, maj] = idx
        else:
            if maj != sort_key[idx - 1] // 6:
                cursor[wid, maj] = idx

    return cursor_kernel


@functools.cache
def make_fill_row_sorted_kernel():
    """Materialize the column-sorted block rows for the transpose (``row_sorted``).

    For each sorted block position ``idx``, gather its row (``nzb_coords[gb,0]``) from the
    scattered global block ``gb = sort_val[idx]`` into a sequential array, once per step, so
    the transpose hot loop reads ``row_idx_sorted[idx]`` sequentially instead of chasing ``gb``.
    """

    @wp.kernel
    def fill_row_sorted_kernel(
        num_nzb: wp.array[wp.int32],
        nzb_start: wp.array[wp.int32],
        sort_val: wp.array[wp.int32],
        nzb_coords: wp.array2d[wp.int32],
        # Outputs:
        row_idx_sorted: wp.array[wp.int32],
    ):
        wid, pos = wp.tid()
        if pos >= num_nzb[wid]:
            return
        idx = nzb_start[wid] + pos
        row_idx_sorted[idx] = nzb_coords[sort_val[idx], 0]

    return fill_row_sorted_kernel


def build_row_index(
    *,
    num_nzb: wp.array,
    nzb_start: wp.array,
    nzb_coords: wp.array,
    row_offset: wp.array,
    total_rows: int,
    max_of_num_nzb: int,
    max_blocks: int = MAX_BLOCKS_PER_ROW,
    out_row_blk: wp.array | None = None,
    out_slot_count: wp.array | None = None,
    device=None,
) -> wp.array:
    """Build (or refill) the per-row block-id index ``row_blk``."""
    n_worlds = num_nzb.shape[0]
    row_blk = out_row_blk if out_row_blk is not None else wp.empty((total_rows, max_blocks), dtype=wp.int32, device=device)
    slot_count = out_slot_count if out_slot_count is not None else wp.empty((total_rows,), dtype=wp.int32, device=device)
    row_blk.fill_(-1)
    slot_count.zero_()
    wp.launch(
        make_row_index_kernel(max_blocks),
        dim=(n_worlds, max_of_num_nzb),
        inputs=[num_nzb, nzb_start, nzb_coords, row_offset],
        outputs=[row_blk, slot_count],
        device=device,
    )
    return row_blk


def build_transpose_index(
    *,
    num_nzb: wp.array,
    nzb_start: wp.array,
    nzb_coords: wp.array,
    total_nnz: int,
    max_of_num_nzb: int,
    max_major_cols: int,
    out_sort_key: wp.array | None = None,
    out_sort_val: wp.array | None = None,
    out_seg_end: wp.array | None = None,
    out_cursor: wp.array | None = None,
    out_row_idx_sorted: wp.array | None = None,
    device=None,
) -> tuple[wp.array, wp.array, wp.array]:
    """Build (or refill) the column-sorted transpose index and per-body cursor.

    Returns ``(sort_key, sort_val, cursor)``. ``sort_key``/``sort_val`` are sized
    ``2 * total_nnz`` (the sort needs scratch); the first ``total_nnz`` entries
    hold the per-world block segments sorted by column. ``cursor`` has shape
    ``(n_worlds, max_major_cols)``.
    """
    n_worlds = num_nzb.shape[0]
    alloc = max(2 * total_nnz, 2)
    sort_key = out_sort_key if out_sort_key is not None else wp.empty((alloc,), dtype=wp.int32, device=device)
    sort_val = out_sort_val if out_sort_val is not None else wp.empty((alloc,), dtype=wp.int32, device=device)
    seg_end = out_seg_end if out_seg_end is not None else wp.empty((n_worlds,), dtype=wp.int32, device=device)
    cursor = out_cursor if out_cursor is not None else wp.empty((n_worlds, max_major_cols), dtype=wp.int32, device=device)
    cursor.fill_(-1)

    wp.launch(
        make_fill_sort_kernel(),
        dim=(n_worlds, max_of_num_nzb),
        inputs=[num_nzb, nzb_start, nzb_coords],
        outputs=[sort_key, sort_val, seg_end],
        device=device,
    )
    if total_nnz > 0:
        wp.utils.segmented_sort_pairs(
            sort_key,
            sort_val,
            total_nnz,
            segment_start_indices=nzb_start,
            segment_end_indices=seg_end,
        )
    wp.launch(
        make_cursor_kernel(),
        dim=(n_worlds, max_of_num_nzb),
        inputs=[num_nzb, nzb_start, sort_key],
        outputs=[cursor],
        device=device,
    )
    if out_row_idx_sorted is not None:
        wp.launch(
            make_fill_row_sorted_kernel(),
            dim=(n_worlds, max_of_num_nzb),
            inputs=[num_nzb, nzb_start, sort_val, nzb_coords],
            outputs=[out_row_idx_sorted],
            device=device,
        )
    return sort_key, sort_val, cursor


# ---------------------------------------------------------------------------
# Fused CR kernel
# ---------------------------------------------------------------------------


@functools.cache
def make_fused_cr_kernel(max_rows: int, max_cols: int, max_blocks: int, block_dim: int = 0):
    """Build the single-kernel sparse CR solver.

    Each block solves one world with ``block_dim`` threads; each thread owns
    ``NR = ceil(max_rows / block_dim)`` constraint rows in registers (one row per
    thread when ``block_dim == max_rows``).

    Args:
        max_rows: Static upper bound on constraint rows per world. Pad to a
            multiple of ``block_dim`` so the per-thread row count divides evenly.
        max_cols: Static upper bound on body DoFs per world.
        max_blocks: Static maximum number of blocks per constraint row.
        block_dim: Threads per block. Defaults to ``max_rows``.

    Launch with ``wp.launch_tiled(dim=n_worlds, block_dim=block_dim)``.
    """

    R = max_rows
    C = max_cols
    MB = max_blocks
    BD = block_dim if block_dim > 0 else max_rows
    NR = (R + BD - 1) // BD  # constraint rows owned per thread
    NB = C // 6  # max bodies per world (transpose output is one vec6 per body)
    # Cooperative per-body transpose: T threads share each body's block walk so every
    # block is read once (no 6x re-read across the body's DoF columns) and the walk is
    # load-balanced across the high-degree-body tail. T = largest power of two with
    # NB*T <= block_dim and T <= warp (so a body's lanes form one shuffle sub-group).
    if NB > BD:
        raise ValueError(f"fused CR requires block_dim >= max bodies ({BD} < {NB}).")
    T = 1
    while T * 2 <= 32 and NB * (T * 2) <= BD:
        T *= 2
    vecNR = wp.types.vector(length=NR, dtype=wp.float32)
    # CR reductions: two-level (warp __shfl + NW-element shared combine) when block_dim % 32 == 0, else tile_sum.
    NW = BD // 32  # warps per block
    warp_reduce = BD % 32 == 0 and NW >= 1

    @wp.func
    def _minv(im: wp.float32, iI: wp.mat33f, tv: vec6f) -> vec6f:
        # Block-diagonal inverse mass applied to one body's 6-vector.
        lin = im * wp.vec3f(tv[0], tv[1], tv[2])
        ang = iI @ wp.vec3f(tv[3], tv[4], tv[5])
        return vec6f(lin[0], lin[1], lin[2], ang[0], ang[1], ang[2])

    @wp.func
    def _rsum(x: wp.float32, t: wp.int32) -> wp.float32:
        # Block-reduce a per-thread CR scalar: two-level warp-shuffle + shared combine (full warps), else tile_sum.
        if wp.static(warp_reduce):
            wv = warp_allreduce_sum(x)
            sw = wp.tile_zeros(shape=wp.static(NW), dtype=wp.float32, storage="shared")
            wp.tile_scatter_add(sw, t // 32, wv, (t % 32) == 0, False)
            s = wp.float32(0.0)
            for i in range(wp.static(NW)):
                s += sw[i]
            return s
        else:
            return wp.tile_sum(wp.tile(x))[0]

    @wp.func
    def _rsum2(a: wp.float32, b: wp.float32, t: wp.int32) -> wp.vec2:
        # Reduce two co-located CR scalars in one pass; two-level for full warps, else one axis-1 tile_sum.
        if wp.static(warp_reduce):
            wa = warp_allreduce_sum(a)
            wb = warp_allreduce_sum(b)
            sw = wp.tile_zeros(shape=wp.static(NW), dtype=wp.vec2, storage="shared")
            wp.tile_scatter_add(sw, t // 32, wp.vec2(wa, wb), (t % 32) == 0, False)
            v = wp.vec2(0.0, 0.0)
            for i in range(wp.static(NW)):
                v += sw[i]
            return v
        else:
            s = wp.tile_sum(wp.tile(wp.vec2(a, b)), axis=1)
            return wp.vec2(s[0], s[1])

    @wp.func
    def _apply_A(
        vv: vecNR,
        p_rowv: vecNR,
        eta_rowv: vecNR,
        w: wp.int32,
        t: wp.int32,
        n: wp.int32,
        nb: wp.int32,
        nze: wp.int32,
        roff: wp.int32,
        nzb_values: wp.array[vec6f],
        nzb_coords: wp.array2d[wp.int32],
        row_blk: wp.array2d[wp.int32],
        sort_key: wp.array[wp.int32],
        sort_val: wp.array[wp.int32],
        row_idx_sorted: wp.array[wp.int32],
        cursor: wp.array2d[wp.int32],
        inv_m: wp.array[wp.float32],
        inv_I: wp.array[wp.mat33f],
        bodies_offset: wp.array[wp.int32],
    ) -> vecNR:
        # A @ vv = P (J M^-1 J^T (P vv)) + eta vv, kept in shared memory via wp.tile (logical index
        # e lives at tile element [e // BD, e % BD]; the tile carries Warp's barrier on cross-lane
        # gathers and overwrites, so no per-call zeroing).
        pv = vecNR()
        for i in range(wp.static(NR)):
            pv[i] = p_rowv[i] * vv[i]
        sv = wp.tile(pv)  # sv[row // BD, row % BD] = P*vv at constraint row
        # Transpose t = J^T (P vv), cooperative per body: T threads share a body's column-sorted
        # block run (strided by T -> load-balanced), each block's vec6 read once into all 6 DoFs,
        # then a width-T shuffle reduction; st[body] holds the body's 6 transpose outputs.
        st = wp.tile_zeros(shape=wp.static(NB), dtype=vec6f, storage="shared")
        body = t // T
        sub = t % T
        nbod = nb // 6
        q0 = wp.float32(0.0)
        q1 = wp.float32(0.0)
        q2 = wp.float32(0.0)
        q3 = wp.float32(0.0)
        q4 = wp.float32(0.0)
        q5 = wp.float32(0.0)
        body_col = body * 6
        if body < nbod:
            cur = cursor[w, body]
            if cur >= 0:
                idx = cur + sub
                while idx < nze and sort_key[idx] == body_col:
                    # Read the block's row from the pre-gathered column-sorted index (sequential),
                    # and its vec6 value from the original coordinate values via the sorted block id.
                    rr = row_idx_sorted[idx]
                    val = nzb_values[sort_val[idx]]
                    xv = sv[rr // BD, rr % BD]
                    q0 += val[0] * xv
                    q1 += val[1] * xv
                    q2 += val[2] * xv
                    q3 += val[3] * xv
                    q4 += val[4] * xv
                    q5 += val[5] * xv
                    idx += T
        q0 = warp_subreduce_sum(q0, T)
        q1 = warp_subreduce_sum(q1, T)
        q2 = warp_subreduce_sum(q2, T)
        q3 = warp_subreduce_sum(q3, T)
        q4 = warp_subreduce_sum(q4, T)
        q5 = warp_subreduce_sum(q5, T)
        bo = bodies_offset[w]  # hoist the per-world body offset out of the row/block loops
        bidx = wp.min(body, NB - 1)  # keep the tile index in-bounds for masked-out lanes
        # Apply block-diagonal M^-1 per body here (writing lane) so the forward reads M^-1(J^T P v) directly.
        qv = vec6f(q0, q1, q2, q3, q4, q5)
        active = body < nbod and sub == 0
        if active:
            qv = _minv(inv_m[bo + body], inv_I[bo + body], qv)
        wp.tile_scatter_add(st, bidx, qv, active, False)
        out = vecNR()
        for i in range(wp.static(NR)):
            row = t + i * BD
            av = wp.float32(0.0)
            if row < n:
                acc = wp.float32(0.0)
                for k in range(wp.static(MB)):
                    packed = row_blk[roff + row, k]
                    if packed >= 0:
                        gb = packed & 0x00FFFFFF  # low 24 bits: global block id
                        bl = packed >> 24  # high bits: within-world body index
                        acc += wp.dot(nzb_values[gb], st[bl])  # st[bl] = M^-1 (J^T P v) for the body
                av = p_rowv[i] * acc + eta_rowv[i] * vv[i]
            out[i] = av
        return out

    @wp.kernel
    def fused_cr_kernel(
        # Problem dims / offsets:
        ncts: wp.array[wp.int32],
        nbd: wp.array[wp.int32],
        row_offset: wp.array[wp.int32],
        vec_off: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        # Raw constraint Jacobian J (coordinate block-sparse), read directly:
        num_nzb: wp.array[wp.int32],
        nzb_start: wp.array[wp.int32],
        nzb_coords: wp.array2d[wp.int32],
        nzb_values: wp.array[vec6f],
        # Forward (per-row) block index and transpose (column-sorted) index:
        row_blk: wp.array2d[wp.int32],
        sort_key: wp.array[wp.int32],
        sort_val: wp.array[wp.int32],
        row_idx_sorted: wp.array[wp.int32],
        cursor: wp.array2d[wp.int32],
        # Inverse mass:
        inv_m: wp.array[wp.float32],
        inv_I: wp.array[wp.mat33f],
        bodies_offset: wp.array[wp.int32],
        # Diagonal preconditioner P and regularization eta:
        precond: wp.array[wp.float32],
        use_precond: wp.int32,
        eta: wp.array[wp.float32],
        # RHS / solution:
        b: wp.array[wp.float32],
        x: wp.array[wp.float32],
        # Params (per-world device arrays; PADMM mutates atol in place for its adaptive path):
        maxiter: wp.array[wp.int32],
        atol: wp.array[wp.float32],
        rtol: wp.array[wp.float32],
        # Outputs:
        out_iters: wp.array[wp.int32],
        out_resid: wp.array[wp.float32],
    ):
        w, t = wp.tid()
        if not world_active[w]:
            if t == 0:
                out_iters[w] = 0
                out_resid[w] = wp.float32(0.0)
            return

        n = ncts[w]
        nb = nbd[w]
        voff = vec_off[w]
        roff = row_offset[w]
        nzs = nzb_start[w]
        nze = nzs + num_nzb[w]

        # Per-thread row state: thread t owns rows {t, t+BD, ...}. Each row loop breaks once
        # row >= n (rows grow with i), so only active rows are touched; the compile-time loop
        # index keeps the vecNR state register-resident.
        p_rowv = vecNR()
        eta_rowv = vecNR()
        b_rowv = vecNR()
        x_rowv = vecNR()
        for i in range(wp.static(NR)):
            row = t + i * BD
            if row >= n:
                break
            pr = wp.float32(1.0)
            if use_precond != 0:
                pr = precond[voff + row]
            p_rowv[i] = pr
            eta_rowv[i] = eta[voff + row]
            b_rowv[i] = b[voff + row]
            x_rowv[i] = x[voff + row]

        bb = wp.float32(0.0)
        for i in range(wp.static(NR)):
            if t + i * BD >= n:
                break
            bb += b_rowv[i] * b_rowv[i]
        at = atol[w]
        rt = rtol[w]
        atol_sq = wp.max(rt * rt * _rsum(bb, t), at * at)

        # r0 = b - A x0
        ax_v = _apply_A(
            x_rowv,
            p_rowv,
            eta_rowv,
            w,
            t,
            n,
            nb,
            nze,
            roff,
            nzb_values,
            nzb_coords,
            row_blk,
            sort_key,
            sort_val,
            row_idx_sorted,
            cursor,
            inv_m,
            inv_I,
            bodies_offset,
        )
        r_rowv = vecNR()
        for i in range(wp.static(NR)):
            r_rowv[i] = b_rowv[i] - ax_v[i]
        p_dirv = r_rowv

        # Ar = A @ r ; Ap = Ar
        ar_rowv = _apply_A(
            r_rowv,
            p_rowv,
            eta_rowv,
            w,
            t,
            n,
            nb,
            nze,
            roff,
            nzb_values,
            nzb_coords,
            row_blk,
            sort_key,
            sort_val,
            row_idx_sorted,
            cursor,
            inv_m,
            inv_I,
            bodies_offset,
        )
        ap_rowv = ar_rowv

        rAr_local = wp.float32(0.0)
        rnorm_local = wp.float32(0.0)
        for i in range(wp.static(NR)):
            if t + i * BD >= n:
                break
            rAr_local += r_rowv[i] * ar_rowv[i]
            rnorm_local += r_rowv[i] * r_rowv[i]
        rr0 = _rsum2(rAr_local, rnorm_local, t)
        rAr = rr0[0]
        r_norm_sq = rr0[1]

        it = wp.int32(0)
        mi = maxiter[w]  # hoist the per-world iteration cap into a register (read once, not per-iter)
        for _it in range(mi):
            if r_norm_sq <= atol_sq:
                break
            it += 1

            denom_local = wp.float32(0.0)
            for i in range(wp.static(NR)):
                if t + i * BD >= n:
                    break
                denom_local += ap_rowv[i] * ap_rowv[i]
            denom = _rsum(denom_local, t)
            alpha = wp.float32(0.0)
            if denom > wp.float32(0.0):
                alpha = rAr / denom

            for i in range(wp.static(NR)):
                if t + i * BD >= n:
                    break
                x_rowv[i] += alpha * p_dirv[i]
                r_rowv[i] -= alpha * ap_rowv[i]

            # Ar = A @ r
            ar_rowv = _apply_A(
                r_rowv,
                p_rowv,
                eta_rowv,
                w,
                t,
                n,
                nb,
                nze,
                roff,
                nzb_values,
                nzb_coords,
                row_blk,
                sort_key,
                sort_val,
                row_idx_sorted,
                cursor,
                inv_m,
                inv_I,
                bodies_offset,
            )

            rAr_new_local = wp.float32(0.0)
            rnorm_local = wp.float32(0.0)
            for i in range(wp.static(NR)):
                if t + i * BD >= n:
                    break
                rAr_new_local += r_rowv[i] * ar_rowv[i]
                rnorm_local += r_rowv[i] * r_rowv[i]
            rr = _rsum2(rAr_new_local, rnorm_local, t)
            rAr_new = rr[0]
            r_norm_sq = rr[1]
            beta = wp.float32(0.0)
            if rAr > wp.float32(0.0):
                beta = rAr_new / rAr
            for i in range(wp.static(NR)):
                if t + i * BD >= n:
                    break
                p_dirv[i] = r_rowv[i] + beta * p_dirv[i]
                ap_rowv[i] = ar_rowv[i] + beta * ap_rowv[i]
            rAr = rAr_new

        for i in range(wp.static(NR)):
            row = t + i * BD
            if row >= n:
                break
            x[voff + row] = x_rowv[i]
        if t == 0:
            out_iters[w] = it
            out_resid[w] = r_norm_sq  # squared residual norm, matching ConjugateResidualSolver

    return fused_cr_kernel
