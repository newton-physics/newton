# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched GPU Reverse Cuthill-McKee reordering across a list of SPD blocks.

Companion to :mod:`rcm` that does the same work but batches all blocks into
a small, fixed number of launches (one launch per RCM stage, not per block).

Motivation
----------

The per-block :func:`rcm.create_rcm_launch` launches roughly ``2 * max_bfs_iters + 5``
kernels **per block**. For a workload with ``B`` blocks this scales linearly
in ``B * max_bfs_iters``. At small problem sizes (e.g. ``n = 256``, ``B = 8``)
the resulting hundreds of launches dominate wall time over the actual compute.

The CUDA complete-traversal path keeps the full BFS inside one persistent
kernel per matrix block. Other devices use a staged fallback with a fixed
``max_dim`` upper bound whose completed iterations are no-ops.

Layout assumptions
------------------

- ``A_flat``: a flat ``wp.array`` containing all block matrices concatenated
  in row-major order. Block ``b``'s matrix starts at offset ``mio[b]`` with
  size ``dims[b] * dims[b]``.
- ``perm_flat``: flat wp.int32 permutation output. Block ``b``'s output starts
  at offset ``vio[b]`` with size ``dims[b]``.
- ``dims``, ``mio``, ``vio``: ``wp.array[wp.int32]`` of length
  ``num_blocks``, precomputed on the device.

API
---

.. code-block:: python

    from newton._src.solvers.kamino._src.linalg.factorize.rcm_batch import (
        create_rcm_batch_launch,
    )

    launch = create_rcm_batch_launch(
        A_flat=A,
        perm_flat=P,
        dims=dims,
        mio=mio,
        vio=vio,
        num_blocks=B,
        max_dim=max(dims_host),
        tol=0.0,
        use_cuda_graph=True,
    )
    launch()  # one zero-arg callable; CUDA-graph capturable
"""

from collections.abc import Callable
from functools import cache

import warp as wp


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


def create_cuda_graph_callback(callback: Callable[[], None], device=None, stream=None) -> Callable[[], None]:
    """Capture ``callback`` into a CUDA graph and return a zero-arg replay fn."""
    with wp.ScopedCapture(device=device, stream=stream) as capture:
        callback()
    graph = capture.graph
    if stream is not None and stream.device != graph.device:
        raise RuntimeError(f"Cannot launch graph from device {graph.device} on stream from device {stream.device}")

    def graph_callback():
        wp.capture_launch(graph)

    return graph_callback


# ---------------------------------------------------------------------------
# Scratch allocation
# ---------------------------------------------------------------------------


def allocate_rcm_batch_scratch(total_vec: int, num_blocks: int, device) -> dict:
    """Preallocate device-side scratch used by the batched RCM launch.

    Sizing:
    - Per-vertex arrays (``degree``, ``level``, ``order_buf``) are sized by
      the union of all block vector offsets (``total_vec = sum(dims)``).
      Each block's slice is ``[vio[b] : vio[b]+dims[b])``.
    - Per-block arrays are sized ``(num_blocks,)`` and indexed by block.
    """
    return {
        "degree": wp.empty(total_vec, dtype=wp.int32, device=device),
        "level": wp.empty(total_vec, dtype=wp.int32, device=device),
        "order_buf": wp.empty(total_vec, dtype=wp.int32, device=device),
        "head": wp.empty(num_blocks, dtype=wp.int32, device=device),
        "current_level": wp.empty(num_blocks, dtype=wp.int32, device=device),
        "discovered": wp.empty(num_blocks, dtype=wp.int32, device=device),
        "reorder_active": wp.empty(num_blocks, dtype=wp.int32, device=device),
        "permutation_valid": wp.zeros(num_blocks, dtype=wp.int32, device=device),
        "permutation_dim": wp.zeros(num_blocks, dtype=wp.int32, device=device),
    }


# ---------------------------------------------------------------------------
# Kernels (one module per dtype)
# ---------------------------------------------------------------------------


@cache
def _make_rcm_batch_kernels(dtype):
    """Kernels are parameterized by `dtype` only. `max_dim` / `num_blocks`
    are passed as runtime ints so the same module can serve any shape.
    """
    module_name = f"rcm_batch_kernels_{getattr(dtype, '__name__', str(dtype))}"
    module = wp.get_module(module_name)
    module.options.update({"enable_backward": False, "default_grid_stride": False})

    @wp.kernel(module=module)
    def prepare_reorder_kernel(
        reuse_permutation: bool,
        dims: wp.array[wp.int32],
        permutation_valid: wp.array[wp.int32],
        permutation_dim: wp.array[wp.int32],
        reorder_active: wp.array[wp.int32],
    ):
        """Mark blocks whose permutation must be computed."""
        b = wp.tid()
        cached = permutation_valid[b] != int(0) and permutation_dim[b] == dims[b]
        reorder_active[b] = wp.where(reuse_permutation and cached, int(0), int(1))

    @wp.kernel(module=module)
    def init_and_degree_kernel(
        num_blocks: int,
        tol: dtype,  # type: ignore[valid-type]
        A: wp.array[dtype],  # type: ignore[valid-type]
        dims: wp.array[wp.int32],  # type: ignore[valid-type]
        mio: wp.array[wp.int32],  # type: ignore[valid-type]
        vio: wp.array[wp.int32],  # type: ignore[valid-type]
        degree: wp.array[wp.int32],  # type: ignore[valid-type]
        level: wp.array[wp.int32],  # type: ignore[valid-type]
        head: wp.array[wp.int32],  # type: ignore[valid-type]
        current_level: wp.array[wp.int32],
        discovered: wp.array[wp.int32],
        reorder_active: wp.array[wp.int32],
    ):
        """Launch dims: ``(num_blocks, max_dim)``.

        Thread ``(b, i)`` computes ``degree[vio[b] + i]`` for vertex ``i`` in
        block ``b`` (if ``i < dims[b]``). Thread ``(b, 0)`` also initializes
        the per-block scalars.
        """
        b, i = wp.tid()
        if b >= num_blocks:
            return
        if reorder_active[b] == int(0):
            return
        n_b = dims[b]
        if i >= n_b:
            return

        vb = vio[b]
        mb = mio[b]

        # Per-vertex init.
        level[vb + i] = int(-1)

        # Degree row scan.
        d = int(0)
        base = mb + i * n_b
        for j in range(n_b):
            if j == i:
                continue
            av = wp.abs(A[base + j])
            if av > tol:
                d += int(1)
        degree[vb + i] = d

        # Per-block scalars: one thread per block sets them.
        if i == 0:
            head[b] = int(0)
            current_level[b] = int(0)
            discovered[b] = int(0)

    @wp.kernel(module=module)
    def select_and_seed_kernel(
        num_blocks: int,
        dims: wp.array[wp.int32],  # type: ignore[valid-type]
        vio: wp.array[wp.int32],  # type: ignore[valid-type]
        degree: wp.array[wp.int32],  # type: ignore[valid-type]
        level: wp.array[wp.int32],  # type: ignore[valid-type]
        order_buf: wp.array[wp.int32],  # type: ignore[valid-type]
        head: wp.array[wp.int32],  # type: ignore[valid-type]
        reorder_active: wp.array[wp.int32],
    ):
        """Select a minimum-degree root and seed each active block."""
        b = wp.tid()
        if b >= num_blocks:
            return
        if reorder_active[b] == int(0):
            return
        n_b = dims[b]
        vb = vio[b]
        best_deg = int(2147483647)
        best_idx = int(0)
        for i in range(n_b):
            d = degree[vb + i]
            if d < best_deg:
                best_deg = d
                best_idx = i
        level[vb + best_idx] = int(0)
        # Atomically claim the first slot; at kernel entry head[b] is 0 and
        # only this thread touches it for block ``b``, so the atomic is
        # effectively a plain write to slot 0.
        slot = wp.atomic_add(head, b, int(1))
        order_buf[vb + slot] = best_idx

    @wp.kernel(module=module)
    def bfs_step_kernel(
        num_blocks: int,
        tol: dtype,  # type: ignore[valid-type]
        A: wp.array[dtype],  # type: ignore[valid-type]
        dims: wp.array[wp.int32],  # type: ignore[valid-type]
        mio: wp.array[wp.int32],  # type: ignore[valid-type]
        vio: wp.array[wp.int32],  # type: ignore[valid-type]
        level: wp.array[wp.int32],  # type: ignore[valid-type]
        order_buf: wp.array[wp.int32],  # type: ignore[valid-type]
        head: wp.array[wp.int32],  # type: ignore[valid-type]
        current_level: wp.array[wp.int32],
        discovered: wp.array[wp.int32],
        reorder_active: wp.array[wp.int32],
    ):
        """Launch dims: ``(num_blocks, max_dim)``. One BFS expansion step.

        ``current_level`` is device-side so the same launch can be replayed
        until every connected component has been traversed.
        """
        b, i = wp.tid()
        if b >= num_blocks:
            return
        if reorder_active[b] == int(0):
            return
        n_b = dims[b]
        if i >= n_b:
            return

        vb = vio[b]
        mb = mio[b]
        cur = current_level[b]

        if level[vb + i] != cur:
            return

        base = mb + i * n_b
        next_lvl = cur + int(1)
        for j in range(n_b):
            if j == i:
                continue
            av = wp.abs(A[base + j])
            if av > tol:
                if level[vb + j] == int(-1):
                    old = wp.atomic_cas(level, vb + j, int(-1), next_lvl)
                    if old == int(-1):
                        wp.atomic_max(discovered, b, int(1))
                        slot = wp.atomic_add(head, b, int(1))
                        order_buf[vb + slot] = j

    @wp.kernel(module=module)
    def advance_or_seed_kernel(
        num_blocks: int,
        dims: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        degree: wp.array[wp.int32],
        level: wp.array[wp.int32],
        order_buf: wp.array[wp.int32],
        head: wp.array[wp.int32],
        current_level: wp.array[wp.int32],
        discovered: wp.array[wp.int32],
        reorder_active: wp.array[wp.int32],
    ):
        """Advance a BFS level or seed the next disconnected component."""
        b = wp.tid()
        if b >= num_blocks or reorder_active[b] == int(0):
            return
        n_b = dims[b]
        if head[b] >= n_b:
            return

        next_level = current_level[b] + int(1)
        current_level[b] = next_level
        if discovered[b] != int(0):
            discovered[b] = int(0)
            return

        vb = vio[b]
        best_deg = int(2147483647)
        best_idx = int(-1)
        for i in range(n_b):
            if level[vb + i] == int(-1):
                d = degree[vb + i]
                if d < best_deg:
                    best_deg = d
                    best_idx = i

        if best_idx >= int(0):
            level[vb + best_idx] = next_level
            slot = head[b]
            head[b] = slot + int(1)
            order_buf[vb + slot] = best_idx

    @wp.kernel(module=module)
    def complete_cuda_kernel(
        num_blocks: int,
        tol: dtype,  # type: ignore[valid-type]
        A: wp.array[dtype],  # type: ignore[valid-type]
        dims: wp.array[wp.int32],
        mio: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        degree: wp.array[wp.int32],
        level: wp.array[wp.int32],
        order_buf: wp.array[wp.int32],
        head: wp.array[wp.int32],
        reorder_active: wp.array[wp.int32],
    ):
        """Traverse each block completely inside one persistent CUDA block."""
        tid = wp.tid()
        block_dim = wp.block_dim()
        lane = tid % block_dim
        b = tid / block_dim
        if b >= num_blocks or reorder_active[b] == int(0):
            return

        n_b = dims[b]
        vb = vio[b]
        mb = mio[b]

        if lane == int(0):
            best_deg = int(2147483647)
            best_idx = int(0)
            for i in range(n_b):
                d = degree[vb + i]
                if d < best_deg:
                    best_deg = d
                    best_idx = i
            level[vb + best_idx] = int(0)
            order_buf[vb] = best_idx
            head[b] = int(1)
        _sync_threads()

        frontier_begin = int(0)
        current_level = int(0)
        while frontier_begin < n_b:
            frontier_end = head[b]
            next_level = current_level + int(1)
            pos = frontier_begin
            while pos < frontier_end:
                source = order_buf[vb + pos]
                base = mb + source * n_b
                j = lane
                while j < n_b:
                    if j != source and wp.abs(A[base + j]) > tol:
                        wp.atomic_cas(level, vb + j, int(-1), next_level)
                    j += block_dim
                pos += int(1)
            _sync_threads()

            chunk_start = int(0)
            while chunk_start < n_b:
                j = chunk_start + lane
                is_new = int(0)
                if j < n_b and level[vb + j] == next_level:
                    is_new = int(1)
                prefix = wp.tile_scan_inclusive(wp.tile(is_new))
                write_base = int(0)
                if lane == block_dim - int(1):
                    write_base = wp.atomic_add(head, b, prefix[block_dim - int(1)])
                write_base_tile = wp.tile(write_base)
                write_base = write_base_tile[block_dim - int(1)]
                if is_new != int(0):
                    order_buf[vb + write_base + prefix[lane] - int(1)] = j
                chunk_start += block_dim
            _sync_threads()

            if lane == int(0) and head[b] == frontier_end and frontier_end < n_b:
                best_deg = int(2147483647)
                best_idx = int(-1)
                for i in range(n_b):
                    if level[vb + i] == int(-1):
                        d = degree[vb + i]
                        if d < best_deg:
                            best_deg = d
                            best_idx = i
                if best_idx >= int(0):
                    level[vb + best_idx] = next_level
                    order_buf[vb + frontier_end] = best_idx
                    head[b] = frontier_end + int(1)
            _sync_threads()
            frontier_begin = frontier_end
            current_level = next_level

    @wp.kernel(module=module)
    def append_unreached_kernel(
        num_blocks: int,
        dims: wp.array[wp.int32],  # type: ignore[valid-type]
        vio: wp.array[wp.int32],  # type: ignore[valid-type]
        level: wp.array[wp.int32],  # type: ignore[valid-type]
        order_buf: wp.array[wp.int32],  # type: ignore[valid-type]
        head: wp.array[wp.int32],  # type: ignore[valid-type]
        reorder_active: wp.array[wp.int32],
    ):
        """Append vertices left by an explicitly truncated traversal."""
        b = wp.tid()
        if b >= num_blocks:
            return
        if reorder_active[b] == int(0):
            return
        n_b = dims[b]
        vb = vio[b]
        pos = head[b]
        for i in range(n_b):
            if level[vb + i] == int(-1):
                order_buf[vb + pos] = i
                pos += int(1)
        head[b] = pos

    @wp.kernel(module=module)
    def reverse_into_perm_kernel(
        num_blocks: int,
        dims: wp.array[wp.int32],  # type: ignore[valid-type]
        vio: wp.array[wp.int32],  # type: ignore[valid-type]
        order_buf: wp.array[wp.int32],  # type: ignore[valid-type]
        perm: wp.array[wp.int32],  # type: ignore[valid-type]
        reorder_active: wp.array[wp.int32],
        permutation_valid: wp.array[wp.int32],
        permutation_dim: wp.array[wp.int32],
    ):
        """Launch dims: ``(num_blocks, max_dim)``. ``perm[i] = order_buf[n-1-i]``."""
        b, i = wp.tid()
        if b >= num_blocks:
            return
        if reorder_active[b] == int(0):
            return
        n_b = dims[b]
        if i >= n_b:
            return
        vb = vio[b]
        perm[vb + i] = order_buf[vb + (n_b - int(1) - i)]
        if i == 0:
            permutation_valid[b] = int(1)
            permutation_dim[b] = n_b

    return {
        "prepare_reorder": prepare_reorder_kernel,
        "init_and_degree": init_and_degree_kernel,
        "select_and_seed": select_and_seed_kernel,
        "bfs_step": bfs_step_kernel,
        "advance_or_seed": advance_or_seed_kernel,
        "complete_cuda": complete_cuda_kernel,
        "append_unreached": append_unreached_kernel,
        "reverse_into_perm": reverse_into_perm_kernel,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _persistent_block_dim(max_dim: int) -> int:
    """Choose enough CUDA lanes to scan matrix rows cooperatively."""
    return min(1024, max(32, 1 << (max_dim - 1).bit_length()))


def create_rcm_batch_launch(
    A_flat: wp.array[wp.float32],
    perm_flat: wp.array[wp.int32],
    dims: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    scratch: dict,
    num_blocks: int,
    max_dim: int,
    tol: float = 0.0,
    max_bfs_iters: int | None = None,
    use_cuda_graph: bool = True,
    reuse_permutation: bool = False,
    device=None,
    stream=None,
) -> Callable[[], None]:
    """Create a single zero-arg callback that runs RCM on all blocks in parallel.

    Parameters
    ----------
    A_flat, perm_flat:
        Flat buffers for the concatenated block matrices and output permutations.
    dims, mio, vio:
        ``wp.int32`` arrays describing the per-block sizes and flat offsets.
    scratch:
        Caller-owned scratch buffers from :func:`allocate_rcm_batch_scratch`.
        The caller must keep this dict alive for the lifetime of the returned
        callback: the Warp CPU backend does not retain strong Python refs to
        recorded-launch inputs, so scratch owned only by this function's
        locals would be collected and the callback would write into freed
        memory on replay.
    num_blocks, max_dim:
        Host-side sizing used to pick fixed launch dimensions.
    max_bfs_iters:
        Optional approximate traversal cap. By default, all components are traversed.
    tol, use_cuda_graph, reuse_permutation, device, stream:
        Reordering and launch options.
    """
    if perm_flat.dtype != wp.int32:
        raise TypeError(f"perm_flat must be wp.int32; got {perm_flat.dtype}")
    dtype = A_flat.dtype

    if device is None:
        device = A_flat.device
    device = wp.get_device(device)
    complete_traversal = max_bfs_iters is None
    if max_bfs_iters is not None:
        max_bfs_iters = min(max(0, max_bfs_iters), max_dim)

    K = _make_rcm_batch_kernels(dtype)

    prepare_reorder_launch = wp.launch(
        K["prepare_reorder"],
        dim=num_blocks,
        inputs=[
            bool(reuse_permutation),
            dims,
            scratch["permutation_valid"],
            scratch["permutation_dim"],
            scratch["reorder_active"],
        ],
        device=device,
        stream=stream,
        record_cmd=True,
    )
    init_and_degree_launch = wp.launch(
        K["init_and_degree"],
        dim=(num_blocks, max_dim),
        inputs=[
            num_blocks,
            float(tol),
            A_flat,
            dims,
            mio,
            vio,
            scratch["degree"],
            scratch["level"],
            scratch["head"],
            scratch["current_level"],
            scratch["discovered"],
            scratch["reorder_active"],
        ],
        device=device,
        stream=stream,
        record_cmd=True,
    )
    select_and_seed_launch = wp.launch(
        K["select_and_seed"],
        dim=num_blocks,
        inputs=[
            num_blocks,
            dims,
            vio,
            scratch["degree"],
            scratch["level"],
            scratch["order_buf"],
            scratch["head"],
            scratch["reorder_active"],
        ],
        device=device,
        stream=stream,
        record_cmd=True,
    )
    bfs_step_launch = wp.launch(
        K["bfs_step"],
        dim=(num_blocks, max_dim),
        inputs=[
            num_blocks,
            float(tol),
            A_flat,
            dims,
            mio,
            vio,
            scratch["level"],
            scratch["order_buf"],
            scratch["head"],
            scratch["current_level"],
            scratch["discovered"],
            scratch["reorder_active"],
        ],
        device=device,
        stream=stream,
        record_cmd=True,
    )
    advance_or_seed_launch = wp.launch(
        K["advance_or_seed"],
        dim=num_blocks,
        inputs=[
            num_blocks,
            dims,
            vio,
            scratch["degree"],
            scratch["level"],
            scratch["order_buf"],
            scratch["head"],
            scratch["current_level"],
            scratch["discovered"],
            scratch["reorder_active"],
        ],
        device=device,
        stream=stream,
        record_cmd=True,
    )
    complete_cuda_launch = None
    if complete_traversal and device.is_cuda:
        block_dim = _persistent_block_dim(max_dim)
        complete_cuda_launch = wp.launch(
            K["complete_cuda"],
            dim=num_blocks * block_dim,
            inputs=[
                num_blocks,
                float(tol),
                A_flat,
                dims,
                mio,
                vio,
                scratch["degree"],
                scratch["level"],
                scratch["order_buf"],
                scratch["head"],
                scratch["reorder_active"],
            ],
            device=device,
            stream=stream,
            block_dim=block_dim,
            record_cmd=True,
        )
    append_unreached_launch = wp.launch(
        K["append_unreached"],
        dim=num_blocks,
        inputs=[
            num_blocks,
            dims,
            vio,
            scratch["level"],
            scratch["order_buf"],
            scratch["head"],
            scratch["reorder_active"],
        ],
        device=device,
        stream=stream,
        record_cmd=True,
    )
    reverse_launch = wp.launch(
        K["reverse_into_perm"],
        dim=(num_blocks, max_dim),
        inputs=[
            num_blocks,
            dims,
            vio,
            scratch["order_buf"],
            perm_flat,
            scratch["reorder_active"],
            scratch["permutation_valid"],
            scratch["permutation_dim"],
        ],
        device=device,
        stream=stream,
        record_cmd=True,
    )

    def traverse_level():
        bfs_step_launch.launch()
        advance_or_seed_launch.launch()

    def callback():
        prepare_reorder_launch.launch()
        init_and_degree_launch.launch()
        if complete_cuda_launch is not None:
            complete_cuda_launch.launch()
        else:
            select_and_seed_launch.launch()
            iteration_count = max_dim if complete_traversal else int(max_bfs_iters)
            for _ in range(iteration_count):
                traverse_level()
        append_unreached_launch.launch()
        reverse_launch.launch()

    if use_cuda_graph:
        return create_cuda_graph_callback(callback, device=device, stream=stream)
    return callback
