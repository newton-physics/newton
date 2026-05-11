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

The batched version amortizes kernel launch overhead by making each stage a
single launch that covers **all** blocks. Each Warp thread block is
responsible for processing a single graph-block's vertex. Out-of-range
threads (``i >= dims[b]``) return early, which lets us safely launch with a
uniform ``max_dim`` over heterogeneous block sizes.

Layout assumptions
------------------

- ``A_flat``: a flat ``wp.array`` containing all block matrices concatenated
  in row-major order. Block ``b``'s matrix starts at offset ``mio[b]`` with
  size ``dims[b] * dims[b]``.
- ``perm_flat``: flat int32 permutation output. Block ``b``'s output starts
  at offset ``vio[b]`` with size ``dims[b]``.
- ``dims``, ``mio``, ``vio``: ``wp.array[int32]`` of length
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

import math
from collections.abc import Callable
from functools import cache

import warp as wp


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
    - Per-block arrays (``head``, ``root``) are sized ``(num_blocks,)``
      and are indexed by block id ``b``.

    The BFS "current level" scalar is *not* allocated here: it is a
    host-side loop counter baked into each pre-recorded ``bfs_step``
    launch, so there is no device-side counter and no intra-launch race
    hazard.
    """
    return {
        "degree": wp.empty(total_vec, dtype=wp.int32, device=device),
        "level": wp.empty(total_vec, dtype=wp.int32, device=device),
        "order_buf": wp.empty(total_vec, dtype=wp.int32, device=device),
        "head": wp.empty(num_blocks, dtype=wp.int32, device=device),
        "root": wp.empty(num_blocks, dtype=wp.int32, device=device),
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

    @wp.kernel(module=module, enable_backward=False)
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
        root: wp.array[wp.int32],  # type: ignore[valid-type]
    ):
        """Launch dims: ``(num_blocks, max_dim)``.

        Thread ``(b, i)`` computes ``degree[vio[b] + i]`` for vertex ``i`` in
        block ``b`` (if ``i < dims[b]``). Thread ``(b, 0)`` also initializes
        the per-block scalars.
        """
        b, i = wp.tid()
        if b >= num_blocks:
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
            root[b] = int(0)

    @wp.kernel(module=module, enable_backward=False)
    def select_and_seed_kernel(
        num_blocks: int,
        dims: wp.array[wp.int32],  # type: ignore[valid-type]
        vio: wp.array[wp.int32],  # type: ignore[valid-type]
        degree: wp.array[wp.int32],  # type: ignore[valid-type]
        level: wp.array[wp.int32],  # type: ignore[valid-type]
        order_buf: wp.array[wp.int32],  # type: ignore[valid-type]
        head: wp.array[wp.int32],  # type: ignore[valid-type]
        root: wp.array[wp.int32],  # type: ignore[valid-type]
    ):
        """Launch dims: ``(num_blocks,)``. Fused root-selection + BFS seed.

        One thread per block does a serialized argmin over that block's
        ``degree`` slice to pick a minimum-degree root, stores it into
        ``root[b]``, then seeds the BFS frontier for that block by writing
        ``level[vb + r] = 0`` and appending ``r`` to the block's
        ``order_buf`` segment (head advances by one). Fine for the intended
        ``n <= ~1000`` regime; merging the two removes one kernel launch
        at the start of every reorder call.
        """
        b = wp.tid()
        if b >= num_blocks:
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
        root[b] = best_idx
        level[vb + best_idx] = int(0)
        # Atomically claim the first slot; at kernel entry head[b] is 0 and
        # only this thread touches it for block ``b``, so the atomic is
        # effectively a plain write to slot 0.
        slot = wp.atomic_add(head, b, int(1))
        order_buf[vb + slot] = best_idx

    @wp.kernel(module=module, enable_backward=False)
    def bfs_step_kernel(
        num_blocks: int,
        cur: int,
        tol: dtype,  # type: ignore[valid-type]
        A: wp.array[dtype],  # type: ignore[valid-type]
        dims: wp.array[wp.int32],  # type: ignore[valid-type]
        mio: wp.array[wp.int32],  # type: ignore[valid-type]
        vio: wp.array[wp.int32],  # type: ignore[valid-type]
        level: wp.array[wp.int32],  # type: ignore[valid-type]
        order_buf: wp.array[wp.int32],  # type: ignore[valid-type]
        head: wp.array[wp.int32],  # type: ignore[valid-type]
    ):
        """Launch dims: ``(num_blocks, max_dim)``. One BFS expansion step.

        The "current BFS level" ``cur`` is passed as a **scalar kernel
        argument** rather than being stored in device memory. The host
        pre-records one launch per iteration with a distinct integer
        ``cur`` baked into each, so every thread in a given launch
        observes the same value. This removes the previous device-side
        ``iter_counter`` scratch array, the per-step 1-thread increment
        launch, and any possibility of an intra-launch race on the
        counter.

        The ``alive`` / ``discovered`` arrays are dropped entirely: when a
        block saturates, no thread in it has ``level == cur`` so the kernel
        does no work for that block. Kernel launch overhead is fixed either
        way, so skipping via an ``alive`` flag saved no time.
        """
        b, i = wp.tid()
        if b >= num_blocks:
            return
        n_b = dims[b]
        if i >= n_b:
            return

        vb = vio[b]
        mb = mio[b]

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
                        slot = wp.atomic_add(head, b, int(1))
                        order_buf[vb + slot] = j

    @wp.kernel(module=module, enable_backward=False)
    def append_unreached_kernel(
        num_blocks: int,
        dims: wp.array[wp.int32],  # type: ignore[valid-type]
        vio: wp.array[wp.int32],  # type: ignore[valid-type]
        level: wp.array[wp.int32],  # type: ignore[valid-type]
        order_buf: wp.array[wp.int32],  # type: ignore[valid-type]
        head: wp.array[wp.int32],  # type: ignore[valid-type]
    ):
        """Launch dims: ``(num_blocks,)``. Appends any vertex with
        ``level == -1`` to each block's ``order_buf`` segment in ascending
        index order. Serialized per block.
        """
        b = wp.tid()
        if b >= num_blocks:
            return
        n_b = dims[b]
        vb = vio[b]
        pos = head[b]
        for i in range(n_b):
            if level[vb + i] == int(-1):
                order_buf[vb + pos] = i
                pos += int(1)
        head[b] = pos

    @wp.kernel(module=module, enable_backward=False)
    def reverse_into_perm_kernel(
        num_blocks: int,
        dims: wp.array[wp.int32],  # type: ignore[valid-type]
        vio: wp.array[wp.int32],  # type: ignore[valid-type]
        order_buf: wp.array[wp.int32],  # type: ignore[valid-type]
        perm: wp.array[wp.int32],  # type: ignore[valid-type]
    ):
        """Launch dims: ``(num_blocks, max_dim)``. ``perm[i] = order_buf[n-1-i]``."""
        b, i = wp.tid()
        if b >= num_blocks:
            return
        n_b = dims[b]
        if i >= n_b:
            return
        vb = vio[b]
        perm[vb + i] = order_buf[vb + (n_b - int(1) - i)]

    return {
        "init_and_degree": init_and_degree_kernel,
        "select_and_seed": select_and_seed_kernel,
        "bfs_step": bfs_step_kernel,
        "append_unreached": append_unreached_kernel,
        "reverse_into_perm": reverse_into_perm_kernel,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _default_bfs_iters(max_dim: int) -> int:
    """Upper bound on BFS depth we actually execute.

    We use ``max_dim`` (the largest block) as the conservative sizing; all
    blocks run the same number of steps in the batched layout. Smaller blocks
    just saturate earlier and the subsequent steps become no-ops for them.

    The classical analytic upper bound is ``2*sqrt(n)`` (Cuthill-McKee on a
    banded matrix with bandwidth ``sqrt(n)``). We keep that full bound as the
    default because lowering it regresses tile-fill on banded-scrambled
    matrices at ~5% density (bandwidth ~ 6 but after scrambling BFS needs
    several more expansion rounds to recover). Any gains from dropping
    launches are then eaten by the extra tiles the factorize/solve kernels
    no longer skip.
    """
    return 2 * int(math.ceil(math.sqrt(max_dim))) + 4


def create_rcm_batch_launch(
    A_flat: wp.array,
    perm_flat: wp.array,
    dims: wp.array,
    mio: wp.array,
    vio: wp.array,
    scratch: dict,
    num_blocks: int,
    max_dim: int,
    tol: float = 0.0,
    max_bfs_iters: int | None = None,
    use_cuda_graph: bool = True,
    device=None,
    stream=None,
) -> Callable[[], None]:
    """Create a single zero-arg callback that runs RCM on all blocks in parallel.

    Parameters
    ----------
    A_flat, perm_flat:
        Flat buffers for the concatenated block matrices and output permutations.
    dims, mio, vio:
        ``int32`` arrays describing the per-block sizes and flat offsets.
    scratch:
        Caller-owned scratch buffers from :func:`allocate_rcm_batch_scratch`.
        The caller must keep this dict alive for the lifetime of the returned
        callback: the Warp CPU backend does not retain strong Python refs to
        recorded-launch inputs, so scratch owned only by this function's
        locals would be collected and the callback would write into freed
        memory on replay.
    num_blocks, max_dim:
        Host-side sizing used to pick fixed launch dimensions.
    tol, max_bfs_iters, use_cuda_graph, device, stream:
        Same semantics as :func:`rcm.create_rcm_launch`.
    """
    if perm_flat.dtype != wp.int32:
        raise TypeError(f"perm_flat must be int32; got {perm_flat.dtype}")
    dtype = A_flat.dtype

    if device is None:
        device = A_flat.device
    if max_bfs_iters is None:
        max_bfs_iters = _default_bfs_iters(max_dim)
    max_bfs_iters = min(max_bfs_iters, max_dim)

    K = _make_rcm_batch_kernels(dtype)

    # Pre-record launches with fixed (num_blocks, max_dim) topology.
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
            scratch["root"],
        ],
        device=device,
        stream=stream,
        record_cmd=True,
    )
    select_and_seed_launch = wp.launch(
        K["select_and_seed"],
        dim=(num_blocks,),
        inputs=[
            num_blocks,
            dims,
            vio,
            scratch["degree"],
            scratch["level"],
            scratch["order_buf"],
            scratch["head"],
            scratch["root"],
        ],
        device=device,
        stream=stream,
        record_cmd=True,
    )
    # Pre-record one bfs_step launch per iteration, each with its own
    # ``cur`` scalar baked in at record time. This removes the need for a
    # device-side iteration counter (and its per-step 1-thread increment
    # launch) and is race-free by construction: every thread in a given
    # launch sees the same compile-time-constant-looking level.
    bfs_step_launches = [
        wp.launch(
            K["bfs_step"],
            dim=(num_blocks, max_dim),
            inputs=[
                num_blocks,
                int(cur),
                float(tol),
                A_flat,
                dims,
                mio,
                vio,
                scratch["level"],
                scratch["order_buf"],
                scratch["head"],
            ],
            device=device,
            stream=stream,
            record_cmd=True,
        )
        for cur in range(max_bfs_iters)
    ]
    append_unreached_launch = wp.launch(
        K["append_unreached"],
        dim=(num_blocks,),
        inputs=[num_blocks, dims, vio, scratch["level"], scratch["order_buf"], scratch["head"]],
        device=device,
        stream=stream,
        record_cmd=True,
    )
    reverse_launch = wp.launch(
        K["reverse_into_perm"],
        dim=(num_blocks, max_dim),
        inputs=[num_blocks, dims, vio, scratch["order_buf"], perm_flat],
        device=device,
        stream=stream,
        record_cmd=True,
    )

    def callback():
        init_and_degree_launch.launch()
        select_and_seed_launch.launch()
        for step in bfs_step_launches:
            step.launch()
        append_unreached_launch.launch()
        reverse_launch.launch()

    if use_cuda_graph:
        return create_cuda_graph_callback(callback, device=device, stream=stream)
    return callback
