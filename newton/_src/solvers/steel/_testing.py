# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test and tuning seam that pins the kernel and launch choices of SolverSteel."""

import contextlib
import contextvars
from collections.abc import Iterator, Mapping

_KERNEL_OVERRIDES: contextvars.ContextVar[Mapping[str, object] | None] = contextvars.ContextVar(
    "steel_kernel_overrides", default=None
)


@contextlib.contextmanager
def kernel_overrides(overrides: Mapping[str, object]) -> Iterator[None]:
    """Pin the kernel and launch choices of every solver constructed in this context.

    A solver reads the overrides once at construction and keeps them. Tasks and threads that
    inherit this context see the overrides; others do not. Keys: ``cholesky_kernel``, ``trisolve_kernel``,
    ``hinv_jt_kernel``, ``delassus_kernel``, ``pgs_kernel`` (split solve on CUDA:
    ``"auto"``, ``"loop"``, ``"tiled"``, ``"tiled_contact"``, ``"streaming"``),
    ``pgs_chunk_size``, ``tile_threads``, ``serial_kernel_block_dim``,
    ``sparse_mass_matrix``, ``diagonal_mass``, ``propagation_tree_kernel`` and the
    ``colored_*`` sweep variants.
    """
    token = _KERNEL_OVERRIDES.set(dict(overrides))
    try:
        yield
    finally:
        _KERNEL_OVERRIDES.reset(token)


def current_kernel_overrides() -> dict[str, object]:
    """Return a copy of the overrides active in the current context."""
    return dict(_KERNEL_OVERRIDES.get() or {})
