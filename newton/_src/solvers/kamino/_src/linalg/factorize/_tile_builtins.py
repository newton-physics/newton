# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Availability checks and fallback for optional Warp tile builtins."""

import inspect
import os
from functools import cache
from pathlib import Path

import warp as wp


def _tile_transpose_update_enabled() -> bool:
    return os.environ.get("NEWTON_KAMINO_DISABLE_TILE_TRANSPOSE_UPDATE") not in {"1", "true", "True"}


def _has_warp_builtin(name: str) -> bool:
    if not _tile_transpose_update_enabled():
        return False

    try:
        from warp._src.context import builtin_functions  # noqa: PLC0415
    except Exception:
        return False
    return name in builtin_functions


def _has_native_left_transpose_update_support() -> bool:
    if not _tile_transpose_update_enabled():
        return False

    try:
        from warp._src import codegen  # noqa: PLC0415

        codegen_source = inspect.getsource(codegen.codegen_snippet)
        tile_header = Path(wp.__file__).resolve().parent / "native" / "tile.h"
        tile_source = tile_header.read_text(encoding="utf-8")
    except Exception:
        return False

    return (
        "template_params" in codegen_source
        and "is_tile(arg.type)" in codegen_source
        and "& {arg.emit()" in codegen_source
        and "tile_add(tile_register_t" in tile_source
        and "tile_add(tile_shared_t" in tile_source
    )


HAS_TILE_MATMUL_TRANSPOSE_UPDATE = _has_warp_builtin("tile_matmul_transpose_update")
HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE = _has_warp_builtin("tile_matmul_left_transpose_update")
HAS_NATIVE_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE = (
    not HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE and _has_native_left_transpose_update_support()
)


@cache
def make_tile_matmul_left_transpose_update_func(block_size: int):
    """Create ``out += alpha * transpose(left) @ right`` for the LLT solve."""
    snippet = """using OutTile = tile_out;
using LeftTile = tile_left;
using RightTile = tile_right;
using T = typename OutTile::Type;
using OutLayout = typename OutTile::Layout;
using LeftLayout = typename LeftTile::Layout;
using RightLayout = typename RightTile::Layout;

static_assert(OutLayout::Shape::N == 2, "out must be 2D");
static_assert(LeftLayout::Shape::N == 2, "left must be 2D");
static_assert(RightLayout::Shape::N == 2, "right must be 2D");
static_assert(LeftLayout::Shape::dim(1) == OutLayout::Shape::dim(0), "left cols must match out rows");
static_assert(RightLayout::Shape::dim(1) == OutLayout::Shape::dim(1), "right cols must match out cols");
static_assert(LeftLayout::Shape::dim(0) == RightLayout::Shape::dim(0), "left/right rows must match");

constexpr int Rows = OutLayout::Shape::dim(0);
constexpr int Cols = OutLayout::Shape::dim(1);
constexpr int K = LeftLayout::Shape::dim(0);

#if defined(__CUDA_ARCH__)
__shared__ T left_values[K * Rows];
__shared__ T right_values[K * Cols];
#else
T left_values[K * Rows];
T right_values[K * Cols];
#endif

left.apply([&](int reg, auto c) { left_values[c[0] * Rows + c[1]] = left.data[reg]; });
right.apply([&](int reg, auto c) { right_values[c[0] * Cols + c[1]] = right.data[reg]; });
WP_TILE_SYNC();

const T a = static_cast<T>(alpha);
for (int linear = WP_TILE_THREAD_IDX; linear < OutLayout::Size; linear += WP_TILE_BLOCK_DIM) {
    auto c = OutLayout::coord_from_linear(linear);
    T sum = T{};
    WP_PRAGMA_UNROLL
    for (int k = 0; k < K; ++k) {
        sum += left_values[k * Rows + c[0]] * right_values[k * Cols + c[1]];
    }
    int reg = linear / WP_TILE_BLOCK_DIM;
    tile_add(out, reg, linear, a * sum);
}
WP_TILE_SYNC();"""

    @wp.func_native(snippet)
    def tile_matmul_left_transpose_update(
        out: wp.tile[float, block_size, 1],
        left: wp.tile[float, block_size, block_size],
        right: wp.tile[float, block_size, 1],
        alpha: float,
    ): ...

    return tile_matmul_left_transpose_update
