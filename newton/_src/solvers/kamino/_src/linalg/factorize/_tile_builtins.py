# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Availability checks and native fallbacks for optional Warp tile builtins."""

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


def _has_native_tile_arg_support() -> bool:
    if not _tile_transpose_update_enabled():
        return False

    try:
        from warp._src import codegen  # noqa: PLC0415

        source = inspect.getsource(codegen.codegen_snippet)
    except Exception:
        return False

    return "template_params" in source and "is_tile(arg.type)" in source and "& {arg.emit()" in source


def _has_native_tile_access_helpers() -> bool:
    if not _tile_transpose_update_enabled():
        return False

    try:
        tile_header = Path(wp.__file__).resolve().parent / "native" / "tile.h"
        source = tile_header.read_text(encoding="utf-8")
    except Exception:
        return False

    return (
        "tile_read(const tile_register_t" in source
        and "tile_read(const tile_shared_t" in source
        and "tile_add(tile_register_t" in source
        and "tile_add(tile_shared_t" in source
    )


def _has_native_tile_update_support() -> bool:
    return _has_native_tile_arg_support() and _has_native_tile_access_helpers()


def _copy_dense_2d_snippet(tile_name: str, layout_name: str, values_name: str, cols_name: str, storage: str) -> str:
    if storage == "register":
        return (
            f"{tile_name}.apply([&](int reg, auto c) {{ "
            f"{values_name}[c[0] * {cols_name} + c[1]] = {tile_name}.data[reg]; }});"
        )
    if storage == "generic":
        return f"""for (int linear = WP_TILE_THREAD_IDX; linear < {layout_name}::Size; linear += WP_TILE_BLOCK_DIM) {{
    auto c = {layout_name}::coord_from_linear(linear);
    int reg = linear / WP_TILE_BLOCK_DIM;
    {values_name}[c[0] * {cols_name} + c[1]] = tile_read({tile_name}, reg, linear);
}}"""
    raise ValueError(f"Unsupported tile storage specialization: {storage!r}")


def _update_output_snippet(
    layout_name: str,
    output_name: str,
    rows_name: str,
    cols_name: str,
    k_name: str,
    left_values_name: str,
    right_values_name: str,
    left_transposed: bool,
    storage: str,
) -> str:
    if left_transposed:
        product = f"{left_values_name}[k * {rows_name} + c[0]] * {right_values_name}[k * {cols_name} + c[1]]"
    else:
        product = f"{left_values_name}[c[0] * {k_name} + k] * {right_values_name}[c[1] * {k_name} + k]"

    if storage == "shared":
        write = f"""const T value = a * sum;
    if constexpr ({layout_name}::Unique)
        {output_name}.data(linear) += value;
    else
        wp::atomic_add(&{output_name}.data(linear), value);"""
    elif storage == "register":
        return f"""const T a = static_cast<T>(alpha);
{output_name}.apply([&](int reg, auto c) {{
    T sum = T{{}};
    WP_PRAGMA_UNROLL
    for (int k = 0; k < {k_name}; ++k) {{
        sum += {product};
    }}
    {output_name}.data[reg] += a * sum;
}});
WP_TILE_SYNC();"""
    elif storage == "generic":
        write = f"""int reg = linear / WP_TILE_BLOCK_DIM;
    tile_add({output_name}, reg, linear, a * sum);"""
    else:
        raise ValueError(f"Unsupported tile storage specialization: {storage!r}")

    return f"""const T a = static_cast<T>(alpha);
for (int linear = WP_TILE_THREAD_IDX; linear < {layout_name}::Size; linear += WP_TILE_BLOCK_DIM) {{
    auto c = {layout_name}::coord_from_linear(linear);
    T sum = T{{}};
    WP_PRAGMA_UNROLL
    for (int k = 0; k < {k_name}; ++k) {{
        sum += {product};
    }}
    {write}
}}
WP_TILE_SYNC();"""


def _make_tile_matmul_transpose_update_snippet(out_storage: str, input_storage: str) -> str:
    copy_left = _copy_dense_2d_snippet("left", "LeftLayout", "left_values", "K", input_storage)
    copy_right = _copy_dense_2d_snippet("right", "RightLayout", "right_values", "K", input_storage)
    update_out = _update_output_snippet(
        "OutLayout", "out", "Rows", "Cols", "K", "left_values", "right_values", False, out_storage
    )
    return f"""using OutTile = tile_out;
using LeftTile = tile_left;
using RightTile = tile_right;
using T = typename OutTile::Type;
using OutLayout = typename OutTile::Layout;
using LeftLayout = typename LeftTile::Layout;
using RightLayout = typename RightTile::Layout;

static_assert(OutLayout::Shape::N == 2, "out must be 2D");
static_assert(LeftLayout::Shape::N == 2, "left must be 2D");
static_assert(RightLayout::Shape::N == 2, "right must be 2D");
static_assert(LeftLayout::Shape::dim(0) == OutLayout::Shape::dim(0), "left rows must match out rows");
static_assert(RightLayout::Shape::dim(0) == OutLayout::Shape::dim(1), "right rows must match out cols");
static_assert(LeftLayout::Shape::dim(1) == RightLayout::Shape::dim(1), "left/right cols must match");

constexpr int Rows = OutLayout::Shape::dim(0);
constexpr int Cols = OutLayout::Shape::dim(1);
constexpr int K = LeftLayout::Shape::dim(1);

#if defined(__CUDA_ARCH__)
__shared__ T left_values[Rows * K];
__shared__ T right_values[Cols * K];
#else
T left_values[Rows * K];
T right_values[Cols * K];
#endif

{copy_left}
{copy_right}
WP_TILE_SYNC();

{update_out}
"""


def _make_tile_matmul_left_transpose_update_snippet(out_storage: str, input_storage: str) -> str:
    copy_left = _copy_dense_2d_snippet("left", "LeftLayout", "left_values", "Rows", input_storage)
    copy_right = _copy_dense_2d_snippet("right", "RightLayout", "right_values", "Cols", input_storage)
    update_out = _update_output_snippet(
        "OutLayout", "out", "Rows", "Cols", "K", "left_values", "right_values", True, out_storage
    )
    return f"""using OutTile = tile_out;
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

{copy_left}
{copy_right}
WP_TILE_SYNC();

{update_out}
"""


HAS_TILE_MATMUL_TRANSPOSE_UPDATE = _has_warp_builtin("tile_matmul_transpose_update")
HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE = _has_warp_builtin("tile_matmul_left_transpose_update")
HAS_NATIVE_TILE_MATMUL_TRANSPOSE_UPDATE = not HAS_TILE_MATMUL_TRANSPOSE_UPDATE and _has_native_tile_update_support()
HAS_NATIVE_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE = (
    not HAS_TILE_MATMUL_LEFT_TRANSPOSE_UPDATE and _has_native_tile_update_support()
)


@cache
def make_tile_matmul_transpose_update_func(
    block_size: int, out_storage: str = "shared", input_storage: str = "register"
):
    """Create ``out += alpha * left @ transpose(right)`` as a native tile function."""
    snippet = _make_tile_matmul_transpose_update_snippet(out_storage, input_storage)

    @wp.func_native(snippet)
    def tile_matmul_transpose_update(
        out: wp.tile[float, block_size, block_size],
        left: wp.tile[float, block_size, block_size],
        right: wp.tile[float, block_size, block_size],
        alpha: float,
    ): ...

    return tile_matmul_transpose_update


@cache
def make_tile_matmul_left_transpose_update_func(
    block_size: int, out_storage: str = "generic", input_storage: str = "register"
):
    """Create ``out += alpha * transpose(left) @ right`` as a native tile function."""
    snippet = _make_tile_matmul_left_transpose_update_snippet(out_storage, input_storage)

    @wp.func_native(snippet)
    def tile_matmul_left_transpose_update(
        out: wp.tile[float, block_size, 1],
        left: wp.tile[float, block_size, block_size],
        right: wp.tile[float, block_size, 1],
        alpha: float,
    ): ...

    return tile_matmul_left_transpose_update
