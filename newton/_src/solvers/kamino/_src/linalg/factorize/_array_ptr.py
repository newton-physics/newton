# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Raw-pointer helpers for byte-level offsets into Warp arrays.

These helpers are stored in their own, separate module without ``from __future__ import annotations``,
as this would break the @wp.func_native due to the wp.array[dtype] annotation otherwise.
"""

from ctypes import sizeof

import warp as wp

###
# Module interface
###

__all__ = [
    "get_float32_array_offset_ptr",
    "get_int32_array_offset_ptr",
]

get_array_ptr_cpp = """return (uint64_t)arr.data;"""
"""Native C++ snippet returning the raw device pointer of a Warp array."""


def make_get_array_offset_ptr_func(dtype):
    """Build a Warp function returning ``base_ptr + start_index * sizeof(dtype)``."""

    # Thin native wrapper exposing arr.data as a wp.uint64 inside kernels.
    @wp.func_native(get_array_ptr_cpp)
    def get_dtype_array_ptr(arr: wp.array[dtype]) -> wp.uint64: ...

    # Byte offset from the array base pointer for element start_index.
    @wp.func
    def get_dtype_array_offset_ptr(arr: wp.array[dtype], start_index: int) -> wp.uint64:
        return get_dtype_array_ptr(arr) + wp.uint64(start_index) * wp.uint64(wp.static(sizeof(dtype._type_)))

    return get_dtype_array_offset_ptr


get_int32_array_offset_ptr = make_get_array_offset_ptr_func(wp.int32)
"""A Warp function to get the offset pointer of a wp.int32 warp array."""

get_float32_array_offset_ptr = make_get_array_offset_ptr_func(wp.float32)
"""A Warp function to get the offset pointer of a wp.float32 warp array."""
