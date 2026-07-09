# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A module defining several core types and aliases specific to Kamino."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import warp as wp

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Unions
###


FloatType = wp.float16 | wp.float32 | wp.float64
IntType = wp.int16 | wp.int32 | wp.int64
VecIntType = wp.vec2s | wp.vec2i | wp.vec2l

ArrayLike = np.ndarray | list | tuple | Iterable
"""An Array-like structure for aliasing various data types compatible with numpy."""


FloatArrayLike = np.ndarray | list[float] | list[list[float]]
"""An Array-like structure for aliasing various floating-point data types compatible with numpy."""


IntArrayLike = np.ndarray | list[int] | list[list[int]]
"""An Array-like structure for aliasing various integer data types compatible with numpy."""


###
# Vectors & Matrices
###

# Aliases for vector/matrix sizes without a warp built-in, but that we use in Kamino


class vec1i(wp.types.vector(length=1, dtype=wp.int32)):
    pass


class vec5i(wp.types.vector(length=5, dtype=wp.int32)):
    pass


class vec6i(wp.types.vector(length=6, dtype=wp.int32)):
    pass


class vec1f(wp.types.vector(length=1, dtype=wp.float32)):
    pass


class vec6f(wp.types.vector(length=6, dtype=wp.float32)):
    pass


class vec7f(wp.types.vector(length=7, dtype=wp.float32)):
    pass


class vec8f(wp.types.vector(length=8, dtype=wp.float32)):
    pass


class mat61f(wp.types.matrix(shape=(6, 1), dtype=wp.float32)):
    pass


class mat63f(wp.types.matrix(shape=(6, 3), dtype=wp.float32)):
    pass


class mat34f(wp.types.matrix(shape=(3, 4), dtype=wp.float32)):
    pass


class mat36f(wp.types.matrix(shape=(3, 6), dtype=wp.float32)):
    pass


class mat66f(wp.types.matrix(shape=(6, 6), dtype=wp.float32)):
    pass


###
# Descriptor
###


@dataclass
class Descriptor:
    """
    Base class for entity descriptor objects.

    A descriptor object is one with a designated name and a unique identifier (UID).
    """

    name: str
    """The name of the entity descriptor."""

    uid: str | None = None
    """The unique identifier (UID) of the entity descriptor."""

    @staticmethod
    def _assert_valid_uid(uid: str) -> str:
        """
        Check if a given UID string is valid.

        Args:
            uid: The UID string to validate.

        Returns:
            The validated UID string.

        Raises:
            ValueError: If the UID string is not valid.
        """
        try:
            val = uuid.UUID(uid, version=4)
        except ValueError as err:
            raise ValueError("Invalid UID string.") from err
        return str(val)

    def __post_init__(self):
        """Post-initialization to handle UID generation and validation."""
        if self.uid is None:
            # Generate a new UID if none is provided
            self.uid = str(uuid.uuid4())
        else:
            # Otherwise, validate the provided UID
            self.uid = self._assert_valid_uid(self.uid)

    def __hash__(self):
        """Returns a hash of the Descriptor based on its UID."""
        return hash((self.name, self.uid))

    def __repr__(self):
        """Returns a human-readable string representation of the Descriptor."""
        return f"Descriptor(\nname={self.name},\nuid={self.uid}\n)"


###
# Array constructors
###

_INT32_INFO = np.iinfo(np.int32)


def _check_int32_range(data: IntArrayLike, func_name: str) -> np.ndarray:
    arr = np.asarray(data)
    if arr.size > 0 and not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{func_name} expected integer data, got dtype={arr.dtype}")
    if arr.size > 0:
        v_min = int(arr.min())
        v_max = int(arr.max())
        if v_min < _INT32_INFO.min or v_max > _INT32_INFO.max:
            raise OverflowError(
                f"int32 overflow: values in [{v_min}, {v_max}] outside [{_INT32_INFO.min}, {_INT32_INFO.max}]"
            )
    return arr


def to_warp_int32_array(
    data: IntArrayLike,
    device: wp.DeviceLike | None = None,
) -> wp.array[wp.int32]:
    """Convert ``data`` to a Warp ``int32`` array, asserting all values fit in ``int32``.

    Use this helper in place of ``wp.array(data, dtype=wp.int32)`` and
    ``wp.from_numpy(data, dtype=wp.int32)`` whenever ``data`` originates from
    Python ints or NumPy integers. Direct ``wp.array`` / ``wp.from_numpy`` calls
    silently truncate values that do not fit in ``int32``; this helper raises
    instead.

    Args:
        data: Integer array-like (Python list, tuple, numpy array, or scalar).
        device: Warp device to allocate on. Defaults to the current scoped device.

    Returns:
        A Warp array with dtype :class:`wp.int32` containing the values of ``data``.

    Raises:
        TypeError: if ``data`` is not integer-typed.
        OverflowError: if any value falls outside ``[-2**31, 2**31 - 1]``.
    """
    arr = _check_int32_range(data, "to_warp_int32_array")
    return wp.array(arr.astype(np.int32, copy=False), dtype=wp.int32, device=device)


def assign_to_warp_int32_array(dst: wp.array[wp.int32], data: IntArrayLike) -> None:
    """Assign ``data`` into an existing Warp ``int32`` array ``dst``, asserting no overflow.

    Use this helper in place of ``dst.assign(data)`` whenever ``dst`` is an
    ``int32`` Warp array and ``data`` originates from NumPy integers. Warp's
    ``.assign(np.ndarray)`` silently truncates NumPy ints that do not fit in
    the target dtype; this helper raises instead.

    Args:
        dst: Pre-allocated Warp array with dtype :class:`wp.int32`.
        data: Integer array-like (Python list, tuple, numpy array, or scalar).

    Raises:
        TypeError: if ``dst`` does not have dtype :class:`wp.int32`, or if ``data``
            is not integer-typed.
        OverflowError: if any value falls outside ``[-2**31, 2**31 - 1]``.
    """
    if dst.dtype is not wp.int32:
        raise TypeError(f"assign_to_warp_int32_array expected destination with dtype wp.int32, got dtype={dst.dtype}")
    arr = _check_int32_range(data, "assign_to_warp_int32_array")
    dst.assign(arr.astype(np.int32, copy=False))
