# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internal utilities for simulation builder index containers."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, MutableSequence
from typing import SupportsIndex, overload

import numpy as np


class _IntIndexList(MutableSequence):
    """Compact array-backed storage for 1D integer indices.

    Stores indices in a typed ``array.array("i")`` for lower per-element overhead
    than a plain Python ``list``.  Hot paths for offset-extension are accelerated
    via NumPy in-place addition on the underlying buffer.
    """

    __slots__ = ("_data",)
    __hash__ = None

    def __init__(self, values: Iterable[int] | None = None):
        self._data = array("i")
        if values is not None:
            self.extend(values)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    @overload
    def __getitem__(self, idx: SupportsIndex) -> int: ...

    @overload
    def __getitem__(self, idx: slice) -> list[int]: ...

    def __getitem__(self, idx: SupportsIndex | slice) -> int | list[int]:
        if isinstance(idx, slice):
            return self._data[idx].tolist()
        return self._data[idx]

    @overload
    def __setitem__(self, idx: SupportsIndex, value: int) -> None: ...

    @overload
    def __setitem__(self, idx: slice, value: Iterable[int]) -> None: ...

    def __setitem__(self, idx: SupportsIndex | slice, value: int | Iterable[int]) -> None:
        if isinstance(idx, slice):
            if isinstance(value, _IntIndexList):
                self._data[idx] = value._data
            elif isinstance(value, Iterable):
                self._data[idx] = array("i", (int(v) for v in value))
            else:
                raise TypeError("iterable assignment expected for slice index")
            return
        if isinstance(value, Iterable):
            raise TypeError("int assignment expected for scalar index")
        self._data[idx] = int(value)

    def __delitem__(self, idx: SupportsIndex | slice) -> None:
        raise NotImplementedError("IntIndexList does not support item deletion")

    def insert(self, idx: SupportsIndex, value: int) -> None:
        raise NotImplementedError("IntIndexList does not support insert; use append() or extend()")

    def __contains__(self, value: object) -> bool:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return int(value) in self._data
        return False

    def __repr__(self) -> str:
        return repr(self._data.tolist())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _IntIndexList):
            return self._data == other._data
        if isinstance(other, array):
            return self._data == other
        if isinstance(other, (list, tuple)):
            return self._data.tolist() == list(other)
        return NotImplemented

    def __copy__(self):
        out = _IntIndexList()
        out._data = self._data[:]
        return out

    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self):
        return self.__copy__()

    def append(self, value: int) -> None:
        self._data.append(int(value))

    def extend(self, values: Iterable[int]) -> None:
        if isinstance(values, _IntIndexList):
            self._data.extend(values._data)
            return
        append = self._data.append
        for value in values:
            append(int(value))

    def clear(self) -> None:
        self._data = array("i")

    def tolist(self) -> list[int]:
        return self._data.tolist()

    def __array__(self, dtype=None, copy=None):
        arr = np.frombuffer(self._data, dtype=np.intc)
        if dtype is None or np.dtype(dtype) == np.dtype(np.intc):
            return arr.copy()
        return arr.astype(dtype)

    def extend_with_offset(self, values: Iterable[int], offset: int) -> None:
        if isinstance(values, _IntIndexList):
            if not values._data:
                return

            start = len(self._data)
            self._data.extend(values._data)
            if offset != 0:
                np.frombuffer(self._data, dtype=np.intc)[start:] += int(offset)
            return

        offset = int(offset)
        append = self._data.append
        if offset == 0:
            for value in values:
                append(int(value))
            return

        for value in values:
            append(int(value) + offset)

    def extend_with_offset_except(self, values: Iterable[int], offset: int, sentinel: int = -1) -> None:
        if isinstance(values, _IntIndexList):
            if not values._data:
                return

            start = len(self._data)
            self._data.extend(values._data)
            if offset != 0:
                new_data = np.frombuffer(self._data, dtype=np.intc)[start:]
                new_data[new_data != int(sentinel)] += int(offset)
            return

        offset = int(offset)
        sentinel = int(sentinel)
        append = self._data.append
        if offset == 0:
            for value in values:
                append(int(value))
            return

        for value in values:
            value_int = int(value)
            append(value_int if value_int == sentinel else value_int + offset)

    def extend_with_offset_nonnegative(self, values: Iterable[int], offset: int) -> None:
        if isinstance(values, _IntIndexList):
            if not values._data:
                return

            start = len(self._data)
            self._data.extend(values._data)
            if offset != 0:
                new_data = np.frombuffer(self._data, dtype=np.intc)[start:]
                new_data[new_data >= 0] += int(offset)
            return

        offset = int(offset)
        append = self._data.append
        if offset == 0:
            for value in values:
                append(int(value))
            return

        for value in values:
            value_int = int(value)
            append(value_int + offset if value_int >= 0 else value_int)


class _IntIndexList2D(MutableSequence):
    """Compact array-backed storage for fixed-width integer tuples.

    Rows are stored flat in a single typed ``array.array("i")`` with all elements
    of each row contiguous.  Offset-extension is accelerated via NumPy.
    """

    __slots__ = ("_data", "_width")
    __hash__ = None

    def __init__(self, rows: Iterable[Iterable[int]] | None = None, width: int = 2):
        self._width = int(width)
        if self._width <= 0:
            raise ValueError("width must be positive")
        self._data = array("i")
        if rows is not None:
            self.extend(rows)

    def __len__(self) -> int:
        return len(self._data) // self._width

    def __iter__(self):
        data = self._data
        width = self._width
        for i in range(0, len(data), width):
            yield tuple(data[i : i + width])

    @overload
    def __getitem__(self, idx: SupportsIndex) -> tuple[int, ...]: ...

    @overload
    def __getitem__(self, idx: slice) -> list[tuple[int, ...]]: ...

    def __getitem__(self, idx: SupportsIndex | slice) -> tuple[int, ...] | list[tuple[int, ...]]:
        row_count = len(self)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(row_count)
            return [self[i] for i in range(start, stop, step)]

        idx_int = int(idx)
        if idx_int < 0:
            idx_int += row_count
        if idx_int < 0 or idx_int >= row_count:
            raise IndexError("index out of range")

        base_idx = idx_int * self._width
        return tuple(self._data[base_idx : base_idx + self._width])

    @overload
    def __setitem__(self, idx: SupportsIndex, value: Iterable[int]) -> None: ...

    @overload
    def __setitem__(self, idx: slice, value: Iterable[Iterable[int]]) -> None: ...

    def __setitem__(self, idx: SupportsIndex | slice, value: Iterable[int] | Iterable[Iterable[int]]) -> None:
        if isinstance(idx, slice):
            replacement = _IntIndexList2D(value, width=self._width)
            n = len(self)
            start, stop, step = idx.indices(n)
            if step == 1:
                # Contiguous row range: map directly to the flat buffer.
                self._data[start * self._width : stop * self._width] = replacement._data
            else:
                row_indices = range(start, stop, step)
                if len(row_indices) != len(replacement):
                    raise ValueError(f"cannot assign {len(replacement)} row(s) to a slice of length {len(row_indices)}")
                w = self._width
                for i, row_idx in enumerate(row_indices):
                    flat = row_idx * w
                    self._data[flat : flat + w] = replacement._data[i * w : (i + 1) * w]
            return

        idx_int = int(idx)
        row_count = len(self)
        if idx_int < 0:
            idx_int += row_count
        if idx_int < 0 or idx_int >= row_count:
            raise IndexError("index out of range")

        row = tuple(int(v) for v in value)
        if len(row) != self._width:
            raise ValueError(f"row must have width {self._width}, got {len(row)}")
        base_idx = idx_int * self._width
        self._data[base_idx : base_idx + self._width] = array("i", row)

    def __delitem__(self, idx: SupportsIndex | slice) -> None:
        raise NotImplementedError("IntIndexList2D does not support item deletion")

    def insert(self, idx: SupportsIndex, value: Iterable[int]) -> None:
        raise NotImplementedError("IntIndexList2D does not support insert; use append() or extend()")

    def __contains__(self, row: object) -> bool:
        if not isinstance(row, tuple) or len(row) != self._width:
            return False
        try:
            row_ints = tuple(int(v) for v in row)
        except (TypeError, ValueError):
            return False

        data = self._data
        width = self._width
        for i in range(0, len(data), width):
            if tuple(data[i : i + width]) == row_ints:
                return True
        return False

    def __repr__(self) -> str:
        return repr(self.tolist())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _IntIndexList2D):
            return self._width == other._width and self._data == other._data
        if isinstance(other, (list, tuple)):
            return self.tolist() == [tuple(row) for row in other]
        return NotImplemented

    def __copy__(self):
        out = _IntIndexList2D(width=self._width)
        out._data = self._data[:]
        return out

    def __deepcopy__(self, memo):
        return self.__copy__()

    def copy(self):
        return self.__copy__()

    def append(self, row: Iterable[int]) -> None:
        values = tuple(int(v) for v in row)
        if len(values) != self._width:
            raise ValueError(f"row must have width {self._width}, got {len(values)}")
        self._data.extend(values)

    def extend(self, rows: Iterable[Iterable[int]]) -> None:
        if isinstance(rows, _IntIndexList2D):
            if rows._width != self._width:
                raise ValueError(f"row width mismatch: expected {self._width}, got {rows._width}")
            self._data.extend(rows._data)
            return

        for row in rows:
            self.append(row)

    def clear(self) -> None:
        self._data = array("i")

    def tolist(self) -> list[tuple[int, ...]]:
        return list(self)

    def extend_with_offset(self, rows: Iterable[Iterable[int]], offset: int) -> None:
        if isinstance(rows, _IntIndexList2D):
            if rows._width != self._width:
                raise ValueError(f"row width mismatch: expected {self._width}, got {rows._width}")
            if not rows._data:
                return

            start = len(self._data)
            self._data.extend(rows._data)
            if offset != 0:
                np.frombuffer(self._data, dtype=np.intc)[start:] += int(offset)
            return

        offset = int(offset)
        if offset == 0:
            for row in rows:
                self.append(row)
            return
        w = self._width
        for row in rows:
            values = tuple(int(v) + offset for v in row)
            if len(values) != w:
                raise ValueError(f"row must have width {w}, got {len(values)}")
            self._data.extend(values)

    def extend_with_offset_except(self, rows: Iterable[Iterable[int]], offset: int, sentinel: int = -1) -> None:
        if isinstance(rows, _IntIndexList2D):
            if rows._width != self._width:
                raise ValueError(f"row width mismatch: expected {self._width}, got {rows._width}")
            if not rows._data:
                return

            start = len(self._data)
            self._data.extend(rows._data)
            if offset != 0:
                new_data = np.frombuffer(self._data, dtype=np.intc)[start:]
                new_data[new_data != int(sentinel)] += int(offset)
            return

        offset = int(offset)
        sentinel = int(sentinel)
        if offset == 0:
            for row in rows:
                self.append(row)
            return
        w = self._width
        for row in rows:
            values = tuple(iv if (iv := int(v)) == sentinel else iv + offset for v in row)
            if len(values) != w:
                raise ValueError(f"row must have width {w}, got {len(values)}")
            self._data.extend(values)

    def extend_with_offset_nonnegative(self, rows: Iterable[Iterable[int]], offset: int) -> None:
        if isinstance(rows, _IntIndexList2D):
            if rows._width != self._width:
                raise ValueError(f"row width mismatch: expected {self._width}, got {rows._width}")
            if not rows._data:
                return

            start = len(self._data)
            self._data.extend(rows._data)
            if offset != 0:
                new_data = np.frombuffer(self._data, dtype=np.intc)[start:]
                new_data[new_data >= 0] += int(offset)
            return

        offset = int(offset)
        if offset == 0:
            for row in rows:
                self.append(row)
            return
        w = self._width
        for row in rows:
            values = tuple((iv := int(v)) + offset if iv >= 0 else iv for v in row)
            if len(values) != w:
                raise ValueError(f"row must have width {w}, got {len(values)}")
            self._data.extend(values)
