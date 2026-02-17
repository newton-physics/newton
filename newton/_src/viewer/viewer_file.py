# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import warp as wp
from warp._src import types as warp_types

from ..core.types import override
from ..geometry import Mesh
from ..sim import Model, State
from .viewer import ViewerBase

# Optional CBOR2 support
try:
    import cbor2

    HAS_CBOR2 = True
except ImportError:
    HAS_CBOR2 = False


T = TypeVar("T")


class RingBuffer(Generic[T]):
    """
    A ring buffer that behaves like a list but only keeps the last N items.

    This class provides a list-like interface while maintaining a fixed capacity.
    When the buffer is full, new items overwrite the oldest items.
    """

    def __init__(self, capacity: int = 100):
        """
        Initialize the ring buffer.

        Args:
            capacity (int): Maximum number of items to store. Default is 100.
        """
        self.capacity = capacity
        self._buffer: list[T] = []
        self._start = 0  # Index of the oldest item
        self._size = 0  # Current number of items

    def append(self, item: T) -> None:
        """Add an item to the buffer."""
        if self._size < self.capacity:
            # Buffer not full yet, just append
            self._buffer.append(item)
            self._size += 1
        else:
            # Buffer is full, overwrite the oldest item
            self._buffer[self._start] = item
            self._start = (self._start + 1) % self.capacity

    def __len__(self) -> int:
        """Return the number of items in the buffer."""
        return self._size

    def __getitem__(self, index: int) -> T:
        """Get an item by index (0 is the oldest item)."""
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")

        if not (0 <= index < self._size):
            raise IndexError(f"Index {index} out of range [0, {self._size})")

        # Convert logical index to physical buffer index
        if self._size < self.capacity:
            # Buffer not full, simple indexing
            return self._buffer[index]
        else:
            # Buffer is full, need to account for wrap-around
            physical_index = (self._start + index) % self.capacity
            return self._buffer[physical_index]

    def __setitem__(self, index: int, value: T) -> None:
        """Set an item by index."""
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")

        if not (0 <= index < self._size):
            raise IndexError(f"Index {index} out of range [0, {self._size})")

        # Convert logical index to physical buffer index
        if self._size < self.capacity:
            # Buffer not full, simple indexing
            self._buffer[index] = value
        else:
            # Buffer is full, need to account for wrap-around
            physical_index = (self._start + index) % self.capacity
            self._buffer[physical_index] = value

    def __iter__(self):
        """Iterate over items in order (oldest to newest)."""
        for i in range(self._size):
            yield self[i]

    def clear(self) -> None:
        """Clear all items from the buffer."""
        self._buffer.clear()
        self._start = 0
        self._size = 0

    def to_list(self) -> list[T]:
        """Convert the ring buffer to a regular list."""
        return [self[i] for i in range(self._size)]

    def from_list(self, items: list[T]) -> None:
        """Replace buffer contents with items from a list."""
        self.clear()
        for item in items:
            self.append(item)


class ArrayCache(Generic[T]):
    """
    Cache that assigns a monotonically increasing index to each unique key and stores an object with it.

    - Keys are uint64-compatible integers (use Python int).
    - Values are stored alongside the assigned index.
    - During serialization, repeated keys return their existing index; new keys return -1 and are added.
    - During deserialization, lookups happen by index and return the associated object or raise if missing.
    """

    def __init__(self):
        self._key_to_entry: dict[int, tuple[int, T]] = {}
        self._index_to_entry: dict[int, T] = {}
        self._next_index: int = 1

    def try_register_pointer_and_value(self, key: int, value: T) -> int:
        """
        Register an object under a numeric key.

        Args:
            key: Unsigned 64-bit compatible integer key
            value: Object to cache

        Returns:
            Existing index if the key already exists; otherwise 0 after inserting a new entry.
        """
        existing_entry = self._key_to_entry.get(key, None)
        if existing_entry is not None:
            existing_index, _ = existing_entry
            return existing_index

        assigned_index = self._next_index
        self._next_index += 1
        self._key_to_entry[key] = (assigned_index, value)
        self._index_to_entry[assigned_index] = value
        return 0

    def try_get_value(self, index: int) -> T:
        """
        Resolve an object by its index.

        Args:
            index: Previously assigned index from try_register_pointer_and_value() or
                  try_register_pointer_and_value_and_index()

        Returns:
            The object associated with the given index.
        """
        return self._index_to_entry[index]

    def try_register_pointer_and_value_and_index(self, key: int, value: T, index: int) -> int:
        """
        Register an object with an explicit, well-defined index (used during deserialization).

        - If the key already exists, the stored index must equal the provided index.
          Returns that index, or raises on mismatch.
        - If the key is new, the provided index must not be used by another entry.
          Adds the mapping and returns the index.
        - Advances the internal next-index counter if necessary.
        """
        existing_entry = self._key_to_entry.get(key, None)
        if existing_entry is not None:
            existing_index, existing_value = existing_entry
            if existing_index != index:
                raise ValueError(
                    f"ArrayCache: key already registered with a different index (have {existing_index}, got {index})"
                )
            return existing_index

        existing_value = self._index_to_entry.get(index, None)
        if existing_value is not None:
            raise ValueError(f"ArrayCache: index {index} already in use for another entry")

        self._key_to_entry[key] = (index, value)
        self._index_to_entry[index] = value
        if index >= self._next_index:
            self._next_index = index + 1
        return index

    def get_index_for_key(self, key: int) -> int:
        """Return the assigned index for an existing key, else raise KeyError."""
        existing_entry = self._key_to_entry.get(key, None)
        if existing_entry is None:
            raise KeyError(f"ArrayCache: key {key} not found")
        return existing_entry[0]

    def clear(self) -> None:
        """Remove all entries and reset the index counter."""
        self._key_to_entry.clear()
        self._index_to_entry.clear()
        self._next_index = 1

    def __len__(self) -> int:
        return len(self._key_to_entry)


def _get_serialization_format(file_path: str) -> str:
    """
    Determine serialization format based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        'json' for .json files, 'cbor2' for .bin files

    Raises:
        ValueError: If file extension is not supported
    """
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".json":
        return "json"
    elif ext == ".bin":
        if not HAS_CBOR2:
            raise ImportError("cbor2 library is required for .bin files. Install with: pip install cbor2")
        return "cbor2"
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Supported extensions: .json, .bin")


def _ptr_key_from_numpy(arr: np.ndarray) -> int:
    # Use the underlying buffer address as a stable key within a process
    # for non-aliased arrays. For views, this still points to the base buffer;
    # since user guarantees no aliasing across arrays, we can use the data address.
    return int(arr.__array_interface__["data"][0])


_NP_TAG = 1 << 60
_WARP_TAG = 2 << 60
_MESH_TAG = 3 << 60


def _np_key(arr: np.ndarray) -> int:
    return _NP_TAG + _ptr_key_from_numpy(arr)


def _warp_key(x) -> int:
    try:
        base = int(x.ptr)
    except Exception:
        base = int(id(x))
    return _WARP_TAG + base


def _mesh_key_from_vertices(vertices: np.ndarray, fallback_obj=None) -> int:
    try:
        base = _ptr_key_from_numpy(vertices)
    except Exception:
        base = int(id(fallback_obj)) if fallback_obj is not None else int(id(vertices))
    return _MESH_TAG + base


def serialize_ndarray(arr: np.ndarray, format_type: str = "json", cache: ArrayCache | None = None) -> dict:
    """
    Serialize a numpy ndarray to a dictionary representation.

    Args:
        arr: The numpy array to serialize.
        format_type: The serialization format ('json' or 'cbor2').

    Returns:
        A dictionary containing the array's type, dtype, shape, and data.
    """
    if format_type == "json":
        return {
            "__type__": "numpy.ndarray",
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "data": json.dumps(arr.tolist()),
        }
    elif format_type == "cbor2":
        try:
            arr_c = np.ascontiguousarray(arr)
            # Required check to test if tobytes will work without using pickle internally
            # arr.view will throw an exception if the dtype is not supported
            arr.view(dtype=np.float32)
            if cache is None:
                return {
                    "__type__": "numpy.ndarray",
                    "dtype": arr.dtype.str,
                    "shape": arr.shape,
                    "order": "C",
                    "binary_data": arr_c.tobytes(order="C"),
                }
            # Cache-aware: assign or reuse an index
            key = _np_key(arr_c)
            idx = cache.try_register_pointer_and_value(key, arr_c)
            if idx == 0:
                # First occurrence: write full payload with index
                assigned = cache.get_index_for_key(key)
                return {
                    "__type__": "numpy.ndarray",
                    "dtype": arr_c.dtype.str,
                    "shape": arr_c.shape,
                    "order": "C",
                    "binary_data": arr_c.tobytes(order="C"),
                    "cache_index": int(assigned),
                }
            else:
                # Reference only
                return {
                    "__type__": "numpy.ndarray_ref",
                    "cache_index": int(idx),
                    "dtype": arr_c.dtype.str,
                    "shape": arr_c.shape,
                    "order": "C",
                }
        except (ValueError, TypeError):
            # Fallback to list serialization for dtypes that can't be serialized as binary
            return {
                "__type__": "numpy.ndarray",
                "dtype": str(arr.dtype),
                "shape": arr.shape,
                "data": arr.tolist(),
                "is_binary": False,
            }
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")


def deserialize_ndarray(data: dict, format_type: str = "json", cache: ArrayCache | None = None) -> np.ndarray:
    """
    Deserialize a dictionary representation back to a numpy ndarray.

    Args:
        data: Dictionary containing the serialized array data.
        format_type: The serialization format ('json' or 'cbor2').

    Returns:
        The reconstructed numpy array.
    """
    if data.get("__type__") == "numpy.ndarray_ref":
        if cache is None:
            raise ValueError("ArrayCache is required to resolve numpy.ndarray_ref")
        ref_index = int(data["cache_index"])
        # Try to resolve immediately; if not yet registered (forward ref), defer
        try:
            return cache.try_get_value(ref_index)
        except KeyError:
            return {"__cache_ref__": {"index": ref_index, "kind": "numpy"}}

    if data.get("__type__") != "numpy.ndarray":
        raise ValueError("Invalid data format for numpy array deserialization")

    dtype = np.dtype(data["dtype"])
    shape = tuple(data["shape"])

    if format_type == "json":
        array_data = json.loads(data["data"])
        return np.array(array_data, dtype=dtype).reshape(shape)
    elif format_type == "cbor2":
        if "binary_data" in data:
            binary = data["binary_data"]
            order = data.get("order", "C")
            arr = np.frombuffer(binary, dtype=dtype)
            arr = arr.reshape(shape, order=order)
            # Register in cache if available and index provided
            if cache is not None and "cache_index" in data:
                # We cannot recover a stable pointer from bytes; use id(arr.data) as key surrogate
                # Since no aliasing is guaranteed, each full array is unique in the stream
                key = _np_key(arr)
                cache.try_register_pointer_and_value_and_index(key, arr, int(data["cache_index"]))
            return arr
        else:
            # Fallback to list deserialization for non-binary data
            array_data = data["data"]
            return np.array(array_data, dtype=dtype).reshape(shape)
    else:
        raise ValueError(f"Unsupported format_type: {format_type}")


def serialize(obj, callback, _visited=None, _path="", format_type="json", cache: ArrayCache | None = None):
    """
    Recursively serialize an object into a dict, handling primitives,
    containers, and custom class instances. Calls callback(obj) for every object
    and replaces obj with the callback's return value before continuing.

    Args:
        obj: The object to serialize.
        callback: A function taking two arguments (the object and current path) and returning the (possibly transformed) object.
        _visited: Internal set to avoid infinite recursion from circular references.
        _path: Internal parameter tracking the current path/member name.
        format_type: The serialization format ('json' or 'cbor2').
    """
    if _visited is None:
        _visited = set()

    # Run through callback first (object may be replaced)
    result = callback(obj, _path)
    if result is not obj:
        return result

    obj_id = id(obj)
    if obj_id in _visited:
        return "<circular_reference>"

    # Add to visited set (stack-like behavior)
    _visited.add(obj_id)

    try:
        # Primitive types
        if isinstance(obj, str | int | float | bool | type(None)):
            return {"__type__": type(obj).__name__, "value": obj}

        # NumPy scalar types
        if isinstance(obj, np.number):
            # Normalize to "numpy.<typename>" for compatibility with deserializer
            return {
                "__type__": f"numpy.{type(obj).__name__}",
                "value": obj.item(),  # Convert numpy scalar to Python scalar
            }

        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return serialize_ndarray(obj, format_type, cache)

        # Mappings (like dict)
        if isinstance(obj, Mapping):
            return {
                "__type__": type(obj).__name__,
                "items": {
                    str(k): serialize(
                        v, callback, _visited, f"{_path}.{k}" if _path else str(k), format_type, cache=cache
                    )
                    for k, v in obj.items()
                },
            }

        # Iterables (like list, tuple, set)
        if isinstance(obj, Iterable) and not isinstance(obj, str | bytes | bytearray):
            return {
                "__type__": type(obj).__name__,
                "items": [
                    serialize(
                        item, callback, _visited, f"{_path}[{i}]" if _path else f"[{i}]", format_type, cache=cache
                    )
                    for i, item in enumerate(obj)
                ],
            }

        # Custom object — serialize attributes
        if hasattr(obj, "__dict__"):
            return {
                "__type__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
                "attributes": {
                    attr: serialize(
                        value, callback, _visited, f"{_path}.{attr}" if _path else attr, format_type, cache=cache
                    )
                    for attr, value in vars(obj).items()
                },
            }

        # Fallback — non-serializable type
        raise ValueError(f"Cannot serialize object of type {type(obj)}")
    finally:
        # Remove from visited set when done (stack-like cleanup)
        _visited.discard(obj_id)


def _is_struct_dtype(dtype) -> bool:
    """Check if a warp dtype is a struct type (decorated with @wp.struct)."""
    return type(dtype).__name__ == "Struct"


def _serialize_warp_dtype(dtype) -> dict:
    """
    Serialize a warp dtype with full metadata for proper reconstruction.

    For built-in types (vec3f, mat33f, etc.), just stores the type string.
    For dynamically created types (vec_t, mat_t), also stores length/shape
    and scalar type metadata to enable reconstruction.

    Args:
        dtype: The warp dtype to serialize.

    Returns:
        A dict containing dtype info that can be used to reconstruct the type.
    """
    dtype_str = str(dtype)
    dtype_name = dtype.__name__

    result = {"__dtype__": dtype_str}

    # Check if this is a dynamically created type that needs extra metadata
    if dtype_name in ("vec_t", "mat_t", "quat_t"):
        # Get scalar type
        try:
            scalar_type = warp_types.type_scalar_type(dtype)
            result["__scalar_type__"] = scalar_type.__name__
        except Exception:
            pass

        # Get length/shape
        try:
            length = warp_types.type_length(dtype)
            result["__type_length__"] = length
        except Exception:
            pass

        # For matrices, also get shape
        if hasattr(dtype, "_shape_"):
            result["__type_shape__"] = list(dtype._shape_)

    return result


def pointer_as_key(obj, format_type: str = "json", cache: ArrayCache | None = None):
    def callback(x, path):
        if isinstance(x, wp.array):
            # Skip arrays with struct dtypes - they can't be serialized
            if _is_struct_dtype(x.dtype):
                return None
            # Get dtype info with metadata for dynamic types
            dtype_info = _serialize_warp_dtype(x.dtype)
            # Use device pointer as cache key
            if cache is not None:
                key = _warp_key(x)
                idx = cache.try_register_pointer_and_value(key, x)
                if idx > 0:
                    return {
                        "__type__": "warp.array_ref",
                        **dtype_info,
                        "cache_index": int(idx),
                    }
                # First occurrence: store full payload plus cache_index
                assigned = cache.get_index_for_key(key)
                return {
                    "__type__": "warp.array",
                    **dtype_info,
                    "cache_index": int(assigned),
                    # Avoid nested cache for raw bytes to keep warp-level dedup authoritative
                    "data": serialize_ndarray(x.numpy(), format_type, cache=None),
                }
            # No cache: fall back to plain encoding
            return {
                "__type__": "warp.array",
                **dtype_info,
                "data": serialize_ndarray(x.numpy(), format_type, cache=None),
            }

        if isinstance(x, wp.HashGrid):
            return {"__type__": "warp.HashGrid", "data": None}

        if isinstance(x, wp.Mesh):
            return {"__type__": "warp.Mesh", "data": None}

        if isinstance(x, Mesh):
            # Use vertices buffer address as mesh key
            mesh_data = {
                "vertices": serialize_ndarray(x.vertices, format_type, cache),
                "indices": serialize_ndarray(x.indices, format_type, cache),
                "is_solid": x.is_solid,
                "has_inertia": x.has_inertia,
                "maxhullvert": x.maxhullvert,
                "mass": x.mass,
                "com": [float(x.com[0]), float(x.com[1]), float(x.com[2])],
                "I": serialize_ndarray(np.array(x.I), format_type, cache),
            }
            if cache is not None:
                mesh_key = _mesh_key_from_vertices(x.vertices, fallback_obj=x)
                idx = cache.try_register_pointer_and_value(mesh_key, x)
                if idx > 0:
                    return {"__type__": "newton.geometry.Mesh_ref", "cache_index": int(idx)}
                assigned = cache.get_index_for_key(mesh_key)
                return {"__type__": "newton.geometry.Mesh", "cache_index": int(assigned), "data": mesh_data}
            return {"__type__": "newton.geometry.Mesh", "data": mesh_data}

        if isinstance(x, wp.Device):
            return {"__type__": "wp.Device", "data": None}

        if callable(x):
            return {"__type__": "callable", "data": None}

        return x

    return serialize(obj, callback, format_type=format_type, cache=cache)


def transfer_to_model(source_dict, target_obj, post_load_init_callback=None, _path=""):
    """
    Recursively transfer values from source_dict to target_obj, respecting the tree structure.
    Only transfers values where both source and target have matching attributes.

    Args:
        source_dict: Dictionary containing the values to transfer (from deserialization).
        target_obj: Target object to receive the values.
        post_load_init_callback: Optional function taking (target_obj, path) called after all children are processed.
        _path: Internal parameter tracking the current path.
    """
    if not hasattr(target_obj, "__dict__"):
        return

    # Handle case where source_dict is not a dict (primitive value)
    if not isinstance(source_dict, dict):
        return

    # Iterate through all attributes of the target object
    for attr_name in dir(target_obj):
        # Skip private/magic methods and properties
        if attr_name.startswith("_"):
            continue

        # Skip if attribute doesn't exist in target or is not settable
        try:
            target_value = getattr(target_obj, attr_name)
        except (AttributeError, TypeError, RuntimeError):
            # Skip attributes that can't be accessed (including CUDA stream on CPU devices)
            continue

        # Skip methods and non-data attributes
        if callable(target_value):
            continue

        # Check if source_dict has this attribute (optimization: single dict lookup)
        source_value = source_dict.get(attr_name, _MISSING := object())
        if source_value is _MISSING:
            continue

        # Handle different types of values
        if hasattr(target_value, "__dict__") and isinstance(source_value, dict):
            # Recursively transfer for custom objects
            # Build path only when needed (optimization: lazy string formatting)
            current_path = f"{_path}.{attr_name}" if _path else attr_name
            transfer_to_model(source_value, target_value, post_load_init_callback, current_path)
        elif isinstance(source_value, list | tuple) and hasattr(target_value, "__len__"):
            # Handle sequences - try to transfer if lengths match or target is empty
            try:
                # Optimization: cache len() call to avoid redundant computation
                target_len = len(target_value)
                if target_len == 0 or target_len == len(source_value):
                    # For now, just assign the value directly
                    # In a more sophisticated implementation, you might want to handle
                    # element-wise transfer for lists of objects
                    setattr(target_obj, attr_name, source_value)
            except (TypeError, AttributeError):
                # If we can't handle the sequence, try direct assignment
                try:
                    setattr(target_obj, attr_name, source_value)
                except (AttributeError, TypeError):
                    # Skip if we can't set the attribute
                    pass
        else:
            # Direct assignment for primitive types and other values
            try:
                setattr(target_obj, attr_name, source_value)
            except (AttributeError, TypeError):
                # Skip if we can't set the attribute (e.g., read-only property)
                pass

    # Call post_load_init_callback after all children have been processed
    if post_load_init_callback is not None:
        post_load_init_callback(target_obj, _path)


def deserialize(data, callback, _path="", format_type="json", cache: ArrayCache | None = None):
    """
    Recursively deserialize a dict back into objects, handling primitives,
    containers, and custom class instances. Calls callback(obj, path) for every object
    and replaces obj with the callback's return value before continuing.

    Args:
        data: The serialized data to deserialize.
        callback: A function taking two arguments (the data dict and current path) and returning the (possibly transformed) object.
        _path: Internal parameter tracking the current path/member name.
        format_type: The serialization format ('json' or 'cbor2').
    """
    # Run through callback first (object may be replaced)
    result = callback(data, _path)
    if result is not data:
        return result

    # If not a dict with __type__, return as-is
    if not isinstance(data, dict) or "__type__" not in data:
        return data

    type_name = data["__type__"]

    # Primitive types
    if type_name in ("str", "int", "float", "bool", "NoneType"):
        return data["value"]

    # NumPy scalar types
    if type_name.startswith("numpy."):
        if type_name == "numpy.ndarray":
            return deserialize_ndarray(data, format_type, cache)
        else:
            # NumPy scalar types
            numpy_type = getattr(np, type_name.split(".")[-1])
            return numpy_type(data["value"])

    # Mappings (like dict)
    if type_name == "dict":
        return {
            k: deserialize(v, callback, f"{_path}.{k}" if _path else k, format_type, cache)
            for k, v in data["items"].items()
        }

    # Iterables (like list, tuple, set)
    if type_name in ("list", "tuple", "set"):
        items = [
            deserialize(item, callback, f"{_path}[{i}]" if _path else f"[{i}]", format_type, cache)
            for i, item in enumerate(data["items"])
        ]
        if type_name == "tuple":
            return tuple(items)
        elif type_name == "set":
            return set(items)
        else:
            return items

    # Custom objects
    if "attributes" in data:
        # For now, return a simple dict representation
        # In a full implementation, you might want to reconstruct the actual class
        return {
            attr: deserialize(value, callback, f"{_path}.{attr}" if _path else attr, format_type, cache)
            for attr, value in data["attributes"].items()
        }

    # Unknown type - return the data as-is
    return data["value"] if isinstance(data, dict) and "value" in data else data


def extract_type_path(class_str: str) -> str:
    """
    Extracts the fully qualified type name from a string like:
    "<class 'warp.types.uint64'>"
    """
    # The format is always "<class '...'>", so we strip the prefix/suffix
    if class_str.startswith("<class '") and class_str.endswith("'>"):
        return class_str[len("<class '") : -len("'>")]
    raise ValueError(f"Unexpected format: {class_str}")


def extract_last_type_name(class_str: str) -> str:
    """
    Extracts the last type name from a string like:
    "<class 'warp.types.uint64'>" -> "uint64"
    """
    if class_str.startswith("<class '") and class_str.endswith("'>"):
        inner = class_str[len("<class '") : -len("'>")]
        return inner.split(".")[-1]
    raise ValueError(f"Unexpected format: {class_str}")


# Mapping of scalar type names to warp scalar types and suffixes
_SCALAR_TYPE_MAP = {
    "float32": (wp.float32, "f"),
    "float64": (wp.float64, "d"),
    "int32": (wp.int32, "i"),
    "int64": (wp.int64, "l"),
    "uint32": (wp.uint32, "u"),
    "uint64": (wp.uint64, "ul"),
    "int16": (wp.int16, "h"),
    "uint16": (wp.uint16, "uh"),
    "int8": (wp.int8, "b"),
    "uint8": (wp.uint8, "ub"),
}

# Mapping numpy dtypes to warp scalar suffixes
_NUMPY_DTYPE_TO_SUFFIX = {
    np.float32: "f",
    np.float64: "d",
    np.int32: "i",
    np.int64: "l",
    np.uint32: "u",
    np.uint64: "ul",
}

# Mapping numpy dtypes to warp scalar types (for dynamic type creation)
_NUMPY_TO_WARP_SCALAR = {
    np.float32: wp.float32,
    np.float64: wp.float64,
    np.int32: wp.int32,
    np.int64: wp.int64,
    np.uint32: wp.uint32,
    np.uint64: wp.uint64,
    np.int16: wp.int16,
    np.uint16: wp.uint16,
    np.int8: wp.int8,
    np.uint8: wp.uint8,
}


def _resolve_warp_dtype(dtype_str: str, serialized_data: dict | None = None, np_arr: np.ndarray | None = None):
    """
    Resolve a dtype string to a warp dtype, with backwards compatibility.

    Uses metadata from serialized_data when available (for new recordings),
    falls back to inferring from numpy array shape (for old recordings).

    Args:
        dtype_str: The dtype name extracted from serialized data.
        serialized_data: Optional dict containing dtype metadata (__scalar_type__, __type_length__, __type_shape__).
        np_arr: Optional numpy array to infer shape for generic types (fallback for old recordings).

    Returns:
        The warp dtype object.

    Raises:
        AttributeError: If the dtype cannot be resolved.
    """
    # Try direct lookup first
    if hasattr(wp, dtype_str):
        return getattr(wp, dtype_str)

    # Try to reconstruct from metadata (new recordings)
    if serialized_data is not None:
        scalar_type_name = serialized_data.get("__scalar_type__")
        type_length = serialized_data.get("__type_length__")
        type_shape = serialized_data.get("__type_shape__")

        if scalar_type_name and scalar_type_name in _SCALAR_TYPE_MAP:
            scalar_type, suffix = _SCALAR_TYPE_MAP[scalar_type_name]

            # Handle vector types (vec_t)
            if dtype_str == "vec_t" and type_length:
                inferred = f"vec{type_length}{suffix}"
                if hasattr(wp, inferred):
                    return getattr(wp, inferred)
                # For non-standard lengths, create dynamically
                return wp.types.vector(type_length, scalar_type)

            # Handle matrix types (mat_t)
            if dtype_str == "mat_t" and type_shape and len(type_shape) == 2:
                rows, cols = type_shape
                inferred = f"mat{rows}{cols}{suffix}"
                if hasattr(wp, inferred):
                    return getattr(wp, inferred)
                # For non-standard shapes, create dynamically
                return wp.types.matrix((rows, cols), scalar_type)

            # Handle quaternion types (quat_t)
            if dtype_str == "quat_t":
                inferred = f"quat{suffix}"
                if hasattr(wp, inferred):
                    return getattr(wp, inferred)

    # Fallback: infer from numpy array shape (for old recordings without metadata)
    if dtype_str == "vec_t" and np_arr is not None and np_arr.ndim >= 1:
        vec_len = np_arr.shape[-1] if np_arr.ndim > 1 else np_arr.shape[0]
        suffix = _NUMPY_DTYPE_TO_SUFFIX.get(np_arr.dtype.type, "f")
        inferred = f"vec{vec_len}{suffix}"
        if hasattr(wp, inferred):
            print(f"[Recorder] Info: Inferred dtype '{inferred}' from data shape for generic 'vec_t'")
            return getattr(wp, inferred)
        # For non-standard vector lengths (e.g., vec5), create dynamically
        scalar_type = _NUMPY_TO_WARP_SCALAR.get(np_arr.dtype.type, wp.float32)
        print(f"[Recorder] Info: Creating dynamic vector type vec{vec_len} with {scalar_type.__name__}")
        return wp.types.vector(vec_len, scalar_type)

    if dtype_str == "mat_t" and np_arr is not None and np_arr.ndim >= 2:
        rows = np_arr.shape[-2] if np_arr.ndim > 2 else np_arr.shape[0]
        cols = np_arr.shape[-1]
        suffix = _NUMPY_DTYPE_TO_SUFFIX.get(np_arr.dtype.type, "f")
        inferred = f"mat{rows}{cols}{suffix}"
        if hasattr(wp, inferred):
            print(f"[Recorder] Info: Inferred dtype '{inferred}' from data shape for generic 'mat_t'")
            return getattr(wp, inferred)
        # For non-standard matrix shapes, create dynamically
        scalar_type = _NUMPY_TO_WARP_SCALAR.get(np_arr.dtype.type, wp.float32)
        print(f"[Recorder] Info: Creating dynamic matrix type mat{rows}x{cols} with {scalar_type.__name__}")
        return wp.types.matrix((rows, cols), scalar_type)

    # If dtype ends with 'f' or 'd', try without suffix (e.g., vec3f -> vec3)
    if dtype_str.endswith("f") or dtype_str.endswith("d"):
        base = dtype_str[:-1]
        if hasattr(wp, base):
            return getattr(wp, base)

    raise AttributeError(f"Cannot resolve warp dtype: '{dtype_str}'")


# returns a model and a state history
def depointer_as_key(data: dict, format_type: str = "json", cache: ArrayCache | None = None):
    """
    Deserialize Newton simulation data using callback approach.

    Args:
        data: The serialized data containing model and states.
        format_type: The serialization format ('json' or 'cbor2').

    Returns:
        The deserialized data structure.
    """

    def callback(x, path):
        # Optimization: extract type once to avoid repeated isinstance and dict lookups
        x_type = x.get("__type__") if isinstance(x, dict) else None

        if x_type == "warp.array_ref":
            if cache is None:
                raise ValueError("ArrayCache required to resolve warp.array_ref")
            ref_index = int(x["cache_index"])
            try:
                return cache.try_get_value(ref_index)
            except KeyError:
                return {"__cache_ref__": {"index": ref_index, "kind": "warp.array"}}

        elif x_type == "warp.array":
            try:
                dtype_str = extract_last_type_name(x["__dtype__"])
                np_arr = deserialize_ndarray(x["data"], format_type, cache)
                # Pass the full serialized dict for metadata, and np_arr as fallback for old recordings
                a = _resolve_warp_dtype(dtype_str, serialized_data=x, np_arr=np_arr)
                result = wp.array(np_arr, dtype=a)
                # Register in cache if provided index present (optimization: single dict lookup)
                cache_index = x.get("cache_index")
                if cache is not None and cache_index is not None:
                    key = _warp_key(result)
                    cache.try_register_pointer_and_value_and_index(key, result, int(cache_index))
                return result
            except Exception as e:
                print(f"[Recorder] Warning: Failed to deserialize warp.array at '{path}': {e}")
                return None

        elif x_type == "warp.HashGrid":
            # Return None or create empty HashGrid as appropriate
            return None

        elif x_type == "warp.Mesh":
            # Return None or create empty Mesh as appropriate
            return None

        elif x_type == "newton.geometry.Mesh_ref":
            if cache is None:
                raise ValueError("ArrayCache required to resolve Mesh_ref")
            ref_index = int(x["cache_index"])
            try:
                return cache.try_get_value(ref_index)
            except KeyError:
                return {"__cache_ref__": {"index": ref_index, "kind": "mesh"}}

        elif x_type == "newton.geometry.Mesh":
            try:
                mesh_data = x["data"]
                vertices = deserialize_ndarray(mesh_data["vertices"], format_type, cache)
                indices = deserialize_ndarray(mesh_data["indices"], format_type, cache)
                # Create the mesh without computing inertia since we'll restore the saved values
                mesh = Mesh(
                    vertices=vertices,
                    indices=indices,
                    compute_inertia=False,
                    is_solid=mesh_data["is_solid"],
                    maxhullvert=mesh_data["maxhullvert"],
                )

                # Restore the saved inertia properties
                mesh.has_inertia = mesh_data["has_inertia"]
                mesh.mass = mesh_data["mass"]
                mesh.com = wp.vec3(*mesh_data["com"])
                mesh.I = wp.mat33(deserialize_ndarray(mesh_data["I"], format_type, cache))
                # Optimization: single dict lookup
                cache_index = x.get("cache_index")
                if cache is not None and cache_index is not None:
                    mesh_key = _mesh_key_from_vertices(vertices, fallback_obj=mesh)
                    cache.try_register_pointer_and_value_and_index(mesh_key, mesh, int(cache_index))
                return mesh
            except Exception as e:
                print(f"[Recorder] Warning: Failed to deserialize Mesh at '{path}': {e}")
                return None

        elif x_type == "callable":
            # Return None for callables as they can't be serialized/deserialized
            return None

        return x

    result = deserialize(data, callback, format_type=format_type, cache=cache)

    def _resolve_cache_refs(obj):
        if isinstance(obj, dict):
            # Optimization: single dict lookup instead of checking membership then accessing
            cache_ref = obj.get("__cache_ref__")
            if cache_ref is not None:
                idx = int(cache_ref["index"])
                # Will raise KeyError with clear message if still missing
                return cache.try_get_value(idx) if cache is not None else obj
            # Recurse into dict
            return {k: _resolve_cache_refs(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve_cache_refs(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_resolve_cache_refs(v) for v in obj)
        if isinstance(obj, set):
            return {_resolve_cache_refs(v) for v in obj}
        return obj

    # Resolve any forward references now that all definitive objects have populated the cache
    return _resolve_cache_refs(result)


class ViewerFile(ViewerBase):
    """
    File-based viewer backend for Newton physics simulations.

    This backend records simulation data to JSON or binary files using the same
    ViewerBase API as other viewers. It captures model structure and state data
    during simulation for later replay or analysis.

    Format is determined by file extension:
    - .json: Human-readable JSON format
    - .bin: Binary CBOR2 format (more efficient)
    """

    def __init__(
        self,
        output_path: str,
        auto_save: bool = True,
        save_interval: int = 100,
        max_history_size: int | None = None,
    ):
        """
        Initialize the File viewer backend for Newton physics simulations.

        Args:
            output_path (str): Path to the output file (.json or .bin)
            auto_save (bool): If True, automatically save periodically during recording
            save_interval (int): Number of frames between auto-saves (when auto_save=True)
            max_history_size (int | None): Maximum number of states to keep in memory.
                If None, uses unlimited history. If set, keeps only the last N states.
        """
        super().__init__()

        if not output_path:
            raise ValueError("output_path must be a non-empty path")
        self.output_path = Path(output_path)
        self.auto_save = auto_save
        self.save_interval = save_interval

        # Recording storage
        if max_history_size is None:
            self.history: list[dict] = []
        else:
            self.history: RingBuffer[dict] = RingBuffer(max_history_size)
        self.raw_model: Model | None = None
        self.deserialized_model: dict | None = None

        self._frame_count = 0
        self._model_recorded = False

    @override
    def set_model(self, model, max_worlds: int | None = None):
        """Override set_model to record the model when it's set."""
        super().set_model(model, max_worlds=max_worlds)

        if model is not None and not self._model_recorded:
            self.record_model(model)
            self._model_recorded = True

    @override
    def log_state(self, state):
        """Override log_state to record the state in addition to standard processing."""
        super().log_state(state)

        # Record the state
        self.record(state)
        self._frame_count += 1

        # Auto-save if enabled
        if self.auto_save and self._frame_count % self.save_interval == 0:
            self._save_recording()

    def save_recording(self, file_path: str | None = None, verbose: bool = False):
        """Save the recorded data to file.

        Args:
            file_path: Optional override for the output path. If omitted, uses
                the ``output_path`` from construction.
            verbose: If True, print status output on success/failure.
        """
        effective_path = file_path if file_path is not None else str(self.output_path)
        self._save_recording(effective_path, verbose=verbose)

    def _save_recording(self, file_path: str | None = None, verbose: bool = True):
        """Internal method to save recording."""
        try:
            effective_path = file_path if file_path is not None else str(self.output_path)
            self._save_to_file(effective_path)
            if verbose:
                print(f"Recording saved to {effective_path} ({self._frame_count} frames)")
        except Exception as e:
            if verbose:
                print(f"Error saving recording: {e}")

    def record(self, state: State):
        """Record a snapshot of the provided simulation state."""
        state_data = {}
        for name, value in state.__dict__.items():
            if isinstance(value, wp.array):
                state_data[name] = wp.clone(value)
        self.history.append(state_data)

    def playback(self, state: State, frame_id: int):
        """Restore a state snapshot from history into a State object."""
        if not (0 <= frame_id < len(self.history)):
            print(f"Warning: frame_id {frame_id} is out of bounds. Playback skipped.")
            return

        state_data = self.history[frame_id]
        for name, value_wp in state_data.items():
            if hasattr(state, name):
                setattr(state, name, value_wp)

    def record_model(self, model: Model):
        """Record a reference to the simulation model for later serialization."""
        self.raw_model = model

    def playback_model(self, model: Model):
        """Populate a Model instance from loaded recording data."""
        if not self.deserialized_model:
            print("Warning: No model data to playback.")
            return

        def post_load_init_callback(target_obj, path):
            if isinstance(target_obj, Mesh):
                target_obj.finalize()

        transfer_to_model(self.deserialized_model, model, post_load_init_callback)

    def _save_to_file(self, file_path: str):
        """Save recorded model and history to disk."""
        try:
            format_type = _get_serialization_format(file_path)
        except ValueError:
            if "." not in os.path.basename(file_path):
                file_path = file_path + ".json"
                format_type = "json"
            else:
                raise

        states_to_save = self.history.to_list() if isinstance(self.history, RingBuffer) else self.history
        data_to_save = {"model": self.raw_model, "states": states_to_save}
        array_cache = ArrayCache()
        serialized_data = pointer_as_key(data_to_save, format_type, cache=array_cache)

        if format_type == "json":
            with open(file_path, "w") as f:
                json.dump(serialized_data, f, indent=2)
        elif format_type == "cbor2":
            cbor_data = cbor2.dumps(serialized_data)
            with open(file_path, "wb") as f:
                f.write(cbor_data)

    def _load_from_file(self, file_path: str):
        """Load recording data from disk, replacing current model/history."""
        try:
            format_type = _get_serialization_format(file_path)
        except ValueError:
            if "." not in os.path.basename(file_path):
                json_path = file_path + ".json"
                if os.path.exists(json_path):
                    file_path = json_path
                    format_type = "json"
                else:
                    raise FileNotFoundError(f"File not found: {file_path} (tried .json extension)") from None
            else:
                raise

        if format_type == "json":
            with open(file_path) as f:
                serialized_data = json.load(f)
        elif format_type == "cbor2":
            with open(file_path, "rb") as f:
                file_data = f.read()
            serialized_data = cbor2.loads(file_data)

        array_cache = ArrayCache()
        raw = depointer_as_key(serialized_data, format_type, cache=array_cache)
        self.deserialized_model = raw["model"]

        loaded_states = raw["states"]
        if isinstance(self.history, RingBuffer):
            self.history.from_list(loaded_states)
        else:
            self.history = loaded_states

    # Abstract method implementations (no-ops for file recording)

    @override
    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        texture=None,
        hidden=False,
        backface_culling=True,
    ):
        """File viewer doesn't render meshes, so this is a no-op."""
        pass

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        """File viewer doesn't render instances, so this is a no-op."""
        pass

    @override
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        """File viewer doesn't render lines, so this is a no-op."""
        pass

    @override
    def log_points(self, name, points, radii, colors, hidden=False):
        """File viewer doesn't render points, so this is a no-op."""
        pass

    @override
    def log_array(self, name, array):
        """File viewer doesn't log arrays visually, so this is a no-op."""
        pass

    @override
    def log_scalar(self, name, value):
        """File viewer doesn't log scalars visually, so this is a no-op."""
        pass

    @override
    def end_frame(self):
        """No frame rendering needed for file viewer."""
        pass

    @override
    def close(self):
        """Save final recording and cleanup."""
        if self._frame_count > 0:
            self._save_recording()
        print(f"ViewerFile closed. Total frames recorded: {self._frame_count}")

    def load_recording(self, file_path: str | None = None, verbose: bool = False):
        """Load a previously recorded file for playback.

        After loading, use load_model() and load_state() to restore the model
        and state at a given frame.

        Args:
            file_path: Optional override for the file path. If omitted, uses
                the ``output_path`` from construction.
            verbose: If True, print status output after loading.
        """
        effective_path = file_path if file_path is not None else str(self.output_path)
        self._load_from_file(effective_path)
        self._frame_count = len(self.history)
        if verbose:
            print(f"Loaded recording with {self._frame_count} frames from {effective_path}")

    def get_frame_count(self) -> int:
        """Return the number of frames in the loaded or recorded session."""
        return self._frame_count

    def has_model(self) -> bool:
        """Return True if the loaded recording contains model data (for playback)."""
        return self.deserialized_model is not None

    def load_model(self, model):
        """Restore a Model from the loaded recording.

        Must be called after load_recording(). The given model is populated
        with the recorded model structure (bodies, shapes, etc.).

        Args:
            model: A Newton Model instance to populate.
        """
        self.playback_model(model)

    def load_state(self, state, frame_id: int):
        """Restore State to a specific frame from the loaded recording.

        Must be called after load_recording(). The given state is updated
        with the state snapshot at frame_id.

        Args:
            state: A Newton State instance to populate.
            frame_id: Frame index in [0, get_frame_count()).
        """
        self.playback(state, frame_id)
