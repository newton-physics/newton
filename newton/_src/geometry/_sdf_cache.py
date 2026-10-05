# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""On-disk cache for cooked texture SDFs.

The mesh SDF cooking pipeline produces a dictionary of plain numpy arrays
(see :func:`newton._src.geometry.sdf_texture.build_sparse_sdf_from_mesh`)
just before the data is uploaded to ``wp.Texture3D`` instances.  Reading
back from a Warp 3D texture is not supported, so this cache snapshots the
data at the pre-upload boundary; on a hit, the GPU upload path
(:func:`newton._src.geometry.sdf_texture.create_sparse_sdf_textures`) runs
unchanged.

Cache layout
------------

For each cached SDF, a single ``{hash}.sdf.npz`` file is written under
the user-supplied ``cache_dir``.  The basename is a content hash of the
mesh and build parameters.  The file is written via ``os.replace`` from
a per-writer ``{hash}.sdf.npz.{pid}.{token}.tmp.npz`` companion to make
replacement atomic; the unique tmp suffix lets multiple processes cook
the same hash concurrently without trampling each other's in-flight
writes.

Cooked array layout (``.npz`` contents)
---------------------------------------

* ``coarse_sdf`` — ``float32 (bg_size_z, bg_size_y, bg_size_x)``: the
  coarse/background SDF samples.
* ``subgrid_data`` — ``float32 | uint16 | uint8 (tex, tex, tex)``:
  packed narrow-band subgrid texture data (dtype follows
  ``quantization_mode``).
* ``subgrid_start_slots`` — ``uint32 (w, h, d)``: indirection from
  coarse cell to subgrid slot.
* ``subgrid_required`` — ``int32 (w*h*d,)``: 1D occupancy flags for
  non-linear subgrids.
* Plus the scalar metadata ``coarse_dims``, ``subgrid_tex_size``,
  ``num_subgrids``, ``min_extents``, ``max_extents``, ``cell_size``,
  ``subgrid_size``, ``quantization_mode``, ``subgrids_min_sdf_value``,
  and ``subgrids_sdf_value_range`` — each stored as a 0-d or
  shape-``(3,)`` numpy array.
* ``__cache_format_version__`` — 0-d ``int32`` matching
  :data:`CACHE_FORMAT_VERSION`.  This embedded version is the
  authoritative invalidator; a mismatch is treated as a miss and the
  file is overwritten on the next cook.
* ``__kind__`` — 0-d ``str`` (``"newton.texture_sdf"``) marker so
  these files are distinguishable from arbitrary ``.npz``'s sharing
  the same directory.
* ``__newton_version__`` — 0-d ``str``: Newton package version that
  produced the cache.  Diagnostic only.
* ``__created_utc__`` — 0-d ``str``: ISO-8601 UTC timestamp of the
  write.  Diagnostic only.

The ``__kind__`` field is validated at load time. The ``__newton_version__`` and
``__created_utc__`` fields are diagnostic only.

To inspect a cache file from a shell, an ``.npz`` is just a zip of
``.npy`` members:

.. code-block:: bash

    python -m zipfile -l <hash>.sdf.npz
    python -c "import numpy as np; f=np.load('<hash>.sdf.npz'); \\
               print(f.files); print(f['__newton_version__'].item())"

Cache key
---------

The hash includes the bytes that determine the cooked output: mesh
vertices/indices/``is_solid``, ``narrow_band_range``,
``target_voxel_size``, effective ``max_resolution``, ``margin``,
``texture_format``, the resolved sign method (``parity`` or
``winding``), the *sign* of ``winding_threshold``, and ``scale``.
``shape_margin`` is *not* part of the key — it is applied at sample
time and is not baked into the cooked dictionary.

Schema versioning
-----------------

Bump :data:`CACHE_FORMAT_VERSION` whenever the cooked dictionary's
shape, dtypes, or quantization conventions change.  Existing on-disk
caches are then transparently invalidated and recooked.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import secrets
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


CACHE_FORMAT_VERSION: int = 2
"""Version of the on-disk cooked-SDF cache format.

Bump when the dictionary returned by
:func:`newton._src.geometry.sdf_texture.build_sparse_sdf_from_mesh`
changes shape, dtypes, or quantization meaning.  Existing cache files
become invalid and are transparently re-cooked.
"""


_VERSION_KEY = "__cache_format_version__"
_KIND_KEY = "__kind__"
_NEWTON_VERSION_KEY = "__newton_version__"
_CREATED_UTC_KEY = "__created_utc__"
_NPZ_SUFFIX = ".sdf.npz"
_KIND = "newton.texture_sdf"
# Match the sentinels in ``sdf_texture``. Valid slots pack three 10-bit
# coordinates into bits 0-29, leaving the two highest uint32 values reserved.
_SLOT_LINEAR = np.uint32(0xFFFFFFFE)
_SLOT_EMPTY = np.uint32(0xFFFFFFFF)

_QUANTIZATION_DTYPES: dict[int, np.dtype] = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.uint16),
    4: np.dtype(np.float32),
}

# Plain ndarray entries that pass through unmodified.
_NDARRAY_KEYS: tuple[str, ...] = (
    "coarse_sdf",
    "subgrid_data",
    "subgrid_start_slots",
    "subgrid_required",
)

# (key, save_dtype, load_caster). ``save_dtype`` is the on-disk dtype;
# ``load_caster`` reconstructs the in-memory representation expected by
# ``create_sparse_sdf_textures``.
_INT_SCALARS: tuple[tuple[str, np.dtype, type], ...] = (
    ("subgrid_tex_size", np.dtype(np.int32), int),
    ("num_subgrids", np.dtype(np.int32), int),
    ("subgrid_size", np.dtype(np.int32), int),
    ("quantization_mode", np.dtype(np.int32), int),
)
_FLOAT_SCALARS: tuple[tuple[str, np.dtype, type], ...] = (
    ("subgrids_min_sdf_value", np.dtype(np.float32), float),
    ("subgrids_sdf_value_range", np.dtype(np.float32), float),
)
_VEC3_SCALARS: tuple[str, ...] = ("min_extents", "max_extents", "cell_size")


def _digest_array(arr: np.ndarray, dtype: np.dtype) -> str:
    """SHA-256 over ``arr`` after a deterministic dtype/contiguity cast."""

    buf = np.ascontiguousarray(arr, dtype=dtype).tobytes()
    return hashlib.sha256(buf).hexdigest()


def _resolve_newton_version() -> str:
    # The relative target is the package root ``newton``; ``..`` from this
    # module would resolve to ``newton._src``, which has no ``__version__``.
    # Importing ``newton._version`` directly side-steps the package's own
    # ``__init__`` and so avoids any import-cycle risk during bootstrap.
    try:
        from newton._version import __version__  # noqa: PLC0415

        return str(__version__)
    except ImportError:
        return "unknown"


def hash_inputs(
    *,
    vertices: np.ndarray,
    indices: np.ndarray,
    is_solid: bool,
    narrow_band_range: tuple[float, float],
    target_voxel_size: float | None,
    max_resolution: int | None,
    margin: float,
    texture_format: str,
    sign_method_resolved: str,
    winding_threshold: float,
    scale: tuple[float, float, float] | None,
) -> str:
    """Compute the cache key for a texture-SDF cook.

    Args:
        vertices: Mesh vertex array, ``(N, 3)``.
        indices: Mesh triangle indices, ``(M * 3,)`` or ``(M, 3)``.
        is_solid: Whether the mesh is treated as solid.
        narrow_band_range: Signed narrow-band distance range [m].
        target_voxel_size: Target voxel size [m] or ``None``.
        max_resolution: Effective maximum grid dimension [voxel] or ``None``.
        margin: Extra AABB padding [m].
        texture_format: Subgrid storage format (``"float32"``, ``"uint16"``,
            ``"uint8"``).
        sign_method_resolved: Resolved sign strategy (``"parity"`` or
            ``"winding"``).
        winding_threshold: Winding-number threshold value; only its sign
            participates in the hash.
        scale: Pre-baked vertex scale or ``None``.

    Returns:
        A 32-character BLAKE2b digest used as the cache filename
        basename.
    """

    # JSON with ``sort_keys=True`` is used purely as a deterministic
    # canonical-bytes encoder for the structured key inputs; the result
    # is fed straight into BLAKE2b and never touches disk.
    payload = {
        "kind": _KIND,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "mesh": {
            "vertices_sha256": _digest_array(vertices, np.dtype(np.float32)),
            "indices_sha256": _digest_array(indices, np.dtype(np.int32)),
            "num_vertices": int(np.asarray(vertices).reshape(-1, 3).shape[0]),
            "num_triangles": int(np.asarray(indices).size // 3),
            "is_solid": bool(is_solid),
        },
        "build_params": {
            "narrow_band_range": [float(narrow_band_range[0]), float(narrow_band_range[1])],
            "target_voxel_size": (None if target_voxel_size is None else float(target_voxel_size)),
            "max_resolution": (None if max_resolution is None else int(max_resolution)),
            "margin": float(margin),
            "texture_format": str(texture_format),
            "sign_method_resolved": str(sign_method_resolved),
            # Sign only: magnitude is derived from mesh orientation and
            # identical (0.5) in absolute value for the parity path.
            "winding_threshold_sign": (1 if winding_threshold >= 0.0 else -1),
            "scale": (None if scale is None else [float(scale[0]), float(scale[1]), float(scale[2])]),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def cache_path(cache_dir: str | os.PathLike[str], hash_hex: str) -> Path:
    """Return the ``.npz`` path for a given cache key."""

    return Path(cache_dir) / f"{hash_hex}{_NPZ_SUFFIX}"


def _require_array(
    npz: np.lib.npyio.NpzFile,
    key: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Return one cache array after exact dtype and shape validation."""
    array = np.asarray(npz[key])
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(
            f"invalid {key}: expected dtype {dtype} and shape {shape}, got dtype {array.dtype} and shape {array.shape}"
        )
    return array


def _validate_sparse_data(data: Mapping[str, Any]) -> None:
    """Validate relationships that keep cached SDF texture accesses in bounds."""
    coarse_dims = data["coarse_dims"]
    if any(dim <= 0 for dim in coarse_dims):
        raise ValueError(f"invalid coarse_dims: expected positive dimensions, got {coarse_dims}")

    subgrid_tex_size = data["subgrid_tex_size"]
    num_subgrids = data["num_subgrids"]
    subgrid_size = data["subgrid_size"]
    quantization_mode = data["quantization_mode"]
    total_subgrids = coarse_dims[0] * coarse_dims[1] * coarse_dims[2]
    if subgrid_tex_size <= 0:
        raise ValueError(f"invalid subgrid_tex_size: expected a positive value, got {subgrid_tex_size}")
    if not 0 <= num_subgrids <= total_subgrids:
        raise ValueError(f"invalid num_subgrids: expected a value in [0, {total_subgrids}], got {num_subgrids}")
    if subgrid_size <= 0:
        raise ValueError(f"invalid subgrid_size: expected a positive value, got {subgrid_size}")
    if quantization_mode not in _QUANTIZATION_DTYPES:
        raise ValueError(f"invalid quantization_mode: {quantization_mode}")

    min_extents = data["min_extents"]
    max_extents = data["max_extents"]
    cell_size = data["cell_size"]
    if not np.all(np.isfinite(min_extents)) or not np.all(np.isfinite(max_extents)):
        raise ValueError("invalid SDF extents: values must be finite")
    if not np.all(max_extents > min_extents):
        raise ValueError("invalid SDF extents: max_extents must exceed min_extents")
    if not np.all(np.isfinite(cell_size)) or not np.all(cell_size > 0.0):
        raise ValueError("invalid cell_size: values must be finite and positive")

    sdf_min = data["subgrids_min_sdf_value"]
    sdf_range = data["subgrids_sdf_value_range"]
    if not np.isfinite(sdf_min):
        raise ValueError("invalid subgrids_min_sdf_value: expected a finite value")
    if not np.isfinite(sdf_range) or sdf_range <= 0.0:
        raise ValueError("invalid subgrids_sdf_value_range: expected a finite positive value")

    coarse_sdf = data["coarse_sdf"]
    expected_coarse_shape = (coarse_dims[2] + 1, coarse_dims[1] + 1, coarse_dims[0] + 1)
    if coarse_sdf.shape != expected_coarse_shape:
        raise ValueError(f"invalid coarse_sdf shape: expected {expected_coarse_shape}, got {coarse_sdf.shape}")
    if not np.all(np.isfinite(coarse_sdf)):
        raise ValueError("invalid coarse_sdf: values must be finite")

    subgrid_data = data["subgrid_data"]
    expected_subgrid_dtype = np.dtype(np.float32) if num_subgrids == 0 else _QUANTIZATION_DTYPES[quantization_mode]
    expected_subgrid_shape = (subgrid_tex_size, subgrid_tex_size, subgrid_tex_size)
    if subgrid_data.dtype != expected_subgrid_dtype or subgrid_data.shape != expected_subgrid_shape:
        raise ValueError(
            f"invalid subgrid_data: expected dtype {expected_subgrid_dtype} and shape {expected_subgrid_shape}, "
            f"got dtype {subgrid_data.dtype} and shape {subgrid_data.shape}"
        )
    if np.issubdtype(subgrid_data.dtype, np.floating) and not np.all(np.isfinite(subgrid_data)):
        raise ValueError("invalid subgrid_data: values must be finite")

    expected_slots_shape = coarse_dims
    subgrid_start_slots = data["subgrid_start_slots"]
    if subgrid_start_slots.shape != expected_slots_shape:
        raise ValueError(
            f"invalid subgrid_start_slots shape: expected {expected_slots_shape}, got {subgrid_start_slots.shape}"
        )

    subgrid_required = data["subgrid_required"]
    if subgrid_required.shape != (total_subgrids,):
        raise ValueError(f"invalid subgrid_required shape: expected {(total_subgrids,)}, got {subgrid_required.shape}")
    if not np.all((subgrid_required == 0) | (subgrid_required == 1)):
        raise ValueError("invalid subgrid_required: values must be 0 or 1")
    if int(np.count_nonzero(subgrid_required)) != num_subgrids:
        raise ValueError("invalid subgrid_required: active count does not match num_subgrids")

    active_slots = subgrid_start_slots < _SLOT_LINEAR
    required_grid = subgrid_required.reshape((coarse_dims[2], coarse_dims[1], coarse_dims[0])).transpose(2, 1, 0)
    if not np.array_equal(active_slots, required_grid.astype(bool)):
        raise ValueError("invalid subgrid_start_slots: active slots do not match subgrid_required")
    inactive_slots = subgrid_start_slots[~active_slots]
    if not np.all((inactive_slots == _SLOT_EMPTY) | (inactive_slots == _SLOT_LINEAR)):
        raise ValueError("invalid subgrid_start_slots: inactive cells must use a recognized sentinel")

    if num_subgrids == 0:
        if subgrid_tex_size != 1:
            raise ValueError("invalid empty subgrid texture: expected size 1")
        return

    samples_per_dim = subgrid_size + 1
    if subgrid_tex_size % samples_per_dim != 0:
        raise ValueError("invalid subgrid_tex_size: size must be divisible by subgrid_size + 1")
    blocks_per_dim = subgrid_tex_size // samples_per_dim
    if blocks_per_dim > 1024 or blocks_per_dim**3 < num_subgrids:
        raise ValueError("invalid subgrid texture capacity")

    encoded_slots = subgrid_start_slots[active_slots]
    if np.unique(encoded_slots).size != num_subgrids:
        raise ValueError("invalid subgrid_start_slots: active slots must be unique")
    if np.any(encoded_slots >> np.uint32(30)):
        raise ValueError("invalid subgrid_start_slots: encoded slot uses reserved bits")
    slot_x = encoded_slots & np.uint32(0x3FF)
    slot_y = (encoded_slots >> np.uint32(10)) & np.uint32(0x3FF)
    slot_z = (encoded_slots >> np.uint32(20)) & np.uint32(0x3FF)
    if np.any(slot_x >= blocks_per_dim) or np.any(slot_y >= blocks_per_dim) or np.any(slot_z >= blocks_per_dim):
        raise ValueError("invalid subgrid_start_slots: encoded slot exceeds the packed texture bounds")


def save_sparse_data(
    cache_dir: str | os.PathLike[str],
    hash_hex: str,
    sparse_data: Mapping[str, Any],
    *,
    newton_version: str | None = None,
) -> Path:
    """Persist a cooked SDF dict to the cache.

    Args:
        cache_dir: Destination directory.  Created if missing.
        hash_hex: Cache key from :func:`hash_inputs`.
        sparse_data: Dictionary returned by
            :func:`newton._src.geometry.sdf_texture.build_sparse_sdf_from_mesh`.
        newton_version: Newton package version string for provenance.
            Resolved from ``newton.__version__`` when ``None``.

    Returns:
        Path to the ``.npz`` file written.

    Raises:
        OSError: On filesystem errors.  Callers should treat any failure
            as non-fatal and fall back to live cooking.
    """

    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    npz_path = cache_path(cache_dir_path, hash_hex)

    arrays: dict[str, np.ndarray] = {
        _VERSION_KEY: np.asarray(CACHE_FORMAT_VERSION, dtype=np.int32),
    }
    for k in _NDARRAY_KEYS:
        arrays[k] = np.asarray(sparse_data[k])
    arrays["coarse_dims"] = np.asarray(tuple(int(v) for v in sparse_data["coarse_dims"]), dtype=np.int32)
    for k, dtype, _ in _INT_SCALARS:
        arrays[k] = np.asarray(int(sparse_data[k]), dtype=dtype)
    for k, dtype, _ in _FLOAT_SCALARS:
        arrays[k] = np.asarray(float(sparse_data[k]), dtype=dtype)
    for k in _VEC3_SCALARS:
        arrays[k] = np.asarray(sparse_data[k], dtype=np.float64).reshape(3)

    arrays[_KIND_KEY] = np.asarray(_KIND, dtype=np.str_)
    arrays[_NEWTON_VERSION_KEY] = np.asarray(
        newton_version if newton_version is not None else _resolve_newton_version(),
        dtype=np.str_,
    )
    arrays[_CREATED_UTC_KEY] = np.asarray(datetime.now(timezone.utc).isoformat(), dtype=np.str_)

    # ``np.savez`` appends ``.npz`` to its target, so the tmp path must
    # already end in ``.npz`` for the post-save ``os.replace`` to find
    # the right file.
    #
    # The tmp filename embeds the PID and a per-call random token so
    # multiple writers cooking the same hash concurrently (CI matrix
    # runs, parallel test workers, multi-process training) cannot trample
    # each other's in-flight ``*.tmp.npz``.  ``os.replace`` is atomic on
    # both POSIX and Windows for same-volume moves, so the published
    # ``{hash}.sdf.npz`` is never observed in a partial state.
    #
    # A second concurrency wrinkle is Windows-specific: ``os.replace``
    # into a destination that another process is *also* currently
    # replacing can raise ``PermissionError``/``OSError`` even though
    # both writers are publishing identical content for a content-hashed
    # key.  We treat that race as a benign "already published by a peer"
    # and discard our own tmp file rather than propagating the error.
    tmp_npz = npz_path.parent / f"{npz_path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp.npz"
    try:
        np.savez(tmp_npz, **arrays)
        try:
            os.replace(tmp_npz, npz_path)
        except OSError as exc:
            # If another writer just published the same content-hash, the
            # cache invariant is satisfied; drop our duplicate tmp file.
            if npz_path.exists():
                logger.debug(
                    "SDF cache: concurrent publish of %s won by peer (%s); discarding tmp file",
                    npz_path.name,
                    exc,
                )
                with contextlib.suppress(OSError):
                    tmp_npz.unlink()
            else:
                raise
    except BaseException:
        # Clean up our own tmp file on any failure (including
        # ``KeyboardInterrupt``) so the cache directory doesn't
        # accumulate stale ``*.tmp.npz``.
        with contextlib.suppress(OSError):
            tmp_npz.unlink()
        raise
    return npz_path


def try_load_sparse_data(
    cache_dir: str | os.PathLike[str],
    hash_hex: str,
) -> dict[str, Any] | None:
    """Load a cooked SDF dict from the cache, or ``None`` on miss.

    Verifies the embedded ``__cache_format_version__`` in the ``.npz``;
    a mismatch, missing file, or any IO/parse error is logged and
    treated as a miss.

    Args:
        cache_dir: Directory holding the cache files.
        hash_hex: Cache key from :func:`hash_inputs`.

    Returns:
        The reconstructed ``sparse_data`` dict suitable for
        :func:`newton._src.geometry.sdf_texture.create_sparse_sdf_textures`,
        or ``None`` if the entry is missing or invalid.
    """

    npz_path = cache_path(cache_dir, hash_hex)

    if not npz_path.exists():
        return None

    try:
        with npz_path.open("rb") as cache_file, np.load(cache_file, allow_pickle=False) as npz:
            embedded = int(_require_array(npz, _VERSION_KEY, np.dtype(np.int32), ()).item())
            if embedded != CACHE_FORMAT_VERSION:
                logger.info(
                    "SDF cache: embedded version %d != %d, treating as miss (%s)",
                    embedded,
                    CACHE_FORMAT_VERSION,
                    npz_path,
                )
                return None

            kind = np.asarray(npz[_KIND_KEY])
            if kind.shape != () or kind.dtype.kind != "U" or str(kind.item()) != _KIND:
                raise ValueError(f"invalid {_KIND_KEY}: expected {_KIND!r}")

            coarse_dims_array = _require_array(npz, "coarse_dims", np.dtype(np.int32), (3,))
            coarse_dims = tuple(int(v) for v in coarse_dims_array)
            if any(dim <= 0 for dim in coarse_dims):
                raise ValueError(f"invalid coarse_dims: expected positive dimensions, got {coarse_dims}")
            total_subgrids = coarse_dims[0] * coarse_dims[1] * coarse_dims[2]
            data: dict[str, Any] = {
                "coarse_sdf": _require_array(
                    npz,
                    "coarse_sdf",
                    np.dtype(np.float32),
                    (coarse_dims[2] + 1, coarse_dims[1] + 1, coarse_dims[0] + 1),
                ),
                "subgrid_start_slots": _require_array(
                    npz,
                    "subgrid_start_slots",
                    np.dtype(np.uint32),
                    coarse_dims,
                ),
                "subgrid_required": _require_array(
                    npz,
                    "subgrid_required",
                    np.dtype(np.int32),
                    (total_subgrids,),
                ),
                "coarse_dims": coarse_dims,
            }
            for k, dtype, cast in _INT_SCALARS:
                data[k] = cast(_require_array(npz, k, dtype, ()).item())
            for k, dtype, cast in _FLOAT_SCALARS:
                data[k] = cast(_require_array(npz, k, dtype, ()).item())
            for k in _VEC3_SCALARS:
                data[k] = _require_array(npz, k, np.dtype(np.float64), (3,))
            subgrid_tex_size = data["subgrid_tex_size"]
            data["subgrid_data"] = np.asarray(npz["subgrid_data"])
            if data["subgrid_data"].ndim != 3 or data["subgrid_data"].shape != (subgrid_tex_size,) * 3:
                raise ValueError(
                    f"invalid subgrid_data shape: expected {(subgrid_tex_size,) * 3}, got {data['subgrid_data'].shape}"
                )
            _validate_sparse_data(data)
            return data
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        logger.warning("SDF cache: failed to load %s: %s", npz_path, exc)
        return None


def write(
    cache_dir: str | os.PathLike[str],
    hash_hex: str,
    sparse_data: Mapping[str, Any],
) -> None:
    """Best-effort persist; logs and swallows ``OSError``.

    Convenience wrapper used by :meth:`SDF.create_from_mesh` so cache
    failures never abort an otherwise-successful cook.
    """

    try:
        save_sparse_data(cache_dir, hash_hex, sparse_data)
    except OSError as exc:
        logger.warning("SDF cache: failed to write %s: %s", cache_dir, exc)


__all__ = [
    "CACHE_FORMAT_VERSION",
    "cache_path",
    "hash_inputs",
    "save_sparse_data",
    "try_load_sparse_data",
    "write",
]
