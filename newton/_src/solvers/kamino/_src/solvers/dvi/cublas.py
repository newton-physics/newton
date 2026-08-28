# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Narrow graph-capturable cuBLAS support for batched DVI response solves."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
from pathlib import Path

import warp as wp

_CUBLAS_SIDE_LEFT = 0
_CUBLAS_FILL_UPPER = 1
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1
_CUBLAS_DIAG_NON_UNIT = 0


def _windows_cublas_candidates() -> tuple[list[Path], list[str]]:
    """Return DLL directories and cuBLAS candidates in preferred order."""
    search_dirs: list[Path] = []
    for name in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(name)
        if root:
            search_dirs.append(Path(root) / "bin")

    # Python 3.8+ no longer uses PATH to resolve dependent DLLs for ctypes.
    # Inspect it explicitly so a system CUDA installation still works when
    # CUDA_PATH is unavailable.
    search_dirs.extend(Path(value) for value in os.environ.get("PATH", "").split(os.pathsep) if value)

    unique_dirs: list[Path] = []
    candidates: list[str] = []
    seen_dirs: set[Path] = set()
    seen_candidates: set[str] = set()
    for directory in search_dirs:
        if directory in seen_dirs or not directory.is_dir():
            continue
        seen_dirs.add(directory)
        libraries = sorted(directory.glob("cublas64_*.dll"), reverse=True)
        if not libraries:
            continue
        unique_dirs.append(directory)
        for library in libraries:
            value = str(library)
            if value not in seen_candidates:
                candidates.append(value)
                seen_candidates.add(value)

    discovered = ctypes.util.find_library("cublas")
    if discovered and discovered not in seen_candidates:
        candidates.append(discovered)
        seen_candidates.add(discovered)
    for major in range(20, 9, -1):
        name = f"cublas64_{major}.dll"
        if name not in seen_candidates:
            candidates.append(name)
            seen_candidates.add(name)
    return unique_dirs, candidates


def _load_cublas_library() -> tuple[ctypes.CDLL, list[object]]:
    if os.name != "nt":
        path = ctypes.util.find_library("cublas")
        if path is None:
            raise OSError("cuBLAS library was not found")
        return ctypes.CDLL(path), []

    directories, candidates = _windows_cublas_candidates()
    directory_handles: list[object] = []
    for directory in directories:
        try:
            directory_handles.append(os.add_dll_directory(str(directory)))
        except (FileNotFoundError, OSError):
            pass

    loader = ctypes.WinDLL
    errors: list[OSError] = []
    for candidate in candidates:
        try:
            # cuBLAS uses CUBLASWINAPI (__stdcall) on Windows.
            return loader(candidate), directory_handles
        except OSError as error:
            errors.append(error)
    for handle in directory_handles:
        close = getattr(handle, "close", None)
        if close is not None:
            close()
    detail = f": {errors[-1]}" if errors else ""
    raise OSError(f"cuBLAS library was not found{detail}")


class _ThreadHandles:
    """Own the cuBLAS handles created by one host thread."""

    def __init__(self, library: ctypes.CDLL):
        self.library = library
        self.handles: dict[int, tuple[wp.context.Device, ctypes.c_void_p]] = {}

    def close(self) -> None:
        handles, self.handles = self.handles, {}
        for device, handle in handles.values():
            try:
                with wp.ScopedDevice(device):
                    self.library.cublasDestroy_v2(handle)
            except Exception:
                # Thread teardown can race interpreter or CUDA shutdown.
                pass

    def __del__(self):
        self.close()


class _Cublas:
    def __init__(self):
        self.lib, self._directory_handles = _load_cublas_library()
        self.lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cublasCreate_v2.restype = ctypes.c_int
        self.lib.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
        self.lib.cublasDestroy_v2.restype = ctypes.c_int
        self.lib.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cublasSetStream_v2.restype = ctypes.c_int
        self.lib.cublasStrsmBatched.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.cublasStrsmBatched.restype = ctypes.c_int
        self.local = threading.local()

    def handle(self, device: wp.context.Device) -> ctypes.c_void_p:
        owner = getattr(self.local, "owner", None)
        if owner is None:
            owner = _ThreadHandles(self.lib)
            self.local.owner = owner
        key = int(device.ordinal)
        entry = owner.handles.get(key)
        if entry is None:
            handle = ctypes.c_void_p()
            with wp.ScopedDevice(device):
                status = self.lib.cublasCreate_v2(ctypes.byref(handle))
            if status != 0:
                raise RuntimeError(f"cublasCreate_v2 failed with status {status}")
            entry = (device, handle)
            owner.handles[key] = entry
        return entry[1]


_cache: list[_Cublas | bool] = []
_lock = threading.Lock()
_alpha = ctypes.c_float(1.0)


def _get_cublas() -> _Cublas | None:
    if not _cache:
        with _lock:
            if not _cache:
                try:
                    library: _Cublas | bool = _Cublas()
                except (AttributeError, OSError):
                    library = False
                _cache.append(library)
    library = _cache[0]
    return library if isinstance(library, _Cublas) else None


def is_batched_trsm_available(device: wp.context.Device) -> bool:
    """Return whether graph-capturable batched triangular solves are available."""
    if not device.is_cuda:
        return False
    library = _get_cublas()
    if library is None:
        return False
    try:
        library.handle(device)
    except RuntimeError:
        return False
    return True


def solve_llt_batched(
    factor_ptrs: wp.array[wp.uint64],
    rhs_ptrs: wp.array[wp.uint64],
    rows: int,
    rhs_count: int,
    batch_count: int,
) -> None:
    """Enqueue ``L L^T X = B`` for row-major ``L`` and RHS-major ``B`` batches."""
    library = _get_cublas()
    if library is None:
        raise RuntimeError("cuBLAS is not available")
    device = rhs_ptrs.device
    handle = library.handle(device)
    with wp.ScopedDevice(device):
        stream = device.stream
        status = library.lib.cublasSetStream_v2(handle, ctypes.c_void_p(stream.cuda_stream))
        if status != 0:
            raise RuntimeError(f"cublasSetStream_v2 failed with status {status}")
        for operation in (_CUBLAS_OP_T, _CUBLAS_OP_N):
            status = library.lib.cublasStrsmBatched(
                handle,
                _CUBLAS_SIDE_LEFT,
                _CUBLAS_FILL_UPPER,
                operation,
                _CUBLAS_DIAG_NON_UNIT,
                rows,
                rhs_count,
                ctypes.byref(_alpha),
                ctypes.c_void_p(factor_ptrs.ptr),
                rows,
                ctypes.c_void_p(rhs_ptrs.ptr),
                rows,
                batch_count,
            )
            if status != 0:
                raise RuntimeError(f"cublasStrsmBatched failed with status {status}")
