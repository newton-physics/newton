# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Backend implementations for Cosserat rod simulation.

This module provides the backend registry and factory function for creating
backend instances.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import CosseratRodModel
    from ..solver import BackendType

from .base import BackendBase


def create_backend(
    backend_type: "BackendType",
    model: "CosseratRodModel",
    device: str = "cuda:0",
    use_cuda_graph: bool = False,
    dll_path: str = "unity_ref",
) -> BackendBase:
    """Create a backend instance.

    Args:
        backend_type: Type of backend to create.
        model: The rod model to operate on.
        device: Warp device string.
        use_cuda_graph: Enable CUDA graph (WARP_GPU only).
        dll_path: Path to DLL directory (REFERENCE only).

    Returns:
        Backend instance.

    Raises:
        ValueError: If backend_type is unknown.
        ImportError: If required dependencies are not available.
    """
    from ..solver import BackendType

    if backend_type == BackendType.REFERENCE:
        from .reference import ReferenceBackend

        return ReferenceBackend(model, dll_path=dll_path)

    elif backend_type == BackendType.NUMPY:
        from .numpy_backend import NumPyBackend

        return NumPyBackend(model)

    elif backend_type == BackendType.WARP_CPU:
        from .warp_cpu import WarpCPUBackend

        return WarpCPUBackend(model)

    elif backend_type == BackendType.WARP_GPU:
        from .warp_gpu import WarpGPUBackend

        return WarpGPUBackend(model, device=device, use_cuda_graph=use_cuda_graph)

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


__all__ = ["BackendBase", "create_backend"]
