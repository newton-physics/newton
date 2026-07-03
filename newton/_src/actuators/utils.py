# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
from typing import Any

import warp as wp


def _require_onnx():
    """Lazy import of the ``onnx`` package with a friendly error message."""
    try:
        import onnx  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only on missing dep
        raise ImportError(
            "Loading neural-controller ONNX checkpoints requires the optional `onnx` package. "
            "Install it with `pip install newton[onnx]`."
        ) from exc
    return onnx


def _require_warp_nn_runtime():
    """Lazy import of the Warp-NN ONNX runtime."""
    try:
        from warp_nn.runtime import OnnxRuntime  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only on missing dep
        raise ImportError(
            "Loading neural-controller ONNX checkpoints requires Warp-NN's ONNX runtime. "
            "Install it with `pip install newton[onnx]`."
        ) from exc
    return OnnxRuntime


def _looks_like_torch_checkpoint(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".pt", ".pth")


def load_metadata(path: str) -> dict[str, Any]:
    """Load only the metadata dict from a checkpoint.

    ONNX metadata is read from the model's ``metadata_props`` field.  A single
    property named ``metadata`` with a JSON-encoded value is preferred;
    otherwise individual ``key/value`` properties are returned verbatim.

    Args:
        path: File path to the checkpoint.

    Returns:
        Metadata mapping (empty dict when no metadata is stored).
    """
    if _looks_like_torch_checkpoint(path):
        return _load_torch_metadata(path)
    onnx = _require_onnx()
    model = onnx.load(path, load_external_data=False)
    return _extract_metadata(model, path)


def load_checkpoint(
    path: str,
    device: str | wp.Device | None = None,
    batch_size: int = 1,
    input_batch_axes: int | dict[str, int] | None = None,
):
    """Load a neural-network checkpoint as ``(model, metadata)``.

    Both ONNX (``.onnx``) and TorchScript / Torch (``.pt`` / ``.pth``)
    checkpoints are accepted. ONNX checkpoints return a Warp-NN runtime;
    Torch checkpoints return the loaded Torch module.

    Args:
        path: File path to the checkpoint.
        device: Warp device string (e.g. ``"cuda:0"``).  ``None`` uses the
            current default device.
        batch_size: Fixed batch dimension used to pre-allocate intermediate
            buffers.
        input_batch_axes: Optional ONNX graph-input batch-axis override passed
            to :class:`warp_nn.runtime.OnnxRuntime`.

    Returns:
        ``(model, metadata)`` where *model* is a Warp-NN runtime for ONNX
        checkpoints or a Torch module for Torch checkpoints.
    """
    if _looks_like_torch_checkpoint(path):
        return _load_torch_checkpoint(path, device=device)

    metadata = load_metadata(path)
    OnnxRuntime = _require_warp_nn_runtime()
    runtime = OnnxRuntime(path, device=device, batch_size=batch_size, input_batch_axes=input_batch_axes)
    return runtime, metadata


def _extract_metadata(model, path: str) -> dict[str, Any]:
    props = {p.key: p.value for p in model.metadata_props}
    if "metadata" in props:
        try:
            metadata = json.loads(props["metadata"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in ONNX metadata property 'metadata' for '{path}'") from exc
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Invalid ONNX metadata property 'metadata' for '{path}': expected a JSON object, "
                f"got {type(metadata).__name__}"
            )
        return metadata
    parsed: dict[str, Any] = {}
    for k, v in props.items():
        try:
            parsed[k] = json.loads(v)
        except json.JSONDecodeError:
            parsed[k] = v
    return parsed


def _parse_metadata_scale(
    metadata: dict[str, Any],
    key: str,
    model_path: str,
    default: float = 1.0,
    fallback_key: str | None = None,
) -> float:
    value_key = key
    value = default
    if key in metadata:
        value = metadata[key]
    elif fallback_key is not None and fallback_key in metadata:
        value_key = fallback_key
        value = metadata[fallback_key]

    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid metadata value for '{value_key}' in '{model_path}': expected a finite non-zero number, got {value!r}"
        ) from exc
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError(
            f"Invalid metadata value for '{value_key}' in '{model_path}': expected a finite non-zero number, got {value!r}"
        )
    return scale


def _runtime_shape(runtime, name: str) -> tuple[int, ...]:
    """Return a runtime tensor shape while isolating Warp-NN private access."""
    shapes = getattr(runtime, "_shapes", None)
    if shapes is None:
        raise AttributeError(
            f"{type(runtime).__name__} does not expose tensor shapes; update Warp-NN to provide shape metadata"
        )
    if name not in shapes:
        raise ValueError(
            f"{type(runtime).__name__} has no shape for tensor '{name}'; available tensors: {sorted(shapes)}"
        )
    return tuple(shapes[name])


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Loading .pt/.pth neural-controller checkpoints requires PyTorch. "
            "Install it with `pip install newton[torch-cu12]` or `pip install newton[torch-cu13]`."
        ) from exc
    return torch


def _load_torch_raw(path: str) -> tuple[Any, dict[str, Any]]:
    """Load a ``.pt`` / ``.pth`` checkpoint."""
    torch = _require_torch()

    extra_files: dict[str, str] = {"metadata.json": ""}
    try:
        net = torch.jit.load(path, map_location="cpu", _extra_files=extra_files)
        meta = json.loads(extra_files["metadata.json"]) if extra_files["metadata.json"] else {}
        return net, meta
    except Exception:
        pass

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        meta = checkpoint.get("metadata", {})
        if "model" not in checkpoint:
            raise ValueError(f"Cannot load checkpoint at '{path}'; dict checkpoint has no 'model' key")
        return checkpoint["model"], meta

    raise ValueError(f"Cannot load checkpoint at '{path}'; expected a TorchScript archive or a dict with a 'model' key")


def _load_torch_metadata(path: str) -> dict[str, Any]:
    _, metadata = _load_torch_raw(path)
    return metadata


def _load_torch_checkpoint(path: str, device: str | wp.Device | None = None):
    """Load a TorchScript / dict checkpoint as a Torch module."""
    model, metadata = _load_torch_raw(path)
    if hasattr(model, "eval"):
        model = model.eval()
    return model, metadata
