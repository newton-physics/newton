# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import warp as wp


def _require_onnx():
    """Lazy import of the ``onnx`` package with a friendly error message."""
    try:
        import onnx  # noqa: PLC0415 - lazy import keeps `onnx` an optional extra
    except ImportError as exc:  # pragma: no cover - exercised only on missing dep
        raise ImportError(
            "Loading neural-controller checkpoints requires the optional `onnx` package. "
            "Install it with `pip install onnx>=1.16.0` or `pip install newton[onnx]`."
        ) from exc
    return onnx


def _looks_like_torch_checkpoint(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".pt", ".pth")


def load_metadata(path: str) -> dict[str, Any]:
    """Load only the metadata dict from an ONNX checkpoint.

    Metadata is read from the model's ``metadata_props`` field.  A single
    property named ``metadata`` with a JSON-encoded value is preferred;
    otherwise individual ``key/value`` properties are returned verbatim.

    Args:
        path: File path to the ``.onnx`` model.

    Returns:
        Metadata mapping (empty dict when no metadata is stored).
    """
    if _looks_like_torch_checkpoint(path):
        return _load_torch_metadata(path)
    onnx = _require_onnx()
    model = onnx.load(path, load_external_data=False)
    return _extract_metadata(model)


def load_checkpoint(path: str, device: str | None = None, batch_size: int = 1):
    """Load a neural-network checkpoint as ``(runtime, metadata)``.

    Both ONNX (``.onnx``) and TorchScript (``.pt`` / ``.pth``) checkpoints
    are accepted.  TorchScript loading is **deprecated**: it emits a
    :class:`DeprecationWarning` and will be removed in a future release.
    Convert legacy ``.pt`` policies to ``.onnx`` once with
    ``torch.onnx.export(...)``.

    Args:
        path: File path to the checkpoint.  ``.onnx`` is preferred;
            ``.pt`` / ``.pth`` is accepted for backward compatibility.
        device: Warp device string (e.g. ``"cuda:0"``).  ``None`` uses the
            current default device.
        batch_size: Fixed batch dimension used to pre-allocate intermediate
            buffers.

    Returns:
        ``(runtime, metadata)`` where *runtime* is callable as
        ``runtime({input_name: warp_array})`` and exposes ``input_names`` /
        ``output_names`` lists, and *metadata* is a configuration dict.
    """
    if _looks_like_torch_checkpoint(path):
        return _load_torch_checkpoint(path, device=device)
    metadata = load_metadata(path)
    # Deferred import: keeps the heavy onnx_runtime module (Warp kernels, etc.)
    # off the import path of every newton.actuators consumer.
    from ..utils.onnx_runtime import OnnxRuntime  # noqa: PLC0415

    runtime = OnnxRuntime(path, device=device, batch_size=batch_size)
    return runtime, metadata


def _extract_metadata(model) -> dict[str, Any]:
    props = {p.key: p.value for p in model.metadata_props}
    if "metadata" in props:
        try:
            return json.loads(props["metadata"])
        except json.JSONDecodeError:
            pass
    parsed: dict[str, Any] = {}
    for k, v in props.items():
        try:
            parsed[k] = json.loads(v)
        except json.JSONDecodeError:
            parsed[k] = v
    return parsed


# ---------------------------------------------------------------------------
# Deprecated TorchScript / dict-checkpoint loader
# ---------------------------------------------------------------------------
#
# Kept only so existing user code that points the neural controllers at a
# ``.pt`` / ``.pth`` file keeps working for one release.  Emits a
# ``DeprecationWarning`` and will be removed in a future release.
#
# The legacy loader supported two formats (mirrors the pre-ONNX behavior):
#   1. TorchScript archive (``torch.jit.save``) with metadata in
#      ``_extra_files={"metadata.json": ...}``.
#   2. Dict checkpoint (``torch.save({"model": net, "metadata": {...}})``).
# Both are preserved; the resulting Torch module is wrapped in an
# OnnxRuntime-shaped adapter so the neural controllers (which now speak the
# ONNX-runtime interface only) keep working with the legacy MLP policies
# that exposed a single ``forward(obs) -> effort`` callable.


_TORCH_DEPRECATION_MSG = (
    "Loading neural-controller checkpoints from TorchScript .pt/.pth files is deprecated "
    "and will be removed in a future release. Convert your checkpoint to ONNX once "
    "(see torch.onnx.export) and load the .onnx file instead."
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Loading legacy .pt/.pth checkpoints requires PyTorch. "
            "Install it (e.g. `pip install newton[torch-cu12]`) or convert the "
            "checkpoint to ONNX (`torch.onnx.export`) and load the .onnx file."
        ) from exc
    return torch


def _load_torch_raw(path: str) -> tuple[Any, dict[str, Any]]:
    """Load a legacy ``.pt`` / ``.pth`` checkpoint, returning ``(model, metadata)``.

    Mirrors the original loader: try ``torch.jit.load`` first (with
    ``metadata.json`` extra-file), fall back to ``torch.load`` and the
    ``{"model": ..., "metadata": ...}`` dict-checkpoint convention.
    """
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
    warnings.warn(_TORCH_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)
    _, metadata = _load_torch_raw(path)
    return metadata


def _load_torch_checkpoint(path: str, device: str | None = None):
    """Wrap a TorchScript / dict checkpoint in an OnnxRuntime-compatible adapter."""
    warnings.warn(_TORCH_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)
    model, metadata = _load_torch_raw(path)
    if hasattr(model, "eval"):
        model = model.eval()
    return _TorchModuleAdapter(model, device=device), metadata


class _TorchModuleAdapter:
    """Adapter that exposes a Torch module via the ``OnnxRuntime`` interface.

    Provides ``input_names`` / ``output_names`` (single-input / single-output
    by convention; that's what every legacy MLP policy used) and a callable
    ``__call__(inputs: dict[str, wp.array]) -> dict[str, wp.array]``.

    Not graph-capturable: torch tensors are not Warp-managed and the call
    crosses the host boundary on .numpy() copy-out.  Legacy ``.pt`` users
    already lived with these constraints; the adapter preserves that behavior
    so existing code keeps running until the deprecation window expires.
    """

    def __init__(self, model, device: str | None = None):
        torch = _require_torch()
        self._torch = torch
        self._model = model
        self._device = device
        self.input_names: list[str] = ["observation"]
        self.output_names: list[str] = ["action"]
        self._shapes: dict[str, tuple[int, ...]] = {}

    def __call__(self, inputs):
        torch = self._torch
        # The adapter only models the single-input/single-output MLP contract.
        # Multi-input calls (e.g. ControllerNeuralLSTM, which passes the input
        # tensor *and* hidden/cell states) would be silently truncated to just
        # the observation, turning a stateful LSTM into a stateless MLP and
        # returning wrong results.  Fail loudly instead.
        if len(inputs) != 1:
            raise NotImplementedError(
                "_TorchModuleAdapter only supports single-input MLP-shaped policies "
                f"(got {len(inputs)} inputs: {sorted(inputs)}). Stateful controllers "
                "such as ControllerNeuralLSTM no longer accept .pt/.pth checkpoints; "
                "re-export the model to ONNX with the metadata properties listed in the "
                "ControllerNeuralLSTM class docstring (input_name, hidden_in_name, "
                "cell_in_name, output_name, hidden_out_name, cell_out_name, num_layers, "
                "hidden_size) and load the resulting .onnx file."
            )
        in_name = self.input_names[0]
        if in_name not in inputs:
            raise KeyError(f"_TorchModuleAdapter: missing input '{in_name}'")
        arr = inputs[in_name]
        x_np = arr.numpy() if hasattr(arr, "numpy") else arr
        x = torch.as_tensor(x_np)
        with torch.no_grad():
            y = self._model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        y_np = y.detach().cpu().numpy()
        out_name = self.output_names[0]
        out = wp.array(y_np, dtype=wp.float32, device=self._device)
        self._shapes[out_name] = tuple(out.shape)
        return {out_name: out}
