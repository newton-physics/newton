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
        import onnx
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


def load_checkpoint(
    path: str,
    device: str | None = None,
    batch_size: int = 1,
    input_batch_axes: int | dict[str, int] | None = None,
):
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
        input_batch_axes: Optional ONNX graph-input batch-axis override passed
            to :class:`newton.utils.OnnxRuntime`.

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

    runtime = OnnxRuntime(path, device=device, batch_size=batch_size, input_batch_axes=input_batch_axes)
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


def _load_legacy_lstm_torch_checkpoint(path: str, device: str | None = None):
    """Load a legacy ``.pt`` LSTM checkpoint and wrap it for the ONNX-shaped LSTM controller.

    Supports the pre-2.0 ``ControllerNeuralLSTM`` checkpoint contract: a
    ``torch.nn.Module`` exposing a ``.lstm`` attribute (``torch.nn.LSTM`` with
    ``batch_first=True``, ``input_size=2``, ``bidirectional=False``,
    ``proj_size=0``).  The wrapped adapter speaks the new ``OnnxRuntime``
    dict-in / dict-out protocol so :class:`ControllerNeuralLSTM` does not
    need to special-case it on the hot path.

    Returns:
        ``(adapter, metadata)`` where ``metadata`` carries every required
        ``ControllerNeuralLSTM`` ONNX-metadata key (``input_name``,
        ``hidden_in_name``, ..., ``num_layers``, ``hidden_size``) so the
        controller can route legacy and ONNX checkpoints through a single
        code path.
    """
    warnings.warn(_TORCH_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)
    model, metadata = _load_torch_raw(path)
    if hasattr(model, "eval"):
        model = model.eval()
    if not hasattr(model, "lstm"):
        raise ValueError(
            f"Legacy .pt LSTM checkpoint at '{path}' must expose a 'lstm' attribute (torch.nn.LSTM); "
            "re-export to ONNX (see ControllerNeuralLSTM docstring) or supply a compatible checkpoint."
        )
    lstm = model.lstm
    if not hasattr(lstm, "num_layers"):
        raise ValueError("Legacy .pt LSTM checkpoint: network.lstm must be a torch.nn.LSTM (missing num_layers)")
    if not getattr(lstm, "batch_first", False):
        raise ValueError("Legacy .pt LSTM checkpoint: network.lstm.batch_first must be True")
    if getattr(lstm, "input_size", None) != 2:
        raise ValueError(
            f"Legacy .pt LSTM checkpoint: network.lstm.input_size must be 2 (pos_error, vel_error); "
            f"got {lstm.input_size}"
        )
    if getattr(lstm, "bidirectional", False):
        raise ValueError("Legacy .pt LSTM checkpoint: network.lstm must not be bidirectional")
    if getattr(lstm, "proj_size", 0) != 0:
        raise ValueError(f"Legacy .pt LSTM checkpoint: network.lstm.proj_size must be 0; got {lstm.proj_size}")

    legacy_meta = dict(metadata) if metadata else {}
    legacy_meta.setdefault("input_name", "observation")
    legacy_meta.setdefault("hidden_in_name", "hidden_in")
    legacy_meta.setdefault("cell_in_name", "cell_in")
    legacy_meta.setdefault("output_name", "effort")
    legacy_meta.setdefault("hidden_out_name", "hidden_out")
    legacy_meta.setdefault("cell_out_name", "cell_out")
    legacy_meta["num_layers"] = int(lstm.num_layers)
    legacy_meta["hidden_size"] = int(lstm.hidden_size)
    return _LegacyLstmTorchAdapter(model, legacy_meta, device=device), legacy_meta


class _LegacyLstmTorchAdapter:
    """Legacy ``.pt`` LSTM adapter exposing the ``OnnxRuntime`` interface.

    Bridges the dict-in / dict-out protocol expected by
    :class:`ControllerNeuralLSTM` to the legacy positional torch call
    ``effort, (h, c) = network(net_input, (hidden, cell))``.

    Not graph-capturable: the call crosses the host boundary on each
    ``.numpy()`` round-trip.  Legacy ``.pt`` users already lived with these
    constraints; the adapter preserves that behavior so existing checkpoints
    keep running until the deprecation window expires.
    """

    def __init__(self, model, metadata: dict[str, Any], device: str | None = None):
        torch = _require_torch()
        self._torch = torch
        self._model = model
        self._device = device
        self._input_name: str = metadata["input_name"]
        self._hidden_in_name: str = metadata["hidden_in_name"]
        self._cell_in_name: str = metadata["cell_in_name"]
        self._output_name: str = metadata["output_name"]
        self._hidden_out_name: str = metadata["hidden_out_name"]
        self._cell_out_name: str = metadata["cell_out_name"]
        self._num_layers = int(metadata["num_layers"])
        self._hidden_size = int(metadata["hidden_size"])
        self.input_names: list[str] = [self._input_name, self._hidden_in_name, self._cell_in_name]
        self.output_names: list[str] = [self._output_name, self._hidden_out_name, self._cell_out_name]
        # Move the module onto the requested device once so per-call inference
        # avoids repeated host transfers.
        self._torch_device = self._resolve_torch_device(device)
        self._model = self._model.to(self._torch_device)
        self._shapes: dict[str, tuple[int, ...]] = {}

    def _resolve_torch_device(self, device: str | None):
        torch = self._torch
        if device is None or device == "cpu":
            return torch.device("cpu")
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device)
        # Warp may be on CUDA while the installed torch is a CPU-only build
        # (or CUDA is unavailable for some other reason).  Fall back to CPU
        # rather than crashing inside ``model.to(...)``; the adapter already
        # round-trips through ``.numpy()`` so cross-device usage just adds
        # one extra host-device copy.
        return torch.device("cpu")

    def _to_torch(self, arr):
        torch = self._torch
        if hasattr(arr, "numpy"):
            np_arr = arr.numpy()
        else:
            np_arr = arr
        return torch.as_tensor(np_arr, device=self._torch_device)

    def __call__(self, inputs):
        torch = self._torch
        if self._input_name not in inputs:
            raise KeyError(f"_LegacyLstmTorchAdapter: missing input '{self._input_name}'")
        if self._hidden_in_name not in inputs:
            raise KeyError(f"_LegacyLstmTorchAdapter: missing input '{self._hidden_in_name}'")
        if self._cell_in_name not in inputs:
            raise KeyError(f"_LegacyLstmTorchAdapter: missing input '{self._cell_in_name}'")

        # Newton hands the LSTM input as ``(1, N, 2)`` (seq-major / layout=0)
        # so the runtime/ONNX path can consume it directly.  The legacy torch
        # ``LSTM`` was built with ``batch_first=True`` and expects ``(N, 1, 2)``;
        # transpose the leading two dims to bridge the two contracts.
        x = self._to_torch(inputs[self._input_name])
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.transpose(0, 1).contiguous()
        h = self._to_torch(inputs[self._hidden_in_name])
        c = self._to_torch(inputs[self._cell_in_name])

        with torch.inference_mode():
            effort, (h_new, c_new) = self._model(x, (h, c))
        if isinstance(effort, (tuple, list)):
            effort = effort[0]

        # ``effort`` is ``(N, 1)`` per the legacy contract; pass through as-is.
        effort_np = effort.detach().cpu().numpy()
        h_np = h_new.detach().cpu().numpy()
        c_np = c_new.detach().cpu().numpy()
        effort_wp = wp.array(effort_np, dtype=wp.float32, device=self._device)
        h_wp = wp.array(h_np, dtype=wp.float32, device=self._device)
        c_wp = wp.array(c_np, dtype=wp.float32, device=self._device)

        self._shapes[self._output_name] = tuple(effort_wp.shape)
        self._shapes[self._hidden_out_name] = tuple(h_wp.shape)
        self._shapes[self._cell_out_name] = tuple(c_wp.shape)

        return {
            self._output_name: effort_wp,
            self._hidden_out_name: h_wp,
            self._cell_out_name: c_wp,
        }


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
