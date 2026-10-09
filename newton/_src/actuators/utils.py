# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import zipfile
from typing import Any

import warp as wp

from ..utils.onnx import load_onnx_runtime

_METADATA_FILE = "metadata.json"


def _require_onnx():
    """Lazy import of the ``onnx`` package with a friendly error message."""
    try:
        import onnx  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only on missing dep
        raise ImportError(
            "Loading neural-drive ONNX checkpoints requires the optional `onnx` package. "
            "Install it with `pip install newton[onnx]`."
        ) from exc
    return onnx


def _looks_like_torch_checkpoint(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".pt", ".pth", ".pt2")


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
    input_batch_axes: int | dict[str, int] | None = None,
    requires_grad: bool = False,
):
    """Load a neural-network checkpoint as ``(model, metadata)``.

    Both ONNX (``.onnx``) and Torch pt2 (``.pt2`` / ``.pt`` / ``.pth`` saved
    with ``torch.export.save``) checkpoints are accepted. ONNX checkpoints
    return a Warp-NN runtime; Torch checkpoints return the loaded Torch module.

    Args:
        path: File path to the checkpoint.
        device: Warp device string (e.g. ``"cuda:0"``).  ``None`` uses the
            current default device.
        input_batch_axes: ONNX graph-input dimensions to make dynamic for
            the actuator batch.
        requires_grad: Whether the runtime allocates gradient storage for its
            own tensors. Required to differentiate the network, since the
            runtime owns intermediate buffers that cannot be given gradients
            after construction. Ignored for Torch checkpoints.

    Returns:
        ``(model, metadata)`` where *model* is a Warp-NN runtime for ONNX
        checkpoints or a Torch module for Torch checkpoints.
    """
    if _looks_like_torch_checkpoint(path):
        return _load_torch_raw(path)

    metadata = load_metadata(path)
    runtime = load_onnx_runtime(
        path,
        device=device,
        input_batch_axes=input_batch_axes,
        requires_grad=requires_grad,
    )
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


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Loading .pt/.pth neural-drive checkpoints requires PyTorch. "
            "Install it with `pip install newton[torch-cu12]` or `pip install newton[torch-cu13]`."
        ) from exc
    return torch


def _parse_metadata_json(text: str | bytes, path: str) -> dict[str, Any]:
    """Parse a checkpoint's ``metadata.json`` payload, requiring a JSON object."""
    if not text:
        return {}
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in checkpoint metadata for '{path}'") from exc
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Invalid checkpoint metadata for '{path}': expected a JSON object, got {type(metadata).__name__}"
        )
    return metadata


def _pt2_archive_entries(path: str) -> dict[str, str]:
    """Map a pt2 archive's zip entry names, minus PyTorch's archive-name prefix, to full names.

    Raises:
        ValueError: If *path* is not a pt2 archive saved with ``torch.export.save``.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            entries = {name.split("/", 1)[-1]: name for name in zf.namelist()}
    except zipfile.BadZipFile:
        entries = {}
    # Torch <= 2.7 wrote export archives without the pt2 "archive_format" marker.
    if "archive_format" not in entries and "serialized_exported_program.json" not in entries:
        raise ValueError(
            f"Cannot load checkpoint at '{path}'; expected a pt2 archive saved with torch.export.save(). "
            f"TorchScript (torch.jit.save) and dict (torch.save) checkpoints are no longer supported; "
            f"re-export the network with torch.export.save()."
        )
    return entries


def _load_torch_raw(path: str) -> tuple[Any, dict[str, Any]]:
    """Load a pt2 checkpoint as ``(module, metadata)``.

    The module runs with the train/eval mode that was captured at export time.
    """
    torch = _require_torch()
    _pt2_archive_entries(path)
    extra_files: dict[str, str] = {_METADATA_FILE: ""}
    exported = torch.export.load(path, extra_files=extra_files)
    return exported.module(), _parse_metadata_json(extra_files[_METADATA_FILE], path)


def _load_torch_metadata(path: str) -> dict[str, Any]:
    """Read pt2 checkpoint metadata without deserializing the network.

    pt2 archives store extra files as plain zip entries under
    ``<archive_name>/extra/`` (``extra_files/`` for Torch <= 2.7 export
    archives), so metadata can be read directly.
    """
    entries = _pt2_archive_entries(path)
    metadata_entry = entries.get(f"extra/{_METADATA_FILE}") or entries.get(_METADATA_FILE)
    if metadata_entry is None:
        return {}
    with zipfile.ZipFile(path) as zf:
        return _parse_metadata_json(zf.read(metadata_entry), path)
