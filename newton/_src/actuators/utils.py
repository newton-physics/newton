# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any


def _load(path: str, *, model: bool) -> tuple[Any | None, dict[str, Any]]:
    """Shared loader for TorchScript and dict checkpoints.

    Args:
        path: File path to the checkpoint.
        model: If ``True``, load and return the network. If ``False``,
            only the metadata is extracted (cheaper).

    Returns:
        ``(network_or_None, metadata)``.
    """
    import torch

    extra_files: dict[str, str] = {"metadata.json": ""}
    try:
        net = torch.jit.load(path, map_location="cpu", _extra_files=extra_files)
        meta = json.loads(extra_files["metadata.json"]) if extra_files["metadata.json"] else {}
        return (net, meta) if model else (None, meta)
    except Exception:
        pass

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        meta = checkpoint.get("metadata", {})
        if model:
            if "model" not in checkpoint:
                raise ValueError(f"Cannot load checkpoint at '{path}'; dict checkpoint has no 'model' key")
            return checkpoint["model"], meta
        return None, meta

    raise ValueError(f"Cannot load checkpoint at '{path}'; expected a TorchScript archive or a dict with a 'model' key")


def load_metadata(path: str) -> dict[str, Any]:
    """Load only the metadata dict from a checkpoint.

    Args:
        path: File path to the checkpoint.
    """
    _, metadata = _load(path, model=False)
    return metadata


def load_checkpoint(path: str) -> tuple[Any, dict[str, Any]]:
    """Load a neural-network checkpoint as ``(model, metadata)``.

    Accepts a TorchScript archive (with optional ``metadata.json`` in
    ``_extra_files``) or a dict checkpoint ``{"model": ..., "metadata": ...}``.

    Args:
        path: File path to the checkpoint.
    """
    network, metadata = _load(path, model=True)
    network.eval()
    return network, metadata
