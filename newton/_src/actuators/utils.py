# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any


def _load(path: str) -> tuple[Any, dict[str, Any]]:
    """Shared loader for TorchScript and dict checkpoints.

    Args:
        path: File path to the checkpoint.

    Returns:
        ``(network, metadata)``.
    """
    import torch

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


def load_metadata(path: str) -> dict[str, Any]:
    """Load only the metadata dict from a checkpoint.

    Args:
        path: File path to the checkpoint.

    Returns:
        Metadata mapping (empty dict when no metadata is stored).
    """
    _, metadata = _load(path)
    return metadata


def load_checkpoint(path: str) -> tuple[Any, dict[str, Any]]:
    """Load a neural-network checkpoint as ``(model, metadata)``.

    Two PyTorch file formats are accepted:

    1. **TorchScript archive** — saved via :func:`torch.jit.save`.
       Metadata is read from an internal ``metadata.json`` entry
       (provided via ``_extra_files`` when saving).

    2. **Dict checkpoint** — a Python dict saved via :func:`torch.save`
       with a ``"model"`` key (the network) and an optional
       ``"metadata"`` key (a dict of configuration attributes).

    Args:
        path: File path to the checkpoint.

    Returns:
        ``(network, metadata)`` where *network* is the loaded model
        (in eval mode) and *metadata* is a configuration dict.
    """
    network, metadata = _load(path)
    network.eval()
    return network, metadata
