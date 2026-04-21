# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any


def load_checkpoint(path: str) -> tuple[Any, dict[str, Any]]:
    """Load a neural-network checkpoint as ``(model, metadata)``.

    Accepts a TorchScript archive (with optional ``metadata.json`` in
    ``_extra_files``) or a dict checkpoint ``{"model": ..., "metadata": ...}``.

    Args:
        path: File path to the checkpoint.
    """
    import torch

    extra_files: dict[str, str] = {"metadata.json": ""}
    try:
        model = torch.jit.load(path, map_location="cpu", _extra_files=extra_files)
        model.eval()
        metadata = json.loads(extra_files["metadata.json"]) if extra_files["metadata.json"] else {}
        return model, metadata
    except Exception:
        pass

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        metadata = checkpoint.get("metadata", {})
        return checkpoint["model"].eval(), metadata

    raise ValueError(f"Cannot load checkpoint at '{path}'; expected a TorchScript archive or a dict with a 'model' key")
