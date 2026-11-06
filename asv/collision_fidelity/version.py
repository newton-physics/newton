# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lightweight identities for benchmark inputs, separate from the tested solver."""

import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from .catalog import load_cases


@lru_cache(maxsize=1)
def _source_and_assets():
    """Hash benchmark code and bundled assets without importing physics modules."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    files = (
        sorted(root.glob("*.py"))
        + sorted(
            path
            for path in (root / "assets").rglob("*")
            if path.is_file() and path.suffix not in (".md", ".rst") and not path.name.endswith("LICENSE.txt")
        )
        + [root.parent / "benchmarks/simulation/bench_collision_fidelity.py"]
    )
    for path in files:
        content = path.read_bytes()
        if path.suffix == ".py":
            content = ast.dump(ast.parse(content), include_attributes=False).encode()
        digest.update(str(path.relative_to(root.parent)).encode() + b"\0" + content + b"\0")
    return digest.hexdigest()


def _case_version(case, source_and_assets):
    return hashlib.sha256((source_and_assets + json.dumps(case, sort_keys=True)).encode()).hexdigest()


def semantic_version(case_id):
    """Identify case inputs and shared definitions, excluding the Newton revision.

    ASV compares solver revisions under this benchmark version. The runner also
    records the installed Newton/Warp revisions; those do not invalidate this
    cross-revision measurement series. ASV's execution-local cache still has a
    separate source-line key for each case and is never a prepared-state cache.
    Asset documentation and license text do not change measurement semantics.
    """
    return _case_version(load_cases()[case_id], _source_and_assets())
