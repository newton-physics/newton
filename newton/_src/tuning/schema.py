# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared schema versioning for the tuning contracts.

The tuning contracts are written to disk and read back by other tools, so they
are public API: a run recorded today must stay readable after the schema moves
on. Every contract therefore carries ``schema_version`` and refuses a version it
does not understand, rather than silently misreading fields that changed
meaning.

Loading is strict: :func:`check_fields` rejects any field a contract does not
define. Bump :data:`SCHEMA_VERSION` for every change to the stored fields -- a
field added, removed, renamed, or given a new meaning -- because an older reader
rejects a file with a field it does not know. Keep reading the previous version
by adding it to :data:`SUPPORTED_SCHEMA_VERSIONS` and converting it on load.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = 1

# Versions this build can read.
SUPPORTED_SCHEMA_VERSIONS = (1,)


def check_schema_version(version: Any, what: str) -> None:
    """Raise unless ``version`` is a schema this build understands.

    Args:
        version: The ``schema_version`` read from a serialized contract.
        what: Contract name, for the error message.
    """
    if version is None:
        raise ValueError(f"{what}: no schema_version. It predates versioning and cannot be read safely.")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{what}: schema_version {version} is not supported by this build "
            f"(supported: {list(SUPPORTED_SCHEMA_VERSIONS)}, current: {SCHEMA_VERSION})."
        )


def check_fields(cls: type, d: Mapping[str, Any], what: str) -> None:
    """Raise unless every key of ``d`` is a field of the dataclass ``cls``.

    Args:
        cls: The contract dataclass ``d`` is read into.
        d: The serialized mapping, without keys the caller handles itself.
        what: Contract name, for the error message.
    """
    unknown = sorted(set(d) - {f.name for f in dataclasses.fields(cls)})
    if unknown:
        raise ValueError(f"{what}: unknown field(s) {unknown}.")
