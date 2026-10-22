# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Attribute parsing helpers for :meth:`CollisionPipeline.create_from_usd`."""

from __future__ import annotations

import math
import operator
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd


def authored(prim: Usd.Prim, path: str, name: str) -> Any:
    """Return the authored value of attribute ``name`` on ``prim``, or ``None`` if unauthored."""
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return None
    value = attr.Get()
    if value is None:
        raise ValueError(f"{path}: authored attribute {name!r} has no value.")
    return value


def integer(path: str, name: str, value: Any, *, allow_minus_one: bool = False, disallow_zero: bool = False) -> int:
    """Validate ``value`` as an integer, optionally allowing ``-1`` and disallowing ``0``."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{path}: {name} must be an integer, got {value!r}.")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{path}: {name} must be an integer, got {value!r}.") from error
    if result < 0 and not (allow_minus_one and result == -1):
        suffix = " or -1" if allow_minus_one else ""
        raise ValueError(f"{path}: {name} must be non-negative{suffix}, got {result}.")
    if result == 0 and disallow_zero:
        suffix = " or -1" if allow_minus_one else ""
        raise ValueError(f"{path}: {name} must be a positive integer{suffix}, got {result}.")
    return result


def token(path: str, name: str, value: Any, allowed: set[str]) -> str:
    """Validate that ``value`` is one of the ``allowed`` token strings."""
    result = str(value)
    if result not in allowed:
        raise ValueError(f"{path}: {name} must be one of {sorted(allowed)}, got {result!r}.")
    return result


def finite_float(
    path: str, name: str, value: Any, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    """Validate ``value`` as a finite float within optional bounds."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {name} must be a number, got {value!r}.") from error
    out_of_range = (minimum is not None and result < minimum) or (maximum is not None and result > maximum)
    if not math.isfinite(result) or out_of_range:
        bounds = ""
        if minimum is not None and maximum is not None:
            bounds = f" in [{minimum}, {maximum}]"
        elif minimum is not None:
            bounds = f" >= {minimum}"
        elif maximum is not None:
            bounds = f" <= {maximum}"
        raise ValueError(f"{path}: {name} must be a finite number{bounds}, got {value!r}.")
    return result


def optional_finite_float(path: str, name: str, value: Any, *, minimum: float) -> float | None:
    """Parse a float attribute using the ``-inf`` sentinel convention for "unset".

    A value below ``minimum`` warns and falls back to "unset" instead of
    raising, mirroring the soft-limit convention used for
    ``newton:hydroelasticStiffness``.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {name} must be a number, got {value!r}.") from error
    if numeric == float("-inf"):
        return None
    if not math.isfinite(numeric) or numeric < minimum:
        warnings.warn(
            f"{path}: {name}={numeric!r} is invalid (must be >= {minimum}); falling back to default.",
            stacklevel=3,
        )
        return None
    return numeric
