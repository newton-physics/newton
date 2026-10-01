# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

_P = ParamSpec("_P")
_R = TypeVar("_R")


class RemovedAttribute:
    """Data descriptor that turns a removed attribute into a hard error.

    Deleting an attribute outright is silent on assignment: ``obj.old_name =
    value`` simply creates an unused instance attribute, so code written
    against the old API keeps running while its writes are ignored. Shadowing
    the name with a data descriptor makes reads, writes, and deletes raise
    :class:`AttributeError` naming the replacement instead.

    Reads raise :class:`AttributeError`, so ``hasattr()`` reports ``False`` and
    ``getattr(obj, name, default)`` still falls back to *default*.
    """

    __slots__ = ("_message", "_removed_in", "_replacement")

    def __init__(self, replacement: str, *, removed_in: str) -> None:
        """Initialize the descriptor.

        Args:
            replacement: Name of the attribute to use instead, unqualified.
            removed_in: Newton release that removed the attribute.
        """
        self._replacement = replacement
        self._removed_in = removed_in
        self._message = f"This attribute was removed in Newton {removed_in}; use {replacement} instead."

    def __set_name__(self, owner: type, name: str) -> None:
        self._message = (
            f"{owner.__name__}.{name} was removed in Newton {self._removed_in}; "
            f"use {owner.__name__}.{self._replacement} instead."
        )

    def __get__(self, instance: object, owner: type | None = None) -> Any:
        raise AttributeError(self._message)

    def __set__(self, instance: object, value: Any) -> None:
        raise AttributeError(self._message)

    def __delete__(self, instance: object) -> None:
        raise AttributeError(self._message)


def deprecate_nonkeyword_arguments(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Warn when keyword-only parameters are supplied positionally."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    positional_count = 0
    keyword_only_names: list[str] = []

    for param in params:
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            keyword_only_names.append(param.name)
        elif not keyword_only_names and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(f"{func.__qualname__}() cannot use deprecate_nonkeyword_arguments with *args")

    if not keyword_only_names:
        return func

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> _R:
        extra_count = len(args) - positional_count
        if extra_count <= 0:
            return func(*args, **kwargs)

        if extra_count > len(keyword_only_names):
            try:
                sig.bind(*args, **kwargs)
            except TypeError as exc:
                raise TypeError(str(exc)) from None
            return func(*args, **kwargs)

        positional_kwargs = dict(zip(keyword_only_names, args[positional_count:], strict=False))
        for name in positional_kwargs:
            if name in kwargs:
                raise TypeError(f"{func.__qualname__}() got multiple values for argument '{name}'")

        kwargs.update(positional_kwargs)
        parameter_list = ", ".join(f"'{name}'" for name in positional_kwargs)
        warnings.warn(
            f"Passing {parameter_list} positionally to {func.__qualname__}() is deprecated. "
            "Pass these arguments as keyword arguments instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args[:positional_count], **kwargs)

    return cast(Callable[_P, _R], wrapper)
