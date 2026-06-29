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
