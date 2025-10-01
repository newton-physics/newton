# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from typing import Any

# NOTE: This file is *imported by Sphinx* when building the docs.
# It must therefore avoid heavy third-party imports that might not be
# available in the documentation environment.

# ---------------------------------------------------------------------------

# Skip handler implementation


def _should_skip_member(
    app: Any,  # Sphinx application (unused)
    what: str,
    name: str,
    obj: Any,
    skip: bool,
    options: Any,  # autodoc options (unused)
) -> bool | None:
    """Determine whether *obj* should be skipped by autodoc.

    We apply two simple rules that make API pages cleaner:
    1.   Private helpers (names that start with an underscore but are not
         special dunder methods) are hidden unless they are explicitly
         marked as public via a ``:meta public:`` field.
    2.   Public members that have **no** docstring are hidden.  This keeps the
         generated documentation focused on the public, documented API.

    Returning ``True`` tells Sphinx to skip the member, ``False`` to include
    it, and ``None`` to fall back to Sphinx's default behaviour.
    """

    # Respect decisions made by other handlers first.
    if skip:
        return True

    # Keep dunder methods that are explicitly requested elsewhere.
    if name.startswith("__") and name.endswith("__"):
        return None  # keep default behaviour

    # Skip private helpers (single underscore) that are not explicitly public.
    if name.startswith("_"):
        # Let users override via :meta public:
        doc = getattr(obj, "__doc__", "") or ""
        if ":meta public:" not in doc:
            return True
        return None

    # If the member is public but undocumented, decide based on its nature.
    doc = getattr(obj, "__doc__", None)

    if not doc:
        # Keep an undocumented callable **only** if it overrides a documented
        # attribute from a base-class.  This covers cases like ``step`` in
        # solver subclasses while still hiding brand-new helpers that have no
        # documentation.

        is_callable = callable(obj) or isinstance(
            obj,
            property | staticmethod | classmethod,
        )

        if is_callable and what == "class":
            # Try to determine the parent class from the qualified name.
            qualname = getattr(obj, "__qualname__", "")
            parts = qualname.split(".")
            if len(parts) >= 2:
                cls_name = parts[-2]
                module = sys.modules.get(obj.__module__)
                parent_cls = getattr(module, cls_name, None)
                if isinstance(parent_cls, type):
                    for base in parent_cls.__mro__[1:]:
                        if hasattr(base, name):
                            return None  # overrides something -> keep it

        # Otherwise hide the undocumented member to keep the page concise.
        return True

    # Default: do not override Sphinx's decision.
    return None


def setup(app):  # type: ignore[override]
    """Hook into the Sphinx build."""

    app.connect("autodoc-skip-member", _should_skip_member)
    # Tell Sphinx our extension is parallel-safe.
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
