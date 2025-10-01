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

"""Sphinx extension to document Warp `@wp.func` functions.

This extension registers a custom *autodoc* documenter that recognises
`warp.types.Function` objects (created by the :pyfunc:`warp.func` decorator),
unwraps them to their original Python function (stored in the ``.func``
attribute) and then delegates all further processing to the standard
:class:`sphinx.ext.autodoc.FunctionDocumenter`.

With this in place, *autosummary* and *autodoc* treat Warp kernels exactly like
regular Python functions: the original signature and docstring are used and
`__all__` filtering works as expected.
"""

from __future__ import annotations

import inspect
from typing import Any

from sphinx.ext.autodoc import FunctionDocumenter

# NOTE: We do **not** import warp at module import time. Doing so would require
# CUDA and other heavy deps during the Sphinx build. Instead, detection of a
# Warp function is performed purely via *duck typing* (checking attributes and
# class name) so the extension is safe even when Warp cannot be imported.


class WarpFunctionDocumenter(FunctionDocumenter):
    """Autodoc documenter that unwraps :pyclass:`warp.types.Function`."""

    objtype = "warpfunc"
    directivetype = "function"
    # Ensure we run *before* the builtin FunctionDocumenter (higher priority)
    priority = FunctionDocumenter.priority + 10

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    @staticmethod
    def _looks_like_warp_function(obj: Any) -> bool:
        """Return *True* if *obj* appears to be a `warp.types.Function`."""
        cls = obj.__class__
        return getattr(cls, "__name__", "") == "Function" and hasattr(obj, "func")

    @classmethod
    def can_document_member(
        cls,
        member: Any,
        member_name: str,
        isattr: bool,
        parent,
    ) -> bool:
        """Return *True* when *member* is a Warp function we can handle."""
        return cls._looks_like_warp_function(member)

    # ------------------------------------------------------------------
    # Autodoc overrides - we proxy to the underlying Python function.
    # ------------------------------------------------------------------
    def _unwrap(self):
        """Return the original Python function or *self.object* as fallback."""
        orig = getattr(self.object, "func", None)
        if orig and inspect.isfunction(orig):
            return orig
        return self.object

    # Each of these hooks replaces *self.object* with the unwrapped function
    # *before* delegating to the base implementation.
    def format_args(self):
        self.object = self._unwrap()
        return super().format_args()

    def get_doc(self, *args: Any, **kwargs: Any) -> list[list[str]]:
        self.object = self._unwrap()
        return super().get_doc(*args, **kwargs)

    def add_directive_header(self, sig: str) -> None:
        self.object = self._unwrap()
        super().add_directive_header(sig)


# ----------------------------------------------------------------------------
# Sphinx extension entry point
# ----------------------------------------------------------------------------


def setup(app):  # type: ignore[override]
    """Register the :class:`WarpFunctionDocumenter` with *app*."""

    app.add_autodocumenter(WarpFunctionDocumenter, override=True)
    # Declare the extension safe for parallel reading/writing
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
