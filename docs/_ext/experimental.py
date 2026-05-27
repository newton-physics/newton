# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sphinx directive for marking experimental Newton features."""

from __future__ import annotations

from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive

DEFAULT_NOTICE = (
    "Experimental feature. API, behavior, defaults, and supported use cases may change without prior notice."
)


class ExperimentalDirective(Directive):
    """Render a standard admonition for experimental features."""

    has_content = True

    def run(self) -> list[nodes.Node]:
        node = nodes.admonition(classes=["experimental"])
        node += nodes.title(text="Experimental")

        if self.content:
            self.state.nested_parse(self.content, self.content_offset, node)
        else:
            node += nodes.paragraph(text=DEFAULT_NOTICE)

        return [node]


def setup(app: Any) -> dict[str, bool]:
    """Register the ``experimental`` directive with Sphinx."""

    app.add_directive("experimental", ExperimentalDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
