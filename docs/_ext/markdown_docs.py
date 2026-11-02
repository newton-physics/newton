# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep Newton's resolved Sphinx documentation readable as Markdown.

The translator only fills gaps in sphinx-markdown-builder 0.6.11. Remove
handlers as upstream gains equivalent support; unsupported nodes must fail
instead of dropping documentation. This extension does not convert HTML.
"""

import json
import pkgutil
import shutil
from pathlib import Path
from types import ModuleType

from docutils import nodes
from sphinx import addnodes
from sphinx.util.osutil import relative_uri
from sphinx_markdown_builder.contexts import TitleContext
from sphinx_markdown_builder.translator import MarkdownTranslator, pushing_context


class NewtonMarkdownTranslator(MarkdownTranslator):
    def visit_admonition(self, node):
        self.ensure_eol(2)

    def depart_admonition(self, node):
        self.ensure_eol(2)

    @pushing_context
    def visit_title(self, node):
        if isinstance(node.parent, nodes.admonition):
            self._push_context(TitleContext(4))
        else:
            super().visit_title(node)

    def depart_abbreviation(self, node):
        if explanation := node.get("explanation"):
            self.add(f" ({explanation})")

    def visit_CodeAreaNode(self, node):
        pass

    def visit_caption(self, node):
        self.ensure_eol(2)

    def depart_caption(self, node):
        self.ensure_eol(2)

    def visit_desc_type_parameter_list(self, node):
        self.add("[")

    def depart_desc_type_parameter_list(self, node):
        self.add("]")

    def visit_desc_type_parameter(self, node):
        if node.parent.index(node):
            self.add(", ")

    def visit_mermaid(self, node):
        self.add(f"```mermaid\n{node['code']}\n```", prefix_eol=2, suffix_eol=2)
        raise nodes.SkipNode

    def visit_image(self, node):
        uri = node["uri"]
        if uri in self.builder.env.images:
            uri = relative_uri(
                self.builder.get_target_uri(self.builder.current_doc_name),
                "_images/" + self.builder.env.images[uri][1],
            )
        self.add(f"![{node.get('alt', 'image')}]({uri})")

    def visit_raw(self, node):
        # Browser controls and HTML redirects are not documentation content.
        if "markdown" in node.get("format", "").split():
            self.add(node.astext())
        raise nodes.SkipNode

    def unknown_visit(self, node):
        raise NotImplementedError(f"Unsupported Markdown node: {type(node).__name__}")


def write_api_index(app, exception):
    """Generate the alphabetical index from the same inventory as API links."""
    if exception is not None or app.builder.name != "markdown":
        return
    for source, (_documents, destination) in app.env.images.items():
        target = Path(app.outdir) / "_images" / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(app.srcdir) / source, target)
    objects = {}
    for name, _display, kind, docname, anchor, _priority in app.env.domains["py"].get_objects():
        if not name.startswith("newton.") or name.startswith("newton._src."):
            continue
        path = app.builder.get_target_uri(docname)
        if not (Path(app.outdir) / path).is_file():
            raise ValueError(f"API page is missing: {path}")
        objects[name] = {"path": f"{path}#{anchor}", "kind": kind}
        try:
            value = pkgutil.resolve_name(name)
        except (AttributeError, ImportError, ValueError):
            continue
        value = getattr(value, "__func__", value)
        if isinstance(value, ModuleType):
            objects[name]["identity"] = value.__name__
            continue
        if type(value).__module__.startswith("warp."):
            value = getattr(value, "func", value)
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if module and qualname:
            objects[name]["identity"] = f"{module}:{qualname}"
    lines = ["# API index", "", f"Newton {app.config.release}", ""]
    lines.extend(f"- [`{name}`]({entry['path']})" for name, entry in sorted(objects.items()))
    (Path(app.outdir) / "genindex.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (Path(app.outdir) / "symbols.json").write_text(
        json.dumps(objects, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_navigation_index(app, exception):
    """Derive the agent entrypoint from the site's actual navigation."""
    if exception is not None or app.builder.name != "markdown":
        return
    lines = [f"# Newton {app.config.release}", "", "> Guides and API reference. Documentation: CC-BY-4.0.", ""]
    for toc in app.env.get_doctree(app.config.root_doc).findall(addnodes.toctree):
        lines.extend([f"## {toc.get('caption') or 'API Reference'}", ""])
        for label, target in toc["entries"]:
            if target == "self":
                continue
            if "://" in target:
                title, link = label or target, target
            else:
                title = label or app.env.titles[target].astext()
                link = app.builder.get_target_uri(target)
                if not (Path(app.outdir) / link).is_file():
                    raise ValueError(f"Indexed documentation page is missing: {link}")
            lines.append(f"- [{title}]({link})")
        lines.append("")
    (Path(app.outdir) / "llms.txt").write_text("\n".join(lines), encoding="utf-8")


def setup(app):
    app.setup_extension("sphinx_markdown_builder")
    app.set_translator("markdown", NewtonMarkdownTranslator, override=True)
    app.connect("build-finished", write_api_index)
    app.connect("build-finished", write_navigation_index)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
