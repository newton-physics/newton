// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

(function () {
  "use strict";

  const MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.esm.min.mjs";

  function hasMermaidBlocks() {
    return document.querySelector("pre.mermaid:not([data-processed])") !== null;
  }

  function loadMermaid() {
    if (window.mermaid) {
      return Promise.resolve(window.mermaid);
    }

    return import(MERMAID_URL).then((module) => module.default);
  }

  async function renderMermaidBlocks() {
    if (!hasMermaidBlocks()) {
      return;
    }

    const mermaid = await loadMermaid();
    mermaid.initialize({
      startOnLoad: false,
      theme: "forest",
      themeVariables: {
        lineColor: "#76b900",
      },
    });

    const nodes = Array.from(document.querySelectorAll("pre.mermaid:not([data-processed])"));
    if (nodes.length === 0) {
      return;
    }

    try {
      await mermaid.run({ nodes });
    } catch (err) {
      for (const node of nodes) {
        if (!node.hasAttribute("data-processed")) {
          node.textContent = `Mermaid render error: ${err && err.message ? err.message : err}`;
        }
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMermaidBlocks);
  } else {
    renderMermaidBlocks();
  }
})();
