// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

(function () {
  "use strict";

  const MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.min.js";
  const MERMAID_INTEGRITY = "sha384-LlKSgo4Eo5GuF/ZrstLti44dE+GC5XAJ7TSu0Nw9Q3vIZF2QMnkRcK7BUoLabYLF";
  let mermaidLoadPromise = null;

  function hasMermaidBlocks() {
    return document.querySelector("pre.mermaid:not([data-processed])") !== null;
  }

  function loadMermaid() {
    if (window.mermaid) {
      return Promise.resolve(window.mermaid);
    }

    if (mermaidLoadPromise) {
      return mermaidLoadPromise;
    }

    mermaidLoadPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = MERMAID_URL;
      script.integrity = MERMAID_INTEGRITY;
      script.crossOrigin = "anonymous";
      script.referrerPolicy = "no-referrer";
      script.onload = () => {
        if (window.mermaid) {
          resolve(window.mermaid);
        } else {
          reject(new Error("Mermaid loaded without exposing window.mermaid"));
        }
      };
      script.onerror = () => reject(new Error("Mermaid failed to load"));
      document.head.appendChild(script);
    });

    return mermaidLoadPromise;
  }

  function replaceWithRenderError(node) {
    node.setAttribute("data-processed", "true");

    const errorNode = document.createElement("div");
    errorNode.className = "admonition warning mermaid-error";

    const title = document.createElement("p");
    title.className = "admonition-title";
    title.textContent = "Mermaid diagram failed to render";

    errorNode.appendChild(title);
    node.replaceWith(errorNode);
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
      console.error("Mermaid render error", err);
      for (const node of nodes) {
        if (!node.hasAttribute("data-processed")) {
          replaceWithRenderError(node);
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
