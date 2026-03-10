.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Newton Physics Documentation
============================

.. image:: /_static/newton-banner.jpg
   :alt: Newton Physics Engine Banner
   :align: center
   :class: newton-banner

.. raw:: html
    
   <br />

**Newton** is a GPU-accelerated, extensible, and differentiable physics simulation engine designed for robotics, research, and advanced simulation workflows. Built on top of NVIDIA Warp and integrating MuJoCo Warp, Newton provides high-performance simulation, modern Python APIs, and a flexible architecture for both users and developers.


Key Features
------------

* **GPU-accelerated**: Leverages NVIDIA Warp for fast, scalable simulation.
* **Multiple solver implementations**: XPBD, VBD, MuJoCo, Featherstone, SemiImplicit.
* **Modular design**: Easily extendable with new solvers and components.
* **Differentiable**: Supports differentiable simulation for machine learning and optimization.
* **Rich Import/Export**: Load models from URDF, MJCF, USD, and more.
* **Open Source**: Maintained by Disney Research, Google DeepMind, and NVIDIA.

.. admonition:: Learn More
   :class: tip

   Start with the :doc:`introduction tutorial </tutorials/00_introduction>` for a
   hands-on walkthrough. For a deeper conceptual introduction, see the
   `DeepWiki Newton Physics page <https://deepwiki.com/newton-physics/newton>`__.


Core Concepts
-------------

.. mermaid::
   :config: {"theme": "forest", "themeVariables": {"lineColor": "#76b900"}}

   graph TD
   A[ModelBuilder] -->|builds| B[Model]
   B --> C[State]
   C --> D[Solver]
   D --> C
   B --> F[Viewer]
   C --> F
   G[Importer] --> A
   I[Application] --> A
   F --> H[Visualization]

- :class:`~newton.ModelBuilder`: The entry point for constructing
  simulation models from primitives or imported assets.
- :class:`~newton.Model`: Encapsulates the physical structure,
  parameters, and configuration of the simulation world, including
  bodies, joints, shapes, and physical properties.
- :class:`~newton.State`: Represents the dynamic state at a given time,
  including positions and velocities that solvers update each step.
- :class:`~newton.Control`: Encodes control inputs such as joint targets
  and forces applied during the simulation loop.
- :doc:`Solver <../api/newton_solvers>`: Advances the simulation by
  integrating physics, handling contacts, and enforcing constraints.
  Newton provides multiple solver backends, including XPBD, VBD,
  MuJoCo, Featherstone, and SemiImplicit.
- **Importer**: Loads models from external formats via
  :meth:`~newton.ModelBuilder.add_urdf`,
  :meth:`~newton.ModelBuilder.add_mjcf`, and
  :meth:`~newton.ModelBuilder.add_usd`.
- :doc:`Viewer <visualization>`: Visualizes the simulation in real time
  or offline.

Simulation Loop
---------------

1. Build or import a model with :class:`~newton.ModelBuilder`.
2. Finalize the builder into a :class:`~newton.Model`.
3. Initialize a :class:`~newton.State` and any :class:`~newton.Control`
   inputs.
4. Step a :doc:`solver <../api/newton_solvers>` to advance the
   simulation.
5. Inspect, render, or export the results.

Quick Links
-----------

- :doc:`installation` — Setup Newton and run a first example in a couple of minutes
- :doc:`tutorials` — Browse the guide's tutorial landing page
- :doc:`Introduction tutorial </tutorials/00_introduction>` — Walk through a first hands-on tutorial
- :doc:`../faq` — Frequently asked questions
- :doc:`development` — For developers and code contributors
- :doc:`../api/newton` — Full API reference

:ref:`Full Index <genindex>`
