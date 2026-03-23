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


.. _guide-core-concepts:

Core Concepts
-------------

.. mermaid::
   :config: {"theme": "forest", "themeVariables": {"lineColor": "#76b900"}}

   flowchart LR
       subgraph Authoring["Model Authoring"]
           direction LR
           P[Python API] --> A[ModelBuilder]

           subgraph Imported["Imported assets"]
               direction TB
               U[URDF]
               M[MJCF]
               S[USD]
           end

           U --> G[Importer]
           M --> G
           S --> G
           G --> A
       end

       B[Model]

       subgraph Loop["Simulation Loop"]
           direction LR
           C[State] --> D[Solver]
           J[Control] --> D
           E[Contacts] --> D
           D --> C2[Updated state]
       end

       subgraph Outputs["Outputs"]
           direction TB
           K[Sensors]
           F[Viewer]
       end

       A -->|builds| B
       B --> C
       B --> J
       B --> E
       C2 --> K
       E --> K
       C2 --> F

- :class:`~newton.ModelBuilder`: The entry point for constructing
  simulation models from primitives or imported assets.
- :class:`~newton.Model`: Encapsulates the physical structure,
  parameters, and configuration of the simulation world, including
  bodies, joints, shapes, and physical properties.
- :class:`~newton.State`: Represents the dynamic state at a given time,
  including positions and velocities that solvers update each step.
  Optional :doc:`extended attributes <../concepts/extended_attributes>`
  store derived quantities such as rigid-body accelerations for
  sensors.
- :class:`~newton.Contacts`: Stores the active contact set produced by
  :meth:`Model.collide <newton.Model.collide>`, with optional extended
  attributes such as contact forces for sensing and analysis.
- :class:`~newton.Control`: Encodes control inputs such as joint targets
  and forces applied during the simulation loop.
- :doc:`Solver <../api/newton_solvers>`: Advances the simulation by
  integrating physics, handling contacts, and enforcing constraints.
  Newton provides multiple solver backends, including XPBD, VBD,
  MuJoCo, Featherstone, and SemiImplicit.
- :doc:`Sensors <../concepts/sensors>`: Compute observations from
  :class:`~newton.State`, :class:`~newton.Contacts`, sites, and shapes.
  Many sensors rely on optional :doc:`extended attributes
  <../concepts/extended_attributes>` that store derived solver outputs.
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
3. Create any sensors, then allocate one or more
   :class:`~newton.State` objects plus :class:`~newton.Control` inputs
   and :class:`~newton.Contacts`.
4. Call :meth:`Model.collide <newton.Model.collide>` to populate the
   contact set for the current state.
5. Step a :doc:`solver <../api/newton_solvers>` using the current
   state, control, and contacts.
6. Update sensors, inspect outputs, render, or export the results.

Quick Links
-----------

- :doc:`installation` — Setup Newton and run a first example in a couple of minutes
- :doc:`tutorials` — Browse the guide's tutorial landing page
- :doc:`Introduction tutorial </tutorials/00_introduction>` — Walk through a first hands-on tutorial
- :doc:`../faq` — Frequently asked questions
- :doc:`development` — For developers and code contributors
- :doc:`../api/newton` — Full API reference

:ref:`Full Index <genindex>`
