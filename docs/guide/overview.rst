Guide
========

Newton Physics is a GPU-accelerated, extensible, and differentiable physics simulation engine designed for robotics, research, and advanced simulation workflows. Built on top of NVIDIA Warp and integrating MuJoCo Warp, Newton provides high-performance simulation, modern Python APIs, and a flexible architecture for both users and developers.

For a deeper dive into the design goals and architecture, see :doc:`architecture`.

.. toctree::
   :maxdepth: 1

   overview
   quickstart
   tutorials
   key-concepts
   user-guide
   development-guide


Key Features
------------

- **GPU-accelerated**: Leverages NVIDIA Warp for fast, scalable simulation.
- **Differentiable**: Supports differentiable simulation for machine learning and optimization.
- **Extensible**: Easily add new solvers, importers, and rendering backends.
- **Rich Import/Export**: Load models from URDF, MJCF, USD, and more.
- **Modern Python API**: Intuitive, object-oriented interface for building, simulating, and visualizing physics models.
- **Open Source**: Maintained by Disney Research, Google DeepMind, and NVIDIA.

.. admonition:: Learn More
   :class: tip

   For a deep conceptual introduction, see the [DeepWiki Newton Physics page](https://deepwiki.com/newton-physics/newton).

High-Level Architecture
----------------------

.. mermaid::

   graph TD
     A[ModelBuilder] -->|builds| B[Model]
     B --> C[State]
     C --> D[Solver]
     D --> E[State (next)]
     B --> F[Renderer]
     B --> G[Importer]
     G --> B
     F --> H[Visualization]

- **ModelBuilder**: Constructs models from primitives or imported assets.
- **Model**: Encapsulates the physical structure, parameters, and configuration.
- **State**: Represents the dynamic state (positions, velocities, etc.).
- **Solver**: Advances the simulation by integrating physics.
- **Renderer**: Visualizes the simulation in real-time or offline.
- **Importer**: Loads models from external formats (URDF, MJCF, USD).

Quick Links
-----------

- :doc:`quickstart` — Get started in minutes
- :doc:`tutorials` — Step-by-step guides and examples
- :doc:`user-guide` — In-depth usage and feature documentation
- :doc:`api` — Full API reference
- :doc:`development-guide` — For contributors and advanced users
- :doc:`architecture` — Background and design goals

.. note::
   Newton is in active development. APIs and features may change. See the :doc:`changelog` for updates. 