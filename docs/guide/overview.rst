Newton Physics Documentation
============================

.. image:: /_static/newton-logo.jpg
   :alt: Newton Physics Engine Logo
   :align: center
   :class: newton-logo

.. raw:: html
    
   <br />

**Newton** is a GPU-accelerated, extensible, and differentiable physics simulation engine designed for robotics, research, and advanced simulation workflows. Built on top of NVIDIA Warp and integrating MuJoCo Warp, Newton provides high-performance simulation, modern Python APIs, and a flexible architecture for both users and developers.


Key Features
------------

* **GPU-accelerated**: Leverages NVIDIA Warp for fast, scalable simulation.
* **Multiple solver implementations** - XPBD, VBD, Mujoco, Featherstone, Euler
* **Modular design** - Easily extendable with new solvers and components
* **Differentiable**: Supports differentiable simulation for machine learning and optimization.
* **Rich Import/Export**: Load models from URDF, MJCF, USD, and more.
* **Open Source**: Maintained by Disney Research, Google DeepMind, and NVIDIA.

.. admonition:: Learn More
   :class: tip

   For a deep conceptual introduction, see the `DeepWiki Newton Physics page <https://deepwiki.com/newton-physics/newton>`__.


High-Level Architecture
-----------------------

.. mermaid::

   graph TD
   A[ModelBuilder] -->|builds| B[Model]
   B --> C[State]
   C --> D[Solver]
   D --> C
   B --> F[Renderer]
   C --> F
   G[Importer] --> A
   I[Application] --> A
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
- :doc:`../changelog` — Full release notes
- :doc:`../faq` — Frequently asked questions
- :doc:`../development-guide` — For developers and code contributors

.. note::
   Newton is in active development. APIs and features may change. See the :doc:`../changelog` for updates. 

:ref:`Full Index <genindex>`
