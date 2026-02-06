Key Concepts
============

Newton Physics is built around a set of core abstractions that make it powerful, extensible, and easy to use for a wide range of simulation tasks. Understanding these concepts will help you get the most out of the engine.

Core Abstractions
-----------------

- :class:`~newton.ModelBuilder`: The entry point for constructing simulation models. You can build models programmatically or import them from standard formats (URDF, MJCF, USD).
- :class:`~newton.Model`: Encapsulates the physical structure, parameters, and configuration of your simulation world, including bodies, joints, shapes, and physical properties.
- :class:`~newton.State`: Represents the dynamic state of the simulation at a given time (positions, velocities, etc.). States are updated by solvers and can be inspected or visualized.
- :class:`~newton.Control`: Encodes the control inputs (e.g., joint targets, forces) applied to the model at each simulation step.
- :doc:`Solver <../api/newton_solvers>`: Advances the simulation by integrating the equations of motion, handling contacts, constraints, and physics integration. Newton supports multiple solver backends (XPBD, VBD, MuJoCo, Featherstone, etc.).
- **Importer**: Loads models from external formats and populates the ModelBuilder (see :meth:`~newton.ModelBuilder.add_urdf`, :meth:`~newton.ModelBuilder.add_mjcf`, :meth:`~newton.ModelBuilder.add_usd`).
- :doc:`Viewer <visualization>`: Visualizes the simulation in real-time (OpenGL) or offline (USD, Omniverse).

Simulation Loop
---------------

1. **Build or import a model** using ModelBuilder.
2. **Initialize the state** (positions, velocities, etc.).
3. **Apply controls** (joint targets, forces).
4. **Step the solver** to advance the simulation.
5. **Render or export** the results.

Further Reading
---------------

- :doc:`tutorials` — Step-by-step guides
- :doc:`../api/newton` — Full API reference
- `DeepWiki: Newton Physics <https://deepwiki.com/newton-physics/newton>`__ — Conceptual background
