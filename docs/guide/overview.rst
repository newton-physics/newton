Newton Physics Documentation
============================

.. image:: /_static/newton-logo.png
   :alt: Newton Physics Engine Logo
   :align: center
   :class: newton-logo

.. raw:: html
    
   <br />

**Newton** is a GPU-accelerated, extensible, and differentiable physics simulation engine designed for robotics, research, and advanced simulation workflows. Built on top of NVIDIA Warp and integrating MuJoCo Warp, Newton provides high-performance simulation, modern Python APIs, and a flexible architecture for both users and developers.

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   key-concepts
   development-guide


Key Features
------------

* **GPU-accelerated**: Leverages NVIDIA Warp for fast, scalable simulation.
* **Multiple solver implementations** - XPBD, VBD, Mujoco, Featherstone, Euler
* **Modular design** - Easily extendable with new solvers and components
* **Differentiable**: Supports differentiable simulation for machine learning and optimization.
* **Rich Import/Export**: Load models from URDF, MJCF, USD, and more.
* **Open Source**: Maintained by Disney Research, Google DeepMind, and NVIDIA.

Example
------------

.. code-block:: python
   :caption: Simple pendulum example
   :linenos:

   import newton as nw
   
   # Create a new physics model
   model = nw.Model()
   
   # Add a pendulum
   body = model.add_body(mass=1.0)
   joint = model.add_joint(
       type=nw.JointType.REVOLUTE,
       parent=model.ground,
       child=body,
       axis=[0, 0, 1]
   )
   
   # Create solver and simulate
   solver = nw.solvers.XPBD(model)
   
   for i in range(100):
       solver.step(dt=0.01)
       print(f"Angle: {joint.get_position()}")


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

:ref:`Full Index <genindex>`\