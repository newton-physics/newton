API Reference
=============

This section provides a comprehensive reference for the Newton Physics Python API. All classes, functions, and modules are documented here, with links to detailed docstrings and usage examples.

Core
----

Core data structures and simulation objects.

.. list-table:: Core Classes
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - Model
     - Simulation model structure
   * - State
     - Dynamic simulation state
   * - Control
     - Actuator/control inputs
   * - ModelBuilder
     - Model construction API
   * - Mesh, SDF, Axis
     - Geometry and math types

See :doc:`api_core` for details.

Solvers
-------

Physics solvers for advancing simulation. Each solver implements a common interface and can be selected based on your needs.

.. list-table:: Solver Classes
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - SolverBase
     - Abstract solver interface
   * - XPBDSolver
     - Position-based dynamics
   * - VBDSolver
     - Vertex Block Descent solver
   * - MuJoCoSolver
     - MuJoCo backend
   * - FeatherstoneSolver
     - Articulated rigid bodies
   * - SemiImplicitSolver
     - Semi-implicit Euler

See :doc:`api_solvers` for details.

Importers
---------

Load models from standard formats.

.. list-table:: Importer Functions
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - parse_urdf
     - Import URDF models
   * - parse_mjcf
     - Import MJCF (MuJoCo XML)
   * - parse_usd
     - Import USD scenes
   * - resolve_usd_from_url
     - Download/resolve USD assets

See :doc:`api_importers` for details.

Renderers
---------

Visualize simulations in real-time or offline.

.. list-table:: Renderer Classes
   :widths: 40 60
   :header-rows: 1

   * - Class
     - Description
   * - SimRendererUsd
     - USD/Omniverse renderer
   * - SimRendererOpenGL
     - Real-time OpenGL renderer

See :doc:`api_renderers` for details.
