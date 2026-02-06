newton.solvers
==============

Solvers are used to integrate the dynamics of a Newton model.
The typical workflow is to construct a :class:`~newton.Model` and a :class:`~newton.State` object, then use a solver to advance the state forward in time
via the :meth:`~newton.solvers.SolverBase.step` method:

.. mermaid::
  :config: {"theme": "forest", "themeVariables": {"lineColor": "#76b900"}}

  flowchart LR
      subgraph Input["Input Data"]
          M[newton.Model]
          S[newton.State]
          C[newton.Control]
          K[newton.Contacts]
          DT[Time step dt]
      end

      STEP["solver.step()"]

      subgraph Output["Output Data"]
          SO["newton.State (updated)"]
      end

      %% Connections
      M --> STEP
      S --> STEP
      C --> STEP
      K --> STEP
      DT --> STEP
      STEP --> SO

Supported Features
------------------

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 0

   * - Solver
     - :abbr:`Integration (Available methods for integrating the dynamics)`
     - Rigid bodies
     - :ref:`Articulations <Articulations>`
     - Particles
     - Cloth
     - Soft bodies
   * - :class:`~newton.solvers.SolverFeatherstone`
     - Explicit
     - ✅
     - ✅ generalized coordinates
     - ✅
     - 🟨 no self-collision
     - ✅
   * - :class:`~newton.solvers.SolverImplicitMPM`
     - Implicit
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverMuJoCo`
     - Explicit, Semi-implicit, Implicit
     - ✅ (uses its own collision pipeline from MuJoCo/mujoco_warp by default, unless ``use_mujoco_contacts`` is set to False)
     - ✅ generalized coordinates
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverSemiImplicit`
     - Semi-implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - 🟨 no self-collision
     - ✅
   * - :class:`~newton.solvers.SolverStyle3D`
     - Implicit
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌
   * - :class:`~newton.solvers.SolverVBD`
     - Implicit
     - ✅
     - ❌
     - ✅
     - ✅
     - ❌
   * - :class:`~newton.solvers.SolverXPBD`
     - Implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - 🟨 no self-collision
     - 🟨 experimental

.. currentmodule:: newton.solvers

.. toctree::
   :hidden:

   newton_solvers_style3d

.. rubric:: Submodules

- :doc:`newton.solvers.style3d <newton_solvers_style3d>`

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   SolverBase
   SolverFeatherstone
   SolverImplicitMPM
   SolverMuJoCo
   SolverNotifyFlags
   SolverSemiImplicit
   SolverStyle3D
   SolverVBD
   SolverXPBD
