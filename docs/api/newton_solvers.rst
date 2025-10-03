newton.solvers
==============

Solvers are used to integrate the dynamics of a Newton model.

Supported Features
------------------

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 0

   * - Solver
     - Integration
     - Rigid bodies
     - :ref:`Articulations <Articulations>`
     - :abbr:`Rigid contacts (Are contacts between rigid bodies simulated?)`
     - Particles
     - Cloth
     - Soft bodies
   * - :class:`~newton.solvers.SolverFeatherstone`
     - Explicit
     - ✅
     - ✅ generalized coordinates
     - ✅
     - ✅
     - 🟨 no self-collision
     - ✅
   * - :class:`~newton.solvers.SolverImplicitMPM`
     - Implicit
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverMuJoCo`
     - Explicit, Semi-implicit, Implicit
     - ✅
     - ✅ generalized coordinates
     - ✅ (uses its own collision pipeline from MuJoCo/mujoco_warp by default, unless ``use_mujoco_contacts`` is set to False)
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverSemiImplicit`
     - Semi-implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - ✅
     - 🟨 no self-collision
     - ✅
   * - :class:`~newton.solvers.SolverStyle3D`
     - Implicit
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌
   * - :class:`~newton.solvers.SolverVBD`
     - Implicit
     - ❌
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌
   * - :class:`~newton.solvers.SolverXPBD`
     - Implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - ✅
     - 🟨 no self-collision
     - 🟨 experimental

.. currentmodule:: newton.solvers

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
