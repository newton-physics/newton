.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton
======

.. py:module:: newton
.. currentmodule:: newton

.. rubric:: Submodules

- :doc:`newton.actuators <newton_actuators>`
- :doc:`newton.geometry <newton_geometry>`
- :doc:`newton.ik <newton_ik>`
- :doc:`newton.math <newton_math>`
- :doc:`newton.selection <newton_selection>`
- :doc:`newton.sensors <newton_sensors>`
- :doc:`newton.solvers <newton_solvers>`
- :doc:`newton.usd <newton_usd>`
- :doc:`newton.utils <newton_utils>`
- :doc:`newton.viewer <newton_viewer>`

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   Axis
   BodyFlags
   CollisionPipeline
   Contacts
   Control
   EqType
   Gaussian
   GeoType
   Heightfield
   InverseDynamics
   JointTargetMode
   JointType
   Mesh
   Model
   ModelBuilder
   ModelFlags
   ParticleFlags
   SDF
   ShapeFlags
   State
   StateFlags
   TetMesh

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   AxisType
   eval_fk
   eval_ik
   eval_inverse_dynamics
   eval_inverse_dynamics_force
   eval_jacobian
   eval_mass_matrix
   intersect_ray

.. rubric:: Constants

.. list-table::
   :header-rows: 1

   * - Name
     - Value
   * - ``MAXVAL``
     - ``10000000000.0``
   * - ``__version__``
     - ``1.5.0.dev0``
   * - ``use_coord_layout_targets``
     - ``False``
