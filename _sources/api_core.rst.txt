Core
====

This section documents the core data structures and simulation objects in Newton Physics.

Classes
-------

.. autosummary::
   :toctree: generated/core
   :template: class.rst

   newton.Model
   newton.State
   newton.Control
   newton.ModelBuilder
   newton.Mesh
   newton.SDF

.. autoclass:: newton.Axis
   :noindex:

.. autoclass:: newton.AxisType
   :noindex:

.. _geometry-types:

Geometry Types
--------------

These constants define the supported geometry types for rigid body shapes in Newton simulations.
They are defined in ``newton.core.types`` and are used when assigning collision geometry to bodies or shapes.

==================  ================================================================
Constant            Description
==================  ================================================================
``GEO_SPHERE``       A sphere defined by a center and radius.
``GEO_BOX``          An axis-aligned box defined by half-extents.
``GEO_CAPSULE``      A capsule defined by two endpoints and a radius.
``GEO_CYLINDER``     A cylinder defined by two endpoints and a radius.
``GEO_CONE``         A cone aligned along an axis with a base radius and height.
``GEO_MESH``         A triangle mesh with vertex and face data.
``GEO_SDF``          A signed distance field represented on a grid.
``GEO_PLANE``        An infinite plane defined by a normal and offset.
``GEO_NONE``         No geometry; used to mark inactive or undefined shapes.
==================  ================================================================

.. _joint-types:

Joint Types
-----------

These constants define the supported types of joints used to constrain motion between rigid bodies.
They are defined in ``newton.core.types`` and used when constructing models.

=====================  ============================================================
Constant               Description
=====================  ============================================================
``JOINT_PRISMATIC``    Allows translation along a single axis (prismatic joint).
``JOINT_REVOLUTE``     Allows rotation around a single axis (revolute/hinge joint).
``JOINT_BALL``         Allows rotation in all directions around a point (ball joint).
``JOINT_FIXED``        Prevents all relative motion (fixed joint).
``JOINT_FREE``         Allows full freedom (no constraints between bodies).
``JOINT_COMPOUND``     A compound joint made from multiple constraints.
``JOINT_UNIVERSAL``    Allows two rotational degrees of freedom (universal joint).
``JOINT_DISTANCE``     Constrains two points to a fixed distance (distance joint).
``JOINT_D6``           6-DoF joint with configurable limits per axis.
=====================  ============================================================


.. _particle-flags:

Particle Flags
--------------

These flags indicate special states or behaviors for particles in Newton simulations.
They are defined in ``newton.core.types`` and can be used to tag or filter particles.

==========================  ======================================
Constant                    Description
==========================  ======================================
``PARTICLE_FLAG_ACTIVE``     The particle is active and should be simulated.
==========================  ======================================

.. _shape-flags:

Shape Flags
--------------

These flags control rendering and collision behavior for shapes in Newton simulations.
They are defined in ``newton.core.types`` and can be bitwise ORd together to configure shape behavior.

=============================  ===============================================================
Constant                       Description
=============================  ===============================================================
``SHAPE_FLAG_VISIBLE``         Shape is visible in rendering.
``SHAPE_FLAG_COLLIDE_SHAPES``  Enables collision with other shapes.
``SHAPE_FLAG_COLLIDE_GROUND``  Enables collision with ground planes or static terrain.
=============================  ===============================================================

.. _model-update-flags:

Update Flags
------------------

These bitmask flags are used with :py:meth:`newton.solvers.SolverBase.notify_model_changed` to inform a solver which parts
of a :class:`newton.Model` were modified after the solver was created. Individual solver implementations
are responsible for interpreting these flags and updating any internal data structures from the model as required.

==========================================  =============================================================
Constant                                    Description
==========================================  =============================================================
``NOTIFY_FLAG_JOINT_PROPERTIES``            Joint transforms or coordinates have changed.
``NOTIFY_FLAG_JOINT_AXIS_PROPERTIES``       Joint axis limits, targets, or modes have changed.
``NOTIFY_FLAG_DOF_PROPERTIES``              Joint DOF state or force buffers have changed.
``NOTIFY_FLAG_BODY_PROPERTIES``             Rigid-body pose or velocity buffers have changed.
``NOTIFY_FLAG_BODY_INERTIAL_PROPERTIES``    Rigid-body mass or inertia tensors have changed.
``NOTIFY_FLAG_SHAPE_PROPERTIES``            Shape transforms or geometry have changed.
==========================================  =============================================================

