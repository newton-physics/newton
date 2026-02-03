newton
======

.. currentmodule:: newton

.. toctree::
   :hidden:

   newton_geometry
   newton_ik
   newton_selection
   newton_sensors
   newton_solvers
   newton_utils
   newton_viewer

.. rubric:: Submodules

- :doc:`newton.geometry <newton_geometry>`
- :doc:`newton.ik <newton_ik>`
- :doc:`newton.selection <newton_selection>`
- :doc:`newton.sensors <newton_sensors>`
- :doc:`newton.solvers <newton_solvers>`
- :doc:`newton.utils <newton_utils>`
- :doc:`newton.viewer <newton_viewer>`

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ActuatorMode
   Axis
   BroadPhaseMode
   CollisionPipeline
   CollisionPipelineUnified
   Contacts
   Control
   EqType
   GeoType
   JointType
   Mesh
   Model
   ModelBuilder
   ParticleFlags
   SAPSortType
   SDF
   ShapeFlags
   State

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   AxisType
   count_rigid_contact_points
   eval_fk
   eval_ik
   get_joint_constraint_count
   get_joint_dof_count
   infer_actuator_mode

.. rubric:: Constants

.. list-table::
   :header-rows: 1

   * - Name
     - Value
   * - MAXVAL
     - 10000000000.0
   * - __version__
     - 0.2.0
