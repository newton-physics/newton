newton.geometry
===============

.. currentmodule:: newton.geometry

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   SDF
   BroadPhaseAllPairs
   BroadPhaseExplicit
   BroadPhaseSAP
   Mesh

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   build_ccd_generic
   compute_shape_inertia
   compute_shape_radius
   create_box
   create_capsule
   create_cone
   create_cylinder
   create_none
   create_plane
   create_sphere
   transform_inertia

.. rubric:: Constants

.. list-table::
   :header-rows: 1

   * - Name
     - Value
   * - GeoType.BOX
     - 1
   * - GeoType.CAPSULE
     - 2
   * - GeoType.CONE
     - 4
   * - GeoType.CYLINDER
     - 3
   * - GeoType.MESH
     - 5
   * - GeoType.NONE
     - 8
   * - GeoType.PLANE
     - 7
   * - GeoType.SDF
     - 6
   * - GeoType.SPHERE
     - 0
   * - MESH_MAXHULLVERT
     - 64
   * - PARTICLE_FLAG_ACTIVE
     - 1
   * - SHAPE_FLAG_COLLIDE_PARTICLES
     - 4
   * - SHAPE_FLAG_COLLIDE_SHAPES
     - 2
   * - SHAPE_FLAG_VISIBLE
     - 1
