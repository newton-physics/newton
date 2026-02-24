newton.geometry
===============

.. currentmodule:: newton.geometry

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   BroadPhaseAllPairs
   BroadPhaseExplicit
   BroadPhaseSAP
   HydroelasticSDF
   NarrowPhase

.. rubric:: Nested Class Aliases

.. py:class:: NarrowPhase.HydroelasticSDF
   :canonical: newton.geometry.HydroelasticSDF

   Alias of :class:`~newton.geometry.HydroelasticSDF`.

   Hydroelastic contact generation with SDF-based collision detection.
   
   This class implements hydroelastic contact modeling between shapes represented
   by Signed Distance Fields (SDFs). It uses an octree-based broadphase to identify
   potentially colliding regions, then applies marching cubes to extract the
   zero-isosurface where both SDFs intersect. Contact points are generated at
   triangle centroids on this isosurface, with contact forces proportional to
   penetration depth and represented area.
   
   The collision pipeline consists of:
       1. Broadphase: Identifies overlapping OBBs of SDF between shape pairs
       2. Octree refinement: Hierarchically subdivides blocks to find iso-voxels
       3. Marching cubes: Extracts contact surface triangles from iso-voxels
       4. Contact generation: Computes contact points, normals, depths, and areas
       5. Optional contact reduction: Bins and reduces contacts per shape pair
   
   Args:
       num_shape_pairs: Maximum number of hydroelastic shape pairs to process.
       total_num_tiles: Total number of SDF blocks across all hydroelastic shapes.
       max_num_blocks_per_shape: Maximum block count for any single shape.
       shape_sdf_block_coords: Block coordinates for each shape's SDF representation.
       shape_sdf_shape2blocks: Mapping from shape index to (start, end) block range.
       shape_material_kh: Hydroelastic stiffness coefficient for each shape.
       n_shapes: Total number of shapes in the simulation.
       config: Configuration options controlling buffer sizes, contact reduction,
           and other behavior. Defaults to :class:`HydroelasticSDF.Config`.
       device: Warp device for GPU computation.
       writer_func: Callback for writing decoded contact data.
   
   Note:
       Use :meth:`_from_model` to construct from a simulation :class:`Model`,
       which automatically extracts the required SDF data and shape information.
   
       Contact IDs are packed into 32-bit integers using 9 bits per voxel axis coordinate.
       For SDF grids larger than 512 voxels per axis, contact ID collisions may occur,
       which can affect contact matching accuracy for warm-starting physics solvers.
   
   See Also:
       :class:`HydroelasticSDF.Config`: Configuration options for this class.

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   collide_box_box
   collide_capsule_box
   collide_capsule_capsule
   collide_plane_box
   collide_plane_capsule
   collide_plane_cylinder
   collide_plane_ellipsoid
   collide_plane_sphere
   collide_sphere_box
   collide_sphere_capsule
   collide_sphere_cylinder
   collide_sphere_sphere
   compute_inertia_shape
   create_empty_sdf_data
   sdf_box
   sdf_capsule
   sdf_cone
   sdf_cylinder
   sdf_mesh
   sdf_plane
   sdf_sphere
   transform_inertia
