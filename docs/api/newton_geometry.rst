.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.geometry
===============

.. py:module:: newton.geometry
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
   ParticleSurface
   TriMeshCollisionInfo

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   build_tri_mesh_collision_info
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
   compute_offset_mesh
   create_empty_sdf_data
   extract_particle_surface
   get_edge_colliding_edges
   get_edge_colliding_edges_count
   get_edge_collision_buffer_edge_index
   get_triangle_colliding_vertices
   get_triangle_colliding_vertices_count
   get_vertex_colliding_triangles
   get_vertex_colliding_triangles_count
   get_vertex_collision_buffer_vertex_index
   sdf_box
   sdf_capsule
   sdf_cone
   sdf_cylinder
   sdf_mesh
   sdf_plane
   sdf_sphere
   transform_inertia

.. rubric:: Deprecated

.. list-table::
   :header-rows: 1

   * - Name
     - Guidance
   * - ``MATCH_BROKEN``
     - Do not rely on this value.
   * - ``MATCH_NOT_FOUND``
     - Do not rely on this value.
