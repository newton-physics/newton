.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.utils
============

.. py:module:: newton.utils
.. currentmodule:: newton.utils

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ColorSpace
   EventTracer
   MeshAdjacency
   MeshAdjacencyData
   RodStiffness

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   bourke_color_map
   cable_straight_points
   color_graph
   color_linear_to_srgb
   color_srgb_to_linear
   compute_world_offsets
   download_asset
   event_scope
   load_texture
   normalize_texture
   plot_graph
   rasterize_mesh_to_heightfield
   remesh_mesh
   rod_parallel_transport_quaternions
   rod_stiffness_from_elastic_moduli
   rod_straight_points_and_quaternions
   run_benchmark
   solidify_mesh
   string_to_warp
   validate_tet_mesh
   validate_triangle_mesh

.. rubric:: Deprecated

.. list-table::
   :header-rows: 1

   * - Name
     - Guidance
   * - ``CableStiffness``
     - Deprecated in 1.6; use RodStiffness instead.
   * - ``create_cable_stiffness_from_elastic_moduli``
     - Deprecated in 1.6; use rod_stiffness_from_elastic_moduli instead.
   * - ``create_parallel_transport_cable_quaternions``
     - Deprecated in 1.6; use rod_parallel_transport_quaternions instead.
   * - ``create_straight_cable_points``
     - Deprecated in 1.6; use cable_straight_points instead.
   * - ``create_straight_cable_points_and_quaternions``
     - Deprecated in 1.6; use rod_straight_points_and_quaternions instead.
