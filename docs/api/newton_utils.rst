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

.. rubric:: Functions

.. autosummary::
   :toctree: _generated
   :signatures: long

   bourke_color_map
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
     - Deprecated in 1.6; pass direct stiffness values to ``newton.ModelBuilder.add_rod()`` instead.
   * - ``create_cable_stiffness_from_elastic_moduli``
     - Deprecated in 1.6; supply elastic material through ``newton.Rod(...)`` or pass direct stiffness values to ``newton.ModelBuilder.add_rod()`` instead.
   * - ``create_parallel_transport_cable_quaternions``
     - Deprecated in 1.6; construct ``rod = newton.Rod(points)`` and read ``rod.quaternions`` instead. For nonzero ``twist_total``, first call ``rod.compute_frames(twist_total=twist_total)``.
   * - ``create_straight_cable_points``
     - Deprecated in 1.6; use ``newton.Rod.create_straight(...).points`` instead.
   * - ``create_straight_cable_points_and_quaternions``
     - Deprecated in 1.6; use ``newton.Rod.create_straight(...)`` instead.
