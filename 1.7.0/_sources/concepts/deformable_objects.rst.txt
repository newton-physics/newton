.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _deformable-objects:

Deformable Objects
==================

Each deformable object has a label, a world index, and ranges identifying its
simulation elements. A rod-backed curve contains bodies and joints. A triangle
surface contains particles, triangles, and bending edges. A tetrahedral volume
contains particles and tetrahedra.

The family names describe the simulation geometry. The builder methods and
supported USD deformable imports populate these lists:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Builder lists
     - Native construction
     - USD deformable import
   * - ``curve_label`` / ``curve_world``
     - :meth:`~newton.ModelBuilder.add_rod`, :meth:`~newton.ModelBuilder.add_rod_graph` (legacy)
     - Cable ``BasisCurves``
   * - ``surface_label`` / ``surface_world``
     - :meth:`~newton.ModelBuilder.add_cloth_mesh`, :meth:`~newton.ModelBuilder.add_cloth_grid`
     - Cloth ``Mesh``
   * - ``volume_label`` / ``volume_world``
     - :meth:`~newton.ModelBuilder.add_soft_mesh`, :meth:`~newton.ModelBuilder.add_soft_grid`
     - Volume ``TetMesh``

Each native call records one deformable object. For example,
:meth:`~newton.ModelBuilder.add_cloth_grid` delegates to
:meth:`~newton.ModelBuilder.add_cloth_mesh` but records the cloth only once.
USD imports record each simulation prim once, including cable prims with several
disconnected curves. Curves welded into a graph instead share one record for the
complete graph; see :ref:`deformable-objects-welded-usd-graphs`. Internal rod calls
for parts of a multi-curve prim do not create extra entries.

Builder identities
------------------

.. experimental::

   The builder's ``curve_label`` / ``curve_world``, ``surface_label`` /
   ``surface_world``, and ``volume_label`` / ``volume_world`` lists may change
   without the normal deprecation period.

Each pair contains one label and world index per deformable object. Native
construction records explicit labels or generates names such as ``curve_0``,
``surface_0``, and ``volume_0``. USD imports use the simulation prim's path, or
the graph identifier for welded curves.
Labels may repeat. An index in these lists is not a body or particle index.

Use the lists to identify assets while composing or cloning a builder:

.. testcode::

   import newton

   prototype = newton.ModelBuilder()
   prototype.add_rod(
       rod=newton.Rod([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)], radius=0.02),
       label="cable",
       body_frame_origin="com",
   )

   # Replace a template name with the application's asset name.
   prototype.curve_label[0] = "gripper_cable"

   scene = newton.ModelBuilder()
   scene.replicate(prototype, 2, label_prefixes=["env_0", "env_1"])
   print(scene.curve_label)
   print(scene.curve_world)

.. testoutput::

   ['env_0/gripper_cable', 'env_1/gripper_cable']
   [0, 1]

Labels are editable. The label and world lists must remain aligned with the
recorded deformable objects, so entries must not be appended, removed, or
reordered manually.

The builder assigns world indices during construction and cloning.
World ``-1`` identifies global deformable objects.
Use :meth:`~newton.ModelBuilder.begin_world` / :meth:`~newton.ModelBuilder.end_world`,
:meth:`~newton.ModelBuilder.add_world`, or :meth:`~newton.ModelBuilder.replicate`
to assign worlds when creating or cloning deformable objects. Do not edit the
world lists: changing an entry does not move its simulation elements.

Composition and fixed-joint collapse
------------------------------------

:meth:`~newton.ModelBuilder.add_builder`, :meth:`~newton.ModelBuilder.add_world`,
and :meth:`~newton.ModelBuilder.replicate` preserve the records and offset their
element ranges. Label prefixes also apply to deformable labels. These records
remain on the builder; :meth:`~newton.ModelBuilder.finalize` does not copy them
to the model. The simulation ranges remain private, and the identity lists do
not provide a public state-selection API.

Labels do not affect :meth:`~newton.ModelBuilder.collapse_fixed_joints`. Complete
curve records follow the remapped indices. If collapse removes part of a curve,
Newton warns and omits its incomplete record. Use ``joints_to_keep`` to preserve
required joints. Particle, triangle, and tetrahedron ranges are not affected by
fixed-joint collapse.

.. _deformable-objects-welded-usd-graphs:

Welded USD graphs
-----------------

A welded USD graph is recorded by one :meth:`~newton.ModelBuilder.add_rod` call,
just like a native graph. Its record includes all created segment bodies and
joints, including the generated root. The source curves are not recorded as
separate deformable objects.

The graph's label is the existing ``graph_component`` identifier in
``path_cable_attrs``. This identifier reuses one source prim's path but labels
the whole graph, not just that curve. Read it from the import result rather
than assuming which source path is chosen.

The return maps still describe each source curve. ``path_cable_map`` keeps its
segment body indices and an empty joint list; ``path_cable_attrs`` keeps its
material and graph identifier. Per-curve joint membership, including shared
joints, remains follow-up work. This does not limit the whole graph's record.

.. code-block:: python

   result = builder.add_usd("harness.usda", return_deformable_results=True)
   branch_path = "/World/Branch"
   graph_label = result["path_cable_attrs"][branch_path]["graph_component"]
   graph_index = builder.curve_label.index(graph_label)
   graph_world = builder.curve_world[graph_index]  # The complete welded graph.
   branch_bodies, branch_joints = result["path_cable_map"][branch_path]
   # branch_bodies contains only the branch's segments; branch_joints is [].
