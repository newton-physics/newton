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

Each native call records one deformable object.
USD imports record each simulation prim once, including cable prims with several
disconnected curves. Curves welded into a graph instead share one record for the
complete graph; see :ref:`deformable-objects-welded-usd-graphs`.

.. _deformable-objects-builder-identities:

Builder labels and worlds
-------------------------

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

Model labels and ranges
-----------------------

.. experimental::

   The model's ``curve_*``, ``surface_*``, and ``volume_*`` label, world,
   count, and range attributes may change without the normal deprecation period.

:meth:`~newton.ModelBuilder.finalize` copies the labels, worlds, and simulation
ranges to public :class:`~newton.Model` attributes. Each family retains its
builder order. Later builder edits do not change an existing model. Treat
model labels, worlds, and ranges as read-only.

Continue with ``scene`` from :ref:`deformable-objects-builder-identities`.
It contains two cables, named ``env_0/gripper_cable`` and
``env_1/gripper_cable``. Inspect their model data without constructing a view:

.. testcode::

   model = scene.finalize(device="cpu")
   print("curve_label:", model.curve_label)
   print("curve_world:", model.curve_world.numpy().tolist())
   print("curve_body_start:", model.curve_body_start.numpy().tolist())
   print("curve_body_end:", model.curve_body_end.numpy().tolist())

.. testoutput::

   curve_label: ['env_0/gripper_cable', 'env_1/gripper_cable']
   curve_world: [0, 1]
   curve_body_start: [0, 2]
   curve_body_end: [2, 4]

The example uses ``device="cpu"`` so it also runs without a GPU. The same
model attributes are available with ``device="cuda:0"``.

Every list or array has one entry per curve. Index ``1`` identifies the second
cable, ``env_1/gripper_cable``. Entry ``1`` of ``curve_world`` is ``1``, so this
cable belongs to world 1. Its body range starts at ``2`` and ends at ``4``.
The end is excluded, so the range contains body indices ``2`` and ``3``:

.. testcode::

   curve_index = 1
   body_start = int(model.curve_body_start.numpy()[curve_index])
   body_end = int(model.curve_body_end.numpy()[curve_index])
   print(model.curve_label[curve_index])
   print(list(range(body_start, body_end)))

.. testoutput::

   env_1/gripper_cable
   [2, 3]

Curve indices identify deformable objects. Body indices identify the rigid
segments in the model's body arrays. Do not use one in place of the other.
Each deformable object has a start and an end. Unlike
:attr:`~newton.Model.articulation_start`, the start arrays have no extra final
entry. Separate ends allow gaps between deformable objects and empty ranges.

.. list-table::
   :header-rows: 1

   * - Family
     - Labels, worlds, and counts
     - Range pairs
   * - Curve
     - ``curve_label``, ``curve_world``, ``curve_count``
     - ``curve_body_start/end``, ``curve_joint_start/end``
   * - Surface
     - ``surface_label``, ``surface_world``, ``surface_count``
     - ``surface_particle_start/end``, ``surface_tri_start/end``, ``surface_edge_start/end``
   * - Volume
     - ``volume_label``, ``volume_world``, ``volume_count``
     - ``volume_particle_start/end``, ``volume_tet_start/end``

``start/end`` denotes two attributes, such as ``surface_particle_start`` and
``surface_particle_end``. Labels are Python lists. Worlds and range endpoints
are one-dimensional ``int32`` Warp arrays on the model device. Empty families
have empty lists and arrays. World ``-1`` still means global.

This follows the label/world pattern of :attr:`~newton.Model.articulation_label`
and :attr:`~newton.Model.articulation_world`. Model attributes describe every
recorded deformable object. The :ref:`deformable selection views <deformable-selection>`
select matching deformable objects and read or update their state.

.. _deformable-objects-curve-joints:

Curve joint ranges
~~~~~~~~~~~~~~~~~~

Curve joint ranges include the joints created with the deformable object.
Native rods and USD cables include each generated free root or the
attachment root created in its place. Earlier rigid-body joints and independent
attachments added later are outside the range. These complete ranges differ
from the rod-joint lists returned by :meth:`~newton.ModelBuilder.add_rod` and
USD's ``path_cable_map``, which exclude roots.

Combining builders
------------------

:meth:`~newton.ModelBuilder.add_builder`, :meth:`~newton.ModelBuilder.add_world`,
and :meth:`~newton.ModelBuilder.replicate` preserve the records and offset their
element ranges. Label prefixes also apply to deformable labels. Finalization
copies the resulting labels, worlds, and ranges to the model. The builder's range
lists remain private; use the public model arrays to inspect finalized ranges.

.. _deformable-objects-collapse:

Fixed-joint collapse
--------------------

Labels do not affect :meth:`~newton.ModelBuilder.collapse_fixed_joints`. Curve
records retain their labels and describe the surviving segment bodies and joints.
A curve with a remaining body is retained even if its joint range is empty.
Newton warns and omits a curve only when none of its segment bodies survive.
Use ``joints_to_keep`` to preserve the original segment layout. Particle,
triangle, and tetrahedron ranges are not affected by fixed-joint collapse.

Coupled solver models
---------------------

Compact coupled-solver models retain only deformable objects whose recorded
elements are all present. Their ranges refer to the compact model's arrays,
not the parent model's arrays. If a solver partition excludes part of a curve's
current recorded ranges, that curve is omitted from the compact model. This is
separate from reducing the ranges during builder collapse. Particle-based
deformables follow the coupled solver's existing particle layout.

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
