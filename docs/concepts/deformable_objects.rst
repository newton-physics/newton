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
curves. Internal rod calls made by the importer do not create extra entries.

Builder identities
------------------

.. experimental::

   The builder's ``curve_label`` / ``curve_world``, ``surface_label`` /
   ``surface_world``, and ``volume_label`` / ``volume_world`` lists may change
   without the normal deprecation period.

Each pair contains one label and world index per deformable object. Native
construction records explicit labels or generates names such as ``curve_0``,
``surface_0``, and ``volume_0``. USD imports use the simulation prim's path.
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

Finalized model identities
--------------------------

.. experimental::

   The model's ``curve_*``, ``surface_*``, and ``volume_*`` identity, count,
   and range attributes may change without the normal deprecation period.

:meth:`~newton.ModelBuilder.finalize` copies the labels, worlds, and simulation
ranges to public :class:`~newton.Model` attributes. Each family retains its
builder order. Later builder edits do not change an existing model. Treat
finalized identities and ranges as read-only.

For example, inspect the cables created above without constructing a view:

.. testcode::

   model = scene.finalize(device="cpu")
   print(model.curve_label)
   print(model.curve_world.numpy().tolist())
   print(model.curve_body_start.numpy().tolist())
   print(model.curve_body_end.numpy().tolist())

.. testoutput::

   ['env_0/gripper_cable', 'env_1/gripper_cable']
   [0, 1]
   [0, 2]
   [2, 4]

There is one entry per recorded deformable object. Here, cable 1 belongs to
world 1 and uses body indices 2 and 3. An end index is excluded from its range.
Unlike :attr:`~newton.Model.articulation_start`, deformable start arrays have
no trailing sentinel. Every start has a separate end, so gaps between objects
and empty ranges are represented directly.

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
recorded object. The :ref:`deformable selection views <deformable-selection>`
select matching objects and read or update their state.

Composition and fixed-joint collapse
------------------------------------

:meth:`~newton.ModelBuilder.add_builder`, :meth:`~newton.ModelBuilder.add_world`,
and :meth:`~newton.ModelBuilder.replicate` preserve the records and offset their
element ranges. Label prefixes also apply to deformable labels. Finalization
copies the resulting identities and ranges to the model. The builder's range
lists remain private; use the public model arrays to inspect finalized ranges.

After finalization, use the family-specific views described in
:ref:`deformable-selection` to select deformable objects and access their state.

Curve joint ranges include the joints created with the deformable object.
Native rods and non-welded USD cables include each generated free root or the
attachment root created in its place. Earlier rigid-body joints and independent
attachments added later are outside the range. These complete ranges differ
from the rod-joint lists returned by :meth:`~newton.ModelBuilder.add_rod` and
USD's ``path_cable_map``, which exclude roots.

A USD curve welded into a shared graph has an empty joint range because the
graph owns those joints. A native call that creates the whole graph records
the graph's joints. The curve view exposes the recorded range through
``ranges("joint")``.

Labels do not affect :meth:`~newton.ModelBuilder.collapse_fixed_joints`. Complete
curve records follow the remapped indices. If collapse removes part of a curve,
Newton warns and omits its incomplete record. Use ``joints_to_keep`` to preserve
required joints. Particle, triangle, and tetrahedron ranges are not affected by
fixed-joint collapse.

Compact coupled-solver models retain only deformable objects whose recorded
elements are all present. Their ranges refer to the compact model's arrays,
not the parent model's arrays. A partial curve is omitted. Particle-based
deformables follow the coupled solver's existing particle layout.
