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

Composition and fixed-joint collapse
------------------------------------

:meth:`~newton.ModelBuilder.add_builder`, :meth:`~newton.ModelBuilder.add_world`,
and :meth:`~newton.ModelBuilder.replicate` preserve the records and offset their
element ranges. Label prefixes also apply to deformable labels. These records
remain on the builder; :meth:`~newton.ModelBuilder.finalize` does not copy them
to the model. The simulation ranges remain private, and the identity lists do
not provide a public state-selection API.

Native rods and USD cables have different joint ranges. Native rod records
include free-root joints created by the call. USD records retain the existing
importer's joint span. For a single curve, it excludes root attachments. For a
prim with multiple curves, the span can include free-root joints between those
curves. A curve welded into a shared graph has an empty joint range because the
graph owns those joints. These private spans are not a public joint-selection API.

Labels do not affect :meth:`~newton.ModelBuilder.collapse_fixed_joints`. Complete
curve records follow the remapped indices. If collapse removes part of a curve,
Newton warns and omits its incomplete record. Use ``joints_to_keep`` to preserve
required joints. Particle, triangle, and tetrahedron ranges are not affected by
fixed-joint collapse.
