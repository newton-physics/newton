.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _deformable-objects:

Deformable Objects
==================

A deformable object is a complete cable, cloth, or soft volume. Newton records
its label, world, and the simulation elements that belong to it. A rod-backed
curve contains bodies and joints. A triangle surface contains particles,
triangles, and bending edges. A tetrahedral volume contains particles and
tetrahedra. Recording this information does not change the simulation.

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

World ``-1`` identifies global deformable objects. The builder assigns world
indices during construction and cloning. Editing an identity does not move its
simulation elements to another world. Keep the label and world entries aligned
with the recorded deformable objects; do not append, remove, or reorder entries
manually. Change existing labels rather than creating replacement records.

Composition and finalization
----------------------------

:meth:`~newton.ModelBuilder.add_builder`, :meth:`~newton.ModelBuilder.add_world`,
and :meth:`~newton.ModelBuilder.replicate` preserve the records and offset their
element ranges. Label prefixes also apply to deformable labels. Finalization
retains the records on the model, but their storage and simulation ranges remain
private. These identity lists do not provide a public state-selection API.

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
