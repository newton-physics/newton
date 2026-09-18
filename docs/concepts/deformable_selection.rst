.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _deformable-selection:

Deformable Selection
====================

.. experimental::

   The deformable selection views may change while the API is developed.

Use :class:`~newton.selection.DeformableCurveView`,
:class:`~newton.selection.DeformableSurfaceView`, or
:class:`~newton.selection.DeformableVolumeView` to find deformables in a finalized
model and read or update their state. The class selects the geometric family:

.. code-block:: python

    from newton.selection import (
        DeformableCurveView,
        DeformableSurfaceView,
        DeformableVolumeView,
    )

    curves = DeformableCurveView(model, "*/gripper_cable")
    surfaces = DeformableSurfaceView(model, "*/table_cloth")
    volumes = DeformableVolumeView(model, "*/soft_toy")

Patterns accept glob strings, lists of globs, and compiled regular expressions,
as in :class:`~newton.selection.ArticulationView`. A broad pattern selects only
the class's family. A pattern with no matches in that family raises ``KeyError``.

Currently, curves expose rod body transforms and velocities. Surfaces and volumes
expose particle positions and velocities. A family does not prescribe a solver or
storage layout. Particle-based curves and solver-owned MPM data are not supported
yet. There is no simulation-representation filter in this version.

Groups and worlds
-----------------

A group is one addressable object, such as a cable, cloth, or soft toy. Native
builder calls and USD imports record groups in the same way. Explicit labels
support application lookup; otherwise native objects get names such as
``surface_0``. USD objects use their prim paths. Labels can repeat.

Each view orders groups by world, then by their order in the model.
``world_starts`` partitions this flat group axis, including empty worlds.
``world_ids`` identifies the world of each row. Zero, one, or several groups may
match in each world. Global groups (world ``-1``) cannot share a view with per-world
groups.

``ranges(kind)`` returns each group's ``[start, end)`` range into the simulation
arrays. ``starts(kind)`` returns device-side starts; treat this array as read-only.
``elements_per_group(kind)`` returns a common element count or raises when the
sizes differ. Raw ranges remain available for groups with different sizes.

Recording labels does not prevent fixed-joint collapse. A curve group is omitted
with a warning if collapse removes one of its bodies or joints. Preserve required
joints using :meth:`~newton.ModelBuilder.collapse_fixed_joints` with
``joints_to_keep`` when complete curve access is needed.

Reading state
-------------

.. code-block:: python

    transforms = curves.get_body_transforms(state)
    positions = surfaces.get_particle_positions(state)

Batched results have shape ``(group_count, elements_per_group)``. The selected
groups must have equal counts for the requested element kind. Getters accept
either a :class:`~newton.Model` for initial values or a :class:`~newton.State`.

Regular layouts return a view into the source arrays, which may have gaps between
rows. Irregular layouts fill a reusable buffer. Do not depend on modifying a
getter result to change the state: use setters instead. Copy a result if it must
remain unchanged, for example when saving reset values. Another read may
overwrite a reusable buffer, and later state changes affect a zero-copy result.

Writing state
-------------

.. code-block:: python

    # Write group 2 from input row 5, and group 0 from input row 1.
    surfaces.set_particle_positions(
        state,
        reset_positions,
        group_indices=[2, 0],
        source_indices=[5, 1],
    )

``group_indices`` selects destination rows in the view, not model-global object
IDs. These rows coincide with world IDs only when exactly one group is selected
in every world. ``source_indices`` selects rows from the input values. Without
it, values contain one compact row per destination. Omitting ``group_indices``
writes all groups. Other groups are untouched.

Host selectors must contain genuine integers. Destination indices must be in
range and unique; source rows may repeat. Device selectors must be one-dimensional
``int32`` arrays on the model device. Out-of-range device indices are ignored.
For duplicate device destinations, the last input row wins. If that row has an
invalid source index, the destination is left unchanged.

Setters change only the supplied model or state. Applications with two alternating
states must reset both when both should retain the reset. Model writes change
initial arrays, not states that were already created.

Construct views and warm up the operations before CUDA graph capture. Preallocate
input values and device selectors. The indexed setters and getters that reuse
staging buffers can then be captured and replayed. A later replay may use changed
values or indices without a host copy.
