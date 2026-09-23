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

Deformable objects and worlds
-----------------------------

A deformable object is represented by a collection of simulation elements.
Examples include a complete cable, cloth, or soft volume. A cable can contain
rigid segments and joints. Cloth contains
particles, triangles, and bending edges. A soft volume contains particles and
tetrahedra: particles store motion, while tetrahedra describe their connections.
The view selects the whole deformable object and provides access to its elements.

This is not an arbitrary collection of scene objects. It does not add a public
Python object wrapper. The `USD deformables proposal
<https://github.com/aousd/OpenUSD-proposals/blob/5d89c0ed46a26de92f4d3fefef3bfad6500c07ce/proposals/physics_deformables/wp_deformable_physics.md#deformable-bodies>`_
uses the term *deformable body*. Here, *deformable object* avoids confusing a
complete cable with the rigid bodies used for its segments. A USD deformable body
can also own visual and collision geometry; these views expose its recorded
simulation elements, not that entire USD hierarchy.

Native builder calls and USD imports record deformable objects in the same way. Explicit labels
support application lookup; otherwise native deformable objects get names such as
``surface_0``. USD deformable objects use their prim paths. Labels can repeat.

Each view orders selected deformable objects by world, then by their order in the model.
For example, suppose deformable objects are added in this order within each world:

.. code-block:: text

    World 0: cloth, cable
    World 1: cloth, cable_0, cable_1, volume, cable_2
    World 2: cloth, cable

A curve view selecting ``cable*`` skips the cloth and volume. They do not create
gaps in the selected list:

.. code-block:: python

    cables = DeformableCurveView(model, "cable*")
    print(cables.labels)
    # ["cable", "cable_0", "cable_1", "cable_2", "cable"]

    world_ids = cables.world_ids.numpy().tolist()
    print(world_ids)
    # [0, 1, 1, 1, 2]

    deformable_object_index = 2  # The third selected cable; Python indices start at 0.
    print(cables.labels[deformable_object_index])  # "cable_1"
    print(world_ids[deformable_object_index])      # 1

``world_ids`` has one entry per selected deformable object. Here, ``"cable_1"`` has
deformable object index 2 and belongs to world 1. Index 3 refers to ``"cable_2"``.

To find all selected cables in a world, use ``deformable_object_ranges()``:

.. code-block:: python

    deformable_object_ranges = cables.deformable_object_ranges()
    print(deformable_object_ranges)
    # [(0, 1), (1, 4), (4, 5)]

    world_id = 1
    start, end = deformable_object_ranges[world_id]
    print(cables.labels[start:end])
    # ["cable_0", "cable_1", "cable_2"]

There is one ``(start, end)`` pair per world. The end is excluded. World 1 has
selected deformable object indices 1, 2, and 3. These are positions in the view, not body or
particle indices. A world with no matches has equal start and end values, so it
remains represented.

The same ranges are available on the device as a compact boundaries array:

.. code-block:: python

    deformable_object_boundaries = cables.deformable_object_boundaries.numpy().tolist()
    print(deformable_object_boundaries)
    # [0, 1, 4, 5]

    world_id = 1
    start = deformable_object_boundaries[world_id]      # 1
    end = deformable_object_boundaries[world_id + 1]    # 4
    print(cables.labels[start:end])
    # ["cable_0", "cable_1", "cable_2"]

The final 5 marks the end of the selected list. Three worlds need four boundaries.
``world_ids`` and ``deformable_object_boundaries`` are Warp arrays on the model's device;
the ``numpy()`` calls above only make their values easy to inspect in Python.
Treat both arrays as read-only.

Global deformable objects (world ``-1``) cannot share a view with deformable
objects assigned to model worlds. A global-only view has one range, ``(0, count)``,
and all its ``world_ids`` are ``-1``. That single range is not indexed by a model
world ID.

Simulation element ranges answer a different question: which bodies or particles
belong to one selected deformable object?

.. code-block:: python

    deformable_object_index = 2  # "cable_1"
    body_start, body_end = cables.ranges("body")[deformable_object_index]
    joint_start, joint_end = cables.ranges("joint")[deformable_object_index]

``ranges(kind)`` returns each deformable object's ``[start, end)`` range into the simulation
arrays. For example, ``body_start`` and ``body_end`` above refer to the model's
body arrays, not selected deformable object indices. There can be gaps between different
cables' body ranges even though their selected deformable object indices are consecutive.
``starts(kind)`` returns device-side starts; treat this array as read-only.
``elements_per_deformable_object(kind)`` returns a common element count or raises when the
sizes differ. Raw ranges remain available for deformable objects with different sizes.

Recording labels does not prevent fixed-joint collapse. A deformable curve is omitted
with a warning if collapse removes one of its bodies or joints. Preserve required
joints using :meth:`~newton.ModelBuilder.collapse_fixed_joints` with
``joints_to_keep`` when complete curve access is needed.

Reading state
-------------

.. code-block:: python

    transforms = curves.get_body_transforms(state)
    positions = surfaces.get_particle_positions(state)

Batched results have shape ``(count, elements_per_deformable_object(kind))``. The selected
deformable objects must have equal counts for the requested element kind. Getters accept
either a :class:`~newton.Model` for initial values or a :class:`~newton.State`.

Regular layouts return a view into the source arrays, which may have gaps between
rows. Irregular layouts fill a reusable buffer. Do not depend on modifying a
getter result to change the state: use setters instead. Copy a result if it must
remain unchanged, for example when saving reset values. Another read may
overwrite a reusable buffer, and later state changes affect a zero-copy result.

Writing state
-------------

.. code-block:: python

    # Write deformable object 2 from input row 5, and deformable object 0 from input row 1.
    surfaces.set_particle_positions(
        state,
        reset_positions,
        deformable_object_indices=[2, 0],
        source_indices=[5, 1],
    )

``deformable_object_indices`` selects destination rows in the view, not model-global deformable object
IDs. These rows coincide with world IDs only when exactly one deformable object is selected
in every world. ``source_indices`` selects rows from the input values. Without
it, values contain one compact row per destination. Omitting ``deformable_object_indices``
writes every selected deformable object. Unselected deformable objects are untouched.

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

Comparison with ArticulationView
--------------------------------

The workflow is similar: select an articulation or a deformable object by label,
then access its parts. An articulation contains links and joints. A deformable
object contains the elements used to simulate it. This comparison uses three
identical worlds, each with one robot, two cables, and one cloth:

.. code-block:: python

    from newton.selection import (
        ArticulationView,
        DeformableCurveView,
        DeformableSurfaceView,
    )

    robots = ArticulationView(model, "robot*")
    cables = DeformableCurveView(model, "cable*")
    cloths = DeformableSurfaceView(model, "cloth*")

    robot_link_transforms = robots.get_link_transforms(state)
    cable_segment_transforms = cables.get_body_transforms(state)
    cloth_particle_positions = cloths.get_particle_positions(state)

Each selection must satisfy its batched-read requirements. There are important
differences:

* :class:`~newton.selection.ArticulationView` requires equal selected articulation
  counts per world, compatible element counts, and regular spacing in the model
  arrays. Deformable views allow uneven deformable object counts per world and irregular
  spacing. Batched reads still require equal counts of the requested element kind.
* ``articulation_ids`` contains model articulation IDs. ``deformable_object_indices``
  contains positions in the chosen view, not model-global IDs.
* Articulation setters use Boolean masks. Deformable setters use destination
  indices and optional independent source rows.
* :meth:`~newton.selection.ArticulationView.get_attribute` is a generic entry
  point. Deformable views currently provide the specific state getters shown above,
  plus element ranges. They do not yet expose a public ``get_attribute()``.

For example, these calls reset velocities in world 1 of a three-world model.
``robot_velocities`` has the full shape returned by ``get_dof_velocities()``.
``cable_velocities`` has one compact row for each destination cable:

.. code-block:: python

    import warp as wp

    world_id = 1
    world_mask = wp.array([False, True, False], dtype=bool, device=model.device)
    robot_velocities = wp.zeros(
        robots.get_dof_velocities(state).shape, dtype=float, device=model.device
    )
    robots.set_dof_velocities(state, robot_velocities, mask=world_mask)
    robots.eval_fk(state, mask=world_mask)

    start, end = cables.deformable_object_ranges()[world_id]
    destination_objects = wp.array(
        list(range(start, end)), dtype=wp.int32, device=model.device
    )
    cable_velocities = wp.zeros(
        (end - start, cables.bodies_per_deformable_object),
        dtype=wp.spatial_vector,
        device=model.device,
    )
    cables.set_body_velocities(
        state,
        cable_velocities,
        deformable_object_indices=destination_objects,
    )

The robot call writes joint velocities; forward kinematics updates its links.
The cable call writes segment velocities directly. These are different state
fields, not interchangeable operations. Prepare buffers outside graph capture.
Both examples target one state; applications with alternating states must manage
resetting both.

See the runnable examples ``python -m newton.examples selection_articulations``
and ``python -m newton.examples selection_deformables`` for complete scenes.
