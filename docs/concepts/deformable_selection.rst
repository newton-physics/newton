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

Inspect all deformable objects
------------------------------

Views read the public model data described in :ref:`deformable-objects`.
Use those arrays directly when all labels, worlds, or ranges are needed,
without applying a selection pattern:

.. code-block:: python

    # Read metadata once during setup, outside simulation steps or graph capture.
    worlds = model.surface_world.numpy()
    starts = model.surface_particle_start.numpy()
    ends = model.surface_particle_end.numpy()
    for label, world, start, end in zip(
        model.surface_label, worlds, starts, ends, strict=True
    ):
        print(label, "world:", world, "particles:", end - start)

Like ``model.articulation_label`` and ``model.articulation_world``, these arrays
describe recorded objects before selection. They include global objects and
objects in uneven worlds. A view contains only its matches, ordered by world.
Its destination indices refer to that selection, not the model inventory.

Read and reset state
--------------------

Getters read either a :class:`~newton.State` or the initial values stored in a
:class:`~newton.Model`:

.. code-block:: python

    transforms = curves.get_body_transforms(state)
    positions = surfaces.get_particle_positions(state)

Each row contains one selected deformable object's values. All selected
deformable objects must have the same count of the requested element kind.
For example, three cloths with four particles each produce a Warp array with
shape ``(3, 4)`` and ``wp.vec3`` elements. Its NumPy representation has shape
``(3, 4, 3)`` because each position has three coordinates.

Save independent reset buffers during setup, then use them to reset the state:

.. code-block:: python

    import warp as wp

    positions_default = wp.clone(surfaces.get_particle_positions(model))
    velocities_default = wp.clone(surfaces.get_particle_velocities(model))

    # Later, reset positions and velocities for all selected cloths.
    surfaces.set_particle_positions(state, positions_default)
    surfaces.set_particle_velocities(state, velocities_default)

Regular layouts return a view into the source arrays, which may have gaps between
rows. Irregular layouts fill a reusable buffer. A later state change or another
read can change the returned values. Use ``wp.clone()`` when a result must stay
unchanged, as in the reset buffers above.

Use setters to change state; do not rely on editing a getter result. Setters
reject input that shares storage with the target array. Copy such input with
``wp.clone()`` first. This prevents a write from overwriting values it still
needs to read, without adding hidden copies to every setter call.

Setters change only the supplied model or state. Applications with two alternating
states must reset both when both should retain the reset. Model writes change
initial arrays, not states that were already created.

Update selected deformable objects
----------------------------------

``deformable_object_indices`` says **which selected deformable objects to change**.
``source_indices`` says **which input rows to read**. For a view containing at
least three cloths, reset only cloths 2 and 0 from their saved rows:

.. code-block:: python

    surfaces.set_particle_positions(
        state,
        positions_default,
        deformable_object_indices=[2, 0],
        source_indices=[2, 0],
    )
    surfaces.set_particle_velocities(
        state,
        velocities_default,
        deformable_object_indices=[2, 0],
        source_indices=[2, 0],
    )

The two lists need not match. With destinations ``[2, 0]`` and source rows
``[0, 2]``, the setter copies saved row 0 to cloth 2 and saved row 2 to cloth 0.
Without ``source_indices``, values contain one compact row per destination.
Omitting ``deformable_object_indices`` writes every selected deformable object.
Unselected deformable objects are untouched.

Destination indices are positions in this view, not model-global IDs. They
coincide with world IDs only when exactly one deformable object is selected in
every world. The next section explains how to look up objects by world.

Host selectors must contain genuine integers. Destination indices must be in
range and unique; source rows may repeat. Device selectors must be one-dimensional
``int32`` arrays on the model device. Out-of-range device indices are ignored.
For duplicate device destinations, the last input row wins. If that row has an
invalid source index, the destination is left unchanged.

Construct views and warm up the operations before CUDA graph capture. Preallocate
independent input values and device selectors. The indexed setters and getters
that reuse staging buffers can then be captured and replayed. A later replay may
use changed values or indices without a host copy.

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

See :ref:`deformable-objects` for how native builder calls and USD imports record
deformable objects, assign labels and worlds, and preserve their simulation ranges.

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

Joint ranges depend on how a cable was created. Native rod calls include any
free-root joints they create. For a single USD curve, the recorded joint span
excludes its root attachment. A USD prim containing several curves can include
root joints between those curves in its span. Curves welded into a shared rod
graph have empty joint ranges because those joints belong to the shared graph.
Do not infer the joint count from the segment count. See :ref:`deformable-objects`.

Recording labels does not prevent fixed-joint collapse. A deformable curve is omitted
with a warning if collapse removes one of its bodies or joints. Preserve required
joints using :meth:`~newton.ModelBuilder.collapse_fixed_joints` with
``joints_to_keep`` when complete curve access is needed.

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

The method style is similar, but the layouts serve different needs:

* ArticulationView keeps a world/articulation layout. It requires equal selected
  articulation counts per world, compatible element counts, and regular spacing
  in the model arrays. Deformable views use one flat row per selected deformable
  object. This allows uneven world counts and irregular spacing without padding.
  Batched reads still require equal counts of the requested element kind.
* Articulation setters use Boolean masks with full-sized input arrays. Deformable
  setters use destination indices and compact input rows, with optional source
  indices for reading from a larger buffer. This supports partial resets without
  requiring a full-sized array for each write.
* :meth:`~newton.selection.ArticulationView.get_attribute` is a generic entry
  point. The deformable views start with state getters, setters, and element
  ranges. Generic attribute access remains a follow-up, not a limitation of
  deformable simulation.

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
