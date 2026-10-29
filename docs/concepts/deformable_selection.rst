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

    import newton
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

An integer list selects indices in the family's Model label array. Indices must
be unique, in ascending order, and within that array:

.. code-block:: python

    cloths = DeformableSurfaceView(model, [0, 2])
    # Select model.surface_label[0] and model.surface_label[2].
    # They become rows 0 and 1 of this view.

Separate classes make the family clear at the call site. For example, a cloth
and a soft volume can share a label without sharing a selection:

.. code-block:: python

    cloths = DeformableSurfaceView(model, "object")
    volumes = DeformableVolumeView(model, "object")

    cloths.ranges(newton.Model.AttributeFrequency.TRIANGLE)
    volumes.ranges(newton.Model.AttributeFrequency.TETRAHEDRON)

Currently, curve views expose rod body transforms and velocities, with body and
joint ranges. Surface and volume views share particle methods: positions,
velocities, inverse masses, and flags. Both expose particle ranges, but surface
views also expose triangle and edge ranges, while volume views expose tetrahedron
ranges. A family does not prescribe a solver or storage layout. Particle-based
curves and solver-owned MPM data are not supported yet.

Read and reset state
--------------------

State getters read either a :class:`~newton.State` or the initial values stored in a
:class:`~newton.Model`:

.. code-block:: python

    transforms = curves.get_body_transforms(state)
    positions = surfaces.get_particle_positions(state)

Each row contains one selected deformable object's values. All selected
deformable objects must have the same count of the requested element kind.
For example, cloth particle reads require matching particle counts, but triangle
counts can differ.
Three cloths with four particles each produce a Warp array with
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

Read particle properties
------------------------

Surface and volume views also read :attr:`~newton.Model.particle_inv_mass` and
:attr:`~newton.Model.particle_flags`. Pass a Model, not a State. Both results have
shape ``(count, particle_count)``, with the same rows and particle
order as the position getter. Inverse masses use ``wp.float32`` in [1/kg]; flags
use ``wp.int32`` bitmasks from :class:`~newton.ParticleFlags`.

See the :ref:`particle-property example <selection-particle-properties>` for
saving independent snapshots with ``wp.clone()``. Do not edit getter results:
they may share Model storage or use a reusable buffer. Applying kinematic targets
and restoring saved properties remain application responsibilities; the views do
not provide property setters.

Update selected deformable objects
----------------------------------

A Boolean ``mask`` chooses which selected deformable objects to update.
Values always contain one row per object in view order. For a view containing
three cloths, reset only cloths 0 and 2 from their saved rows:

.. code-block:: python

    mask_cloths = wp.array([True, False, True], dtype=bool, device=model.device)
    surfaces.set_particle_positions(state, positions_default, mask=mask_cloths)
    surfaces.set_particle_velocities(state, velocities_default, mask=mask_cloths)

The middle cloth is untouched. ``mask=None`` updates the whole selection;
an all-false mask changes nothing. Values must still have their full shape.
A mask entry refers to a selected object, not a model-global ID or a particle.
It corresponds to a world only when exactly one object is selected in every
real model world. See :ref:`deformable-selection-worlds` for the mapping.

Host Boolean sequences are accepted outside capture. Numeric lists are not masks.
Device masks must have shape ``(count,)``, Boolean dtype, and the view's device.
Strided masks are supported. Setters do not modify their masks or input values.

When a reset supplies object indices, convert them into a caller-owned mask:

.. code-block:: python

    surfaces.set_mask_from_indices(mask_cloths, [2, 0, 2])
    # mask_cloths is [True, False, True]. Values stay in view order.
    surfaces.set_particle_positions(state, positions_default, mask=mask_cloths)

The helper replaces the complete mask. Repeated indices select an object once;
an empty selector clears it. Host indices must be integers within the view's
range; booleans are rejected. Device indices must be one-dimensional ``int32``
arrays on the view's device. Invalid device indices are ignored, so a
fixed-capacity selector can use ``-1`` for unused entries. No source-row
selection or compact-value mode is provided.

For one matching object in every model world, environment IDs can be used as
object indices. Verify that mapping once during setup. Then reset both active
states from the same full buffers:

.. code-block:: python

    # Setup: surfaces selects exactly one named cloth in every world.
    assert surfaces.worlds == list(range(model.world_count))
    indices_world = wp.array([0, 2], dtype=wp.int32, device=model.device)
    mask_cloths = wp.zeros(surfaces.count, dtype=bool, device=model.device)

    # Runtime: indices_world may be filled on the device before each reset.
    surfaces.set_mask_from_indices(mask_cloths, indices_world)
    for state in (state_0, state_1):
        surfaces.set_particle_positions(state, positions_default, mask=mask_cloths)
        surfaces.set_particle_velocities(state, velocities_default, mask=mask_cloths)

Missing matches, multiple objects in one world, and global objects invalidate
this shortcut. Use selected-object indices or fill the object mask directly
in those cases. A global-only view has world IDs of ``-1``.

.. _deformable-selection-worlds:

Deformable objects and worlds
-----------------------------

Each row represents a complete deformable object, such as a cable or cloth.
The view provides access to its recorded simulation elements.
See :ref:`deformable-objects` for how native calls and USD imports record these elements.

Each view orders selected deformable objects by world, then by their order in the model.
This orders the view's rows; it does not reorder the model's arrays.
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

    worlds = cables.worlds
    print(worlds)
    # [0, 1, 1, 1, 2]

    deformable_object_index = 2  # The third selected cable; Python indices start at 0.
    print(cables.labels[deformable_object_index])  # "cable_1"
    print(worlds[deformable_object_index])        # 1

``worlds`` has one entry per selected deformable object. Here, ``"cable_1"`` has
deformable object index 2 and belongs to world 1. Index 3 refers to ``"cable_2"``.

``worlds_device`` is a Warp copy of this list on the model's device:

.. code-block:: python

    print(cables.worlds_device.numpy().tolist())
    # [0, 1, 1, 1, 2]

Both mappings are created with the view. Treat them as read-only: editing one
does not update the other or change the selection. The ``numpy()`` call above
only makes the device values easy to inspect in Python. Use the Python list for
host-side setup and the Warp array in device kernels.

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

``count_per_world`` includes every model world, even those without a match.
It is ``None`` when their counts differ. For example, suppose worlds 0 and 2
each contain two selected cloths, and world 1 has none:

.. code-block:: python

    cloth = DeformableSurfaceView(model, "cloth")
    print(cloth.count)                       # 4

    # Python list on the host: one world ID per selected cloth object.
    print(cloth.worlds)                      # [0, 0, 2, 2]

    # Warp array on model.device: the same IDs, in the same order.
    # Use .numpy() here only to inspect the values on the host.
    print(cloth.worlds_device.numpy().tolist())  # [0, 0, 2, 2]

    print(cloth.deformable_object_ranges())  # [(0, 2), (2, 2), (2, 4)]
    print(cloth.count_per_world)             # None: the counts are [2, 0, 2]

    world_id = 1
    start, end = cloth.deformable_object_ranges()[world_id]
    print(cloth.labels[start:end])           # []

    # If worlds 0 and 2 also each have two matching robot articulations:
    from newton.selection import ArticulationView

    robots = ArticulationView(model, "robot*", homogeneous=False)
    # Warp array: one ID per selected world, not per selected object.
    print(robots.world_ids.numpy().tolist())  # [0, 2]

Deformable ``worlds`` and articulation ``world_ids`` have different meanings.
``worlds`` and its device copy, ``worlds_device``, have one entry per selected
deformable object, so world IDs can repeat. ``ArticulationView.world_ids`` has
one entry per selected world. The difference is not just host versus device
storage.

For the earlier cable selection, the ranges are also available on the device
as a compact boundaries array:

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
``deformable_object_boundaries`` is also a read-only Warp array on the model's device.

Global deformable objects (world ``-1``) cannot share a view with deformable
objects assigned to model worlds. A global-only view has one range, ``(0, count)``,
and all its ``worlds`` and ``worlds_device`` entries are ``-1``. That single range is not indexed by a model
world ID. Its ``count_per_world`` equals ``count``.

Simulation element ranges answer a different question: which bodies or particles
belong to one selected deformable object?

.. code-block:: python

    deformable_object_index = 2  # "cable_1"
    body_start, body_end = cables.ranges(newton.Model.AttributeFrequency.BODY)[deformable_object_index]
    joint_start, joint_end = cables.ranges(newton.Model.AttributeFrequency.JOINT)[deformable_object_index]

``ranges(kind)`` returns each deformable object's ``[start, end)`` range into the simulation
arrays. For example, ``body_start`` and ``body_end`` above refer to the model's
body arrays, not selected deformable object indices. There can be gaps between different
cables' body ranges even though their selected deformable object indices are consecutive.
``starts(kind)`` returns device-side starts; treat this array as read-only.
``element_count(kind)`` is the general count query. It returns the common count
per selected deformable object, or ``None`` when sizes differ. ``particle_count``
and ``body_count`` are convenience properties for the corresponding state
elements. Raw ranges remain available when sizes differ.

Use a :class:`~newton.Model.AttributeFrequency` member for ``kind``. Its lowercase
name is also accepted as a convenience:

.. code-block:: python

    kind = newton.Model.AttributeFrequency.PARTICLE
    cloths.element_count(kind)  # Particles per selected cloth, or None.
    cloths.particle_count      # The same count through the convenience property.
    cloths.ranges(kind)        # Each cloth's [start, end) particle range.
    cloths.ranges("particle")  # Equivalent string form.

For cable root joints and later attachments, see :ref:`deformable-objects-curve-joints`.
For changes to the ranges after collapse, see :ref:`deformable-objects-collapse`.

Different deformable object sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For differently sized deformable objects, use separate views for batched state access:

.. code-block:: python

    cables_short = DeformableCurveView(model, "cable_10_segments*")
    cables_long = DeformableCurveView(model, "cable_30_segments*")
    transforms_short = cables_short.get_body_transforms(state)
    transforms_long = cables_long.get_body_transforms(state)

These labels are an application naming choice, not a size filter. The view
checks the actual element counts, including any changes from fixed-joint collapse.
A broader view can still expose raw ranges. Masking out a differently sized
deformable object does not make batched access valid.

Inspect all deformable objects
------------------------------

Use the :ref:`public model arrays <deformable-objects-model-arrays>` to inspect
all deformable objects without a selection pattern, including global objects.
Read this metadata during setup, outside simulation steps or graph capture.
A view contains only its matches; its row indices are not model-global indices.

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

See :ref:`selection-array-shapes` for the data and mask layouts. Both views use
Boolean masks with full-sized input arrays. Deformable masks have one entry per
selected deformable object, in the order shown by ``view.labels``. Selecting a
world with several selected deformable objects requires several mask entries.
ArticulationView also provides :meth:`~newton.selection.ArticulationView.get_attribute`;
generic attribute access remains a follow-up for deformable views.

These calls reset velocities in world 1.
``robot_velocities`` has the full shape returned by ``get_dof_velocities()``.
``cable_velocities`` also has the full shape returned by its getter:

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
    mask_cables = wp.zeros(cables.count, dtype=bool, device=model.device)
    cables.set_mask_from_indices(mask_cables, range(start, end))
    cable_velocities = wp.zeros(
        cables.get_body_velocities(state).shape,
        dtype=wp.spatial_vector,
        device=model.device,
    )
    cables.set_body_velocities(state, cable_velocities, mask=mask_cables)

The robot call writes joint velocities; forward kinematics updates its links.
The cable call writes segment velocities directly. These are different state
fields, not interchangeable operations. Prepare buffers outside graph capture.

The deformable conversion helper maps a list of selected-object indices to
mask membership. It is not the same as ArticulationView's
``get_model_articulation_mask()``, which maps a view mask to model articulation
IDs. Neither call changes the view's fixed membership.

Getter buffers
---------------

Here, *layout* means where each selected deformable object's elements sit in the
model arrays. Consider three cloths with four particles each:

* Particle starts ``[0, 8, 16]`` are evenly spaced. The getter can return a view
  into the source array without copying values. The gaps between cloths are allowed.
* Particle starts ``[0, 8, 20]`` are unevenly spaced. The getter copies the
  selected values into a reusable buffer. All three cloths still have four particles.

A change to the source array or another read can change the returned values. Use
``wp.clone()`` when a result must stay unchanged, as in the reset buffers above.
If a solver replaces a source array, call the getter again. Previously returned
results still refer to the old storage.

Each source Model or State has its own cached results. The cache does not keep
discarded states alive. Getter results still held by your code remain valid.
The view keeps its model alive, as :class:`~newton.selection.ArticulationView` does.

State write requirements
------------------------

Outside CUDA capture, setters also accept numeric host sequences and NumPy
arrays. Like :class:`~newton.selection.ArticulationView`, they reshape host inputs
in row-major order when the total number of components matches. The final NumPy
axis has length 3 for particle positions and velocities, 7 for body transforms,
and 6 for body velocities. Warp inputs must already have the required shape,
dtype, and device. Values are validated even with an all-false mask.

Use setters to change state; do not rely on editing a getter result. Setters
reject input that shares storage with the target array. Copy such input with
``wp.clone()`` first to avoid overwriting values the setter still needs to read.

Setters change only the supplied model or state. Applications with two alternating
states must reset both when both should retain the reset. Model writes change
initial arrays, not states that were already created.

State arrays must have the expected rank, dtype, and device, and cover every
selected range. Strided arrays and extra elements are allowed. These checks do
not verify model ownership: the caller must ensure that the array uses the
view's element numbering. See the getter and setter API references for full
validation rules, for example
:meth:`~newton.selection.DeformableSurfaceView.set_particle_positions`.

CUDA graph capture
------------------

Construct views and warm up operations before CUDA graph capture. Preallocate
independent values and masks. Setters allocate no temporary arrays with device
inputs, including when those inputs require gradients. Host values and masks
raise ``RuntimeError`` during capture. Getters that reuse buffers can also be
captured after their first call. Mask and value contents can change between
inference replays without rebuilding the graph.

Getter buffers first allocated during capture belong to that graph. They are
not reused by ordinary reads or other captures. Warm up before capturing to
reuse buffers without allocating inside the graph.

``set_mask_from_indices()`` also supports capture with preallocated device
indices and a mask. Host indices raise ``RuntimeError`` during capture.

A captured graph keeps using the data and gradient arrays from capture. Keep
the source models or states and getter results alive while replaying it. If those
arrays are replaced, warm up the getters and capture a new graph.

Gradients
---------

Getters and setters support ``wp.Tape``. Create the state with
``model.state(requires_grad=True)`` when gradients are needed. Reads propagate
gradients to the source state. Masked writes propagate gradients to the input values.
Inverse-mass reads also propagate gradients when the source Model has gradients
enabled. Integer flags do not have gradients.

Keep values, masks, and source arrays unchanged until ``tape.backward()`` finishes.
Use separate state buffers for successive steps and separate masks for different
selections on the same tape. Setters do not save a copy of the mask for each call.

For evenly spaced elements, getter results share the source's gradient storage.
For uneven spacing, the copy operation passes gradients back to the source.
If the source's gradient buffer or ``requires_grad`` setting changes, call the
getter again to use the current gradient storage. Zeroing gradients does not
replace the cached result.

For a captured forward/backward graph, change inputs only between complete replays.
