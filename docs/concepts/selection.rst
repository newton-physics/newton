.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _selection-api:

Selection API
=============

Selection views find articulations or deformable objects by label in a finalized
model. They arrange the selected data into arrays for reading and updating state,
without requiring you to track each joint, body, or particle index.

Choose a view
-------------

The views are available from :mod:`newton.selection`:

* :class:`~newton.selection.ArticulationView` selects articulations and accesses
  their joints, links, shapes, and attributes.
* :class:`~newton.selection.DeformableCurveView` selects curves. It currently
  accesses the body transforms and velocities of rods.
* :class:`~newton.selection.DeformableSurfaceView` selects surfaces, such as
  cloth, and accesses their particle positions and velocities.
* :class:`~newton.selection.DeformableVolumeView` selects volumes, such as
  tetrahedral soft bodies, and accesses their particle positions and velocities.

.. experimental::

   The three deformable view classes may change while the API is developed.

Patterns can be exact labels, globs such as ``"cloth*"``, lists of globs, or
compiled regular expressions. See :ref:`label-matching` for matching rules.
A view's selection is fixed after construction. A Boolean mask chooses which
of those matches a later write updates; it does not change the selection.

Select, read, and update
------------------------

This complete example creates one cloth in each of three worlds. It gives only
the middle cloth an upward velocity of 1 m/s. Each cloth has four particles.

.. testcode:: selection-overview

    import warp as wp

    import newton
    from newton.selection import DeformableSurfaceView

    cloth = newton.ModelBuilder()
    cloth.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1, dim_y=1,
        cell_x=0.5, cell_y=0.5,
        mass=0.1,
        label="cloth",
    )
    builder = newton.ModelBuilder()
    builder.replicate(cloth, copy_count=3)
    model = builder.finalize(device="cpu")
    state = model.state()

    cloths = DeformableSurfaceView(model, "cloth")
    velocities = wp.clone(cloths.get_particle_velocities(state))
    velocities.fill_(wp.vec3(0.0, 0.0, 1.0))
    mask_cloths = wp.array([False, True, False], dtype=bool, device=model.device)
    cloths.set_particle_velocities(state, velocities, mask=mask_cloths)

    updated = cloths.get_particle_velocities(state)
    print(updated.shape)
    print(updated.numpy()[:, 0, 2].tolist())

.. testoutput:: selection-overview

    (3, 4)
    [0.0, 1.0, 0.0]

The Warp array has one row per cloth and one ``wp.vec3`` per particle. The
second output shows the first particle's vertical velocity in each cloth.
The other two cloths are unchanged.

Clone a getter result before editing it: it may share the state's storage or
use a reusable read buffer. Use the setter to write the changes. Values retain
the full view shape, even when the mask selects only one row. Getters can also
read a model's initial state arrays, for example to save reset values.

.. _selection-particle-properties:

Surface and volume views can also snapshot inverse masses and flags from the
Model. These scalar properties have the same object and particle layout:

.. testcode:: selection-overview

    inverse_masses_default = wp.clone(cloths.get_particle_inv_masses(model))
    flags_default = wp.clone(cloths.get_particle_flags(model))
    print(inverse_masses_default.shape, flags_default.shape)

.. testoutput:: selection-overview

    (3, 4) (3, 4)

.. _selection-array-shapes:

Array and mask shapes
---------------------

Articulation data is arranged by world, then by selected articulation.
Deformable data has one row per selected deformable object, ordered by world.
For example:

.. list-table::
   :header-rows: 1
   :widths: 38 30 32

   * - Selection
     - Example Warp data shape
     - State setter mask shape
   * - Three worlds, two robots per world, four DOFs per robot
     - DOF velocities: ``(3, 2, 4)``
     - ``(3,)`` or ``(3, 2)``
   * - Three worlds with one, three, and one cloth, four particles per cloth
     - Particle velocities: ``(5, 4)`` of ``wp.vec3``
     - ``(5,)``

For articulations, a ``(3,)`` mask selects whole worlds. A ``(3, 2)`` mask can
select either robot independently in each world. For deformables, each mask
entry selects one deformable object. In the cloth example from the table, rows
1, 2, and 3 belong to world 1; a single mask entry does not select that whole world.
Use :ref:`deformable-selection-worlds` to map worlds to selected rows.

Deformable views allow different object counts per world and gaps between objects
in the model arrays. They keep empty worlds in their object ranges, while
``worlds`` and ``worlds_device`` map each selected deformable object to its model world.
Batched reads and writes still require equal counts of the requested element
kind, such as particles per cloth.

For articulation layout requirements and world mapping, see
:class:`~newton.selection.ArticulationView`.

Further reading
---------------

* :ref:`Articulations` covers articulation construction and selection, joint and
  link filters, and partial layouts. The
  :class:`~newton.selection.ArticulationView` reference lists its state and
  attribute methods.
* :ref:`deformable-selection` covers object ranges, different sizes, reset masks,
  reusable buffers, CUDA graph capture, and gradients.
* :ref:`deformable-objects` explains how builders and USD imports assign labels
  and record the simulation elements of each deformable object.
