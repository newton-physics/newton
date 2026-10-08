.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _solver_observables:

Solver Observables
==================

.. experimental::

   The solver observable API may change while additional solvers and observable
   categories are migrated to it.

Quantities produced by a solver but not required to advance simulation belong
in :class:`newton.solvers.SolverObservables`, separately from
:class:`~newton.State` and :class:`~newton.Contacts`. Request only the arrays
an application needs by composing :class:`newton.solvers.SolverObservableFlags`
members in a set:

.. code-block:: python

   from newton.solvers import SolverObservableFlags, SolverMuJoCo

   solver = SolverMuJoCo(model)
   observables = solver.observables(
       {
           SolverObservableFlags.BODY_QDD,
           SolverObservableFlags.BODY_PARENT_F,
       }
   )

   solver.step(state_in, state_out, control, contacts, dt, observables=observables)
   acceleration = observables.body_qdd
   parent_wrench = observables.body_parent_f

Allocate an observable container once and reuse it across steps. The container is
owned by the solver instance that allocated it. For contact-indexed observables,
construct a :class:`~newton.CollisionPipeline` first. It publishes the resolved
rigid and soft contact capacities on the model; the solver allocates from those
capacities without needing a :class:`~newton.Contacts` instance:

.. code-block:: python

   pipeline = newton.CollisionPipeline(model)
   solver = newton.solvers.SolverXPBD(model)
   observables = solver.observables({SolverObservableFlags.CONTACT_F})
   contacts = pipeline.contacts()

   pipeline.collide(state_in, contacts)
   solver.step(state_in, state_out, control, contacts, dt, observables=observables)
   contact_force = observables.contact_f

All requested arrays are allocated when ``observables()`` returns. An unrequested
field is ``None``; a requested field with zero capacity is an empty array.
``Model.rigid_contact_max`` and ``Model.soft_contact_max`` use ``None`` for
uninitialized capacities and nonnegative integers for resolved capacities.
Requesting contact observables before pipeline construction raises an error.
Body-only observables do not require a pipeline.

The live contact counts do not determine allocation sizes. ``contact_f`` has
``model.rigid_contact_max + model.soft_contact_max`` entries, with rigid slots
followed by soft slots. Kernels use the contact counts to process valid entries.
Allocation freezes the model's contact capacities: later changes, including a
replacement pipeline with different capacities, are rejected. Configure or
rebuild pipelines before requesting contact-indexed observables.

The first solver step or consumer read binds the observable container to its
contact storage after validating device and both capacities. Newly allocated
forces are zero and can be read by a sensor or viewer before the first step.
Custom consumers can use :meth:`SolverObservables.bind_contacts()
<newton.solvers.SolverObservables.bind_contacts>` for the same validation.
Subsequent steps and consumers must use that same storage. Allocate observables and
contacts before graph capture; neither ordinary nor conditional graph execution
needs deferred observable allocation. Other solver scratch buffers may still need
their usual warmup.

Native collision backends
^^^^^^^^^^^^^^^^^^^^^^^^^

With MuJoCo's internal collision detection, construct the solver first and size
the pipeline for the backend's export capacity:

.. code-block:: python

   solver = newton.solvers.SolverMuJoCo(model)
   pipeline = newton.CollisionPipeline(
       model, rigid_contact_max=solver.get_max_contact_count(), soft_contact_max=0
   )
   observables = solver.observables({SolverObservableFlags.CONTACT_F})
   contacts = pipeline.contacts()
   solver.step(state_in, state_out, control, contacts, dt, observables=observables)

There is no ``pipeline.collide()`` call in this mode: the solver fills contact
geometry and forces. Native Kamino similarly seeds the model's rigid capacity
when constructed before the pipeline. Both backends reject insufficient
pipeline capacity during contact-observable allocation instead of resizing arrays
inside a step.

A solver advertises available entries through
:attr:`~newton.solvers.SolverBase.supported_observable_flags` and rejects an
unsupported request during allocation. Passing an observable container to a
different solver, or using contact observables with different contact storage, is
also rejected.

Standard observables
--------------------

.. list-table::
   :header-rows: 1
   :widths: 29 40 31

   * - Flag and field
     - Description
     - Solvers
   * - ``BODY_QDD`` / ``observables.body_qdd``
     - Rigid-body center-of-mass spatial accelerations in the world frame
     - :class:`~newton.solvers.SolverMuJoCo` with MuJoCo Warp and
       :class:`~newton.solvers.SolverKamino`
   * - ``BODY_PARENT_F`` / ``observables.body_parent_f``
     - Incoming parent-joint wrenches on rigid bodies
     - :class:`~newton.solvers.SolverMuJoCo` with MuJoCo Warp,
       :class:`~newton.solvers.SolverFeatherstone`, and
       :class:`~newton.solvers.SolverXPBD`
   * - ``CONTACT_F`` / ``observables.contact_f``
     - Contact spatial forces aligned with the bound contacts
     - :class:`~newton.solvers.SolverMuJoCo` with MuJoCo Warp,
       :class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverVBD`, and
       :class:`~newton.solvers.SolverKamino`

:class:`~newton.solvers.SolverKamino` computes acceleration as the discrete
step-average ``(body_qd_out - body_qd_in) / dt``. Across an impact, this includes
the velocity impulse divided by ``dt``. MuJoCo Warp computes ``BODY_QDD`` and
``BODY_PARENT_F`` even with ``disable_sensors=True`` by enabling the
post-constraint RNE stage independently of sensors.
The native MuJoCo CPU backend (``use_mujoco_cpu=True``) supports only
``SolverMuJoCo.ObservableFlags.QFRC_ACTUATOR``; body and contact observable
requests are rejected. MuJoCo Warp supports body observables on both CPU and GPU.

Kamino exports contact points in the step's input body frames and world-frame
wrenches about the input centers of mass.
Use the input state when consuming those contacts, including for native contact
detection. In-place callers must preserve the input poses separately if a
consumer needs to transform the contact points back to world coordinates.

:class:`~newton.solvers.experimental.coupled.SolverCoupled` exposes a body
observable when every entry that owns bodies supports that flag. It allocates an
entry-local container for each sub-solver and gathers owned rows into parent
model order. Contact observables are not exposed by the coupled wrapper because
filtered entry contacts require an explicit contact-index remapping contract.

Solver-specific observables
---------------------------

Solvers can derive their container from
:class:`~newton.solvers.SolverObservables` and define a separate observable enum. For
example, MuJoCo adds ``SolverMuJoCo.ObservableFlags.QFRC_ACTUATOR`` and returns a
``SolverMuJoCo.Observables`` instance with ``qfrc_actuator``:

.. code-block:: python

   solver = newton.solvers.SolverMuJoCo(model)
   observables = solver.observables(
       {
           newton.solvers.SolverObservableFlags.BODY_QDD,
           solver.ObservableFlags.QFRC_ACTUATOR,
       }
   )

Sets may contain members from both enums without coordinating bit values.
Observable enums must derive directly from :class:`enum.Enum`, not
:class:`enum.IntEnum` or a string-mixin enum. Integer and string enum members
can compare equal across enum classes and silently collide in a set.

Declare each array on a ``@dataclass(eq=False)`` container using
:meth:`SolverObservables.field() <newton.solvers.SolverObservables.field>`.
The declaration associates the field with a flag, Warp dtype, and row frequency.
Flag values are opaque identifiers: they do not need to match Python field names.
Frequency means indexing domain, not how often an observable is updated.

Derived containers inherit the standard fields and any custom fields from their
base classes. There is no separate frequency map or handwritten initializer.
The generic allocator handles standard and custom arrays alike. Inherited
declarations do not imply that a solver can compute those quantities:
``SUPPORTED_OBSERVABLE_FLAGS`` remains the capability declaration, and
``solver.supported_observable_flags`` reports the effective instance capabilities.
Contact dependencies follow the declared frequency:

.. code-block:: python

   from dataclasses import dataclass
   from enum import Enum

   import warp as wp

   class CustomObservableFlags(Enum):
       CONTACT_PRESSURE = 0

   @dataclass(eq=False)
   class CustomObservables(newton.solvers.SolverObservables):
       contact_pressure: wp.array[float] | None = newton.solvers.SolverObservables.field(
           flag=CustomObservableFlags.CONTACT_PRESSURE,
           dtype=float,
           frequency=newton.Model.AttributeFrequency.CONTACT_RIGID,
       )
       """Rigid contact pressure [Pa], shape (rigid_contact_max,)."""

   class CustomSolver(newton.solvers.SolverBase):
       OBSERVABLES_TYPE = CustomObservables
       SUPPORTED_OBSERVABLE_FLAGS = frozenset({CustomObservableFlags.CONTACT_PRESSURE})

The dataclass helper defaults every declared array to ``None`` and excludes it
from constructor arguments. Only ``solver.observables(flags)`` allocates the
requested arrays, in declaration order, on the model's device with the requested
gradient setting. ``eq=False`` preserves object identity and hashing for
articulation-view caches. A derived container may redeclare a field with a new
dtype or frequency; its siblings and base class are not modified. Two fields
cannot declare the same flag. Missing declarations, duplicate flags, and missing
dataclass decorators are rejected before array allocation.

The base solver requires pipeline initialization, freezes capacities, and binds
contact storage for any requested contact frequency, including custom fields
requested without ``CONTACT_F``. Body- and joint-indexed fields do not require
a collision pipeline.

Use :meth:`~newton.solvers.SolverObservables.is_requested` in stepping code.
For example, after validating the container and advancing
the simulation, compute a custom diagnostic only when it is requested:

.. code-block:: python

   self.validate_observables(observables, contacts)
   if observables is not None and observables.is_requested(CustomObservableFlags.CONTACT_PRESSURE):
       wp.launch(
           compute_contact_pressure,
           dim=contacts.rigid_contact_max,
           inputs=[contacts.rigid_contact_count],
           outputs=[observables.contact_pressure],
           device=self.model.device,
       )

``compute_contact_pressure`` is a solver-specific kernel, not provided by this
snippet. ``observables.is_requested(flag)`` tests the request, not freshness.
The request remains true for zero-length arrays and while they are being allocated.

Customizing the factory
^^^^^^^^^^^^^^^^^^^^^^^

Most custom fields require no method override: their dtype and row frequency
fully describe allocation. The public solver lifecycle has two methods:

* :meth:`SolverBase.observables() <newton.solvers.SolverBase.observables>` creates
  the container and its requested arrays. For additional initialization,
  override ``observables(flags, *, requires_grad=None)`` and delegate declared
  array allocation to ``super().observables(...)``. Perform backend preflight
  checks before delegating, preserve the gradient option, and finish any
  auxiliary allocation before returning the container.
* :meth:`SolverBase.validate_observables() <newton.solvers.SolverBase.validate_observables>`
  checks ownership and contact storage at the start of ``step()``, before
  launching work or modifying outputs.

There are no separate public allocation or preparation hooks. Built-in solvers
share internal construction bookkeeping so failed backend setup does not freeze
contact capacities. Kamino's factory also allocates saved input poses, and the
coupled solver's factory creates entry-local containers. These are implementation
details, not additional extension points.

All factory allocations finish before graph capture. Neither ``step()`` nor
``select()`` calls the factory. The field declarations describe runtime
diagnostics and do not introduce USD-authorable model attributes.

Contact row domains
-------------------

Three experimental frequencies describe the existing contact storage layouts:

* ``CONTACT_RIGID``: ``rigid_contact_max`` rigid-rigid slots.
* ``CONTACT_SOFT``: ``soft_contact_max`` soft-rigid slots. This does not include
  the separate soft self-contact storage.
* ``CONTACT``: the packed sum of both capacities, used by ``contact_f`` for
  compatibility. The soft segment begins at ``rigid_contact_max``, not at the
  live rigid contact count.

These frequencies currently describe solver observables, not builder custom
attributes. Capacity determines allocation; live counts determine which slots
can be read. Packed storage does not guarantee that a solver produces forces for
both segments. :class:`~newton.sensors.SensorContact` currently consumes only
rigid-rigid forces, using ``rigid_contact_count`` and rigid shape endpoints.
Soft-force production and soft-contact sensor aggregation are separate future
features.

Selection
---------

:class:`~newton.selection.ArticulationView` reads frequencies from a
``SolverObservables`` source rather than requiring field registration on the
model. Standard and custom fields using supported static layouts, such as
``BODY`` and ``JOINT_DOF``, can therefore be selected directly:

.. code-block:: python

   view = newton.selection.ArticulationView(model, "robot_*")
   accelerations = view.get_attribute("body_qdd", observables)

The container must belong to the same model and the field must be allocated.
Custom string frequencies still need the model's articulation-ownership
metadata. Raw contact rows do not have stable, unique articulation ownership:
their order changes during collision detection and their endpoints can belong
to different articulations. The view rejects contact frequencies. Filter using
contact endpoints or reduce forces to a ``BODY``-frequency field before using
an articulation view; automatic contact filtering and reduction are not provided.

Sensors
-------

Solver-dependent sensors expose a composable ``solver_observable_flags`` set. A
caller can union the requirements of multiple consumers, allocate one
container, and pass it through the step:

.. code-block:: python

   imu = newton.sensors.SensorIMU(model, sites="imu_*", request_state_attributes=False)
   contact_sensor = newton.sensors.SensorContact(
       model, sensing_shapes="foot_*", request_contact_attributes=False
   )

   flags = imu.solver_observable_flags | contact_sensor.solver_observable_flags
   # Construct a pipeline with a compatible capacity before requesting contacts.
   observables = solver.observables(flags)

   solver.step(state_in, state_out, control, contacts, dt, observables=observables)
   imu.update(state_out, observables=observables)
   contact_sensor.update(state_in, contacts, observables=observables)

The viewer follows the same pattern:

.. code-block:: python

   viewer.log_contacts(contacts, state_in, observables=observables)

Substep scheduling
------------------

Allocate the union of needed observables once, then use
:meth:`~newton.solvers.SolverObservables.select` to create reusable subsets.
Each subset has the same concrete container type and shares the selected array
objects, including gradients, with the original container. Omitted fields are
``None`` in the subset, while the original container and its arrays are unchanged.
The subset's read-only ``flags`` and ``is_requested(flag)`` describe only its
selected fields. No observable arrays are allocated or copied by ``select()``.

For example, an IMU may need acceleration every substep, while a contact sensor
only needs the final substep's forces. After configuring the solver, collision
pipeline, and sensors as above:

.. code-block:: python

   flags = imu.solver_observable_flags | contact_sensor.solver_observable_flags
   observables = solver.observables(flags)
   every_substep = observables.select(imu.solver_observable_flags)
   substep_dt = frame_dt / num_substeps

   for substep in range(num_substeps):
       is_last = substep == num_substeps - 1
       requested = observables if is_last else every_substep
       # Run the usual collision update here if using external collision detection.
       solver.step(state_in, state_out, control, contacts, substep_dt, observables=requested)
       state_in, state_out = state_out, state_in
       imu.update(state_in, observables=every_substep)

   contact_sensor.update(state_in, contacts, observables=observables)

If all fields are needed only at the end, pass ``observables=None`` on earlier
substeps. ``observables.select(set())`` also requests no diagnostics.
Selecting a field absent from the source raises :class:`ValueError`; selecting
an existing subset can only narrow it. Select from the original container to
create a different combination. Do not temporarily replace fields with ``None``.

The same selection mechanism handles standard and custom flags. Simple derived
containers inherit it unchanged. Containers with nested observable containers
should override ``select()``, call ``super()``, and select their children in the
returned object. :class:`~newton.solvers.experimental.coupled.SolverCoupled`
does this for its entry-local containers. Other custom metadata is shallow-copied.
Contact-indexed subsets share one contact binding with their source and siblings,
even if they are created before the first step. A body-only subset needs no
contact binding even when the original container includes contact fields.

Create selections before graph capture and reuse them. A fixed Python substep
schedule is captured with the graph; changing a Python selection afterward does
not change an existing graph. Device-driven choices require graph control flow
with the desired selections captured in its branches.

Skipped arrays retain their previous values. ``is_requested()`` does not mean
those values were updated, and mixed-rate containers do not represent a single
time snapshot. Consume sensors only after their required fields have been
produced, before reusing the corresponding state or contact geometry. Last-substep
forces are not frame-averaged forces; averaging or integration requires sampling
the relevant substeps. Contact rows may change between collision updates.

Selection controls observable writes and optional diagnostic work, not calculations
needed by the dynamics or other requested fields. Legacy extended attributes
can still request their own work. MuJoCo currently keeps its post-constraint RNE
stage enabled after its first request, so omitting those observables does not
necessarily avoid that internal computation on subsequent steps.

.. _vbd_contact_forces:

Contact forces from SolverVBD
-----------------------------

:class:`~newton.solvers.SolverVBD` exports one wrench per contact record it solves -- body-body
contacts, and rigid-soft particle, edge, and face records against rigid shapes -- through
:attr:`~newton.solvers.SolverObservables.contact_f`:

1. Create the :class:`~newton.CollisionPipeline` to establish contact capacities, then request
   ``SolverObservableFlags.CONTACT_F`` with ``solver.observables()``. This allocates the entire
   force array before stepping or graph capture; no contact attributes need to be requested.
2. Each frame, run collision detection and pass the container to ``solver.step(...,
   observables=observables)``. The step evaluates the wrenches directly into the array only
   when ``CONTACT_F`` is selected, without changing the simulation result. Omit ``observables``
   or pass a selection without ``CONTACT_F`` to skip reporting on earlier substeps.
3. Rows ``[0, rigid_contact_max)`` hold the body-body contacts (``i < rigid_contact_count``) when
   VBD integrates the rigid bodies: the force on body 0 by body 1 with its torque about body 0's
   center of mass, the convention shared by every solver writing ``contact_f``. With
   ``integrate_with_external_rigid_solver=True``, VBD's rigid rows are zero: request and consume
   the external solver's forces through its own observable container. Rows
   ``rigid_contact_max + i`` for ``i < soft_contact_count`` hold the rigid-soft records. Rows
   past either active count are zero.

Each soft row is expressed in world frame. Its first three entries are the force [N] exerted on
the contacted shape's body (``model.shape_body[contacts.soft_contact_shape[i]]``, ``-1`` for a
static shape) by the soft feature; its last three entries are the torque [N·m] of that force about
the body's center of mass, or about the world origin for a static shape. The force acts at the
shape-side contact point (``soft_contact_body_pos`` mapped to world space). Negate the force to get
the force on the soft contact point, and distribute it to the record's particles with
``soft_contact_barycentric``. All values are the solver's own contact law evaluated once at the
final configuration of the step -- the forces the last iteration balanced -- not time-step
averages; records without penetration are zero.

For soft contacts this is the penalty law at the final per-contact stiffness, damping while
the contact point approaches the surface, and regularized Coulomb friction on the step's slip,
bounded by the elastic normal load. Body-body contacts use the compliant ALM or legacy AVBD
contact law with its multipliers. Both evaluations retain the step-start pose history used
by the iterations.

Soft self-contact forces are not exported. :class:`~newton.sensors.SensorContact` reads the
rigid-contact rows, so it reports VBD's body-body contacts, but it does not read the soft rows.
Construct it with ``request_contact_attributes=False`` and pass the container as
``sensor.update(state_out, contacts, observables=observables)``.

.. testcode::

   import numpy as np
   import warp as wp
   import newton
   from newton.solvers import SolverObservableFlags

   builder = newton.ModelBuilder()
   builder.add_ground_plane()
   builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.045), vel=wp.vec3(0.0), mass=1.0, radius=0.05)
   ball = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.099), wp.quat_identity()))
   ball_shape = builder.add_shape_sphere(ball, radius=0.1)
   builder.color()
   model = builder.finalize()

   pipeline = newton.CollisionPipeline(model)
   contacts = pipeline.contacts()
   solver = newton.solvers.SolverVBD(model)
   observables = solver.observables({SolverObservableFlags.CONTACT_F})
   state_in, state_out = model.state(), model.state()

   pipeline.collide(state_in, contacts)
   solver.step(state_in, state_out, None, contacts, dt=1.0 / 60.0, observables=observables)

   # Body-body rows: force on body 0 by body 1; flip the sign of rows where the ball is shape 1.
   n_rigid = int(contacts.rigid_contact_count.numpy()[0])
   rigid = observables.contact_f.numpy()[:n_rigid]
   shape0 = contacts.rigid_contact_shape0.numpy()[:n_rigid]
   sign = np.where(shape0 == ball_shape, 1.0, -1.0)
   force_on_ball = (sign[:, None] * rigid[:, :3]).sum(axis=0)  # [N], world frame

   # Soft rows follow the rigid rows.
   n_soft = int(contacts.soft_contact_count.numpy()[0])
   start = contacts.rigid_contact_max
   wrenches = observables.contact_f.numpy()[start : start + n_soft]
   force_on_shape = wrenches[:, :3]  # [N], on the ground (static shape), world frame
   torque = wrenches[:, 3:]  # [N·m], about the world origin for the static ground

   # Reaction on the soft side: negate and distribute by barycentric weights.
   corners = contacts.soft_contact_indices.numpy()[:n_soft]
   weights = contacts.soft_contact_barycentric.numpy()[:n_soft]
   particle_force = np.zeros((model.particle_count, 3))
   for row in range(n_soft):
       for corner, weight in zip(corners[row], weights[row]):
           if corner >= 0:
               particle_force[corner] -= weight * force_on_shape[row]

   print(n_rigid > 0, force_on_ball[2] > 0.0, n_soft, force_on_shape[0, 2] < 0.0, particle_force[0, 2] > 0.0)

.. testoutput::

   True True 1 True True

Deprecated extended attributes
------------------------------

.. deprecated:: 1.7

   Request solver-produced arrays from the solver instead of extending
   ``State`` or ``Contacts``.

The following compatibility paths remain available for a deprecation period:

.. list-table::
   :header-rows: 1
   :widths: 37 63

   * - Deprecated destination
     - Replacement
   * - ``State.body_qdd``
     - ``SolverObservableFlags.BODY_QDD`` and ``observables.body_qdd``
   * - ``State.body_parent_f``
     - ``SolverObservableFlags.BODY_PARENT_F`` and ``observables.body_parent_f``
   * - ``State.mujoco.qfrc_actuator``
     - ``SolverMuJoCo.ObservableFlags.QFRC_ACTUATOR`` and
       ``observables.qfrc_actuator``
   * - ``Contacts.force``
     - ``SolverObservableFlags.CONTACT_F`` and ``observables.contact_f``
   * - ``Model.request_state_attributes()`` and
       ``ModelBuilder.request_state_attributes()``
     - ``solver.observables({...})``
   * - ``Model.request_contact_attributes()`` and
       ``ModelBuilder.request_contact_attributes()``
     - ``solver.observables({...})``
   * - ``solver.update_contacts()``
     - Pass contact observables to ``solver.step(..., observables=observables)``

The request methods and legacy force export via ``update_contacts()`` emit
:class:`DeprecationWarning`. MuJoCo and Kamino keep geometry-only ``update_contacts()``
supported without a warning when ``contacts.force`` is ``None``.
``SensorIMU`` and ``SensorContact`` retain their legacy
``request_state_attributes=True`` and ``request_contact_attributes=True`` defaults
during the deprecation period; explicitly pass ``False`` when using solver observables.
Setting either option to ``True`` is deprecated in Newton 1.7 and emits one
sensor-specific warning at the caller's location. Passing ``False`` does not
emit a warning or request legacy fields.

``Contacts.EXTENDED_ATTRIBUTES`` and direct ``requested_attributes={"force"}``
remain compatibility APIs. New integrations should not allocate
``Contacts.force``.

``State.EXTENDED_ATTRIBUTES`` remains a compatibility registry for the three
deprecated state destinations listed above.

This deprecation does not affect custom attributes registered with
:meth:`ModelBuilder.add_custom_attribute <newton.ModelBuilder.add_custom_attribute>`.
Existing custom model, state, and control data remain supported; the migration
only covers built-in solver-produced diagnostics. The new ``CONTACT*`` frequencies
are reserved for solver observables: ``ModelBuilder.add_custom_attribute()`` rejects
them because builder finalization does not allocate dynamic contact rows.
