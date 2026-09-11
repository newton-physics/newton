.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _solver_observables:

Solver Observables
==================

.. experimental::

   The solver observable API may change while additional solvers and observable
   categories are migrated to it.

See the standalone `Solver Observables architecture guide
<../_static/solver_observables_architecture.html>`_ for diagrams of the ownership,
allocation, extension, and sensor-consumption flows.

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

The first ``solver.step(..., contacts, ..., observables=observables)`` binds the observable
container to that contact storage after validating device and both capacities.
Subsequent steps and sensors must use that same storage. Allocate observables and
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
     - :class:`~newton.solvers.SolverMuJoCo` and
       :class:`~newton.solvers.SolverKamino`
   * - ``BODY_PARENT_F`` / ``observables.body_parent_f``
     - Incoming parent-joint wrenches on rigid bodies
     - :class:`~newton.solvers.SolverMuJoCo`,
       :class:`~newton.solvers.SolverFeatherstone`, and
       :class:`~newton.solvers.SolverXPBD`
   * - ``CONTACT_F`` / ``observables.contact_f``
     - Contact spatial forces aligned with the bound contacts
     - :class:`~newton.solvers.SolverMuJoCo` with MuJoCo Warp,
       :class:`~newton.solvers.SolverXPBD`, and
       :class:`~newton.solvers.SolverKamino`

:class:`~newton.solvers.SolverKamino` computes acceleration as the discrete
step-average ``(body_qd_out - body_qd_in) / dt``. Across an impact, this includes
the velocity impulse divided by ``dt``. MuJoCo Warp requires sensors to remain
enabled when requesting ``BODY_QDD`` or ``BODY_PARENT_F``; stepping with
``disable_sensors=True`` and either observable raises an error.

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

Each flag's string value names its array field. Declare that field's row domain
in the container's ``ATTRIBUTE_FREQUENCIES`` mapping, using
:class:`~newton.Model.AttributeFrequency` or a custom string frequency. Lookup
inherits base declarations; subclasses need only list their additional fields.
Missing declarations are rejected before allocation. Frequency means indexing
domain, not how often an observable is updated.

Override ``_allocate_observables()`` and call ``super()`` to allocate inherited
arrays before custom arrays. ``SUPPORTED_OBSERVABLE_FLAGS`` is the only
capability flag set. Contact dependencies follow the declared frequency:

.. code-block:: python

   class CustomObservableFlags(Enum):
       CONTACT_PRESSURE = "contact_pressure"

   class CustomObservables(newton.solvers.SolverObservables):
       ATTRIBUTE_FREQUENCIES = {
           "contact_pressure": newton.Model.AttributeFrequency.CONTACT_RIGID,
       }

       def __init__(self, flags=()):
           super().__init__(flags)
           self.contact_pressure = None

   class CustomSolver(newton.solvers.SolverBase):
       OBSERVABLES_TYPE = CustomObservables
       SUPPORTED_OBSERVABLE_FLAGS = frozenset({CustomObservableFlags.CONTACT_PRESSURE})

       def _allocate_observables(self, observables, *, requires_grad):
           super()._allocate_observables(observables, requires_grad=requires_grad)
           if CustomObservableFlags.CONTACT_PRESSURE in observables:
               observables.contact_pressure = wp.zeros(
                   self.model.rigid_contact_max,
                   dtype=float,
                   device=self.model.device,
                   requires_grad=requires_grad,
               )

The snippet assumes ``from enum import Enum`` and ``import warp as wp``.
The base solver requires pipeline initialization, freezes capacities, and binds
contact storage for any requested contact frequency, including custom fields
requested without ``CONTACT_F``. Body- and joint-indexed fields do not require
a collision pipeline.

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

   imu = newton.sensors.SensorIMU(model, sites="imu_*")
   contact_sensor = newton.sensors.SensorContact(model, sensing_shapes="foot_*")

   flags = imu.solver_observable_flags | contact_sensor.solver_observable_flags
   # Construct a pipeline with a compatible capacity before requesting contacts.
   observables = solver.observables(flags)

   solver.step(state_in, state_out, control, contacts, dt, observables=observables)
   imu.update(state_out, solver_observables=observables)
   contact_sensor.update(state_out, contacts, solver_observables=observables)

The viewer follows the same pattern:

.. code-block:: python

   viewer.log_contacts(contacts, state_out, solver_observables=observables)

Deprecated extended attributes
------------------------------

.. deprecated:: 1.6

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

The request methods and ``update_contacts()`` emit
:class:`DeprecationWarning`. ``SensorIMU`` and ``SensorContact`` no longer
request extended attributes by default. Set their respective
``request_state_attributes=True`` or ``request_contact_attributes=True`` only
while migrating legacy code.

``Contacts.EXTENDED_ATTRIBUTES`` and direct ``requested_attributes={"force"}``
remain compatibility APIs. New integrations should not allocate
``Contacts.force``.

``State.EXTENDED_ATTRIBUTES`` remains a compatibility registry for the three
deprecated state destinations listed above.

This deprecation does not affect custom attributes registered with
:meth:`ModelBuilder.add_custom_attribute <newton.ModelBuilder.add_custom_attribute>`.
Custom model, state, control, and contact data remain supported; the migration
only covers built-in solver-produced diagnostics.
