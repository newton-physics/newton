.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _solver_outputs:

Solver Outputs
==============

.. experimental::

   The solver output API may change while additional solvers and output
   categories are migrated to it.

See the standalone `Solver Outputs architecture guide
<../_static/solver_outputs_architecture.html>`_ for diagrams of the ownership,
allocation, extension, and sensor-consumption flows.

Quantities produced by a solver but not required to advance simulation belong
in :class:`newton.solvers.SolverOutputs`, separately from
:class:`~newton.State` and :class:`~newton.Contacts`. Request only the arrays
an application needs by composing :class:`newton.solvers.SolverOutputFlags`
members in a set:

.. code-block:: python

   from newton.solvers import SolverOutputFlags, SolverMuJoCo

   solver = SolverMuJoCo(model)
   outputs = solver.outputs(
       {
           SolverOutputFlags.BODY_QDD,
           SolverOutputFlags.BODY_PARENT_F,
       }
   )

   solver.step(state_in, state_out, control, contacts, dt, outputs=outputs)
   acceleration = outputs.body_qdd
   parent_wrench = outputs.body_parent_f

Allocate an output container once and reuse it across steps. The container is
owned by the solver instance that allocated it. For contact-indexed outputs,
construct a :class:`~newton.CollisionPipeline` first. It publishes the resolved
rigid and soft contact capacities on the model; the solver allocates from those
capacities without needing a :class:`~newton.Contacts` instance:

.. code-block:: python

   pipeline = newton.CollisionPipeline(model)
   solver = newton.solvers.SolverXPBD(model)
   outputs = solver.outputs({SolverOutputFlags.CONTACT_F})
   contacts = pipeline.contacts()

   pipeline.collide(state_in, contacts)
   solver.step(state_in, state_out, control, contacts, dt, outputs=outputs)
   contact_force = outputs.contact_f

All requested arrays are allocated when ``outputs()`` returns. An unrequested
field is ``None``; a requested field with zero capacity is an empty array.
``Model.rigid_contact_max`` and ``Model.soft_contact_max`` use ``None`` for
uninitialized capacities and nonnegative integers for resolved capacities.
Requesting contact outputs before pipeline construction raises an error.
Body-only outputs do not require a pipeline.

The live contact counts do not determine allocation sizes. ``contact_f`` has
``model.rigid_contact_max + model.soft_contact_max`` entries, with rigid slots
followed by soft slots. Kernels use the contact counts to process valid entries.
Allocation freezes the model's contact capacities: later changes, including a
replacement pipeline with different capacities, are rejected. Configure or
rebuild pipelines before requesting contact-indexed outputs.

The first ``solver.step(..., contacts, ..., outputs=outputs)`` binds the output
container to that contact storage after validating device and both capacities.
Subsequent steps and sensors must use that same storage. Allocate outputs and
contacts before graph capture; neither ordinary nor conditional graph execution
needs deferred output allocation. Other solver scratch buffers may still need
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
   outputs = solver.outputs({SolverOutputFlags.CONTACT_F})
   contacts = pipeline.contacts()
   solver.step(state_in, state_out, control, contacts, dt, outputs=outputs)

There is no ``pipeline.collide()`` call in this mode: the solver fills contact
geometry and forces. Native Kamino similarly seeds the model's rigid capacity
when constructed before the pipeline. Both backends reject insufficient
pipeline capacity during contact-output allocation instead of resizing arrays
inside a step.

A solver advertises available entries through
:attr:`~newton.solvers.SolverBase.supported_output_flags` and rejects an
unsupported request during allocation. Passing an output container to a
different solver, or using contact outputs with different contact storage, is
also rejected.

Standard outputs
----------------

.. list-table::
   :header-rows: 1
   :widths: 29 40 31

   * - Flag and field
     - Description
     - Solvers
   * - ``BODY_QDD`` / ``outputs.body_qdd``
     - Rigid-body spatial accelerations
     - :class:`~newton.solvers.SolverMuJoCo`
   * - ``BODY_PARENT_F`` / ``outputs.body_parent_f``
     - Incoming parent-joint wrenches on rigid bodies
     - :class:`~newton.solvers.SolverMuJoCo`,
       :class:`~newton.solvers.SolverFeatherstone`, and
       :class:`~newton.solvers.SolverXPBD`
   * - ``CONTACT_F`` / ``outputs.contact_f``
     - Contact spatial forces aligned with the bound contacts
     - :class:`~newton.solvers.SolverMuJoCo` with MuJoCo Warp,
       :class:`~newton.solvers.SolverXPBD`, and
       :class:`~newton.solvers.SolverKamino`

:class:`~newton.solvers.experimental.coupled.SolverCoupled` exposes a body
output when every entry that owns bodies supports that flag. It allocates an
entry-local container for each sub-solver and gathers owned rows into parent
model order. Contact outputs are not exposed by the coupled wrapper because
filtered entry contacts require an explicit contact-index remapping contract.

Solver-specific outputs
-----------------------

Solvers can derive their container from
:class:`~newton.solvers.SolverOutputs` and define a separate output enum. For
example, MuJoCo adds ``SolverMuJoCo.OutputFlags.QFRC_ACTUATOR`` and returns a
``SolverMuJoCo.Outputs`` instance with ``qfrc_actuator``:

.. code-block:: python

   solver = newton.solvers.SolverMuJoCo(model)
   outputs = solver.outputs(
       {
           newton.solvers.SolverOutputFlags.BODY_QDD,
           solver.OutputFlags.QFRC_ACTUATOR,
       }
   )

Sets may contain members from both enums without coordinating bit values.
Output enums must derive directly from :class:`enum.Enum`, not
:class:`enum.IntEnum` or a string-mixin enum. Integer and string enum members
can compare equal across enum classes and silently collide in a set.

Override ``_allocate_outputs()`` and call ``super()`` to allocate inherited
arrays before custom arrays. Custom contact-indexed flags also belong in
``CONTACT_OUTPUT_FLAGS`` so the base solver requires pipeline initialization,
freezes capacities, and binds contact storage even when ``CONTACT_F`` itself is
not requested:

.. code-block:: python

   class CustomSolver(newton.solvers.SolverBase):
       OUTPUTS_TYPE = CustomOutputs
       SUPPORTED_OUTPUT_FLAGS = frozenset({CustomOutputFlags.CONTACT_PRESSURE})
       CONTACT_OUTPUT_FLAGS = (
           newton.solvers.SolverBase.CONTACT_OUTPUT_FLAGS
           | {CustomOutputFlags.CONTACT_PRESSURE}
       )

       def _allocate_outputs(self, outputs, *, requires_grad):
           super()._allocate_outputs(outputs, requires_grad=requires_grad)
           if CustomOutputFlags.CONTACT_PRESSURE in outputs:
               outputs.contact_pressure = wp.zeros(
                   self.model.rigid_contact_max,
                   dtype=float,
                   device=self.model.device,
                   requires_grad=requires_grad,
               )

Here ``CustomOutputs`` derives from ``SolverOutputs`` and initializes
``contact_pressure`` to ``None``. Custom fields sized by bodies, particles, or
solver-owned dimensions do not need ``CONTACT_OUTPUT_FLAGS``.

Sensors
-------

Solver-dependent sensors expose a composable ``solver_output_flags`` set. A
caller can union the requirements of multiple consumers, allocate one
container, and pass it through the step:

.. code-block:: python

   imu = newton.sensors.SensorIMU(model, sites="imu_*")
   contact_sensor = newton.sensors.SensorContact(model, sensing_shapes="foot_*")

   flags = imu.solver_output_flags | contact_sensor.solver_output_flags
   # Construct a pipeline with a compatible capacity before requesting contacts.
   outputs = solver.outputs(flags)

   solver.step(state_in, state_out, control, contacts, dt, outputs=outputs)
   imu.update(state_out, outputs=outputs)
   contact_sensor.update(state_out, contacts, outputs=outputs)

The viewer follows the same pattern:

.. code-block:: python

   viewer.log_contacts(contacts, state_out, outputs=outputs)

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
     - ``SolverOutputFlags.BODY_QDD`` and ``outputs.body_qdd``
   * - ``State.body_parent_f``
     - ``SolverOutputFlags.BODY_PARENT_F`` and ``outputs.body_parent_f``
   * - ``State.mujoco.qfrc_actuator``
     - ``SolverMuJoCo.OutputFlags.QFRC_ACTUATOR`` and
       ``outputs.qfrc_actuator``
   * - ``Contacts.force``
     - ``SolverOutputFlags.CONTACT_F`` and ``outputs.contact_f``
   * - ``Model.request_state_attributes()`` and
       ``ModelBuilder.request_state_attributes()``
     - ``solver.outputs({...})``
   * - ``Model.request_contact_attributes()`` and
       ``ModelBuilder.request_contact_attributes()``
     - ``solver.outputs({...})``
   * - ``solver.update_contacts()``
     - Pass contact outputs to ``solver.step(..., outputs=outputs)``

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
