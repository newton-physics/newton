.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _extended_attributes:

Extended Attributes
===================

Newton's :class:`~newton.State` and :class:`~newton.Contacts` objects can optionally carry extra arrays that are not always needed.
These *extended attributes* are allocated on demand when explicitly requested, reducing memory usage for simulations that don't need them.

.. _extended_contact_attributes:

Extended Contact Attributes
---------------------------

Extended contact attributes are optional arrays on :class:`~newton.Contacts` (e.g., contact forces for sensors).
Request them via :meth:`Model.request_contact_attributes <newton.Model.request_contact_attributes>` or :meth:`ModelBuilder.request_contact_attributes <newton.ModelBuilder.request_contact_attributes>` before creating a :class:`~newton.Contacts` object.

.. testcode::

   import newton

   builder = newton.ModelBuilder()
   body = builder.add_body(mass=1.0)
   builder.add_shape_sphere(body, radius=0.1)
   model = builder.finalize()

   # Request the "force" extended attribute directly
   model.request_contact_attributes("force")

   contacts = model.contacts()
   print(contacts.force is not None)

.. testoutput::

   True

Some components request attributes transparently.  For example,
:class:`~newton.sensors.SensorContact` requests ``"force"`` at init time, so
creating the sensor before allocating contacts is sufficient:

.. testcode::

   import warp as wp
   import newton
   from newton.sensors import SensorContact

   builder = newton.ModelBuilder()
   builder.add_ground_plane()
   body = builder.add_body(xform=wp.transform((0, 0, 0.1), wp.quat_identity()))
   builder.add_shape_sphere(body, radius=0.1, label="ball")
   model = builder.finalize()

   sensor = SensorContact(model, sensing_obj_shapes="ball")
   contacts = model.contacts()
   print(contacts.force is not None)

.. testoutput::

   True

The canonical list is :attr:`Contacts.EXTENDED_ATTRIBUTES <newton.Contacts.EXTENDED_ATTRIBUTES>`:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Attribute
     - Description
   * - :attr:`~newton.Contacts.force`
     - Contact spatial forces (used by :class:`~newton.sensors.SensorContact`)


.. _extended_state_attributes:

Extended State Attributes
-------------------------

Extended state attributes are optional arrays on :class:`~newton.State` (e.g., accelerations for sensors).
Request them via :meth:`Model.request_state_attributes <newton.Model.request_state_attributes>` or :meth:`ModelBuilder.request_state_attributes <newton.ModelBuilder.request_state_attributes>` before calling :meth:`Model.state() <newton.Model.state>`.

.. testcode::

   import newton

   builder = newton.ModelBuilder()
   body = builder.add_body(mass=1.0)
   builder.request_state_attributes("body_qdd")
   model = builder.finalize()

   state = model.state()
   print(state.body_qdd is not None)

.. testoutput::

   True

The canonical list is :attr:`State.EXTENDED_ATTRIBUTES <newton.State.EXTENDED_ATTRIBUTES>`:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Attribute
     - Description
   * - :attr:`~newton.State.body_qdd`
     - Rigid-body spatial accelerations (used by :class:`~newton.sensors.SensorIMU`)
   * - :attr:`~newton.State.body_parent_f`
     - Rigid-body parent interaction wrenches
   * - ``State.mujoco.qfrc_actuator``
     - Actuator forces in generalized (joint DOF) coordinates, namespaced under ``state.mujoco.qfrc_actuator``.
       Only populated by :class:`~newton.solvers.SolverMuJoCo`.


Notes
-----

- Some components transparently request attributes they need. For example, :class:`~newton.sensors.SensorIMU` requests ``body_qdd`` and :class:`~newton.sensors.SensorContact` requests ``force``.
  Create sensors before allocating State/Contacts for this to work automatically.
- Solvers populate extended attributes they support. :class:`~newton.solvers.SolverMuJoCo` populates ``body_qdd``, ``body_parent_f``, ``mujoco:qfrc_actuator``, and ``force``. :class:`~newton.solvers.SolverFeatherstone` populates ``body_parent_f`` directly from its RNEA backward pass. :class:`~newton.solvers.SolverXPBD` populates ``body_parent_f`` and ``force``; XPBD's reported wrenches are approximate (it applies relaxation factors to each constraint correction and is not momentum-conserving), so they should be treated as the *applied* constraint reaction rather than an exact analytic value. For simple decoupled cases (e.g. a single dynamic body suspended from a kinematic or world parent) the XPBD values converge to within the integrator's first-order time-stepping bias.
