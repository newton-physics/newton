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

   pipeline = newton.CollisionPipeline(model)
   contacts = pipeline.contacts()
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

   sensor = SensorContact(model, sensing_shapes="ball")
   pipeline = newton.CollisionPipeline(model)
   contacts = pipeline.contacts()
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
     - Contact spatial forces (used by :class:`~newton.sensors.SensorContact`). Rigid-contact
       rows come first, followed by one row per soft (particle-shape) contact.


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
     - Rigid-body center-of-mass spatial accelerations in the world frame (used by
       :class:`~newton.sensors.SensorIMU`)
   * - :attr:`~newton.State.body_parent_f`
     - Rigid-body parent interaction wrenches
   * - ``State.mujoco.qfrc_actuator``
     - Actuator forces in generalized (joint DOF) coordinates, namespaced under ``state.mujoco.qfrc_actuator``.
       Only populated by :class:`~newton.solvers.SolverMuJoCo`.


Notes
-----

- Some components transparently request attributes they need. For example, :class:`~newton.sensors.SensorIMU` requests ``body_qdd`` and :class:`~newton.sensors.SensorContact` requests ``force``.
  Create sensors before allocating State/Contacts for this to work automatically.
- Solvers populate extended attributes they support. :class:`~newton.solvers.SolverMuJoCo`
  populates ``body_qdd``, ``body_parent_f``, ``mujoco:qfrc_actuator``, and ``force``.
  :class:`~newton.solvers.SolverFeatherstone` populates ``body_parent_f`` directly
  from its RNEA backward pass. :class:`~newton.solvers.SolverXPBD` populates
  ``body_parent_f`` and ``force``; XPBD's reported wrenches are approximate (it
  applies relaxation factors to each constraint correction and is not
  momentum-conserving), so they should be treated as the *applied* constraint
  reaction rather than an exact analytic value. For simple decoupled cases (e.g. a
  single dynamic body suspended from a kinematic or world parent) the XPBD values
  converge to within the integrator's first-order time-stepping bias.
  :class:`~newton.solvers.SolverVBD` populates ``force`` for the contacts it solves,
  evaluated at the final configuration of each step: the rigid-contact rows when it
  integrates the rigid bodies, and the soft-contact rows (row ``rigid_contact_max + i``
  for soft contact ``i``) with its rigid-soft wrenches for particle, edge, and face
  records -- the force on the contacted shape's body and its torque about that body's
  COM (the world origin for a static shape). With an external rigid solver the rigid
  rows are left to that solver. See :meth:`~newton.solvers.SolverVBD.update_contacts`
  for the full convention and :ref:`vbd_contact_forces` for a guide.
  :class:`~newton.solvers.SolverKamino` populates ``body_qdd`` as the discrete
  step-average acceleration ``(body_qd_out - body_qd_in) / dt``; across an
  impact, this includes the velocity impulse divided by ``dt``.

.. _vbd_contact_forces:

Contact forces from SolverVBD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~newton.solvers.SolverVBD` exports one wrench per contact record it solves -- body-body
contacts, and rigid-soft particle, edge, and face records against rigid shapes -- through
:attr:`~newton.Contacts.force`:

1. Request ``"force"`` on the builder or the finalized model *before* creating the
   :class:`~newton.Contacts` buffer; a buffer allocated earlier has ``force is None`` and the
   export stays off.
2. Each frame, run collision detection, step the solver with that buffer, then call
   :meth:`~newton.solvers.SolverVBD.update_contacts` with the same buffer. The step evaluates
   the wrenches only when ``contacts.force`` is allocated; the extra cost is one kernel over
   each contact capacity, and the evaluation never changes the simulation result.
3. Rows ``[0, rigid_contact_max)`` hold the body-body contacts (``i < rigid_contact_count``) when
   VBD integrates the rigid bodies: the force on body 0 by body 1 with its torque about body 0's
   center of mass, the convention shared by every solver writing ``force``. With an external rigid
   solver these rows are left to that solver; call its ``update_contacts`` first, because
   :meth:`~newton.solvers.SolverXPBD.update_contacts` clears the whole array. Rows
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

Soft self-contact forces are not exported. :class:`~newton.sensors.SensorContact` reads the
rigid-contact rows, so it reports VBD's body-body contacts, but it does not read the soft rows.

.. testcode::

   import numpy as np
   import warp as wp
   import newton

   builder = newton.ModelBuilder()
   builder.add_ground_plane()
   builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.045), vel=wp.vec3(0.0), mass=1.0, radius=0.05)
   ball = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.099), wp.quat_identity()))
   ball_shape = builder.add_shape_sphere(ball, radius=0.1)
   builder.color()
   builder.request_contact_attributes("force")  # before pipeline.contacts()
   model = builder.finalize()

   pipeline = newton.CollisionPipeline(model)
   contacts = pipeline.contacts()
   solver = newton.solvers.SolverVBD(model, rigid_compliant_alm=True)
   state_in, state_out = model.state(), model.state()

   pipeline.collide(state_in, contacts)
   solver.step(state_in, state_out, None, contacts, dt=1.0 / 60.0)
   solver.update_contacts(contacts, state_out)

   # Body-body rows: force on body 0 by body 1; flip the sign of rows where the ball is shape 1.
   n_rigid = int(contacts.rigid_contact_count.numpy()[0])
   rigid = contacts.force.numpy()[:n_rigid]
   shape0 = contacts.rigid_contact_shape0.numpy()[:n_rigid]
   sign = np.where(shape0 == ball_shape, 1.0, -1.0)
   force_on_ball = (sign[:, None] * rigid[:, :3]).sum(axis=0)  # [N], world frame

   # Soft rows follow the rigid rows.
   n_soft = int(contacts.soft_contact_count.numpy()[0])
   start = contacts.rigid_contact_max
   wrenches = contacts.force.numpy()[start : start + n_soft]
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
