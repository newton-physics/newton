.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Sensors
=======

Sensors in Newton provide a way to extract measurements and observations from the simulation. They compute derived
quantities that are commonly needed for control, reinforcement learning, robotics applications, and analysis.

Overview
--------

Most Newton sensors follow a common pattern:

1. **Initialization**: Configure the sensor with the model and specify what to measure
2. **Update**: Call ``sensor.update(state, ...)`` during the simulation loop to compute measurements
3. **Access**: Read results from sensor attributes (typically as Warp arrays)

.. note::

   Sensors automatically request any :doc:`extended attributes <extended_attributes>` they need
   (e.g. ``body_qdd``, ``Contacts.force``) at init, so ``State`` and ``Contacts`` objects created afterwards will
   include them.

   ``SensorContact`` additionally requires a call to ``solver.update_contacts()`` before ``sensor.update()``.

   ``SensorTiledCamera`` writes results to output arrays passed into ``update()`` rather than storing them as sensor
   attributes.

.. testcode::

   import warp as wp
   import newton
   from newton.sensors import SensorIMU

   # Build the model
   builder = newton.ModelBuilder()
   builder.add_ground_plane()
   body = builder.add_body(xform=wp.transform((0, 0, 1), wp.quat_identity()))
   builder.add_shape_sphere(body, radius=0.1)
   builder.add_site(body, label="imu_0")
   model = builder.finalize()

   # 1. Create sensor and specify what to measure
   imu = SensorIMU(model, sites="imu_*")

   # Create solver and state
   solver = newton.solvers.SolverMuJoCo(model)
   state = model.state()

   # Simulation loop
   for _ in range(100):
       state.clear_forces()
       solver.step(state, state, None, None, dt=1.0 / 60.0)

       # 2. Compute measurements from the current state
       imu.update(state)

       # 3. Results stored on sensor attributes
       acc = imu.accelerometer.numpy()   # (n_sensors, 3) linear acceleration
       gyro = imu.gyroscope.numpy()      # (n_sensors, 3) angular velocity

   print("accelerometer shape:", acc.shape)
   print("gyroscope shape:", gyro.shape)

.. testoutput::

   accelerometer shape: (1, 3)
   gyroscope shape: (1, 3)

.. _label-matching:

Label Matching
--------------

Several Newton APIs accept **label patterns** to select bodies, shapes, joints, sites, etc. by name. Parameters that
support label matching accept one of the following:

- A **list of integer indices** -- selects directly by index.
- A **single string pattern** -- selects all entries whose label matches the pattern via :func:`fnmatch.fnmatch`
  (supports ``*`` and ``?`` wildcards).
- A **list of string patterns** -- selects all entries whose label matches at least one of the patterns.

Examples::

   # single pattern: all shapes whose label starts with "foot_"
   SensorIMU(model, sites="foot_*")

   # list of patterns: union of two groups
   SensorContact(model, sensing_obj_shapes=["*Plate*", "*Flap*"])

   # list of indices: explicit selection
   SensorFrameTransform(model, shapes=[0, 3, 7], reference_sites=[1])

Available Sensors
-----------------

Newton provides five sensor types. See the
:doc:`API reference <../api/newton_sensors>` for constructor arguments,
attributes, and usage examples.

* :class:`~newton.sensors.SensorContact` -- contact forces between bodies or shapes, including friction decomposition,
  with optional per-counterpart breakdown.
* :class:`~newton.sensors.SensorFrameTransform` -- relative transforms of shapes/sites with respect to reference sites.
* :class:`~newton.sensors.SensorIMU` -- linear acceleration and angular velocity at site frames.
* :class:`~newton.sensors.SensorRaycast` -- *(deprecated)* ray-based depth images from a virtual camera; use
  :class:`~newton.sensors.SensorTiledCamera` instead.
* :class:`~newton.sensors.SensorTiledCamera` -- raytraced color and depth rendering across multiple worlds.

Extended Attributes
-------------------

Some sensors depend on extended attributes that are not allocated by default:

- ``SensorIMU`` requires ``State.body_qdd`` (rigid-body accelerations). By
  default it requests this from the model at construction, so subsequent
  ``model.state()`` calls allocate it automatically.
- ``SensorContact`` requires ``Contacts.force`` (per-contact spatial force
  wrenches). By default it requests this from the model at construction, so
  subsequent ``model.contacts()`` calls allocate it automatically. The solver
  must also support populating contact forces.

Performance Considerations
--------------------------

Sensors are designed to be efficient and GPU-friendly, computing results in
parallel where possible. Create each sensor once during setup and reuse it
every step -- this lets Newton pre-allocate output arrays and avoid per-frame
overhead.

Sensors that depend on extended attributes (e.g. ``body_qdd``,
``Contacts.force``) may add nontrivial cost to the solver step itself, since
the solver must compute and store these additional quantities regardless of
whether the sensor is evaluated after each step.

See Also
--------

* :doc:`sites` -- using sites as sensor attachment points and reference frames
* :doc:`../api/newton_sensors` -- full sensor API reference
* :doc:`extended_attributes` -- optional ``State``/``Contacts`` arrays required by some sensors
* ``newton.examples.sensors.example_sensor_contact`` -- SensorContact example
* ``newton.examples.sensors.example_sensor_imu`` -- SensorIMU example
* ``newton.examples.sensors.example_sensor_tiled_camera`` -- SensorTiledCamera example
