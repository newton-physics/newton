.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Sensors
=======

Sensors in Newton provide a way to extract measurements and observations from the simulation state. They compute derived quantities that are commonly needed for control, reinforcement learning, robotics applications, and analysis.

Overview
--------

Newton sensors follow a consistent pattern:

1. **Initialization**: Configure the sensor with the model and specify what to measure
2. **Update**: Call ``sensor.update(...)`` during the simulation loop to compute measurements
3. **Access**: Read results from sensor attributes (typically as Warp arrays)

Sensors are designed to be efficient and GPU-friendly, computing results in parallel where possible.

.. _label-matching:

Label Matching
--------------

Several Newton APIs accept **label patterns** to select bodies, shapes, joints,
etc. by name. Parameters that support label matching accept one of the following:

- A **list of integer indices** — selects directly by index.
- A **single string pattern** — selects all entries whose label matches the
  pattern via :func:`fnmatch.fnmatch` (supports ``*`` and ``?`` wildcards).
- A **list of string patterns** — selects all entries whose label matches at
  least one of the patterns.

Examples::

   # single pattern: all shapes whose label starts with "foot_"
   SensorIMU(model, sites="foot_*")

   # list of patterns: union of two groups
   SensorContact(model, sensing_obj_shapes=["*Plate*", "*Flap*"])

   # list of indices: explicit selection
   SensorFrameTransform(model, shapes=[0, 3, 7], reference_sites=[1])

Available Sensors
-----------------

Newton currently provides five sensor types:

* :class:`~newton.sensors.SensorContact` -- Detects and reports contact forces between bodies or shapes
* :class:`~newton.sensors.SensorFrameTransform` -- Computes relative transforms between reference frames
* :class:`~newton.sensors.SensorIMU` -- Measures linear acceleration and angular velocity at site frames
* :class:`~newton.sensors.SensorRaycast` -- Depth camera simulation via ray casting; outputs distance to scene geometry
* :class:`~newton.sensors.SensorTiledCamera` -- Raytraced rendering across multiple worlds

.. _sensorcontact:

SensorContact
-------------

:class:`~newton.sensors.SensorContact` measures contact forces between a set of *sensing* bodies or shapes and, optionally, a set of *counterpart* bodies or shapes. Outputs are stored in:

- :attr:`~newton.sensors.SensorContact.net_force`: net contact force [N] per (sensing object, counterpart) in world frame
- :attr:`~newton.sensors.SensorContact.sensing_obj_transforms`: world-frame transforms of each sensing object

If no counterparts are specified, the sensor reports the total contact force for each sensing object (one column). With counterparts, you get a force matrix; use ``include_total=True`` to add a total column.

Basic Usage
~~~~~~~~~~

``SensorContact`` requires exactly one of ``sensing_obj_bodies`` or ``sensing_obj_shapes`` to define the sensing objects. Optionally specify ``counterpart_bodies`` or ``counterpart_shapes`` to measure force per counterpart. It requires contact forces via the :doc:`extended attribute <extended_attributes>` :attr:`Contacts.force <newton.Contacts.force>`.

By default, the sensor requests the ``force`` attribute from the model during construction. Create the sensor before creating a :class:`~newton.Contacts` object and pass :meth:`model.get_requested_contact_attributes() <newton.Model.get_requested_contact_attributes>` when constructing Contacts so that ``force`` is allocated. Call :meth:`~newton.sensors.SensorContact.update` with both ``state`` and ``contacts`` after collision (e.g. after a solver step and a call to ``solver.update_contacts``).

.. testcode:: sensors-contact-basic

   from newton.sensors import SensorContact
   import newton

   builder = newton.ModelBuilder()
   body_a = builder.add_body(mass=1.0, label="a")
   builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
   body_b = builder.add_body(mass=1.0, label="b")
   builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.1)
   model = builder.finalize()

   sensor = SensorContact(model, sensing_obj_bodies=["a", "b"])
   state = model.state()
   contacts = newton.Contacts(
       64, 0, device=model.device,
       requested_attributes=model.get_requested_contact_attributes(),
   )
   sensor.update(state, contacts)
   forces = sensor.net_force.numpy()   # shape (n_sensing_objs, n_counterparts)
   xforms = sensor.sensing_obj_transforms.numpy()

State / Contacts Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SensorContact`` depends on contact forces computed by the collision/solver and stored in ``contacts.force``:

- **Allocate**: create ``SensorContact`` before :class:`~newton.Contacts` (or call :meth:`newton.Model.request_contact_attributes` with ``"force"`` yourself), then construct Contacts with ``requested_attributes=model.get_requested_contact_attributes()``.
- **Populate**: run collision and a solver step, then call ``solver.update_contacts(contacts, state)`` (or equivalent) so that ``contacts.force`` is filled before ``sensor.update(state, contacts)``.

SensorFrameTransform
--------------------

The ``SensorFrameTransform`` computes the relative pose (position and orientation) of objects with respect to reference frames. This is essential for:

* End-effector pose tracking in robotics
* Sensor pose computation (cameras, IMUs relative to world or body frames)
* Object tracking and localization tasks
* Reinforcement learning observations

Basic Usage
~~~~~~~~~~~

The sensor takes shape indices (which can include sites or regular shapes) and computes their transforms relative to reference site frames:

.. testcode:: sensors-basic

   from newton.sensors import SensorFrameTransform
   import newton
   
   # Create model with sites
   builder = newton.ModelBuilder()
   
   base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
   ref_site = builder.add_site(base, label="reference")
   j_free = builder.add_joint_free(base)
   
   end_effector = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
   ee_site = builder.add_site(end_effector, label="end_effector")
   
   # Add a revolute joint to connect bodies
   j_revolute = builder.add_joint_revolute(
       parent=base,
       child=end_effector,
       axis=newton.Axis.X,
       parent_xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
       child_xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
   )
   builder.add_articulation([j_free, j_revolute])
   
   model = builder.finalize()
   state = model.state()
   
   # Create sensor
   sensor = SensorFrameTransform(
       model,
       shapes=[ee_site],              # What to measure
       reference_sites=[ref_site]     # Reference frame(s)
   )
   
   # In simulation loop (after eval_fk)
   newton.eval_fk(model, state.joint_q, state.joint_qd, state)
   sensor.update(state)
   transforms = sensor.transforms.numpy()  # Array of relative transforms

Transform Computation
~~~~~~~~~~~~~~~~~~~~~

The sensor computes: ``X_ro = inverse(X_wr) * X_wo``

Where:
- ``X_wo`` is the world transform of the object (shape/site)
- ``X_wr`` is the world transform of the reference site
- ``X_ro`` is the resulting transform expressing the object's pose in the reference frame's coordinate system

This gives you the position and orientation of the object as observed from the reference frame.

Multiple Objects and References
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sensor supports measuring multiple objects, optionally with different reference frames:

.. testcode:: sensors-multiple

   from newton.sensors import SensorFrameTransform
   
   # Setup model with multiple sites
   builder = newton.ModelBuilder()
   body1 = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
   site1 = builder.add_site(body1, label="site1")
   body2 = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
   site2 = builder.add_site(body2, label="site2")
   body3 = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
   site3 = builder.add_site(body3, label="site3")
   ref_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
   ref_site = builder.add_site(ref_body, label="ref_site")

   # Sensor2: multiple objects with one reference per object (counts must match)
   ref1 = builder.add_site(body1, label="ref1")
   ref2 = builder.add_site(body2, label="ref2")
   ref3 = builder.add_site(body3, label="ref3")
   
   model = builder.finalize()
   state = model.state()
   
   # Multiple objects, single reference
   sensor1 = SensorFrameTransform(
       model,
       shapes=[site1, site2, site3],
       reference_sites=[ref_site]  # Broadcasts to all objects
   )
   
   sensor2 = SensorFrameTransform(
       model,
       shapes=[site1, site2, site3],
       reference_sites=[ref1, ref2, ref3]  # One per object
   )
   
   newton.eval_fk(model, state.joint_q, state.joint_qd, state)
   sensor2.update(state)
   transforms = sensor2.transforms.numpy()  # Shape: (num_objects, 7)
   
   # Extract position and rotation for first object
   import warp as wp
   xform = wp.transform(*transforms[0])
   pos = wp.transform_get_translation(xform)
   quat = wp.transform_get_rotation(xform)

Objects vs Reference Frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Objects** (``shapes``): Can be any shape index, including both regular shapes and sites
- **Reference frames** (``reference_sites``): Must be site indices (validated at initialization)

This design reflects the common use case where reference frames are explicitly defined coordinate systems (sites), while measurements can be taken of any geometric entity.

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

The sensor is optimized for GPU execution:

- Computes world transforms only once for all unique shapes/sites involved
- Uses pre-allocated Warp arrays to minimize memory overhead
- Parallel computation of all relative transforms

For best performance, create the sensor once during initialization and reuse it throughout the simulation, rather than recreating it each frame.

.. _sensorimu:

SensorIMU
---------

:class:`~newton.sensors.SensorIMU` measures inertial quantities at one or more sites; each site defines an IMU frame. Outputs are stored in two arrays:

- :attr:`~newton.sensors.SensorIMU.accelerometer`: linear acceleration (specific force)
- :attr:`~newton.sensors.SensorIMU.gyroscope`: angular velocity

Basic Usage
~~~~~~~~~~~

``SensorIMU`` takes a list of site indices and computes IMU readings at each site. It requires rigid-body accelerations via the :doc:`extended attribute <extended_attributes>` :attr:`State.body_qdd <newton.State.body_qdd>`.

By default, the sensor requests ``body_qdd`` from the model during construction, so that subsequent calls to :meth:`Model.state() <newton.Model.state>` allocate it.
If you need to allocate the State before constructing the sensor, you must request ``body_qdd`` on the model yourself before calling :meth:`Model.state() <newton.Model.state>`.


.. testcode:: sensors-imu-basic

   from newton.sensors import SensorIMU
   import newton

   builder = newton.ModelBuilder()
   body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
   s1 = builder.add_site(body, label="imu1")
   s2 = builder.add_site(body, label="imu2")
   model = builder.finalize()

   imu = SensorIMU(model, sites=[s1, s2])
   state = model.state()

   imu.update(state)
   acc = imu.accelerometer.numpy()  # shape: (2, 3)
   gyro = imu.gyroscope.numpy()      # shape: (2, 3)

State / Solver Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SensorIMU`` depends on body accelerations computed by the solver and stored in ``state.body_qdd``:

- **Allocate**: ensure ``body_qdd`` is allocated on the State (typically by constructing ``SensorIMU`` before calling :meth:`Model.state() <newton.Model.state>`).
- **Populate**: use a solver that actually fills ``state.body_qdd`` (for example, :class:`~newton.solvers.SolverMuJoCo` computes body accelerations).

.. _sensorraycast:

SensorRaycast
-------------

:class:`~newton.sensors.SensorRaycast` simulates a depth camera by casting rays from a virtual camera through each pixel and recording the distance to the closest intersection with scene geometry (rigid-body shapes and, optionally, particles). Outputs are stored in:

- :attr:`~newton.sensors.SensorRaycast.depth_image`: per-pixel depth [m], shape ``(height, width)``; positive values are distance to the closest surface, ``-1.0`` indicates no hit

The camera uses a right-handed frame: ``camera_direction`` (forward), ``camera_up``, and ``camera_right`` (cross of forward and up). Configure vertical field of view (``fov_radians``), resolution (``width``, ``height``), and ``max_distance`` (rays beyond this distance report no hit).

Basic Usage
~~~~~~~~~~

``SensorRaycast`` is constructed with the model and camera parameters (position, direction, up, vertical FOV, width, height). Call :meth:`~newton.sensors.SensorRaycast.update` with the current ``state`` so body poses are used for shape raycasting. For articulated models, run :func:`newton.eval_fk` before ``update`` so that body poses (``state.body_q``) are current. Optionally pass ``include_particles=True`` to also intersect rays with particles in ``state`` (requires the model to define particle geometry).

.. testcode:: sensors-raycast-basic

   import math
   import warp as wp
   from newton.sensors import SensorRaycast
   import newton

   builder = newton.ModelBuilder()
   body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()))
   builder.add_shape_sphere(body, radius=0.5)
   model = builder.finalize()
   state = model.state()

   newton.eval_fk(model, state.joint_q, state.joint_qd, state)

   sensor = SensorRaycast(
       model,
       camera_position=(0.0, 0.0, 0.0),
       camera_direction=(0.0, 0.0, 1.0),
       camera_up=(0.0, 1.0, 0.0),
       fov_radians=math.pi / 4,
       width=64,
       height=48,
       max_distance=10.0,
   )
   sensor.update(state)
   depth = sensor.depth_image.numpy()   # shape (48, 64); positive = distance, -1.0 = no hit

State / Solver Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SensorRaycast`` only needs body poses in ``state.body_q`` for shape raycasting; no extended State or Contacts attributes are required. Ensure body poses are up to date (e.g. call :func:`newton.eval_fk` for generalized-coordinate models before ``sensor.update(state)``). For ``include_particles=True``, the state must have ``particle_q`` and the model must have valid ``particle_radius`` (and ``particle_max_radius``).

See Also
--------

* :doc:`sites` — Using sites as reference frames
* :doc:`../api/newton_sensors` — Full sensor API reference
* :doc:`extended_attributes` — Optional State/Contacts arrays (e.g., ``State.body_qdd``, ``Contacts.force``) required by some sensors.
* ``newton.examples.sensors.example_sensor_contact`` — SensorContact example
* ``newton.examples.sensors.example_sensor_imu`` — SensorIMU example
* ``newton.examples.sensors.example_sensor_tiled_camera`` — SensorTiledCamera example
