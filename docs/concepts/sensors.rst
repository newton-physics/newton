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

Available Sensors
-----------------

Newton currently provides five sensor types:

* :class:`~newton.sensors.SensorContact` -- Detects and reports contact information between bodies (TODO: document)
* :class:`~newton.sensors.SensorFrameTransform` -- Computes relative transforms between reference frames
* :class:`~newton.sensors.SensorIMU` -- Measures linear acceleration and angular velocity at site frames
* :class:`~newton.sensors.SensorRaycast` -- Performs ray casting for distance measurements and collision detection (TODO: document)
* :class:`~newton.sensors.SensorTiledCamera` -- Raytraced rendering across multiple worlds

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
   
   base = builder.add_link(mass=1.0, I_m=wp.mat33(np.eye(3)))
   ref_site = builder.add_site(base, key="reference")
   j_free = builder.add_joint_free(base)
   
   end_effector = builder.add_link(mass=1.0, I_m=wp.mat33(np.eye(3)))
   ee_site = builder.add_site(end_effector, key="end_effector")
   
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
   sensor.update(model, state)
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
   body1 = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))
   site1 = builder.add_site(body1, key="site1")
   body2 = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))
   site2 = builder.add_site(body2, key="site2")
   body3 = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))
   site3 = builder.add_site(body3, key="site3")
   ref_body = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))
   ref_site = builder.add_site(ref_body, key="ref_site")

   # Multiple objects, multiple references (must match in count) for sensor 2
   ref1 = builder.add_site(body1, key="ref1")
   ref2 = builder.add_site(body2, key="ref2")
   ref3 = builder.add_site(body3, key="ref3")
   
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
   sensor2.update(model, state)
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

:class:`~newton.sensors.SensorIMU` measures inertial quantities at one or more sites; each site defines the IMU frame. Outputs are stored in two arrays:

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
   body = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))
   s1 = builder.add_site(body, key="imu1")
   s2 = builder.add_site(body, key="imu2")
   model = builder.finalize()

   imu = SensorIMU(model, sites=[s1, s2])
   state = model.state()

   imu.update(state)
   acc = imu.accelerometer.numpy()  # shape: (2, 3)
   gyro = imu.gyroscope.numpy()      # shape: (2, 3)

State / Solver Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SensorIMU`` depends on body accelerations computed by the solver and stored in ``state.body_qdd``:

- Allocate: ensure ``body_qdd`` is allocated on the State (typically by constructing ``SensorIMU`` before calling :meth:`Model.state() <newton.Model.state>`).
- Populate: use a solver that actually fills ``state.body_qdd`` (for example, :class:`~newton.solvers.SolverMuJoCo` computes body accelerations).

See Also
--------

* :doc:`sites` — Using sites as reference frames
* :doc:`../api/newton_sensors` — Full sensor API reference
* :doc:`extended_attributes` — Optional State/Contacts arrays (e.g., ``State.body_qdd``, ``Contacts.force``) required by some sensors.
* ``newton.examples.sensors.example_sensor_contact`` — SensorContact example
* ``newton.examples.sensors.example_sensor_imu`` — SensorIMU example
* ``newton.examples.sensors.example_sensor_tiled_camera`` — SensorTiledCamera example
