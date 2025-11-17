Sensors
=======

Sensors in Newton provide a way to extract measurements and observations from the simulation state. They compute derived quantities that are commonly needed for control, reinforcement learning, robotics applications, and analysis.

Overview
--------

Newton sensors follow a consistent pattern:

1. **Initialization**: Configure the sensor with the model and specify what to measure
2. **Update**: Call ``sensor.update(model, state)`` during the simulation loop to compute measurements
3. **Access**: Read results from sensor attributes (typically as Warp arrays)

Sensors are designed to be efficient and GPU-friendly, computing results in parallel where possible.

Available Sensors
-----------------

Newton currently provides three sensor types:

* **ContactSensor** - Detects and reports contact information between bodies (TODO: document)
* **RaycastSensor** - Performs ray casting for distance measurements and collision detection (TODO: document)
* **FrameTransformSensor** - Computes relative transforms between reference frames

FrameTransformSensor
--------------------

The ``FrameTransformSensor`` computes the relative pose (position and orientation) of objects with respect to reference frames. This is essential for:

* End-effector pose tracking in robotics
* Sensor pose computation (cameras, IMUs relative to world or body frames)
* Object tracking and localization tasks
* Reinforcement learning observations

Basic Usage
~~~~~~~~~~~

The sensor takes shape indices (which can include sites or regular shapes) and computes their transforms relative to reference site frames:

.. testcode:: sensors-basic

   from newton.sensors import FrameTransformSensor
   import newton
   
   # Create model with sites
   builder = newton.ModelBuilder()
   
   base = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))
   ref_site = builder.add_site(base, key="reference")
   builder.add_joint_free(base)
   
   end_effector = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))
   ee_site = builder.add_site(end_effector, key="end_effector")
   
   # Add a revolute joint to connect bodies
   builder.add_joint_revolute(
       parent=base,
       child=end_effector,
       axis=newton.Axis.X,
       parent_xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
       child_xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
   )
   
   model = builder.finalize()
   state = model.state()
   
   # Create sensor
   sensor = FrameTransformSensor(
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

   from newton.sensors import FrameTransformSensor
   
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
   
   # Add joints
   builder.add_joint_free(body1)
   builder.add_joint_free(body2)
   builder.add_joint_free(body3)
   builder.add_joint_free(ref_body)

   # Multiple objects, multiple references (must match in count) for sensor 2
   ref1 = builder.add_site(body1, key="ref1")
   ref2 = builder.add_site(body2, key="ref2")
   ref3 = builder.add_site(body3, key="ref3")
   
   model = builder.finalize()
   state = model.state()
   
   # Multiple objects, single reference
   sensor1 = FrameTransformSensor(
       model,
       shapes=[site1, site2, site3],
       reference_sites=[ref_site]  # Broadcasts to all objects
   )
   
   sensor2 = FrameTransformSensor(
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

See Also
--------

* :doc:`sites` — Using sites as reference frames
* :doc:`../api/newton_sensors` — Full sensor API reference
* ``newton.examples.sensors.example_sensor_contact`` — ContactSensor example

