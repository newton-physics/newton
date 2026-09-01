.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Sensors
========

Sensors in Newton provide a way to extract measurements and observations from the simulation. They compute derived
quantities that are commonly needed for control, reinforcement learning, robotics applications, and analysis.

Overview
--------

Most Newton sensors follow a common pattern:

1. **Initialization**: Configure the sensor with the model and specify what to measure
2. **Update**: Call ``sensor.update(state, ...)`` during the simulation loop to compute measurements
3. **Access**: Read results from sensor attributes (typically as Warp arrays)

.. note::

   Solver-dependent sensors expose ``solver_output_flags``. Combine these sets,
   allocate :doc:`solver outputs <solver_outputs>` once, and pass the
   container to the solver and sensor updates.

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
   outputs = solver.outputs(imu.solver_output_flags)
   state = model.state()

   # Simulation loop
   for _ in range(100):
       state.clear_forces()
       solver.step(state, state, None, None, dt=1.0 / 60.0, outputs=outputs)

       # 2. Compute measurements from the current state
       imu.update(state, outputs=outputs)

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
- A **list of string patterns** -- selects all entries whose label matches at least one pattern.
- A **compiled string regular expression** -- selects all entries whose entire label or name matches the expression via
  :meth:`re.Pattern.fullmatch`.

Ordinary strings always use glob syntax. Compile a pattern with :func:`re.compile` to opt into regular-expression
syntax. Callers who want a regular expression to match a substring can add ``.*`` around that substring explicitly.
For :class:`~newton.selection.ArticulationView`, ``pattern`` is matched against full articulation labels. Joint and
link filters are matched against the final path component of each label.

.. code-block:: python

   import re

   # single pattern: all shapes whose label starts with "foot_"
   SensorIMU(model, sites="foot_*")

   # compiled regular expression: full-match an environment and object label
   SensorIMU(model, sites=re.compile(r"/World/envs/env_[0-9]+/imu_(left|right)"))

   # list of patterns: union of two groups
   SensorContact(model, sensing_shapes=["*Plate*", "*Flap*"])

   # list of indices: explicit selection
   SensorFrameTransform(model, shapes=[0, 3, 7], reference_sites=[1])

Available Sensors
-----------------

Newton provides five sensor types. See the
:doc:`API reference <../api/newton_sensors>` for constructor arguments,
attributes, and usage examples.

* :class:`~newton.sensors.SensorContact` -- contact forces between bodies or shapes, with friction decomposition,
  optional per-counterpart force matrices, and force-weighted contact positions.
* :class:`~newton.sensors.SensorFrameTransform` -- relative transforms of shapes/sites with respect to reference sites.
* :class:`~newton.sensors.SensorIMU` -- linear acceleration and angular velocity at site frames.
* :class:`~newton.sensors.SensorTiledCamera` -- raytraced color and depth rendering across multiple worlds.

Camera Rays from USD and Calibration Data
-----------------------------------------

``SensorTiledCamera`` can build standard USD pinhole camera rays directly. For lens models without standard USD
attributes, read the attributes you use in your pipeline and pass the numeric values into the matching helper:

.. code-block:: python

   from pxr import Usd

   from newton.sensors import SensorTiledCamera

   stage = Usd.Stage.Open("scene.usda")
   usd_camera = stage.GetPrimAtPath("/World/Camera")

   sensor = SensorTiledCamera(model)
   camera_rays = sensor.utils.compute_camera_rays_usd_pinhole(640, 480, usd_camera)
   camera_transforms = sensor.utils.compute_camera_transforms_usd(usd_camera)

   color = sensor.utils.create_color_image_output(640, 480)
   sensor.update(
       state,
       camera_transforms,
       camera_rays,
       color_image=color,
   )

For OpenCV-calibrated pinhole cameras, call
:meth:`~newton.sensors.SensorTiledCamera.Utils.compute_camera_rays_pinhole_opencv` with the calibrated intrinsics and
radial, tangential, and optional thin-prism coefficients.

For fisheye cameras, extract the calibration values from your chosen USD attributes and call one of
:meth:`~newton.sensors.SensorTiledCamera.Utils.compute_camera_rays_fisheye_opencv`,
:meth:`~newton.sensors.SensorTiledCamera.Utils.compute_camera_rays_fisheye_ftheta`, or
:meth:`~newton.sensors.SensorTiledCamera.Utils.compute_camera_rays_fisheye_kannala_brandt`. Each fisheye helper builds
rays for one camera; pass ``out_rays`` and ``camera_index`` to fill a shared ray buffer.

Solver Outputs
--------------

``SensorIMU`` requires ``SolverOutputFlags.BODY_QDD`` and ``SensorContact``
requires ``SolverOutputFlags.CONTACT_F``. Their ``solver_output_flags``
properties provide these requirements without mutating the model. Union the
sets when both sensors are present and bind contact-indexed output to the
contacts buffer:

.. code-block:: python

   flags = imu.solver_output_flags | contact_sensor.solver_output_flags
   outputs = solver.outputs(flags, contacts=contacts)

   solver.step(state_in, state_out, control, contacts, dt, outputs=outputs)
   imu.update(state_out, outputs=outputs)
   contact_sensor.update(state_out, contacts, outputs=outputs)

Performance Considerations
--------------------------

Sensors are designed to be efficient and GPU-friendly, computing results in
parallel where possible. Create each sensor once during setup and reuse it
every step -- this lets Newton pre-allocate output arrays and avoid per-frame
overhead.

Requested solver outputs may add nontrivial cost to the solver step itself.
Request only the flags consumed by the application and reuse the allocation.

See Also
--------

* :doc:`sites` -- using sites as sensor attachment points and reference frames
* :doc:`../api/newton_sensors` -- full sensor API reference
* :doc:`solver_outputs` -- optional arrays produced by solvers
* ``newton.examples.sensors.example_sensor_contact`` -- SensorContact example
* ``newton.examples.sensors.example_sensor_imu`` -- SensorIMU example
* ``newton.examples.sensors.example_sensor_tiled_camera`` -- SensorTiledCamera example
