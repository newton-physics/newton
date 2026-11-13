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

   Solver-dependent sensors expose ``solver_observable_kinds``. Combine these sets,
   allocate :doc:`solver observables <solver_observables>` once, and pass the
   container to the solver and sensor updates.

   ``SensorCamera`` writes results to output arrays passed into ``update()`` rather than storing them as sensor
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
   imu = SensorIMU(model, sites="imu_*", request_state_attributes=False)

   # Create solver and state
   solver = newton.solvers.SolverMuJoCo(model)
   observables = solver.observables(kinds=imu.solver_observable_kinds)
   state = model.state()

   # Simulation loop
   for _ in range(100):
       state.clear_forces()
       solver.step(state, state, None, None, dt=1.0 / 60.0, observables=observables)

       # 2. Compute measurements from the current state
       imu.update(state, observables=observables)

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
   SensorIMU(model, sites="foot_*", request_state_attributes=False)

   # compiled regular expression: full-match an environment and object label
   SensorIMU(model, sites=re.compile(r"/World/envs/env_[0-9]+/imu_(left|right)"), request_state_attributes=False)

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
* :class:`~newton.sensors.SensorCamera` -- raytraced color, HDR color, depth, forward-depth, normal, albedo, and
  shape-index rendering; one view per camera transform, mapped to worlds via a per-view selector. Deformable
  visual meshes render too; see :ref:`deformable-visual-camera-output`.
* :class:`~newton.sensors.SensorTiledCamera` -- deprecated; superseded by :class:`~newton.sensors.SensorCamera`.

Camera Traversal Backends
-------------------------

:class:`~newton.sensors.SensorCamera` and :func:`~newton.intersect_ray` traverse
shapes with Warp BVHs or, when ``newton[optix]`` is installed, with OptiX
hardware ray tracing. By default (``backend=None``) they use OptiX when it
supports the model and device and Warp otherwise, so installing the extra is
enough to enable it:

.. code-block:: python

   camera = SensorCamera(model)
   print(camera.backend)  # RaycastBackend.OPTIX when available

Pass :attr:`RaycastBackend.WARP <newton.RaycastBackend.WARP>` to always use
Warp, or :attr:`RaycastBackend.OPTIX <newton.RaycastBackend.OPTIX>` to require
OptiX; requesting OptiX when it cannot be used raises :class:`RuntimeError`
with the reason.

.. experimental::

   ``newton.RaycastBackend.OPTIX`` is experimental. The ``warp-optix~=0.3.0``
   extra currently installs on Python 3.12, on x86-64 Linux and Windows, and
   requires an NVIDIA CUDA device supported by OptiX.

OptiX shares mesh geometry between instances and spatially packs worlds into
one instance acceleration structure. Rays are offset into the selected world's
cell and clipped to its bounds so they cannot hit neighboring worlds. The cell
extent grows on the GPU from the model's shape bounds when geometry moves.
Global bounded shapes occupy an additional cell, while planes are intersected
analytically. Camera rays, image outputs, materials, lights, and shading are the
same for both backends. Packing uses float32 coordinates, so results are not
bit-identical to Warp.

Continue to refit the model's shape BVH after changing poses or sizes. The
camera refits its OptiX instances from those transforms on every update; pass
``sync_shapes=False`` to skip the refit for an unchanged scene. Mesh topology
and the set of enabled shapes must remain fixed for the camera's lifetime.

The OptiX backend does not support:

- CUDA graph capture. Render outside graph capture, or create the camera with
  ``backend=RaycastBackend.WARP``; simulation may still use CUDA graphs.
  Raycasts with the default backend use Warp while a graph is being captured.
- Particles, deformable surfaces, Gaussian shapes, and shape variants. Cameras
  of such models use Warp by default. Raycasts query rigid shapes only.

Raycasts build their OptiX acceleration lazily on the model.
:meth:`~newton.Model.bvh_refit_shapes` updates it, and
:meth:`~newton.Model.bvh_build_shapes` invalidates it, including when the shape
selection changes. World selection, optional outputs, and miss values are the
same for both backends.

The optional package declares ``Apache-2.0 AND BSD-3-Clause AND
LicenseRef-NVIDIA-OptiX``. It is installed separately; Newton does not redistribute
its binaries or change their license terms.

Camera Rays from USD and Calibration Data
-----------------------------------------

Camera image allocation helpers share the ``allocate_image_`` prefix:
``color``, ``color_hdr``, ``depth``, ``depth_forward``, ``normal``, ``albedo``,
and ``shape_index``. These helpers return zero-initialized arrays on the model's
device. For example, use
:meth:`~newton.sensors.SensorCamera.allocate_image_color` for a packed RGBA
buffer.

:class:`~newton.sensors.SensorCamera` renders one view per world-space camera transform passed to
:meth:`~newton.sensors.SensorCamera.update`. The caller owns the camera-space rays and the per-view transforms.
The ray bundle for a standard USD pinhole camera can be built directly with
:meth:`~newton.sensors.SensorCamera.compute_camera_rays_pinhole_usd`, and the matching world-space per-view
transforms with :meth:`~newton.sensors.SensorCamera.compute_camera_transforms_usd` (which converts the USD stage's
up axis to the model's and composes an optional import ``xform``). For lens models without standard USD attributes,
read the attributes you use in your pipeline and pass the numeric values into the matching helper:

.. code-block:: python

   from pxr import Usd

   from newton.sensors import SensorCamera

   stage = Usd.Stage.Open("scene.usda")
   usd_camera = stage.GetPrimAtPath("/World/Camera")

   camera = SensorCamera(model)
   camera.create_default_light()

   # Camera-space rays for one 640x480 pinhole camera, on the model device.
   camera_rays = SensorCamera.compute_camera_rays_pinhole_usd(640, 480, usd_camera, device=model.device)

   # World-space transform per view, read from the USD camera(s).
   camera_transforms = camera.compute_camera_transforms_usd(usd_camera)
   view_count = camera_transforms.shape[0]

   color = camera.allocate_image_color(view_count, 640, 480)

   # update() syncs deformable-mesh points from state by default; refit the
   # shape/particle BVHs first on any frame whose geometry moved.
   camera.update(state, camera_transforms, camera_rays, color_image=color)

For OpenCV-calibrated pinhole cameras, call
:meth:`~newton.sensors.SensorCamera.compute_camera_rays_pinhole_opencv` with the calibrated intrinsics and
radial, tangential, and optional thin-prism coefficients.

For fisheye cameras, extract the calibration values from your chosen USD attributes and call one of
:meth:`~newton.sensors.SensorCamera.compute_camera_rays_fisheye_opencv`,
:meth:`~newton.sensors.SensorCamera.compute_camera_rays_fisheye_ftheta`, or
:meth:`~newton.sensors.SensorCamera.compute_camera_rays_fisheye_kannala_brandt`. Each helper builds a single-camera
``(height, width, sample_count, 2)`` ray bundle. Set ``sample_count`` to generate
subpixel rays and select ``SensorCamera.AntiAliasing.SSAA`` or ``MSAA`` through
``SensorCamera.RenderConfig.anti_aliasing`` to resolve them.

``MSAA`` traces every ray for coverage but shades each covered shape, particle,
or deformable-mesh face using the first ray in bundle order that hits it. That
shaded color is reused for the other rays hitting the same surface, so shading
may come from an off-center subpixel ray. Use ``SSAA`` to shade every ray.
The built-in ray helpers put the pixel-center ray first when ``sample_count`` is
at least three, with the remaining rays evenly spaced around it. With two samples,
they use an off-center pair so the sample pattern remains centered on the pixel.

Solver Observables
------------------

``SensorIMU`` requires ``SolverBase.ObservableKind.BODY_QDD`` and ``SensorContact``
requires ``SolverBase.ObservableKind.CONTACT_F``. ``SensorContact`` uses only the
linear part of each ``CONTACT_F`` wrench (force [N]); the torque part is ignored.
Each sensor's ``solver_observable_kinds`` property provides its requirements
without mutating the model. Union the sets when both sensors are present.
Pass a contacts buffer to the factory when requesting contact-indexed
observables, and pass each solver step's buffer to the sensor:

.. code-block:: python

   kinds = imu.solver_observable_kinds | contact_sensor.solver_observable_kinds
   observables = solver.observables(kinds=kinds, contacts=contacts)

   solver.step(state_in, state_out, control, contacts, dt, observables=observables)
   imu.update(state_out, observables=observables)
   contact_sensor.update(state_out, contacts, observables=observables)

Contact observables are sized from the supplied contacts' capacities, not the
current number of contacts. See :ref:`solver_observables` for
pipeline setup with native collision backends and graph capture.

Both :class:`~newton.solvers.SolverKamino` and
:class:`~newton.solvers.SolverMuJoCo` populate ``body_qdd``. Kamino reports
the discrete step-average center-of-mass acceleration in the world frame;
impact steps therefore include the velocity impulse divided by the step duration.

Performance Considerations
--------------------------

Sensors are designed to be efficient and GPU-friendly, computing results in
parallel where possible. Create each sensor once during setup and reuse it
every step -- this lets Newton pre-allocate output arrays and avoid per-frame
overhead.

Requested solver observables may add nontrivial cost to the solver step itself.
Request only the kinds consumed by the application and reuse the allocation.

See Also
--------

* :doc:`sites` -- using sites as sensor attachment points and reference frames
* :doc:`../api/newton_sensors` -- full sensor API reference
* :doc:`solver_observables` -- optional arrays produced by solvers
* ``newton.examples.sensors.example_sensor_contact`` -- SensorContact example
* ``newton.examples.sensors.example_sensor_imu`` -- SensorIMU example
* ``newton.examples.sensors.example_sensor_camera`` -- SensorCamera example (run with ``python -m newton.examples sensor_camera``)
