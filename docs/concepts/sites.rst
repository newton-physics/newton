Sites (Abstract Markers)
========================

**Sites** are abstract reference points that don't participate in physics simulation or collision detection. They are lightweight markers used for:

* Sensor attachment points (IMU, camera, raycast origins)
* Frame of reference definitions for measurements
* Debugging and visualization reference points
* Spatial tendon attachment points and routing

Overview
--------

Sites in Newton are implemented as a special type of shape with the following properties:

* **No collision**: Sites never collide with any objects (shapes or particles)
* **No mass contribution**: Sites have zero density and don't affect body inertia
* **Transform-based**: Sites have position and orientation relative to their parent body
* **Shape types**: Sites can use any geometric primitive (sphere, box, capsule, etc.) for visualization
* **Visibility**: Sites can be visible (for debugging) or invisible (for runtime use)

Creating Sites
--------------

Sites are created using the ``add_site()`` method on ModelBuilder:

.. testcode:: sites-basic

   builder = newton.ModelBuilder()
   
   # Create a body
   body = builder.add_body(mass=1.0)
   
   # Add a site at body origin
   imu_site = builder.add_site(
       body=body,
       key="imu"
   )
   
   # Add a site with offset and rotation
   camera_site = builder.add_site(
       body=body,
       xform=wp.transform(
           wp.vec3(0.5, 0, 0.2),  # Position
           wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 3.14159/4)  # Orientation
       ),
       type=newton.GeoType.BOX,
       scale=(0.05, 0.05, 0.02),
       visible=True,
       key="camera"
   )

Sites can also be attached to the world frame (body=-1) to create fixed reference points:

.. testcode:: sites-world

   builder = newton.ModelBuilder()
   
   # World-frame reference site
   world_origin = builder.add_site(
       body=-1,
       xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
       key="world_origin"
   )

Alternative: Using Shape Methods with ``as_site=True``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sites can also be created using shape creation methods (``add_shape_sphere``, ``add_shape_box``, ``add_shape_capsule``, ``add_shape_cylinder``) by passing ``as_site=True``. This is particularly useful when programmatically generating shapes or conditionally creating sites:

.. testcode:: sites-shape-methods

   builder = newton.ModelBuilder()
   body = builder.add_body(mass=1.0)
   
   # Create sites using shape methods
   sphere_site = builder.add_shape_sphere(
       body=body,
       radius=0.05,
       as_site=True,
       key="sphere_marker"
   )
   
   box_site = builder.add_shape_box(
       body=body,
       hx=0.1, hy=0.1, hz=0.1,
       as_site=True,
       key="box_marker"
   )
   
   # Useful for conditional creation
   is_sensor_point = True
   shape_idx = builder.add_shape_sphere(
       body=body,
       radius=0.05,
       as_site=is_sensor_point,  # Conditionally a site
       key="measurement_point"
   )

When ``as_site=True``, the shape is automatically configured with all site invariants (no collision, zero density, collision_group=0), regardless of any custom configuration passed

Importing Sites
---------------

Sites are automatically imported from MJCF and USD files, with optional control over what gets loaded.

MJCF Import
~~~~~~~~~~~

MuJoCo sites are directly mapped to Newton sites, preserving type, position, orientation, and size:

.. code-block:: xml

   <mujoco>
       <worldbody>
           <body name="robot">
               <!-- Sites with various types and orientations -->
               <site name="sensor_site" type="sphere" size="0.02" pos="0.1 0 0"/>
               <site name="marker_site" type="box" size="0.05 0.05 0.05" 
                     quat="1 0 0 0" rgba="0 1 0 0.5"/>
           </body>
       </worldbody>
   </mujoco>

By default, sites are loaded along with collision and visual shapes. You can control this behavior with the ``parse_sites`` and ``parse_visuals`` parameters:

.. code-block:: python

   builder = newton.ModelBuilder()
   
   # Load only collision shapes and sites (no visual shapes)
   builder.add_mjcf("robot.xml", parse_sites=True, parse_visuals=False)
   
   # Load only collision shapes (no sites or visual shapes)
   builder.add_mjcf("robot.xml", parse_sites=False, parse_visuals=False)

USD Import
~~~~~~~~~~

Sites in USD are identified by the ``MjcSiteAPI`` schema applied to geometric primitives:

.. code-block:: usda

   def Xform "robot" (
       prepend apiSchemas = ["PhysicsRigidBodyAPI"]
   ) {
       def Sphere "imu_site" (
           prepend apiSchemas = ["MjcSiteAPI"]
       ) {
           double radius = 0.02
           double3 xformOp:translate = (0.1, 0, 0)
           uniform token[] xformOpOrder = ["xformOp:translate"]
       }
   }

Similar to MJCF import, you can control whether sites and visual shapes are loaded using the ``load_sites`` and ``load_visual_shapes`` parameters:

.. code-block:: python

   builder = newton.ModelBuilder()
   
   # Load only collision shapes and sites (no visual shapes)
   builder.add_usd("robot.usda", load_sites=True, load_visual_shapes=False)
   
   # Load only collision shapes (no sites or visual shapes)
   builder.add_usd("robot.usda", load_sites=False, load_visual_shapes=False)

By default, both ``load_sites`` and ``load_visual_shapes`` are set to ``True``.

Using Sites with Sensors
------------------------

Sites are commonly used as reference frames for sensors, particularly the ``SensorFrameTransform`` which computes relative poses between objects and reference frames.

For detailed information on using sites with sensors, see :doc:`sensors`.

MuJoCo Interoperability
-----------------------

When using ``SolverMuJoCo``, Newton sites are automatically exported to MuJoCo's native site representation:

.. testcode:: sites-mujoco

   from newton.solvers import SolverMuJoCo
   
   # Create a simple model with a site
   builder = newton.ModelBuilder()
   body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
   site = builder.add_site(body=body, key="sensor")
   model = builder.finalize()
   
   # Create MuJoCo solver (sites are exported by default)
   solver = SolverMuJoCo(model)

Sites are exported with their visual properties (color, size) and can be used with MuJoCo's native sensors and actuators. To disable site export, pass ``include_sites=False`` to the ``convert_to_mjc()`` method if calling it manually.

Implementation Details
----------------------

Sites are internally represented as shapes with the ``ShapeFlags.SITE`` flag set. This allows them to leverage Newton's existing shape infrastructure while maintaining distinct behavior:

* Sites are filtered out from collision detection pipelines
* Site density is automatically set to zero during creation
* Sites can be queried and filtered using the Selection API with shape-frequency operations

This implementation approach provides maximum flexibility while keeping the codebase maintainable and avoiding duplication.

See Also
--------

* :doc:`sensors` — Using sites with sensors for measurements
* :doc:`custom_attributes` — Attaching custom data to sites and other entities
* :doc:`../api/newton_sensors` — Full sensor API reference
* :doc:`usd_parsing` — Details on USD schema handling

