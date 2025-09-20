
Conventions
===========

This document covers the various conventions used across physics engines and graphics frameworks when working with Newton and other simulation systems.

Spatial Twist Conventions
--------------------------

Twists in Modern Robotics
~~~~~~~~~~~~~~~~~~~~~~~~~~

In robotics, a **twist** is a 6-dimensional velocity vector combining angular
and linear velocity. *Modern Robotics* (Lynch & Park) defines two equivalent
representations of a rigid body's twist, depending on the coordinate frame
used:

* **Body twist** (:math:`V_b`):
  uses the body's *body frame* (often at the body's center of mass).
  Here :math:`\omega_b` is the angular velocity expressed in the body frame,
  and :math:`v_b` is the linear velocity of a point at the body origin
  (e.g. the COM) expressed in the body frame.  
  Thus :math:`V_b = (\omega_b,\;v_b)` gives the body's own-frame view of its
  motion.

* **Spatial twist** (:math:`V_s`):
  uses the fixed *space frame* (world/inertial frame).
  :math:`v_s` represents the linear velocity of a hypothetical point on the
  moving body that is instantaneously at the world origin, and
  :math:`\omega_s` is the angular velocity expressed in world coordinates.  Equivalently,

  .. math::

     v_s \;=\; \dot p \;-\; \omega_s \times p,

  where :math:`p` is the vector from the world origin to the body origin.
  Hence :math:`V_s = (v_s,\;\omega_s)` is called the **spatial twist**.
  *Note:* :math:`v_s` is **not** simply the COM velocity
  (that would be :math:`\dot p`); it is the velocity of the *world origin* as
  if rigidly attached to the body.

In summary, *Modern Robotics* lets us express the same physical motion either
in the body frame or in the world frame.  The angular velocity is identical
up to coordinate rotation; the linear component depends on the chosen
reference point (world origin vs. body origin).

Physics-Engine Conventions (Drake, MuJoCo, Isaac)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most physics engines store the **COM linear velocity** together with the
**angular velocity** of the body, typically both in world coordinates.  This
corresponds conceptually to a twist taken at the COM and expressed in the
world frame, though details vary:

* **Drake**  
  Drake's multibody library uses full spatial vectors with explicit frame
  names.  The default, :math:`V_{MB}^{E}`, reads "velocity of frame *B*
  measured in frame *M*, expressed in frame *E*."  In normal use
  :math:`V_{WB}^{W}` (body *B* in world *W*, expressed in *W*) contains
  :math:`(\omega_{WB}^{W},\;v_{WB_o}^{W})`, i.e. both components in the world
  frame.  This aligns with the usual physics-engine convention.

* **MuJoCo**  
  MuJoCo employs a *mixed-frame* format for free bodies:  
  the linear part :math:`(v_x,v_y,v_z)` is in the world frame, while the
  angular part :math:`(\omega_x,\omega_y,\omega_z)` is expressed in the **body
  frame**.  The choice follows from quaternion integration (angular velocities
  "live" in the quaternion's tangent space, a local frame).

* **Isaac Lab / Isaac Gym**  
  NVIDIA's Isaac tools provide **both** linear and angular velocities in the
  world frame.  The root-state tensor returns
  :math:`(v_x,v_y,v_z,\;\omega_x,\omega_y,\omega_z)` all expressed globally.
  This matches Bullet/ODE/PhysX practice.

.. _Twist conventions:

Newton Conventions
~~~~~~~~~~~~~~~~~~

**Newton** follows the standard physics engine convention for most solvers, 
aligning with Isaac Lab's approach, but with one important exception:

* **Standard Newton solvers** (XPBD, Euler, SolverMuJoCo etc.)  
  Newton's :attr:`State.body_qd` stores **both** linear and angular velocities 
  in the world frame. The linear velocity represents the COM velocity in world 
  coordinates, while the angular velocity is also expressed in world coordinates.
  This matches the Isaac Lab convention exactly. Note that Newton will automatically
  convert from this convention to MuJoCo's mixed-frame format when using the SolverMuJoCo.

* **Featherstone solver**  
  Newton's Featherstone implementation uses the **spatial twist** convention 
  from *Modern Robotics*. Here :attr:`State.body_qd` represents a spatial 
  twist :math:`V_s = (v_s, \omega_s)` where :math:`v_s` is the linear 
  velocity of a hypothetical point on the moving body that is instantaneously 
  at the world origin, **not** the COM velocity.

Summary of Conventions
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 33 27 22

   * - **System**
     - **Linear velocity** (translation)
     - **Angular velocity** (rotation)
     - **Twist term**
   * - *Modern Robotics* — **body twist**
     - Body origin (e.g. COM), **body frame**
     - **Body frame**
     - "Body twist" (:math:`V_b`)
   * - *Modern Robotics* — **spatial twist**
     - World origin, **world frame**
     - **World frame**
     - "Spatial twist" (:math:`V_s`)
   * - **Drake**
     - Body origin (COM), **world frame**
     - **World frame**
     - Spatial velocity :math:`V_{WB}^{W}`
   * - **MuJoCo**
     - COM, **world frame**
     - **Body frame**
     - Mixed-frame 6-vector
   * - **Isaac Gym / Sim**
     - COM, **world frame**
     - **World frame**
     - "Root" linear / angular velocity
   * - **Newton** (standard solvers)
     - COM, **world frame**
     - **World frame**
     - Physics engine convention
   * - **Newton** (Featherstone)
     - World origin, **world frame**
     - **World frame**
     - Spatial twist :math:`V_s`

Mapping Between Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Body ↔ Spatial (Modern Robotics)**  
For body pose :math:`T_{sb}=(R,p)`:

.. math::

   \omega_s \;=\; R\,\omega_b,
   \qquad
   v_s \;=\; R\,v_b \;+\; \omega_s \times p.

This is :math:`V_s = \mathrm{Ad}_{(R,p)}\,V_b`;
the inverse uses :math:`R^{\mathsf T}` and :math:`-R^{\mathsf T}p`.

**Physics engine → MR**  
Given engine values :math:`(v_{\text{com}}^{W},\;\omega^{W})`
(world-frame COM velocity and angular velocity):

1. Spatial twist at COM  

   :math:`V_{WB}^{W} = (v_{\text{com}}^{W},\;\omega^{W})`

2. Body-frame twist  

   :math:`\omega_b = R^{\mathsf T}\omega^{W}`,
   :math:`v_b = R^{\mathsf T}v_{\text{com}}^{W}`.

3. Shift to another origin offset :math:`r` from COM:  

   :math:`v_{\text{origin}}^{W} = v_{\text{com}}^{W} + \omega^{W}\times r^{W}`,
   where :math:`r^{W}=R\,r`.

**MuJoCo conversion**  
Rotate its local angular velocity by :math:`R` to world frame
(or rotate a world-frame twist back into the body frame for MuJoCo).

In all cases the conversion boils down to the **reference point**
(COM vs. another point) and the **frame** (world vs. body) used for each
component.  Physics is unchanged; any linear velocity at one point follows
:math:`v_{\text{new}} = v + \omega\times r`.

Quaternion Ordering Conventions
--------------------------------

Different physics engines and graphics frameworks use different conventions 
for storing quaternion components. This can cause significant confusion when 
transferring data between systems or when interfacing with external libraries.

The quaternion :math:`q = w + xi + yj + zk` where :math:`w` is the scalar 
(real) part and :math:`(x, y, z)` is the vector (imaginary) part, can be 
stored in memory using different orderings:

.. list-table:: Quaternion Component Ordering
   :header-rows: 1
   :widths: 30 35 35

   * - **System**
     - **Storage Order**
     - **Description**
   * - **Newton / Warp**
     - ``(x, y, z, w)``
     - Vector part first, scalar last
   * - **Isaac Lab / Isaac Sim**
     - ``(w, x, y, z)``
     - Scalar first, vector part last
   * - **MuJoCo**
     - ``(w, x, y, z)``
     - Scalar first, vector part last
   * - **USD (Universal Scene Description)**
     - ``(x, y, z, w)``
     - Vector part first, scalar last

**Important Notes:**

* **Mathematical notation** typically writes quaternions as :math:`q = w + xi + yj + zk` 
  or :math:`q = (w, x, y, z)`, but this doesn't dictate storage order.

* **Conversion between systems** requires careful attention to component ordering.
  For example, converting from Isaac Lab to Newton requires reordering:
  ``newton_quat = (isaac_quat[1], isaac_quat[2], isaac_quat[3], isaac_quat[0])``

* **Rotation semantics** remain the same regardless of storage order—only the 
  memory layout differs.

* **Warp's quat type** uses ``(x, y, z, w)`` ordering, accessible via:
  ``quat[0]`` (x), ``quat[1]`` (y), ``quat[2]`` (z), ``quat[3]`` (w).

When working with multiple systems, always verify quaternion ordering in your 
data pipeline to avoid unexpected rotations or orientations.

Coordinate System and Up Axis Conventions
------------------------------------------

Different physics engines, graphics frameworks, and content creation tools use 
different conventions for coordinate systems and up axis orientation. This can 
cause significant confusion when transferring assets between systems or when 
setting up physics simulations from existing content.

The **up axis** determines which coordinate axis points "upward" in the world, 
affecting gravity direction, object placement, and overall scene orientation.

.. list-table:: Coordinate System and Up Axis Conventions
   :header-rows: 1
   :widths: 30 20 25 25

   * - **System**
     - **Up Axis**
     - **Handedness**
     - **Notes**
   * - **Newton**
     - ``Z`` (default)
     - Right-handed
     - Configurable via ``Axis.X/Y/Z``
   * - **MuJoCo**
     - ``Z`` (default)
     - Right-handed
     - Standard robotics convention
   * - **USD**
     - ``Y`` (default)
     - Right-handed
     - Configurable as ``Y`` or ``Z``
   * - **Isaac Lab / Isaac Sim**
     - ``Z`` (default)
     - Right-handed
     - Follows robotics conventions

**Important Design Principle:**

Newton itself is **coordinate system agnostic** and can work with any choice 
of up axis. The physics calculations and algorithms do not depend on a specific 
coordinate system orientation. However, it becomes essential to track the 
conventions used by various assets and data sources to enable proper conversion 
and integration at runtime.

**Common Integration Scenarios:**

* **USD to Newton**: Convert from USD's Y-up (or Z-up) to Newton's configured up axis
* **MuJoCo to Newton**: Convert from MuJoCo's Z-up to Newton's configured up axis  
* **Mixed asset pipelines**: Track up axis per asset and apply appropriate transforms

**Conversion Between Systems:**

When converting assets between coordinate systems with different up axes, 
apply the appropriate rotation transforms:

* **Y-up ↔ Z-up**: 90° rotation around the X-axis
* **Maintain right-handedness**: Ensure coordinate system handedness is preserved

**Example Configuration:**

.. code-block:: python

   import newton
   
   # Configure Newton for Z-up coordinate system (robotics convention)
   builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
   
   # Or use Y-up (graphics/animation convention)  
   builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
   
   # Gravity vector will automatically align with the chosen up axis:
   # - Y-up: gravity = (0, -9.81, 0)
   # - Z-up: gravity = (0, 0, -9.81)

Collision Primitive Conventions
-------------------------------

This section documents the conventions used for collision primitive shapes in Newton and compares them with other physics engines and formats. Understanding these conventions is essential when:

* Creating collision geometry programmatically with ModelBuilder
* Debugging unexpected collision behavior after asset import
* Understanding center of mass calculations for asymmetric shapes

Newton Collision Primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Newton defines collision primitives with consistent conventions across all shape types. The following table summarizes the key parameters and properties for each primitive:

.. list-table:: Newton Collision Primitive Specifications
   :header-rows: 1
   :widths: 15 20 35 30

   * - **Shape**
     - **Origin**
     - **Parameters**
     - **Notes**
   * - **Box**
     - Geometric center
     - ``hx``, ``hy``, ``hz`` (half-extents)
     - Edges aligned with local axes
   * - **Sphere**
     - Center
     - ``radius``
     - Uniform in all directions
   * - **Capsule**
     - Geometric center
     - ``radius``, ``half_height``
     - Extends along Z-axis; half_height excludes hemispherical caps
   * - **Cylinder**
     - Geometric center
     - ``radius``, ``half_height``
     - Extends along Z-axis
   * - **Cone**
     - Geometric center
     - ``radius`` (base), ``half_height``
     - Extends along Z-axis; base at -half_height, apex at +half_height
   * - **Plane**
     - Shape frame origin
     - ``width``, ``length`` (or 0,0 for infinite)
     - Normal along +Z of shape frame
   * - **Mesh**
     - User-defined
     - Vertex and triangle arrays
     - General triangle mesh (can be non-convex)

**Shape Orientation and Alignment**

All Newton primitives that have a primary axis (capsule, cylinder, cone) are aligned along the Z-axis in their local coordinate frame. The shape's transform determines its final position and orientation in the world or parent body frame.

**Center of Mass Considerations**

For most primitives, the center of mass coincides with the geometric origin. The cone is a notable exception:

* **Cone COM**: Located at ``(0, 0, -half_height/2)`` in the shape's local frame, which is 1/4 of the total height from the base toward the apex.

Collision Primitive Conventions Across Engines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following tables compare how different engines and formats define common collision primitives:

**Sphere Primitives**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - **System**
     - **Parameter Convention**
     - **Notes**
   * - **Newton**
     - ``radius``
     - Origin at center
   * - **MuJoCo**
     - ``size[0]`` = radius
     - Origin at center
   * - **USD (UsdGeomSphere)**
     - ``radius`` attribute
     - Origin at center
   * - **USD Physics**
     - ``radius`` attribute
     - Origin at center

**Box Primitives**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - **System**
     - **Parameter Convention**
     - **Notes**
   * - **Newton**
     - Half-extents (``hx``, ``hy``, ``hz``)
     - Distance from center to face
   * - **MuJoCo**
     - Half-sizes in ``size`` attribute
     - Can use ``fromto`` (Newton importer doesn't support)
   * - **USD (UsdGeomCube)**
     - ``size`` attribute (full dimensions)
     - Edge length, not half-extent
   * - **USD Physics**
     - ``halfExtents`` attribute
     - Matches Newton convention

**Capsule Primitives**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - **System**
     - **Parameter Convention**
     - **Notes**
   * - **Newton**
     - ``radius``, ``half_height`` (excludes caps)
     - Total length = 2*(radius + half_height)
   * - **MuJoCo**
     - ``size[0]`` = radius, ``size[1]`` = half-length (excludes caps)
     - Can also use ``fromto`` for endpoints
   * - **USD (UsdGeomCapsule)**
     - ``radius``, ``height`` (excludes caps)
     - Full height of cylindrical portion
   * - **USD Physics**
     - ``radius``, ``halfHeight`` (excludes caps)
     - Similar to Newton

**Cylinder Primitives**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - **System**
     - **Parameter Convention**
     - **Notes**
   * - **Newton**
     - ``radius``, ``half_height``
     - Extends along Z-axis
   * - **MuJoCo**
     - ``size[0]`` = radius, ``size[1]`` = half-length
     - Can use ``fromto``; Newton's MJCF importer maps to capsule
   * - **USD (UsdGeomCylinder)**
     - ``radius``, ``height`` (full height)
     - Visual shape
   * - **USD Physics**
     - ``radius``, ``halfHeight``
     - Newton's USD importer creates actual cylinders

**Cone Primitives**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - **System**
     - **Parameter Convention**
     - **Notes**
   * - **Newton**
     - ``radius`` (base), ``half_height``
     - COM offset at -half_height/2
   * - **MuJoCo**
     - Not supported
     - N/A
   * - **USD (UsdGeomCone)**
     - ``radius``, ``height`` (full height)
     - Visual representation
   * - **USD Physics**
     - ``radius``, ``halfHeight``
     - Physics representation

**Plane Primitives**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - **System**
     - **Definition Method**
     - **Normal Direction**
   * - **Newton**
     - Transform-based or plane equation
     - +Z of shape frame
   * - **MuJoCo**
     - Size and orientation in body frame
     - +Z of geom frame
   * - **USD**
     - No standard plane primitive
     - Implementation-specific

**Mesh Primitives**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - **System**
     - **Mesh Type**
     - **Notes**
   * - **Newton**
     - General triangle mesh
     - Can be non-convex
   * - **MuJoCo**
     - Convex hull only for collision
     - Visual mesh can be non-convex
   * - **USD (UsdGeomMesh)**
     - General polygon mesh
     - Visual representation
   * - **USD Physics**
     - Implementation-dependent
     - May use convex approximation

Import Handling
~~~~~~~~~~~~~~~

Newton's importers automatically handle convention differences when loading assets. No manual conversion is required when using these importers—they automatically transform shapes to Newton's conventions.

Practical Considerations
~~~~~~~~~~~~~~~~~~~~~~~~

**Creating Shapes Programmatically**

When using ModelBuilder to create shapes:

.. code-block:: python

   # Sphere - simple radius parameter
   builder.add_shape_sphere(body=0, radius=1.0)
   
   # Box uses half-extents
   builder.add_shape_box(body=0, hx=1.0, hy=0.5, hz=0.25)
   
   # Capsule half_height excludes caps
   # Total length = 2 * (radius + half_height) = 2 * (0.5 + 1.0) = 3.0
   builder.add_shape_capsule(body=0, radius=0.5, half_height=1.0)
   
   # Cylinder extends along Z-axis
   builder.add_shape_cylinder(body=0, radius=0.5, half_height=1.0)
   
   # Cone - note the COM offset affects dynamics
   # Base at z=-1.0, apex at z=+1.0, COM at z=-0.5
   builder.add_shape_cone(body=0, radius=0.5, half_height=1.0)
   
   # Plane normal points along +Z of shape frame
   builder.add_shape_plane(width=10.0, length=10.0)  # Bounded plane
   builder.add_shape_plane(width=0.0, length=0.0)    # Infinite plane
   
   # Mesh - general triangle mesh (can be non-convex)
   builder.add_shape_mesh(body=0, mesh=my_mesh)