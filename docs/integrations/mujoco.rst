.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

MuJoCo Integration
==================

:class:`~newton.solvers.SolverMuJoCo` translates a Newton :class:`~newton.Model`
into a `MuJoCo <https://github.com/google-deepmind/mujoco>`_ model and steps it
with `mujoco_warp <https://github.com/google-deepmind/mujoco_warp>`_.
Because MuJoCo has its own modelling conventions, some Newton properties are
mapped differently or not at all.  The sections below describe each conversion
in detail.

Coordinate conventions
----------------------

**Quaternion order.**
  Newton stores quaternions as ``(x, y, z, w)``; MuJoCo uses ``(w, x, y, z)``.
  The solver converts between the two automatically.  Be aware of this when
  inspecting raw MuJoCo model or data objects (e.g. via ``save_to_mjcf`` or
  the ``mj_model``/``mj_data`` attributes on the solver).


Joint types
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Newton type
     - MuJoCo type(s)
     - Notes
   * - ``FREE``
     - ``mjJNT_FREE``
     - Initial pose taken from ``body_q``.
   * - ``BALL``
     - ``mjJNT_BALL``
     - Per-axis actuators mapped via ``gear``.
   * - ``REVOLUTE``
     - ``mjJNT_HINGE``
     -
   * - ``PRISMATIC``
     - ``mjJNT_SLIDE``
     -
   * - ``D6``
     - Up to 3 × ``mjJNT_SLIDE`` + 3 × ``mjJNT_HINGE``
     - Each active linear/angular DOF becomes a separate MuJoCo joint
       (``_lin``/``_ang`` suffixes, with numeric indices when multiple axes
       are active).
   * - ``FIXED``
     - *(no joint)*
     - The child body is nested directly under its parent.  If the fixed joint
       connects to the world, the body is created as a **mocap** body.


Geometry types
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Newton type
     - MuJoCo type
     - Notes
   * - ``SPHERE``
     - ``mjGEOM_SPHERE``
     -
   * - ``CAPSULE``
     - ``mjGEOM_CAPSULE``
     -
   * - ``CYLINDER``
     - ``mjGEOM_CYLINDER``
     -
   * - ``BOX``
     - ``mjGEOM_BOX``
     -
   * - ``ELLIPSOID``
     - ``mjGEOM_ELLIPSOID``
     -
   * - ``PLANE``
     - ``mjGEOM_PLANE``
     - Must be attached to the world body.  Rendered size defaults to
       ``5 × 5 × 5``.
   * - ``HFIELD``
     - ``mjGEOM_HFIELD``
     - Heightfield data is normalized ``[0, 1]``; the geom origin is shifted
       by ``min_z`` so the lowest point is at the correct world height.
   * - ``MESH`` / ``CONVEX_MESH``
     - ``mjGEOM_MESH``
     - MuJoCo only supports **convex** collision meshes.  Non-convex meshes are
       convex-hulled automatically, which changes the collision boundary.
       ``maxhullvert`` is forwarded from the mesh source when set.

**Sites** (shapes with the ``SITE`` flag) are converted to MuJoCo sites, which
are non-colliding reference frames used for sensor attachment and spatial
tendons.


Shape parameters
----------------

**Friction.**
  Newton's ``shape_material_mu``, ``shape_material_mu_torsional``, and
  ``shape_material_mu_rolling`` map directly to MuJoCo's three-element
  geom ``friction`` vector: ``(sliding, torsional, rolling)``.

**Stiffness and damping (solref).**
  Newton's ``shape_material_ke`` (stiffness) and ``shape_material_kd``
  (damping) are converted to MuJoCo's geom ``solref`` ``(timeconst, dampratio)``
  pair.  When either value is zero or negative, the solver falls back to MuJoCo's defaults
  (``timeconst = 0.02``, ``dampratio = 1.0``).

**Joint-limit stiffness and damping (solref_limit).**
  ``joint_limit_ke`` and ``joint_limit_kd`` are forwarded as negative
  ``solref_limit`` values ``(-ke, -kd)``, which MuJoCo interprets as direct
  stiffness/damping rather than time-constant/damp-ratio.

**Margin.**
  Newton's ``shape_margin`` maps to MuJoCo ``geom_margin``.

**MuJoCo-specific custom attributes.**
  Many MuJoCo-specific parameters are stored in Newton's ``mujoco``
  custom-attribute namespace and forwarded to the MuJoCo model when present.
  These cover geom properties, joint properties, equality constraints, tendons,
  general actuators, and solver options.  See
  :ref:`mujoco-custom-attributes-and-frequencies` below for the full catalog
  and :doc:`/concepts/custom_attributes` for the general custom-attribute
  mechanism.


Collision filtering
-------------------

Newton uses integer ``shape_collision_group`` labels to control which shapes
can collide.  MuJoCo uses ``contype``/``conaffinity`` bitmasks with a different
semantic: two geoms collide when
``(contype_A & conaffinity_B) || (contype_B & conaffinity_A)`` is non-zero.

The solver bridges the two systems with **graph coloring**.  Shapes that must
*not* collide are assigned the same color; each color maps to one bit in
``contype``.  ``conaffinity`` is set to the complement so that same-color
geoms never match.  Up to 32 colors are supported (one per ``contype`` bit).
If the graph requires more than 32 colors, shapes with color index ≥ 32 fall
back to MuJoCo defaults (``contype=1``, ``conaffinity=1``) and will collide
with all other shapes, silently bypassing the intended filtering.

Non-colliding shapes (no ``COLLIDE_SHAPES`` flag, or ``collision_group == 0``)
get ``contype = conaffinity = 0``.

Additionally, body pairs for which all shape-shape combinations are filtered
are registered as ``<exclude>`` elements.


Mass and inertia
----------------

Bodies with positive mass have their mass, center-of-mass offset (``ipos``),
and inertia tensor set explicitly (``explicitinertial="true"``).  When the
inertia tensor is diagonal the solver uses ``diaginertia``; otherwise it uses
``fullinertia``.

Zero-mass bodies (e.g. sensor frames) omit mass and inertia entirely, letting
MuJoCo derive them from child geoms (``inertiafromgeom="auto"``).


Actuators
---------

Newton's per-DOF ``joint_target_mode`` determines which MuJoCo general
actuators are created:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - MuJoCo actuator(s)
   * - ``POSITION``
     - One actuator: ``gainprm = [kp]``, ``biasprm = [0, -kp, -kd]``.
   * - ``VELOCITY``
     - One actuator: ``gainprm = [kd]``, ``biasprm = [0, 0, -kd]``.
   * - ``POSITION_VELOCITY``
     - Two actuators — a position actuator (``gainprm = [kp]``,
       ``biasprm = [0, -kp, 0]``) and a velocity actuator
       (``gainprm = [kd]``, ``biasprm = [0, 0, -kd]``).
   * - ``NONE``
     - No actuator created for this DOF.

``joint_effort_limit`` is forwarded as ``actfrcrange`` on the joint (for
prismatic, revolute, and D6 joints) or as ``forcerange`` on the actuator (for
ball joints).

Additional MuJoCo general actuators (motors, etc.) can be attached through
custom attributes and are appended after the joint-target actuators.


Equality constraints
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Newton type
     - MuJoCo type
     - Notes
   * - ``CONNECT``
     - ``mjEQ_CONNECT``
     - Anchor forwarded in ``data[0:3]``.
   * - ``WELD``
     - ``mjEQ_WELD``
     - Relative pose and torque scale forwarded.
   * - ``JOINT``
     - ``mjEQ_JOINT``
     - Polynomial coefficients forwarded in ``data[0:5]``.
   * - Mimic
     - ``mjEQ_JOINT``
     - ``coef0`` / ``coef1`` mapped to polynomial coefficients.
       Only ``REVOLUTE`` and ``PRISMATIC`` joints are supported.

``eq_solref`` custom attributes are forwarded when present.


Solver options
--------------

Solver parameters follow a three-level resolution priority:

1. **Constructor argument** — value passed to :class:`~newton.solvers.SolverMuJoCo`.
2. **Custom attribute** (``model.mujoco.<option>``) — supports per-world values.
   These attributes are typically populated automatically when importing USD or
   MJCF assets.
3. **Default** — the table below lists Newton defaults alongside MuJoCo
   defaults for reference.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Option
     - Newton default
     - MuJoCo default
     - Notes
   * - ``solver``
     - ``newton``
     - ``newton``
     -
   * - ``integrator``
     - ``implicitfast``
     - ``euler``
     - ``implicitfast`` provides better stability for stiff systems.
   * - ``cone``
     - ``pyramidal``
     - ``pyramidal``
     -
   * - ``iterations``
     - 100
     - 100
     -
   * - ``ls_iterations``
     - 50
     - 50
     -


Multi-world support
-------------------

When ``separate_worlds=True`` (the default for GPU mode with multiple worlds),
the solver builds a
MuJoCo model from the **first world** only and replicates it across all worlds
via ``mujoco_warp``.  This requires all Newton worlds to be structurally
identical (same bodies, joints, and shapes).  Global entities (those with a
negative world index) may only include static shapes — they are shared across
all worlds without replication.


Collision pipeline
------------------

By default :class:`~newton.solvers.SolverMuJoCo` uses MuJoCo's built-in
collision detection (``use_mujoco_contacts=True``).  Alternatively, you can set
``use_mujoco_contacts=False`` and pass contacts computed by Newton's own
collision pipeline into :meth:`~newton.solvers.SolverMuJoCo.step`.  Newton's
pipeline supports non-convex meshes, SDF-based contacts, and hydroelastic
contacts, which are not available through MuJoCo's collision detection.


Caveats
-------

**geom_gap is always zero.**
  MuJoCo's ``gap`` parameter controls *inactive* contact generation — contacts
  that are detected but do not produce constraint forces until penetration
  exceeds the gap threshold.  Newton does not use this concept: when the MuJoCo
  collision pipeline is active it runs every step, so there is no benefit to
  keeping inactive contacts around.  Setting ``geom_gap > 0`` would inflate
  ``geom_margin``, which disables MuJoCo's multicontact and degrades contact
  quality.  Therefore :class:`~newton.solvers.SolverMuJoCo` always sets
  ``geom_gap = 0`` regardless of the Newton :attr:`~newton.ModelBuilder.ShapeConfig.gap`
  value.  MJCF/USD ``gap`` values are still imported into
  :attr:`~newton.ModelBuilder.ShapeConfig.gap` in the Newton model, but they are
  not forwarded to the MuJoCo solver.

**shape_collision_radius is ignored.**
  MuJoCo computes bounding-sphere radii (``geom_rbound``) internally from the
  geometry definition.  Newton's ``shape_collision_radius`` is not forwarded.

**Non-convex meshes are convex-hulled.**
  MuJoCo only supports convex collision geometry.  If a Newton ``MESH`` shape
  is non-convex, MuJoCo will automatically compute its convex hull, changing
  the effective collision boundary.

**Velocity limits are not forwarded.**
  Newton's ``joint_velocity_limit`` has no MuJoCo equivalent and is ignored.

**Body ordering must be depth-first.**
  The solver sorts joints in depth-first topological order for MuJoCo's
  kinematic tree.  If the Newton model's joint order differs, a warning is
  emitted because kinematics may diverge.


.. _mujoco-kinematic-links-and-fixed-roots:

Kinematic Links and Fixed Roots
-------------------------------

Newton only allows ``is_kinematic=True`` on articulation roots, so in the
MuJoCo exporter a "kinematic link" always means a kinematic root body. Any
descendants of that root can still be dynamic and are exported normally.

At runtime, :class:`~newton.solvers.SolverMuJoCo` keeps kinematic roots
user-prescribed rather than dynamically integrated:

- When converting MuJoCo state back to Newton, the previous Newton
  :attr:`newton.State.joint_q` and :attr:`newton.State.joint_qd` values are
  passed through for kinematic roots instead of being overwritten from MuJoCo's
  integrated ``qpos`` and ``qvel``.
- Applied body wrenches and joint forces targeting kinematic bodies are ignored
  on the MuJoCo side.
- Kinematic bodies still participate in contacts, so they can act as moving or
  fixed obstacles for dynamic bodies.

During export, :class:`~newton.solvers.SolverMuJoCo` maps roots according to
their joint type:

- **Kinematic roots with non-fixed joints** are exported as ordinary MuJoCo
  joints with the same Newton joint type and DOFs. The solver assigns a very
  large internal armature to those DOFs so MuJoCo treats them like prescribed,
  effectively infinite-mass coordinates.
- **Roots attached to world with a fixed joint** are exported as MuJoCo mocap
  bodies. This applies to both kinematic and non-kinematic Newton roots
  attached to world by :class:`~newton.JointType.FIXED`. MuJoCo has no joint
  coordinates for a fixed root, so Newton drives the pose through
  ``mjData.mocap_pos`` and ``mjData.mocap_quat`` instead.
- **World-attached shapes that are not part of an articulation** remain
  ordinary static MuJoCo geometry rather than mocap bodies.

If you edit :attr:`newton.Model.joint_X_p` or :attr:`newton.Model.joint_X_c`
for a fixed-root articulation after constructing the solver, call
``solver.notify_model_changed(newton.solvers.SolverNotifyFlags.JOINT_PROPERTIES)``
to synchronize the updated fixed-root poses into MuJoCo.


.. _mujoco-custom-attributes-and-frequencies:

Custom Attributes and Frequencies
---------------------------------

:meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` registers
MuJoCo-specific custom attributes in the ``mujoco`` namespace and custom
frequencies for variable-length entity types.  Call it on a
:class:`~newton.ModelBuilder` **before** loading assets.  After
:meth:`~newton.ModelBuilder.finalize`, the attributes are accessible as
``model.mujoco.<name>``.

See the :meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` API
documentation for the full catalog of registered frequencies and attributes,
and :doc:`/concepts/custom_attributes` for background on Newton's
custom-attribute system.


.. _mujoco-usd-schemas:

Supported ``mjc:`` USD Attributes
---------------------------------

When loading USD assets, Newton can parse MuJoCo-specific attributes via the
``mjc:`` USD attribute prefix.  Most of these attributes come from the
`mjcPhysics USD schema <https://github.com/google-deepmind/mujoco/blob/main/src/experimental/usd/mjcPhysics/generatedSchema.usda>`_
developed by the MuJoCo team.  The schema is not yet published as a registered
USD schema, so Newton reads ``mjc:``-prefixed attributes directly rather than
relying on applied schemas.

Newton also defines a small number of ``mjc:``-prefixed attributes that are
**not** in the mjcPhysics schema (marked with :sup:`ext` in the tables below).
These correspond to MuJoCo XML attributes that do not yet have a schema
counterpart.

Attributes reach Newton through two code paths:

- **Schema resolver** (``SchemaResolverMjc``) — maps ``mjc:`` attributes to
  Newton's built-in model properties (e.g. ``armature``, ``margin``, ``ke``).
- **Custom attribute registration** — attributes registered via
  :meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` with a
  ``usd_attribute_name`` starting with ``mjc:`` are parsed into the
  ``mujoco`` namespace (e.g. ``model.mujoco.condim``).  See
  :ref:`mujoco-custom-attribute-parsing` for details.

The tables below list every ``mjc:`` attribute that Newton reads from USD.
Attributes not listed here are ignored during import.

**Scene — built-in properties** (``PhysicsScene``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Newton property
     - Default
   * - ``mjc:option:iterations``
     - ``max_solver_iterations``
     - 100
   * - ``mjc:option:timestep``
     - ``time_steps_per_second``
     - 0.002 (→ 500 Hz)
   * - ``mjc:flag:gravity``
     - ``gravity_enabled``
     - ``True``

**Scene — solver options (per-world)** (``PhysicsScene`` → ``mujoco.*``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Custom attribute
     - Default
   * - ``mjc:option:impratio``
     - ``mujoco.impratio``
     - 1.0
   * - ``mjc:option:tolerance``
     - ``mujoco.tolerance``
     - 1e-8
   * - ``mjc:option:ls_tolerance``
     - ``mujoco.ls_tolerance``
     - 0.01
   * - ``mjc:option:ccd_tolerance``
     - ``mujoco.ccd_tolerance``
     - 1e-6
   * - ``mjc:option:density``
     - ``mujoco.density``
     - 0.0
   * - ``mjc:option:viscosity``
     - ``mujoco.viscosity``
     - 0.0
   * - ``mjc:option:wind``
     - ``mujoco.wind``
     - ``(0, 0, 0)``
   * - ``mjc:option:magnetic``
     - ``mujoco.magnetic``
     - ``(0, -0.5, 0)``

**Scene — solver options (per-model)** (``PhysicsScene`` → ``mujoco.*``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Custom attribute
     - Default
   * - ``mjc:option:iterations``
     - ``mujoco.iterations``
     - 100
   * - ``mjc:option:ls_iterations``
     - ``mujoco.ls_iterations``
     - 50
   * - ``mjc:option:ccd_iterations``
     - ``mujoco.ccd_iterations``
     - 35
   * - ``mjc:option:sdf_iterations``
     - ``mujoco.sdf_iterations``
     - 10
   * - ``mjc:option:sdf_initpoints``
     - ``mujoco.sdf_initpoints``
     - 40
   * - ``mjc:option:integrator``
     - ``mujoco.integrator``
     - 3 (``implicitfast``)
   * - ``mjc:option:solver``
     - ``mujoco.solver``
     - 2 (``newton``)
   * - ``mjc:option:cone``
     - ``mujoco.cone``
     - 0 (``pyramidal``)
   * - ``mjc:option:jacobian``
     - ``mujoco.jacobian``
     - 2 (``auto``)

**Joint — built-in properties** (``PhysicsRevoluteJoint``, ``PhysicsPrismaticJoint``, etc.):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Newton property
     - Default
   * - ``mjc:armature``
     - ``armature``
     - 0.0
   * - ``mjc:frictionloss`` :sup:`ext`
     - ``friction``
     - 0.0
   * - ``mjc:solref``
     - ``limit_*_ke`` / ``limit_*_kd``
     - ``[0.02, 1.0]``

The ``mjc:solref`` attribute is mapped to per-axis limit stiffness and damping
for all joint DOFs (``transX``, ``transY``, ``transZ``, ``rotX``, ``rotY``,
``rotZ``, ``linear``, ``angular``).

.. note::

   The ``solref`` → ``ke``/``kd`` conversion currently produces mass-normalized
   values rather than force-based values.  See
   `issue #2009 <https://github.com/newton-physics/newton/issues/2009>`_
   for details.

**Joint — custom attributes** (joint prims → ``mujoco.*``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Custom attribute
     - Default
   * - ``mjc:margin``
     - ``mujoco.limit_margin``
     - 0.0
   * - ``mjc:solimplimit`` :sup:`ext`
     - ``mujoco.solimplimit``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
   * - ``mjc:solreffriction`` :sup:`ext`
     - ``mujoco.solreffriction``
     - ``(0.02, 1.0)``
   * - ``mjc:solimpfriction`` :sup:`ext`
     - ``mujoco.solimpfriction``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
   * - ``mjc:stiffness`` :sup:`ext`
     - ``mujoco.dof_passive_stiffness``
     - 0.0
   * - ``mjc:damping``
     - ``mujoco.dof_passive_damping``
     - 0.0
   * - ``mjc:springref`` :sup:`ext`
     - ``mujoco.dof_springref``
     - 0.0
   * - ``mjc:ref`` :sup:`ext`
     - ``mujoco.dof_ref``
     - 0.0
   * - ``mjc:actuatorgravcomp``
     - ``mujoco.jnt_actgravcomp``
     - ``False``

**Shape — built-in properties** (``PhysicsCollisionAPI``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Newton property
     - Default
   * - ``mjc:maxhullvert``
     - ``max_hull_vertices``
     - -1
   * - ``mjc:margin``
     - ``margin``
     - 0.0
   * - ``mjc:gap``
     - ``gap``
     - 0.0
   * - ``mjc:solref``
     - ``ke`` / ``kd``
     - ``[0.02, 1.0]``

.. note::

   Newton computes ``margin = mjc:margin - mjc:gap``.
   The ``solref`` → ``ke``/``kd`` conversion has the same
   mass-normalization issue as the joint mapping.  See
   `issue #2009 <https://github.com/newton-physics/newton/issues/2009>`_
   for details.

**Shape — custom attributes** (collision prims → ``mujoco.*``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Custom attribute
     - Default
   * - ``mjc:condim``
     - ``mujoco.condim``
     - 3
   * - ``mjc:priority``
     - ``mujoco.geom_priority``
     - 0
   * - ``mjc:solimp``
     - ``mujoco.geom_solimp``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
   * - ``mjc:solmix``
     - ``mujoco.geom_solmix``
     - 1.0

**Material** (``PhysicsMaterialAPI``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Newton property
     - Default
   * - ``mjc:torsionalfriction`` :sup:`ext`
     - ``mu_torsional``
     - 0.005
   * - ``mjc:rollingfriction`` :sup:`ext`
     - ``mu_rolling``
     - 0.0001
   * - ``mjc:priority``
     - ``priority``
     - 0
   * - ``mjc:solmix``
     - ``weight``
     - 1.0
   * - ``mjc:solref``
     - ``stiffness`` / ``damping``
     - ``[0.02, 1.0]``

.. note::

   The ``solref`` → ``stiffness``/``damping`` conversion has the same
   mass-normalization issue as the joint mapping.  See
   `issue #2009 <https://github.com/newton-physics/newton/issues/2009>`_
   for details.

**Body** (``PhysicsRigidBodyAPI``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Newton property
     - Default
   * - ``mjc:damping``
     - ``rigid_body_linear_damping``
     - 0.0
   * - ``mjc:gravcomp`` :sup:`ext`
     - ``mujoco.gravcomp``
     - 0.0

**Equality constraint** (``mujoco.*``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Custom attribute
     - Default
   * - ``mjc:solref``
     - ``mujoco.eq_solref``
     - ``(0.02, 1.0)``
   * - ``mjc:solimp``
     - ``mujoco.eq_solimp``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``

**Actuator** (``MjcActuator``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Newton property
     - Default
   * - ``mjc:ctrlRange:min`` / ``max``
     - ``ctrl_low`` / ``ctrl_high``
     - 0.0
   * - ``mjc:forceRange:min`` / ``max``
     - ``force_low`` / ``force_high``
     - 0.0
   * - ``mjc:actRange:min`` / ``max``
     - ``act_low`` / ``act_high``
     - 0.0
   * - ``mjc:lengthRange:min`` / ``max``
     - ``length_low`` / ``length_high``
     - 0.0
   * - ``mjc:gainPrm``
     - ``gainPrm`` / ``mujoco.actuator_gainprm``
     - ``[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
   * - ``mjc:gainType``
     - ``gainType`` / ``mujoco.actuator_gaintype``
     - ``"fixed"``
   * - ``mjc:biasPrm``
     - ``biasPrm`` / ``mujoco.actuator_biasprm``
     - ``[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
   * - ``mjc:biasType``
     - ``biasType`` / ``mujoco.actuator_biastype``
     - ``"none"``
   * - ``mjc:dynPrm``
     - ``dynPrm`` / ``mujoco.actuator_dynprm``
     - ``[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
   * - ``mjc:dynType``
     - ``dynType`` / ``mujoco.actuator_dyntype``
     - ``"none"``
   * - ``mjc:gear``
     - ``gear`` / ``mujoco.actuator_gear``
     - ``[1, 0, 0, 0, 0, 0]``
   * - ``mjc:actDim``
     - ``mujoco.actuator_actdim``
     - -1 (auto)
   * - ``mjc:actEarly``
     - ``mujoco.actuator_actearly``
     - ``False``
   * - ``mjc:ctrlLimited``
     - ``mujoco.actuator_ctrllimited``
     - ``"auto"``
   * - ``mjc:forceLimited``
     - ``mujoco.actuator_forcelimited``
     - ``"auto"``
   * - ``mjc:actLimited``
     - ``mujoco.actuator_actlimited``
     - ``"auto"``
   * - ``mjc:inheritRange``
     - *(resolves range from transmission target)*
     -
   * - ``mjc:target``
     - *(USD relationship → transmission target)*
     -

**Tendon** (``MjcTendon``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Custom attribute
     - Default
   * - ``mjc:stiffness`` :sup:`ext`
     - ``mujoco.tendon_stiffness``
     - 0.0
   * - ``mjc:damping``
     - ``mujoco.tendon_damping``
     - 0.0
   * - ``mjc:frictionloss`` :sup:`ext`
     - ``mujoco.tendon_frictionloss``
     - 0.0
   * - ``mjc:limited`` :sup:`ext`
     - ``mujoco.tendon_limited``
     - 2 (auto)
   * - ``mjc:range:min`` / ``max`` :sup:`ext`
     - ``mujoco.tendon_range``
     - ``(0, 0)``
   * - ``mjc:margin``
     - ``mujoco.tendon_margin``
     - 0.0
   * - ``mjc:solreflimit`` :sup:`ext`
     - ``mujoco.tendon_solref_limit``
     - ``(0.02, 1.0)``
   * - ``mjc:solimplimit`` :sup:`ext`
     - ``mujoco.tendon_solimp_limit``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
   * - ``mjc:solreffriction`` :sup:`ext`
     - ``mujoco.tendon_solref_friction``
     - ``(0.02, 1.0)``
   * - ``mjc:solimpfriction`` :sup:`ext`
     - ``mujoco.tendon_solimp_friction``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
   * - ``mjc:armature``
     - ``mujoco.tendon_armature``
     - 0.0
   * - ``mjc:springlength`` :sup:`ext`
     - ``mujoco.tendon_springlength``
     - ``(-1, -1)``
   * - ``mjc:actuatorfrclimited``
     - ``mujoco.tendon_actuator_force_limited``
     - 2 (auto)
   * - ``mjc:actuatorfrcrange:min`` / ``max``
     - ``mujoco.tendon_actuator_force_range``
     - ``(0, 0)``

:sup:`ext` = Newton extension — not defined in the mjcPhysics schema.  These
correspond to MuJoCo XML attributes that do not yet have a schema counterpart.


Unsupported mjcPhysics schema attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following mjcPhysics schema attributes are **not** parsed by Newton.
Authoring them on USD prims has no effect.

- **Compiler options** (``mjc:compiler:*``): all 14 compiler attributes
  (``alignFree``, ``angle``, ``autoLimits``, ``balanceInertia``, etc.)
- **Flags** (``mjc:flag:*``): all flags except ``mjc:flag:gravity``
  (``actuation``, ``clampctrl``, ``contact``, ``equality``, ``multiccd``,
  ``nativeccd``, ``warmstart``, etc.)
- **Keyframe** (``MjcKeyframe``): ``mjc:qpos``, ``mjc:qvel``,
  ``mjc:mpos``, ``mjc:mquat``
- **Shape / collision**: ``mjc:inertia`` (mesh inertia mode),
  ``mjc:shellinertia``, ``mjc:group``
- **Actuator**: ``mjc:act``, ``mjc:ctrl``, ``mjc:crankLength``,
  ``mjc:jointInParent``, ``mjc:refSite``, ``mjc:sliderSite``,
  ``mjc:group``
- **Solver options**: ``mjc:option:noslip_iterations``,
  ``mjc:option:noslip_tolerance``, ``mjc:option:actuatorgroupdisable``,
  ``mjc:option:o_friction``, ``mjc:option:o_margin``,
  ``mjc:option:o_solimp``, ``mjc:option:o_solref``


.. _mujoco-custom-attribute-parsing:

Custom attribute parsing from USD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When :meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` registers
a custom attribute with a ``usd_attribute_name`` (e.g.
``usd_attribute_name="mjc:condim"``), the attribute is parsed from USD prims
during :meth:`~newton.ModelBuilder.add_usd`.  This is how most MuJoCo-specific
parameters (solver options, joint impedance, tendon properties, etc.) reach
Newton from USD files.

The full catalog of registered custom attributes — including those not parsed
from USD — is documented in the
:meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` API docstring.
After :meth:`~newton.ModelBuilder.finalize`, custom attributes are accessible as
``model.mujoco.<name>``, ``state.mujoco.<name>``, or
``control.mujoco.<name>`` depending on their assignment.
