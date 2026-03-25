.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

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

**Custom frequencies:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Frequency
     - Description
   * - ``mujoco:pair``
     - Explicit contact pairs (MJCF ``<contact><pair>``).
   * - ``mujoco:actuator``
     - General MuJoCo actuators (MJCF ``<actuator>`` / USD ``MjcActuator``).
   * - ``mujoco:tendon``
     - Fixed and spatial tendons (MJCF ``<tendon>`` / USD ``MjcTendon``).
   * - ``mujoco:tendon_joint``
     - Per-joint entries inside fixed tendons.
   * - ``mujoco:tendon_wrap``
     - Per-element entries inside spatial tendon wrap paths.

**Geom / shape attributes** (frequency: ``SHAPE``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``condim``
     - ``int32``
     - Contact dimensionality (default 3).
   * - ``geom_priority``
     - ``int32``
     - Contact-parameter mixing priority (default 0).
   * - ``geom_solimp``
     - ``vec5``
     - Solver impedance parameters.
   * - ``geom_solmix``
     - ``float32``
     - Solver mixing weight (default 1.0).

**Joint / DOF attributes** (frequency: ``JOINT_DOF``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``limit_margin``
     - ``float32``
     - Joint-limit margin [m or rad].
   * - ``solimplimit``
     - ``vec5``
     - Solver impedance for joint limits.
   * - ``solreffriction``
     - ``vec2``
     - Solver reference for joint friction.
   * - ``solimpfriction``
     - ``vec5``
     - Solver impedance for joint friction.
   * - ``dof_passive_stiffness``
     - ``float32``
     - Passive spring stiffness.
   * - ``dof_passive_damping``
     - ``float32``
     - Passive damping coefficient.
   * - ``dof_springref``
     - ``float32``
     - Spring reference position [m or rad].
   * - ``dof_ref``
     - ``float32``
     - Joint reference position [m or rad].
   * - ``jnt_actgravcomp``
     - ``bool``
     - Per-DOF actuator gravity compensation flag.

**Body attributes** (frequency: ``BODY``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``gravcomp``
     - ``float32``
     - Gravity compensation scaling factor.

**Equality constraint attributes** (frequency: ``EQUALITY_CONSTRAINT``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``eq_solref``
     - ``vec2``
     - Solver reference for equality constraints.
   * - ``eq_solimp``
     - ``vec5``
     - Solver impedance for equality constraints.

**Solver options — per-world** (frequency: ``WORLD``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``impratio``
     - ``float32``
     - Impedance ratio (default 1.0).
   * - ``tolerance``
     - ``float32``
     - Solver tolerance (default 1e-8).
   * - ``ls_tolerance``
     - ``float32``
     - Line-search tolerance (default 0.01).
   * - ``ccd_tolerance``
     - ``float32``
     - CCD tolerance (default 1e-6).
   * - ``density``
     - ``float32``
     - Medium density for viscous forces.
   * - ``viscosity``
     - ``float32``
     - Medium viscosity.
   * - ``wind``
     - ``vec3``
     - Wind velocity.
   * - ``magnetic``
     - ``vec3``
     - Magnetic flux.

**Solver options — per-model** (frequency: ``ONCE``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``iterations``
     - ``int32``
     - Maximum solver iterations (default 100).
   * - ``ls_iterations``
     - ``int32``
     - Maximum line-search iterations (default 50).
   * - ``ccd_iterations``
     - ``int32``
     - Maximum CCD iterations (default 35).
   * - ``sdf_iterations``
     - ``int32``
     - Maximum SDF iterations (default 10).
   * - ``sdf_initpoints``
     - ``int32``
     - SDF initial sample points (default 40).
   * - ``integrator``
     - ``int32``
     - Integration scheme (default 3 = ``implicitfast``).
   * - ``solver``
     - ``int32``
     - Constraint solver (default 2 = ``newton``).
   * - ``cone``
     - ``int32``
     - Friction cone type (default 0 = ``pyramidal``).
   * - ``jacobian``
     - ``int32``
     - Jacobian type (default 2 = ``auto``).
   * - ``autolimits``
     - ``bool``
     - Enable automatic limit inference (default ``True``).

**Pair attributes** (frequency: ``mujoco:pair``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``pair_world``
     - ``int32``
     - World index for this pair.
   * - ``pair_geom1``
     - ``int32``
     - First shape index.
   * - ``pair_geom2``
     - ``int32``
     - Second shape index.
   * - ``pair_condim``
     - ``int32``
     - Contact dimensionality (default 3).
   * - ``pair_solref``
     - ``vec2``
     - Solver reference.
   * - ``pair_solreffriction``
     - ``vec2``
     - Solver reference for friction.
   * - ``pair_solimp``
     - ``vec5``
     - Solver impedance.
   * - ``pair_margin``
     - ``float32``
     - Contact margin.
   * - ``pair_gap``
     - ``float32``
     - Contact gap.
   * - ``pair_friction``
     - ``vec5``
     - Five-element friction vector.

**Actuator attributes** (frequency: ``mujoco:actuator``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``joint_dof_label``
     - ``str``
     - DOF label strings (frequency: ``JOINT_DOF``).
   * - ``actuator_trnid``
     - ``vec2i``
     - Transmission target index pair.
   * - ``actuator_target_label``
     - ``str``
     - Target path label resolved from USD.
   * - ``actuator_trntype``
     - ``int32``
     - Transmission type (0 = joint).
   * - ``actuator_dyntype``
     - ``int32``
     - Activation dynamics type (0 = none).
   * - ``actuator_gaintype``
     - ``int32``
     - Gain type (0 = fixed).
   * - ``actuator_biastype``
     - ``int32``
     - Bias type (0 = none).
   * - ``actuator_world``
     - ``int32``
     - World index.
   * - ``actuator_ctrllimited``
     - ``int32``
     - Control-range limiting tri-state (2 = auto).
   * - ``actuator_forcelimited``
     - ``int32``
     - Force-range limiting tri-state (2 = auto).
   * - ``actuator_ctrlrange``
     - ``vec2``
     - Control range.
   * - ``actuator_has_ctrlrange``
     - ``int32``
     - Whether ``ctrlrange`` was explicitly authored.
   * - ``actuator_forcerange``
     - ``vec2``
     - Force range.
   * - ``actuator_has_forcerange``
     - ``int32``
     - Whether ``forcerange`` was explicitly authored.
   * - ``actuator_gear``
     - ``vec6``
     - Gear ratio vector.
   * - ``actuator_cranklength``
     - ``float32``
     - Crank length for slider-crank transmissions.
   * - ``actuator_dynprm``
     - ``vec10``
     - Activation dynamics parameters.
   * - ``actuator_gainprm``
     - ``vec10``
     - Gain parameters.
   * - ``actuator_biasprm``
     - ``vec10``
     - Bias parameters.
   * - ``actuator_actlimited``
     - ``int32``
     - Activation-range limiting tri-state (2 = auto).
   * - ``actuator_actrange``
     - ``vec2``
     - Activation range.
   * - ``actuator_has_actrange``
     - ``int32``
     - Whether ``actrange`` was explicitly authored.
   * - ``actuator_actdim``
     - ``int32``
     - Activation state dimension (-1 = auto).
   * - ``actuator_actearly``
     - ``bool``
     - Apply activation at start of step.
   * - ``ctrl``
     - ``float32``
     - Control signal (assignment: ``CONTROL``).
   * - ``ctrl_source``
     - ``int32``
     - Control source enum.

**Tendon attributes** (frequency: ``mujoco:tendon``):

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``tendon_world``
     - ``int32``
     - World index.
   * - ``tendon_stiffness``
     - ``float32``
     - Spring stiffness.
   * - ``tendon_damping``
     - ``float32``
     - Damping coefficient.
   * - ``tendon_frictionloss``
     - ``float32``
     - Friction loss.
   * - ``tendon_limited``
     - ``int32``
     - Length-limit tri-state (2 = auto).
   * - ``tendon_range``
     - ``vec2``
     - Length range.
   * - ``tendon_margin``
     - ``float32``
     - Length-limit margin.
   * - ``tendon_solref_limit``
     - ``vec2``
     - Solver reference for length limits.
   * - ``tendon_solimp_limit``
     - ``vec5``
     - Solver impedance for length limits.
   * - ``tendon_solref_friction``
     - ``vec2``
     - Solver reference for friction.
   * - ``tendon_solimp_friction``
     - ``vec5``
     - Solver impedance for friction.
   * - ``tendon_armature``
     - ``float32``
     - Armature.
   * - ``tendon_springlength``
     - ``vec2``
     - Spring rest length (-1 = use model length).
   * - ``tendon_joint_adr``
     - ``int32``
     - Start address into joint arrays.
   * - ``tendon_joint_num``
     - ``int32``
     - Number of joints in this tendon.
   * - ``tendon_actuator_force_range``
     - ``vec2``
     - Actuator force range.
   * - ``tendon_actuator_force_limited``
     - ``int32``
     - Actuator force limiting tri-state (2 = auto).
   * - ``tendon_label``
     - ``str``
     - Tendon name string.
   * - ``tendon_type``
     - ``int32``
     - Tendon type (0 = fixed, 1 = spatial).
   * - ``tendon_wrap_adr``
     - ``int32``
     - Start address into wrap-path arrays.
   * - ``tendon_wrap_num``
     - ``int32``
     - Number of wrap elements.
   * - ``tendon_joint``
     - ``int32``
     - Joint index (frequency: ``mujoco:tendon_joint``).
   * - ``tendon_coef``
     - ``float32``
     - Joint coefficient (frequency: ``mujoco:tendon_joint``).
   * - ``tendon_wrap_type``
     - ``int32``
     - Wrap element type (frequency: ``mujoco:tendon_wrap``).
   * - ``tendon_wrap_shape``
     - ``int32``
     - Shape index for geom wraps (frequency: ``mujoco:tendon_wrap``).
   * - ``tendon_wrap_sidesite``
     - ``int32``
     - Side-site shape index (frequency: ``mujoco:tendon_wrap``).
   * - ``tendon_wrap_prm``
     - ``float32``
     - Wrap parameter (frequency: ``mujoco:tendon_wrap``).

See :doc:`/concepts/custom_attributes` for background on Newton's
custom-attribute system.


.. _mujoco-usd-schemas:

Mjc USD Schemas
---------------

When loading USD assets, Newton can parse MuJoCo-specific attributes via the
``mjc:`` USD attribute prefix.  This is handled by the internal
``SchemaResolverMjc`` resolver, which maps ``mjc:``-prefixed USD attributes to
Newton model properties during :meth:`~newton.ModelBuilder.add_usd`.  The
``mjc:`` convention means that MuJoCo attributes are named ``mjc:attr`` in USD
files rather than ``newton:mujoco:attr``.

The following tables list supported ``mjc:`` attributes per USD prim type.

**Scene** (``PhysicsScene``):

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

**Joint** (``PhysicsRevoluteJoint``, ``PhysicsPrismaticJoint``, etc.):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Newton property
     - Default
   * - ``mjc:armature``
     - ``armature``
     - 0.0
   * - ``mjc:frictionloss``
     - ``friction``
     - 0.0
   * - ``mjc:solref``
     - ``limit_*_ke`` / ``limit_*_kd``
     - ``[0.02, 1.0]``

The ``mjc:solref`` attribute is mapped to per-axis limit stiffness and damping
for all joint DOFs (``transX``, ``transY``, ``transZ``, ``rotX``, ``rotY``,
``rotZ``, ``linear``, ``angular``).

**Shape** (``PhysicsCollisionAPI``):

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

**Material** (``PhysicsMaterialAPI``):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - USD attribute
     - Newton property
     - Default
   * - ``mjc:torsionalfriction``
     - ``mu_torsional``
     - 0.005
   * - ``mjc:rollingfriction``
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
     - ``gainPrm``
     - ``[1, 0, …, 0]``
   * - ``mjc:gainType``
     - ``gainType``
     - ``"fixed"``
   * - ``mjc:biasPrm``
     - ``biasPrm``
     - ``[0, …, 0]``
   * - ``mjc:biasType``
     - ``biasType``
     - ``"none"``
   * - ``mjc:dynPrm``
     - ``dynPrm``
     - ``[1, 0, …, 0]``
   * - ``mjc:dynType``
     - ``dynType``
     - ``"none"``
   * - ``mjc:gear``
     - ``gear``
     - ``[1, 0, 0, 0, 0, 0]``
