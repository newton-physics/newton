.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

MuJoCo Integration
==================

:class:`~newton.solvers.SolverMuJoCo` wraps `mujoco_warp
<https://github.com/google-deepmind/mujoco_warp>`_ behind Newton's standard
solver interface.

Because MuJoCo has its own modelling conventions, many Newton properties
are mapped differently or not at all. The sections below describe which
Newton concepts the solver supports, how each is mapped to MuJoCo, how
state is exchanged at every step, and where each piece of the conversion
lives in the source. MuJoCo-specific behaviour that has no Newton-core
equivalent is exposed through the :ref:`custom-attribute namespace
<mujoco-custom-attributes>`. A :ref:`code pointers <mujoco-code-pointers>`
section at the bottom collects the most useful anchor points.


Joint types
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Newton type
     - MuJoCo equivalent
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
     - The child body is nested directly under its parent. A fixed joint
       connecting to the world produces a **mocap** body, driven via
       ``mjData.mocap_pos`` / ``mjData.mocap_quat``.
   * - ``DISTANCE``, ``CABLE``
     - *unsupported*
     - Not forwarded to MuJoCo.


Geometry types
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Newton type
     - MuJoCo equivalent
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
     - Must be attached to the world body. Rendered size defaults to
       ``5 × 5 × 5``.
   * - ``HFIELD``
     - ``mjGEOM_HFIELD``
     - Heightfield data is normalized to ``[0, 1]``; the geom origin is
       shifted by ``min_z`` so the lowest point is at the correct world
       height.
   * - ``MESH`` / ``CONVEX_MESH``
     - ``mjGEOM_MESH``
     - MuJoCo only supports **convex** collision meshes. Non-convex meshes
       are convex-hulled automatically, which changes the collision
       boundary. ``maxhullvert`` is forwarded from the mesh source when set.
   * - ``CONE``, ``GAUSSIAN``
     - *unsupported*
     - Not present in the MuJoCo geom-type map.

**Sites** (shapes with the ``SITE`` flag) are converted to MuJoCo sites —
non-colliding reference frames used for sensor attachment and spatial
tendon wrap anchors.

SDF-based and hydroelastic collisions are not part of the MuJoCo geometry
model; they are only available through Newton's collision pipeline (see
`Collision pipeline`_ below).


Mass and inertia
----------------

Bodies with positive mass are exported with ``explicitinertial=True``,
forwarding mass, centre-of-mass offset (``ipos``), and inertia tensor.
When the inertia tensor is diagonal it is passed as a 3-element vector
(preserving MuJoCo's sameframe optimisation, ``body_simple=1``); otherwise
it is passed as the full 6-element symmetric form.

Zero-mass bodies (e.g. sensor frames, reference links) omit mass and
inertia entirely. The solver sets the compiler option
``inertiafromgeom = auto`` globally, so MuJoCo derives inertia from child
geoms for any body that was exported without an explicit inertia tensor.


Actuators
---------

Newton's per-DOF ``joint_target_mode`` creates MuJoCo general actuators:

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

``joint_effort_limit`` is forwarded as ``actfrcrange`` on the joint
(prismatic, revolute, and D6) or as ``forcerange`` on the actuator (ball).

The full MuJoCo general-actuator model (arbitrary gain/bias/dynamics types
and parameters, explicit transmission targets, ctrl/force/act ranges) is
only reachable through the :ref:`mujoco-custom-attributes` namespace.
Additional actuators declared this way are appended after the joint-target
actuators — see ``SolverMuJoCo._init_actuators``.


Equality constraints
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Newton type
     - MuJoCo equivalent
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
     - ``coef0`` / ``coef1`` mapped to polynomial coefficients. Only
       ``REVOLUTE`` and ``PRISMATIC`` joints are supported.


Tendons and contact pairs
-------------------------

Newton's core API does not currently expose tendons or explicit MuJoCo-style
``<pair>`` contact overrides as first-class concepts. Both are implemented
through the MuJoCo :ref:`custom-attribute namespace <mujoco-custom-attributes>`:

- **Tendons** (fixed and spatial) — populated on import from MJCF/USD and
  parsed into MuJoCo's tendon structures by ``SolverMuJoCo._init_tendons``.
  Unsupported wrap types and degenerate tendon definitions produce warnings
  rather than hard errors.
- **Contact pairs** — explicit geom-pair contact overrides are parsed by
  ``SolverMuJoCo._init_pairs``.


Collision filtering
-------------------

Newton uses integer ``shape_collision_group`` labels to control which shapes
can collide. MuJoCo uses ``contype``/``conaffinity`` bitmasks with different
semantics: two geoms collide when
``(contype_A & conaffinity_B) || (contype_B & conaffinity_A)`` is non-zero.

The solver bridges the two systems with **graph coloring**. Shapes that must
*not* collide are assigned the same color; each color maps to one bit in
``contype``. ``conaffinity`` is set to the complement so same-color geoms
never match. Up to 32 colors are supported (one per ``contype`` bit). If the
graph requires more than 32 colors, shapes with color index ≥ 32 fall back
to MuJoCo defaults (``contype=1``, ``conaffinity=1``) and will collide with
all other shapes, silently bypassing the intended filtering.

Non-colliding shapes (no ``COLLIDE_SHAPES`` flag, or
``collision_group == 0``) get ``contype = conaffinity = 0``. Body pairs for
which all shape-shape combinations are filtered are emitted as
``<exclude>`` elements.


Collision pipeline
------------------

By default :class:`~newton.solvers.SolverMuJoCo` uses MuJoCo's built-in
collision detection (``use_mujoco_contacts=True``). Alternatively, you can
set ``use_mujoco_contacts=False`` and pass contacts computed by Newton's
own collision pipeline into :meth:`~newton.solvers.SolverMuJoCo.step`.
Newton's pipeline supports non-convex meshes, SDF-based contacts, and
hydroelastic contacts, which are not available through MuJoCo's collision
detection.


Multi-world support
-------------------

When ``separate_worlds=True`` (the default for GPU mode with multiple
worlds), the solver builds a MuJoCo model from the **first world** only and
replicates it across all worlds via ``mujoco_warp``. This requires all
Newton worlds to be structurally identical (same bodies, joints, and
shapes). Global entities (those with a negative world index) may only
include static shapes — they are shared across all worlds without
replication.


Runtime state synchronisation
-----------------------------

Each call to :meth:`~newton.solvers.SolverMuJoCo.step` goes through the
same three-phase cycle:

1. **Push Newton → MuJoCo.** Joint positions and velocities
   (``state_in.joint_q`` / ``joint_qd``) are copied into MuJoCo's
   ``qpos`` / ``qvel`` with quaternion reordering. Control is applied via
   ``control.joint_target_pos`` / ``joint_target_vel`` (→ MuJoCo
   ``ctrl``), ``control.joint_f`` (→ ``qfrc_applied``), and
   ``state_in.body_f`` (→ ``xfrc_applied``). Direct MuJoCo actuator
   control is exposed through ``control.mujoco.ctrl``. When
   ``use_mujoco_contacts=False``, Newton-side contacts are converted into
   ``mjData.contact`` before the integrator runs.
2. **Integrate.** ``mujoco_warp`` (or the CPU MuJoCo backend when
   ``use_mujoco_cpu=True``) steps the MuJoCo model forward by ``dt``.
3. **Pull MuJoCo → Newton.** Integrated ``qpos`` / ``qvel`` are copied
   back into ``state_out.joint_q`` / ``joint_qd``, with quaternion
   reordering; ``body_q`` / ``body_qd`` are recomputed from the new joint
   state. Kinematic roots pass through unchanged (see
   `Kinematic links and fixed roots`_).

Contacts are **not** pulled back into a Newton ``Contacts`` object
automatically. Call :meth:`~newton.solvers.SolverMuJoCo.update_contacts`
when you need contact points, forces, or material indices in Newton form.

Push, pull, and contact-conversion are implemented by
``SolverMuJoCo._apply_mjc_control``, ``SolverMuJoCo._update_newton_state``,
and :meth:`~newton.solvers.SolverMuJoCo.update_contacts`, using kernels
from ``newton/_src/solvers/mujoco/kernels.py`` — see
`Where the conversion lives`_ for the full anchor list.


Solver options
--------------

MuJoCo solver parameters (``iterations``, ``ls_iterations``, ``solver``,
``integrator``, ``cone``, ``jacobian``, ``tolerance``, ``ls_tolerance``,
``impratio``, ``wind``, ``viscosity``, ``density``, ``magnetic``, and the
CCD / SDF iteration counts) follow a three-level resolution priority:

1. **Constructor argument** passed to :class:`~newton.solvers.SolverMuJoCo`
   — one value, applied to all worlds. The full list of kwargs, their
   types, and their defaults is documented on the class itself.
2. **Custom attribute** (``model.mujoco.<option>``) — supports per-world
   values. Typically populated automatically by USD or MJCF import.
3. **Default** — if neither of the above is set, the MuJoCo default is
   used, with one Newton-opinionated exception: ``integrator`` defaults
   to ``implicitfast`` (MuJoCo's default is ``euler``) for better
   stability on stiff systems.


.. _mujoco-custom-attributes:

Custom attributes
-----------------

Many MuJoCo-specific parameters have no counterpart in Newton's core API.
The solver exposes them through a dedicated ``mujoco`` custom-attribute
namespace (``model.mujoco.<name>``), populated from MJCF elements and from
attributes in the OpenUSD MuJoCo schema (``mjc:*``). To enable the
namespace, call
:meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` on the
:class:`~newton.ModelBuilder` **before** loading any asset or adding
joints/shapes manually::

    import newton
    from newton.solvers import SolverMuJoCo

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    # ...then import MJCF / USD or add bodies/joints/shapes manually...
    model = builder.finalize()

The authoritative list of registered attributes — names, defaults, dtypes,
MJCF / USD source names, and the category each belongs to — is the body
of :meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` itself.
See :doc:`/concepts/custom_attributes` for how Newton's custom-attribute
system works in general.

A subset of ``mjc:*`` USD attributes is additionally mapped onto Newton's
built-in properties during USD import by
:class:`~newton.usd.SchemaResolverMjc` (``newton/_src/usd/schemas.py``) —
for example, joint-limit stiffness/damping from ``mjc:solreflimit``, and
torsional/rolling friction from ``mjc:torsionalfriction`` /
``mjc:rollingfriction``. Attributes not handled by the schema resolver
land in the ``mujoco`` namespace unchanged.


Unsupported MuJoCo features
---------------------------

The sections above describe what Newton forwards *into* MuJoCo. In the
other direction, MuJoCo has several modelling concepts that are not
imported when loading an MJCF or USD asset into Newton, and that
:class:`~newton.solvers.SolverMuJoCo` does not reconstruct at export time:

- **Sensors** (``<sensor>`` — force/torque, IMU, gyro, accelerometer,
  rangefinder, touch, camera-based, …). Newton has its own sensor
  pipeline (:doc:`/concepts/sensors`) that is independent of the MuJoCo
  solver.
- **Cameras and lights** declared in MJCF/USD. Newton uses its own viewer
  and lighting pipeline; camera/light primitives in the source asset are
  ignored.
- **Keyframes** (``<keyframe>``) — MuJoCo's saved-state / reset mechanism
  is not imported.
- **Composite and flex** (``<composite>``, ``<flex>``) — MuJoCo's built-in
  deformables and soft bodies. Newton has dedicated solvers for cloth,
  MPM, and FEM; they are not part of the MuJoCo integration.
- **Skinned meshes** (``<skin>``) — visualisation-only, not imported.
- **User plugins** (``<plugin>``) — MuJoCo's plugin mechanism for custom
  passive forces or dynamics is not supported.
- **User data and arbitrary custom elements** (``<custom>``, ``<numeric>``,
  ``<text>``) — not imported. Newton-specific user data should use the
  Newton custom-attribute system instead.
- **Actuator transmissions** — not all MuJoCo transmission types are
  supported. See :class:`~newton.solvers.SolverMuJoCo.TrnType` for the
  full list.

Smaller limitations are documented inline where they are most relevant —
see `Caveats`_ below for ``gap``, collision-radius, convex-hull fallback,
and velocity limits; and the unsupported rows in `Joint types`_ and
`Geometry types`_.


Caveats
-------

**geom_gap is always zero.**
  MuJoCo's ``gap`` parameter controls *inactive* contact generation —
  contacts that are detected but do not produce constraint forces until
  penetration exceeds the gap threshold. Newton does not use this concept:
  when the MuJoCo collision pipeline is active it runs every step, so
  there is no benefit to keeping inactive contacts around. Setting
  ``geom_gap > 0`` would inflate ``geom_margin``, which disables MuJoCo's
  multicontact and degrades contact quality. Therefore
  :class:`~newton.solvers.SolverMuJoCo` always sets ``geom_gap = 0``
  regardless of the Newton
  :attr:`~newton.ModelBuilder.ShapeConfig.gap` value. MJCF/USD ``gap``
  values are still imported into
  :attr:`~newton.ModelBuilder.ShapeConfig.gap` on the Newton model, but
  they are not forwarded to the MuJoCo solver.

**shape_collision_radius is ignored.**
  MuJoCo computes bounding-sphere radii (``geom_rbound``) internally from
  the geometry definition. Newton's ``shape_collision_radius`` is not
  forwarded.

**Non-convex meshes are convex-hulled.**
  MuJoCo only supports convex collision geometry. Non-convex ``MESH``
  shapes are automatically convex-hulled at conversion time, changing the
  effective collision boundary.

**Velocity limits are not forwarded.**
  Newton's ``joint_velocity_limit`` has no MuJoCo equivalent and is
  ignored.


.. _mujoco-kinematic-links-and-fixed-roots:

Kinematic links and fixed roots
-------------------------------

Newton only allows ``is_kinematic=True`` on articulation roots, so in the
MuJoCo exporter a "kinematic link" always means a kinematic root body. Any
descendants of that root can still be dynamic and are exported normally.

At runtime, :class:`~newton.solvers.SolverMuJoCo` keeps kinematic roots
user-prescribed rather than dynamically integrated:

- When converting MuJoCo state back to Newton, the previous Newton
  :attr:`newton.State.joint_q` and :attr:`newton.State.joint_qd` values are
  passed through for kinematic roots instead of being overwritten from
  MuJoCo's integrated ``qpos`` and ``qvel``.
- Applied body wrenches and joint forces targeting kinematic bodies are
  ignored on the MuJoCo side.
- Kinematic bodies still participate in contacts, so they can act as
  moving or fixed obstacles for dynamic bodies.

During export, :class:`~newton.solvers.SolverMuJoCo` maps roots according
to their joint type:

- **Kinematic roots with non-fixed joints** are exported as ordinary MuJoCo
  joints with the same Newton joint type and DOFs. The solver assigns a
  very large internal armature to those DOFs so MuJoCo treats them like
  prescribed, effectively infinite-mass coordinates.
- **Roots attached to world with a fixed joint** are exported as MuJoCo
  mocap bodies. This applies to both kinematic and non-kinematic Newton
  roots attached to world by :class:`~newton.JointType.FIXED`. MuJoCo has
  no joint coordinates for a fixed root, so Newton drives the pose through
  ``mjData.mocap_pos`` and ``mjData.mocap_quat`` instead.
- **World-attached shapes that are not part of an articulation** remain
  ordinary static MuJoCo geometry rather than mocap bodies.

If you edit :attr:`newton.Model.joint_X_p` or :attr:`newton.Model.joint_X_c`
for a fixed-root articulation after constructing the solver, call
``solver.notify_model_changed(newton.solvers.SolverNotifyFlags.JOINT_PROPERTIES)``
to synchronise the updated fixed-root poses into MuJoCo.


.. _mujoco-code-pointers:

Where the conversion lives
--------------------------

For readers navigating the source, the following symbols are the most
useful entry points. Symbols with a leading underscore are **internal
entry points** — stable enough to navigate to, but not part of the public
API and subject to change.

- :meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` —
  authoritative registry of every MuJoCo-specific custom attribute and
  frequency.
- :meth:`~newton.solvers.SolverMuJoCo.step` — per-step integration entry
  point.
- ``SolverMuJoCo._convert_to_mjc`` — Newton ``Model`` (and optional
  ``State``) → MuJoCo ``mjModel`` / ``mjData`` (orchestrator).
- ``SolverMuJoCo._init_pairs`` / ``_init_actuators`` / ``_init_tendons`` —
  category-specific parsers that consume the MuJoCo custom attributes.
- ``SolverMuJoCo._apply_mjc_control`` and
  ``SolverMuJoCo._update_newton_state`` — per-step control and state sync
  between Newton and MuJoCo.
- ``newton/_src/solvers/mujoco/kernels.py`` — Warp kernels for coordinate,
  contact, and state conversion (``quat_wxyz_to_xyzw``,
  ``convert_mj_coords_to_warp_kernel``,
  ``convert_newton_contacts_to_mjwarp_kernel``, ``convert_solref``, …).
- :class:`~newton.usd.SchemaResolverMjc`
  (``newton/_src/usd/schemas.py``) — USD ``mjc:*`` attribute → Newton
  built-in property mapping.
