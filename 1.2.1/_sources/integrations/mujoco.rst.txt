.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

MuJoCo Integration
==================

:class:`~newton.solvers.SolverMuJoCo` wraps `mujoco_warp
<https://github.com/google-deepmind/mujoco_warp>`_ behind Newton's standard
solver interface. Newton uses compatible-release pins (``~=``) on both ``mujoco``
and ``mujoco-warp`` to keep the two version-aligned; see
:github:`pyproject.toml` for the current pins.

Because MuJoCo has its own modeling conventions, many Newton properties
are mapped differently or not at all. The sections below describe which
Newton concepts the solver supports, how each is mapped to MuJoCo, how
state is exchanged at every step, and where each piece of the conversion
lives in the source. MuJoCo-specific behavior that has no Newton-core
equivalent is exposed through the :ref:`custom-attribute namespace
<mujoco-custom-attributes>`. A :ref:`code pointers <mujoco-code-pointers>`
section at the bottom collects the most useful anchor points.

.. note::

   References to ``mjModel`` / ``mjData`` fields below (e.g.
   ``mjData.contact``, ``mjData.mocap_pos``) use the canonical names
   from MuJoCo's `mjModel
   <https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel>`_
   and `mjData
   <https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata>`_
   reference. ``mujoco_warp`` exposes the same fields on its
   GPU-resident analogues.


Joint types
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Newton type
     - MuJoCo equivalent
     - Notes
   * - :attr:`~newton.JointType.FREE`
     - ``mjJNT_FREE``
     - Initial pose taken from ``body_q``.
   * - :attr:`~newton.JointType.BALL`
     - ``mjJNT_BALL``
     - Per-axis actuators mapped via ``gear``.
   * - :attr:`~newton.JointType.REVOLUTE`
     - ``mjJNT_HINGE``
     -
   * - :attr:`~newton.JointType.PRISMATIC`
     - ``mjJNT_SLIDE``
     -
   * - :attr:`~newton.JointType.D6`
     - Up to 3 × ``mjJNT_SLIDE`` + 3 × ``mjJNT_HINGE``
     - Each active linear/angular DOF becomes a separate MuJoCo joint with
       a ``_lin`` or ``_ang`` suffix; a numeric index is appended when more
       than one axis is active in the same group (e.g. ``_lin0``,
       ``_lin1``).
   * - :attr:`~newton.JointType.FIXED`
     - *(no joint)*
     - The child body is nested directly under its parent. A fixed joint
       connecting to the world produces a **mocap** body, driven via
       ``mjData.mocap_pos`` / ``mjData.mocap_quat``.
   * - :attr:`~newton.JointType.DISTANCE`
     - *(no joint)*
     - The distance constraint is dropped, but the body bookkeeping is
       handled like a free body (counted in MuJoCo's free-body slots).
   * - :attr:`~newton.JointType.CABLE`
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
   * - :attr:`~newton.GeoType.SPHERE`
     - ``mjGEOM_SPHERE``
     -
   * - :attr:`~newton.GeoType.CAPSULE`
     - ``mjGEOM_CAPSULE``
     -
   * - :attr:`~newton.GeoType.CYLINDER`
     - ``mjGEOM_CYLINDER``
     -
   * - :attr:`~newton.GeoType.BOX`
     - ``mjGEOM_BOX``
     -
   * - :attr:`~newton.GeoType.ELLIPSOID`
     - ``mjGEOM_ELLIPSOID``
     -
   * - :attr:`~newton.GeoType.PLANE`
     - ``mjGEOM_PLANE``
     - Must be attached to a static body (``body=-1``); attaching to a
       non-static body raises ``ValueError`` at conversion time. Planes
       are infinite for collision in MuJoCo regardless of size; the
       configured :attr:`~newton.Model.shape_scale` only affects rendering, defaulting to
       ``5 × 5 × 5`` when unset.
   * - :attr:`~newton.GeoType.HFIELD`
     - ``mjGEOM_HFIELD``
     - Heightfield data is stored normalized to ``[0, 1]`` on the Newton
       :class:`~newton.Heightfield` source and forwarded as-is. The geom
       origin is shifted by ``min_z`` so the lowest point is at the
       correct world height.
   * - :attr:`~newton.GeoType.MESH` / :attr:`~newton.GeoType.CONVEX_MESH`
     - ``mjGEOM_MESH``
     - MuJoCo only supports **convex** collision meshes. Non-convex
       meshes are convex-hulled by MuJoCo's compiler (not by Newton),
       which changes the collision boundary. The mesh source's
       ``maxhullvert`` is forwarded.
   * - :attr:`~newton.GeoType.CONE`, :attr:`~newton.GeoType.GAUSSIAN`
     - *unsupported*
     - Not present in the MuJoCo geom-type map.

**Sites** (shapes with the ``SITE`` flag) are converted to MuJoCo sites —
non-colliding reference frames used for sensor attachment and spatial
tendon wrap anchors. Only ``SPHERE``, ``CAPSULE``, ``CYLINDER``, and
``BOX`` are MuJoCo-native site geom types; other types silently fall
back to ``SPHERE``.

Several Newton collision features — for example non-convex trimesh,
SDF-based contacts, and hydroelastic contacts — are not part of the
MuJoCo geometry model. They are only available through Newton's
collision pipeline (see `Collision pipeline`_ below).


Actuators
---------

Newton's per-DOF :attr:`~newton.Model.joint_target_mode` creates MuJoCo general actuators:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - MuJoCo actuator(s)
   * - :attr:`~newton.JointTargetMode.POSITION`
     - One actuator: ``gainprm = [kp]``, ``biasprm = [0, -kp, -kd]``.
   * - :attr:`~newton.JointTargetMode.VELOCITY`
     - One actuator: ``gainprm = [kd]``, ``biasprm = [0, 0, -kd]``.
   * - :attr:`~newton.JointTargetMode.POSITION_VELOCITY`
     - Two actuators — a position actuator (``gainprm = [kp]``,
       ``biasprm = [0, -kp, 0]``) and a velocity actuator
       (``gainprm = [kd]``, ``biasprm = [0, 0, -kd]``).
   * - :attr:`~newton.JointTargetMode.NONE`,
       :attr:`~newton.JointTargetMode.EFFORT`
     - No MuJoCo actuator created.

:attr:`~newton.Model.joint_effort_limit` is forwarded as ``actfrcrange`` on the joint
(prismatic, revolute, and D6) or as ``forcerange`` on the actuator (ball).

The full MuJoCo general-actuator model (arbitrary gain/bias/dynamics types
and parameters, explicit transmission targets, ctrl/force/act ranges) is
only reachable through the ``mujoco`` :ref:`custom-attribute namespace <mujoco-custom-attributes>`.
Additional actuators declared this way are appended after the joint-target
actuators — see ``SolverMuJoCo._init_actuators``.


Equality constraints
--------------------

Each row's ``data[...]`` reference below points into MuJoCo's
`equality.data <https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality>`_
array; slot layout depends on the constraint type.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Newton type
     - MuJoCo equivalent
     - Notes
   * - :attr:`~newton.EqType.CONNECT`
     - ``mjEQ_CONNECT``
     - Anchor forwarded in ``data[0:3]``.
   * - :attr:`~newton.EqType.WELD`
     - ``mjEQ_WELD``
     - Anchor forwarded in ``data[0:3]``, relative pose in ``data[3:10]``,
       torque scale in ``data[10]``.
   * - :attr:`~newton.EqType.JOINT`
     - ``mjEQ_JOINT``
     - Polynomial coefficients forwarded in ``data[0:5]``.
   * - Mimic
     - ``mjEQ_JOINT``
     - Added via :meth:`~newton.ModelBuilder.add_constraint_mimic`. Maps
       ``coef0`` / ``coef1`` to polynomial coefficients. Only
       :attr:`~newton.JointType.REVOLUTE` and
       :attr:`~newton.JointType.PRISMATIC` joints are supported.

**Loop closures.** Newton joints with no associated articulation
(``joint_articulation == -1``) are treated as loop closures rather than
tree joints. They are not emitted as MuJoCo joints; instead, the solver
synthesises equality constraints:

- :attr:`~newton.JointType.FIXED` → ``mjEQ_WELD`` (constrains all 6 DOFs).
- :attr:`~newton.JointType.REVOLUTE` → two ``mjEQ_CONNECT`` constraints,
  the second offset by 0.1 m along the hinge axis, so 5 DOFs are
  constrained and one rotational DOF remains free.
- :attr:`~newton.JointType.BALL` → one ``mjEQ_CONNECT`` (3 translational
  DOFs constrained, all 3 rotational DOFs free).

Other joint types in this configuration are not supported and produce a
warning. Loop-joint DOFs and coordinates are excluded from MuJoCo's
``nq`` / ``nv``.


Tendons
-------

Newton's core API does not currently expose tendons (fixed or spatial)
as first-class concepts. They are implemented through the MuJoCo
:ref:`custom-attribute namespace <mujoco-custom-attributes>`: populated
on import from MJCF/USD and parsed into MuJoCo's tendon structures by
``SolverMuJoCo._init_tendons``. Spatial tendons support ``site``,
``geom``, and ``pulley`` wrap elements; any other wrap type and any
degenerate tendon definition produces a warning and is skipped rather
than raising.


Collision pipeline
------------------

:class:`~newton.solvers.SolverMuJoCo` uses MuJoCo's built-in collision
detection by default. Construct it with ``use_mujoco_contacts=False``
to feed contacts computed by Newton's own collision pipeline into
:meth:`~newton.solvers.SolverMuJoCo.step` instead.
Newton's pipeline supports non-convex meshes, SDF-based contacts, and
hydroelastic contacts, which are not available through MuJoCo's collision
detection.

**Multi-contact CCD.** Constructing
:class:`~newton.solvers.SolverMuJoCo` with ``enable_multiccd=True``
allows up to four contact points per geom pair instead of one. Pairs
where either geom has non-zero MuJoCo ``geom_margin`` still fall back
to a single contact regardless of the flag (see *Margin zeroing*
below for how Newton's :attr:`~newton.Model.shape_margin` is forwarded
to it).

**Margin zeroing.** ``mujoco_warp`` rejects non-zero geom margins on
box-box pairs (its default NATIVECCD path) and on any box/mesh pair
when ``enable_multiccd=True``. To stay compatible :class:`~newton.solvers.SolverMuJoCo` zeroes
``geom_margin`` model-wide at compile time whenever a box geom exists,
or whenever ``enable_multiccd=True`` is combined with mesh geoms; geoms
with non-zero authored margins emit a warning when
``use_mujoco_contacts=True``. The Newton model's :attr:`~newton.Model.shape_margin` array
is left untouched, and when ``use_mujoco_contacts=False`` the authored
margins are restored at runtime through ``update_geom_properties_kernel``.


Contact pairs
-------------

Newton's core API does not expose explicit MuJoCo-style ``<pair>``
contact overrides. They are implemented through the MuJoCo
:ref:`custom-attribute namespace <mujoco-custom-attributes>` and
parsed into MuJoCo's geom-pair contact structures by
``SolverMuJoCo._init_pairs``.


Multi-world support
-------------------

Constructing :class:`~newton.solvers.SolverMuJoCo` with
``separate_worlds=True`` (the default for GPU mode with multiple
worlds) builds a MuJoCo model from the **first world** only and
replicates it across all worlds via ``mujoco_warp``. This requires
all Newton worlds to be structurally identical (same bodies, joints,
and shapes); :class:`~newton.solvers.SolverMuJoCo` validates this at
construction and raises ``ValueError`` on a mismatch.

Bodies, joints, equality constraints, and mimic constraints cannot have
a negative world index — assigning any of them to the global world
raises ``ValueError``. Only shapes may live in the global world (-1);
they are shared across all worlds without replication.


Runtime state synchronization
-----------------------------

Each call to :meth:`~newton.solvers.SolverMuJoCo.step` goes through the
same three-phase cycle:

1. **Push Newton → MuJoCo.** ``SolverMuJoCo._apply_mjc_control`` and
   ``SolverMuJoCo._update_mjc_data`` transfer the Newton ``State``
   and ``Control`` inputs to MuJoCo's working data. When
   ``use_mujoco_contacts=False``, Newton-side contacts are also
   converted before the integrator runs. The joint-state re-sync
   frequency can be controlled via the ``update_data_interval``
   kwarg for substepping schemes.
2. **Integrate.** ``mujoco_warp`` steps the MuJoCo model forward by ``dt``.
3. **Pull MuJoCo → Newton.** ``SolverMuJoCo._update_newton_state``
   populates the output ``State`` from the integrated MuJoCo data.
   Kinematic roots pass through unchanged from ``state_in`` (see
   `Kinematic links and fixed roots`_).

Contacts are **not** pulled back into a Newton ``Contacts`` object
automatically. Call :meth:`~newton.solvers.SolverMuJoCo.update_contacts`
when you need contact points, forces, or material indices in Newton form.

Push, pull, and contact-conversion are implemented by
``SolverMuJoCo._apply_mjc_control``, ``SolverMuJoCo._update_newton_state``,
and :meth:`~newton.solvers.SolverMuJoCo.update_contacts`, using kernels
from :github:`newton/_src/solvers/mujoco/kernels.py` — see
`Code pointers`_ for the full anchor list.


Solver options
--------------

MuJoCo solver parameters follow a three-level resolution priority:

1. **Constructor argument** passed to :class:`~newton.solvers.SolverMuJoCo`
   — one value, applied to all worlds. The full list of kwargs, their
   types, and their defaults is documented on the class itself.
2. **Custom attribute** (``model.mujoco.<option>``) — supports per-world
   values. Typically populated automatically by USD or MJCF import.
3. **Default** — if neither of the above is set, the MuJoCo default is
   used, with one Newton-opinionated exception: ``integrator`` defaults
   to ``implicitfast`` (MuJoCo's default is ``euler``) for better
   stability on stiff systems.

These values are read once during :class:`~newton.solvers.SolverMuJoCo`
construction. Editing ``model.mujoco.<option>`` afterwards has no
effect — the resolved value is already baked into the underlying
MuJoCo model.

See MuJoCo's `solver documentation
<https://mujoco.readthedocs.io/en/stable/computation/index.html>`_ and
`\<option\> XML reference
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#option>`_ for
what each parameter does and when to tune it.


.. _mujoco-custom-attributes:

MuJoCo-specific parameters in USD and MJCF
------------------------------------------

MuJoCo has parameters with no counterpart in Newton's core API.
:class:`~newton.ModelBuilder` handles them during MJCF / USD import
via two mechanisms.

**Custom-attribute namespace.** A dedicated ``mujoco`` custom-attribute
namespace (``model.mujoco.<name>``) is populated from MJCF elements
and from attributes in the OpenUSD MuJoCo schema (``mjc:*``). To
enable the namespace, call
:meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` on the
:class:`~newton.ModelBuilder` **before** adding anything to it::

    import newton
    from newton.solvers import SolverMuJoCo

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    # ...then add anything (e.g. import MJCF / USD, add joints, ...)
    model = builder.finalize()

The authoritative list of registered attributes — names, defaults,
dtypes, MJCF / USD source names, and the category each belongs to —
is the body of
:meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes`
itself. See :doc:`/concepts/custom_attributes` for how Newton's
custom-attribute system works in general.

**Direct mapping to Newton built-ins.** Some MuJoCo-specific
attributes are mapped onto Newton's built-in properties during import
(rather than the ``mujoco`` namespace) — for example, joint-limit
stiffness and damping derived from ``solreflimit``. The MJCF parser
handles this inline (:github:`newton/_src/utils/import_mjcf.py`); USD
goes through :class:`~newton.usd.SchemaResolverMjc`
(:github:`newton/_src/usd/schemas.py`).


Unsupported MuJoCo features
---------------------------

The sections above describe what Newton forwards *into* MuJoCo. In the
other direction, MuJoCo has several modeling concepts that are not
imported when loading an MJCF or USD asset into Newton, and that
:class:`~newton.solvers.SolverMuJoCo` does not reconstruct during conversion:

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
- **Skinned meshes** (``<skin>``) — visualization-only, not imported.
- **User plugins** (``<plugin>``) — MuJoCo's plugin mechanism for custom
  passive forces or dynamics is not supported.
- **User data and arbitrary custom elements** (``<custom>``, ``<numeric>``,
  ``<text>``) — not imported. Newton-specific user data should use the
  Newton custom-attribute system instead.
- **Actuator transmissions** — only ``joint``, ``tendon``, ``site``, and
  ``body`` transmissions are supported (see
  :class:`~newton.solvers.SolverMuJoCo.TrnType` for the enum). MuJoCo's
  ``jointinparent`` and ``slidercrank`` transmissions are not converted;
  actuators using them are skipped at construction with a warning.

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
  the geometry definition. Newton's :attr:`~newton.Model.shape_collision_radius` is not
  forwarded.

**Non-convex meshes are convex-hulled.**
  MuJoCo only supports convex collision geometry. Non-convex ``MESH``
  shapes are automatically convex-hulled at conversion time, changing the
  effective collision boundary.

**Velocity limits are not forwarded.**
  Newton's :attr:`~newton.Model.joint_velocity_limit` has no MuJoCo equivalent and is
  ignored.

**Kinematic-root armature override.**
  DOFs of kinematic articulation roots have their
  :attr:`~newton.Model.joint_armature` replaced with a very large
  internal value (``1e10``) so MuJoCo treats them as effectively
  prescribed. The user-supplied armature on those DOFs is silently
  discarded. See `Kinematic links and fixed roots`_.

**Collision filtering bitmask fallback.**
  Newton's :attr:`~newton.Model.shape_collision_group` (see
  :ref:`Collision Groups`) is translated to MuJoCo's ``contype`` /
  ``conaffinity`` via graph coloring
  (:github:`newton/_src/sim/graph_coloring.py`). Up to 32 colors are
  supported (one per ``contype`` bit). If the filtering graph requires
  more, shapes with color index ≥ 32 fall back to ``contype=1`` /
  ``conaffinity=1`` and silently collide with every other shape,
  bypassing the intended filtering and adding extra contact pairs to
  the broadphase.


.. _mujoco-kinematic-links-and-fixed-roots:

Kinematic links and fixed roots
-------------------------------

Newton only allows ``is_kinematic=True`` on articulation roots, so a
"kinematic link" in this section always means a kinematic root body.
Any descendants of that root can still be dynamic and are converted
normally.

At runtime, :class:`~newton.solvers.SolverMuJoCo` keeps kinematic roots
user-prescribed rather than dynamically integrated:

- When converting MuJoCo state back to Newton, the previous Newton
  :attr:`~newton.State.joint_q` and :attr:`~newton.State.joint_qd` values
  are passed through for kinematic roots instead of being overwritten
  from MuJoCo's integrated ``qpos`` and ``qvel``.
- Applied body wrenches and joint forces targeting kinematic bodies
  are ignored on the MuJoCo side.
- Kinematic bodies still participate in contacts, so they can act as
  moving or fixed obstacles for dynamic bodies.

During Newton-to-MuJoCo conversion (at
:class:`~newton.solvers.SolverMuJoCo` construction), roots are mapped
by joint type:

- **Kinematic roots with non-fixed joints** become ordinary MuJoCo
  joints with the same Newton joint type and DOFs. A very large
  internal armature is assigned to those DOFs so MuJoCo treats them
  as prescribed, effectively infinite-mass coordinates.
- **Roots attached to world with a fixed joint** become MuJoCo mocap
  bodies (whether kinematic or not). MuJoCo has no joint coordinates
  for a fixed root, so Newton drives the pose through
  ``mjData.mocap_pos`` and ``mjData.mocap_quat`` instead.
- **World-attached shapes that are not part of an articulation**
  remain ordinary static MuJoCo geometry rather than mocap bodies.

If you edit :attr:`~newton.Model.joint_X_p` or :attr:`~newton.Model.joint_X_c`
for a fixed-root articulation after constructing the solver, call
:meth:`~newton.solvers.SolverBase.notify_model_changed` with the
:attr:`~newton.solvers.SolverNotifyFlags.JOINT_PROPERTIES` flag to
synchronize the updated fixed-root poses into MuJoCo.


.. _mujoco-code-pointers:

Code pointers
-------------

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
- ``SolverMuJoCo._apply_mjc_control``,
  ``SolverMuJoCo._update_mjc_data``, and
  ``SolverMuJoCo._update_newton_state`` — per-step control, data, and
  state sync between Newton and MuJoCo.
- :meth:`~newton.solvers.SolverMuJoCo.update_contacts` — explicit pull
  of MuJoCo's resolved contacts into a Newton ``Contacts`` object
  (default per-step path does not pull contacts back).
- :meth:`~newton.solvers.SolverBase.notify_model_changed` —
  re-synchronize MuJoCo state after editing the Newton ``Model`` (e.g.
  fixed-root pose changes via :attr:`~newton.Model.joint_X_p` / :attr:`~newton.Model.joint_X_c`).
- :github:`newton/_src/solvers/mujoco/kernels.py` — Warp kernels for
  coordinate, contact, and state conversion (``quat_wxyz_to_xyzw``,
  ``convert_mj_coords_to_warp_kernel``,
  ``convert_newton_contacts_to_mjwarp_kernel``, ``convert_solref``, …).
- :class:`~newton.usd.SchemaResolverMjc`
  (:github:`newton/_src/usd/schemas.py`) — USD ``mjc:*`` attribute →
  Newton built-in property mapping.
