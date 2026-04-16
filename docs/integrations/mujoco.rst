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


.. _mujoco-parameters:

MuJoCo model attributes
-----------------------

The tables below list every Newton property and custom attribute that
:class:`~newton.solvers.SolverMuJoCo` reads, organized by entity type.  Each
table shows:

- The Newton property or ``model.mujoco.<name>`` custom attribute.
- The USD attribute that populates it (standard ``physics:`` schemas,
  ``mjc:``-prefixed attributes from the
  `mjcPhysics USD schema <https://github.com/google-deepmind/mujoco/blob/main/src/experimental/usd/mjcPhysics/generatedSchema.usda>`_,
  or ``newton:mujoco:`` attributes for Newton's own round-trip namespace).
- The MJCF XML attribute that populates it during
  :meth:`~newton.ModelBuilder.add_mjcf`, where applicable.
- The MuJoCo model field it maps to.

Standard Newton properties (``joint_limit_lower``, ``shape_material_mu``, etc.)
are populated from standard USD physics schemas, MJCF XML elements, or set
programmatically.  MuJoCo-specific parameters use the ``mujoco``
custom-attribute namespace.  Attributes marked :sup:`ext` are Newton extensions
— ``mjc:``-prefixed USD attributes without a schema counterpart yet.

**Setup.**
  Call :meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` on a
  :class:`~newton.ModelBuilder` **before** loading assets so that USD and MJCF
  importers can populate the ``mujoco.*`` attributes.  After
  :meth:`~newton.ModelBuilder.finalize`, they are accessible as
  ``model.mujoco.<name>``.  See :doc:`/concepts/custom_attributes` for
  background on Newton's custom-attribute system.


.. _mujoco-solver-options:

Solver options
^^^^^^^^^^^^^^

Solver parameters follow a three-level resolution priority:

1. **Constructor argument** — value passed to :class:`~newton.solvers.SolverMuJoCo`.
2. **Custom attribute** (``model.mujoco.<option>``) — supports per-world values.
   These attributes are typically populated automatically when importing USD or
   MJCF assets.
3. **Default** — the tables below list Newton defaults alongside MuJoCo
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

**Per-world options** (one value per Newton world, read from ``PhysicsScene``):

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 20

   * - Custom attribute
     - USD attribute
     - Default
     - Notes
   * - ``mujoco.impratio``
     - ``mjc:option:impratio``
     - 1.0
     -
   * - ``mujoco.tolerance``
     - ``mjc:option:tolerance``
     - 1e-8
     -
   * - ``mujoco.ls_tolerance``
     - ``mjc:option:ls_tolerance``
     - 0.01
     -
   * - ``mujoco.ccd_tolerance``
     - ``mjc:option:ccd_tolerance``
     - 1e-6
     -
   * - ``mujoco.density``
     - ``mjc:option:density``
     - 0.0
     - Medium density [kg/m\ :sup:`3`]
   * - ``mujoco.viscosity``
     - ``mjc:option:viscosity``
     - 0.0
     - Medium viscosity [Pa·s]
   * - ``mujoco.wind``
     - ``mjc:option:wind``
     - ``(0, 0, 0)``
     - Wind velocity [m/s]
   * - ``mujoco.magnetic``
     - ``mjc:option:magnetic``
     - ``(0, -0.5, 0)``
     - Magnetic flux [T]

**Per-model options** (single value, read from ``PhysicsScene``):

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 20

   * - Custom attribute
     - USD attribute
     - Default
     - Notes
   * - ``mujoco.iterations``
     - ``mjc:option:iterations``
     - 100
     - Also sets built-in ``max_solver_iterations``
   * - ``mujoco.ls_iterations``
     - ``mjc:option:ls_iterations``
     - 50
     -
   * - ``mujoco.ccd_iterations``
     - ``mjc:option:ccd_iterations``
     - 35
     -
   * - ``mujoco.sdf_iterations``
     - ``mjc:option:sdf_iterations``
     - 10
     -
   * - ``mujoco.sdf_initpoints``
     - ``mjc:option:sdf_initpoints``
     - 40
     -
   * - ``mujoco.integrator``
     - ``mjc:option:integrator``
     - 3 (``implicitfast``)
     -
   * - ``mujoco.solver``
     - ``mjc:option:solver``
     - 2 (``newton``)
     -
   * - ``mujoco.cone``
     - ``mjc:option:cone``
     - 0 (``pyramidal``)
     -
   * - ``mujoco.jacobian``
     - ``mjc:option:jacobian``
     - 2 (``auto``)
     -

**Scene — built-in properties** (read from ``PhysicsScene`` via ``SchemaResolverMjc``):

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 20

   * - Newton property
     - USD attribute
     - Default
     - Notes
   * - ``max_solver_iterations``
     - ``mjc:option:iterations``
     - 100
     -
   * - ``time_steps_per_second``
     - ``mjc:option:timestep``
     - 0.002 (→ 500 Hz)
     - USD stores the step *duration*; Newton inverts it
   * - ``gravity_enabled``
     - ``mjc:flag:gravity``
     - ``True``
     -
   * - ``gravity``
     -
     - ``(0, 0, -9.81)``
     - 3D gravity vector → MuJoCo ``opt.gravity``


.. _mujoco-joint-attributes:

Joint attributes
^^^^^^^^^^^^^^^^

**Standard Newton properties:**

.. list-table::
   :header-rows: 1
   :widths: 20 18 22 18 22

   * - Newton property
     - MuJoCo field
     - USD attribute
     - MJCF attribute
     - Notes
   * - ``joint_limit_lower`` / ``upper``
     - ``jnt_range``
     - ``physics:lowerLimit`` / ``upperLimit``
     - ``range``
     - Angular limits converted to degrees for MuJoCo
   * - ``joint_limit_ke`` / ``kd``
     - ``jnt_solref``
     - ``newton:*:limitStiffness`` / ``Damping``
     - ``solreflimit``
     - Forwarded as negative solref ``(-ke, -kd)``
   * - ``joint_armature``
     - ``dof_armature``
     - ``newton:armature``
     - ``armature``
     -
   * - ``joint_friction``
     - ``dof_frictionloss``
     - ``newton:friction``
     - ``frictionloss``
     -
   * - ``joint_target_ke`` / ``kd``
     - actuator ``gainprm`` / ``biasprm``
     - ``physics:drive:stiffness`` / ``damping``
     -
     - See :ref:`mujoco-actuator-attributes`
   * - ``joint_effort_limit``
     - ``jnt_actfrcrange``
     - ``physics:drive:forceLimit``
     - ``actuatorfrcrange``
     - Per-actuator ``forcerange`` for ball joints

The ``mjc:`` USD attributes below are an alternative source for
``armature``, ``friction``, and ``limit_ke``/``limit_kd`` (the same
properties listed above).  When both a standard USD attribute and an
``mjc:`` attribute are authored, the resolution order is
Newton → PhysX → MuJoCo.

.. list-table::
   :header-rows: 1
   :widths: 25 25 20 30

   * - Newton property
     - ``mjc:`` USD attribute
     - Default
     - Notes
   * - ``armature``
     - ``mjc:armature``
     - 0.0
     -
   * - ``friction``
     - ``mjc:frictionloss`` :sup:`ext`
     - 0.0
     -
   * - ``limit_*_ke`` / ``limit_*_kd``
     - ``mjc:solref``
     - ``[0.02, 1.0]``
     - Mapped to per-axis limit stiffness / damping for all DOFs

.. note::

   The imported ``solref`` values are currently incorrectly mapped to
   ``ke``/``kd``: ``solref`` is mass-normalized, while the ``ke``/``kd``
   Newton properties have force-based units.  See
   `issue #2009 <https://github.com/newton-physics/newton/issues/2009>`_
   for details.

**Custom attributes** (``mujoco.*`` namespace):

.. list-table::
   :header-rows: 1
   :widths: 25 22 18 15 20

   * - Custom attribute
     - USD attribute
     - MJCF attribute
     - Default
     - Notes
   * - ``mujoco.limit_margin``
     - ``mjc:margin`` :sup:`ext`
     - ``margin``
     - 0.0
     - [m or rad]
   * - ``mujoco.solimplimit``
     - ``mjc:solimplimit`` :sup:`ext`
     -
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
     -
   * - ``mujoco.solreffriction``
     - ``mjc:solreffriction`` :sup:`ext`
     -
     - ``(0.02, 1.0)``
     -
   * - ``mujoco.solimpfriction``
     - ``mjc:solimpfriction`` :sup:`ext`
     -
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
     -
   * - ``mujoco.dof_passive_stiffness``
     - ``mjc:stiffness`` :sup:`ext`
     - ``stiffness``
     - 0.0
     -
   * - ``mujoco.dof_passive_damping``
     - ``mjc:damping``
     - ``damping``
     - 0.0
     -
   * - ``mujoco.dof_springref``
     - ``mjc:springref`` :sup:`ext`
     - ``springref``
     - 0.0
     - [m or rad]
   * - ``mujoco.dof_ref``
     - ``mjc:ref`` :sup:`ext`
     - ``ref``
     - 0.0
     - [m or rad]
   * - ``mujoco.jnt_actgravcomp``
     - ``mjc:actuatorgravcomp``
     - ``actuatorgravcomp``
     - ``False``
     -


.. _mujoco-shape-attributes:

Shape attributes
^^^^^^^^^^^^^^^^

**Standard Newton properties:**

.. list-table::
   :header-rows: 1
   :widths: 20 18 22 18 22

   * - Newton property
     - MuJoCo field
     - USD attribute
     - MJCF attribute
     - Notes
   * - ``shape_material_mu``
     - ``geom_friction[0]``
     - ``physics:dynamicFriction``
     - ``friction[0]``
     - Sliding friction
   * - ``shape_material_mu_torsional``
     - ``geom_friction[1]``
     - ``newton:torsionalFriction``
     - ``friction[1]``
     -
   * - ``shape_material_mu_rolling``
     - ``geom_friction[2]``
     - ``newton:rollingFriction``
     - ``friction[2]``
     -
   * - ``shape_material_ke`` / ``kd``
     - ``geom_solref``
     - ``newton:contact_ke`` / ``kd``
     - ``solref``
     - Converted via ``convert_solref()``; falls back to
       ``(0.02, 1.0)`` when zero or negative
   * - ``shape_margin``
     - ``geom_margin``
     - ``newton:contactMargin``
     - ``margin``
     -

The ``mjc:`` USD attributes below are an alternative source for
``margin``, ``gap``, ``ke``/``kd``, and ``max_hull_vertices``.

.. list-table::
   :header-rows: 1
   :widths: 25 25 20 30

   * - Newton property
     - ``mjc:`` USD attribute
     - Default
     - Notes
   * - ``max_hull_vertices``
     - ``mjc:maxhullvert``
     - -1
     -
   * - ``margin``
     - ``mjc:margin``
     - 0.0
     - Newton computes ``margin = mjc:margin − mjc:gap``
   * - ``gap``
     - ``mjc:gap``
     - 0.0
     -
   * - ``ke`` / ``kd``
     - ``mjc:solref``
     - ``[0.02, 1.0]``
     - See the ``solref`` note under :ref:`mujoco-joint-attributes`

**Custom attributes** (``mujoco.*`` namespace):

.. list-table::
   :header-rows: 1
   :widths: 25 22 18 15 20

   * - Custom attribute
     - USD attribute
     - MJCF attribute
     - Default
     - Notes
   * - ``mujoco.condim``
     - ``mjc:condim``
     -
     - 3
     -
   * - ``mujoco.geom_priority``
     - ``mjc:priority``
     - ``priority``
     - 0
     -
   * - ``mujoco.geom_solimp``
     - ``mjc:solimp``
     - ``solimp``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
     -
   * - ``mujoco.geom_solmix``
     - ``mjc:solmix``
     - ``solmix``
     - 1.0
     -


.. _mujoco-material-attributes:

Material attributes
^^^^^^^^^^^^^^^^^^^

The ``mjc:`` USD attributes below are an alternative source for material
properties on ``PhysicsMaterialAPI`` prims.  These map to the same Newton
properties listed in the :ref:`mujoco-shape-attributes` standard table
(e.g. ``mu_torsional``, ``stiffness``) but are read from the material prim
rather than the collision prim.

.. list-table::
   :header-rows: 1
   :widths: 25 25 20 30

   * - Newton property
     - USD attribute
     - Default
     - Notes
   * - ``mu_torsional``
     - ``mjc:torsionalfriction`` :sup:`ext`
     - 0.005
     -
   * - ``mu_rolling``
     - ``mjc:rollingfriction`` :sup:`ext`
     - 0.0001
     -
   * - ``priority``
     - ``mjc:priority`` :sup:`ext`
     - 0
     -
   * - ``weight``
     - ``mjc:solmix`` :sup:`ext`
     - 1.0
     -
   * - ``stiffness`` / ``damping``
     - ``mjc:solref`` :sup:`ext`
     - ``[0.02, 1.0]``
     - See the ``solref`` note under :ref:`mujoco-joint-attributes`


.. _mujoco-body-attributes:

Body attributes
^^^^^^^^^^^^^^^

**Standard Newton properties:**

.. list-table::
   :header-rows: 1
   :widths: 20 18 22 18 22

   * - Newton property
     - MuJoCo field
     - USD attribute
     - MJCF attribute
     - Notes
   * - ``body_mass``
     - ``body_mass``
     - ``physics:mass``
     - ``inertial/mass``
     - Zero-mass bodies use ``inertiafromgeom="auto"``
   * - ``body_inertia``
     - ``body_inertia`` / ``body_iquat``
     - ``physics:diagonalInertia``
     - ``inertial/diaginertia``
     - Eigendecomposed; ``fullinertia`` also supported
   * - ``body_com``
     - ``body_ipos``
     - ``physics:centerOfMass``
     - ``inertial/pos``
     - Center-of-mass offset
   * - ``body_q``
     - ``body_pos`` / ``body_quat``
     - ``xformOp:*``
     - ``pos`` / ``quat``
     - Initial pose for free-joint bodies

**Custom attributes** (``mujoco.*`` namespace):

.. list-table::
   :header-rows: 1
   :widths: 25 22 18 15 20

   * - Custom attribute
     - USD attribute
     - MJCF attribute
     - Default
     - Notes
   * - ``mujoco.gravcomp``
     - ``mjc:gravcomp`` :sup:`ext`
     - ``gravcomp``
     - 0.0
     -


.. _mujoco-actuator-attributes:

Actuator attributes
^^^^^^^^^^^^^^^^^^^

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

**Built-in properties** (from ``MjcActuator`` USD prims or MJCF ``<actuator>`` elements):

.. list-table::
   :header-rows: 1
   :widths: 25 30 20 25

   * - Newton property
     - USD attribute
     - Default
     - Notes
   * - ``ctrl_low`` / ``ctrl_high``
     - ``mjc:ctrlRange:min`` / ``max``
     - 0.0
     -
   * - ``force_low`` / ``force_high``
     - ``mjc:forceRange:min`` / ``max``
     - 0.0
     - [N]
   * - ``act_low`` / ``act_high``
     - ``mjc:actRange:min`` / ``max``
     - 0.0
     -
   * - ``length_low`` / ``length_high``
     - ``mjc:lengthRange:min`` / ``max``
     - 0.0
     - [m]
   * - ``gainPrm``
     - ``mjc:gainPrm``
     - ``[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
     -
   * - ``gainType``
     - ``mjc:gainType``
     - ``"fixed"``
     -
   * - ``biasPrm``
     - ``mjc:biasPrm``
     - ``[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
     -
   * - ``biasType``
     - ``mjc:biasType``
     - ``"none"``
     -
   * - ``dynPrm``
     - ``mjc:dynPrm``
     - ``[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
     -
   * - ``dynType``
     - ``mjc:dynType``
     - ``"none"``
     -
   * - ``gear``
     - ``mjc:gear``
     - ``[1, 0, 0, 0, 0, 0]``
     -

**Custom attributes** (``mujoco.*`` namespace):

.. list-table::
   :header-rows: 1
   :widths: 35 30 15 20

   * - Custom attribute
     - USD attribute
     - Default
     - Notes
   * - ``mujoco.actuator_gainprm``
     - ``mjc:gainPrm``
     - *(see above)*
     - Same USD attr as built-in ``gainPrm``
   * - ``mujoco.actuator_gaintype``
     - ``mjc:gainType``
     - *(see above)*
     - Same USD attr as built-in ``gainType``
   * - ``mujoco.actuator_biasprm``
     - ``mjc:biasPrm``
     - *(see above)*
     - Same USD attr as built-in ``biasPrm``
   * - ``mujoco.actuator_biastype``
     - ``mjc:biasType``
     - *(see above)*
     - Same USD attr as built-in ``biasType``
   * - ``mujoco.actuator_dynprm``
     - ``mjc:dynPrm``
     - *(see above)*
     - Same USD attr as built-in ``dynPrm``
   * - ``mujoco.actuator_dyntype``
     - ``mjc:dynType``
     - *(see above)*
     - Same USD attr as built-in ``dynType``
   * - ``mujoco.actuator_gear``
     - ``mjc:gear``
     - *(see above)*
     - Same USD attr as built-in ``gear``
   * - ``mujoco.actuator_actdim``
     - ``mjc:actDim``
     - -1 (auto)
     -
   * - ``mujoco.actuator_actearly``
     - ``mjc:actEarly``
     - ``False``
     -
   * - ``mujoco.actuator_ctrllimited``
     - ``mjc:ctrlLimited``
     - 2 (auto)
     - Tri-state: 0 = no, 1 = yes, 2 = auto
   * - ``mujoco.actuator_forcelimited``
     - ``mjc:forceLimited``
     - 2 (auto)
     - Tri-state
   * - ``mujoco.actuator_actlimited``
     - ``mjc:actLimited``
     - 2 (auto)
     - Tri-state
   * - ``mujoco.actuator_ctrlrange``
     - ``mjc:ctrlRange:min`` / ``max``
     - ``(0, 0)``
     -
   * - ``mujoco.actuator_forcerange``
     - ``mjc:forceRange:min`` / ``max``
     - ``(0, 0)``
     - [N]
   * - ``mujoco.actuator_actrange``
     - ``mjc:actRange:min`` / ``max``
     - ``(0, 0)``
     -

The ``mjc:target`` relationship on ``MjcActuator`` prims resolves the
transmission target.  ``mjc:inheritRange`` (default: 0 = disabled) copies
the target joint's limits as the actuator's ``ctrlrange``.

**Internal attributes** (``newton:mujoco:*`` USD namespace):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Custom attribute
     - USD attribute
     - Notes
   * - ``mujoco.actuator_trnid``
     - ``newton:mujoco:actuator_trnid``
     - Transmission target index pair
   * - ``mujoco.actuator_world``
     - ``newton:mujoco:actuator_world``
     - World index
   * - ``mujoco.actuator_cranklength``
     - ``newton:mujoco:actuator_cranklength``
     - Slider-crank length [m]
   * - ``mujoco.ctrl``
     - ``newton:mujoco:ctrl``
     - Direct MuJoCo actuator control signal
   * - ``mujoco.ctrl_source``
     - ``newton:mujoco:ctrl_source``
     - Control source (direct / joint target)
   * - ``mujoco.autolimits``
     - ``newton:mujoco:autolimits``
     - Auto-compute joint limits (default: ``True``)


.. _mujoco-equality-attributes:

Equality constraint attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

**Standard Newton properties:**

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Newton property
     - MuJoCo field
     - Notes
   * - ``equality_constraint_type``
     - ``eq_type``
     -
   * - ``equality_constraint_anchor``
     - ``eq_data[0:3]``
     - CONNECT anchor point
   * - ``equality_constraint_relpose``
     - ``eq_data``
     - WELD relative pose
   * - ``equality_constraint_polycoef``
     - ``eq_data[0:5]``
     - JOINT polynomial coefficients
   * - ``equality_constraint_torquescale``
     - ``eq_data[10]``
     - WELD torque scale
   * - ``equality_constraint_enabled``
     - ``eq_active``
     -
   * - ``constraint_mimic_joint0`` / ``joint1``
     - ``eq_obj1id`` / ``eq_obj2id``
     - Mimic follower / leader joint
   * - ``constraint_mimic_coef0`` / ``coef1``
     - ``eq_data[0:2]``
     - Mimic polynomial coefficients
   * - ``constraint_mimic_enabled``
     - ``eq_active``
     -

**Custom attributes** (``mujoco.*`` namespace):

.. list-table::
   :header-rows: 1
   :widths: 25 22 18 15 20

   * - Custom attribute
     - USD attribute
     - MJCF attribute
     - Default
     - Notes
   * - ``mujoco.eq_solref``
     - ``mjc:solref`` :sup:`ext`
     - ``solref``
     - ``(0.02, 1.0)``
     -
   * - ``mujoco.eq_solimp``
     - ``mjc:solimp`` :sup:`ext`
     - ``solimp``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
     -


.. _mujoco-tendon-attributes:

Tendon attributes
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 22 18 15 20

   * - Custom attribute
     - USD attribute
     - MJCF attribute
     - Default
     - Notes
   * - ``mujoco.tendon_stiffness``
     - ``mjc:stiffness`` :sup:`ext`
     - ``stiffness``
     - 0.0
     - [N/m]
   * - ``mujoco.tendon_damping``
     - ``mjc:damping`` :sup:`ext`
     - ``damping``
     - 0.0
     - [N·s/m]
   * - ``mujoco.tendon_frictionloss``
     - ``mjc:frictionloss`` :sup:`ext`
     - ``frictionloss``
     - 0.0
     - [N]
   * - ``mujoco.tendon_limited``
     - ``mjc:limited`` :sup:`ext`
     - ``limited``
     - 2 (auto)
     - Tri-state
   * - ``mujoco.tendon_range``
     - ``mjc:range:min`` / ``max`` :sup:`ext`
     - ``range``
     - ``(0, 0)``
     - [m]
   * - ``mujoco.tendon_margin``
     - ``mjc:margin`` :sup:`ext`
     - ``margin``
     - 0.0
     - [m]
   * - ``mujoco.tendon_solref_limit``
     - ``mjc:solreflimit`` :sup:`ext`
     - ``solreflimit``
     - ``(0.02, 1.0)``
     -
   * - ``mujoco.tendon_solimp_limit``
     - ``mjc:solimplimit`` :sup:`ext`
     - ``solimplimit``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
     -
   * - ``mujoco.tendon_solref_friction``
     - ``mjc:solreffriction`` :sup:`ext`
     - ``solreffriction``
     - ``(0.02, 1.0)``
     -
   * - ``mujoco.tendon_solimp_friction``
     - ``mjc:solimpfriction`` :sup:`ext`
     - ``solimpfriction``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
     -
   * - ``mujoco.tendon_armature``
     - ``mjc:armature`` :sup:`ext`
     - ``armature``
     - 0.0
     - [kg]
   * - ``mujoco.tendon_springlength``
     - ``mjc:springlength`` :sup:`ext`
     - ``springlength``
     - ``(-1, -1)``
     - [m]; ``-1`` = use model length
   * - ``mujoco.tendon_actuator_force_limited``
     - ``mjc:actuatorfrclimited`` :sup:`ext`
     - ``actuatorfrclimited``
     - 2 (auto)
     - Tri-state
   * - ``mujoco.tendon_actuator_force_range``
     - ``mjc:actuatorfrcrange:min`` / ``max`` :sup:`ext`
     - ``actuatorfrcrange``
     - ``(0, 0)``
     - [N]

Spatial tendons additionally use ``mjc:path`` (relationship),
``mjc:path:indices``, and ``mjc:path:coef`` to describe the wrap path.

**Internal attributes** (``newton:mujoco:*`` USD namespace):

These are populated automatically during MJCF/USD import or set
programmatically.  Newton reads them from USD using the ``newton:mujoco:``
prefix.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Custom attribute
     - USD attribute
     - Notes
   * - ``mujoco.tendon_world``
     - ``newton:mujoco:tendon_world``
     - World index
   * - ``mujoco.tendon_type``
     - ``newton:mujoco:tendon_type``
     - 0 = fixed, 1 = spatial
   * - ``mujoco.tendon_joint_adr``
     - *(computed)*
     - Start address into fixed-tendon joint arrays
   * - ``mujoco.tendon_joint_num``
     - *(computed)*
     - Number of joints in fixed tendon
   * - ``mujoco.tendon_wrap_adr``
     - ``newton:mujoco:tendon_wrap_adr``
     - Start address into wrap arrays
   * - ``mujoco.tendon_wrap_num``
     - ``newton:mujoco:tendon_wrap_num``
     - Number of wrap elements
   * - ``mujoco.tendon_joint``
     - ``newton:mujoco:tendon_joint``
     - Joint index per fixed-tendon entry
   * - ``mujoco.tendon_coef``
     - ``newton:mujoco:tendon_coef``
     - Joint coefficient per fixed-tendon entry
   * - ``mujoco.tendon_wrap_type``
     - ``newton:mujoco:tendon_wrap_type``
     - Wrap element type (site / geom / pulley)
   * - ``mujoco.tendon_wrap_shape``
     - ``newton:mujoco:tendon_wrap_shape``
     - Wrap element shape index
   * - ``mujoco.tendon_wrap_sidesite``
     - ``newton:mujoco:tendon_wrap_sidesite``
     - Side-site shape index
   * - ``mujoco.tendon_wrap_prm``
     - ``newton:mujoco:tendon_wrap_prm``
     - Wrap element parameter


.. _mujoco-contact-pair-attributes:

Contact pair attributes
^^^^^^^^^^^^^^^^^^^^^^^

Explicit contact pairs (``mujoco:pair`` frequency) are populated from
MJCF import, USD import (via ``newton:mujoco:*`` attributes), or
programmatic assignment.

.. list-table::
   :header-rows: 1
   :widths: 25 22 18 15 20

   * - Custom attribute
     - USD attribute
     - MJCF attribute
     - Default
     - Notes
   * - ``mujoco.pair_world``
     - ``newton:mujoco:pair_world``
     -
     - 0
     - World index
   * - ``mujoco.pair_geom1`` / ``pair_geom2``
     - ``newton:mujoco:pair_geom1`` / ``newton:mujoco:pair_geom2``
     - ``geom1`` / ``geom2``
     - -1
     - Shape indices
   * - ``mujoco.pair_condim``
     - ``newton:mujoco:pair_condim``
     - ``condim``
     - 3
     -
   * - ``mujoco.pair_solref``
     - ``newton:mujoco:pair_solref``
     - ``solref``
     - ``(0.02, 1.0)``
     -
   * - ``mujoco.pair_solreffriction``
     - ``newton:mujoco:pair_solreffriction``
     - ``solreffriction``
     - ``(0.02, 1.0)``
     -
   * - ``mujoco.pair_solimp``
     - ``newton:mujoco:pair_solimp``
     - ``solimp``
     - ``(0.9, 0.95, 0.001, 0.5, 2.0)``
     -
   * - ``mujoco.pair_margin``
     - ``newton:mujoco:pair_margin``
     - ``margin``
     - 0.0
     -
   * - ``mujoco.pair_gap``
     - ``newton:mujoco:pair_gap``
     - ``gap``
     - 0.0
     -
   * - ``mujoco.pair_friction``
     - ``newton:mujoco:pair_friction``
     - ``friction``
     - ``(1, 1, 0.005, 0.0001, 0.0001)``
     - ``(slide, slide, torsion, roll, roll)``


.. _mujoco-custom-frequencies:

Custom frequencies
^^^^^^^^^^^^^^^^^^

:meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes` also registers
five custom frequencies for variable-length entity types:

- ``mujoco:pair`` — explicit contact pairs
- ``mujoco:actuator`` — general MuJoCo actuators
- ``mujoco:tendon`` — fixed and spatial tendons
- ``mujoco:tendon_joint`` — per-joint entries in fixed tendons
- ``mujoco:tendon_wrap`` — wrap path elements in spatial tendons

:sup:`ext` = Newton extension — not defined in the mjcPhysics schema.  These
correspond to MuJoCo XML attributes that do not yet have a schema counterpart.


.. _mujoco-unsupported-attributes:

Unsupported mjcPhysics schema attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following mjcPhysics schema attributes are **not** parsed by Newton
from USD.  Authoring them on USD prims has no effect.  Some of these are
parsed from MJCF (noted below).

- **Compiler options** (``mjc:compiler:*``): all 14 compiler attributes
  (``alignFree``, ``angle``, ``autoLimits``, ``balanceInertia``, etc.)
- **Flags** (``mjc:flag:*``): all flags except ``mjc:flag:gravity``
  (``actuation``, ``clampctrl``, ``contact``, ``equality``, ``multiccd``,
  ``nativeccd``, ``warmstart``, etc.)
- **Keyframe** (``MjcKeyframe``): ``mjc:qpos``, ``mjc:qvel``,
  ``mjc:mpos``, ``mjc:mquat``
- **Shape / collision**: ``mjc:inertia`` (mesh inertia mode),
  ``mjc:shellinertia``, ``mjc:group``
- **Actuator**: ``mjc:act``,
  ``mjc:jointInParent``, ``mjc:refSite``, ``mjc:sliderSite``,
  ``mjc:group``.
  Note: ``mjc:crankLength`` and ``mjc:ctrl`` are not parsed from USD but
  **are** parsed from MJCF (see ``mujoco.actuator_cranklength`` and
  ``mujoco.ctrl`` above).
- **Solver options**: ``mjc:option:noslip_iterations``,
  ``mjc:option:noslip_tolerance``, ``mjc:option:actuatorgroupdisable``,
  ``mjc:option:o_friction``, ``mjc:option:o_margin``,
  ``mjc:option:o_solimp``, ``mjc:option:o_solref``


.. _mujoco-runtime-state-and-control:

Runtime state and control
^^^^^^^^^^^^^^^^^^^^^^^^^

During :meth:`~newton.solvers.SolverMuJoCo.step`, the solver reads from
:class:`~newton.State` and :class:`~newton.Control` and writes results back to
the next :class:`~newton.State`.

**Inputs** (Newton → MuJoCo):

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Newton property
     - MuJoCo field
     - Notes
   * - ``state.joint_q``
     - ``qpos``
     - Quaternion order converted (xyzw → wxyz)
   * - ``state.joint_qd``
     - ``qvel``
     -
   * - ``state.body_f``
     - ``xfrc_applied``
     - External body wrenches
   * - ``control.joint_target_pos``
     - actuator ``ctrl``
     - For POSITION / POSITION_VELOCITY modes
   * - ``control.joint_target_vel``
     - actuator ``ctrl``
     - For VELOCITY / POSITION_VELOCITY modes
   * - ``control.joint_f``
     - ``qfrc_applied``
     - Joint-space forces; free-joint forces go to ``xfrc_applied``
   * - ``control.mujoco.ctrl``
     - ``ctrl``
     - Direct MuJoCo actuator control (general actuators)

**Outputs** (MuJoCo → Newton):

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Newton property
     - MuJoCo field
     - Notes
   * - ``state.joint_q``
     - ``qpos``
     - Quaternion order converted back (wxyz → xyzw)
   * - ``state.joint_qd``
     - ``qvel``
     -
   * - ``state.body_q``
     - *(FK)*
     - Computed via :func:`~newton.eval_articulation_fk`
   * - ``state.body_qd``
     - *(FK)*
     - Computed via :func:`~newton.eval_articulation_fk`
   * - ``state.body_qdd``
     - ``cacc``
     - Body accelerations; only when field is non-None
   * - ``state.body_parent_f``
     - ``cfrc_int``
     - Incoming joint wrenches; only when field is non-None
   * - ``state.mujoco.qfrc_actuator``
     - ``qfrc_actuator``
     - Actuator forces in joint space; only when field is non-None


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

**joint_enabled is not supported.**
  Newton's per-joint ``joint_enabled`` flag has no effect in the MuJoCo solver.

**DISTANCE and CABLE joints are not supported.**
  These joint types cannot be represented in MuJoCo and will raise an error.

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


