.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

MuJoCo Solver
=============

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
   * - :attr:`~newton.JointType.ROD`
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


.. _joint-limit-stiffness-and-damping:

Joint-limit stiffness and damping
---------------------------------

:attr:`~newton.Model.joint_limit_ke` and
:attr:`~newton.Model.joint_limit_kd` are force-space gains (for example,
``N·m/rad`` and ``N·m·s/rad`` for revolute joints). MuJoCo converts
``solreflimit`` to an effective limit response using the owning DOF's
``dof_invweight0`` and the limit impedance parameter
``dmax = solimplimit[1]``:

.. math::

   k_\mathrm{eff} = k_\mathrm{stored} /
   (\mathrm{dof\_invweight0} \cdot (1 - dmax))

To keep Newton's force-space meaning,
:class:`~newton.solvers.SolverMuJoCo` first scales the direct
stiffness/damping pair by ``factor = dof_invweight0 * (1 - dmax)`` and then
converts that pair to MuJoCo's positive ``(timeconst, dampratio)`` convention:

.. math::

   \begin{aligned}
   k_\mathrm{stored} &= ke \cdot factor \\
   b_\mathrm{stored} &= kd \cdot factor \\
   \mathrm{timeconst} &= 2 / b_\mathrm{stored} \\
   \mathrm{dampratio} &= b_\mathrm{stored} /
      (2 \sqrt{k_\mathrm{stored}})
   \end{aligned}

The positive convention preserves the same unclamped force-space response as
the equivalent direct stiffness/damping pair while allowing MuJoCo's
``refsafe`` timestep clamp to soften limits that are too stiff for the step
size. This update runs after MuJoCo has compiled or refreshed
``dof_invweight0``. If ``joint_limit_ke <= 0`` or ``joint_limit_kd <= 0``, the
solver restores MuJoCo's default ``solreflimit`` value ``(0.02, 1.0)``.

MJCF- or USD-authored ``solreflimit`` values are already native MuJoCo
parameters, so they are preserved verbatim through the
``model.mujoco.solreflimit`` custom attribute and are not rescaled. Neither
MJCF import nor USD import through the MuJoCo schema derives generic
``joint_limit_ke`` or ``joint_limit_kd`` values from ``solreflimit``; those
gains retain Newton builder defaults or values supplied by a generic schema.
An authored native value wins in :class:`~newton.solvers.SolverMuJoCo`, while
the generic gains remain available to other solvers. Imported joints that did
not author ``solreflimit`` keep MuJoCo's implicit default ``(0.02, 1.0)`` until
either generic gain is configured or edited, including edits made before
constructing the solver, at which point the Newton force-space scaling above is
used. ``model.mujoco.solreflimit_gain_baseline`` records the generic gains seen
at import time so that edits made before construction can be detected.

``model.mujoco.solreflimit_mode`` records how ``solreflimit`` should be
interpreted: Newton force-space gains, a raw authored MuJoCo value, or an
implicit MJCF default. This extra flag is needed because the two-component
``solreflimit`` value alone cannot distinguish an unauthored value from an
authored native value such as ``solreflimit="0 0"`` or USD
``mjc:solreflimit = [0, 0]``.

.. note::

   ``SolverMuJoCo(..., save_to_mjcf=path)`` is not a fully semantic
   round-trip for ``SOLREF_MODE_FORCE_SPACE`` joints. MJCF only stores
   ``solreflimit``; it has no field for "use Newton force-space
   scaling with these gains". The exporter therefore only writes
   ``solreflimit`` for ``SOLREF_MODE_RAW`` joints (where the authored
   value carries the full intent). ``joint_limit_ke`` /
   ``joint_limit_kd`` from the original ``SOLREF_MODE_FORCE_SPACE`` /
   ``SOLREF_MODE_MJCF_DEFAULT`` joints will not be preserved; reapply
   them on the rebuilt model if you need those force-space gains.


.. _shape-material-contact-stiffness-and-damping:

Shape-material contact stiffness and damping
--------------------------------------------

:attr:`~newton.Model.shape_material_ke` and
:attr:`~newton.Model.shape_material_kd` are force-space stiffness and
damping (``N/m`` and ``N·s/m``), but their realized response depends on the
active mapping. On the force-space Newton-contacts path,
:class:`~newton.solvers.SolverMuJoCo` mixes the two shapes' gains and scales the
pair by ``1 - dmax`` and the sum of the bodies' translational
``body_invweight0[..., 0]`` values before writing per-contact ``solref``. Here
``dmax = solimp[1]``. This inverse-weight scaling is Newton's implemented
force-space contract, but it is not the full scalar contact effective mass
:math:`(J M^{-1} J^T)^{-1}` for arbitrary articulated or off-center contacts.

``model.mujoco.solref_mode`` (per shape) records how
``shape_material_ke`` / ``shape_material_kd`` and ``mujoco.solref``
combine, with the same three states as joint limits. The list describes existing
model and implementation behavior; the constants are not a public selection
API:

* ``SOLREF_MODE_FORCE_SPACE`` — Newton force-space gains; the per-contact
  factor above applies.
* ``SOLREF_MODE_RAW`` — forward the authored ``mujoco.solref`` (e.g.
  from an MJCF/USD import) unchanged.
* ``SOLREF_MODE_MJCF_DEFAULT`` — registered default; preserves MuJoCo's
  compile-time contact dynamics and the legacy unit-mass numerical
  ``convert_solref(ke, kd, 1, 1)`` round-trip in ``geom_solref``.

.. _mujoco-contact-solref-conversion:

Contact ``solref`` conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For positive numeric ``ke`` and ``kd``, the legacy
``convert_solref(ke, kd, 1, 1)`` mapping writes:

.. math::

   \mathrm{timeconst} = 2 / kd, \qquad
   \mathrm{dampratio} = kd / (2 \sqrt{ke})

Under this unit-mass numerical mapping, ``sqrt(ke)`` behaves like a frequency
and holding the numerical damping ratio fixed requires ``kd`` to scale with
``sqrt(ke)``. This is not a dimensionally physical critical-damping rule for an
arbitrary contact mass. The force-space path applies the implemented
``body_invweight0`` factor above before the same conversion; the formula then
describes the scaled numerical ``solref``, not an exact Jacobian-derived contact
response. See the implementation in
:github:`newton/_src/solvers/mujoco/kernels.py` and its force-space contact tests
in :github:`newton/tests/test_mujoco_solver.py`.

With MuJoCo's default ``refsafe`` guard enabled, MuJoCo-Warp evaluates positive
``solref`` using ``max(timeconst, 2 * dt)`` for the active constraint without
rewriting the stored value. Direct-format negative ``solref`` bypasses this
clamp. Once a requested positive ``timeconst`` falls below that floor, raising
the gains at a fixed damping ratio no longer hardens the effective response;
reduce the step passed to :meth:`~newton.solvers.SolverMuJoCo.step` instead. See
MuJoCo's `refsafe option
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-flag-refsafe>`__.

These ``SOLREF_MODE_*`` names describe internal mode values; they are not
public Newton symbols. MJCF/USD import selects the appropriate authored/default
mode automatically. Do not import the constants from ``newton._src`` to change
the mode from user code.

.. note::

   ``use_mujoco_contacts=True`` and the MuJoCo CPU backend do not
   apply the per-contact two-body factor — MuJoCo's internal
   ``contact_params`` averages per-geom ``solref``, which cannot
   reproduce the inverse-mass sum. ``SOLREF_MODE_FORCE_SPACE`` shapes
   fall back to the legacy ``convert_solref(ke, kd, 1, 1)``
   approximation on those paths.

For parameter interpretation, stability tradeoffs, and task-oriented guidance,
see :ref:`Tuning MuJoCo`.

.. _mujoco-contact-friction-solreffriction:

Contact friction ``solreffriction`` mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For :class:`~newton.solvers.SolverMuJoCo`, ``kf`` maps to MuJoCo's per-contact
``solreffriction`` when the MuJoCo Warp backend uses elliptic friction cones
(``use_mujoco_cpu=False``, ``cone="elliptic"``) with Newton contacts
(``use_mujoco_contacts=False``). It targets the force-space friction slope
``f = -kf * v`` below the Coulomb limit. The mapping is exact when the sum of
MuJoCo's translational ``body_invweight0`` values matches the contact's inverse
effective mass (the relevant diagonal of :math:`J M^{-1} J^T`) and the contact
operates at its maximum impedance ``dmax``. Very large ``kf`` saturates at
MuJoCo's refsafe stability bound, where the reference time constant is clamped
to twice the timestep.

The two shapes' ``kf`` values combine with the usual priority/``solmix``
weighting. A resolved ``kf = 0`` makes the contact frictionless
(``condim = 1``), removing its sliding, torsional, and rolling friction rows.
If a positive ``kf`` cannot produce a positive, finite inverse-weight
denominator, ``solreffriction`` remains unset and MuJoCo inherits the normal
``solref``. The mapping is independent of the shape's ``solref_mode`` above,
which only governs the normal-direction ``solref``.

The slope is calibrated for the sliding friction rows. With ``condim > 3``, the
torsional and rolling rows share the same per-contact ``solreffriction`` and
MuJoCo scales their regularization by the corresponding friction-coefficient
ratios, so their effective damping deviates from ``kf`` accordingly.

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


.. _mujoco-equality-constraints:

Equality constraints
--------------------

Use :class:`~newton.solvers.SolverMuJoCo.EqType` for MuJoCo equality
constraint types. The top-level :class:`newton.EqType` alias is deprecated
in Newton 1.4.

Each row's ``data[...]`` reference below points into MuJoCo's
`equality.data <https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality>`_
array; slot layout depends on the constraint type.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Newton type
     - MuJoCo equivalent
     - Notes
   * - :attr:`~newton.solvers.SolverMuJoCo.EqType.CONNECT`
     - ``mjEQ_CONNECT``
     - Anchor forwarded in ``data[0:3]``.
   * - :attr:`~newton.solvers.SolverMuJoCo.EqType.WELD`
     - ``mjEQ_WELD``
     - Anchor forwarded in ``data[0:3]``, relative pose in ``data[3:10]``,
       torque scale in ``data[10]``.
   * - :attr:`~newton.solvers.SolverMuJoCo.EqType.JOINT`
     - ``mjEQ_JOINT``
     - Polynomial coefficients forwarded in ``data[0:5]``.
   * - Mimic
     - ``mjEQ_JOINT``
     - Added via :meth:`~newton.ModelBuilder.add_constraint_mimic`. Maps
       ``coef0`` / ``coef1`` to polynomial coefficients. Only
       :attr:`~newton.JointType.REVOLUTE` and
       :attr:`~newton.JointType.PRISMATIC` joints are supported.

Newton's core API does not expose equality constraints as a dedicated
builder call. Construct them through the MuJoCo
:ref:`custom-attribute namespace <mujoco-custom-attributes>` with
:meth:`~newton.ModelBuilder.add_custom_values` using the
``mujoco:equality_constraint_*`` keys, then read or update finalized
fields via ``model.mujoco.equality_constraint_*``.

For example, add a connect constraint between two body indices in the
active world as follows. Fields that do not apply to connect constraints
retain their registered defaults.

.. code-block:: python

   import newton
   import warp as wp

   builder.add_custom_values(
       **{
           "mujoco:equality_constraint_type": int(
               newton.solvers.SolverMuJoCo.EqType.CONNECT
           ),
           "mujoco:equality_constraint_body1": body1,
           "mujoco:equality_constraint_body2": body2,
           "mujoco:equality_constraint_anchor": wp.vec3(0.0, 0.0, 0.0),
           "mujoco:equality_constraint_enabled": True,
           "mujoco:equality_constraint_world": builder.current_world,
       }
   )

.. _mujoco-loop-closures:

Loop closures
-------------

Loop-closing joints (see :ref:`Loop closure` for the general authoring
pattern) are not emitted as MuJoCo joints; instead the solver constrains
the relative motion of the two bodies according to the joint type:

- :attr:`~newton.JointType.FIXED` — all 6 relative DOFs constrained
  (relative position and orientation locked).
- :attr:`~newton.JointType.REVOLUTE` — 5 DOFs constrained; one rotational
  DOF about the hinge axis remains free.
- :attr:`~newton.JointType.BALL` — the 3 translational DOFs constrained;
  all 3 rotational DOFs remain free.

Other joint types used as loop closures
(:attr:`~newton.JointType.PRISMATIC`, :attr:`~newton.JointType.FREE`,
:attr:`~newton.JointType.DISTANCE`, :attr:`~newton.JointType.ROD`) emit a
warning and are silently skipped — the loop is *not* closed. A
:attr:`~newton.JointType.D6` is dispatched by its degrees of freedom: one
angular axis behaves as a revolute closure and three as a ball closure;
any other configuration is skipped.

Only the kinematic coupling implied by the joint type is enforced. Any
drive (``joint_target_q`` / ``joint_target_qd``, PD gains,
``control.joint_f``), joint limits, armature, friction, and
effort/velocity limits authored on the loop-closing joint are **ignored**
by :class:`~newton.solvers.SolverMuJoCo`. Loop-joint DOFs and coordinates
are excluded from MuJoCo's ``nq`` / ``nv``.


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


.. _mujoco-collision-pipeline:

Collision pipeline
------------------

:class:`~newton.solvers.SolverMuJoCo` uses MuJoCo's built-in collision
detection by default. Construct it with ``use_mujoco_contacts=False``
to feed contacts computed by Newton's own collision pipeline into
:meth:`~newton.solvers.SolverMuJoCo.step` instead.
Newton's pipeline supports non-convex meshes, SDF-based contacts, and
hydroelastic contacts, which are not available through MuJoCo's collision
detection.

Collision filtering
~~~~~~~~~~~~~~~~~~~

MuJoCo gives every geom two 32-bit masks, ``contype`` and ``conaffinity``.
Together, these masks decide whether two geoms are allowed to collide. For a
candidate pair ``a, b``, the mask test passes when
``(contype_a & conaffinity_b) != 0`` or
``(contype_b & conaffinity_a) != 0``. MuJoCo then applies other selection
rules, including same-body suppression and body-wide ``<exclude>`` elements;
explicit ``<pair>`` elements bypass the automatic mask test. See MuJoCo's
`collision selection documentation
<https://mujoco.readthedocs.io/en/stable/computation/index.html#selection>`__
and the `geom mask attributes
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom>`__.

**Importing MJCF masks.**

:func:`~newton.utils.parse_mjcf` resolves inherited ``contype`` and
``conaffinity`` values and determines which shape pairs may collide. It stores
the same result in Newton collision groups and explicit excluded pairs. The
original 32-bit values are also retained as
``model.mujoco.contype`` and ``model.mujoco.conaffinity`` custom attributes
for a lossless round trip back to MuJoCo.

A *mask domain* is the set of shapes whose mask bits were authored together.
Each :func:`~newton.utils.parse_mjcf` call creates a new domain and records it
in the internal ``model.mujoco.collision_mask_domain`` attribute. The domain
is only a source label. It is not another collision mask and does not enable
or disable contacts. When :meth:`~newton.ModelBuilder.add_builder` combines
separately built models, it gives the copied domains new IDs so they remain
distinct from domains already in the destination builder.

**Choosing masks for a MuJoCo solver.**

With ``use_mujoco_contacts=True``, preserved source masks are forwarded
verbatim only when every selected collision shape has masks from the same
domain and those masks already enforce all active Newton pair filters.
Same-body filtering and imported body-wide ``<exclude>`` elements also count
as enforced. This path preserves the source MJCF exactly. For a single import,
the original masks therefore remain the source of truth even if its Newton
collision groups are later edited.

A native shape, shapes from more than one domain, or a new Newton pair filter
can make the original masks unsafe to reuse. The solver then lists the shape
pairs that Newton allows and creates new MuJoCo masks that reproduce that
list.

**Example: combining two MJCF files.**

Suppose file A and file B both use bit 0. In file A, bit 0 may control contacts
between its floor and spheres. File B may reuse bit 0 for its own shapes. The
files were authored independently, so that shared number says nothing about
how a shape from A should interact with a shape from B.

After both files are added to one builder, Newton's collision groups and
excluded pairs define those new cross-file interactions. Copying the original
masks would make MuJoCo treat bit 0 as one global rule and could allow or block
the wrong cross-file pairs. Because the shapes have different domains, the
solver instead creates new masks from Newton's final list of allowed pairs.

**Compiling Newton filtering to MuJoCo masks.**

MuJoCo provides only 32 mask bits. One bit can encode all collisions between
one set of shapes and another set. In graph terminology, that rule is a
complete bipartite graph, or biclique. The solver tries to reproduce Newton's
full list of allowed pairs using at most 32 such rules. It is guaranteed to
find an exact result for up to 33 selected shapes and often handles much larger
models whose collision groups have a regular structure.

If the rules do not fit in 32 bits, the solver uses the established legacy
graph-color approximation, which may allow extra contacts. Finding an exact
result requires checking every shape pair, so models above 256 selected shapes
or 1,024 explicit excluded pairs skip directly to that fallback.

.. _mujoco-margin-gap-mapping:

Margin and gap mapping
~~~~~~~~~~~~~~~~~~~~~~

:attr:`~newton.Model.shape_margin` maps to MuJoCo
``geom_margin`` and :attr:`~newton.Model.shape_gap` maps to ``geom_gap``;
authored contact-pair values similarly map to ``pair_margin`` and ``pair_gap``.
The margin mapping is subject to *Margin zeroing* below. The solver forwards gap
values at construction and runtime property updates. MuJoCo reports surface
distances in ``(margin, margin + gap]`` as inactive contacts without contact
force. See :ref:`margin and gap semantics
<margin-gap-semantics>` for Newton's contact geometry and MuJoCo's `margin and
gap model
<https://mujoco.readthedocs.io/en/stable/computation/index.html#margin-and-gap>`__
for the three contact regimes.

MJCF and USD margin/gap values use direct MuJoCo 3.9+ semantics. Pass
``legacy_margin_gap=True`` to :meth:`~newton.ModelBuilder.add_mjcf` or
:meth:`~newton.ModelBuilder.add_usd` only when reproducing Newton's pre-3.9
import translation.

**Multi-contact CCD.** Constructing
:class:`~newton.solvers.SolverMuJoCo` with ``enable_multiccd=True``
allows up to four contact points per geom pair instead of one. Pairs
where either geom has non-zero MuJoCo ``geom_margin`` still fall back
to a single contact regardless of the flag (see *Margin zeroing*
below for how Newton's :attr:`~newton.Model.shape_margin` is forwarded
to it). MuJoCo Warp currently remains single-contact for cylinder--box;
see :ref:`Geometry Pair Contact Behavior`.

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


Sleeping
--------

MuJoCo Warp can exclude stationary constraint islands from collision
detection and the compact Newton solver. Newton keeps this optimization
disabled by default. Enable it when simulating many passive objects that
spend substantial time at rest. Sleeping adds island bookkeeping and uses the
compact-solver path even while everything remains awake, so scenes in which
most degrees of freedom remain awake can be slower than with sleeping disabled.
Benchmark representative workloads before enabling it::

    solver = SolverMuJoCo(
        model,
        enable_sleeping=True,
        sleep_tolerance=0.01,
    )

Sleeping is only supported by Newton's MuJoCo Warp GPU path. It requires
``solver="newton"`` and ``use_mujoco_contacts=True`` so MuJoCo Warp's
collision pipeline can wake sleeping bodies, and it does not support the RK4
integrator. Unsupported combinations raise ``ValueError`` during solver
construction. When an imported MJCF contains
``<option><flag sleep="enable"/></option>``, leaving ``enable_sleeping`` as
``None`` honors that setting. An explicit constructor value takes precedence.

``sleep_tolerance`` is the scaled generalized-velocity threshold [m/s] below
which an island becomes eligible to sleep. It follows the normal
`Solver options`_ resolution order, including per-world
``model.mujoco.sleep_tolerance`` values. The default is ``0.001``.

MuJoCo's per-tree ``body/sleep`` policy is available as the body-frequency
``model.mujoco.sleep_policy`` custom attribute and is imported from MJCF.
The supported values are :attr:`~newton.solvers.SolverMuJoCo.SleepPolicy.AUTO`,
:attr:`~newton.solvers.SolverMuJoCo.SleepPolicy.NEVER`,
:attr:`~newton.solvers.SolverMuJoCo.SleepPolicy.ALLOWED`, and
:attr:`~newton.solvers.SolverMuJoCo.SleepPolicy.INIT`. The policy must be
assigned to the moving root body of a MuJoCo kinematic tree. For a
programmatically built model, register the MuJoCo custom attributes before
adding the body::

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    root = builder.add_link(
        custom_attributes={
            "mujoco:sleep_policy": SolverMuJoCo.SleepPolicy.INIT,
        },
    )

``INIT`` makes the tree start asleep and is the intended way to construct a
solver with compact ``nvmax`` storage. The setting applies to the shared model,
so replicated worlds receive the same initial tree policies. Their default
joint velocities must also match because MuJoCo Warp uses one shared initial
sleep state; solver construction rejects differing per-world defaults.

``nvmax`` sizes the compact solver's active-DOF workspace per world. If it is
``None``, MuJoCo Warp uses the model's full ``nv``. This is the safe starting
point but does not reduce compact-workspace memory. Tune a smaller value to
the maximum number of DOFs expected to be awake simultaneously. The capacity
must include every DOF that is awake in the initial MuJoCo state; Newton rejects
smaller values before allocating solver data. ``nvmax`` is rejected when
sleeping is disabled because MuJoCo Warp would otherwise allocate unused
compact-solver workspace. If more than ``nvmax`` DOFs become active later,
MuJoCo Warp sets the ``NVMAX`` bit in
``solver.mjw_data.overflow``; increase ``nvmax`` and recreate the solver.

MuJoCo Warp automatically wakes islands for applied forces and contacts.
Actuated trees are not allowed to sleep by default. If their sleep policy is
explicitly changed to allow sleeping, changing actuator controls alone does not
wake them; apply a force or set a nonzero velocity first. Newton-side
joint-position edits wake only the affected sleeping trees.
:meth:`~newton.solvers.SolverMuJoCo.reset` restores the initial sleep state in
selected worlds, while
:meth:`~newton.solvers.SolverMuJoCo.notify_model_changed` wakes all worlds
after model-property updates. The sleeping path supports whole-step CUDA graph
capture.

See MuJoCo's `sleeping-islands documentation
<https://mujoco.readthedocs.io/en/stable/programming/simulation.html#sleeping-islands>`_
and `MuJoCo Warp compact-solver guide
<https://mujoco.readthedocs.io/en/latest/mjwarp/#compact-solver>`_ for the
underlying sleep policy, wake triggers, and ``nvmax`` tuning guidance.


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
(rather than the ``mujoco`` namespace). The MJCF parser handles these
inline (:github:`newton/_src/utils/import_mjcf.py`); USD goes through
:class:`~newton.usd.SchemaResolverMjc`
(:github:`newton/_src/usd/schemas.py`). Native MuJoCo solver parameters
such as MJCF ``solreflimit`` or USD ``mjc:solreflimit`` remain in the
``mujoco`` namespace and do not overwrite Newton's generic force-space
joint-limit gains.

MuJoCo joint ``damping`` maps to :attr:`~newton.Model.joint_damping`.
When importing MuJoCo-authored USD, opt into that mapping explicitly::

    from newton.usd import SchemaResolverMjc, SchemaResolverNewton

    builder.add_usd(
        stage,
        schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
    )

Registering MuJoCo custom attributes alone enables the ``model.mujoco``
namespace but does not map ``mjc:damping`` to the built-in property.


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
- **Actuator transmissions** — ``joint``, ``jointinparent``, ``tendon``,
  ``site``, ``body``, and ``slidercrank`` transmissions are supported (see
  :class:`~newton.solvers.SolverMuJoCo.TrnType` for the enum).

Smaller limitations are documented inline where they are most relevant —
see `Caveats`_ below for collision-radius, convex-hull fallback, and
velocity limits; and the unsupported rows in `Joint types`_ and
`Geometry types`_.


Caveats
-------

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

**Collision filtering has a 32-bit capacity.**
  The solver creates MuJoCo masks that reproduce Newton's allowed collision
  pairs when they fit in 32 bits. See `Collision filtering`_ for the behavior
  of imported masks and the warned fallback used when the rules do not fit.


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
  ``mjData.mocap_pos`` and ``mjData.mocap_quat`` instead. A D6 root
  with all axes locked (e.g. imported from a generic USD
  ``PhysicsJoint``) is treated the same way.
- **World-attached shapes that are not part of an articulation**
  remain ordinary static MuJoCo geometry rather than mocap bodies.

If you edit :attr:`~newton.Model.joint_X_p` or :attr:`~newton.Model.joint_X_c`
for a fixed-root articulation after constructing the solver, call
:meth:`~newton.solvers.SolverBase.notify_model_changed` with the
:attr:`~newton.ModelFlags.JOINT_PROPERTIES` flag to
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
