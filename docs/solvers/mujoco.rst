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


.. _native-mjcf-import:

Native MJCF import
------------------

.. experimental::

   :meth:`ModelBuilder.add_mjcf_native` is an independent, opt-in importer.
   Its coverage and mapping to Newton may change.

Choose the importer explicitly:

.. code-block:: python

   builder.add_mjcf("robot.xml")         # Existing Python importer
   builder.add_mjcf_native("robot.xml")  # Native MuJoCo parser and compiler

The existing method has no new dependency. The native method requires MuJoCo
(``uv sync --extra sim``) and accepts the same keyword arguments. It copies
native model data into the builder and retains no ``MjSpec`` or ``MjModel``.
Each source must pass MuJoCo compilation independently. Imported assets can
then be composed in Newton. Builder finalization resolves parameters supported
by Newton's core representation; remaining MuJoCo-specific parameters resolve
when ``SolverMuJoCo`` compiles the assembled model, as described below.
Native defaults, includes, frames, replication, mesh processing, and inertia
inference run in MuJoCo. A custom ``path_resolver`` uses the existing include
and asset-path loader before native parsing. Unknown XML attributes are errors;
they are not interpreted as Newton custom attributes.

Both importers raise ``FileNotFoundError`` for a missing source file. For
missing include files, legacy import raises ``FileNotFoundError`` while native
parsing without a custom resolver raises ``ValueError`` with MuJoCo's diagnostic
and the original exception as its cause. Custom path-resolution errors propagate
unchanged.

MuJoCo rejects some repeated-include layouts, including sibling includes of the
same file. To expand sibling or nested repeated includes using Newton's existing
loader before native parsing, supply a filesystem resolver:

.. code-block:: python

   from pathlib import Path

   def resolve(base_dir, filename):
       return str((Path(base_dir or ".") / filename).resolve())

   builder.add_mjcf_native("robot.xml", path_resolver=resolve)

This path still rejects include cycles and preserves the legacy loader's
per-file asset directories and missing-include ``FileNotFoundError``.
MuJoCo validates the expanded document, including object-name uniqueness;
repeated includes do not automatically rename objects.

The importer supports the following rigid-body features:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Area
     - Native import coverage
   * - Bodies and geometry
     - Compiled mass and inertia, mocap and gravity compensation, primitives,
       meshes, heightfields, materials, textures, sites, fluid coefficients,
       surface velocity, and adhesion.
   * - Joints
     - Free, ball, scalar, and compound joints; references, limits,
       asymmetric effort bounds, friction, armature, polynomial passive forces,
       and ``springdamper``.
   * - Tendons and equalities
     - Fixed and spatial tendons, wrapping and pulleys, polynomial passive
       forces, body/site connect and weld constraints, and joint/tendon
       polynomial equalities. Tendons and equalities can reference individual
       compound-joint axes.
   * - Actuators
     - Scalar-control transmissions, general/shortcut parameters, activation
       dynamics, explicit and inherited ranges, gear, muscle placeholders,
       DC motors, groups, and delay/history.
   * - Contacts and options
     - Contact pairs, exclusions, masks, raw constraint parameters, registered
       solver options, native flags, and actuator-length compiler settings.
       The selected solver backend must support the requested options.

Native compiler optimizations such as ``fusestatic`` and ``alignfree`` run
before conversion. Filtering runs after compilation, so retained bodies keep
their compiled inertia even when some shapes are excluded. Placement,
attachment, and joint collapsing remain available. Gravity and timestep remain
controlled by Newton and the solver. Model-wide options are shared, while
per-world options belong to the current builder world;
use ``parse_mujoco_options=False`` when attaching an asset that should not
replace the assembled model's settings.

Without a custom resolver, file-backed native imports use MuJoCo's asset-path
rules. The legacy importer resolves ``assetdir`` and ``meshdir`` separately for
each included file. For portable nested includes, omit these
per-file directory overrides and write explicit asset paths relative to the
XML file declaring each asset, such as ``file="arm_assets/mesh.obj"``. MuJoCo
checks the main model's directory first, so avoid using the same relative path
for different assets in the main file and its includes.

For example, migrate an included ``models/robot.xml`` from
``<compiler assetdir="assets"/>`` with ``file="cube.stl"`` to
``file="assets/cube.stl"``, removing the ``assetdir`` attribute. Do the same for
per-file ``meshdir`` and ``texturedir`` overrides, retaining other compiler
attributes. The files stay in their existing directories. Legacy directory
scoping remains available through ``add_mjcf()`` or the custom resolver above;
native import without a resolver requires updating those asset paths.

Both importers honor the MJCF default ``type="sphere"`` when a geom references
a mesh without specifying or inheriting a type. The mesh is used to fit the
sphere. Specify ``type="mesh"`` on the geom or in its defaults when a mesh
collider is intended.

Planar mesh colliders remain supported through ``add_mjcf()``. Native compilation
requires at least four mesh vertices and rejects planar collision hulls, even
with explicit body inertia. A planar visual mesh with at least four vertices,
``inertia="shell"``, disabled contact masks, and no explicit contact pairs can
compile. For planar colliders imported through the legacy path,
``SolverMuJoCo(use_mujoco_contacts=False)`` keeps the original Newton mesh and
uses a temporary nonplanar mesh only for solver compilation. That workaround
starts with known body inertia; native import cannot apply it unchanged while
inferring inertia from the source geometry.

Zero-mass geoms and fixed bodies are supported. A moving body with zero mass
of its own can also compile when a fixed descendant supplies positive mass and
inertia in the same import. A descendant separated by another joint does not
satisfy that compiler check. Native import compiles each source before adding
it to the builder, so an authored massless moving body cannot rely on a payload
attached by a later import. Assemble that subtree in the source before native import,
or use ``add_mjcf()`` for the incomplete model. This differs from the deferred
parameter resolution below: the source must first pass native compilation.
For a free root, a source without a root joint can instead use ``floating=True``:
Newton adds the free joint after native compilation, allowing later payload
attachment. This does not resolve massless joints authored inside the subtree.

For solid primitive geoms with explicit mass, native import uses MuJoCo's
volume cutoff of ``1e-14`` cubic meters. Volumes at or below it contribute no
mass or inertia; larger volumes retain the authored mass. The legacy importer
uses ``1e-12`` for spheres, cylinders, capsules, and ellipsoids and
no cutoff for boxes. Newton's final inertia validation still applies.

Actuator ``dampratio`` and ``inheritrange``, joint ``springdamper``, muscle
parameter placeholders, unspecified actuator length ranges, automatic tendon
spring lengths, and automatic weld relative poses retain their authored values.
Builder finalization resolves ``inheritrange`` from each world's joint or tendon
limits, fixed-tendon spring lengths from the assembled spring coordinates,
and automatic weld poses from the assembled reference configuration.
Weld resolution includes nonzero anchors and updates converted Newton loop-joint
frames. Fixed-tendon lengths use ``joint_rest_q`` plus ``dof_ref``, including
compound-joint mappings; initial joint state does not affect them.
The model receives concrete ranges, lengths, and poses, so subsequent edits survive
solver construction and rebuilding. The builder keeps its automatic declarations
for later composition and re-finalization. Fixed-tendon lengths remain fixed
across solver constant refreshes, including signed lengths equal to ``-1``;
rebuild the solver after editing these lengths on the model.
Joint-target damping and passive ``springdamper`` resolve at builder finalization
when the composed inertia is supported, as described below.
Muscle placeholders, automatic actuator length ranges, and spatial tendon spring
lengths are resolved again when
:class:`~newton.solvers.SolverMuJoCo` compiles the assembled builder. Attaching
a hand therefore contributes its inertia to the arm's damping and muscle force
scaling. Native parameters without a Newton core equivalent live in typed
``mujoco`` custom attributes, whose references survive builder copying and
composition. These are Newton storage fields for standard MJCF parameters,
not extensions to the XML format. Explicit zero values remain distinct from
automatic values, and
limit flags retain their ``auto`` / ``true`` / ``false`` state. Native compilation
validates ranges and infers activation dimensions. Inactive scalar joint ranges
are not preserved in solver models or MJCF export. Native import retains source
bounds only for actuators awaiting ``inheritrange`` resolution; finite Newton
joint bounds take precedence at finalization. Native ball-joint ranges remain
solver-specific because Newton's per-axis limits cannot express them.

Passive joint stiffness and rest positions populate ``joint_stiffness`` and
``joint_rest_q`` during import, so other Newton solvers can consume them.
File values override builder defaults, including explicit zeros. Scalar rest
positions use ``springref - ref`` when either value is authored; otherwise
``default_joint_cfg.rest_q`` applies in Newton units. Automatic ``springdamper``
resolves stiffness and damping against each world's assembled inertia during
finalization. Changing either imported coefficient before finalization overrides
that coefficient, including an explicit zero. Ball and free joints use MuJoCo's
average inverse-inertia convention; free-joint translation accounts for the
body-origin versus COM velocity convention. Resolved parameters are consumed
on the model, leaving the builder reusable for further composition.

Fixed tendons and joint equalities use the shared Newton joint-reference fields
for scalar joints. Compound joints additionally retain the selected DOF;
tendon equalities use the shared tendon-reference fields. Site equalities
retain shape references because body and joint IDs cannot represent their targets.
One-ended joint equalities remain MuJoCo equality rows rather than Newton mimic
constraints. Scaling adjusts polynomial coefficients for slide and linear-tendon
coordinates.

Joint ``solreflimit`` values, including inherited defaults, take precedence
over builder limit gains. When omitted, customized builder ``limit_ke`` /
``limit_kd`` are converted using the assembled model's effective inertia;
otherwise MuJoCo's default is retained. Later gain edits use the existing
``JOINT_DOF_FORCE_PROPERTIES`` notification. Import performs two native
parses to distinguish authored values from defaults, and one compilation.

Inherited control/activation ranges resolve during builder finalization,
including distinct per-world ranges.
Subsequent explicit range edits use ``notify_model_changed(ACTUATOR_PROPERTIES)``.
Scaling adjusts translational length ranges and servo controls while preserving
force controls and angular ranges. Other solvers consume the Newton core
representation; MuJoCo-only parameters require ``SolverMuJoCo``.

With ``ctrl_direct=False``, representable scalar position/velocity servos use
Newton joint targets. Position-only ``dampratio`` servos retain joint-target
control: builder finalization publishes damping into ``joint_target_kd``
using the assembled inertia and respecting explicit gain edits. Unsupported
core-inertia cases remain deferred to ``SolverMuJoCo``. One position servo and
one velocity servo on the same DOF share
``POSITION_VELOCITY`` targets when the position servo has no damping of its
own. Actuator ``group`` and ``inheritrange`` settings preserve joint-target
control; inherited ranges resolve from the assembled joint limits during builder
finalization. Control limits apply to both control sources. MuJoCo-Warp does not
support group disabling; the solver warns when it is requested on that backend.
Use ``use_mujoco_cpu=True`` to honor ``actuatorgroupdisable``.
Control history (``delay``, ``nsample``, and ``interp``) also preserves joint-target
control. MuJoCo records the converted scalar commands and applies their delay
and interpolation through its existing history buffers.
The imported timestep initializes history timestamps; ``step(..., dt=...)``
still determines the timestep used for simulation.
Other actuators, including those with activation dynamics and additional servos
that cannot share target gains, use ``control.mujoco.ctrl``.
Use ``ctrl_direct=True`` for native actuator control throughout.

Native import preserves actuator damping, armature, and single-input DC motors
(``voltage``, ``pos``, or ``vel``) through the shared solver attributes. DC motors
use ``control.mujoco.ctrl`` and the solver's existing construction-time parameter
and replicated-world restrictions. Linear position/velocity control ranges
scale with the import; voltage ranges do not. Actuator armature retains the
solver's CPU-only support and warning on the Warp backend.

Ball position/velocity servos without activation dynamics also use Newton joint
targets by default. Their history settings, groups, gains, and
authored control/force ranges apply to all three angular axes, replacing the authored
``gear`` transmission. Position targets use a quaternion and velocity targets
use a three-component angular velocity. A compatible position/velocity pair
therefore produces six MuJoCo actuators, matching the legacy importer.
With ``ctrl_direct=True``, each authored ball actuator remains one scalar
actuator with its original ``gear`` and a slot in ``control.mujoco.ctrl``.

For unit-gear scalar servos without activation dynamics, native import splits
position/velocity pairs with position ``kv`` or ``dampratio`` damping this way: the
first servo uses joint targets and the second uses its MJCF-order slot in
``control.mujoco.ctrl``. Set that direct-control slot for the second command;
it does not read the corresponding joint target. The legacy importer instead
shares joint-target damping between these servos.
This pair needs three independent coefficients: position stiffness, position
damping, and velocity gain. Newton's joint-target model provides only ``ke`` and
``kd``; combining the damping terms changes the response to a velocity command
and can also change per-actuator force limiting.

Other Newton solvers do not evaluate ``control.mujoco.ctrl``. For unit-gear
hinge/slide position and velocity servos, an external actuator pipeline can
preserve independent controls. Import with ``ctrl_direct=True``
to avoid also creating joint-target drives. Evaluate each servo separately with
:class:`~newton.actuators.Actuator` and :class:`~newton.actuators.DrivePD`, then
accumulate its effort into ``control.joint_f`` before stepping the solver.
A position servo's velocity target is zero; a separate velocity servo consumes
the commanded velocity. Apply authored control limits before evaluating the
drives. Keep their effort limits separate, and clear
``control.joint_f`` before each actuator evaluation pass. This explicit pipeline
requires a caller change; native import does not create it automatically.
Preserving authored ball, tendon, and other coupled transmissions requires
additional mappings; the scalar PD path does not represent them.

Native import maps compound joints to Newton's convention: slides precede
hinges, preserving declaration order within each group. Joint properties and
actuator targets follow their corresponding DOFs. Translations act along axes
fixed in the parent joint frame, so a slide authored after a hinge can move
differently than in MuJoCo. All axes use the first authored joint's anchor,
matching the legacy importer, even when subsequent joints specify different
positions. This can also change kinematics relative to MuJoCo.

Native mesh import retains triangle surfaces, including concavities, using
MuJoCo's compiled mesh coordinates and compensating geom transforms.
World-space geometry is preserved, but vertex indexing and mesh coordinate
frames can differ from USD or legacy MJCF imports. Contact reduction and SDF
voxelization can therefore produce different contact samples and trajectories
even with matching mass properties and solver settings.

Both MJCF importers retain the existing mesh SDF policy: build an SDF explicitly
on the imported :class:`~newton.Mesh` with ``mesh.build_sdf()`` before
``finalize()``. Builder resolution defaults do not generate SDFs for imported
MJCF meshes. These are Newton collision settings, separate from MuJoCo's
unsupported SDF geom type.

The importer is not a complete replacement for every MuJoCo feature.
It rejects a ball joint combined with other joints on the same body;
flexes, plugins, SDF geometry, actuator callbacks, PID dynamics,
and actuators requiring multiple control values. These need
additional Newton representations or solver support. Sensors, cameras, lights,
keyframes, skins, custom elements, and MuJoCo user data are omitted with a
warning. Custom extraction from parsed elements is available through
:ref:`native-mjcf-custom-attributes`. XML-string callbacks and nonstandard XML
attributes still require the existing importer.
Native parsing validates MJCF before conversion. Use ``childclass``
for descendant defaults and author body properties, pair endpoints, and mesh
reference poses on their concrete elements. References to generated Newton
hierarchical labels are also legacy behavior; native parsing resolves authored
MJCF names.

Joint damping and armature and geometry gap and friction use explicit MJCF values, inherited
MJCF defaults, then builder defaults. Explicit zeros override inherited values.
Authored sliding friction sets both Newton friction coefficients; omitted friction
retains the builder's static/dynamic pair.
The fallback gap is ``default_shape_cfg.gap`` when set, otherwise ``rigid_gap``.
Builder gaps are in meters and are unaffected by import ``scale``; file-authored
gaps scale with the model. ``armature_scale`` applies to both authored and
fallback armature values. The MJCF ``<freejoint>`` shortcut still ignores joint
defaults. Other mapped physical fields retain MuJoCo's built-in defaults when
not authored. Newton-only settings, such as shape friction stiffness,
retain builder defaults. Contact ``solref`` values, including
native defaults, also populate Newton contact stiffness/damping using the
existing importer's approximation; the raw values are retained for
``SolverMuJoCo``. Built-in textures supply a tinted mean-color approximation
for shapes without texture images; mesh texture images remain available.
Meshes retain non-negative material roughness and metallic values; negative
sentinels leave them unspecified. RGB texture layers take precedence over RGBA.
Opacity follows Newton's importer clamping convention, and non-finite mesh
reference poses are rejected before modifying the builder. Nonlinear joint
equalities warn when converted to generic linear mimic constraints, while
retaining their full polynomial for ``SolverMuJoCo``.
Include/compiler asset paths follow MuJoCo's
resolution rules, and filtering happens after compilation, so excluded assets
must still compile. Generated names, mesh storage, and equality ordering may
also differ.
Conversion checks account for every retained joint DOF and actuator row; explicit
filtering and root-joint replacement are excluded from these checks.

Coverage tests compare compiled parameters, kinematics, and actuator forces
with native MuJoCo, including attachment followed by fixed-joint collapsing
and replication. Joint ``springdamper`` and fixed-tendon spring lengths resolve
independently per world. Automatic spatial-tendon spring lengths and muscle
length ranges still depend on the compiled template, so batched worlds must
agree on the inputs that determine them. Runtime
notification tests cover native joint and tendon bounds, inherited actuator
ranges, and body/site/tendon equalities. Default and sentinel tests compare
explicit zero, automatic, disabled, inherited, and invalid values with MuJoCo.
These are regression checks for the supported contract, not a proof of
arbitrary MJCF equivalence. Newton's solver and geometry limitations below
still apply.

The Menagerie integration tests also compare a separately imported UR5e and
Allegro hand attached in Newton with the same sources composed in MuJoCo.
The contact test closes a finger against a stationary sphere for 0.2 seconds
in 34 worlds, checking contact pairs, force impulses, and joint motion without
copying reference parameters into Newton. This covers MuJoCo Warp contacts;
it does not establish grasp stability or parity with Newton's collision pipeline.

.. _native-mjcf-custom-attributes:

Custom attribute extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Register an experimental
:attr:`ModelBuilder.CustomAttribute.mjcf_native_value_transformer` to extract
standard MJCF fields or application metadata from ``user`` arrays:

.. code-block:: python

   builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
       name="surface_tag",
       dtype=wp.float32,
       frequency=newton.Model.AttributeFrequency.SHAPE,
       default=-1.0,
       mjcf_native_value_transformer=lambda element, context:
           element.userdata[0] if len(element.userdata) else None,
   ))
   builder.add_mjcf_native("robot.xml")

The callback receives a typed ``MjSpec`` element after native parsing resolves
defaults and includes, but before import scaling and compilation. The context
contains ``spec``, the ``CustomAttribute`` as ``attr``, and import ``scale``.
Return a dtype-compatible value, or ``None`` to leave the custom attribute at
its Newton default. Returned numeric arrays are copied. Treat native objects
as read-only and do not retain them. Values follow the parsed spec's units and
sentinels; the importer does not scale custom values automatically.

Supported frequencies are:

* ``BODY``: body elements, excluding the world body.
* ``SHAPE``: geom and site elements.
* ``JOINT``: the first retained authored joint of each Newton joint.
* ``JOINT_DOF``: each source joint, broadcast to its DOFs and reordered with them.
* ``WORLD`` and ``ONCE``: the whole spec, when ``parse_mujoco_options=True``.
* ``mujoco:actuator``, ``mujoco:tendon``, ``mujoco:pair``, and
  ``mujoco:equality_constraint``: the corresponding native elements. Register
  these frequencies first with ``SolverMuJoCo.register_custom_attributes(builder)``.

``WORLD`` values belong to the current builder world, including global world
``-1`` outside a world context. ``ONCE`` values remain model-wide.

Callbacks run on parsed elements before filtering. Values are attached only
to elements retained by compilation and conversion. Synthetic fixed/base joints
keep attribute defaults, and replaced root joints do not transfer source joint
values. Builder attachment, copying, replication, and collapsing subsequently
use the ordinary custom-attribute rules. Other frequencies raise ``ValueError``;
callback exceptions propagate before any bodies, shapes, or joints are added.

This hook is independent of ``mjcf_attribute_name`` and the legacy
``mjcf_value_transformer``. It receives typed values, including built-in defaults,
and cannot distinguish an omitted field from an explicitly authored default.
Unknown XML attributes remain native parser errors.

To compare compatibility with the existing importer, run the same MJCF,
site, and actuator tests against both entry points:

.. code-block:: bash

    uv run --extra dev --extra sim -m newton.tests.compare_mjcf_importers --output /tmp/mjcf-comparison.json

The comparison retains the existing assertions, reports failures without
marking them as expected, and exits unsuccessfully if either importer fails.
The JSON report includes tracebacks, tests blocked by fixture failures, and
import call counts inside tests and shared fixtures. Use ``--importer native``
or ``--importer legacy`` to select one implementation and ``-k`` to filter
by test name substring. Failures can indicate missing conversion behavior,
Newton-specific XML extensions, stricter native validation, or different
internal representations; the pass rate is not a measure of MJCF completeness.


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

Joint reference values (``ref``)
--------------------------------

Legacy MJCF import treats disabled hinge and slide ranges like omitted ranges,
using the builder's default joint limits. It retains inactive ranges locally
only to resolve actuator ``inheritrange`` during import. Native import uses
unlimited scalar bounds and resolves ``inheritrange`` at builder finalization.
Neither importer preserves inactive scalar ranges in ``jnt_range`` or
``save_to_mjcf`` output; their values do not affect joint-limit forces.

MuJoCo scalar joints (hinge and slide) may carry a reference value
(MJCF ``ref``, exposed as the ``mujoco.dof_ref`` custom attribute): the
authored body poses correspond to ``qpos = ref``, and MuJoCo interprets
limits, position-actuator controls, and keyframes as absolute ``qpos``
values. Newton has no reference concept — its scalar joint coordinates
are relative to the authored pose — so the two conventions are reconciled
at the solver boundary:

* In Newton, ``joint_q``, joint limits
  (``joint_limit_lower`` / ``joint_limit_upper``), and position targets
  (``Control.joint_target_q``) are relative to the authored pose. The MJCF
  importer and MuJoCo-converter USD importer shift authored joint ranges by
  ``-ref`` accordingly. ``joint_qd`` is unaffected because ``ref`` changes
  position coordinates only.
* MuJoCo-side quantities keep MuJoCo's absolute convention: the solver
  adds ``ref`` back when it writes ``qpos``, ``jnt_range``, and
  position-actuator ``ctrl``, so the compiled MuJoCo model and its
  runtime observables (``qpos``, sensors) match the authored MJCF and
  native MuJoCo exactly.
* Attributes in the ``mujoco.*`` custom-attribute namespace (for example
  authored ``actuator_ctrlrange``) are native MuJoCo data and remain in
  MuJoCo's absolute units.

Changing ``mujoco.dof_ref`` at runtime (via
:attr:`~newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES` or the broad
:attr:`~newton.ModelFlags.JOINT_DOF_PROPERTIES`) shifts exported
``qpos0``, ``jnt_range``, and position controls with the new reference.
Native MuJoCo attributes remain absolute and are not shifted. Core
:attr:`~newton.Model.joint_rest_q` values are relative to the authored pose;
the solver exports them as ``qpos_spring = joint_rest_q + ref`` for scalar
joints.

.. _mujoco-joint-spring-migration:

Passive spring migration
~~~~~~~~~~~~~~~~~~~~~~~~

MuJoCo's ``ref`` is the coordinate assigned to the authored pose;
``springref`` is the spring equilibrium in that coordinate system.
Newton stores the relative coordinate ``rest_q = springref - ref``,
after conversion to meters or radians. If ``springref`` is omitted,
MuJoCo defaults it to zero, giving ``rest_q = -ref``.

.. deprecated:: 1.8

    Use :attr:`~newton.Model.joint_stiffness` instead of
    ``model.mujoco.dof_passive_stiffness``. The old name remains a warning
    alias to the same DOF-sized array for at least one minor release.
    Reads and writes through this alias emit a ``NewtonDeprecationWarning``.
    In-place updates affect the core array directly.

Legacy ``mujoco:dof_passive_stiffness`` and ``mujoco:dof_springref``
custom-attribute inputs remain accepted by registered builders but are also
deprecated since Newton 1.8. Finalization converts explicit programmatic inputs
once into the core arrays and emits a ``NewtonDeprecationWarning``.
Ordinary MJCF and USD spring imports populate the core arrays without these
deprecation warnings. For a legacy stiffness
input without ``springref``, the default is zero in MuJoCo coordinates
(``rest_q = -ref``), unless a core rest coordinate is already configured.
An explicit ``rest_q=0.0`` also takes precedence over that default; use
``rest_q=None`` to leave it unspecified. Conflicting explicitly configured
core rest coordinates and authored legacy rest values raise an error.

Runtime ``model.mujoco.dof_springref`` is deprecated since Newton 1.8. When the
model has no joints with more coordinates than DOFs (free, ball, distance, or
rod) and no nonzero ``mujoco:dof_ref`` at finalization, the DOF and coordinate
layouts coincide and ``springref`` equals ``rest_q``, so the old name remains a
warning alias to :attr:`~newton.Model.joint_rest_q` for at least one minor
release. The alias
tracks ``rest_q``, so it no longer reports the absolute spring reference if
``dof_ref`` is edited later. For other models, reads, assignments, and
deletions raise an ``AttributeError`` with migration guidance. Replace runtime
accesses with ``joint_rest_q``, subtract ``ref`` from legacy values, and index
by :attr:`~newton.Model.joint_q_start`.

To change a spring at runtime, edit :attr:`~newton.Model.joint_stiffness`
or :attr:`~newton.Model.joint_rest_q` and notify the solver with
:attr:`~newton.ModelFlags.JOINT_DOF_PROPERTIES`. Subsequent ``dof_ref``
changes leave the Newton rest coordinate fixed and export
``qpos_spring = joint_rest_q + ref``, regardless of whether the spring was
authored through core fields, legacy builder inputs, MJCF, or USD. This replaces
the legacy rule that kept the absolute MuJoCo spring reference fixed.

BALL rest quaternions are transformed from the child joint frame to MuJoCo's
body frame. A :attr:`~newton.ModelFlags.JOINT_PROPERTIES` notification refreshes
this conversion after child-frame edits, using the last published rest pose.
Use :attr:`~newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES` to publish rest
edits separately, or :attr:`~newton.ModelFlags.JOINT_DOF_PROPERTIES` for a full
joint DOF update.


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
       correct world height. MJCF's base depth is preserved in the per-shape
       ``mujoco:hfield_base`` custom attribute [m], after import scaling, and
       multiplied by the shape's Z scale on conversion. This depth extends
       below the lowest elevation; it does not change the height samples.
       MJCF import requires exactly four positive ``size`` values, matching
       MuJoCo; missing, zero, or negative sizes are rejected.
       For native Newton fields, an absent or negative value (the default) retains the legacy
       ``1e-4`` m MuJoCo base. This attribute affects only the MuJoCo solver;
       recreate the solver after changing it.
   * - :attr:`~newton.GeoType.MESH` / :attr:`~newton.GeoType.CONVEX_MESH`
     - ``mjGEOM_MESH``
     - MuJoCo only supports **convex** collision meshes. Non-convex
       meshes are convex-hulled by MuJoCo's compiler (not by Newton),
       which changes the collision boundary. The mesh source's
       ``maxhullvert`` is forwarded.
   * - :attr:`~newton.GeoType.CONE`
     - ``mjGEOM_MESH``
     - MuJoCo has no cone primitive, so Newton tessellates the cone into a
       32-segment mesh at conversion time. Collision uses that convex
       polyhedral approximation. Changing :attr:`~newton.Model.shape_scale`
       after construction is unsupported; recreate the solver to resize the
       cone. Eager shape-property notifications validate cone scales and raise
       ``ValueError`` on changes. This host-side check is skipped during CUDA
       graph capture and replay: keep cone scales fixed, and call
       :meth:`~newton.solvers.SolverMuJoCo.notify_model_changed` with
       :attr:`~newton.ModelFlags.SHAPE_PROPERTIES` outside capture to validate
       any preceding edits.
   * - :attr:`~newton.GeoType.GAUSSIAN`
     - *unsupported*
     - Not present in the MuJoCo geom-type map.

**Sites** (shapes with the ``SITE`` flag) are converted to MuJoCo sites —
non-colliding reference frames used for sensor attachment and spatial
tendon wrap anchors. Only ``SPHERE``, ``CAPSULE``, ``CYLINDER``, and
``BOX`` are MuJoCo-native site geom types; other types silently fall
back to ``SPHERE``. MuJoCo ignores the third size component of a cylinder
site, so the MJCF importer sets ``scale[2]`` (Newton's barrel radius) of
cylinder sites to ``0.0``.

.. deprecated:: 1.8

    The ``mujoco:site_size_is_display`` shape attribute is no longer populated
    by the MJCF importer. Passing it to :meth:`~newton.ModelBuilder.add_site`
    emits a ``NewtonDeprecationWarning``; if it is ``True``, ``scale[2]`` of a
    cylinder site is set to ``0.0``. Remove the attribute, and set ``scale[2]``
    to ``0.0`` directly where it was ``True``. The attribute will be removed in
    a future release.

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

Authored raw ``solreflimit`` values are validated during solver construction
and eager :attr:`~newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES` or
:attr:`~newton.ModelFlags.JOINT_DOF_PROPERTIES` notifications on both backends. CUDA graph capture skips this host validation and leaves it
pending until the next eager solref update. Graph replay does not validate
values; call :meth:`~newton.solvers.SolverMuJoCo.notify_model_changed` with
either force-update flag outside capture after reassigning raw values to
check them. Joint-transform notifications do not validate joint-limit
parameters or consume pending validation.

On the MuJoCo Warp backend, runtime joint- and tendon-limit updates are
stored in ``solver.mjw_model.jnt_solref``, ``tendon_solref_lim``, and
``tendon_range`` per world. The corresponding arrays in the host template
``solver.mj_model`` are not updated; inspect the Warp arrays for current
values. The MuJoCo CPU backend keeps the host arrays synchronized.

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
force-space contract. For pyramidal contacts with ``condim > 1``, the factor
also includes ``mu**2 * (1 + mu**2) / (impratio * (condim - 1))`` to account
for the summed pyramid-edge normal response. Here ``mu`` is the final sliding
friction after contact overrides. Elliptic and frictionless contacts retain
the inverse-weight factor alone. The same mapping applies to positive
per-contact stiffness and damping overrides; zero per-contact damping retains
the critical-damping fallback.

This resting-contact approximation is not the full scalar contact effective mass
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

   ``use_mujoco_contacts=True`` does not apply the per-contact two-body
   factor — MuJoCo's internal ``contact_params`` averages per-geom
   ``solref``, which cannot reproduce the inverse-mass sum.
   ``SOLREF_MODE_FORCE_SPACE`` shapes fall back to the legacy
   ``convert_solref(ke, kd, 1, 1)`` approximation on that path.

For parameter interpretation, stability tradeoffs, and task-oriented guidance,
see :ref:`Tuning MuJoCo`.

.. _mujoco-contact-friction-solreffriction:

Contact friction ``solreffriction`` mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For :class:`~newton.solvers.SolverMuJoCo`, ``kf`` maps to MuJoCo's per-contact
``solreffriction`` with elliptic friction cones (``cone="elliptic"``) and Newton
contacts (``use_mujoco_contacts=False``). It targets the force-space friction slope
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

Native MJCF scalar joints retain their imported asymmetric or disabled force
limits while ``joint_effort_limit`` equals its imported value. A different value
selects a symmetric limit, including a zero-force clamp for zero. This applies
to builder edits, solver construction, and ``JOINT_DOF_FORCE_PROPERTIES``
notifications. Restoring the imported value restores the source limit. Each
replicated world retains its own choice; untouched unlimited worlds stay
unlimited when another world enables a limit.

For native MJCF scalar position-only drives with ``dampratio`` and unit gear,
:meth:`~newton.ModelBuilder.finalize` resolves :attr:`~newton.Model.joint_target_kd`
from the composed articulation's reference-pose inertia, joint armature, and
position gain. Ball drives resolve each Newton angular target axis separately.
Other Newton solvers can use the resulting damping without constructing
``SolverMuJoCo``. Import stores provisional damping derived from the source model;
changing it before finalization, including to zero, overrides automatic resolution.
Assigning the same value leaves it automatic. Ball targets track overrides per
axis. Finalization leaves the builder unchanged, so it can be copied, composed,
and finalized again.

Resolution runs after the initial body variant has been selected and shares one
batched mass-matrix calculation between passive springs and target damping.
Actuator or tendon armature excludes only its owning articulation; unassigned
ownership conservatively excludes the whole model. Those cases, unsupported
topology, and singular or numerically ill-conditioned inertia retain deferred
resolution through ``SolverMuJoCo``. Resolved articulations do not have their core
coefficients rewritten by solver construction. Deferred passive springs resolve
once from each simulated world's physical inverse inertia after its properties
are published to MuJoCo, preserving independent stiffness and damping overrides.
Kinematic constraint locking does not contribute to their effective inertia.
The CPU backend resolves only the world it simulates; unrepresented worlds retain
their placeholders. Legacy MJCF joint targets resolve
at solver construction, initializing zero damping entries and preserving nonzero
overrides. Native imports track explicit overrides, including zeros, across both
resolution paths.

Once resolved, spring and target gains are concrete coefficients: later inertia
edits, including body-variant changes, do not recompute them. Set the core gains
explicitly, or edit and finalize the builder again. With ``SolverMuJoCo``, notify
:attr:`~newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES` after model gain edits.
Direct MuJoCo actuators retain their existing runtime ``dampratio`` resolution.

The full MuJoCo general-actuator model (arbitrary gain/bias/dynamics types
and parameters, explicit transmission targets, ctrl/force/act ranges) is
only reachable through the ``mujoco`` :ref:`custom-attribute namespace <mujoco-custom-attributes>`.
The :doc:`newton.solvers.mujoco </api/newton_solvers_mujoco>` helpers
``add_actuator_dcmotor``, ``add_actuator_general``, ``add_actuator_motor``, ``add_actuator_position``,
and ``add_actuator_velocity`` author complete
actuator rows from a typed ``ActuatorTarget``. Scalar and D6 joints support
local or absolute DOF selection. Ball/free joints instead require an explicit
native MuJoCo gear vector: ball-joint torque is in the child body frame;
free-joint force is in the world frame at the body origin, and its torque is
in the child body frame. These are different from Newton's parent-frame,
COM-based DOF convention. Use :attr:`~newton.Control.joint_f` for forces in
Newton's DOF convention.
Additional actuators declared this way are appended after the joint-target
actuators — see ``SolverMuJoCo._init_actuators``.

MJCF ``<dcmotor>`` actuators follow this MuJoCo-specific path. The importer uses
MuJoCo's native parser to resolve electrical, controller, thermal, cogging, and
LuGre parameters, including inherited defaults, into the existing actuator
parameter arrays for both the CPU and MuJoCo-Warp backends. Importing these
shortcuts requires the optional ``mujoco`` package and raises ``ImportError``
with installation guidance when it is unavailable. Install ``newton[sim]`` to
include MuJoCo. Ordinary MJCF and compiled ``<general>`` actuator import do not
require MuJoCo; parsing DC-motor shortcuts does not require MuJoCo-Warp.
Commands use the MJCF actuator order in ``control.mujoco.ctrl``. The supported
``input`` signatures each consume one control value: ``voltage``, ``pos``, or
``vel``, using native MuJoCo keywords only.
MuJoCo's ``ff``, ``none``, and combined multi-input signatures are not yet
supported by Newton's one-control-per-actuator mapping. Compiled MJCF
``<general>`` and USD ``MjcActuator`` rows preserve the low-level parameter
arrays and a supported ``input`` or ``mjc:ctrlSpec`` value, respectively.

The MJCF importer and :func:`~newton.solvers.mujoco.add_actuator_dcmotor`
set :attr:`~newton.solvers.SolverMuJoCo.CtrlType.DCMOTOR`
to mark resolved DC-motor parameters as construction-time
properties; use the helper or importer to populate the associated parameters.
Runtime edits to ``actuator_gainprm``,
``actuator_biasprm``, ``actuator_dynprm``, and ``actuator_forcerange`` are ignored
for these rows to preserve the native compiler's output. Compiled MJCF and USD
DC motors instead use ``CtrlType.GENERAL`` with ``actuator_gaintype`` set to DC motor:
their low-level arrays remain runtime-editable, including after saving and
reimporting compiled MJCF. Compilation preserves the dynamics, not the shortcut's
authoring-time parameter ownership.

Actuator damping is preserved on both backends. MuJoCo-Warp 3.15 folds it into
joint and tendon damping; Newton retains that contribution when publishing
joint or tendon property updates. Actuator armature is preserved only on the
native MuJoCo CPU backend (``use_mujoco_cpu=True``); the Warp backend emits a
:class:`~newton.NewtonWarning` when it is nonzero. This limitation applies to all actuator
types, not only DC motors. MJCF ``damping`` accepts one, two, or three coefficients:
the first is stored in ``mujoco:actuator_damping`` and the remaining two in
``mujoco:actuator_dampingpoly``. Partial overrides retain inherited trailing
coefficients; otherwise omitted coefficients default to zero. Compiled USD rows can author
the corresponding ``mjc:damping`` scalar and ``mjc:dampingPoly`` pair. These
passive properties are set at solver construction, must be uniform across
replicated worlds, and are not refreshed by ``notify_model_changed(ACTUATOR_PROPERTIES)``.
The solver rejects mismatched passive properties or control signatures across worlds,
including on compiled ``GENERAL`` rows. Worlds must also have matching actuator row counts.
Warp's folded damping uses the construction-time transmission gear. For actuators
with nonzero linear or polynomial damping, ``actuator_gear`` must match the compiled
gear in every world and remain unchanged. Construction and eager property
notifications reject mismatches; captured notifications retain the compiled gear.
Recreate the solver to change these gears. Undamped Warp actuators still support
per-world and runtime gear edits. Native CPU actuator gears remain runtime-editable.
DC-motor trajectory parity with native MuJoCo is tested with ``integrator="euler"``;
MuJoCo-Warp's approximate implicit DC-motor derivatives can produce different
trajectories with ``implicitfast``.

When using ``separate_worlds=True``, corresponding high-level ``<dcmotor>``
actuators (including those created by the helper) must have identical parameters because MuJoCo-Warp replicates one
compiled template model across worlds.

This native, stateful MuJoCo actuator is distinct from
:class:`~newton.actuators.ClampingDCMotor`, which is a stateless Newton
clamping stage and does not model winding current, thermal effects, cogging, or
LuGre friction.

For programmatic authoring, use the same typed targets as the other MuJoCo helpers::

    from newton.solvers import mujoco

    actuator = mujoco.add_actuator_dcmotor(
        builder,
        target=mujoco.ActuatorTarget.joint(joint),
        motorconst=(0.05,),
        resistance=2.0,
        input="voltage",
        ctrlrange=(-12.0, 12.0),
    )

The helper registers MuJoCo custom attributes automatically and returns the
actuator row index for ``control.mujoco.ctrl``. It requires native MuJoCo to
resolve shortcut parameters, but does not compile a model or import MuJoCo-Warp.


.. _mujoco-equality-constraints:

Equality constraints
--------------------

Use :class:`~newton.solvers.SolverMuJoCo.EqType` for MuJoCo equality
constraint types.

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
   * - :attr:`~newton.solvers.SolverMuJoCo.EqType.TENDON`
     - ``mjEQ_TENDON``
     - Couples two ``mujoco:tendon`` lengths (``equality_constraint_tendon1``
       and ``equality_constraint_tendon2``). Polynomial coefficients forwarded
       in ``data[0:5]``. MJCF and programmatic authoring only; USD import is not
       supported yet.
   * - Mimic
     - ``mjEQ_JOINT``
     - Added via :meth:`~newton.ModelBuilder.set_joint_mimic`. Maps the
       offset / multiplier in :attr:`~newton.Model.joint_mimic_coeffs` to
       polynomial coefficients. Revolute, prismatic, and D6 joints are
       supported. Multi-axis D6 relationships produce one equality per
       matching pair of scalar axes.

Newton's core API does not expose equality constraints as a dedicated
builder call. Construct them with the
:doc:`newton.solvers.mujoco </api/newton_solvers_mujoco>` helpers
``add_equality_connect``, ``add_equality_weld``, ``add_equality_joint``, and
``add_equality_tendon``,
then read or update finalized fields via
``model.mujoco.equality_constraint_*``:

.. code-block:: python

   from newton.solvers import mujoco

   mujoco.add_equality_connect(builder, body1, body2, anchor=(0.0, 0.0, 0.0))

The helpers write rows in the MuJoCo
:ref:`custom-attribute namespace <mujoco-custom-attributes>`. The equivalent
lower-level form uses :meth:`~newton.ModelBuilder.add_custom_values` with the
``mujoco:equality_constraint_*`` keys; fields that do not apply to connect
constraints retain their registered defaults.

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
       }
   )

``mujoco:equality_constraint_world`` has ``references="world"``, so omitting
it assigns the row to ``builder.current_world``. Pass an explicit world index,
including ``-1`` for the global world, when that is the intended owner instead.

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

MuJoCo's fixed and spatial tendons use the
:ref:`custom-attribute namespace <mujoco-custom-attributes>`: populated
on import from MJCF/USD and parsed into MuJoCo's tendon structures by
``SolverMuJoCo._init_tendons``. To author tendons programmatically, use
``add_tendon_fixed`` and ``add_tendon_spatial`` from
:doc:`newton.solvers.mujoco </api/newton_solvers_mujoco>`. Spatial tendons support ``site``,
``geom``, and ``pulley`` wrap elements; any other wrap type and any
degenerate tendon definition produces a warning and is skipped rather
than raising.

These provider tendons are separate from Newton's experimental
:doc:`massless routed tendons </tendons>` used by XPBD and VBD. There is no
automatic conversion between native ``Model.tendon_*`` routes and
``model.mujoco.tendon_*`` data. USD ``MjcTendon`` import and tendon-actuator
target resolution use the provider path; see
:doc:`/concepts/usd_parsing` for supported path attributes.

For force-based limit gains, set ``model.mujoco.tendon_limit_ke`` (stiffness
[N/m]) and ``tendon_limit_kd`` (damping [N s/m]), select force-space authoring
with ``tendon_solref_limit_mode = 0``, then notify
:attr:`~newton.ModelFlags.TENDON_PROPERTIES`. For angular tendon coordinates,
the corresponding units are [N m/rad] and [N m s/rad]. The solver converts
these gains to ``solreflimit`` using tendon inverse inertia and impedance,
and refreshes the conversion after inertia changes, as it does for joint limits.
MuJoCo impedance and timestep clamping still affect transient response.

``tendon_solref_limit_mode`` follows the joint-limit authoring convention:
0 selects Newton force gains, 1 preserves authored MuJoCo parameters, and 2
uses MuJoCo's implicit default. Tendons default to mode 1 and report their
equivalent force gains after solver initialization. In force-space mode,
zero stiffness disables the limit without changing the authored ``tendon_range``.
Limit damping is independent of passive ``tendon_damping``. Tendon limits must
already be enabled in the imported model.

.. code-block:: python

    model.mujoco.tendon_limit_ke.fill_(100.0)
    model.mujoco.tendon_limit_kd.fill_(20.0)
    model.mujoco.tendon_solref_limit_mode.zero_()  # Newton force-space mode
    solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)


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

With :class:`~newton.solvers.SolverMuJoCo` using MuJoCo Warp,
``integrator="rk4"`` and ``use_mujoco_contacts=False`` reuse contact positions,
distances, and normals at RK4's intermediate stages. This can reduce contact
accuracy as bodies move, and the solver warns at construction. Use
``integrator="implicitfast"`` or ``use_mujoco_contacts=True`` to avoid this
limitation.

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

**Margin zeroing.** Earlier ``mujoco_warp`` releases rejected non-zero geom
margins on box-box pairs (its default NATIVECCD path) and on any box/mesh pair
when ``enable_multiccd=True``. MuJoCo Warp 3.17 accepts them, but the MuJoCo Warp
backend of :class:`~newton.solvers.SolverMuJoCo` still zeroes ``geom_margin`` and
``pair_margin`` model-wide at compile time whenever a box geom exists and NATIVECCD
or multi-CCD is enabled, or whenever ``enable_multiccd=True`` is combined with mesh
geoms; geoms and pairs with non-zero authored margins emit a warning when
``use_mujoco_contacts=True``. The Newton model's :attr:`~newton.Model.shape_margin` array
is left untouched, and when ``use_mujoco_contacts=False`` the authored
margins are restored at runtime through ``update_geom_properties_kernel``.
The CPU backend (``use_mujoco_cpu=True``) keeps the authored geom and pair
margins, so its contacts can differ from the MuJoCo Warp backend's for such models.


Contact pairs
-------------

Newton's core API does not expose explicit MuJoCo-style ``<pair>``
contact overrides. They are implemented through the MuJoCo
:ref:`custom-attribute namespace <mujoco-custom-attributes>` and
parsed into MuJoCo's geom-pair contact structures by
``SolverMuJoCo._init_pairs``. To author a pair programmatically, use
``add_contact_pair`` from :doc:`newton.solvers.mujoco </api/newton_solvers_mujoco>`.


Multi-world support
-------------------

Constructing :class:`~newton.solvers.SolverMuJoCo` with
``separate_worlds=True`` (the default for GPU mode with multiple
worlds) builds a shared MuJoCo topology across worlds via ``mujoco_warp``.
Bodies, joints, equality constraints, mimic relationships, primitive geoms,
and sites must have matching layouts. With ``use_mujoco_cpu=False``, mesh,
heightfield, and cone assets, scales, and convex-hull counts may differ across worlds. Collision
flags may also differ when Newton supplies contacts. Native MuJoCo contacts
require matching collision flags, groups, and exclusions for corresponding geom
slots. With native contacts,
per-world mesh scales are compiled into the MuJoCo assets, so changing their
:attr:`~newton.Model.shape_scale` after construction raises ``ValueError``;
recreate the solver to resize them. The solver validates
compatibility at construction and raises ``ValueError`` on a mismatch.

Bodies, joints, equality constraints, and mimic relationships cannot have
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
   frequency can be controlled via the deprecated
   ``update_data_interval`` kwarg; it will be removed and the joint state
   will then be synchronized on every step.
2. **Integrate.** ``mujoco_warp`` steps the MuJoCo model forward by ``dt``.
3. **Pull MuJoCo → Newton.** ``SolverMuJoCo._update_newton_state``
   populates the output ``State`` from the integrated MuJoCo data.
   Kinematic roots pass through unchanged from ``state_in`` (see
   `Kinematic links and fixed roots`_).

Request ``CONTACT_F`` with ``solver.observables(..., contacts=contacts)`` and pass
the container to ``solver.step(..., observables=observables)``. Forces are written
to ``observables.contact_f``; with ``use_mujoco_contacts=True``, the step also
fills the supplied ``contacts`` with contact geometry. See :ref:`solver_observables`.

Separated speculative contacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``use_mujoco_contacts=False``, Newton's collision pipeline can generate
separated candidates using velocity expansion. By default, MuJoCo activates
constraints only inside the authored collision margin. Generating a candidate
alone therefore does not prevent an object from crossing a thin collider in
one dynamics step.

Set ``use_speculative_contacts=True`` to add frictionless normal constraints
for separated candidates. Their target bounds closing velocity by the current
clearance divided by the physics timestep. Collision margins and the material
response of penetrating contacts retain their authored values. Friction begins
when the surfaces enter the authored margin.

.. experimental::

   The ``SolverMuJoCo(use_speculative_contacts=True)`` mode may change without
   prior notice. It requires Newton contacts on the MuJoCo Warp backend and
   Euler or implicitfast integration. It does not provide continuous collision
   detection or guarantee zero penetration: candidate coverage, rotation,
   contact compliance, forces, and timestep still matter.

This is a runtime solver option, like the choice of contact backend. It does
not add an asset-authoring attribute or change the USD collision schema.

Pass a velocity-expansion horizon to the collision pipeline before each step::

    pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.05)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverMuJoCo(
        model, use_mujoco_contacts=False, use_speculative_contacts=True
    )
    pipeline.collide(state_in, contacts, dt=dt)
    solver.step(state_in, state_out, control, contacts, dt)

For scheduled collision refreshes, the expansion horizon must cover the
interval until the next refresh. The solver's velocity bound uses each
individual dynamics timestep. Speculative constraints can consume additional
constraint rows, so check contact and constraint capacities for the scene.

Push, pull, and contact export are implemented by
``SolverMuJoCo._apply_mjc_control``, ``SolverMuJoCo._update_newton_state``,
and ``SolverMuJoCo._populate_contact_observables``. The MuJoCo Warp path uses kernels
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

Per-world options use the ``WORLD`` layout, shape ``(world_count + 1,)``,
whose final entry belongs to global world ``-1`` (see :ref:`Per-world attributes`).
Separate MuJoCo worlds read the entry of their Newton world. A single MuJoCo world
reads world ``0`` when that world owns bodies and the global entry otherwise, as in
models built without :meth:`~newton.ModelBuilder.begin_world`, where importers store
options in the global entry. :attr:`~newton.Model.gravity` follows the same rule;
a MuJoCo world has one gravity, so dynamic global bodies that share it with world ``0``
bodies must have the same gravity, or construction raises :class:`ValueError`.
Later changes to either gravity entry or to a body's kinematic flag are not checked;
such a MuJoCo world follows world ``0``.

These values are read once during :class:`~newton.solvers.SolverMuJoCo`
construction. Editing ``model.mujoco.<option>`` afterwards has no
effect — the resolved value is already baked into the underlying
MuJoCo model.

See MuJoCo's `solver documentation
<https://mujoco.readthedocs.io/en/stable/computation/index.html>`_ and
`\<option\> XML reference
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#option>`_ for
what each parameter does and when to tune it.


Imported option flags
---------------------

MJCF ``<option><flag>`` settings are stored as boolean ``ONCE`` custom
attributes, ``model.mujoco.flag_<name>``, with ``True`` meaning enabled.
The same attributes can be authored on a USD PhysicsScene through
``MjcSceneAPI`` names ``mjc:flag:<name>``. Supported names are
``constraint``, ``equality``, ``frictionloss``, ``limit``, ``contact``,
``spring``, ``damper``, ``clampctrl``, ``warmstart``,
``actuation``, ``refsafe``, ``sensor``, ``eulerdamp``, ``nativeccd``,
``island``, ``multiccd``, ``energy``, and ``invdiscrete``.

``enable_multiccd``, ``disable_contacts``, and ``disable_sensors`` default
to ``None``: explicit constructor arguments take precedence over model
custom attributes, then Newton defaults. By default, ``multiccd``,
``energy``, and ``invdiscrete`` are disabled; the other listed features
are enabled. ``sleep`` uses the existing ``model.mujoco.enable_sleeping``
attribute described below and requires ``island`` to remain enabled.
Unknown flags and the Warp-unsupported
``override``, ``fwdinv``, ``diagexact``, ``midphase``, and ``autoreset``
flags are warned about and ignored during MJCF import. ``filterparent``
is also warned about and ignored because imported parent-child collision
filters are not controlled by this flag.

``gravity="disable"`` sets Newton's builder gravity to zero, affecting
all solvers and content already added to the shared builder. Inside
``begin_world()`` / ``end_world()``, only that world's gravity is changed;
outside a world, the global/default gravity is changed. A later import
with ``gravity="enable"`` does not restore the previous gravity vector.

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
itself.
See :doc:`/concepts/custom_attributes` for how Newton's
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

MuJoCo joint ``damping`` maps to :attr:`~newton.Model.joint_damping`;
``stiffness`` maps to :attr:`~newton.Model.joint_stiffness`, and scalar
``springref - ref`` maps to :attr:`~newton.Model.joint_rest_q`.
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
  joints with the same Newton joint type and DOFs, held by a weld
  constraint; see `Kinematic welds`_.
- **Roots attached to world with a fixed joint** become MuJoCo mocap
  bodies (whether kinematic or not). MuJoCo has no joint coordinates
  for a fixed root, so Newton drives the pose through
  ``mjData.mocap_pos`` and ``mjData.mocap_quat`` instead. A D6 root
  with all axes locked (e.g. imported from a generic USD
  ``PhysicsJoint``) is treated the same way.
- **Massless kinematic roots** also become mocap bodies and cannot be
  switched to dynamic. This applies when the root and the bodies attached
  to it by fixed joints together lack mass or inertia. Newton poses the
  mocap body from :attr:`~newton.State.joint_q`, but MuJoCo treats its
  velocity as zero, so contacts and dynamic descendants respond to its pose
  but not to its prescribed velocity. Give the root mass and inertia when
  that coupling matters, for example for a conveyor that carries objects
  through friction.
- **World-attached shapes that are not part of an articulation**
  remain ordinary static MuJoCo geometry rather than mocap bodies.

If you edit :attr:`~newton.Model.joint_X_p` or :attr:`~newton.Model.joint_X_c`
for a fixed-root articulation after constructing the solver, call
:meth:`~newton.solvers.SolverBase.notify_model_changed` with the
:attr:`~newton.ModelFlags.JOINT_PROPERTIES` flag to
synchronize the updated fixed-root poses into MuJoCo.

Kinematic welds
~~~~~~~~~~~~~~~

MuJoCo has no prescribed degrees of freedom. Its documented way to drive a
body that interacts physically is to weld it to a moving target (see
`MoCap bodies <https://mujoco.readthedocs.io/en/stable/modeling.html#mocap-bodies>`__).
:class:`~newton.solvers.SolverMuJoCo` uses a variant of this: each kinematic
root keeps its joint and is held by a weld constraint to the world without
stiffness or damping. Before every step the weld is re-centered on the root's
current pose, because MuJoCo's weld adds acceleration away from its reference
that grows with the root's velocity. The weld locks the root's acceleration,
so through each step the root keeps the pose and velocity pushed from
:attr:`~newton.State.joint_q` and :attr:`~newton.State.joint_qd`. Constraints
and contacts treat the root as immovable, while the mass matrix stays physical
and the root keeps its authored :attr:`~newton.Model.joint_armature`.

A weld keeps the root's angular velocity, and the velocity of its joint origin,
constant through each step. For revolute, prismatic, ball, and free roots, and
for D6 roots with at most one angular axis, this is the same as keeping the
pushed joint velocities. A D6 root with two or three angular axes instead keeps
its body angular velocity, so its joint velocities drift within a step, by
about ω²·Δt for angular speed ω. Between sparse pushes (the deprecated
``update_data_interval > 1``), such a root coasts at constant body angular
velocity rather than at constant joint velocities. Newton still returns the
prescribed state, and each push resets the drift, but contacts and dynamic
descendants see the motion MuJoCo integrates.

The weld is resolved by MuJoCo's constraint solver, so dynamic bodies attached
to a kinematic root by joints follow fixed-base dynamics only as closely as
the solver converges. The default Newton solver reproduces them closely. The
CG solver can stop before it resolves the weld, so dynamic descendants of a
kinematic root can drift noticeably from fixed-base motion; use
``solver="newton"`` for kinematic roots that carry articulated dynamic bodies.

MuJoCo's ``equality`` and ``constraint`` disable flags, set through MJCF
``<flag equality="disable"/>`` or ``model.mujoco.flag_equality`` and
``model.mujoco.flag_constraint``, also turn these welds off. The solver
therefore raises ``ValueError`` for a kinematic root held by a weld while
either flag is disabled.

.. _mujoco-kinematic-switching:

Switching kinematic roots at runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MuJoCo cannot add constraints to a compiled model, so by default the solver
gives every movable root an inactive weld, and :attr:`~newton.Model.body_flags` can switch
roots between kinematic and dynamic per world, followed by
``notify_model_changed(ModelFlags.BODY_PROPERTIES)``. Inactive welds add no
constraint rows; roots made kinematic later share ``njmax`` with contacts.
Body-flag checks run on the host and are skipped for notifications replayed
from a CUDA graph.

With ``kinematic_switchable_roots=False``, only roots that are kinematic in
some world at solver creation get a weld, so the compiled MuJoCo model has no
unused welds. To start such a root dynamic, mark it kinematic before creating
the solver and switch it right after:

.. testcode:: mujoco-kinematic-switching

   from newton.solvers import SolverMuJoCo

   builder = newton.ModelBuilder()
   crate = builder.add_body(is_kinematic=True)
   builder.add_shape_box(crate, hx=0.1, hy=0.1, hz=0.1)
   model = builder.finalize()
   solver = SolverMuJoCo(model, kinematic_switchable_roots=False)

   model.body_flags.fill_(int(newton.BodyFlags.DYNAMIC))
   solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
   # The crate can now be switched back to kinematic at any time.


Updating joint force properties
-------------------------------

Use :attr:`~newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES` with
:meth:`~newton.solvers.SolverMuJoCo.notify_model_changed` to publish joint
friction, damping, target gains/modes, effort limits, :attr:`~newton.Model.joint_stiffness`, and
limit coefficients/bounds. This includes the MuJoCo ``solreffriction``,
``solimpfriction``, ``limit_margin``, ``solimplimit``,
and ``solreflimit`` custom attributes. Optional attributes that are absent
retain their solver values. General MuJoCo actuator properties still use
:attr:`~newton.ModelFlags.ACTUATOR_PROPERTIES`.

The force flag skips mass-constant recomputation, reference-pose updates,
tendon-limit refreshes, and contact-cache invalidation. Joint-limit coefficients
use the cached inverse weights. Compiled actuator length ranges are preserved,
and pending tendon-limit edits are left unpublished.
When sleeping is enabled, updated parameters wake all worlds. The MuJoCo Warp
path supports CUDA graph capture; the MuJoCo CPU backend copies values to its
host model and cannot be captured.

Use :attr:`~newton.ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES` for
:attr:`~newton.Model.joint_armature` changes. Reference-pose changes, including
MuJoCo ``dof_ref`` and :attr:`~newton.Model.joint_rest_q`, use
:attr:`~newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES`. These paths recompute constants;
reference updates also shift limit ranges to the new reference. Both flags
are intended for reset-time or domain-randomization changes. Constant
recomputation is substantially more expensive than force updates, so avoid
using either flag every simulation step.
:attr:`~newton.ModelFlags.JOINT_PROPERTIES` only publishes joint transforms and
axes, without recomputing constants or applying pending reference edits. Combine
flags with ``|`` when several categories change. The existing
:attr:`~newton.ModelFlags.JOINT_DOF_PROPERTIES` flag retains its integer value
and full-update behavior, including force, armature, and reference properties.

**Tendon-limit dependency:** Notifications that recompute constants also refresh
tendon limits using the current ``model.mujoco`` limit modes, gains, raw
``solref`` values, and ranges. Those pending edits are therefore published even
without :attr:`~newton.ModelFlags.TENDON_PROPERTIES`. This includes joint
inertial/reference updates and the broad joint-DOF, body, body-inertial, and
actuator notifications. Publish tendon edits explicitly with
:attr:`~newton.ModelFlags.TENDON_PROPERTIES` before these updates rather than
relying on them to preserve pending tendon-limit changes. Force-only joint
notifications do not recompute constants and leave tendon limits unchanged.

For example, actuator models with load-dependent friction can compute effective
budgets and publish them in the same graph as the simulation step. Given device
arrays ``friction_budget`` and ``damping_budget`` with one entry per Newton DOF:

.. code-block:: python

    with wp.ScopedCapture(device=model.device) as capture:
        wp.copy(model.joint_friction, friction_budget)
        wp.copy(model.joint_damping, damping_budget)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        solver.step(state_in, state_out, control, contacts, dt)

The budget computation can precede these copies in the same graph. For an
actuator controlling only some DOFs, scatter into those Newton model entries
before notifying the solver; other entries retain their authored values. No
MuJoCo DOF mapping or backend array access is needed.

The model arrays hold the **effective total** friction and damping. Publication
replaces solver values rather than adding an actuator contribution to authored
passive damping. Retain any authored baseline separately when combining
contributions. Apply reset-time inertia and configuration changes first, then
compute and publish the next step's budgets. Full notifications also copy the
current budgets. This API synchronizes existing runtime model arrays; it does
not add actuator external-load feedback or authored USD properties.


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
- ``SolverMuJoCo._populate_contact_observables`` — requested contact
  geometry and force export during ``step()``.
- :meth:`~newton.solvers.SolverMuJoCo.update_contacts` — explicit
  geometry-only contact export.
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
