.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _usd_parsing:

USD Parsing and Schema Resolver System
========================================

Newton provides USD (Universal Scene Description) ingestion and schema resolver pipelines that enable integration of physics assets authored for different simulation solvers.

Understanding USD and UsdPhysics
--------------------------------

USD (Universal Scene Description) is Pixar's open-source framework for interchange of 3D computer graphics data. It provides an ecosystem for describing 3D scenes with hierarchical composition, animation, and metadata.
UsdPhysics is the standard USD schema for physics simulation, defining for instance:

* Rigid bodies (``UsdPhysics.RigidBodyAPI``)
* Collision shapes (``UsdPhysics.CollisionAPI``)
* Joints and constraints (``UsdPhysics.Joint``)
* Materials and contact properties (``UsdPhysics.MaterialAPI``)
* Scene-level physics settings (``UsdPhysics.Scene``)

However, UsdPhysics provides only a basic foundation. Different physics solvers like PhysX and MuJoCo often require additional attributes not covered by these standard schemas.
PhysX and MuJoCo have their own schemas for describing physics assets. While some of these attributes are *conceptually* common between many solvers, many are solver-specific.
Even among the common attributes, the names and semantics may differ and they are only conceptually similar. Therefore, some transformation is needed to make these attributes usable by Newton.
Newton's schema resolver system automatically handles these differences, allowing assets authored for any solver to work with Newton's simulation. See :ref:`schema_resolvers` for more details.


Newton's USD Import System
--------------------------

Newton's :meth:`newton.ModelBuilder.add_usd` method provides a USD import pipeline that:

* Parses standard UsdPhysics schema for basic rigid body simulation setup
* Resolves common solver attributes that are conceptually similar between different solvers through configurable schema resolvers
* Handles priority-based attribute resolution when multiple solvers define conflicting values for conceptually similar properties
* Collects solver-specific attributes preserving solver-native attributes for potential use in the solver
* Supports parsing of custom Newton model/state/control attributes for specialized simulation requirements

Deformable Bodies
-----------------

.. experimental::

   Deformable-body import targets the **proposed** AOUSD UsdPhysics Deformables schema,
   which is not yet registered in any USD distribution. This is pre-release coverage:
   the schema, its attribute names, and the import behavior may change without notice as
   the proposal evolves.

   It is an initial implementation of a **subset** of the proposal -- see the supported
   subset and limitations below -- and is not fully proposal-compliant. It also does not
   import native OmniPhysics/PhysX deformable assets (see the vendor-namespace note below).

:meth:`newton.ModelBuilder.add_usd` imports deformable bodies authored with the `AOUSD UsdPhysics
Deformables proposal
<https://github.com/aousd/OpenUSD-proposals/blob/5d89c0ed46a26de92f4d3fefef3bfad6500c07ce/proposals/physics_deformables/wp_deformable_physics.md>`_,
across three families:

* **Curve / cable** -- a linear ``UsdGeom.BasisCurves`` with ``PhysicsCurvesDeformableSimAPI``
  becomes a rod: a chain of capsule bodies joined by cable joints, usable by any solver that
  supports cable joints. A ``wrap=periodic`` curve also gets a body for the closing segment.
* **Surface / cloth** -- a ``UsdGeom.Mesh`` with ``PhysicsSurfaceDeformableSimAPI`` becomes
  cloth: particles with FEM triangles and bending edges. Polygonal faces (such as quads) are
  fan-triangulated on import.
* **Volume** -- a ``UsdGeom.TetMesh`` with ``PhysicsVolumeDeformableSimAPI`` becomes a soft
  body. Under a ``PhysicsDeformableBodyAPI`` ancestor exactly one simulation TetMesh is
  selected; other TetMeshes in that hierarchy are graphics/collision geometry and are not
  simulated. A bare ``UsdGeom.TetMesh`` without these APIs keeps the older material-density
  import.

Material attributes are read from the standard ``physics:`` namespace, as the proposal defines.
Vendor namespaces (``omniphysics:``, ``physxDeformableBody:``) are an opt-in fallback for the
same proposal-shaped attributes on bound materials: a schema resolver (e.g.
``SchemaResolverPhysx``) declares them, and they are consulted only for deformable attributes.
That is the full extent of the vendor support -- the resolver does not translate
OmniPhysics/PhysX applied schemas (such as ``PhysxDeformableSurfaceAPI``), renamed attributes,
concrete attachment prims, pose purposes, or hierarchy conventions. A native Omni/PhysX
deformable asset that does not apply the AOUSD simulation APIs is not recognized as a
deformable and imports as ordinary (static) geometry. One temporary exception: a TetMesh
material that authors its values only under the vendor namespaces is still read without a
resolver, with a ``DeprecationWarning``, so existing assets keep their stiffness and density
during the deprecation window.

Supported subset
~~~~~~~~~~~~~~~~

The first release deliberately supports a narrow, predictable set of inputs:

* Valid, enabled, **dynamic** cable, cloth, and volume simulation prims that use the AOUSD
  deformable APIs. A bound simulation material supplies thickness, stiffness, and density;
  unauthored material properties fall back to documented builder defaults.
* The points and topology **as currently authored**. Newton builds the deformable at that pose;
  a standalone cable's ``restShapePoints`` may affect stiffness normalization but never
  establishes an initial strain state.
* Point attachments only where the authored constraint can be represented without moving any
  geometry: hard cable-to-xform attachments, and hard, coincident cable-to-cable junctions.
* ``PhysicsElementCollisionFilter`` prims filter collisions between the paired element groups
  of their two sources (imported cables, rigid bodies, or collider prims; a count of ``0`` or
  an empty counts array selects all elements). Cloth and volume element sources warn and are
  skipped, as are filters targeting a dedicated deformable collider (that collider is not
  represented in the model).
* Standard ``physics:filteredPairs`` relationships are honored for shape-backed participants:
  a rigid collider or body, a cable (every segment shape), or a deformable body prim owning a
  cable. The relationship can be authored on either endpoint. Pairs naming a cloth or volume
  deformable warn and are not lowered (they are particles, not shapes), as do pairs whose
  target is missing or produced no collision participant.
* ``UsdPhysicsCollisionGroup`` membership is **not** applied to deformables; deformable
  collision filtering is per-pair only (the standard ``physics:filteredPairs`` and
  ``PhysicsElementCollisionFilter`` support above).
* Every imported deformable can be found by prim path in the import results (see below).

Anything outside this set warns and is skipped, or is recorded as unsupported in the returned
attributes. It never silently becomes a different physical model. In particular: disabled
(``physics:bodyEnabled = false``) and kinematic (``physics:kinematicEnabled = true``)
deformables are skipped, malformed topology or curves are skipped, and compliant
(finite-stiffness) attachments and non-coincident cable junctions are kept as data but not
imported. Dynamic and static friction belong to collision geometry and are not mapped onto
the deformable collision approximation yet.

Limitations
~~~~~~~~~~~

Known gaps of the experimental importer, tracked as follow-ups:

* **Rest state** -- authored rest geometry is not imported as the deformable's simulated rest
  configuration. Cloth and volume rest attributes are ignored with a warning, and welded cable
  graphs drop ``restShapePoints``. For a standalone cable, a valid ``restShapePoints`` supplies
  only the segment lengths used to convert the material moduli into joint stiffness; the rod
  itself is still built relaxed at the current ``points`` pose, and mass distribution also uses
  the current geometry. A body saved in a deformed pose therefore resumes relaxed at that pose
  instead of springing back.
* **Springy attachments** -- attachments with a finite stiffness are not simulated. They are
  preserved in ``path_attachment_attrs`` with their authored stiffness and damping (silently
  hardening them would change the authored physics); only hard attachments (unauthored or
  infinite stiffness; damping does not affect hardness) become joints.
* **Body state** -- ``startsAsleep`` and ``simulationOwner`` are not read. Kinematic
  deformables are skipped rather than simulated.
* **Per-element materials** -- ``GeomSubset`` physics material bindings warn and are
  ignored; one material applies to the whole simulation prim.
* **Collision participation** -- follows the rigid semantics: a deformable collides when
  its simulation geometry (or, approximated with a warning, another prim in its deformable
  body hierarchy) carries an enabled ``PhysicsCollisionAPI``; ``physics:collisionEnabled``
  falls back to true. Cables without an enabled collider import as non-colliding rods
  (dynamics only, per the proposal). Cloth and volume deformables cannot disable particle
  collision in Newton yet: they warn and import colliding. A welded cable graph shares one
  shape configuration, so any collision-enabled member curve makes the whole graph collide
  (mixed authoring warns).
* **Collision and graphics geometry** -- separate collision or render geometry under a
  deformable body is not simulated or driven (embedding is not implemented): untagged
  PointBased graphics geometry warns and is skipped (a static import would leave a frozen
  copy behind), and a dedicated point-based collider (every one warns) only toggles the
  simulation geometry's collision as described above and never becomes a separate rigid
  shape. Deformable-owned geometry is owned exclusively by the deformable importer: when a
  deformable is skipped as kinematic or malformed it imports as nothing, with a warning,
  rather than falling back to a rigid representation. A disabled
  (``physics:bodyEnabled = false``) deformable follows the rigid-body precedent instead:
  it is not simulated, but its collision geometry persists as static colliders (TetMesh
  and BasisCurves simulation geometry has no static representation and stays out).
* **Cable frames and stiffness** -- if per-point normals are missing, segment orientation is
  synthesized. One stiffness value, computed from the mean segment length, applies to a whole
  curve or graph, so curves with very uneven segment lengths lose per-segment accuracy. When a
  standalone cable authors valid ``restShapePoints``, its segment lengths are used for this
  conversion; the current ``points`` still define the rod's constructed and relaxed pose.
* **Thickness fallbacks** -- without an authored thickness the importer assumes a default
  (2 mm cloth shell thickness, 2.5 mm cable radius) for the mass, stiffness, and
  collision-radius conversions, and warns with the assumed value. Author
  ``physics:thickness`` on the material to override.
* **Single-segment curves** -- an open two-point curve (one segment) is warned and skipped;
  the rod representation needs at least two segments. A periodic two-point curve closes into
  two segments and imports.

**Mass distribution** follows the proposal's precedence order. Per-point ``physics:masses`` on
the simulation geometry win. Next comes the ``PhysicsDeformableBodyAPI`` ``mass`` total, then
the body or material density. Density- and total-derived masses are spread over the **current**
geometry (segment lengths, triangle areas, tet volumes). The proposal spreads them over the rest
shape instead, but rest state is not imported yet (see the limitations above), so a deformed
saved pose shifts mass with it.

Every imported deformable can be looked up by its prim path in the mapping
:meth:`~newton.ModelBuilder.add_usd` returns when called with ``return_deformable_results=True``:
``path_cable_map`` holds each cable's body and joint indices, and ``path_cloth_map`` /
``path_soft_map`` hold each cloth's and soft body's ``[start, end)`` particle and topology
ranges. Without the flag the return shape carries no deformable entries.

A ``PhysicsAttachment`` prim ties two sites together. Each side has a target relationship
(``src0``, ``src1``) pointing at the prim it attaches to, a site ``type`` (``type0``, ``type1``)
naming what on that prim is attached -- ``point``, ``segment``, ``face``, ``tetrahedron``, or
``xform`` -- and ``indices``/``coords`` locating the site on the target (for example, which cable
segment and the ``(u, s, t)`` position along it).

The importer supports attachments on cables. When ``src0`` is an imported cable, ``type0`` is
``point`` or ``segment``, and ``type1`` is ``xform``, each attachment site becomes a ball joint.
The joint connects the cable segment body (for an interior cable point, one flanking segment
body -- each site is a single point-point constraint) to the target: an xform, a rigid body
(kinematic bodies included), or the world frame. The created joints are returned in
``path_attachment_map``. Every parsed attachment, including unsupported ones, is described in
``path_attachment_attrs``. A finite attachment stiffness cannot be represented yet, so
compliant attachments warn and are preserved as metadata instead of becoming joints. Attachments on cloth or volume sites warn and are kept in
``path_attachment_attrs`` until Newton has a constraint for them.

A ``point``->``point`` attachment between two imported cables can be a weld. Welding happens
only when the attachment is **hard** (no authored stiffness, or infinite; authored damping
does not affect hardness) **and** the two attached points sit at the same position. Such a junction is shared structure,
not a runtime constraint: the two points become one node, and every curve connected through such
junctions is built as one rod graph with a single :meth:`~newton.ModelBuilder.add_rod_graph`
call (one capsule body per segment, junction nodes shared). Welded junction attachments are
absorbed into the graph, so they appear in neither ``path_attachment_map`` nor
``path_attachment_attrs``. A springy or non-coincident cable-to-cable attachment is **not**
welded. It warns and is kept as unsupported in ``path_attachment_attrs``, so the authored
geometry and the constraint intent are never silently rewritten. Cable-to-xform attachments on
the same curves still import as described above.

Each imported cable is wrapped into its own articulation, labelled ``"<path>_articulation"``
(a multi-curve prim labels per curve: ``"<path>_curveN_articulation"``).
The model is therefore ready for :meth:`~newton.ModelBuilder.finalize` with no extra steps.
A welded rod graph gets one articulation per connected component; each of its curves keeps its
own body range but shares that articulation. Attachment joints that tie a cable to other bodies
close a loop, so they stay outside the articulation.

.. code-block:: python

    result = builder.add_usd("cables.usda", return_deformable_results=True)
    # Look up an imported cable by prim path:
    cable_bodies, cable_joints = result["path_cable_map"]["/World/Cable"]
    model = builder.finalize()  # cables are already wrapped and finalize-ready

The :meth:`~newton.ModelBuilder.add_usd` return dict carries ``path_cable_attrs``,
``path_cloth_attrs`` and ``path_soft_attrs``, mapping each prim path to its attributes exactly
as authored, independent of any solver. The cable and cloth entries expose the parsed
``material`` moduli and the ``resolved_density``. The volume entry exposes the
``resolved_density`` (a volume material's ``youngsModulus`` / ``poissonsRatio`` are applied to
the built soft body and not repeated there). The cable and cloth ``material`` keeps moduli the
imported rod and membrane cannot express -- for example cable ``shearStiffness`` /
``twistStiffness`` -- so a solver with a richer cable or surface model can rebuild the
deformable from the import without re-parsing the stage. A cable entry carries a
``graph_component`` identifier only when the curve was welded into a rod graph; curves of one
graph share it, and independent or fallback cables have no such key.

.. note::

   Solver tuning that is not part of the AOUSD schema (e.g. damping) is not imported; supply it
   on the builder or model after import.

Material Color Spaces
---------------------

Newton stores imported mesh colors and base-color textures in display/sRGB
space, matching the rest of the public model API. During USD import, scalar
``UsdPreviewSurface`` ``diffuseColor`` and ``baseColor`` values are resolved
from the authored input and normalized to that Newton convention.

If a scalar color has no USD color-space metadata, Newton follows the USD
Preview Surface convention and treats it as linear Rec.709 before converting it
to display/sRGB. If the attribute has authored ``colorSpace`` metadata or
inherits a color space through ``UsdColorSpaceAPI``, Newton uses
``Usd.ColorSpaceAPI.ComputeColorSpaceName`` to determine the effective color
space. Linear/raw color spaces are converted to display/sRGB; display/sRGB
colors such as ``srgb_rec709_scene`` are kept as authored.

Texture inputs are handled similarly at the color-texture boundary. Newton reads
``UsdUVTexture.sourceColorSpace`` first, falls back to color-space metadata on
the file attribute, and converts linear/raw color textures to display/sRGB when
they are loaded. Display/sRGB textures stay display-encoded.

Mass and Inertia Precedence
---------------------------

.. seealso::

   :ref:`Mass and Inertia` for general concepts: the programmatic API,
   density-based inference, and finalize-time validation.

For rigid bodies with ``UsdPhysics.MassAPI`` applied, Newton resolves each inertial property
(mass, inertia, center of mass) independently.  Effective attribute values take precedence;
``UsdPhysics.RigidBodyAPI.ComputeMassProperties(...)`` provides baseline values for the rest.
Per the schema's value semantics, an attribute is effective only when its value is usable:
mass and density must be positive and finite, ``diagonalInertia`` nonzero with finite,
nonnegative components, ``principalAxes`` nonzero, and ``centerOfMass`` finite.  Explicitly
authoring a schema fallback value is equivalent to leaving the attribute unauthored.

1. ``newton:inertia`` (from ``NewtonMassAPI``) is a compact 6-element symmetric tensor
   ``[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`` already in the body frame.  When authored, it
   overrides ``physics:diagonalInertia`` and ``physics:principalAxes``.
2. Effective ``physics:mass``, ``physics:diagonalInertia``, and ``physics:centerOfMass`` are
   applied directly when present.  If ``physics:principalAxes`` is not effective, identity
   rotation is used.
3. When ``physics:mass`` is effective but inertia is not, the inertia
   accumulated from collision shapes is scaled by ``authored_mass / accumulated_mass``.
   Shell colliders (``newton:massModel = "shell"``) contribute shell-derived inertia to the
   accumulation before this scaling is applied.
4. For any remaining unresolved properties, Newton falls back to
   ``UsdPhysics.RigidBodyAPI.ComputeMassProperties(...)``.
   In this fallback path, collider contributions use a two-level precedence:

   a. If collider ``UsdPhysics.MassAPI`` has effective ``mass`` and ``diagonalInertia``, those
      values are converted to unit-density collider mass information.
   b. Otherwise, Newton derives unit-density collider mass information from collider
      geometry.  When ``NewtonMassAPI`` is applied to the collider, ``newton:massModel``
      controls whether inertia is derived from the full volume (``"solid"``, default) or a
      thin shell at the surface (``"shell"``).  For shell shapes,
      ``newton:shellThickness`` sets the wall thickness [m] measured inward from the outer
      surface; the sentinel ``-inf`` (default) falls back to ``newton:contactMargin``.

   A collider is skipped (with warning) only if neither path provides usable collider mass
   information.

   .. note::

      The callback payload provided by Newton in this path is unit-density collider shape
      information (volume/COM/inertia basis). Collider density authored via ``UsdPhysics.MassAPI``
      (for example, ``physics:density``) or via bound ``UsdPhysics.MaterialAPI`` is still applied
      by USD during ``ComputeMassProperties(...)``. In other words, unit-density callback data does
      not mean authored densities are ignored.

If resolved mass is non-positive, inverse mass is set to ``0``.

.. tip::

   For the most predictable results, fully author ``physics:mass``, ``physics:diagonalInertia``,
   ``physics:principalAxes``, and ``physics:centerOfMass`` on each rigid body.  This avoids any
   fallback heuristics and is also the fastest import path since ``ComputeMassProperties(...)``
   can be skipped entirely.

.. _schema_resolvers:

Schema Resolvers
----------------

Schema resolvers bridge the gap between solver-specific USD schemas and Newton's internal representation. They remap attributes authored for PhysX, MuJoCo, or other solvers to the equivalent Newton properties, handle priority-based resolution when multiple solvers define the same attribute, and collect solver-native attributes for inspection or custom pipelines.

.. experimental::

   The ``schema_resolvers`` argument in :meth:`newton.ModelBuilder.add_usd` may change without prior notice.

Solver Attribute Remapping
~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with USD assets authored for other physics solvers like PhysX or MuJoCo, Newton's schema resolver system can automatically remap various solver attributes to Newton's internal representation. This enables Newton to use physics properties from assets originally designed for other simulators without manual conversion.

The following tables show examples of how solver-specific attributes are mapped to Newton's internal representation. Some attributes map directly while others require mathematical transformations.

**PhysX Attribute Remapping Examples:**

The table below shows PhysX attribute remapping examples:

.. list-table:: PhysX Attribute Remapping
   :header-rows: 1
   :widths: 30 30 40

   * - **PhysX Attribute**
     - **Newton Equivalent**
     - **Transformation**
   * - ``physxJoint:armature``
     - ``armature``
     - Direct mapping
   * - ``physxArticulation:enabledSelfCollisions``
     - ``self_collision_enabled`` (per articulation)
     - Direct mapping

**Newton articulation remapping:**

On articulation root prims (with ``PhysicsArticulationRootAPI`` or ``NewtonArticulationRootAPI``), the following is resolved:

.. list-table:: Newton Articulation Remapping
   :header-rows: 1
   :widths: 30 30 40

   * - **Newton Attribute**
     - **Resolved key**
     - **Transformation**
   * - ``newton:selfCollisionEnabled``
     - ``self_collision_enabled``
     - Direct mapping

The parser resolves ``self_collision_enabled`` from either ``newton:selfCollisionEnabled`` or ``physxArticulation:enabledSelfCollisions`` (in resolver priority order). The ``enable_self_collisions`` argument to :meth:`newton.ModelBuilder.add_usd` is used as the default when neither attribute is authored.

**Newton Joint Attribute Remapping:**

On joint prims (``RevoluteJoint``, ``PrismaticJoint``, ``D6Joint``), the following ``NewtonJointAPI`` attributes are resolved uniformly across all DOFs of the joint:

.. list-table:: Newton Joint Attribute Remapping
   :header-rows: 1
   :widths: 30 30 40

   * - **Newton Attribute**
     - **Resolved key**
     - **Notes**
   * - ``newton:armature``
     - ``armature``
     - Direct mapping
   * - ``newton:damping``
     - ``damping``
     - Passive velocity damping
   * - ``newton:friction``
     - ``friction``
     - Direct mapping
   * - ``newton:velocityLimit``
     - ``velocity_limit``
     - ``+inf`` = unlimited (builder default)
   * - ``newton:limitStiffness``
     - ``limit_ke``
     - ``-inf`` = engine default; ``+inf`` = hard limit
   * - ``newton:limitDamping``
     - ``limit_kd``
     - ``-inf`` = engine default; ignored when ``limitStiffness`` is ``+inf``

Angular joints store gains per-degree in USD and the importer converts to per-radian internally.

**MuJoCo Attribute Remapping Examples:**

The table below shows MuJoCo attribute remapping examples, including both direct mappings and transformations:

.. list-table:: MuJoCo Attribute Remapping
   :header-rows: 1
   :widths: 30 30 40

   * - **MuJoCo Attribute**
     - **Newton Equivalent**
     - **Transformation**
   * - ``mjc:armature``
     - ``armature``
     - Direct mapping
   * - ``mjc:margin``
     - ``margin``
     - Direct mapping (identity under MuJoCo 3.9+). Pass ``legacy_margin_gap=True`` to :meth:`~newton.ModelBuilder.add_usd` for the pre-3.9 ``margin = mjc:margin - mjc:gap`` translation.
   * - ``mjc:gap``
     - ``gap``
     - Direct mapping

**Example USD with remapped attributes:**

The following USD example demonstrates how PhysX attributes are authored in a USD file. The schema resolver automatically applies the transformations shown in the table above during import:

.. code-block:: usda

   #usda 1.0

   def PhysicsScene "Scene" (
       prepend apiSchemas = ["PhysxSceneAPI"]
   ) {
       # PhysX scene settings that Newton can understand
       uint physxScene:maxVelocityIterationCount = 16  # → max_solver_iterations = 16
   }

   def RevoluteJoint "elbow_joint" (
       prepend apiSchemas = ["PhysxJointAPI", "PhysxLimitAPI:angular"]
   ) {
       # PhysX joint attributes remapped to Newton
       float physxJoint:armature = 0.1  # → armature = 0.1
       # PhysX limit attributes (applied via PhysxLimitAPI:angular)
       float physxLimit:angular:stiffness = 1000.0  # → limit_angular_ke = 1000.0
       float physxLimit:angular:damping = 10.0  # → limit_angular_kd = 10.0

       # Initial joint state
       float state:angular:physics:position = 1.57  # → joint_q = 1.57 rad
   }

   def Mesh "collision_shape" (
       prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
   ) {
       # PhysX collision settings (gap = contactOffset - restOffset)
       float physxCollision:contactOffset = 0.05
       float physxCollision:restOffset = 0.01   # → gap = 0.04
   }

Priority-Based Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple physics solvers define conflicting attributes for the same property, the user can define which solver attributes should be preferred by configuring the resolver order.

**Resolution Hierarchy:**

The attribute resolution process follows a three-layer fallback hierarchy to determine which value to use:

1. **Authored Values**: Resolvers are queried in priority order; the first resolver that finds an authored value on the prim returns it and remaining resolvers are not consulted.
2. **Importer Defaults**: If no authored value is found, Newton's importer uses a property-specific fallback (e.g. ``builder.default_joint_cfg.armature`` for joint armature). This takes precedence over schema-level defaults.
3. **Approximated Schema Defaults**: If neither an authored value nor an importer default is available, Newton falls back to a hardcoded approximation of each solver's schema default, defined in Newton's resolver mapping. These approximations will be replaced by actual USD schema defaults in a future release.

**Configuring Resolver Priority:**

The order of resolvers in the ``schema_resolvers`` list determines priority, with earlier entries taking precedence. To demonstrate this, consider a USD asset where the same joint has conflicting armature values authored for different solvers:

.. code-block:: usda

   def RevoluteJoint "shoulder_joint" {
       float newton:armature = 0.01
       float physxJoint:armature = 0.02
       float mjc:armature = 0.03
   }

By changing the order of resolvers in the ``schema_resolvers`` list, different attribute values will be selected from the same USD file. The following examples show how the same asset produces different results based on resolver priority:

.. testcode::
   :skipif: True

   from newton import ModelBuilder
   from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

   builder = ModelBuilder()

   # Configuration 1: Newton priority
   result_newton = builder.add_usd(
       source="conflicting_asset.usda",
       schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()]
   )
   # Result: Uses newton:armature = 0.01

   # Configuration 2: PhysX priority
   builder2 = ModelBuilder()
   result_physx = builder2.add_usd(
       source="conflicting_asset.usda",
       schema_resolvers=[SchemaResolverPhysx(), SchemaResolverNewton(), SchemaResolverMjc()]
   )
   # Result: Uses physxJoint:armature = 0.02

   # Configuration 3: MuJoCo priority
   builder3 = ModelBuilder()
   result_mjc = builder3.add_usd(
       source="conflicting_asset.usda",
       schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton(), SchemaResolverPhysx()]
   )
   # Result: Uses mjc:armature = 0.03


Solver-Specific Attribute Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some attributes are solver-specific and cannot be directly used by Newton's simulation. The schema resolver system preserves these solver-specific attributes during import, making them accessible as part of the parsing results. This is useful for:

* Debugging and inspection of solver-specific properties
* Future compatibility when Newton adds support for additional attributes
* Custom pipelines that need to access solver-native properties
* Sim-to-sim transfer where you might need to rebuild assets for other solvers

**Solver-Specific Attribute Namespaces:**

Each solver has its own namespace prefixes for solver-specific attributes. The table below shows the namespace conventions and provides examples of attributes that would be collected from each solver:

.. list-table:: Solver-Specific Namespaces
   :header-rows: 1
   :widths: 20 40 40

   * - **Engine**
     - **Namespace Prefixes**
     - **Example Attributes**
   * - **PhysX**
     - ``physx``, ``physxScene``, ``physxRigidBody``, ``physxCollision``, ``physxArticulation``
     - ``physxArticulation:enabledSelfCollisions``, ``physxSDFMeshCollision:meshScale``
   * - **MuJoCo**
     - ``mjc``
     - ``mjc:model:joint:testMjcJointScalar``, ``mjc:state:joint:testMjcJointVec3``
   * - **Newton**
     - ``newton``
     - ``newton:maxHullVertices``, ``newton:contactGap``

**Accessing Collected Solver-Specific Attributes:**

The collected attributes are returned in the result dictionary and can be accessed by solver namespace:

.. testcode::
   :skipif: True

   from newton import ModelBuilder
   from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

   builder = ModelBuilder()
   result = builder.add_usd(
       source="physx_humanoid.usda",
       schema_resolvers=[SchemaResolverPhysx(), SchemaResolverNewton()],
   )

   # Access collected solver-specific attributes
   solver_attrs = result.get("schema_attrs", {})

   if "physx" in solver_attrs:
       physx_attrs = solver_attrs["physx"]
       for prim_path, attrs in physx_attrs.items():
           if "physxJoint:armature" in attrs:
               armature_value = attrs["physxJoint:armature"]
               print(f"PhysX joint {prim_path} has armature: {armature_value}")

Custom Attributes from USD
--------------------------

USD assets can define custom attributes that become part of the model/state/control attributes, see :ref:`custom_attributes` for more information.
Besides the programmatic way of defining custom attributes through the :meth:`newton.ModelBuilder.add_custom_attribute` method, Newton's USD importer also supports declaring custom attributes from within a USD stage.

**Overview:**

Custom attributes enable users to:

* Extend Newton's data model with application-specific properties
* Store per-body/joint/dof/shape data directly in USD assets
* Implement custom simulation behaviors driven by USD-authored data
* Organize related attributes using namespaces

**Declaration-First Pattern:**

Custom attributes must be declared on the ``PhysicsScene`` prim with metadata before being used on individual prims:

1. **Declare on PhysicsScene**: Define attributes with ``customData`` metadata specifying assignment and frequency
2. **Assign on individual prims**: Override default values using shortened attribute names

**Declaration Format:**

.. code-block:: usda

   custom <type> newton:namespace:attr_name = default_value (
       customData = {
           string assignment = "model|state|control|contact"
           string frequency = "body|shape|joint|joint_dof|joint_coord|articulation"
       }
   )

Where:

* **namespace** (optional): Custom namespace for organizing related attributes (omit for default namespace)
* **attr_name**: User-defined attribute name
* **assignment**: Storage location (``model``, ``state``, ``control``, ``contact``)
* **frequency**: Per-entity granularity (``body``, ``shape``, ``joint``, ``joint_dof``, ``joint_coord``, ``articulation``)

**Supported Data Types:**

The system automatically infers data types from authored USD values. The following table shows the mapping between USD types and Warp types used internally by Newton:

.. list-table:: Custom Attribute Data Types
   :header-rows: 1
   :widths: 25 25 50

   * - **USD Type**
     - **Warp Type**
     - **Example**
   * - ``float``
     - ``wp.float32``
     - Scalar values
   * - ``bool``
     - ``wp.bool``
     - Boolean flags
   * - ``int``
     - ``wp.int32``
     - Integer values
   * - ``float2``
     - ``wp.vec2``
     - 2D vectors
   * - ``float3``
     - ``wp.vec3``
     - 3D vectors, positions
   * - ``float4``
     - ``wp.vec4``
     - 4D vectors
   * - ``quatf``/``quatd``
     - ``wp.quat``
     - Quaternions (with automatic reordering)

**Assignment Types:**

The ``assignment`` field in the declaration determines where the custom attribute data will be stored. The following table describes each assignment type and its typical use cases:

.. list-table:: Custom Attribute Assignments
   :header-rows: 1
   :widths: 15 25 60

   * - **Assignment**
     - **Storage Location**
     - **Use Cases**
   * - ``model``
     - ``Model`` object
     - Static configuration, physical properties, metadata
   * - ``state``
     - ``State`` object
     - Dynamic quantities, targets, sensor readings
   * - ``control``
     - ``Control`` object
     - Control parameters, actuator settings, gains
   * - ``contact``
     - Contact container
     - Contact-specific properties (future use)

**USD Authoring with Custom Attributes:**

The following USD example demonstrates the complete workflow for authoring custom attributes. Note how attributes are first declared on the ``PhysicsScene`` with their metadata, then assigned with specific values on individual prims:

.. code-block:: usda

   # robot_with_custom_attrs.usda
   #usda 1.0

   def PhysicsScene "physicsScene" {
       # Declare custom attributes with metadata (default namespace)
       custom float newton:mass_scale = 1.0 (
           customData = {
               string assignment = "model"
               string frequency = "body"
           }
       )
       custom float3 newton:local_marker = (0.0, 0.0, 0.0) (
           customData = {
               string assignment = "model"
               string frequency = "body"
           }
       )
       custom bool newton:is_sensor = false (
           customData = {
               string assignment = "model"
               string frequency = "body"
           }
       )
       custom float3 newton:target_position = (0.0, 0.0, 0.0) (
           customData = {
               string assignment = "state"
               string frequency = "body"
           }
       )

       # Declare namespaced custom attributes (namespace_a)
       custom float newton:namespace_a:mass_scale = 1.0 (
           customData = {
               string assignment = "state"
               string frequency = "body"
           }
       )
       custom float newton:namespace_a:gear_ratio = 1.0 (
           customData = {
               string assignment = "model"
               string frequency = "joint"
           }
       )
       custom float2 newton:namespace_a:pid_gains = (0.0, 0.0) (
           customData = {
               string assignment = "control"
               string frequency = "joint"
           }
       )

       # Articulation frequency attribute
       custom float newton:articulation_stiffness = 100.0 (
           customData = {
               string assignment = "model"
               string frequency = "articulation"
           }
       )
   }

   def Xform "robot_body" (
       prepend apiSchemas = ["PhysicsRigidBodyAPI"]
   ) {
       # Assign values to declared attributes (default namespace)
       custom float newton:mass_scale = 1.5
       custom float3 newton:local_marker = (0.1, 0.2, 0.3)
       custom bool newton:is_sensor = true
       custom float3 newton:target_position = (1.0, 2.0, 3.0)

       # Assign values to namespaced attributes (namespace_a)
       custom float newton:namespace_a:mass_scale = 2.5
   }

   def RevoluteJoint "joint1" {
       # Assign joint attributes (namespace_a)
       custom float newton:namespace_a:gear_ratio = 2.25
       custom float2 newton:namespace_a:pid_gains = (100.0, 10.0)
   }

   # Articulation frequency attributes must be defined on the prim with PhysicsArticulationRootAPI
   def Xform "robot_articulation" (
       prepend apiSchemas = ["PhysicsArticulationRootAPI"]
   ) {
       # Assign articulation-level attributes
       custom float newton:articulation_stiffness = 150.0
   }

.. note::
   Attributes with ``frequency = "articulation"`` store per-articulation values and must be
   authored on USD prims that have the ``PhysicsArticulationRootAPI`` schema applied.

**Accessing Custom Attributes in Python:**

After importing the USD file with the custom attributes shown above, they become accessible as properties on the appropriate objects (``Model``, ``State``, or ``Control``) based on their assignment. The following example shows how to import and access these attributes:

.. code-block:: python

   from newton import ModelBuilder

   builder = ModelBuilder()

   # Import the USD file with custom attributes (from example above)
   result = builder.add_usd(
       source="robot_with_custom_attrs.usda",
   )

   model = builder.finalize()
   state = model.state()
   control = model.control()

   # Access default namespace model-assigned attributes
   body_mass_scale = model.mass_scale.numpy()        # Per-body scalar
   local_markers = model.local_marker.numpy()        # Per-body vec3
   sensor_flags = model.is_sensor.numpy()            # Per-body bool

   # Access default namespace state-assigned attributes
   target_positions = state.target_position.numpy()  # Per-body vec3

   # Access namespaced attributes (namespace_a)
   # Note: Same attribute name can exist in different namespaces with different assignments
   namespaced_mass = state.namespace_a.mass_scale.numpy()  # Per-body scalar (state assignment)
   gear_ratios = model.namespace_a.gear_ratio.numpy()       # Per-joint scalar
   pid_gains = control.namespace_a.pid_gains.numpy()        # Per-joint vec2

   arctic_stiff = model.articulation_stiffness.numpy()      # Per-articulation scalar

**Namespace Isolation:**

Attributes with the same name in different namespaces are completely independent and stored separately. This allows the same attribute name to be used for different purposes across namespaces. In the example above, ``mass_scale`` appears in both the default namespace (as a model attribute) and in ``namespace_a`` (as a state attribute). These are treated as completely separate attributes with independent values, assignments, and storage locations.

.. _sdf_hydroelastic_usd:

SDF Collision and Hydroelastic Contact
--------------------------------------

Newton configures SDF-based mesh collision and hydroelastic contact through the
``NewtonSDFCollisionAPI`` codeless schema (applied to ``Gprim`` shapes). The
schema covers two related but distinct concerns:

* **SDF collision** — resolution, narrow band, AABB padding, texture format.
  These attributes have a one-to-one mapping with PhysX's
  ``PhysxSDFMeshCollisionAPI``, so assets authored for PhysX can be ported.
* **Hydroelastic contact** — opt-in via ``newton:hydroelasticEnabled``,
  parameterized by ``newton:hydroelasticStiffness``. **Newton-only**: PhysX
  has no equivalent schema, so hydroelastic configuration must be authored
  fresh on a Newton-ready asset; there is nothing to port.

Collision margin and gap are inherited from ``NewtonCollisionAPI`` and
covered alongside the SDF mapping below.

.. note::

   ``NewtonSDFCollisionAPI`` and ``NewtonMeshCollisionAPI`` are **independent
   collision representations** and should not be applied to the same prim. If
   both are present, the importer emits a warning and uses the SDF
   configuration. Because ``physics:approximation`` is inherited from
   ``PhysicsMeshCollisionAPI``, it is a mesh-collider concept and is ignored
   (with a warning) on prims that apply ``NewtonSDFCollisionAPI``.

Newton's USD importer reads only the ``newton:*`` attributes. ``physx*:*``
attributes are collected by the schema resolver (see :ref:`schema_resolvers`)
but not applied. To make a PhysX-authored asset work in Newton, author the
Newton schemas alongside (or in place of) the PhysX ones — manually or with
the :ref:`programmatic helper <porting_physx_sdf_assets>` below.

SDF Attribute Mapping (PhysX ↔ Newton)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following tables show the one-to-one mapping between PhysX and Newton schema
attributes, along with any unit or semantic differences.

**Collision margin (applies to any collider):**

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - **PhysX**
     - **Newton**
     - **Notes**
   * - ``physxCollision:restOffset``
     - ``newton:contactMargin``
     - Direct mapping [m].
   * - ``physxCollision:contactOffset``
     - ``newton:contactGap``
     - Semantic shift — see note below.

.. note::

   Newton's ``contactGap`` is additive on top of ``contactMargin``; PhysX's
   ``contactOffset`` is measured from the original shape surface. Convert as
   ``newton:contactGap = physxCollision:contactOffset - physxCollision:restOffset``.

**SDF collision configuration (per mesh shape):**

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - **PhysX**
     - **Newton**
     - **Notes**
   * - ``physxSDFMeshCollision:sdfResolution``
     - ``newton:sdfMaxResolution``
     - Direct mapping; must be divisible by 8.
   * - ``physxSDFMeshCollision:sdfNarrowBandThickness``
     - ``newton:sdfNarrowBandInner``,
       ``newton:sdfNarrowBandOuter``
     - Absolute distances [m]. Split into inner / outer halves — see
       *Narrow band split* below.
   * - ``physxSDFMeshCollision:sdfMargin``
     - ``newton:sdfPadding``
     - Absolute distance [m]. PhysX authors a fraction of the mesh AABB
       diagonal; multiply by the diagonal before authoring on Newton.
   * - ``physxSDFMeshCollision:sdfBitsPerSubgridPixel``
     - ``newton:sdfTextureFormat``
     - ``BitsPerPixel8/16/32`` → ``uint8`` / ``uint16`` / ``float32``.
   * - *(no equivalent)*
     - ``newton:sdfTargetVoxelSize``
     - Absolute voxel size [m]; when ``> 0``, overrides ``sdfMaxResolution``.

.. note::

   **Narrow band split.** PhysX authors a single thickness around the
   surface. Newton splits it into inner / outer halves: set
   ``newton:sdfNarrowBandInner`` to the negated value and
   ``newton:sdfNarrowBandOuter`` to the positive value in meters
   (e.g. ``-0.02`` and ``0.02``).

Hydroelastic Attributes (Newton-only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PhysX does not expose a hydroelastic configuration schema for rigid-body
contacts, so there is no PhysX attribute to map from. Hydroelastic contacts
are authored fresh on ``NewtonSDFCollisionAPI``:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - **Attribute**
     - **Default**
     - **Notes**
   * - ``newton:hydroelasticEnabled``
     - ``false``
     - Opt-in. Both shapes in a contact pair must set ``true`` for
       hydroelastic contacts to be generated.
   * - ``newton:hydroelasticStiffness``
     - ``1e10`` [N/m³]
     - Contact stiffness coefficient. Authored alone (without
       ``hydroelasticEnabled=true``) it is a material parameter and does
       **not** turn hydroelastic on.

Hydroelastic contact uses the same SDF representation as SDF collision, so
its configuration lives on the same applied API. A mesh that opts into
hydroelastic must also have an SDF source (an ``sdfMaxResolution`` or
``sdfTargetVoxelSize`` authored, or an attached ``mesh.sdf``); the importer
validates this at parse time.

Authoring Examples
~~~~~~~~~~~~~~~~~~

A typical PhysX-authored SDF mesh collider looks like this:

.. code-block:: usda

   #usda 1.0

   def Xform "Body"
   (
       prepend apiSchemas = ["PhysicsRigidBodyAPI"]
   )
   {
       def Mesh "CollisionMesh"
       (
           prepend apiSchemas = [
               "PhysicsCollisionAPI",
               "PhysxCollisionAPI",
               "PhysxSDFMeshCollisionAPI",
           ]
       )
       {
           float physxCollision:restOffset = 0.002
           float physxCollision:contactOffset = 0.005
           uint physxSDFMeshCollision:sdfResolution = 256
           float physxSDFMeshCollision:sdfNarrowBandThickness = 0.01
           float physxSDFMeshCollision:sdfMargin = 0.01
           token physxSDFMeshCollision:sdfBitsPerSubgridPixel = "BitsPerPixel16"
       }
   }

The same asset using Newton schemas:

.. code-block:: usda

   #usda 1.0

   def Xform "Body"
   (
       prepend apiSchemas = ["PhysicsRigidBodyAPI"]
   )
   {
       def Mesh "CollisionMesh"
       (
           prepend apiSchemas = ["NewtonSDFCollisionAPI"]
       )
       {
           float newton:contactMargin = 0.002
           float newton:contactGap = 0.003
           uniform int newton:sdfMaxResolution = 256
           uniform float newton:sdfNarrowBandInner = -0.02
           uniform float newton:sdfNarrowBandOuter = 0.02
           uniform float newton:sdfPadding = 0.02
           uniform token newton:sdfTextureFormat = "uint16"
       }
   }

Two details are worth highlighting:

* ``newton:contactGap = 0.003`` (PhysX ``contactOffset=0.005`` minus
  ``restOffset=0.002``), because Newton's gap is additive on top of the margin.
* PhysX authors the narrow-band thickness as a fraction of the mesh AABB
  diagonal; Newton authors absolute distances [m] split into inner/outer
  halves. For a unit-cube-ish mesh, a PhysX fraction of ``0.01`` corresponds
  to roughly ``±0.02`` m (``0.01 * sqrt(3)``).

To also opt into hydroelastic contacts, set ``newton:hydroelasticEnabled=true``
on the same ``NewtonSDFCollisionAPI`` and author ``newton:hydroelasticStiffness``:

.. code-block:: usda

   def Mesh "CollisionMesh"
   (
       prepend apiSchemas = ["NewtonSDFCollisionAPI"]
   )
   {
       uniform int newton:sdfMaxResolution = 256
       bool newton:hydroelasticEnabled = true
       float newton:hydroelasticStiffness = 1e7
   }

.. _porting_physx_sdf_assets:

Programmatic Porting from PhysX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following helper rewrites PhysX SDF attributes to their Newton equivalents
on every prim that has ``PhysxSDFMeshCollisionAPI`` applied. It is a starting
point — edit the token map if your assets use different bit-depth or texture
conventions. A ``defaults`` argument controls how unauthored PhysX attributes
are handled: ``PortDefaults.NEWTON`` (default) leaves the corresponding
Newton attributes unauthored so the Newton importer applies its own schema
defaults, while ``PortDefaults.PHYSX`` backfills from the PhysX schema
defaults so the ported authoring preserves PhysX-fidelity behavior. The
snippet below is executed as part of the documentation test suite, so it
doubles as an end-to-end regression against Newton's importer.

.. testcode::

   import enum
   import math

   from pxr import Sdf, Usd, UsdGeom, UsdPhysics

   _BITS_TO_FORMAT = {
       "BitsPerPixel8": "uint8",
       "BitsPerPixel16": "uint16",
       "BitsPerPixel32": "float32",
   }

   # PhysX per-shape schema defaults for SDF attributes. restOffset / contactOffset
   # default to -inf in PhysX (defer to scene-level defaults) and therefore have no
   # per-shape value to backfill — they are intentionally omitted here.
   _PHYSX_DEFAULTS = {
       "physxSDFMeshCollision:sdfResolution": 256,
       "physxSDFMeshCollision:sdfNarrowBandThickness": 0.01,
       "physxSDFMeshCollision:sdfMargin": 0.01,
       "physxSDFMeshCollision:sdfBitsPerSubgridPixel": "BitsPerPixel16",
   }


   class PortDefaults(enum.Enum):
       """Source for unauthored attributes during porting: ``NEWTON`` leaves
       the Newton attribute unauthored (importer applies its own schema
       defaults); ``PHYSX`` backfills from PhysX schema defaults."""

       NEWTON = "newton"
       PHYSX = "physx"

   def _get(prim, name):
       attr = prim.GetAttribute(name)
       if not attr or not attr.HasAuthoredValue():
           return None
       return attr.Get()

   def _physx_attr(prim, name, defaults):
       """Read a PhysX attribute, falling back to its schema default in PHYSX mode."""
       v = _get(prim, name)
       if v is None and defaults is PortDefaults.PHYSX:
           return _PHYSX_DEFAULTS.get(name)
       return v

   def _set(prim, name, type_name, value):
       prim.CreateAttribute(name, type_name, custom=False).Set(value)

   def _has_applied_schema(prim, name):
       # Read apiSchemas metadata directly so detection works even when the
       # schema type is not registered with the USD runtime.
       op = prim.GetMetadata("apiSchemas")
       if op is None:
           return False
       return any(name in items for items in (op.explicitItems, op.prependedItems,
                                              op.appendedItems, op.addedItems,
                                              op.orderedItems) if items)

   def _bbox_diag(prim) -> float | None:
       """World-space AABB diagonal [m] of ``prim``, or ``None`` if empty."""
       cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                 includedPurposes=[UsdGeom.Tokens.default_])
       rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
       if rng.IsEmpty():
           return None
       size = rng.GetSize()
       diag = size.GetLength()
       return float(diag) if diag > 0 else None

   def port_physx_sdf_to_newton(
       stage: Usd.Stage, defaults: PortDefaults = PortDefaults.NEWTON,
   ) -> int:
       """Rewrite PhysxSDFMeshCollisionAPI / PhysxCollisionAPI attrs on each
       prim as NewtonSDFCollisionAPI / NewtonCollisionAPI attrs. Returns the
       number of prims modified. See :class:`PortDefaults` for the meaning of
       ``defaults``."""
       modified = 0
       for prim in stage.Traverse():
           if not _has_applied_schema(prim, "PhysxSDFMeshCollisionAPI"):
               continue

           # Margins: newton:contactMargin == restOffset;
           # newton:contactGap == contactOffset - restOffset. PhysX uses -inf
           # as a "defer to scene default" sentinel on both fields; skip the
           # mapping in that case so we don't write -inf into newton:contactMargin
           # (which has no sentinel).
           rest = _get(prim, "physxCollision:restOffset")
           contact = _get(prim, "physxCollision:contactOffset")
           if rest is not None and rest != float("-inf"):
               _set(prim, "newton:contactMargin", Sdf.ValueTypeNames.Float, float(rest))
           if (
               rest is not None and contact is not None
               and rest != float("-inf") and contact != float("-inf")
           ):
               _set(prim, "newton:contactGap", Sdf.ValueTypeNames.Float,
                    float(contact) - float(rest))

           # SDF resolution (direct).
           res = _physx_attr(prim, "physxSDFMeshCollision:sdfResolution", defaults)
           if res is not None:
               _set(prim, "newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, int(res))

           # PhysX authors narrow band / margin as a fraction of the mesh
           # AABB diagonal. Newton authors absolute distances [m], so
           # multiply by the bbox diagonal at port time.
           diag = _bbox_diag(prim)

           # Narrow band: single fraction -> (inner=-t*diag, outer=+t*diag).
           t = _physx_attr(prim, "physxSDFMeshCollision:sdfNarrowBandThickness", defaults)
           if t is not None and diag is not None:
               abs_t = float(t) * diag
               _set(prim, "newton:sdfNarrowBandInner", Sdf.ValueTypeNames.Float, -abs_t)
               _set(prim, "newton:sdfNarrowBandOuter", Sdf.ValueTypeNames.Float, abs_t)

           # Margin: fraction -> absolute [m] via diag.
           m = _physx_attr(prim, "physxSDFMeshCollision:sdfMargin", defaults)
           if m is not None and diag is not None:
               _set(prim, "newton:sdfPadding", Sdf.ValueTypeNames.Float,
                    float(m) * diag)

           # Texture format token translation.
           bits = _physx_attr(prim, "physxSDFMeshCollision:sdfBitsPerSubgridPixel", defaults)
           if bits is not None:
               fmt = _BITS_TO_FORMAT.get(str(bits))
               if fmt is not None:
                   _set(prim, "newton:sdfTextureFormat", Sdf.ValueTypeNames.Token, fmt)

           # Apply the Newton SDF API so the importer honors schema defaults
           # for any attributes left unauthored. NewtonSDFCollisionAPI inherits
           # NewtonCollisionAPI and PhysicsCollisionAPI; listing them explicitly
           # would be redundant.
           prim.AddAppliedSchema("NewtonSDFCollisionAPI")
           modified += 1
       return modified

   # Round-trip demo: PhysX-authored stage -> port -> verify.
   stage = Usd.Stage.CreateInMemory()
   UsdPhysics.Scene.Define(stage, "/World")
   body = stage.DefinePrim("/World/Body", "Xform")
   UsdPhysics.RigidBodyAPI.Apply(body)
   mesh = UsdGeom.Mesh.Define(stage, "/World/Body/CollisionMesh")
   mesh.CreatePointsAttr([(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5),
                          (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
                          (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)])
   mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
   mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
                                     2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5])
   p = mesh.GetPrim()
   UsdPhysics.CollisionAPI.Apply(p)
   p.AddAppliedSchema("PhysxCollisionAPI")
   p.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
   p.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.002)
   p.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.005)
   p.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.UInt).Set(256)
   p.CreateAttribute("physxSDFMeshCollision:sdfNarrowBandThickness",
                     Sdf.ValueTypeNames.Float).Set(0.01)
   p.CreateAttribute("physxSDFMeshCollision:sdfMargin", Sdf.ValueTypeNames.Float).Set(0.02)
   p.CreateAttribute("physxSDFMeshCollision:sdfBitsPerSubgridPixel",
                     Sdf.ValueTypeNames.Token).Set("BitsPerPixel16")

   assert port_physx_sdf_to_newton(stage) == 1

   # After porting, each mapped Newton attribute is authored with the
   # expected value and the Newton API schemas are applied. Float comparisons
   # are approximate because USD stores floats as float32.
   def _close(a, b, tol=1e-6):
       return abs(a - b) <= tol

   assert _close(_get(p, "newton:contactMargin"), 0.002)
   assert _close(_get(p, "newton:contactGap"), 0.003)
   assert _get(p, "newton:sdfMaxResolution") == 256
   _diag = math.sqrt(3)  # unit cube [-0.5, 0.5]^3
   assert _close(_get(p, "newton:sdfNarrowBandInner"), -0.01 * _diag, tol=1e-5)
   assert _close(_get(p, "newton:sdfNarrowBandOuter"), 0.01 * _diag, tol=1e-5)
   assert _close(_get(p, "newton:sdfPadding"), 0.02 * _diag, tol=1e-5)
   assert _get(p, "newton:sdfTextureFormat") == "uint16"
   assert _has_applied_schema(p, "NewtonSDFCollisionAPI")

   # PHYSX-mode demo: applied API with no authored attrs -> backfill from PhysX defaults.
   stage2 = Usd.Stage.CreateInMemory()
   mesh2 = UsdGeom.Mesh.Define(stage2, "/Body/CollisionMesh")
   mesh2.CreatePointsAttr([(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5),
                           (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
                           (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)])
   mesh2.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
   mesh2.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
                                      2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5])
   p2 = mesh2.GetPrim()
   p2.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
   assert port_physx_sdf_to_newton(stage2, defaults=PortDefaults.PHYSX) == 1
   assert _get(p2, "newton:sdfMaxResolution") == 256
   assert _close(_get(p2, "newton:sdfNarrowBandOuter"), 0.01 * _diag, tol=1e-5)
   assert _close(_get(p2, "newton:sdfPadding"), 0.01 * _diag, tol=1e-5)
   assert _get(p2, "newton:sdfTextureFormat") == "uint16"

   # NEWTON mode (default): leave Newton attrs unauthored; importer applies its own schema defaults.
   stage3 = Usd.Stage.CreateInMemory()
   mesh3 = UsdGeom.Mesh.Define(stage3, "/Body/CollisionMesh")
   mesh3.CreatePointsAttr([(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5),
                           (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
                           (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)])
   mesh3.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
   mesh3.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
                                      2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5])
   p3 = mesh3.GetPrim()
   p3.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
   assert port_physx_sdf_to_newton(stage3) == 1  # default is PortDefaults.NEWTON
   assert _get(p3, "newton:sdfMaxResolution") is None
   assert _get(p3, "newton:sdfNarrowBandOuter") is None
   assert _get(p3, "newton:sdfPadding") is None
   assert _has_applied_schema(p3, "NewtonSDFCollisionAPI")

   # Xform-scale demo: PhysX fractional SDF distances are fractions of the
   # SCALED collision shape's AABB, so we use ComputeWorldBound (which
   # captures ancestor xformOps) rather than ComputeLocalBound.
   import pxr.Gf as _Gf  # noqa: PLC0415
   stage4 = Usd.Stage.CreateInMemory()
   parent = UsdGeom.Xform.Define(stage4, "/Body")
   parent.AddScaleOp().Set(_Gf.Vec3f(2.0, 3.0, 1.0))
   mesh4 = UsdGeom.Mesh.Define(stage4, "/Body/CollisionMesh")
   mesh4.CreatePointsAttr([(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5),
                           (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
                           (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)])
   mesh4.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
   mesh4.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
                                      2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5])
   p4 = mesh4.GetPrim()
   p4.AddAppliedSchema("PhysxSDFMeshCollisionAPI")
   p4.CreateAttribute("physxSDFMeshCollision:sdfMargin", Sdf.ValueTypeNames.Float).Set(0.01)
   assert port_physx_sdf_to_newton(stage4) == 1
   # Scaled world-space diagonal: sqrt((1*2)^2 + (1*3)^2 + (1*1)^2) = sqrt(14)
   _scaled_diag = math.sqrt(14.0)
   assert _close(_get(p4, "newton:sdfPadding"), 0.01 * _scaled_diag, tol=1e-5)

The helper leaves the original ``PhysxSDFMeshCollisionAPI`` attributes in place
so the asset continues to work with PhysX; you can optionally remove them after
verifying the Newton import.

.. tip::

   Applying ``NewtonSDFCollisionAPI`` declares the prim's SDF configuration.
   The importer fills in schema defaults for any attributes that are not
   authored (e.g. ``sdfMaxResolution=64``; ``sdfPadding`` defaults to the
   value of ``newton:contactGap``).

   For **mesh** shapes, applying the API causes ``ModelBuilder.finalize()``
   to build a deferred SDF on the underlying ``Mesh``. For **primitive**
   shapes (sphere, box, capsule, cylinder, cone, ellipsoid), the SDF
   parameters are recorded but no texture SDF is generated unless
   ``newton:hydroelasticEnabled = true`` is also set — primitive texture
   SDFs are produced today only along the hydroelastic contact path.

   Hydroelastic contacts are folded into the same API and are **opt-in**:
   ``newton:hydroelasticEnabled`` defaults to ``false``. Set it to
   ``true`` to enable hydroelastic contacts on a shape
   (``hydroelasticStiffness`` defaults to ``1e10``).
   ``newton:hydroelasticStiffness`` authored on its own is a material
   parameter and does **not** flip hydroelastic contacts on. To remove
   the configuration entirely, remove ``NewtonSDFCollisionAPI`` from the
   prim (e.g. via USD variant sets) — there is no separate toggle
   attribute.


Limitations
-----------

Importing USD files where many (> 30) mesh colliders are under the same rigid body
can result in a crash in ``UsdPhysics.LoadUsdPhysicsFromRange``.  This is a known
thread-safety issue in OpenUSD and will be fixed in a future release of
``usd-core``.  It can be worked around by setting the work concurrency limit to 1
before ``pxr`` initializes its thread pool.

.. note::

   Setting the concurrency limit to 1 disables multi-threaded USD processing
   globally and may degrade performance of other OpenUSD workloads in the same
   process.

Choose **one** of the two approaches below — do not combine them.
``PXR_WORK_THREAD_LIMIT`` is evaluated once when ``pxr`` is first imported and
cached for the lifetime of the process; after that point,
``Work.SetConcurrencyLimit()`` cannot override it.  Conversely, if the env var
*is* set, calling ``Work.SetConcurrencyLimit()`` has no effect.

**Option A — environment variable (before any USD import):**

.. code-block:: python

   import os
   os.environ["PXR_WORK_THREAD_LIMIT"] = "1"  # must precede any pxr import

   from newton import ModelBuilder

   builder = ModelBuilder()
   result = builder.add_usd(
       source="rigid_body_with_many_mesh_colliders.usda",
   )

**Option B —** ``Work.SetConcurrencyLimit`` **(only when the env var is not set):**

.. code-block:: python

   from pxr import Work
   import os

   if "PXR_WORK_THREAD_LIMIT" not in os.environ:
       Work.SetConcurrencyLimit(1)

   from newton import ModelBuilder

   builder = ModelBuilder()
   result = builder.add_usd(
       source="rigid_body_with_many_mesh_colliders.usda",
   )

.. seealso::

   `threadLimits.h`_ (API reference) and `threadLimits.cpp`_ (implementation)
   document the precedence rules between the environment variable and the API.

   .. _threadLimits.h: https://openusd.org/dev/api/thread_limits_8h.html
   .. _threadLimits.cpp: https://github.com/PixarAnimationStudios/OpenUSD/blob/release/pxr/base/work/threadLimits.cpp
