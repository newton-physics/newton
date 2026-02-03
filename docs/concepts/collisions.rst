.. _Collisions:

Collisions
==========

Newton provides a flexible collision detection system for rigid-rigid and soft-rigid contacts. The pipeline handles broad phase culling, narrow phase contact generation, and filtering.

Newton's collision system is also compatible with MuJoCo-imported models via MJWarp, enabling advanced contact models (SDF, hydroelastic) for MuJoCo scenes. See ``examples/mjwarp/`` for usage.

.. _Collision Pipelines Overview:

Collision Pipelines
-------------------

Newton provides two collision pipeline implementations:

**CollisionPipeline** (Standard)
  Uses precomputed shape pairs determined during model finalization. Efficient when the number of potentially colliding pairs is limited (NxN minus filtering rules). For scenes with many shapes where most pairs are filtered out, this avoids runtime broad phase overhead.

**CollisionPipelineUnified**
  Supports multiple broad phase algorithms. Use this when the number of potential collision pairs would be large (e.g., hundreds of shapes that could all collide). Also required for advanced contact models like SDF-based and hydroelastic contacts. See :ref:`Unified Pipeline` for details.

Basic usage:

.. code-block:: python

    # Default: uses CollisionPipeline with precomputed pairs
    contacts = model.collide(state)

    # Or explicitly create a pipeline
    from newton import CollisionPipelineUnified, BroadPhaseMode
    
    pipeline = CollisionPipelineUnified.from_model(
        model,
        broad_phase_mode=BroadPhaseMode.SAP,
    )
    contacts = model.collide(state, collision_pipeline=pipeline)

.. _Supported Shape Types:

Supported Shape Types
---------------------

Newton supports the following geometry types via :class:`~newton.GeoType`:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - ``PLANE``
     - Infinite plane (ground)
   * - ``SPHERE``
     - Sphere primitive
   * - ``CAPSULE``
     - Cylinder with hemispherical ends
   * - ``BOX``
     - Axis-aligned box
   * - ``CYLINDER``
     - Cylinder
   * - ``CONE``
     - Cone
   * - ``ELLIPSOID``
     - Ellipsoid
   * - ``MESH``
     - Triangle mesh (arbitrary, including non-convex)
   * - ``CONVEX_MESH``
     - Convex hull mesh

.. note::
   **Heightfields** (``HFIELD``) are not implemented. Convert heightfield terrain to a triangle mesh.

.. note::
   **SDF is a collision option**, not a standalone shape type. Enable SDF generation on any shape 
   via ``sdf_max_resolution`` or ``sdf_target_voxel_size`` to precompute a signed distance field 
   from the shape's geometry. The SDF provides O(1) distance queries that accelerate collision 
   detection, especially for mesh-mesh pairs. The collision pipeline decides when to use SDF data 
   based on efficiency.

.. _Shapes and Bodies:

Shapes and Rigid Bodies
-----------------------

Collision shapes are attached to rigid bodies. Each shape has:

- **Body index** (``shape_body``): The rigid body this shape is attached to. Use ``body=-1`` for static/world-fixed shapes.
- **Local transform** (``shape_transform``): Position and orientation relative to the body frame.
- **Scale** (``shape_scale``): 3D scale factors applied to the shape geometry.
- **Thickness** (``shape_thickness``): Surface thickness used in contact generation (see :ref:`Shape Configuration`).
- **Source geometry** (``shape_source``): Reference to the underlying geometry object (e.g., :class:`~newton.Mesh`, :class:`~newton.SDF`).

During collision detection, shapes are transformed to world space using their parent body's pose:

.. code-block:: python

    # Shape world transform = body_pose * shape_local_transform
    X_world_shape = body_q[shape_body] * shape_transform[shape_id]

Contacts are generated between shapes, not bodies. Depending on the type of solver, the motion of the bodies is affected by forces or constraints that resolve the penetrations between their attached shapes.

.. _Collision Filtering:

Collision Filtering
-------------------

Both pipelines use the same filtering rules based on world indices and collision groups.

.. _World IDs:

World Indices
^^^^^^^^^^^^^

World indices enable multi-world simulations, primarily for reinforcement learning, where objects belonging to different worlds coexist but do not interact through contacts:

- **Index -1**: Global entities that collide with all worlds (e.g., ground plane)
- **Index 0, 1, 2, ...**: World-specific entities that only interact within their world

.. testcode::

    builder = newton.ModelBuilder()
    
    # Global ground (default world -1, collides with all worlds)
    builder.add_ground_plane()
    
    # Robot template
    robot_builder = newton.ModelBuilder()
    body = robot_builder.add_link()
    robot_builder.add_shape_sphere(body, radius=0.5)
    joint = robot_builder.add_joint_free(body)
    robot_builder.add_articulation([joint])
    
    # Instantiate in separate worlds - robots won't collide with each other
    builder.add_world(robot_builder)  # World 0
    builder.add_world(robot_builder)  # World 1

    model = builder.finalize()

For heterogeneous worlds, use :meth:`~newton.ModelBuilder.begin_world` and :meth:`~newton.ModelBuilder.end_world`.

.. note::
   MJWarp does not currently support heterogeneous environments (different models per world).

World indices are stored in :attr:`Model.shape_world`, :attr:`Model.body_world`, etc.

.. _Collision Groups:

Collision Groups
^^^^^^^^^^^^^^^^

Collision groups control which shapes collide within the same world:

- **Group 0**: Collisions disabled
- **Positive groups (1, 2, ...)**: Collide with same group or any negative group
- **Negative groups (-1, -2, ...)**: Collide with shapes in any positive or negative group, except shapes in the same negative group

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 55

   * - Group A
     - Group B
     - Collide?
     - Reason
   * - 0
     - Any
     - ❌
     - Group 0 disables collision
   * - 1
     - 1
     - ✅
     - Same positive group
   * - 1
     - 2
     - ❌
     - Different positive groups
   * - 1
     - -2
     - ✅
     - Positive with any negative
   * - -1
     - -1
     - ❌
     - Same negative group
   * - -1
     - -2
     - ✅
     - Different negative groups

.. testcode::

    builder = newton.ModelBuilder()
    
    # Group 1: only collides with group 1 and negative groups
    body1 = builder.add_body()
    builder.add_shape_sphere(body1, radius=0.5, cfg=builder.ShapeConfig(collision_group=1))
    
    # Group -1: collides with everything (except other -1)
    body2 = builder.add_body()
    builder.add_shape_sphere(body2, radius=0.5, cfg=builder.ShapeConfig(collision_group=-1))

    model = builder.finalize()

**Self-collision within articulations**

Self-collisions within an articulation can be enabled or disabled with ``enable_self_collisions`` when loading models. By default, adjacent body collisions (parent-child pairs connected by joints) are disabled via ``collision_filter_parent=True``.

.. code-block:: python

    # Enable self-collisions when loading models
    builder.add_usd("robot.usda", enable_self_collisions=True)
    builder.add_mjcf("robot.xml", enable_self_collisions=True)
    
    # Or control per-shape (also applies to max-coordinate jointed bodies)
    cfg = builder.ShapeConfig(collision_group=-1, collision_filter_parent=False)

**Controlling particle collisions**

Use ``has_shape_collision`` and ``has_particle_collision`` for fine-grained control over what a shape collides with. Setting both to ``False`` is equivalent to ``collision_group=0``.

.. code-block:: python

    # Shape that only collides with particles (not other shapes)
    cfg = builder.ShapeConfig(has_shape_collision=False, has_particle_collision=True)
    
    # Shape that only collides with other shapes (not particles)
    cfg = builder.ShapeConfig(has_shape_collision=True, has_particle_collision=False)

UsdPhysics Collision Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Newton follows the `UsdPhysics collision filtering specification <https://openusd.org/dev/api/usd_physics_page_front.html#usdPhysics_collision_filtering>`_,
which provides two complementary mechanisms for controlling which shapes collide:

1. **Collision Groups** - Group-based filtering using ``UsdPhysicsCollisionGroup``
2. **Pairwise Filtering** - Explicit shape pair exclusions using ``physics:filteredPairs``

**Collision Groups**

In UsdPhysics, shapes can be assigned to collision groups defined by ``UsdPhysicsCollisionGroup`` prims.
When importing USD files, Newton reads the ``collisionGroups`` attribute from each shape and maps
each unique collision group name to a positive integer ID (starting from 1). Shapes in different
collision groups will not collide with each other unless their groups are configured to interact.

.. code-block:: usda

    # Define a collision group in USD
    def "CollisionGroup_Robot" (
        prepend apiSchemas = ["PhysicsCollisionGroup"]
    ) {
    }

    # Assign shape to a collision group
    def Sphere "RobotPart" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    ) {
        rel physics:collisionGroup = </CollisionGroup_Robot>
    }

When loading this USD, Newton automatically assigns each collision group a unique integer ID
and sets the shape's ``collision_group`` accordingly.

**Pairwise Filtering**

For fine-grained control, UsdPhysics supports explicit pair filtering via the ``physics:filteredPairs``
relationship. This allows excluding specific shape pairs from collision detection regardless of their
collision groups.

.. code-block:: usda

    # Exclude specific shape pairs in USD
    def Sphere "ShapeA" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    ) {
        rel physics:filteredPairs = [</ShapeB>]
    }

Newton reads these relationships during USD import and converts them to
:attr:`ModelBuilder.shape_collision_filter_pairs`.

**Collision Enabled Flag**

Shapes with ``physics:collisionEnabled=false`` are excluded from all collisions by adding filter
pairs against all other shapes in the scene.

Shape Collision Filter Pairs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :attr:`ModelBuilder.shape_collision_filter_pairs` list stores explicit shape pair exclusions.
This is Newton's internal representation for pairwise filtering (including pairs imported from
UsdPhysics ``physics:filteredPairs`` relationships).

.. code-block:: python

    builder = newton.ModelBuilder()
    
    # Add shapes
    body = builder.add_body()
    shape_a = builder.add_shape_sphere(body, radius=0.5)
    shape_b = builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)

    # Exclude this specific pair from collision detection
    builder.add_shape_collision_filter_pair(shape_a, shape_b)

Filter pairs are automatically populated in several cases:

- **Adjacent bodies**: Parent-child body pairs connected by joints (when ``collision_filter_parent=True``). Also applies to max-coordinate jointed bodies.
- **Same-body shapes**: Shapes attached to the same rigid body
- **Disabled self-collision**: All shape pairs within an articulation when ``enable_self_collisions=False``
- **USD filtered pairs**: Pairs defined by ``physics:filteredPairs`` relationships in USD files
- **USD collision disabled**: Shapes with ``physics:collisionEnabled=false`` (filtered against all other shapes)

The resulting filter pairs are stored in :attr:`~newton.Model.shape_collision_filter_pairs` as a set of
``(shape_index_a, shape_index_b)`` tuples (canonical order: ``a < b``).

**USD Import Example**

.. code-block:: python

    # Newton automatically imports UsdPhysics collision filtering
    builder = newton.ModelBuilder()
    builder.add_usd("scene.usda")
    
    # Collision groups and filter pairs are now populated:
    # - shape_collision_group: integer IDs mapped from UsdPhysicsCollisionGroup
    # - shape_collision_filter_pairs: pairs from physics:filteredPairs relationships
    
    model = builder.finalize()

.. _Standard Pipeline:

Standard Pipeline
-----------------

:class:`~newton.CollisionPipeline` is the default implementation. Shape pairs are precomputed during :meth:`~newton.ModelBuilder.finalize` based on filtering rules.

**When to use:**

- Limited number of potential collision pairs (filtering rules eliminate most NxN combinations)
- Static collision topology (pairs don't change at runtime)
- Scenes with <100 potentially colliding shapes

**Limitations:**

- Does not support advanced contact models (SDF-based, hydroelastic, cylinder, cone primitives)
- Pairs fixed at finalization - inefficient if NxN minus filtering is still large

.. note::
   Development is focused on ``CollisionPipelineUnified``, which will replace the standard 
   ``CollisionPipeline`` in the future once feature and performance parity is achieved. 
   New contact models (SDF, hydroelastic, cylinder/cone primitives) are only being added to 
   the unified pipeline. Consider using ``CollisionPipelineUnified`` with 
   ``BroadPhaseMode.EXPLICIT`` if you need static pairs with advanced contact models.

Standard Pipeline Shape Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 12 9 9 9 9 9 9 9 9 9

   * - 
     - Plane
     - Sphere
     - Capsule
     - Box
     - Cylinder
     - Cone
     - Mesh
     - SDF
     - Particle
   * - **Plane**
     - 
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - 
     - ✅
   * - **Sphere**
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - 
     - ✅
   * - **Capsule**
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - 
     - ✅
   * - **Box**
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - 
     - ✅
   * - **Cylinder**
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - ✅
   * - **Cone**
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - ✅
   * - **Mesh**
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - 
     - ✅
   * - **SDF**
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - ✅
   * - **Particle**
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

Empty cells indicate unsupported pairs. Use :class:`~newton.CollisionPipelineUnified` for cylinder, cone, SDF, and hydroelastic contact generation.

The pipeline is created automatically when calling :meth:`~newton.Model.collide` without arguments:

.. code-block:: python

    contacts = model.collide(state)  # Uses CollisionPipeline internally

.. _Unified Pipeline:

Unified Pipeline
----------------

:class:`~newton.CollisionPipelineUnified` provides configurable broad phase algorithms:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Mode
     - Description
   * - **NxN**
     - All-pairs AABB broad phase. O(N²), optimal for small scenes (<100 shapes).
   * - **SAP**
     - Sweep-and-prune AABB broad phase. O(N log N), better for larger scenes with spatial coherence.
   * - **EXPLICIT**
     - Uses precomputed shape pairs like standard pipeline, but with unified pipeline's contact algorithms. Combines static pair efficiency with advanced contact models.

.. code-block:: python

    from newton import CollisionPipelineUnified, BroadPhaseMode
    
    # NxN for small scenes
    pipeline = CollisionPipelineUnified.from_model(model, broad_phase_mode=BroadPhaseMode.NXN)
    
    # SAP for larger scenes
    pipeline = CollisionPipelineUnified.from_model(model, broad_phase_mode=BroadPhaseMode.SAP)
    
    contacts = model.collide(state, collision_pipeline=pipeline)

.. _Shape Compatibility:

Unified Pipeline Shape Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The unified pipeline supports collision detection between all shape type combinations:

.. list-table::
   :header-rows: 1
   :widths: 12 9 9 9 9 9 9 9 9 9

   * - 
     - Plane
     - Sphere
     - Capsule
     - Box
     - Cylinder
     - Cone
     - Mesh
     - SDF
     - Particle
   * - **Plane**
     - 
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
   * - **Sphere**
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
   * - **Capsule**
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
   * - **Box**
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
   * - **Cylinder**
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
   * - **Cone**
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
   * - **Mesh**
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅⚠️
     - ✅⚠️
     - 
   * - **SDF**
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅⚠️
     - ✅
     - 
   * - **Particle**
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 
     - 

**Legend:** ⚠️ = Can be slow for meshes with high triangle counts

Ellipsoid and ConvexMesh are also fully supported. The only unsupported type is ``HFIELD`` (heightfield) - convert to mesh instead.

.. note::
   **SDF** in this table refers to shapes with precomputed SDF data (via ``sdf_max_resolution`` or ``sdf_target_voxel_size``). SDFs are generated from a shape's primary geometry and provide O(1) distance queries.

.. note::
   ``CollisionPipelineUnified`` does not currently support particle (soft body) collisions. Use the standard pipeline for particle-shape contacts. The unified pipeline is under active development.

.. _Narrow Phase:

Narrow Phase Algorithms
-----------------------

After broad phase identifies candidate pairs, the narrow phase generates contact points.

**MPR (Minkowski Portal Refinement)**

The primary algorithm for convex shape pairs. Uses support mapping functions to find the closest points between shapes via Minkowski difference sampling. Works with all convex primitives (sphere, box, capsule, cylinder, cone, ellipsoid) and convex meshes.

**Multi-contact Generation**

For shape pairs, multiple contact points are generated for stable stacking and resting contacts. For the standard collision pipeline, the maximum contacts per shape pair is controlled by ``rigid_contact_max_per_pair`` in :meth:`Model.collide`. The unified collision pipeline automatically estimates buffer sizes based on the model.

.. _Mesh Collisions:

Mesh Collision Handling
^^^^^^^^^^^^^^^^^^^^^^^

Mesh collisions use different strategies depending on the pair type:

**Mesh vs Primitive (e.g., Sphere, Box)**

Uses BVH (Bounding Volume Hierarchy) queries to find nearby triangles, then generates contacts between primitive vertices and triangle surfaces, plus triangle vertices against the primitive.

**Mesh vs Plane**

Projects mesh vertices onto the plane and generates contacts for vertices below the plane surface.

**Mesh vs Mesh**

Two approaches available:

1. **BVH-based** (default when no SDF configured): Iterates mesh vertices against the other mesh's BVH. 
   Performance scales with triangle count - can be very slow for complex meshes.

2. **SDF-based** (recommended): Uses precomputed signed distance fields for fast queries. 
   Enable by setting ``sdf_max_resolution`` or ``sdf_target_voxel_size`` on shapes.

.. warning::
   If SDF is not precomputed, mesh-mesh contacts fall back to on-the-fly BVH distance queries 
   which are **significantly slower**. For production use with complex meshes, configure SDF 
   via ``sdf_max_resolution`` or ``sdf_target_voxel_size``:

   .. code-block:: python

       cfg = builder.ShapeConfig(
           sdf_max_resolution=64,  # Precompute SDF for fast mesh-mesh collision
       )
       builder.add_shape_mesh(body, mesh=my_mesh, cfg=cfg)

.. _Contact Reduction:

Contact Reduction
^^^^^^^^^^^^^^^^^

Contact reduction is enabled by default. For scenes with many mesh-mesh interactions that generate thousands of contacts, reduction selects a significantly smaller representative set that maintains stable contact behavior while improving solver performance.

**How it works:**

1. Contacts are binned by normal direction (20 icosahedron face directions)
2. Within each bin, contacts are scored by spatial distribution and penetration depth
3. Representative contacts are selected using configurable depth thresholds (betas)

To disable reduction, set ``reduce_contacts=False`` when creating the pipeline.

**Configuring contact reduction (SDFHydroelasticConfig):**

For hydroelastic and SDF-based contacts, use :class:`~newton.SDFHydroelasticConfig` to tune reduction behavior:

.. code-block:: python

    from newton import SDFHydroelasticConfig

    config = SDFHydroelasticConfig(
        reduce_contacts=True,           # Enable contact reduction
        betas=(10.0, -0.5),             # Scoring thresholds (default)
        sticky_contacts=0.0,            # Temporal persistence (0 = disabled)
        normal_matching=True,           # Align reduced normals with aggregate force
        moment_matching=False,          # Match friction moments (experimental)
    )

    pipeline = CollisionPipelineUnified.from_model(model, sdf_hydroelastic_config=config)

**Understanding betas:**

The ``betas`` tuple controls how contacts are scored for selection. Each beta value (first element, second element, etc.) produces a separate set of representative contacts per normal bin:

- **Positive beta** (e.g., ``10.0``): Score = ``spatial_position + depth * beta``. Higher values favor deeper contacts.
- **Negative beta** (e.g., ``-0.5``): Score = ``spatial_position * depth^(-beta)`` for penetrating contacts.
  This weighs spatial distribution more heavily for shallow contacts.

The default ``(10.0, -0.5)`` provides a balance: one set prioritizes penetration depth,
another prioritizes spatial coverage. More betas = more contacts retained but better coverage.

.. note::
   The beta scoring behavior is subject to refinement. The unified collision pipeline 
   is under active development and these parameters may change in future releases.

**Other reduction options:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``sticky_contacts``
     - Temporal persistence threshold. When > 0, contacts from previous frames within this distance are preserved to prevent jittering. Default: 0.0 (disabled).
   * - ``normal_matching``
     - Rotates selected contact normals so their weighted sum aligns with the aggregate force direction 
       from all unreduced contacts. Preserves net force direction after reduction. Default: True.
   * - ``moment_matching``
     - Preserves torsional friction by adding an anchor contact at the depth-weighted centroid and 
       scaling friction coefficients. This ensures the reduced contact set produces similar resistance 
       to rotational sliding as the original contacts. Experimental. Default: False.
   * - ``margin_contact_area``
     - Lower bound on contact area. Hydroelastic stiffness is ``area * k_eff``, but contacts 
       within the contact margin that are not yet penetrating (speculative contacts) have zero 
       geometric area. This provides a floor value so they still generate repulsive force. Default: 0.01.

.. _Shape Configuration:

Shape Configuration
-------------------

Shape collision behavior is controlled via :class:`~newton.ModelBuilder.ShapeConfig`:

**Collision control:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``collision_group``
     - Collision group ID. 0 disables collisions. Default: 1.
   * - ``collision_filter_parent``
     - Filter collisions with adjacent body (parent in articulation or connected via joint). Default: True.
   * - ``has_shape_collision``
     - Whether shape collides with other shapes. Default: True.
   * - ``has_particle_collision``
     - Whether shape collides with particles. Default: True.

**Geometry parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``thickness``
     - Surface thickness. Pairwise: summed (``t_a + t_b``). Creates visible gap at rest. Essential for thin shells and cloth to improve simulation stability and reduce self-intersections. Default: 1e-5.
   * - ``contact_margin``
     - AABB expansion for early contact detection. Pairwise: max. The margin only affects contact generation; effective rest distance is not affected and is only governed by ``thickness``. Increasing the margin can help avoid tunneling of fast-moving objects because contacts are detected at a greater distance between objects. Must be >= ``thickness``. Default: None (uses ``builder.rigid_contact_margin``, which defaults to 0.1).
   * - ``is_solid``
     - Whether shape is solid or hollow. Affects inertia and SDF sign. Default: True.
   * - ``is_hydroelastic``
     - Whether the shape uses SDF-based hydroelastic contacts. Both shapes in a pair must have this enabled. See :ref:`Hydroelastic Contacts`. Default: False.
   * - ``k_hydro``
     - Contact stiffness for hydroelastic collisions. Used by MuJoCo, Featherstone, SemiImplicit when ``is_hydroelastic=True``. Default: 1.0e10.

.. note::
   **Contact generation**: A contact is created when ``d < max(margin_a, margin_b)``, where 
   ``d = surface_distance - (thickness_a + thickness_b)``. The solver enforces ``d >= 0``, 
   so objects at rest settle with surfaces separated by ``thickness_a + thickness_b``.

**SDF configuration (generates SDF from shape geometry):**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``sdf_max_resolution``
     - Maximum SDF grid dimension (must be divisible by 8). Either this or ``sdf_target_voxel_size`` enables SDF.
   * - ``sdf_target_voxel_size``
     - Target voxel size for SDF. Takes precedence over ``sdf_max_resolution``.
   * - ``sdf_narrow_band_range``
     - SDF narrow band distance range (inner, outer). Default: (-0.1, 0.1).

Example:

.. code-block:: python

    cfg = builder.ShapeConfig(
        collision_group=-1,           # Collide with everything
        thickness=0.001,              # 1mm thickness
        contact_margin=0.01,          # 1cm margin
        sdf_max_resolution=64,        # Enable SDF for mesh
    )
    builder.add_shape_mesh(body, mesh=my_mesh, cfg=cfg)

**Builder default margin:**

The builder's ``rigid_contact_margin`` (default 0.1) applies to shapes without explicit ``contact_margin``. Alternatively, use ``builder.default_shape_cfg.contact_margin``.

.. _Common Patterns:

Common Patterns
---------------

**Creating static/ground geometry**

Use ``body=-1`` to attach shapes to the static world frame:

.. code-block:: python

    builder = newton.ModelBuilder()
    
    # Static ground plane
    builder.add_ground_plane()  # Convenience method
    
    # Or manually create static shapes
    builder.add_shape_plane(body=-1, xform=wp.transform_identity())
    builder.add_shape_box(body=-1, hx=5.0, hy=5.0, hz=0.1)  # Static floor
    builder.add_shape_mesh(body=-1, mesh=terrain_mesh)      # Static terrain

**Setting default shape configuration**

Use ``builder.default_shape_cfg`` to set defaults for all shapes:

.. code-block:: python

    builder = newton.ModelBuilder()
    
    # Set defaults before adding shapes
    builder.default_shape_cfg.ke = 1.0e6
    builder.default_shape_cfg.kd = 1000.0
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.is_hydroelastic = True
    builder.default_shape_cfg.sdf_max_resolution = 64

**Running collision less frequently**

For performance, you can run collision detection less often than simulation substeps:

.. code-block:: python

    collide_every_n_substeps = 2
    
    for frame in range(num_frames):
        for substep in range(sim_substeps):
            if substep % collide_every_n_substeps == 0:
                contacts = model.collide(state, collision_pipeline=pipeline)
            solver.step(state_0, state_1, control, contacts, dt=sim_dt)
            state_0, state_1 = state_1, state_0

**Soft contacts (particle-shape)**

Soft contacts are generated automatically when particles are present. They use a separate margin:

.. code-block:: python

    # Add particles
    builder.add_particle(pos=wp.vec3(0, 0, 1), vel=wp.vec3(0, 0, 0), mass=1.0)
    
    # Soft contact margin is set at collision time
    contacts = model.collide(state, soft_contact_margin=0.01)
    
    # Access soft contact data
    n_soft = contacts.soft_contact_count.numpy()[0]
    particles = contacts.soft_contact_particle.numpy()[:n_soft]
    shapes = contacts.soft_contact_shape.numpy()[:n_soft]

.. _Contact Generation:

Contact Data
------------

The :class:`~newton.Contacts` class stores the results from the collision detection step
and is consumed by the solver :meth:`~newton.solvers.SolverBase.step` method for contact handling.

.. note::
   Contact forces are not part of the :class:`~newton.Contacts` class - it only stores geometric 
   contact information. See :class:`~newton.SensorContact` for computing contact forces.

**Rigid contacts:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``rigid_contact_count``
     - Number of active rigid contacts (scalar).
   * - ``rigid_contact_shape0``, ``rigid_contact_shape1``
     - Indices of colliding shapes.
   * - ``rigid_contact_point0``, ``rigid_contact_point1``
     - World-space contact points on each shape.
   * - ``rigid_contact_offset0``, ``rigid_contact_offset1``
     - Contact point offsets in body-local space.
   * - ``rigid_contact_normal``
     - Contact normal direction (from shape0 to shape1).
   * - ``rigid_contact_thickness0``, ``rigid_contact_thickness1``
     - Shape thickness at each contact point.

**Soft contacts (particle-shape):**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``soft_contact_count``
     - Number of active soft contacts.
   * - ``soft_contact_particle``
     - Particle indices.
   * - ``soft_contact_shape``
     - Shape indices.
   * - ``soft_contact_body_pos``, ``soft_contact_body_vel``
     - Contact position and velocity on shape.
   * - ``soft_contact_normal``
     - Contact normal.

Example usage:

.. code-block:: python

    contacts = model.collide(state)
    
    n = contacts.rigid_contact_count.numpy()[0]
    points0 = contacts.rigid_contact_point0.numpy()[:n]
    points1 = contacts.rigid_contact_point1.numpy()[:n]
    normals = contacts.rigid_contact_normal.numpy()[:n]
    
    # Shape indices
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]

.. _Collide Method:

Model.collide() Parameters
--------------------------

The :meth:`Model.collide` method accepts the following parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``state``
     - Current simulation state (required).
   * - ``collision_pipeline``
     - Optional custom collision pipeline. If None, creates/reuses default.
   * - ``rigid_contact_max_per_pair``
     - Maximum contacts per shape pair. None = no limit.
   * - ``soft_contact_max``
     - Maximum soft contacts to allocate.
   * - ``soft_contact_margin``
     - Margin for soft contact generation. Default: 0.01.
   * - ``edge_sdf_iter``
     - Iterations for edge-SDF contact search. Default: 10.
   * - ``requires_grad``
     - Enable gradient computation. Default: model.requires_grad.

.. _Hydroelastic Contacts:

Hydroelastic Contacts
---------------------

Hydroelastic contacts are an **opt-in** feature that generates contact areas (not just points) using SDF-based collision detection. This provides more realistic and continuous force distribution, particularly useful for robotic manipulation scenarios.

**Default behavior (hydroelastic disabled):**

When ``is_hydroelastic=False`` (default), shapes use **hard SDF contacts** - point contacts computed from SDF distance queries. This is efficient and suitable for most rigid body simulations.

**Opt-in hydroelastic behavior:**

When ``is_hydroelastic=True`` on **both** shapes in a pair, the system generates distributed contact areas instead of point contacts. This is useful for:

- More stable and continuous contact forces for non-convex shape interactions
- Better force distribution across large contact patches
- Realistic friction behavior for flat-on-flat contacts

**Requirements:**

- Both shapes in a pair must have ``is_hydroelastic=True``
- Shapes must have SDF enabled (``sdf_max_resolution`` or ``sdf_target_voxel_size``)
- Only volumetric shapes supported (not planes, heightfields, or non-watertight meshes)

.. code-block:: python

    cfg = builder.ShapeConfig(
        is_hydroelastic=True,   # Opt-in to hydroelastic contacts
        sdf_max_resolution=64,  # Required for hydroelastic
        k_hydro=1.0e11,         # Contact stiffness
    )
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg)

**How it works:**

1. SDF intersection finds overlapping regions between shapes
2. Marching cubes extracts the contact iso-surface
3. Contact points are distributed across the surface area
4. Optional contact reduction selects representative points

**Hydroelastic stiffness (k_hydro):**

The ``k_hydro`` parameter on each shape controls area-dependent contact stiffness. For a pair, the effective stiffness is computed as the harmonic mean: ``k_eff = 2 * k_a * k_b / (k_a + k_b)``. Tune this for desired penetration behavior.

Contact reduction options for hydroelastic contacts are configured via :class:`~newton.SDFHydroelasticConfig` (see :ref:`Contact Reduction`).

.. _Contact Material Properties:

Contact Materials
-----------------

Shape material properties control contact resolution. Configure via :class:`~newton.ModelBuilder.ShapeConfig`:

.. list-table::
   :header-rows: 1
   :widths: 10 30 22 19 19

   * - Property
     - Description
     - Solvers
     - ShapeConfig
     - Model Array
   * - ``mu``
     - Dynamic friction coefficient
     - All
     - :attr:`~newton.ModelBuilder.ShapeConfig.mu`
     - :attr:`~newton.Model.shape_material_mu`
   * - ``ke``
     - Contact elastic stiffness
     - SemiImplicit, Featherstone, MuJoCo
     - :attr:`~newton.ModelBuilder.ShapeConfig.ke`
     - :attr:`~newton.Model.shape_material_ke`
   * - ``kd``
     - Contact damping
     - SemiImplicit, Featherstone, MuJoCo
     - :attr:`~newton.ModelBuilder.ShapeConfig.kd`
     - :attr:`~newton.Model.shape_material_kd`
   * - ``kf``
     - Friction damping coefficient
     - SemiImplicit, Featherstone
     - :attr:`~newton.ModelBuilder.ShapeConfig.kf`
     - :attr:`~newton.Model.shape_material_kf`
   * - ``ka``
     - Adhesion distance
     - SemiImplicit, Featherstone
     - :attr:`~newton.ModelBuilder.ShapeConfig.ka`
     - :attr:`~newton.Model.shape_material_ka`
   * - ``restitution``
     - Bounciness (requires ``enable_restitution=True`` in solver)
     - XPBD
     - :attr:`~newton.ModelBuilder.ShapeConfig.restitution`
     - :attr:`~newton.Model.shape_material_restitution`
   * - ``torsional_friction``
     - Resistance to spinning at contact
     - XPBD, MuJoCo
     - :attr:`~newton.ModelBuilder.ShapeConfig.torsional_friction`
     - :attr:`~newton.Model.shape_material_torsional_friction`
   * - ``rolling_friction``
     - Resistance to rolling motion
     - XPBD, MuJoCo
     - :attr:`~newton.ModelBuilder.ShapeConfig.rolling_friction`
     - :attr:`~newton.Model.shape_material_rolling_friction`
   * - ``k_hydro``
     - Hydroelastic stiffness
     - SemiImplicit, Featherstone, MuJoCo
     - :attr:`~newton.ModelBuilder.ShapeConfig.k_hydro`
     - :attr:`~newton.Model.shape_material_k_hydro`

Example:

.. code-block:: python

    cfg = builder.ShapeConfig(
        mu=0.8,           # High friction
        ke=1.0e6,         # Stiff contact
        kd=1000.0,        # Damping
        restitution=0.5,  # Bouncy (XPBD only)
    )

.. _USD Collision:

USD Integration
---------------

Custom collision properties can be authored in USD:

.. code-block:: usda

    def Xform "Box" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
    ) {
        custom int newton:collision_group = 1
        custom int newton:world = 0
        custom float newton:contact_ke = 100000.0
        custom float newton:contact_kd = 1000.0
        custom float newton:contact_kf = 1000.0
        custom float newton:contact_ka = 0.0
        custom float newton:contact_thickness = 0.00001
    }

See :doc:`custom_attributes` and :doc:`usd_parsing` for details.

.. _Performance:

Performance
-----------

- Use **EXPLICIT** or standard pipeline when collision pairs are limited (<100 shapes with most pairs filtered)
- Use **SAP** for >100 shapes with spatial coherence
- Use **NxN** for small scenes (<100 shapes) or uniform spatial distribution
- Minimize global entities (world=-1) as they interact with all worlds
- Use positive collision groups to reduce candidate pairs
- Use world indices for parallel simulations (essential for RL with many environments)
- Contact reduction is enabled by default for mesh-heavy scenes
- Adjust ``rigid_contact_max_per_pair`` to limit memory usage in complex scenes

See Also
--------

**Imports:**

.. code-block:: python

    import newton
    from newton import (
        CollisionPipeline,
        CollisionPipelineUnified,
        BroadPhaseMode,
        Contacts,
        GeoType,
    )
    from newton.geometry import SDFHydroelasticConfig

**API Reference:**

- :class:`~newton.CollisionPipeline` - Standard collision pipeline
- :class:`~newton.CollisionPipelineUnified` - Unified pipeline with broad phase options
- :class:`~newton.BroadPhaseMode` - Broad phase algorithm selection
- :class:`~newton.Contacts` - Contact data container
- :class:`~newton.GeoType` - Shape geometry types
- :class:`~newton.ModelBuilder.ShapeConfig` - Shape configuration options
- :meth:`~newton.Model.collide` - Collision detection method
- :class:`~newton.geometry.SDFHydroelasticConfig` - Hydroelastic contact configuration

**Model attributes:**

- :attr:`~newton.Model.shape_collision_group` - Per-shape collision groups
- :attr:`~newton.Model.shape_world` - Per-shape world indices
- :attr:`~newton.Model.shape_contact_margin` - Per-shape contact margins
- :attr:`~newton.Model.shape_thickness` - Per-shape thickness values

**Related documentation:**

- :doc:`custom_attributes` - USD custom attributes for collision properties
- :doc:`usd_parsing` - USD import options including collision settings
- :doc:`sites` - Non-colliding reference points
