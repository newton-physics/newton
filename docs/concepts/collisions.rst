.. _Collisions:

Collisions
==========

Newton provides a flexible collision detection system that supports both rigid-rigid and soft-rigid contacts. The collision pipeline handles broad phase culling, narrow phase contact generation, and filtering based on world indices and collision groups.

.. _Collision Pipeline:

Collision Pipeline
------------------

The collision pipeline is responsible for detecting overlapping geometry and generating contact points between shapes. Newton provides two pipeline implementations with different trade-offs:

**CollisionPipeline**
  The original implementation that uses precomputed shape pairs for collision detection. 
  Shape pairs are determined during model finalization based on filtering rules.
  This pipeline is simple and efficient when the set of potentially colliding pairs is static.

**CollisionPipelineUnified**
  An alternative implementation that supports multiple broad phase algorithms:
  
  - **NxN**: All-pairs AABB broad phase (O(N²), optimal for small scenes <100 shapes)
  - **SAP**: Sweep-and-prune AABB broad phase (O(N log N), better for larger scenes)
  - **EXPLICIT**: Uses precomputed shape pairs (most efficient when pairs are known in advance)
  
  The unified pipeline is more flexible and performs better in scenes with dynamic topology or many shapes.
  
  .. note::
     CollisionPipelineUnified is currently work in progress and does not yet support all contact types 
     (e.g., some soft-rigid contact scenarios). Use the default CollisionPipeline if you encounter compatibility issues.

To use a collision pipeline, create it from your model and pass it to :meth:`Model.collide`:

.. code-block:: python

    # Create unified pipeline with NxN broad phase (default)
    collision_pipeline = newton.CollisionPipelineUnified.from_model(
        model,
        rigid_contact_max_per_pair=10,
        broad_phase_mode=newton.BroadPhaseMode.NXN,
    )
    
    # Use the pipeline for collision detection
    contacts = model.collide(state, collision_pipeline=collision_pipeline)

If no collision pipeline is provided, :meth:`Model.collide` creates a default :class:`CollisionPipeline` instance using precomputed pairs.

.. _World IDs:

World Indices
-------------

World indices enable multi-world simulations where multiple independent simulation instances coexist without interacting. Each entity (particle, body, shape, joint, articulation) has an associated world index that controls collision filtering:

- **Index -1**: Global entities shared across all worlds (e.g., ground plane, environmental obstacles)
- **Index 0, 1, 2, ...**: World-specific entities that only interact within their world

Collision rules based on world indices:

1. Entities from different worlds (except -1) **do not collide** with each other
2. Global entities (index -1) **collide with all worlds**
3. Within the same world, collision groups determine fine-grained interactions

World indices are automatically managed when using :meth:`ModelBuilder.add_world` to instantiate multiple copies of a scene:

.. testcode::

    builder = newton.ModelBuilder()
    
    # Create global ground plane (collides with all worlds)
    builder.add_ground_plane()
    
    # Create robot builder
    robot_builder = newton.ModelBuilder()
    robot_body = robot_builder.add_link()
    robot_builder.add_shape_sphere(robot_body, radius=0.5)
    joint = robot_builder.add_joint_free(robot_body)
    robot_builder.add_articulation([joint])
    
    # Instantiate robots in separate worlds
    builder.add_world(robot_builder)  # Creates world 0
    builder.add_world(robot_builder)  # Creates world 1  
    builder.add_world(robot_builder)  # Creates world 2
    
    model = builder.finalize()
    
    # Robots from different worlds won't collide with each other,
    # but all robots will collide with the global ground plane

World indices are stored in :attr:`Model.shape_world`, :attr:`Model.particle_world`, :attr:`Model.body_world`, etc.

For heterogeneous worlds (where each world has different contents), use the :meth:`begin_world` and :meth:`end_world` methods:

.. code-block:: python

    builder = newton.ModelBuilder()
    
    # Global ground plane (default world -1)
    builder.add_ground_plane()
    
    # World 0: Robot with arm
    builder.begin_world(key="robot_arm")
    arm_base = builder.add_body()
    builder.add_shape_box(arm_base, hx=0.5, hy=0.5, hz=0.5)
    # ... add more robot parts
    builder.end_world()
    
    # World 1: Quadruped
    builder.begin_world(key="quadruped")
    quad_body = builder.add_body()
    builder.add_shape_sphere(quad_body, radius=0.3)
    # ... add legs and joints
    builder.end_world()
    
    model = builder.finalize()

**Performance benefits**

Using different worlds significantly improves both collision detection and solver performance:

- **Collision detection**: Shapes from different worlds are automatically filtered during broad phase, 
  reducing the number of candidate pairs that need to be checked. This results in substantial performance 
  gains when simulating many independent environments.
  
- **Solver performance**: Separating non-interacting entities into different worlds can improve solver 
  efficiency by reducing unnecessary constraint coupling between independent systems.

For large-scale parallel simulations (e.g., reinforcement learning with thousands of environments), 
using world indices is essential for maintaining good performance.

.. _Collision Groups:

Collision Groups
----------------

Collision groups provide fine-grained control over which shapes collide within the same world. Each shape has a collision group ID (:attr:`Model.shape_collision_group`) that determines interaction rules:

- **Group 0**: No collisions (disabled)
- **Positive groups (1, 2, 3, ...)**: Exclusive groups that only collide with themselves and negative groups
- **Negative groups (-1, -2, -3, ...)**: Universal groups that collide with everything except their negative counterpart

Collision group rules:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Group A
     - Group B
     - Collision?
   * - 0
     - Any
     - ❌ No
   * - 1
     - 1
     - ✅ Yes (same positive group)
   * - 1
     - 2
     - ❌ No (different positive groups)
   * - 1
     - -1
     - ✅ Yes (positive with negative)
   * - -1
     - -1
     - ❌ No (same negative group)
   * - -1
     - -2
     - ✅ Yes (different negative groups)

To assign collision groups, use the :class:`ModelBuilder.ShapeConfig`:

.. testcode::

    builder = newton.ModelBuilder()
    
    # Body 1: Collision group 1
    body1 = builder.add_body()
    cfg1 = builder.ShapeConfig(collision_group=1)
    builder.add_shape_sphere(body1, radius=0.5, cfg=cfg1)
    
    # Body 2: Collision group 2 (won't collide with group 1)
    body2 = builder.add_body()
    cfg2 = builder.ShapeConfig(collision_group=2)
    builder.add_shape_sphere(body2, radius=0.5, cfg=cfg2)
    
    # Body 3: Collision group -1 (collides with groups 1 and 2)
    body3 = builder.add_body()
    cfg3 = builder.ShapeConfig(collision_group=-1)
    builder.add_shape_sphere(body3, radius=0.5, cfg=cfg3)
    
    model = builder.finalize()

.. _Filtering Rules:

Filtering Rules
---------------

Collision filtering combines world indices and collision groups to determine if two shapes should generate contacts. The filtering logic is implemented in the kernel function :func:`test_world_and_group_pair`:

.. code-block:: python

    def test_world_and_group_pair(world_a, world_b, group_a, group_b):
        """Test if two entities should collide.
        
        Returns True if entities should collide, False otherwise.
        """
        # Rule 1: Check world indices first
        if world_a != -1 and world_b != -1 and world_a != world_b:
            return False  # Different worlds don't collide
        
        # Rule 2: If same world or at least one is global (-1),
        #         check collision groups
        if group_a == 0 or group_b == 0:
            return False  # Group 0 disables collisions
        
        if group_a > 0:
            # Positive group: collides with same group or negative groups
            return group_a == group_b or group_b < 0
        
        if group_a < 0:
            # Negative group: collides with everything except itself
            return group_a != group_b
        
        return False

The filtering happens during the broad phase, ensuring that only valid shape pairs proceed to narrow phase contact generation.

.. _Common Collision Patterns:

Common Patterns
---------------

**Self-collision within an articulation**

By default, shapes within the same articulation use ``collision_filter_parent=True``, which prevents parent-child body collisions. To enable full self-collision:

.. code-block:: python

    builder.add_shape_box(
        body=body_id,
        hx=0.5, hy=0.5, hz=0.5,
        cfg=builder.ShapeConfig(
            collision_group=-1,  # Collide with everything
            collision_filter_parent=False,  # Enable parent-child collision
        )
    )

When loading from USD or MJCF, use the ``enable_self_collisions`` flag:

.. code-block:: python

    builder.add_usd("robot.usda", enable_self_collisions=True)
    builder.add_mjcf("robot.xml", enable_self_collisions=True)

**Separate robot instances**

Use world indices to prevent collision between robot copies while allowing each to interact with the environment:

.. code-block:: python

    # Global environment
    builder.add_ground_plane()
    obstacles = builder.add_body()
    builder.add_shape_box(obstacles, hx=1, hy=1, hz=1)
    
    # Robot instances in separate worlds
    for i in range(num_robots):
        builder.add_world(robot_builder)

**Layer-based collision**

Use positive collision groups to implement collision layers:

.. code-block:: python

    # Layer 1: Player objects (only collide with themselves)
    player_cfg = builder.ShapeConfig(collision_group=1)
    
    # Layer 2: Enemy objects (only collide with themselves)
    enemy_cfg = builder.ShapeConfig(collision_group=2)
    
    # Layer 3: Environment (collides with all layers)
    env_cfg = builder.ShapeConfig(collision_group=-1)

**Disabling specific shapes**

Set collision group to 0 to disable collision for specific shapes:

.. code-block:: python

    # Visual-only shape with no collision
    builder.add_shape_mesh(
        body=body_id,
        mesh=visual_mesh,
        cfg=builder.ShapeConfig(collision_group=0)
    )

.. _Contact Generation:

Contact Generation
------------------

After the broad phase identifies potentially colliding shape pairs, the narrow phase generates contact points with geometric details:

- **Contact point** (:attr:`Contacts.rigid_contact_point`): World-space position of contact
- **Contact normal** (:attr:`Contacts.rigid_contact_normal`): Direction from shape A to shape B
- **Contact depth** (:attr:`Contacts.rigid_contact_depth`): Penetration depth (negative for separation)
- **Contact shape IDs** (:attr:`Contacts.rigid_contact_shape0`, :attr:`Contacts.rigid_contact_shape1`): Indices of colliding shapes

The maximum number of contacts per shape pair is controlled by ``rigid_contact_max_per_pair`` (default: 10). Use :func:`Model.collide` to generate contacts:

.. code-block:: python

    contacts = model.collide(state)
    
    # Access contact data
    num_contacts = contacts.rigid_contact_count[0]  # Scalar count
    points = contacts.rigid_contact_point.numpy()[:num_contacts]
    normals = contacts.rigid_contact_normal.numpy()[:num_contacts]
    depths = contacts.rigid_contact_depth.numpy()[:num_contacts]

Additional Topics
-----------------

**Contact margins**

Contact margins expand AABBs during broad phase to detect contacts slightly before actual penetration, improving stability for implicit integrators.

Rigid contact margins are controlled per-shape via ``ShapeConfig.contact_margin``:

.. code-block:: python

    # Set margin for specific shape
    cfg = builder.ShapeConfig(contact_margin=0.05)
    builder.add_shape_box(body=body_id, hx=1.0, hy=1.0, hz=1.0, cfg=cfg)
    
    # Or set default margin for all shapes
    builder.rigid_contact_margin = 0.05  # Must be set before finalize()

Soft contact margins are specified via the ``soft_contact_margin`` parameter in :meth:`Model.collide`. Default: 0.01.

**Soft-rigid contacts**

Contacts between particles and shapes are generated separately via :attr:`Contacts.soft_contact_*` arrays. Soft contact generation is automatically enabled when particles are present in the model.

.. _Contact Material Properties:

**Contact material properties**

Shape contact material properties control how contacts are resolved by different solvers. Not all properties are used by all solvers:

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Property
     - Description
     - Used by
   * - :attr:`~newton.Model.shape_material_ke`
     - Contact elastic stiffness
     - SemiImplicit, Featherstone, MuJoCo
   * - :attr:`~newton.Model.shape_material_kd`
     - Contact damping coefficient
     - SemiImplicit, Featherstone, MuJoCo
   * - :attr:`~newton.Model.shape_material_kf`
     - Contact friction damping coefficient
     - SemiImplicit, Featherstone
   * - :attr:`~newton.Model.shape_material_ka`
     - Contact adhesion distance
     - SemiImplicit, Featherstone
   * - :attr:`~newton.Model.shape_material_mu`
     - Coefficient of friction
     - all solvers
   * - :attr:`~newton.Model.shape_material_restitution`
     - Coefficient of restitution (bounciness).
     - XPBD
   * - :attr:`~newton.Model.shape_material_torsional_friction`
     - Coefficient of torsional friction (resistance to spinning at contact point)
     - XPBD, MuJoCo
   * - :attr:`~newton.Model.shape_material_rolling_friction`
     - Coefficient of rolling friction (resistance to rolling motion)
     - XPBD, MuJoCo

For solvers SemiImplicit and Featherstone, contact forces are computed using the ``ke``, ``kd``, ``kf``, and ``ka`` parameters. 
For position-based solvers (XPBD), the ``restitution`` parameter controls velocity reflection at contacts. To take effect, enable restitution in solver constructor via ``enable_restitution=True``.

The MuJoCo solver converts ``ke`` and ``kd`` to MuJoCo's ``solref`` parameters (timeconst and dampratio) for its constraint-based contact model.

**USD collision attributes**

Custom collision groups and world indices can be authored in USD using custom attributes:

.. code-block:: usda

    def Xform "Box" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
    ) {
        custom int newton:collision_group = 1
        custom int newton:world = 0
    }

See :doc:`custom_attributes` and :doc:`usd_parsing` for details on USD integration.

**Performance considerations**

- Use **EXPLICIT** broad phase when shape pairs are known and static
- Use **SAP** broad phase for scenes with >100 shapes and spatially coherent motion
- Use **NxN** broad phase for small scenes or when shapes are uniformly distributed
- Minimize global entities (world=-1) as they interact with all worlds
- Use positive collision groups to reduce the number of candidate pairs

