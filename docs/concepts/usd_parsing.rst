.. _usd_parsing:

USD Parsing and Schema Resolver System
========================================

Newton provides USD (Universal Scene Description) ingestion and schema resolver pipelines that enable integration of physics assets authored for different simulation solvers. This system allows Newton to use existing USD assets authored for other simulation solvers.

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
Newton's schema resolver system automatically handles these differences, allowing assets authored for any solver to work with Newton's simulation. See the next section for more details.


Newton's USD Import System
--------------------------

Newton's :meth:`newton.ModelBuilder.add_usd` method provides a USD import pipeline that:

* Parses standard UsdPhysics schema for basic rigid body simulation setup
* Resolves common solver attributes that are conceptually similar between different solvers through configurable schema resolvers
* Handles priority-based attribute resolution when multiple solvers define conflicting values for conceptually similar properties
* Collects solver-specific attributes preserving solver-native attributes for potential use in the solver
* Supports parsing of custom Newton model/state/control attributes for specialized simulation requirements

Mass and Inertia Precedence
---------------------------

For rigid bodies with ``UsdPhysics.MassAPI`` applied, Newton resolves inertial properties using the
following precedence:

1. If both ``physics:mass`` and ``physics:diagonalInertia`` are authored on the rigid body, those
   authored values are used directly (no mass-property recomputation).
   If ``physics:centerOfMass`` is missing, ``(0, 0, 0)`` is used.
   If ``physics:principalAxes`` is missing, identity rotation is used.
2. If either mass or diagonal inertia is missing, Newton falls back to
   ``UsdPhysics.RigidBodyAPI.ComputeMassProperties(...)``.
   In this fallback path, collider contributions use a two-level precedence:

   a. If collider ``UsdPhysics.MassAPI`` has authored ``mass`` and ``diagonalInertia``, those
      authored values are converted to unit-density collider mass information.
   b. Otherwise, Newton derives unit-density collider mass information from collider geometry.

   A collider is skipped (with warning) only if neither path provides usable collider mass
   information.

   .. note::

      The callback payload provided by Newton in this path is unit-density collider shape
      information (volume/COM/inertia basis). Collider density authored via ``UsdPhysics.MassAPI``
      (for example, ``physics:density``) or via bound ``UsdPhysics.MaterialAPI`` is still applied
      by USD during ``ComputeMassProperties(...)``. In other words, unit-density callback data does
      not mean authored densities are ignored.

When a body mass value is resolved in either path, Newton always overwrites body mass and inverse
mass for the imported body. If resolved mass is non-positive, inverse mass is set to ``0``.

.. _schema_resolvers:

1. Solver Attribute Remapping
-----------------------------

.. note::

  Using the ``schema_resolvers`` argument in :meth:`newton.ModelBuilder.add_usd` to collect solver-specific attributes is an experimental feature that may be removed or changed significantly in the future.

When working with USD assets authored for other physics solvers like PhysX or MuJoCo, Newton's schema resolver system can automatically remap various solver attributes to Newton's internal representation. This enables Newton to use physics properties from assets originally designed for other simulators without manual conversion.

The following tables show examples of how solver-specific attributes are mapped to Newton's internal representation. Some attributes map directly while others require mathematical transformations.

**PhysX Attribute Remapping Examples:**

The table below demonstrates PhysX attribute remapping with both direct mapping and transformation examples:

.. list-table:: PhysX Attribute Remapping
   :header-rows: 1
   :widths: 30 30 40

   * - **PhysX Attribute**
     - **Newton Equivalent**
     - **Transformation**
   * - ``physxJoint:armature``
     - ``armature``
     - Direct mapping
   * - ``physxScene:timeStepsPerSecond``
     - ``time_step``
     - ``1.0 / timeStepsPerSecond``

**MuJoCo Attribute Remapping Examples:**

Similarly, MuJoCo attributes can be remapped to Newton. Note that ``mjc:solref`` requires a more complex transformation to convert MuJoCo's solver reference parameters into stiffness and damping coefficients:

.. list-table:: MuJoCo Attribute Remapping  
   :header-rows: 1
   :widths: 30 30 40

   * - **MuJoCo Attribute**
     - **Newton Equivalent**
     - **Transformation**
   * - ``mjc:armature``
     - ``armature``
     - Direct mapping
   * - ``mjc:solref``
     - ``stiffness``/``damping``
     - ``k = 1/(timeconst^2)``, ``b = 2*dampratio/timeconst``

**Example USD with remapped attributes:**

The following USD example demonstrates how PhysX attributes are authored in a USD file. The schema resolver automatically applies the transformations shown in the table above during import:

.. code-block:: usda

   #usda 1.0
   
   def PhysicsScene "Scene" (
       prepend apiSchemas = ["PhysxSceneAPI"]
   ) {
       # PhysX scene settings that Newton can understand
       uint physxScene:timeStepsPerSecond = 120  # → time_step = 1/120 = 0.0083
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
       # PhysX collision settings
       float physxCollision:contactOffset = 0.02  # → contact_margin = 0.02
   }

2. Priority-Based Resolution
----------------------------

When multiple physics solvers define conflicting attributes for the same property, the user can define which solver attributes should be preferred by configuring the resolver order.

**Resolution Hierarchy:**

The attribute resolution process follows a three-layer fallback hierarchy to determine which value to use:

1. **Authored Values**: First resolver in priority order with an authored value wins
2. **Explicit Defaults**: User-provided default parameter in ``Resolver.get_value(default=...)`` calls with a non-None value wins if no authored value is found
3. **Schema Mapping Defaults**: Resolver-specific default values from the schema definition if no authored value or explicit default is found

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
   from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx, SchemaResolverMjc
   
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


3. Solver-Specific Attribute Collection
----------------------------------------

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
     - ``newton:hullVertexLimit``, ``newton:contactMargin``

**Accessing Collected Solver-Specific Attributes:**

The collected attributes are returned in the result dictionary and can be accessed by solver namespace:

.. testcode::
   :skipif: True

   from newton import ModelBuilder
   from newton._src.usd.schemas import SchemaResolverPhysx, SchemaResolverNewton
   
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

4. Custom Attribute Framework
-----------------------------

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
       
       # ARTICULATION frequency attribute
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
   **Articulation Frequency Attributes**: Attributes with ``frequency = "articulation"`` store per-articulation values and must be authored on USD prims that have the ``PhysicsArticulationRootAPI`` schema applied. The USD parser automatically detects articulation root prims and extracts custom articulation attributes when importing the scene.

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
   
   # Access ARTICULATION frequency attributes
   arctic_stiff = model.articulation_stiffness.numpy()       # Per-articulation scalar (ARTICULATION frequency)
   
   # Namespace isolation: model.mass_scale and state.namespace_a.mass_scale are independent
   # - model.mass_scale has value 1.5 for robot_body (default namespace, model assignment)
   # - state.namespace_a.mass_scale has value 2.5 for the same robot_body (namespace_a, state assignment)
   # These are separate attributes stored in different objects with independent values

**Namespace Isolation:**

Attributes with the same name in different namespaces are completely independent and stored separately. This allows the same attribute name to be used for different purposes across namespaces. In the example above, ``mass_scale`` appears in both the default namespace (as a model attribute) and in ``namespace_a`` (as a state attribute). These are treated as completely separate attributes with independent values, assignments, and storage locations.