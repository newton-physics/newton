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

Newton's ``parse_usd()`` function provides a USD import pipeline that:

* Parses standard UsdPhysics schema for basic rigid body simulation setup
* Resolves common solver attributes that are conceptually similar between different solvers through configurable schema resolvers
* Handles priority-based attribute resolution when multiple solvers define conflicting values for conceptually similar properties
* Collects solver-specific attributes preserving solver-native attributes for potential use in the solver
* Supports parsing of custom Newton model/state/control attributes for specialized simulation requirements

1. Solver Attribute Remapping
-----------------------------

When working with USD assets authored for other physics solvers like PhysX or MuJoCo, Newton's schema resolver system can automatically remap various solver attributes to Newton's internal representation. This enables Newton to use physics properties from assets originally designed for other simulators without manual conversion.

Example: Consider the physics time step parameter:

* Newton uses ``newton:timeStep`` (direct time step value in seconds)
* PhysX uses ``physxScene:timeStepsPerSecond`` (frequency, requiring inversion to get the time step)
* MuJoCo uses ``mjc:option:timestep`` (direct time step value in seconds)

Newton can use the time step value from any of these sources and convert it to the internal representation via the schema resolver system.
This allows the USD parser to ingest assets authored for other physics solvers without manual intervention.


Example USD with PhysX Attributes that are remapped to Newton's internal representation:

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

The resolver system follows a three-layer fallback hierarchy:

1. **Authored Values**: First resolver in priority order with an authored value wins
2. **Explicit Defaults**: User-provided default parameter in ``Resolver.get_value(default=...)`` calls with a non-None value wins if no authored value is found
3. **Schema Mapping Defaults**: Resolver-specific default values from the schema definition if no authored value or explicit default is found

Configuring Resolver Priority:

The order of resolvers in the ``schema_resolvers`` list determines priority, with earlier entries taking precedence. For example, 
consider a USD asset with conflicting armature values from different solvers:

.. code-block:: usda

   def RevoluteJoint "shoulder_joint" {
       float newton:armature = 0.01
       float physxJoint:armature = 0.02  
       float mjc:armature = 0.03
   }

.. testcode::
   :skipif: True

   from newton import ModelBuilder
   from newton.utils.schema_resolver import SchemaResolverNewton, SchemaResolverPhysx, SchemaResolverMjc
   
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

Some attributes are solver-specific and cannot be directly used by Newton's simulation. These are usually prefixed with defined terms like ``physxCollision``, ``physxRigidBody``, ``physxSDFMeshCollision``, etc. For MuJoCo, these could be specialized attributes that don't have direct Newton equivalents.

The schema resolver system preserves these solver-specific attributes during import, making them accessible as part of the parsing results. This is useful for:

* Debugging and inspection of solver-specific properties
* Future compatibility when Newton adds support for additional attributes  
* Custom pipelines that need to access solver-native properties
* Sim-to-sim transfer where you might need to rebuild assets for other solvers

**Solver-Specific Attribute Namespaces:**

Each resolver collects attributes from specific namespaces defined in its ``extra_attr_namespaces``:

* PhysX: ``physxScene``, ``physxRigidBody``, ``physxCollision``, ``physxConvexHullCollision``, ``physxSDFMeshCollision``, ``physxMaterial``, ``physxArticulation``
* MuJoCo: ``mjc``  
* Newton: ``newton``

Accessing Collected Solver-Specific Attributes:

.. testcode::
   :skipif: True

   from newton import ModelBuilder
   from newton.utils.schema_resolver import SchemaResolverPhysx, SchemaResolverNewton
   
   builder = ModelBuilder()
   result = builder.add_usd(
       source="physx_humanoid.usda", 
       schema_resolvers=[SchemaResolverPhysx(), SchemaResolverNewton()],
       collect_solver_specific_attrs=True
   )
   
   # Access the collected solver-specific attributes
   solver_attrs = result["solver_specific_attrs"]
   
   # Inspect PhysX-specific attributes
   if "physx" in solver_attrs:
       physx_attrs = solver_attrs["physx"]
       for prim_path, attrs in physx_attrs.items():
           print(f"\nPrim: {prim_path}")
           for attr_name, attr_value in attrs.items():
               print(f"  {attr_name}: {attr_value}")
   
   # Example output:
   # Found PhysX attributes on 12 prims
   #
   # Prim: /World/Humanoid/torso
   # physxRigidBody:retainAccelerations: True
   # physxRigidBody:enableCCD: False
   #
   # Prim: /World/Humanoid/left_hand  
   # physxSDFMeshCollision:sdfResolution: 256
   # physxSDFMeshCollision:sdfSubgridResolution: 6


.. note::
   When ``collect_solver_specific_attrs=False``, the parser skips scanning for solver-specific namespaces, which can improve import performance for large USD files.

4. Custom Attribute Framework
-----------------------------

USD assets can define custom attributes that become part of the model/state/control attributes. Newton's schema resolver system supports these custom attributes that follow a structured naming convention and are automatically parsed and integrated into the simulation model.

For a comprehensive guide to custom attributes, including declaration, authoring via Python API, use cases, and constraints, see :doc:`custom_attributes`.

This section focuses specifically on USD authoring and integration of custom attributes. Custom attributes enable users to:

* Extend Newton's data model with application-specific properties
* Store per-body/joint/dof/shape data directly in USD assets  
* Implement custom simulation behaviors driven by USD-authored data

**Custom Attribute Naming Convention:** 

Newton supports two naming formats for custom attributes in USD:

1. **Default namespace:** ``newton:assignment:frequency:attribute_name``
2. **Custom namespace:** ``newton:assignment:namespace:frequency:attribute_name``

Where:

* **assignment**: Determines where the attribute is stored (``model``, ``state``, ``control``, or ``contact``)
* **namespace** (optional): Custom namespace for organizing related attributes
* **frequency**: Defines the per-entity granularity (``body``, ``shape``, ``joint``, ``joint_dof``, or ``joint_coord``)
* **attribute_name**: User-defined attribute name

Assignment Types:

.. list-table:: Custom Attribute Assignments
   :header-rows: 1
   :widths: 15 25 60

   * - Assignment
     - Storage Location
     - Use Cases
   * - ``model``
     - ``Model`` object
     - Static configuration, physical properties, metadata that doesn't change
   * - ``state``
     - ``State`` object  
     - Dynamic quantities, targets, sensor readings, time-varying data
   * - ``control``
     - ``Control`` object
     - Control parameters, actuator settings, PID gains, command limits
   * - ``contact``
     - Contact system
     - Contact-specific properties

Frequency Types:

.. list-table:: Custom Attribute Frequencies  
   :header-rows: 1
   :widths: 20 80

   * - Frequency
     - Description
   * - ``body``
     - One value per rigid body in the model
   * - ``shape``
     - One value per collision shape
   * - ``joint``
     - One value per joint
   * - ``joint_dof``
     - One value per joint degree of freedom
   * - ``joint_coord``
     - One value per joint coordinate

Supported Data Types:

The system infers Warp data types from authored USD values:

.. list-table:: Custom Attribute Data Types
   :header-rows: 1
   :widths: 25 25 50

   * - USD Type
     - Warp Type
     - Example Usage
   * - ``float``
     - ``wp.float32``
     - Scalar values, gains, thresholds
   * - ``bool``
     - ``wp.bool``
     - Boolean flags, enable/disable states
   * - ``int``
     - ``wp.int32``
     - Integer indices, counts, modes
   * - ``float2``
     - ``wp.vec2``
     - 2D vectors, ranges, limits
   * - ``float3``
     - ``wp.vec3``
     - 3D vectors, positions, orientations
   * - ``float4``
     - ``wp.vec4``
     - 4D vectors, extended parameters
   * - ``quatf``/``quatd``
     - ``wp.quat``
     - Quaternions (with automatic normalization and reordering from USD convention to Newton's convention)

Example USD Authoring with Custom Attributes:

.. code-block:: usda

   #usda 1.0
   
   def Xform "Robot" {
       def Xform "torso" (
           prepend apiSchemas = ["PhysicsRigidBodyAPI"]
       ) {
           # Default namespace - stored directly on Model object
           float newton:model:body:float_attr = 1.5
           float3 newton:model:body:vec3_attr = (0.1, 0.2, 0.3)
           bool newton:model:body:bool_attr = true
           
           # Custom namespace "namespace_a" - stored on model.namespace_a object
           float newton:model:namespace_a:body:float_attr = 0.05
           float3 newton:model:namespace_a:body:vec3_attr = (1.0, 0.5, 0.0)
           
           # Custom namespace "namespace_b" - stored on state.namespace_b object
           bool newton:state:namespace_b:body:bool_attr = true
           float newton:state:namespace_b:body:float_attr = 0.005
           
           # Default namespace state attribute
           float3 newton:state:body:vec3_attr = (1.0, 2.0, 3.0)
       }
       
       def RevoluteJoint "shoulder_joint" {
           # Default namespace joint properties
           float newton:model:joint:float_attr = 2.25
           
           # Custom namespace control attributes
           float newton:control:namespace_a:joint_dof:float_attr = 150.0
           float newton:control:namespace_a:joint_dof:float_attr2 = 0.01
           
           # Custom namespace model attributes
           float newton:model:namespace_b:joint:float_attr = 1000.0
       }
       
       def Mesh "gripper_finger" (
           prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
       ) {
           # Default namespace shape attributes
           float newton:model:shape:float_attr = 5000.0
           
           # Custom namespace shape attributes
           int newton:model:namespace_a:shape:int_attr = 1
           int newton:model:namespace_a:shape:int_attr2 = 2
       }
   }

Importing and Accessing Custom Attributes:

.. testcode::
   :skipif: True

   from newton import ModelBuilder
   from newton.utils.schema_resolver import SchemaResolverNewton

   builder = ModelBuilder()
   result = builder.add_usd(
       source="robot_with_custom_attrs.usda",
       schema_resolvers=[SchemaResolverNewton()],
       collect_solver_specific_attrs=True
   )
   
   model = builder.finalize()
   state = model.state()
   control = model.control()
   
   # Access default namespace MODEL attributes
   body_float_attrs = model.float_attr.numpy()  # Per-body scalar array
   body_vec3_attrs = model.vec3_attr.numpy()   # Per-body vec3 array
   body_bool_attrs = model.bool_attr.numpy()      # Per-body bool array
   joint_float_attrs = model.float_attr.numpy()       # Per-joint scalar array
   
   print(f"Body float attributes: {body_float_attrs}")
   print(f"Body bool attributes: {body_bool_attrs}")
   
   # Access custom namespace "namespace_a" MODEL attributes
   namespace_a_float = model.namespace_a.float_attr.numpy()  # Per-body scalar
   namespace_a_vec3 = model.namespace_a.vec3_attr.numpy()  # Per-body vec3
   namespace_a_int = model.namespace_a.int_attr.numpy()  # Per-shape int
   
   print(f"Namespace A float: {namespace_a_float}")
   print(f"Namespace A vec3: {namespace_a_vec3}")
   
   # Access default namespace STATE attributes
   state_vec3_attrs = state.vec3_attr.numpy()  # Per-body vec3 array
   
   # Access custom namespace "namespace_b" STATE attributes  
   namespace_b_bool = state.namespace_b.bool_attr.numpy()  # Per-body bool
   namespace_b_float = state.namespace_b.float_attr.numpy()  # Per-body float
   
   print(f"State vec3 attributes: {state_vec3_attrs}")
   print(f"Namespace B bool: {namespace_b_bool}")
   
   # Access custom namespace "namespace_a" CONTROL attributes
   namespace_a_control_float = control.namespace_a.float_attr.numpy()  # Per-joint-DOF scalar
   namespace_a_control_float2 = control.namespace_a.float_attr2.numpy()   # Per-joint-DOF scalar
   
   print(f"Namespace A control float: {namespace_a_control_float}")
   print(f"Namespace A control float2: {namespace_a_control_float2}")

This custom attribute framework allows embedding application-specific data directly into USD assets, enabling data-driven simulations.

For detailed information about declaring custom attributes via Python API, default values, validation constraints, and complete usage examples, refer to :doc:`custom_attributes`.