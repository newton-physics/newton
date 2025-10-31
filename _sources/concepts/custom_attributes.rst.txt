Custom Attributes
=================

Newton's simulation model uses flat buffer arrays to represent physical properties and simulation state. These arrays can be extended with user-defined custom attributes to store application-specific data alongside the standard physics quantities.

Use Cases
---------

Custom attributes enable a wide range of simulation extensions:

* **Per-body properties**: Store thermal properties, material composition, sensor IDs, or hardware specifications
* **Advanced control**: Store PD gains, velocity limits, control modes, or actuator parameters per-joint or per-DOF
* **Visualization**: Attach colors, labels, rendering properties, or UI metadata to simulation entities
* **Multi-physics coupling**: Store quantities like surface stress, temperature fields, or electromagnetic properties
* **Reinforcement learning**: Store observation buffers, reward weights, optimization parameters, or policy-specific data directly on entities

Custom attributes follow Newton's flat array indexing scheme, enabling efficient GPU-parallel access while maintaining flexibility for domain-specific extensions.

Overview
--------

Newton organizes simulation data into four primary objects, each containing flat arrays indexed by simulation entities: 

* **Model Object** - Static configuration and physical properties that remain constant during simulation
* **State Object** - Dynamic quantities that evolve during simulation
* **Control Object** - Control inputs and actuator commands
* **Contact Object** - Contact-specific properties

Custom attributes extend these objects with user-defined arrays that follow the same indexing scheme as Newton's built-in attributes.

Declaring Custom Attributes
----------------------------

Custom attributes must be declared before use. Each declaration specifies the following properties:

* **frequency**: Array size and indexing pattern (``BODY``, ``SHAPE``, ``JOINT``, ``JOINT_DOF``, ``JOINT_COORD``, or ``ARTICULATION``)
* **assignment**: Which simulation object owns the attribute (``MODEL``, ``STATE``, ``CONTROL``, ``CONTACT``)  
* **dtype**: Warp data type (``wp.float32``, ``wp.vec3``, ``wp.quat``, etc.)
* **default** (optional): Default value for entities that don't explicitly specify values. If not specified, dtype-specific defaults are used: 0.0 for floats, 0 for integers, False for booleans, and zero vectors for vector types.
* **namespace** (optional): Hierarchical organization for grouping related attributes

When **no namespace** is specified, attributes are added directly to their assignment object (e.g., ``model.temperature``). When a **namespace** is provided, Newton creates a namespace container to organize related attributes hierarchically (e.g., ``model.namespace_a.float_attr``).

The following example demonstrates declaring attributes with and without namespaces, and with explicit default values:

.. testcode::

   from newton import ModelBuilder, ModelAttributeFrequency, ModelAttributeAssignment
   import warp as wp
   
   builder = ModelBuilder()
   
   # Default namespace attributes - added directly to assignment objects
   builder.add_custom_attribute(
       name="temperature",
       frequency=ModelAttributeFrequency.BODY,
       dtype=wp.float32,
       default=20.0,  # Explicit default value
       assignment=ModelAttributeAssignment.MODEL
   )
   # → Accessible as: model.temperature
   
   builder.add_custom_attribute(
       name="velocity_limit",
       frequency=ModelAttributeFrequency.BODY,
       dtype=wp.vec3,
       default=(1.0, 1.0, 1.0),  # Default vector value
       assignment=ModelAttributeAssignment.STATE
   )
   # → Accessible as: state.velocity_limit
   
   # Namespaced attributes - organized under namespace containers
   builder.add_custom_attribute(
       name="float_attr",
       frequency=ModelAttributeFrequency.BODY,
       dtype=wp.float32,
       default=0.5,
       assignment=ModelAttributeAssignment.MODEL,
       namespace="namespace_a"
   )
   # → Accessible as: model.namespace_a.float_attr
   
   builder.add_custom_attribute(
       name="bool_attr",
       frequency=ModelAttributeFrequency.SHAPE,
       dtype=wp.bool,
       default=False,
       assignment=ModelAttributeAssignment.MODEL,
       namespace="namespace_a"
   )
   # → Accessible as: model.namespace_a.bool_attr
   
   # Articulation frequency attributes - one value per articulation
   builder.add_custom_attribute(
       name="articulation_stiffness",
       frequency=ModelAttributeFrequency.ARTICULATION,
       dtype=wp.float32,
       default=100.0,
       assignment=ModelAttributeAssignment.MODEL
   )
   # → Accessible as: model.articulation_stiffness

**Default Value Behavior:**

When entities don't explicitly specify custom attribute values, the default value is used:

.. testcode::

   # First body uses the default value (20.0)
   body1 = builder.add_body(mass=1.0)
   
   # Second body overrides with explicit value
   body2 = builder.add_body(
       mass=1.0,
       custom_attributes={"temperature": 37.5}
   )
   
   # Articulation attributes: create multiple articulations with custom values
   for i in range(3):
       builder.add_articulation(
           custom_attributes={
               "articulation_stiffness": 100.0 + float(i) * 50.0  # 100, 150, 200
           }
       )
       base = builder.add_body(mass=1.0)
   
   # After finalization, access both types of attributes
   model = builder.finalize()
   temps = model.temperature.numpy()
   arctic_stiff = model.articulation_stiffness.numpy()
   
   print(f"Body 1: {temps[body1]}")  # 20.0 (default)
   print(f"Body 2: {temps[body2]}")  # 37.5 (authored)
   print(f"Articulation 0: {arctic_stiff[0]}")  # 100.0
   print(f"Articulation 2: {arctic_stiff[2]}")  # 200.0

.. testoutput::

   Body 1: 20.0
   Body 2: 37.5
   Articulation 0: 100.0
   Articulation 2: 200.0

.. note::
   Uniqueness is determined by the full identifier (namespace + name):
     
     - ``model.float_attr`` (key: ``"float_attr"``) and ``model.namespace_a.float_attr`` (key: ``"namespace_a:float_attr"``) can coexist
     - ``model.float_attr`` (key: ``"float_attr"``) and ``state.namespace_a.float_attr`` (key: ``"namespace_a:float_attr"``) can coexist
     - ``model.float_attr`` (key: ``"float_attr"``) and ``state.float_attr`` (key: ``"float_attr"``) cannot coexist - same key
     - ``model.namespace_a.float_attr`` and ``state.namespace_a.float_attr`` cannot coexist - same key ``"namespace_a:float_attr"``
   
Authoring Custom Attributes
----------------------------

After declaration, values are assigned through the standard entity creation API (``add_body``, ``add_shape``, ``add_joint``). For default namespace attributes, use the attribute name directly. For namespaced attributes, use the format ``"namespace:attr_name"``.

The following example creates bodies and shapes with custom attribute values:

.. testcode::

   # Create a body with both default and namespaced attributes
   body_id = builder.add_body(
       mass=1.0,
       custom_attributes={
           "temperature": 37.5,                  # default → model.temperature
           "velocity_limit": [2.0, 2.0, 2.0],    # default → state.velocity_limit  
           "namespace_a:float_attr": 0.5,        # namespaced → model.namespace_a.float_attr
       }
   )
   
   # Create a shape with a namespaced attribute
   shape_id = builder.add_shape_box(
       body=body_id,
       hx=0.1, hy=0.1, hz=0.1,
       custom_attributes={
           "namespace_a:bool_attr": True,  # → model.namespace_a.bool_attr
       }
   )

For joints, Newton provides three frequency types to store different granularities of data. The system determines how to process attribute values based on the declared frequency:

* **JOINT frequency** → One value per joint
* **JOINT_DOF frequency** → List of values with one per degree of freedom
* **JOINT_COORD frequency** → List of values with one per position coordinate

The following example demonstrates declaring and authoring attributes for each joint frequency type:

.. testcode::

   # Declare joint attributes with different frequencies
   builder.add_custom_attribute(
       "int_attr",
       ModelAttributeFrequency.JOINT,
       dtype=wp.int32
   )
   builder.add_custom_attribute(
       "float_attr_dof",
       ModelAttributeFrequency.JOINT_DOF,
       dtype=wp.float32
   )
   builder.add_custom_attribute(
       "float_attr_coord",
       ModelAttributeFrequency.JOINT_COORD,
       dtype=wp.float32
   )
   
   # Create a D6 joint with 2 DOFs (1 linear + 1 angular) and 2 coordinates
   parent = builder.add_body(mass=1.0)
   child = builder.add_body(mass=1.0)
   
   cfg = ModelBuilder.JointDofConfig
   joint_id = builder.add_joint_d6(
       parent=parent,
       child=child,
       linear_axes=[cfg(axis=[1, 0, 0])],      # 1 linear DOF
       angular_axes=[cfg(axis=[0, 0, 1])],     # 1 angular DOF
       custom_attributes={
           "int_attr": 5,                      # JOINT frequency: single value
           "float_attr_dof": [100.0, 200.0],   # JOINT_DOF frequency: list with 2 values (one per DOF)
           "float_attr_coord": [0.5, 0.7],     # JOINT_COORD frequency: list with 2 values (one per coordinate)
       }
   )

Accessing Custom Attributes
----------------------------

After finalization, custom attributes become accessible as Warp arrays. Default namespace attributes are accessed directly on their assignment object, while namespaced attributes are accessed through their namespace container.

The following example shows how to access all the attributes we declared and authored above:

.. testcode::

   # Finalize the model
   model = builder.finalize()
   state = model.state()
   
   # Access default namespace attributes (direct access on assignment objects)
   temperatures = model.temperature.numpy()
   velocity_limits = state.velocity_limit.numpy()
   
   print(f"Temperature: {temperatures[body_id]}")
   print(f"Velocity limit: {velocity_limits[body_id]}")
   
   # Access namespaced attributes (via namespace containers)
   namespace_a_body_floats = model.namespace_a.float_attr.numpy()
   namespace_a_shape_bools = model.namespace_a.bool_attr.numpy()
   
   print(f"Namespace A body float: {namespace_a_body_floats[body_id]}")
   print(f"Namespace A shape bool: {bool(namespace_a_shape_bools[shape_id])}")

.. testoutput::

   Temperature: 37.5
   Velocity limit: [2. 2. 2.]
   Namespace A body float: 0.5
   Namespace A shape bool: True

Custom attributes follow the same GPU/CPU synchronization rules as built-in attributes and can be modified during simulation.

USD Integration
---------------

Custom attributes can be authored in USD files using a declaration-first pattern, similar to the Python API. Declarations are placed on the PhysicsScene prim, and individual prims can then assign values to these attributes.

**USD Declaration Pattern:**

1. **Declare on PhysicsScene**: Define custom attributes with metadata specifying assignment and frequency
2. **Assign on Prims**: Override default values using the attribute name

**Declaration Format (on PhysicsScene prim):**

.. code-block:: usda

   def PhysicsScene "physicsScene" {
       # Default namespace attributes
       custom float newton:float_attr = 0.0 (
           customData = {
               string assignment = "model"
               string frequency = "body"
           }
       )
       custom float3 newton:vec3_attr = (0.0, 0.0, 0.0) (
           customData = {
               string assignment = "state"
               string frequency = "body"
           }
       )
       
       # ARTICULATION frequency attribute
       custom float newton:articulation_stiffness = 100.0 (
           customData = {
               string assignment = "model"
               string frequency = "articulation"
           }
       )
       
       # Custom namespace attributes
       custom float newton:namespace_a:some_attrib = 150.0 (
           customData = {
               string assignment = "control"
               string frequency = "joint_dof"
           }
       )
       custom bool newton:namespace_a:bool_attr = false (
           customData = {
               string assignment = "model"
               string frequency = "shape"
           }
       )
   }

**Assignment Format (on individual prims):**

.. code-block:: usda

   def Xform "robot_arm" (
       prepend apiSchemas = ["PhysicsRigidBodyAPI"]
   ) {
       # Override declared attributes with custom values
       custom float newton:float_attr = 850.0
       custom float3 newton:vec3_attr = (1.0, 0.5, 0.3)
       custom float newton:namespace_a:some_attrib = 250.0
   }
   
   def Mesh "gripper" (
       prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
   ) {
       custom bool newton:namespace_a:bool_attr = true
   }

After importing the USD file, attributes are accessible following the same patterns as programmatically declared attributes:

.. testcode::
   :skipif: True

   from newton import ModelBuilder
   
   builder_usd = ModelBuilder()
   builder_usd.add_usd("robot_arm.usda")
   
   model = builder_usd.finalize()
   state = model.state()
   control = model.control()
   
   # Access default namespace attributes
   float_values = model.float_attr.numpy()
   vec3_values = state.vec3_attr.numpy()
   
   # Access namespaced attributes
   namespace_a_floats = model.namespace_a.float_attr.numpy()
   namespace_b_floats = state.namespace_b.float_attr.numpy()
   control_floats = control.namespace_a.float_attr_dof.numpy()

For more information about USD integration and the schema resolver system, see :doc:`usd_parsing`.

Validation and Constraints
---------------------------

The custom attribute system enforces several constraints to ensure correctness:

* Attributes must be declared via ``add_custom_attribute()`` before use (raises ``AttributeError`` otherwise)
* Each attribute must be used with entities matching its declared frequency (raises ``ValueError`` otherwise)
* Each full attribute identifier (namespace + name) can only be declared once with a specific assignment, frequency, and dtype
* The same attribute name can exist in different namespaces because they create different full identifiers (e.g., ``model.float_attr`` uses key ``"float_attr"`` while ``state.namespace_a.float_attr`` uses key ``"namespace_a:float_attr"``)

