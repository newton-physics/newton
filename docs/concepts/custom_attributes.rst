.. _custom_attributes:

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

Custom attributes must be declared before use via the :meth:`newton.ModelBuilder.add_custom_attribute` method. Each declaration specifies the following properties:

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
       ModelBuilder.CustomAttribute(
           name="temperature",
           frequency=ModelAttributeFrequency.BODY,
           dtype=wp.float32,
           default=20.0,  # Explicit default value
           assignment=ModelAttributeAssignment.MODEL
       )
   )
   # → Accessible as: model.temperature
   
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="velocity_limit",
           frequency=ModelAttributeFrequency.BODY,
           dtype=wp.vec3,
           default=(1.0, 1.0, 1.0),  # Default vector value
           assignment=ModelAttributeAssignment.STATE
       )
   )
   # → Accessible as: state.velocity_limit
   
   # Namespaced attributes - organized under namespace containers
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="float_attr",
           frequency=ModelAttributeFrequency.BODY,
           dtype=wp.float32,
           default=0.5,
           assignment=ModelAttributeAssignment.MODEL,
           namespace="namespace_a"
       )
   )
   # → Accessible as: model.namespace_a.float_attr
   
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="bool_attr",
           frequency=ModelAttributeFrequency.SHAPE,
           dtype=wp.bool,
           default=False,
           assignment=ModelAttributeAssignment.MODEL,
           namespace="namespace_a"
       )
   )
   # → Accessible as: model.namespace_a.bool_attr
   
   # Articulation frequency attributes - one value per articulation
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="articulation_stiffness",
            frequency=ModelAttributeFrequency.ARTICULATION,
            dtype=wp.float32,
            default=100.0,
            assignment=ModelAttributeAssignment.MODEL
       )
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
       base = builder.add_link(mass=1.0)
       joint = builder.add_joint_free(child=base)
       builder.add_articulation(
           joints=[joint],
           custom_attributes={
               "articulation_stiffness": 100.0 + float(i) * 50.0  # 100, 150, 200
           }
       )
   
   # After finalization, access both types of attributes
   model = builder.finalize()
   temps = model.temperature.numpy()
   arctic_stiff = model.articulation_stiffness.numpy()
   
   print(f"Body 1: {temps[body1]}")  # 20.0 (default)
   print(f"Body 2: {temps[body2]}")  # 37.5 (authored)
   print(f"Articulation 2: {arctic_stiff[2]}")  # 100.0
   print(f"Articulation 4: {arctic_stiff[4]}")  # 200.0

.. testoutput::

   Body 1: 20.0
   Body 2: 37.5
   Articulation 2: 100.0
   Articulation 4: 200.0

.. note::
   Uniqueness is determined by the full identifier (namespace + name):
     
   - ``model.float_attr`` (key: ``"float_attr"``) and ``model.namespace_a.float_attr`` (key: ``"namespace_a:float_attr"``) can coexist
   - ``model.float_attr`` (key: ``"float_attr"``) and ``state.namespace_a.float_attr`` (key: ``"namespace_a:float_attr"``) can coexist
   - ``model.float_attr`` (key: ``"float_attr"``) and ``state.float_attr`` (key: ``"float_attr"``) cannot coexist - same key
   - ``model.namespace_a.float_attr`` and ``state.namespace_a.float_attr`` cannot coexist - same key ``"namespace_a:float_attr"``
   
**Registering Custom Attributes for a Solver:**

Before setting up the scene and loading assets, make sure to allow the solver you are using to register its custom attributes
in the :class:`newton.ModelBuilder` via the :meth:`newton.solvers.SolverBase.register_custom_attributes` method.

For example, to allow the MuJoCo solver to register its custom attributes, you can do:

.. testcode::

   from newton.solvers import SolverMuJoCo

   builder_mujoco = ModelBuilder()

   # First register the custom attributes for the MuJoCo solver
   SolverMuJoCo.register_custom_attributes(builder_mujoco)

   # Build a scene with a body and a shape
   body = builder_mujoco.add_link()
   joint = builder_mujoco.add_joint_free(body)
   builder_mujoco.add_articulation([joint])
   shape = builder_mujoco.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)

   # Finalize the model and allocate arrays for the custom attributes
   model_mujoco = builder_mujoco.finalize()

   # Now the model has the custom attributes registered by the MuJoCo solver
   # in the "mujoco" namespace.
   assert hasattr(model_mujoco, "mujoco")
   assert hasattr(model_mujoco.mujoco, "condim")
   assert np.allclose(model_mujoco.mujoco.condim.numpy(), [3])

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
* **JOINT_DOF frequency** → Values per degree of freedom (list, dict, or scalar for single-DOF joints)
* **JOINT_COORD frequency** → Values per position coordinate (list, dict, or scalar for single-coordinate joints)

For ``JOINT_DOF`` and ``JOINT_COORD`` frequencies, values can be provided in three formats:

1. **List format**: Explicit values for all DOFs/coordinates (e.g., ``[100.0, 200.0]`` for 2-DOF joint)
2. **Dict format**: Sparse specification mapping indices to values (e.g., ``{0: 100.0, 2: 300.0}`` sets only DOF 0 and 2)
3. **Scalar format**: Single value for single-DOF/single-coordinate joints, automatically expanded to a list

The following example demonstrates declaring and authoring attributes for each joint frequency type:

.. testcode::

   # Declare joint attributes with different frequencies
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="int_attr",
           frequency=ModelAttributeFrequency.JOINT,
           dtype=wp.int32
       )
   )
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="float_attr_dof",
           frequency=ModelAttributeFrequency.JOINT_DOF,
           dtype=wp.float32
       )
   )
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="float_attr_coord",
           frequency=ModelAttributeFrequency.JOINT_COORD,
           dtype=wp.float32
       )
   )
   
   # Create a D6 joint with 2 DOFs (1 linear + 1 angular) and 2 coordinates
   parent = builder.add_link(mass=1.0)
   child = builder.add_link(mass=1.0)
   
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
   builder.add_articulation([joint_id])
   
   # Scalar format for single-DOF joints (automatically expanded to list)
   parent2 = builder.add_link(mass=1.0)
   child2 = builder.add_link(mass=1.0)
   revolute_joint = builder.add_joint_revolute(
       parent=parent2,
       child=child2,
       axis=[0, 0, 1],
       custom_attributes={
           "float_attr_dof": 150.0,    # Scalar for 1-DOF joint (expanded to [150.0])
           "float_attr_coord": 0.8,    # Scalar for 1-coord joint (expanded to [0.8])
       }
   )
   builder.add_articulation([revolute_joint])
   
   # Dict format for sparse specification (only set specific DOF/coord indices)
   parent3 = builder.add_link(mass=1.0)
   child3 = builder.add_link(mass=1.0)
   d6_joint = builder.add_joint_d6(
       parent=parent3,
       child=child3,
       linear_axes=[cfg(axis=[1, 0, 0]), cfg(axis=[0, 1, 0])],  # 2 linear DOFs
       angular_axes=[cfg(axis=[0, 0, 1])],                      # 1 angular DOF
       custom_attributes={
           "float_attr_dof": {0: 100.0, 2: 300.0},  # Dict: only DOF 0 and 2 specified
       }
   )
   builder.add_articulation([d6_joint])

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

Custom String Frequencies
=========================

While enum frequencies (``BODY``, ``SHAPE``, ``JOINT``, etc.) cover most use cases, some solver-specific or user-defined data structures have counts that are independent of built-in entity types. Custom string frequencies enable these use cases.

Motivation
----------

Consider MuJoCo's ``<contact><pair>`` elements, which define explicit contact pairs between geometries with custom solver parameters. These pairs:

* Have their own count independent of bodies, shapes, or joints
* Reference shapes by index (which must be remapped when merging worlds)
* Need world assignment for multi-world simulations

A custom string frequency like ``"mujoco:pair"`` allows all pair-related attributes to share the same indexing scheme, with validation ensuring they stay synchronized.

Declaring Custom String Frequencies
-----------------------------------

To use a custom string frequency, pass a string instead of an enum value for the ``frequency`` parameter:

.. code-block:: python

   from newton import ModelBuilder, ModelAttributeFrequency
   import warp as wp
   
   builder = ModelBuilder()
   
   # Built-in enum frequency (standard pattern)
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="body_temp",
           frequency=ModelAttributeFrequency.BODY,  # Enum: one per body
           dtype=wp.float32,
       )
   )
   
   # Custom string frequency (for custom entity types)
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="item_value",
           frequency="item",  # String: custom entity type
           dtype=wp.float32,
           namespace="myns",  # Recommended for organization
       )
   )
   # → Frequency resolves to "myns:item" via namespace

**Namespace Resolution:** When a string frequency is used with a namespace, the ``frequency_key`` property automatically prepends the namespace, matching how attribute keys work. For example, ``frequency="item"`` with ``namespace="myns"`` resolves to ``"myns:item"``. This avoids redundancy and ensures consistency.

Adding Values with ``add_custom_values()``
------------------------------------------

Unlike enum frequencies where values are assigned during entity creation (``add_body``, ``add_shape``, etc.), custom string frequency values are appended using the :meth:`~newton.ModelBuilder.add_custom_values` method:

.. code-block:: python

   # Declare related attributes sharing the same frequency
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="item_id",
           frequency="item",
           dtype=wp.int32,
           namespace="myns",
       )
   )
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="item_value",
           frequency="item",
           dtype=wp.float32,
           default=1.0,
           namespace="myns",
       )
   )
   
   # Append values (all attributes with same frequency should be added together)
   builder.add_custom_values(**{
       "myns:item_id": 100,
       "myns:item_value": 2.5,
   })
   builder.add_custom_values(**{
       "myns:item_id": 101,
       "myns:item_value": 3.0,
   })
   
   model = builder.finalize()
   print(model.myns.item_id.numpy())    # [100, 101]
   print(model.myns.item_value.numpy()) # [2.5, 3.0]

The method returns a dict mapping attribute keys to the indices where values were added, which can be useful for building cross-references.

Validation at Finalize Time
---------------------------

A key benefit of custom string frequencies is automatic validation: all attributes sharing the same frequency must have the same count at ``finalize()`` time.

.. code-block:: python

   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(name="pair_a", frequency="pair", dtype=wp.int32, namespace="test")
   )
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(name="pair_b", frequency="pair", dtype=wp.int32, namespace="test")
   )
   
   # Add values to pair_a but not pair_b
   builder.add_custom_values(**{"test:pair_a": 1})
   builder.add_custom_values(**{"test:pair_a": 2})
   
   # This will raise ValueError at finalize():
   # "Custom attributes with frequency 'test:pair' have inconsistent counts:
   #  expected 2 (from test:pair_a), but 'test:pair_b' has 0 values."
   model = builder.finalize()  # Raises!

This validation prevents subtle bugs from mismatched array sizes, which would otherwise cause indexing errors or incorrect simulation behavior.

Multi-World Merging with ``references``
---------------------------------------

When using ``add_world()`` to create multi-world simulations, entity indices must be remapped. The ``references`` field specifies how attribute values should be transformed during merging.

.. code-block:: python

   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="pair_world",
           frequency="pair",
           dtype=wp.int32,
           namespace="mujoco",
           references="world",  # Replaced with current_world during merge
       )
   )
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="pair_shape1",
           frequency="pair",
           dtype=wp.int32,
           namespace="mujoco",
           references="shape",  # Offset by shape count during merge
       )
   )

**Supported reference types:**

* **Built-in entities**: ``"body"``, ``"shape"``, ``"joint"``, ``"joint_dof"``, ``"joint_coord"``, ``"articulation"`` — values are offset by the corresponding entity count
* **Special handling**: ``"world"`` — values are replaced with ``current_world`` (not offset)
* **Custom frequencies**: Any custom frequency key (e.g., ``"mujoco:pair"``) — values are offset by that frequency's count
* **Custom attributes**: Any attribute key (e.g., ``"mujoco:pair_data"``) — values are offset by that attribute's value count

**Example: Multi-world merging with contact pairs:**

.. code-block:: python

   # Template builder with one contact pair
   template = ModelBuilder()
   SolverMuJoCo.register_custom_attributes(template)
   template.add_mjcf("robot.xml")  # Has shapes 0,1 and pair(geom1=0, geom2=1)
   
   # Main builder merges two copies
   main = ModelBuilder()
   SolverMuJoCo.register_custom_attributes(main)
   main.add_world(template)  # world 0: shapes 0,1; pair(0,1)
   main.add_world(template)  # world 1: shapes 2,3; pair(2,3) <- indices offset!
   
   model = main.finalize()
   print(model.mujoco.pair_world.numpy())   # [0, 1]
   print(model.mujoco.pair_geom1.numpy())   # [0, 2]  <- offset by shape count
   print(model.mujoco.pair_geom2.numpy())   # [1, 3]  <- offset by shape count

Querying Custom Frequency Counts
--------------------------------

After finalization, use :meth:`~newton.Model.get_custom_frequency_count` to query the count for a custom frequency:

.. code-block:: python

   model = builder.finalize()
   pair_count = model.get_custom_frequency_count("mujoco:pair")
   print(f"Number of contact pairs: {pair_count}")

Selection and ArticulationView
------------------------------

Custom string frequency attributes are **not accessible** via :class:`~newton.ArticulationView`. This is by design: custom frequencies represent entity types that aren't inherently tied to articulation structure.

For example, contact pairs can span multiple articulations or involve non-articulated bodies, so there's no sensible way to slice them per-articulation. Attempting to access a custom frequency attribute through ArticulationView raises ``AttributeError`` with a clear message.

If you need per-articulation custom data, use a built-in enum frequency like ``ARTICULATION``, ``JOINT``, or ``BODY``.

Design Rationale
----------------

**Why strings instead of extending the enum?**

Using strings for custom frequencies provides several benefits:

1. **No core changes needed**: Solvers and users can define custom entity types without modifying Newton's core enums
2. **Namespacing**: The ``"namespace:entity"`` pattern naturally prevents conflicts between solvers
3. **Validation**: String frequencies enable the same-count validation that catches synchronization bugs
4. **Flexibility**: Any number of custom entity types can be defined without coordination

**Why require explicit ``add_custom_values()`` calls?**

Custom entity types don't map to existing builder methods (``add_body``, ``add_joint``, etc.), so a dedicated API is needed. The explicit call pattern ensures:

1. All related attributes are added together (reducing synchronization bugs)
2. The caller controls the indexing (important for building cross-references)
3. Clear separation from entity-based attribute assignment

**Why validate at ``finalize()`` time?**

Validating at finalize time rather than during ``add_custom_values()`` calls:

1. Allows flexible ordering (attributes can be added in any order)
2. Catches all synchronization bugs in one place
3. Provides clear error messages with full context

Summary: Enum vs String Frequencies
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Enum Frequency (``BODY``, ``SHAPE``, etc.)
     - String Frequency (``"mujoco:pair"``, etc.)
   * - Array size
     - Entity count (e.g., ``body_count``)
     - Number of ``add_custom_values()`` calls
   * - Index assignment
     - Implicit (entity creation order)
     - Explicit (append order)
   * - Value assignment
     - Via ``add_body(..., custom_attributes={})``, etc.
     - Via ``add_custom_values()``
   * - Multi-world merging
     - Automatic (entity offsets)
     - Via ``references`` field
   * - Validation
     - N/A (tied to entity count)
     - All same-frequency attrs must have same count
   * - ArticulationView
     - Supported
     - Not supported (not per-articulation)
   * - Use case
     - Per-entity properties
     - Custom entity types with independent counts

