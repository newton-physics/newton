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
* **Solver-specific data**: Store contact pair parameters, tendon properties, or other solver-specific entity types

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

Custom attributes must be declared before use via the :meth:`newton.ModelBuilder.add_custom_attribute` method. Each declaration specifies:

* **name**: Attribute name
* **frequency**: Determines array size and indexing—either a :class:`~newton.Model.AttributeFrequency` enum value (``BODY``, ``SHAPE``, ``JOINT``, ``JOINT_DOF``, ``JOINT_COORD``, ``ARTICULATION``) or a string for custom frequencies
* **dtype**: Warp data type (``wp.float32``, ``wp.vec3``, ``wp.quat``, etc.)
* **assignment**: Which simulation object owns the attribute (``MODEL``, ``STATE``, ``CONTROL``, ``CONTACT``)
* **default** (optional): Default value for unspecified entities
* **namespace** (optional): Hierarchical organization for grouping related attributes
* **references** (optional): For multi-world merging, specifies how values are transformed (e.g., ``"body"``, ``"shape"``, ``"world"``, or a custom frequency key)

When **no namespace** is specified, attributes are added directly to their assignment object (e.g., ``model.temperature``). When a **namespace** is provided, Newton creates a namespace container (e.g., ``model.mujoco.damping``).

.. testcode::

   from newton import Model, ModelBuilder
   import warp as wp
   
   builder = ModelBuilder()
   
   # Default namespace attributes - added directly to assignment objects
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="temperature",
           frequency=Model.AttributeFrequency.BODY,
           dtype=wp.float32,
           default=20.0,  # Explicit default value
           assignment=Model.AttributeAssignment.MODEL
       )
   )
   # → Accessible as: model.temperature
   
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="velocity_limit",
           frequency=Model.AttributeFrequency.BODY,
           dtype=wp.vec3,
           default=(1.0, 1.0, 1.0),  # Default vector value
           assignment=Model.AttributeAssignment.STATE
       )
   )
   # → Accessible as: state.velocity_limit
   
   # Namespaced attributes - organized under namespace containers
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="float_attr",
           frequency=Model.AttributeFrequency.BODY,
           dtype=wp.float32,
           default=0.5,
           assignment=Model.AttributeAssignment.MODEL,
           namespace="namespace_a"
       )
   )
   # → Accessible as: model.namespace_a.float_attr
   
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="bool_attr",
           frequency=Model.AttributeFrequency.SHAPE,
           dtype=wp.bool,
           default=False,
           assignment=Model.AttributeAssignment.MODEL,
           namespace="namespace_a"
       )
   )
   # → Accessible as: model.namespace_a.bool_attr
   
   # Articulation frequency attributes - one value per articulation
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="articulation_stiffness",
           frequency=Model.AttributeFrequency.ARTICULATION,
           dtype=wp.float32,
           default=100.0,
           assignment=Model.AttributeAssignment.MODEL
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
   
   # Articulation attributes: create articulations with custom values
   # Each add_articulation creates one articulation at the next index
   for i in range(3):
       base = builder.add_link(mass=1.0)
       joint = builder.add_joint_free(child=base)
       builder.add_articulation(
           joints=[joint],
           custom_attributes={
               "articulation_stiffness": 100.0 + float(i) * 50.0  # 100, 150, 200
           }
       )
   
   # After finalization, access attributes
   model = builder.finalize()
   temps = model.temperature.numpy()
   arctic_stiff = model.articulation_stiffness.numpy()
   
   print(f"Body 1: {temps[body1]}")  # 20.0 (default)
   print(f"Body 2: {temps[body2]}")  # 37.5 (authored)
   # Articulation indices reflect all articulations in the model
   # (including any implicit ones from add_body)
   print(f"Articulations: {len(arctic_stiff)}")
   print(f"Last articulation stiffness: {arctic_stiff[-1]}")  # 200.0

.. testoutput::

   Body 1: 20.0
   Body 2: 37.5
   Articulations: 5
   Last articulation stiffness: 200.0

.. note::
   Uniqueness is determined by the full identifier (namespace + name):
     
   - ``model.float_attr`` (key: ``"float_attr"``) and ``model.namespace_a.float_attr`` (key: ``"namespace_a:float_attr"``) can coexist
   - ``model.float_attr`` (key: ``"float_attr"``) and ``state.namespace_a.float_attr`` (key: ``"namespace_a:float_attr"``) can coexist
   - ``model.float_attr`` (key: ``"float_attr"``) and ``state.float_attr`` (key: ``"float_attr"``) cannot coexist - same key
   - ``model.namespace_a.float_attr`` and ``state.namespace_a.float_attr`` cannot coexist - same key ``"namespace_a:float_attr"``

**Registering Solver Attributes:**

Before loading assets, register solver-specific attributes:

.. testcode::

   from newton.solvers import SolverMuJoCo

   builder_mujoco = ModelBuilder()
   SolverMuJoCo.register_custom_attributes(builder_mujoco)
   
   # Now build your scene...
   body = builder_mujoco.add_link()
   joint = builder_mujoco.add_joint_free(body)
   builder_mujoco.add_articulation([joint])
   shape = builder_mujoco.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
   
   model_mujoco = builder_mujoco.finalize()
   assert hasattr(model_mujoco, "mujoco")
   assert hasattr(model_mujoco.mujoco, "condim")

Authoring Custom Attributes
----------------------------

After declaration, values are assigned through the standard entity creation API (``add_body``, ``add_shape``, ``add_joint``). For default namespace attributes, use the attribute name directly. For namespaced attributes, use the format ``"namespace:attr_name"``.

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

**Joint Frequency Types:**

For joints, Newton provides three frequency types to store different granularities of data:

* **JOINT frequency** → One value per joint
* **JOINT_DOF frequency** → Values per degree of freedom (list, dict, or scalar for single-DOF joints)
* **JOINT_COORD frequency** → Values per position coordinate (list, dict, or scalar for single-coordinate joints)

For ``JOINT_DOF`` and ``JOINT_COORD`` frequencies, values can be provided in three formats:

1. **List format**: Explicit values for all DOFs/coordinates (e.g., ``[100.0, 200.0]`` for 2-DOF joint)
2. **Dict format**: Sparse specification mapping indices to values (e.g., ``{0: 100.0, 2: 300.0}`` sets only DOF 0 and 2)
3. **Scalar format**: Single value for single-DOF/single-coordinate joints, automatically expanded to a list

.. testcode::

   # Declare joint attributes with different frequencies
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="int_attr",
           frequency=Model.AttributeFrequency.JOINT,
           dtype=wp.int32
       )
   )
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="float_attr_dof",
           frequency=Model.AttributeFrequency.JOINT_DOF,
           dtype=wp.float32
       )
   )
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="float_attr_coord",
           frequency=Model.AttributeFrequency.JOINT_COORD,
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
* The same attribute name can exist in different namespaces because they create different full identifiers

Custom Frequencies
==================

While enum frequencies (``BODY``, ``SHAPE``, ``JOINT``, etc.) cover most use cases, some data structures have counts independent of built-in entity types. Custom frequencies address this by allowing a string instead of an enum for the ``frequency`` parameter.

**Example use case:** MuJoCo's ``<contact><pair>`` elements define contact pairs between geometries. These pairs have their own count independent of bodies or shapes, and their indices must be remapped when merging worlds.

Declaring Custom Frequencies
----------------------------

Pass a string instead of an enum for the ``frequency`` parameter:

.. code-block:: python

   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(
           name="pair_geom1",
           frequency="pair",  # Custom frequency (string)
           dtype=wp.int32,
           namespace="mujoco",
       )
   )
   # → Frequency resolves to "mujoco:pair" via namespace

When a namespace is provided, it is automatically prepended to the frequency string, matching how attribute keys work.

Adding Values
-------------

Custom frequency values are appended using :meth:`~newton.ModelBuilder.add_custom_values`:

.. code-block:: python

   # Declare attributes sharing a frequency
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(name="item_id", frequency="item", dtype=wp.int32, namespace="myns")
   )
   builder.add_custom_attribute(
       ModelBuilder.CustomAttribute(name="item_value", frequency="item", dtype=wp.float32, namespace="myns")
   )
   
   # Append values together
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

**Validation:** All attributes sharing a custom frequency must have the same count at ``finalize()`` time. This catches synchronization bugs early.

Multi-World Merging
-------------------

When using ``add_world()`` to create multi-world simulations, the ``references`` field specifies how attribute values should be transformed:

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
           name="pair_geom1",
           frequency="pair",
           dtype=wp.int32,
           namespace="mujoco",
           references="shape",  # Offset by shape count during merge
       )
   )

Supported reference types:

* ``"body"``, ``"shape"``, ``"joint"``, ``"joint_dof"``, ``"joint_coord"``, ``"articulation"`` — offset by entity count
* ``"world"`` — replaced with ``current_world``
* Custom frequency keys (e.g., ``"mujoco:pair"``) — offset by that frequency's count

Querying Counts
---------------

Use :meth:`~newton.Model.get_custom_frequency_count` to get the count for a custom frequency (raises ``KeyError`` if unknown):

.. code-block:: python

   model = builder.finalize()
   pair_count = model.get_custom_frequency_count("mujoco:pair")
   
   # Or check directly without raising:
   pair_count = model.custom_frequency_counts.get("mujoco:pair", 0)

.. note::
   When querying, use the **resolved** frequency key with namespace prefix (e.g., ``"mujoco:pair"``), not the raw string used in the declaration (``"pair"``). This matches how attribute keys work: ``model.get_attribute_frequency("mujoco:condim")`` for a namespaced attribute.

ArticulationView Limitations
----------------------------

Custom frequency attributes are not accessible via :class:`~newton.ArticulationView` because they represent entity types that aren't tied to articulation structure. For per-articulation data, use enum frequencies like ``ARTICULATION``, ``JOINT``, or ``BODY``.
