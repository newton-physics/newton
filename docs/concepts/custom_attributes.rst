Custom Attributes
=================

Newton's simulation model uses flat buffer arrays to represent physical properties and simulation state. 
These arrays can be extended with user-defined custom attributes to store application-specific data alongside the standard physics quantities.
In the following, you will learn how to declare, author, and access custom attributes in Newton through Python API as well as how to author/access them from USD files.


Newton organizes simulation data into three primary objects, each containing flat arrays indexed by simulation entities: 

**Model Object**
   Static configuration and physical properties that remain constant during simulation:
   
   * ``body_mass`` - per-body mass values
   * ``shape_material_mu`` - per-shape friction coefficients
   * ``joint_limit_ke`` - per-DOF limit stiffness values

**State Object**
   Dynamic quantities that evolve during simulation:
   
   * ``body_q`` - per-body positions and orientations
   * ``body_qd`` - per-body spatial velocities
   * ``joint_q`` - per-coordinate joint positions
   * ``joint_qd`` - per-DOF joint velocities

**Control Object**
   Control inputs and actuator commands:
   
   * ``joint_act`` - per-DOF actuator forces/torques

Custom attributes extend these objects with user-defined arrays that follow the same indexing scheme as Newton's built-in attributes.

Declaring Custom Attributes
----------------------------

Custom attributes must be declared before use. Each declaration specifies three required properties: frequency, assignment, and data type.

The **frequency** determines the array size and indexing pattern (``BODY``, ``SHAPE``, ``JOINT``, ``JOINT_DOF``, or ``JOINT_COORD``). The **assignment** determines which simulation object owns the attribute (``MODEL`` for static properties, ``STATE`` for dynamic quantities, ``CONTROL`` for control inputs, or ``CONTACT`` for contact-specific data). The **dtype** specifies the Warp data type such as ``wp.float32``, ``wp.vec3``, ``wp.quat``, or other Warp types.

The following example shows how to declare custom attributes with different frequencies and assignments:

.. code-block:: python

   from newton import ModelBuilder, ModelAttributeFrequency, ModelAttributeAssignment
   import warp as wp
   
   builder = ModelBuilder()
   
   # Declare a MODEL attribute with BODY frequency
   builder.add_custom_attribute(
       name="temperature",
       frequency=ModelAttributeFrequency.BODY,
       dtype=wp.float32,
       assignment=ModelAttributeAssignment.MODEL
   )
   
   # Declare a STATE attribute with BODY frequency
   builder.add_custom_attribute(
       name="velocity_limit",
       frequency=ModelAttributeFrequency.BODY,
       dtype=wp.vec3,
       assignment=ModelAttributeAssignment.STATE
   )
   
   # Declare a CONTROL attribute with JOINT_DOF frequency
   builder.add_custom_attribute(
       name="gain",
       frequency=ModelAttributeFrequency.JOINT_DOF,
       dtype=wp.float32,
       assignment=ModelAttributeAssignment.CONTROL
   )

.. note::
   Custom attribute names must be unique across all assignments. The same name cannot be used for both MODEL and STATE attributes.

Authoring Custom Attributes
----------------------------

After declaration, custom attributes are authored through the standard entity creation API (``add_body``, ``add_shape``, ``add_joint``, etc.). Values are provided in a dictionary structure grouped by assignment.

For the attributes declared above, the following shows how to assign values when creating bodies:

.. code-block:: python

   # Add a body with custom attributes
   body_id = builder.add_body(
       mass=1.0,
       custom_attributes={
           ModelAttributeAssignment.MODEL: {
               "temperature": 37.5,
           },
           ModelAttributeAssignment.STATE: {
               "velocity_limit": [2.0, 2.0, 2.0],
           }
       }
   )

For joints, attribute names can use prefixes to specify different frequencies. Attributes without a prefix use JOINT frequency (one value per joint), while attributes with a ``dof_`` prefix use JOINT_DOF frequency (requiring a list of values, one per DOF), and attributes with a ``coord_`` prefix use JOINT_COORD frequency (requiring a list of values, one per coordinate).

The following shows how to declare and author joint attributes with these different frequencies:

.. code-block:: python

   # Declare joint attributes with different frequencies
   builder.add_custom_attribute(
       "joint_type", 
       ModelAttributeFrequency.JOINT, 
       dtype=wp.int32,
       assignment=ModelAttributeAssignment.MODEL
   )
   builder.add_custom_attribute(
       "dof_stiffness", 
       ModelAttributeFrequency.JOINT_DOF, 
       dtype=wp.float32,
       assignment=ModelAttributeAssignment.MODEL
   )
   builder.add_custom_attribute(
       "coord_offset", 
       ModelAttributeFrequency.JOINT_COORD, 
       dtype=wp.float32,
       assignment=ModelAttributeAssignment.MODEL
   )

After declaring these joint attributes, values can be assigned when creating joints:

.. code-block:: python

   # Author joint attributes
   builder.add_joint_d6(
       parent=parent_body,
       child=child_body,
       linear_axes=[...],
       angular_axes=[...],
       custom_attributes={
           ModelAttributeAssignment.MODEL: {
               "joint_type": 2,
               "dof_stiffness": [100.0, 150.0, 200.0],  # Three DOFs
               "coord_offset": [0.1, 0.2, 0.3],         # Three coordinates
           }
       }
   )

Accessing Custom Attributes
----------------------------

After authoring custom attributes on entities, they become accessible as arrays on their assigned objects after finalization.

Using the same attributes declared and authored above (``temperature`` and ``velocity_limit``), the following demonstrates how to access the data:

.. code-block:: python

   # Build the model
   model = builder.finalize()
   state = model.state()
   
   # Access MODEL attributes
   temperatures = model.temperature.numpy()
   print(f"Body temperature: {temperatures[body_id]}")
   
   # Access STATE attributes  
   velocity_limits = state.velocity_limit.numpy()
   print(f"Velocity limit: {velocity_limits[body_id]}")

Custom attributes follow the same GPU/CPU synchronization rules as built-in attributes and can be modified during simulation.

USD Integration and Default Values
-----------------------------------

Custom attributes can be authored directly in USD files using Newton's naming convention. The USD parser automatically discovers and integrates these attributes during import. For more information about USD integration and the schema resolver system for custom attributes, see the Custom Attribute Framework section in :doc:`usd_parsing`.

The following USD file demonstrates how to author custom attributes for bodies and joints:

.. code-block:: usda

   #usda 1.0
   
   def Xform "robot_arm" (
       prepend apiSchemas = ["PhysicsRigidBodyAPI"]
   ) {
       # Model assignment - static properties
       float newton:model:body:thermal_capacity = 850.0
       int newton:model:body:component_id = 42
       bool newton:model:body:has_sensor = true
       
       # State assignment - dynamic quantities
       float3 newton:state:body:target_position = (1.0, 0.5, 0.3)
       float newton:state:body:energy_level = 100.0
   }
   
   def RevoluteJoint "elbow" {
       # Joint model properties
       float newton:model:joint:gear_ratio = 2.5
       
       # Control assignment  
       float newton:control:joint:max_effort = 50.0
   }

After authoring custom attributes in USD, they can be imported and accessed as shown below:

.. code-block:: python

   from newton import ModelBuilder
   from newton._src.utils.import_usd import parse_usd
   
   builder = ModelBuilder()
   parse_usd(builder, source="robot_arm.usda")
   
   model = builder.finalize()
   state = model.state()
   control = model.control()
   
   # Custom attributes are automatically available
   thermal_capacity = model.thermal_capacity.numpy()
   target_positions = state.target_position.numpy()
   max_efforts = control.max_effort.numpy()

Custom attributes use default values for entities that don't explicitly specify values. When declaring an attribute, users can provide a ``default`` parameter. If not specified, dtype-specific defaults are used: 0.0 for floats, 0 for integers, False for booleans, and zero vectors for vector types. The following demonstrates this behavior:

.. code-block:: python

   # Declare with explicit default
   builder.add_custom_attribute(
       name="temperature",
       frequency=ModelAttributeFrequency.BODY,
       dtype=wp.float32,
       default=20.0,
       assignment=ModelAttributeAssignment.MODEL
   )
   
   body1 = builder.add_body(mass=1.0)  # Uses default: 20.0
   
   body2 = builder.add_body(
       mass=1.0,
       custom_attributes={
           ModelAttributeAssignment.MODEL: {"temperature": 65.0}  # Override default
       }
   )

After creating bodies with and without explicit values, the arrays reflect both authored and default values:

.. code-block:: python
   
   model = builder.finalize()
   temps = model.temperature.numpy()
   # temps[body1] = 20.0 (default)
   # temps[body2] = 65.0 (authored)

Validation and Constraints
---------------------------

The custom attribute system enforces several constraints to ensure correctness. Attributes must be declared via ``add_custom_attribute()`` before use, otherwise an ``AttributeError`` is raised. Each attribute must be used with entities matching its declared frequency (e.g., a BODY-frequency attribute cannot be used with shapes) and with its declared assignment (e.g., a STATE-assigned attribute cannot be authored in the MODEL assignment dictionary). Violations of frequency or assignment constraints raise ``ValueError``. Additionally, each attribute name must be unique across all assignments—the same name cannot be declared for both MODEL and STATE assignments.

Use Cases
---------

Custom attributes enable a wide range of simulation extensions. 
They can store per-body thermal properties, shape material composition that affect simulation behaviors. 
For hardware-in-the-loop simulation, custom attributes can tag bodies and joints with sensor IDs, actuator types, or hardware specifications. 
Custom controllers can store more advanced control parameters such as gains, velocity limits, or control modes per-joint. 
Visualization pipelines can attach colors, labels, or rendering properties to simulation entities. 
For multi-physics coupling, custom attributes can store quantities such as surface stress for fluid simulations that interact with rigid bodies. 
In reinforcement learning applications, observation buffers, reward weights, or optimization parameters can be stored directly on simulation entities which will simplify the indexing and access of these data on the learning side.

