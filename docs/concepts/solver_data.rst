.. _SolverDataFields:

Solver Data Fields
==================

Overview
--------

While Newton's state object completely defines the time-varying state of the simulation, solvers may compute additional time-varying quantities
that are of interest for sensors, learning frameworks, or debugging. The ``SolverData`` class serves as a solver-agnostic interface and registry
for solvers to provide such simulation data, which may include accelerations, contact forces, constraint forces. Fields are organized by frequency
(``body_``, ``shape_``, ``joint_``, ``contact_``, etc.).

SolverData defines *generic* fields, which are specified in the SolverData class. It also supports *solver-specific* fields, which can
be registered through ``register_custom_field()``, enabling solvers to expose additional data that is not part of the generic SolverData API.

Solvers providing extra data implement ``get_data_fields()``, which returns the names of the data fields supported by the solver. These fields
can be requested via ``require_data()``, triggering their allocation. After the solver step, the data can be read from the attributes on ``Solver.data``.

For generic fields, all coordinate frames and transformations follow the :ref:`Twist conventions in Newton <Twist conventions>`, where applicable.


Frequency System
~~~~~~~~~~~~~~~~

Fields in SolverData are organized by frequency (like in the Selection API), indicating what simulation entity they are associated with:

- **articulation**: Per articulation data
- **body**: Per rigid body data
- **contact**: Per contact data
- **joint**: Per joint data
- **joint_dof**: Per joint degree of freedom
- **joint_coord**: Per joint coordinate
- **shape**: Per shape data (includes sites)

Field names have a prefix indicating their frequency followed by an underscore (e.g., ``body_acceleration``, ``contact_force_scalar``).

API Reference
-------------

Methods
~~~~~~~

.. py:method:: require_data(*fields: str) -> None

   Require the solver to provide specified data fields. Fields must be listed in ``get_data_fields()``.

   :param fields: Variable number of field names to require
   :raises TypeError: If a requested field is not supported by the solver
   :raises NotImplementedError: If the solver does not support the SolverData interface

.. py:method:: set_field_active(*fields, active=True)

   Activate or deactivate specified fields. Deactivated fields remain allocated but are not computed.

   :param fields: Variable number of field names to activate/deactivate
   :param active: Whether to activate (True) or deactivate (False) the fields
   :raises RuntimeError: If fields have not been required before activation/deactivation

.. py:method:: register_custom_field(field_name: str, field_frequency: str, field_type: type) -> None

   Register a custom solver-specific field. Intended to be called by solver implementations.

   :param field_name: Name of the custom field
   :param field_frequency: Frequency prefix for the field
   :param field_type: Data type of the field

.. py:method:: get_attribute_frequency(name: str) -> str

   Get the frequency of an attribute based on its name prefix.

   :param name: Name of the attribute
   :return: The frequency of the attribute
   :raises AttributeError: If the attribute frequency is not known

Properties
~~~~~~~~~~

.. py:property:: device

   Get the device used by the solver.

   :return: The ``wp.Device`` used by the solver

Integration with Solvers
------------------------

Solver Requirements
~~~~~~~~~~~~~~~~~~~

Solvers supporting the SolverData interface must:

1. Implement ``get_data_fields()`` to return supported fields and their sizes
2. Populate required and activated fields during ``step()``

Example Solver Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class MySolver(SolverBase):
        def get_data_fields(self) -> dict[str, int]:
            """Return supported data fields and their sizes."""
            return {
                "body_acceleration": self.model.body_count,
                "contact_force_scalar": self.max_contacts,
                # ... other supported fields
            }

        def step(self, state_in, state_out, control, contacts, dt):
            # Perform simulation step
            # ...

            # Populate required fields
            if self.data and "body_acceleration" in self.data.required_fields:
                if self.data.required_fields["body_acceleration"]:
                    # Compute and store body accelerations
                    compute_accelerations(self.data.body_acceleration)

Field Reference
---------------

Available Fields
~~~~~~~~~~~~~~~~

The following fields are currently defined in SolverData:

.. list-table:: SolverData Fields
   :header-rows: 1
   :widths: 30 15 55

   * - Field Name
     - Frequency
     - Description
   * - ``body_acceleration``
     - body
     - Linear and angular acceleration of the body COM in world frame (``wp.spatial_vector``)
   * - ``body_parent_joint_force``
     - body
     - Parent joint force and torque (``wp.spatial_vector``)
   * - ``contact_force_scalar``
     - contact
     - Magnitude of contact force (``float``)
   * - ``contact_force_vector_c``
     - contact
     - Contact force vector in contact frame (``wp.vec3f``)
   * - ``contact_torque_vector_c``
     - contact
     - Contact torque vector in contact frame (``wp.vec3f``)
   * - ``contact_frame_w``
     - contact
     - Unit vectors z and x defining the contact frame in world frame (``mat32``)

See Also
--------

- :class:`newton.SolverBase` - Base class for all solvers
- :ref:`Twist conventions` - Coordinate frame and transformation conventions
- ``newton/examples/example_solver_data.py`` - Example usage of SolverData
