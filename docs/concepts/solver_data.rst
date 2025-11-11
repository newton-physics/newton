.. _SolverDataFields:

Solver Data Fields
==================

Overview
--------

While Newton's ``State`` object completely defines the time-varying state of the simulation, solvers may compute additional time-varying quantities
that are of interest for sensors, learning frameworks, or debugging. The ``SolverData`` class serves as a solver-agnostic interface for solvers to provide such simulation data, which may include accelerations, contact forces, constraint forces. These are stored on ``State.data``. Fields are organized by frequency
(``body_``, ``shape_``, ``joint_``, ``contact_``, etc.).

The logic for tracking and allocating fields lives in ``SolverData``, whereas the ``Data`` class contains the fields definitions and allocations.

SolverData defines *generic* fields, which are specified in the ``SolverData`` class, and also supports *solver-specific* custom fields. Solvers
surface supported fields via two methods on ``SolverBase``:

- ``get_generic_data_fields() -> dict[str, int]``: Declares the generic fields supported by the solver and their sizes.
- ``get_custom_data_fields() -> list[CustomDataField]``: Declares solver-defined fields that are not part of the generic API.

Use the ``SolverBase.data_fields`` property to enumerate all supported field names. These fields can be requested via ``require_data()``.

The required fields must be allocated on each `State` object using ``Solver.allocate_data()``. Once written during the solver step, the
data can be read from the attributes on ``State.data``.
For generic fields, all coordinate frames and transformations follow the :ref:`Twist conventions in Newton <Twist conventions>`, where applicable.


Frequency System
~~~~~~~~~~~~~~~~

Fields in Data are organized by frequency (like in the Selection API), indicating what simulation entity they are associated with:

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

Usage
~~~~~

To access a data field, e.g. ``Data.body_acceleration``:

1. Require the field from the solver: ``solver.require_data("body_acceleration")``.
2. Register the field on the state: ``solver.allocate_data(state_0)``.
3. Access the data field: ``state_0.body_acceleration``.


Example Usage

.. testcode:: python

    import warp as wp
    import newton

    # create model and instantiate solver
    builder = newton.ModelBuilder()
    sphere = builder.add_body()
    builder.add_shape_sphere(sphere, radius=0.2)
    builder.add_joint_free(sphere)

    model = builder.finalize()
    solver = newton.solvers.SolverMuJoCo(model)

    solver.require_data("body_acceleration")  # require field from solver

    state = model.state()
    solver.allocate_data(state)  # allocate field on state


Methods
~~~~~~~

.. py:method:: require_data(*fields: str) -> None

   Require the solver to provide specified data fields. Fields must be listed in ``SolverBase.data_fields``.

   :param fields: Variable number of field names to require
   :raises TypeError: If a requested field is not supported by the solver
   :raises NotImplementedError: If the solver does not support the SolverData interface

.. py:method:: set_field_active(*fields, active=True)

   Activate or deactivate specified fields. Deactivated fields remain allocated but are not computed.

   :param fields: Variable number of field names to activate/deactivate
   :param active: Whether to activate (True) or deactivate (False) the fields
   :raises RuntimeError: If fields have not been required before activation/deactivation


Integration with Solvers
------------------------

Solver Requirements
~~~~~~~~~~~~~~~~~~~

Solvers supporting the SolverData interface must:

1. Implement ``get_generic_data_fields()`` to return supported generic fields and their sizes.
2. Implement ``get_custom_data_fields()`` to return supported custom fields as ``CustomDataField`` entries.
3. Populate required and activated fields on `State.data` (as provided in ``data.required_fields``) during ``step()``.

Example Solver Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode:: python

    import warp as wp
    from newton.solvers import CustomDataField, SolverBase

    class MySolver(SolverBase):
        def get_generic_data_fields(self) -> dict[str, int]:
            """Return supported generic data fields and their sizes."""
            return {
                "body_acceleration": self.model.body_count,
                "contact_force_scalar": self.max_contacts,
                # ... other generic fields
            }

        def get_custom_data_fields(self) -> list[CustomDataField]:
            """Return solver-specific fields that are not part of the generic API."""
            return [
                CustomDataField(
                    name="body_my_metric",
                    frequency="body",
                    field_type=wp.array(dtype=float),
                    size=self.model.body_count,
                    namespace="my_solver",
                )
            ]

        def step(self, state_in, state_out, control, contacts, dt):
            # Perform simulation step
            # ...

            # Populate required fields
            if state_out.data and self.data.required_fields.get("body_acceleration", False):
                compute_accelerations(state_out.data.body_acceleration)

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
- ``SolverBase.data_fields`` - Property exposing all supported field names
- :ref:`Twist conventions` - Coordinate frame and transformation conventions
- ``newton/examples/example_solver_data.py`` - Example usage of SolverData
