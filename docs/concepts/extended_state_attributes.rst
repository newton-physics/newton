.. _extended_state_attributes:

Extended State Attributes
=========================

Newton’s :class:`~newton.State` object can optionally carry extra arrays that are not always needed (e.g., accelerations for sensors).
These are called *extended state attributes* and are allocated by the :meth:`Model.state <newton.Model.state>` method, if they have been previously requested on the Model or ModelBuilder.

Allocation of State Attributes
------------------------------

- Core state attributes are allocated automatically based on what exists in the model (for example, rigid bodies imply :attr:`~newton.State.body_q`/:attr:`~newton.State.body_qd`).
- Extended state attributes are allocated and computed only if you request them before calling :meth:`Model.state() <newton.Model.state>`.
- You can request them either on the finalized model (:meth:`Model.request_state_attributes <newton.Model.request_state_attributes>`) or earlier on the builder (:meth:`ModelBuilder.request_state_attributes <newton.ModelBuilder.request_state_attributes>`).
- Once an attribute has been requested, subsequent requests for the same attribute have no effect.

Example:

.. code-block:: python

  builder = newton.ModelBuilder()
  # build/import model ...
  builder.request_state_attributes("body_qdd")  # can request on the builder
  model = builder.finalize()
  model.request_state_attributes("body_parent_f")  # can also request on the finalized model 

  state = model.state()  # state.body_qdd and state.body_parent_f are allocated


List of extended state attributes
---------------------------------

The canonical list of requestable extended state attribute *names* is :attr:`State.EXTENDED_STATE_ATTRIBUTES <newton.State.EXTENDED_STATE_ATTRIBUTES>`.

The following optional State attributes can currently be requested and allocated by :meth:`Model.state() <newton.Model.state>`:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Attribute name
     - Description
   * - :attr:`body_qdd <newton.State.body_qdd>`
     - Rigid-body spatial accelerations (used by :class:`~newton.sensors.SensorIMU`)
   * - :attr:`body_parent_f <newton.State.body_parent_f>`
     - Rigid-body parent interaction wrenches

Notes
-----

- Some components transparently request the attributes they need. For example, :class:`~newton.sensors.SensorIMU` requires ``body_qdd`` and requests it from the model you pass in.
  For this to work, you must create the sensor before you allocate the State via :meth:`Model.state() <newton.Model.state>`. See :ref:`sensorimu`.
- Solvers only populate optional outputs they explicitly support. When an extended state attribute is allocated on the State, a supporting solver will update it during its :meth:`~newton.solvers.SolverBase.step` method.
  Currently, only :class:`~newton.solvers.SolverMuJoCo` supports populating ``body_qdd`` and ``body_parent_f``.
