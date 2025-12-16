Extended State Attributes
=========================

Newton’s :class:`~newton.State` can optionally carry extra arrays that are not always needed (e.g., accelerations for sensors).
These are called extended state attributes and are allocated by the :meth:`newton.Model.state` method, if they have been requested on the model or builder.

Allocation of State Attributes
------------------------------

- Core state attributes are allocated automatically based on what exists in the model (e.g., rigid bodies imply ``body_q``/``body_qd``).
- Extended state attributes are optional and allocated and computed only if you request them before calling :meth:`newton.Model.state`.
- You can request them either on the finalized model (:meth:`newton.Model.request_state_attributes`) or earlier on the builder (:meth:`newton.ModelBuilder.request_state_attributes`).
- Once an attribute has been requested, subsequent requests for the same attribute have no effect.

Example:

.. code-block:: python

   import newton

   builder = newton.ModelBuilder()
   # build/import model ...
   builder.request_state_attributes("body_qdd")  # can request on the builder
   model = builder.finalize()
   model.request_state_attributes("body_parent_f")  # can also request on the finalized model 

   state = model.state()  # state.body_qdd and state.body_parent_f are allocated


List of extended state attributes
---------------------------------

The following optional State attributes can currently be requested and allocated by :meth:`newton.Model.state`:

- ``body_qdd``: rigid-body spatial accelerations (used by :class:`newton.sensors.SensorIMU`)
- ``body_parent_f``: rigid-body parent interaction wrenches

Notes
-----

- Some components transparently request the attributes they need. For example, :class:`newton.sensors.SensorIMU` requires ``body_qdd`` and requests it from the model you pass in.
  For this to work, you must create the sensor before you allocate the State via :meth:`newton.Model.state`. See :ref:`sensorimu`.
- Solvers only populate optional outputs they explicitly support. When an extended state attribute is allocated on the State, a supporting solver will update it during its step().
  Today, :class:`newton.solvers.SolverMuJoCo` supports populating ``body_qdd`` and ``body_parent_f`` when those arrays are present on the State.


