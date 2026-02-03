.. _extended_attributes:

Extended Attributes
===================

Newton's :class:`~newton.State` and :class:`~newton.Contacts` objects can optionally carry extra arrays that are not always needed.
These *extended attributes* are allocated on demand when explicitly requested, reducing memory usage for simulations that don't need them.

.. _extended_contact_attributes:

Extended Contact Attributes
---------------------------

Extended contact attributes are optional arrays on :class:`~newton.Contacts` (e.g., contact forces for sensors).
Request them via :meth:`Model.request_contact_attributes <newton.Model.request_contact_attributes>` or :meth:`ModelBuilder.request_contact_attributes <newton.ModelBuilder.request_contact_attributes>` before creating a :class:`~newton.Contacts` object.

.. code-block:: python

   builder = newton.ModelBuilder()
   # build/import model ...
   model = builder.finalize()

   sensor = newton.sensors.SensorContact(model, ...)  # transparently requests "force"
   contacts = newton.Contacts(
       rigid_contact_max,
       soft_contact_max,
       requested_attributes=model.get_requested_contact_attributes(),
   )

The canonical list is :attr:`Contacts.EXTENDED_ATTRIBUTES <newton.Contacts.EXTENDED_ATTRIBUTES>`:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Attribute
     - Description
   * - :attr:`~newton.Contacts.force`
     - Contact spatial forces (used by :class:`~newton.sensors.SensorContact`)


.. _extended_state_attributes:

Extended State Attributes
-------------------------

Extended state attributes are optional arrays on :class:`~newton.State` (e.g., accelerations for sensors).
Request them via :meth:`Model.request_state_attributes <newton.Model.request_state_attributes>` or :meth:`ModelBuilder.request_state_attributes <newton.ModelBuilder.request_state_attributes>` before calling :meth:`Model.state() <newton.Model.state>`.

.. code-block:: python

   builder = newton.ModelBuilder()
   # build/import model ...
   builder.request_state_attributes("body_qdd")
   model = builder.finalize()

   state = model.state()  # state.body_qdd is now allocated

The canonical list is :attr:`State.EXTENDED_ATTRIBUTES <newton.State.EXTENDED_ATTRIBUTES>`:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Attribute
     - Description
   * - :attr:`~newton.State.body_qdd`
     - Rigid-body spatial accelerations (used by :class:`~newton.sensors.SensorIMU`)
   * - :attr:`~newton.State.body_parent_f`
     - Rigid-body parent interaction wrenches


Notes
-----

- Some components transparently request attributes they need. For example, :class:`~newton.sensors.SensorIMU` requests ``body_qdd`` and :class:`~newton.sensors.SensorContact` requests ``force``.
  Create sensors before allocating State/Contacts for this to work automatically.
- Solvers populate extended attributes they support. Currently, :class:`~newton.solvers.SolverMuJoCo` populates ``body_qdd``, ``body_parent_f``, and ``force``.
