{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}
{% if fullname in ["newton.geometry.SensorCamera", "newton.sensors.SensorCamera"] %}   :no-index:
{% endif %}   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: groupwise
