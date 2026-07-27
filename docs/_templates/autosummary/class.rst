{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}
{% if fullname in ["newton.geometry.CameraSensor", "newton.sensors.CameraSensor"] %}   :no-index:
{% endif %}   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: groupwise
