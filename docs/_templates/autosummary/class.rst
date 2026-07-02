{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: groupwise
{%- if fullname == "newton.sensors.SensorTiledCamera" %}
   :exclude-members: ClearData, GaussianRenderMode, RenderConfig, RenderLightType, RenderOrder
{%- endif %}
