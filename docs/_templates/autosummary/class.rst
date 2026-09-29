{{ fullname | escape | underline }}

.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
{% if fullname in ("newton.selection.DeformableCurveView", "newton.selection.DeformableSurfaceView", "newton.selection.DeformableVolumeView") %}   :inherited-members:
{% else %}   :show-inheritance:
{% endif %}   :member-order: groupwise
