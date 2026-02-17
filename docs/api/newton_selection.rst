newton.selection
================

.. currentmodule:: newton.selection

.. _label-matching:

Label Matching
--------------

Several Newton APIs accept **label patterns** to select bodies, shapes, joints,
etc. by name. Parameters that support label matching accept one of the following:

- A **list of integer indices** — selects directly by index.
- A **single string pattern** — selects all entries whose label matches the
  pattern via :func:`fnmatch.fnmatch` (supports ``*`` and ``?`` wildcards).
- A **list of string patterns** — selects all entries whose label matches at
  least one of the patterns.

Examples::

   # single pattern: all shapes whose label starts with "foot_"
   SensorIMU(model, sites="foot_*")

   # list of patterns: union of two groups
   SensorContact(model, sensing_obj_shapes=["*Plate*", "*Flap*"])

   # list of indices: explicit selection
   SensorFrameTransform(model, shapes=[0, 3, 7], reference_sites=[1])

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ArticulationView
