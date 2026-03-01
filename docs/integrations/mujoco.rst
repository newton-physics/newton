MuJoCo Integration
==================

:class:`~newton.solvers.SolverMuJoCo` translates a Newton :class:`~newton.Model`
into a `MuJoCo <https://github.com/google-deepmind/mujoco>`_ model and steps it
with `mujoco_warp <https://github.com/google-deepmind/mujoco_warp>`_.
Because MuJoCo has its own modelling conventions, some Newton properties are
mapped differently or not at all.

Caveats
-------

**geom_gap is always zero.**
  MuJoCo's ``gap`` parameter controls *inactive* contact generation — contacts
  that are detected but do not produce constraint forces until penetration
  exceeds the gap threshold.  Newton does not use this concept: when the MuJoCo
  collision pipeline is active it runs every step, so there is no benefit to
  keeping inactive contacts around.  Setting ``geom_gap > 0`` would inflate
  ``geom_margin``, which disables MuJoCo's multicontact and degrades contact
  quality.  Therefore :class:`~newton.solvers.SolverMuJoCo` always sets
  ``geom_gap = 0`` regardless of the Newton :attr:`~newton.ModelBuilder.ShapeConfig.gap`
  value.  MJCF/USD ``gap`` values are still imported into
  :attr:`~newton.ModelBuilder.ShapeConfig.gap` in the Newton model, but they are
  not forwarded to the MuJoCo solver.

**shape_collision_radius is ignored.**
  MuJoCo computes bounding-sphere radii (``geom_rbound``) internally from the
  geometry definition.  Newton's ``shape_collision_radius`` is not forwarded.
