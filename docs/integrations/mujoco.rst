.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

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

Kinematic Links and Fixed Roots
-------------------------------

Newton only allows ``is_kinematic=True`` on articulation roots, so in the
MuJoCo exporter a "kinematic link" always means a kinematic root body. Any
descendants of that root can still be dynamic and are exported normally.

At runtime, :class:`~newton.solvers.SolverMuJoCo` keeps kinematic roots
user-prescribed rather than dynamically integrated:

- When converting MuJoCo state back to Newton, the previous Newton
  :attr:`newton.State.joint_q` and :attr:`newton.State.joint_qd` values are
  passed through for kinematic roots instead of being overwritten from MuJoCo's
  integrated ``qpos`` and ``qvel``.
- Applied body wrenches and joint forces targeting kinematic bodies are ignored
  on the MuJoCo side.
- Kinematic bodies still participate in contacts, so they can act as moving or
  fixed obstacles for dynamic bodies.

During export, :class:`~newton.solvers.SolverMuJoCo` maps roots according to
their joint type:

- **Kinematic roots with non-fixed joints** are exported as ordinary MuJoCo
  joints with the same Newton joint type and DOFs. The solver assigns a very
  large internal armature to those DOFs so MuJoCo treats them like prescribed,
  effectively infinite-mass coordinates.
- **Roots attached to world with a fixed joint** are exported as MuJoCo mocap
  bodies. This applies to both kinematic and non-kinematic Newton roots
  attached to world by :class:`~newton.JointType.FIXED`. MuJoCo has no joint
  coordinates for a fixed root, so Newton drives the pose through
  ``mjData.mocap_pos`` and ``mjData.mocap_quat`` instead.
- **World-attached shapes that are not part of an articulation** remain
  ordinary static MuJoCo geometry rather than mocap bodies.

If you edit :attr:`newton.Model.joint_X_p` or :attr:`newton.Model.joint_X_c`
for a fixed-root articulation after constructing the solver, call
``solver.notify_model_changed(newton.solvers.SolverNotifyFlags.JOINT_PROPERTIES)``
to synchronize the updated fixed-root poses into MuJoCo.
