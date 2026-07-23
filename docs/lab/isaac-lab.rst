.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Isaac Lab Integration
=====================

For details about Isaac Lab support for Newton, see the
`Isaac Lab documentation <https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html>`_.

Collision Updates During Substeps
---------------------------------

Isaac Lab controls Newton collision updates with
`NewtonCfg.collision_decimation <https://isaac-sim.github.io/IsaacLab/develop/source/api/lab_newton/isaaclab_newton.physics.html#isaaclab_newton.physics.NewtonCfg.collision_decimation>`_.
With multiple solver substeps, ``0`` runs collision detection once per physics
tick, ``1`` refreshes contacts between every substep, and ``N`` refreshes them
every N substeps. Values greater than or equal to ``num_substeps`` do not add a
mid-tick collision update.

When debugging contact chatter, penetration, or grasping, start with ``1``.
If that helps, increase the value and keep the largest interval that preserves
the required behavior. This setting belongs to Isaac Lab; in a standalone
Newton loop, control the cadence by choosing when to call
:meth:`~newton.CollisionPipeline.collide`. See
:ref:`collision-frequency-in-the-simulation-loop` for loop patterns.
