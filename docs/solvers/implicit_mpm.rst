.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Implicit MPM Runtime Controls
=============================

:class:`~newton.solvers.SolverImplicitMPM` can change grid resolution during a
rollout while preserving particle motion and constitutive history. Applications
own coarse-to-fine schedules; Newton owns grid rebuilding and state preservation.

.. experimental::

   ``set_voxel_size()``, ``set_particle_radius()``, and the ``state_prev``
   argument to ``project_outside()`` may change without prior notice.

Changing grid resolution
------------------------

Call either setter between simulation steps and recapture any graphs using
the solver before replay:

.. code-block:: python

   graph = None
   solver.set_voxel_size(0.06, state=state)
   # Warm up and capture the enclosing simulation again before graph replay.

Both sparse and dense grids support this operation. Fixed grids reject it.
Particle count, mass, radius, positions, velocities, elastic strain, plastic
volume, stress, and deformation frames are preserved. Collider margins retain
their values in meters. Grid-backed history is discarded; particle history
remains available. The existing native contact cache needs no additional mode.

Finer sparse grids may exceed the capacities selected in
:class:`~newton.solvers.SolverImplicitMPM.Config`. Reserve enough cells and
hierarchy nodes for the finest planned grid. If rebuilding fails, the solver
restores the previous voxel size and resources, and particle state is unchanged.
Resolution is uniform across the solver's worlds, rather than spatially adaptive.

Coarser grids can reduce grid work but also have fewer collision degrees of
freedom near thin walls. Particle work and convergence still affect frame time;
measure warmed-up rollouts at equal simulated time with task-specific accuracy
checks before claiming a speedup.

Changing particle support size
------------------------------

:meth:`~newton.solvers.SolverImplicitMPM.set_particle_radius` updates every
particle's support radius and the model's maximum radius, independently of grid
spacing. It refreshes derived material data. Count, masses, motion, and history
remain unchanged; particles are not split, merged, or respawned. Support volume
is ``8 * radius**3`` and density is mass divided by that volume. Changing radius
at fixed mass therefore changes density, without inherently reducing cost.

Swept collision projection
--------------------------

Supply the start state to catch particle centers crossing completely through
thin walls:

.. code-block:: python

   # Preserve start.particle_q and start.body_q until projection completes.
   solver.step(start, end, control, contacts, dt)
   # Set end.body_q to the collider poses at the end of this same interval.
   solver.project_outside(end, end, dt, state_prev=start)

If integration overwrites its inputs, save start positions and collider poses
separately. Omitting ``state_prev`` retains endpoint-only projection.

The sweep is exact for linear particle-center trajectories and translating
rigid meshes. Rotations use straight segments in collider space and still
require substeps for large angular motion. Up to four contacts and a final
endpoint projection are applied; exhausting that budget discards the remaining
displacement. Collider margins apply to the endpoint correction. Finite-radius
sweeps and deforming collider trajectories are unsupported.

Projection corrects particle motion after the solve without reaction impulses
on rigid bodies or volume recovery. Continue validating timestep, contact
accuracy, and volume behavior.

Moving-bowl example
-------------------

.. code-block:: console

   uv run --extra examples -m newton.examples mpm_bowl
   uv run --extra examples -m newton.examples mpm_bowl --resize-time 3 --resize-voxel-size 0.06

The CUDA example combines a sparse grid, swept projection, and a kinematic bowl.
Its two sliders update voxel size and support radius, then recapture the graph
using disposable warm-up states to preserve live particle and material state.

The new APIs schedule runtime operations and therefore do not require new USD
schemas. Existing particle-size authoring and initial solver configuration are
unchanged. Training curricula and success-rate decisions belong in applications
such as Isaac Lab.
