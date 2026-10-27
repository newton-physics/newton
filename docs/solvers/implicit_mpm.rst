.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Implicit MPM Runtime Controls
=============================

:class:`~newton.solvers.SolverImplicitMPM` can change grid resolution during a
rollout without restarting particle motion. This supports coarse-to-fine
training schedules while retaining the MPM constitutive model and particle
state layout. The application owns the schedule; Newton owns grid rebuilding
and state preservation.

.. experimental::

   ``set_voxel_size()``, ``set_particle_radius()``, and the ``state_prev``
   argument to ``project_outside()`` may change without prior notice.

Changing grid resolution
------------------------

At a simulation-step boundary, discard any captured graphs that use the
solver, then rebuild around the current state:

.. code-block:: python

   graph = None
   solver.set_voxel_size(0.06, state=state)
   # Warm up and capture the enclosing simulation again before graph replay.

Both sparse and dense grids support this operation. Fixed grids reject it.
The solver preserves particle count, mass, radius, positions, velocities,
elastic strain, plastic volume, stress, and deformation frames. Collider
margins retain their values in meters. Grid-backed history is discarded
because it belongs to the previous discretization; particle history remains
available on the new grid. The existing native contact cache is used without
an additional cache mode.

Finer sparse grids may exceed the capacities selected in
:class:`~newton.solvers.SolverImplicitMPM.Config`. Reserve enough cells and
hierarchy nodes for the finest planned grid. If rebuilding fails, the solver
restores the previous voxel size and resources, and particle state is unchanged.
Resolution is uniform across the solver's worlds; this is not spatially
adaptive mesh refinement.

For a sufficiently sampled volume, coarser grids generally need fewer grid
unknowns. This does not guarantee a proportional speedup: particle work,
particles per cell, contact density, and solver convergence also contribute to
frame time. Measure a warmed-up rollout without rendering, at equal simulated
time and with task-specific accuracy checks. Coarser collision discretizations
also have fewer degrees of freedom near thin walls.

Changing particle support size
------------------------------

.. code-block:: python

   graph = None
   solver.set_particle_radius(0.016)
   # Recapture graphs before replaying them.

This operation updates every particle's support radius and the model's maximum
radius, then refreshes derived material data. Particle count, masses, motion,
and constitutive history remain unchanged. It does not split, merge, or respawn
particles. Newton derives support volume as ``8 * radius**3`` and density as
mass divided by that volume, so changing radius at fixed mass changes density.
This is a material-model change, not automatically a performance optimization.
Particle support size and numerical grid spacing are independent controls.

Swept collision projection
--------------------------

Endpoint projection can miss a thin wall when a particle crosses completely
through it in one step. Supplying the start state enables center-trajectory
queries:

.. code-block:: python

   # Preserve start.particle_q and start.body_q until projection completes.
   solver.step(start, end, control, contacts, dt)
   # Set end.body_q to the collider poses at the end of this same interval.
   solver.project_outside(end, end, dt, state_prev=start)

If integration overwrites its input positions, save those start positions and
collider poses in a separate state before stepping. The existing endpoint-only
call remains available when ``state_prev`` is omitted.

The sweep is exact for linear particle-center trajectories and translating
rigid meshes. Rotations use straight segments in collider space and still
require substeps for large angular motion. Up to four contacts and a final
endpoint projection are applied; exhausting that budget discards the remaining
displacement. Collider margins apply to the endpoint correction. Finite-radius
sweeps and deforming collider trajectories are unsupported.

Projection changes particle positions and velocities after the solve. It does
not add reaction impulses to rigid bodies or restore volume lost through the
correction. It complements a resolved contact solve; it is not a replacement
for validating timestep, contact accuracy, and volume behavior.

Moving-bowl example
-------------------

.. code-block:: console

   uv run --extra examples -m newton.examples mpm_bowl
   uv run --extra examples -m newton.examples mpm_bowl --resize-time 3 --resize-voxel-size 0.06

The CUDA example uses a sparse grid, swept projection, and a kinematic bowl.
Its controls change voxel size and particle support radius in place. The example
discards and recaptures its simulation graph after each change, using disposable
warm-up states so live particle and material state are preserved.

The new APIs schedule runtime operations and therefore do not require new USD
schemas. Existing particle-size authoring and initial solver configuration are
unchanged. Training curricula and success-rate decisions belong in applications
such as Isaac Lab.
