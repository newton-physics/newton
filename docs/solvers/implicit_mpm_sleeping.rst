.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Implicit MPM sleeping
====================

.. experimental::

    Particle sleeping, bounded working sets, and topology caching are
    opt-in runtime execution controls and may change without prior notice.

For a large granular bed disturbed locally by a robot or tool,
:class:`~newton.solvers.SolverImplicitMPM` can omit quiet material from the
simulation grid. Its persistent device state retains every original particle
identity, material, position, velocity, and constitutive history. This reduces
the active solve; it does not remove the full particle backing allocation.

Configure sleeping before capturing the simulation:

The fully captured path requires CUDA, standard PIC, and positive bounded
grid capacities. It also requires a Warp runtime supporting persistent
allocations during conditional-body capture; stock Warp versions without that
support reject allocation nodes in these branches. Unsupported solver layouts
retain their ordinary stepping path.

.. code-block:: python

    config = SolverImplicitMPM.Config(
        grid_type="sparse", max_active_cell_count=16384,
        reuse_grid_topology=True,
    )
    solver = SolverImplicitMPM(model, config)
    solver.set_particle_sleeping(
        velocity_threshold=0.01, sleep_time=0.5,
        collider_lookahead=0.1, stress_filter_time=0.1,
    )

Three-dimensional activity patches wake around moving particles, applied
forces, and swept collider bounds. Velocity gradients and changing stress
retain activity locally. Quiet-time hysteresis and an optional planning
interval reduce membership churn; urgent collider, force, and fast-spill
requests remain checked every step. Generated rigid-collider bounds are
cached; mutable custom meshes retain their geometry checks. The solver-owned
device array :attr:`~newton.solvers.SolverImplicitMPM.particle_activity` uses
0 for sleeping, 1 for stationary support, and 2 for awake particles.

A surrounding support halo freezes nearby material while retaining its
history. These boundary conditions approximate settled material: they may
freeze creep or alter forces. Validate thresholds against a fully active
reference for the intended material. Rendering and surface reconstruction
continue to include sleeping material.

Bounded working sets and graph replay
------------------------------------

After initial settling and classification, call
:meth:`~newton.solvers.SolverImplicitMPM.set_particle_working_set` outside
capture to reserve a smaller standard-PIC evaluation pool. Original particle
IDs index material and history throughout compaction. The full state remains
on the GPU; only awake and support evaluations enter the bounded working set.
Changing capacities replaces solver buffers: release existing graphs, warm
up the new layout, and capture again. Ordinary membership changes reuse those
buffers and do not require outer graph recapture.

An optional ``page_size`` argument groups contiguous original particle IDs into
evaluation pages. For example, ``page_size=64`` scans a bitmap 64 times smaller
than the particle array and only rebuilds the page list when that bitmap changes.
Each lane's activity still refreshes every step, so a sleeping lane remains
excluded while another particle in its page is active. Capacities include padded
page lanes and must be divisible by the page size. Paging retains canonical
state in VRAM; it does not add out-of-core storage. Benchmark it against the
default dense compaction because unused page lanes can increase evaluation cost.
The device array ``particle_working_set_required_count`` reports the required
slot extent for explicit capacity diagnostics outside the steady loop.

Topology caching reuses unchanged grid partitions and solver structure.
Device occupancy checks rebuild dirty structure while particle assignments,
interpolation weights, material values, and contact values remain current.
The cache does not lower resolution, change the timestep, or activate extra
particles. It requires a supported grid, standard PIC, and no gradients.

The captured stepping path requires no device-to-host residency count.
The persistent :attr:`~newton.solvers.SolverImplicitMPM.step_status` device
array is zero for a valid rollout and nonzero after a capacity fault. On
overflow the solver preserves particle state instead of advancing a clipped
subset, and the failure remains sticky until an explicit reset. GPU consumers
can use the status directly. A ``None`` status identifies a legacy layout
that retains its normal host checks. Before consuming a rollout on the host, call
:meth:`~newton.solvers.SolverImplicitMPM.check_sparse_grid_rebuild_status` at
that boundary; this explicit check synchronizes. Increase insufficient
capacities outside capture and recapture before resuming. Merely clearing a
failure does not make an undersized buffer sufficient.

These controls manage runtime state and have no USD authoring representation.
They are disabled by default. For a fully active reference, omit
``set_particle_sleeping`` and ``set_particle_working_set`` while keeping the
material, geometry, resolution, and timestep unchanged.
