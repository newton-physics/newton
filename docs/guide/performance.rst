.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _performance-guide:

Performance
===========

This guide describes how to measure and improve simulation performance, from
CUDA graph capture and host synchronization to collision detection and solver
settings. For custom Warp kernels, see Warp's `performance guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance.html>`__.

Establish the required simulation behavior before optimizing (see
:ref:`Simulation Tuning`). Execution changes should preserve that behavior;
changes to geometry, contact models, or solver settings need an accuracy check.

Measuring Performance
---------------------

Use a representative scene and choose an accuracy metric and tolerance before
comparing configurations. Check that the state contains no NaN or Inf values
and that contact and constraint buffers do not overflow (see
:ref:`performance-buffer-capacity`). Compare against a reference using a
physical quantity such as penetration depth, force balance, or body trajectory.
For solver tuning, increase substeps or iterations until the reference metric
converges.

Measure a baseline, then profile it to identify the dominant cost. Common
bottlenecks include kernel launch overhead, host synchronization, collision
detection, solver work, buffer operations, and rendering. Their relative cost
depends on the scene and world count. Change one setting at a time and record
both timing and accuracy.

.. _performance-benchmarking:

Benchmarking
^^^^^^^^^^^^

The examples accept ``--benchmark``, which uses a null viewer, warms up, and
reports frames per second. An optional argument limits the run to that many
seconds; it also stops after ``--num-frames`` frames (100 by default):

.. code-block:: console

    uv run -m newton.examples basic_shapes --benchmark 10 --num-frames 1000

When timing your own code, warm up the simulation and replay any captured graph
at least once before starting the timer. Synchronize the device before and
after the timed region so that the measurement includes GPU execution. Time
several frames together to reduce measurement overhead. Report setup time,
including model construction and kernel compilation, separately from stepping.

Record the hardware and software versions, world count, timestep, substeps,
solver settings, and collision frequency. State whether rendering, sensors,
controls, and resets are included. Compare configurations over the same amount
of simulated time and report the accuracy metric alongside the timing. To track
performance changes in Newton itself, use the airspeed velocity suite described
in the :doc:`development guide </guide/development>`.

.. _performance-profiling:

Profiling
---------

Profile startup and steady-state stepping separately to distinguish setup costs
from work that repeats every frame. This section covers what is specific to
Newton; for the profiling tools themselves, see Warp's `profiling guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html>`__.

Startup
^^^^^^^

Startup covers importing modules, building and finalizing the model, compiling
and loading kernels, allocating buffers, and capturing graphs. Use Python's
:mod:`cProfile` to identify expensive host calls, or time each phase with device
synchronization at both boundaries to include GPU work. Warp prints module load
and compile times. Later runs reuse compiled kernels from the kernel cache, so
the first run after a code or option change is slower. To compile kernels
before latency-sensitive work, or to ship them precompiled, see Warp's guide to
`reducing compilation and startup time
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/reducing_compilation_and_startup_time.html>`__.

Stepping
^^^^^^^^

How you profile the step depends on whether it runs eagerly or as a captured
graph.

When the step runs eagerly, :class:`warp.ScopedTimer` with ``synchronize=True``
measures a region, and its ``cuda_filter`` option records the duration of each
kernel and memory operation (see Warp's `CUDA activity profiling
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#cuda-activity-profiling>`__).
:class:`~utils.EventTracer` uses CUDA events to time the functions that
:class:`~solvers.SolverMuJoCo` instruments.

For captured steps, Warp's activity timing reports the graph launch as a
single operation. Time the graph as a whole, profile the eager version, or
record timed CUDA events during capture to measure regions within the graph
(see Warp's `graph timing guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#timing-in-cuda-graphs>`__).
To check what a graph contains,
:func:`warp.capture_debug_dot_print` writes it as a Graphviz file.

Profilers and synchronizing timers add overhead and change the timing they
measure. Use them to find the expensive parts, and compare configurations with
the same instrumentation, or none, on both sides.

Nsight Systems
^^^^^^^^^^^^^^

For a timeline of the whole application, run it under NVIDIA Nsight Systems
without a viewer:

.. code-block:: console

    nsys profile -o h1 --trace=cuda,nvtx,osrt \
        uv run -m newton.examples robot_h1 --viewer null --num-frames 300

Open ``h1.nsys-rep`` in the Nsight Systems GUI, or print summaries with
``nsys stats h1.nsys-rep``. In the report, check:

- **Gaps on the GPU row** between kernels during steady-state stepping. Gaps
  mean the GPU waits for the CPU, usually because of launch overhead or a host
  synchronization.
- **Synchronization and memory copies** in the CUDA API row, such as
  ``cudaStreamSynchronize`` or device-to-host copies inside the step. Each one
  points to a host read in the loop.
- **Kernel time summary** (``cuda_gpu_kern_sum`` in ``nsys stats``) to see
  which kernels dominate the step.
- **Startup**: compilation, module loading, and graph capture before the first
  replay, kept separate from the steady state.

Pass ``use_nvtx=True`` to :class:`warp.ScopedTimer` to label regions on the
timeline. To record only steady-state stepping, see Warp's guide to `limiting
the profiler capture range
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#limiting-the-profiler-capture-range>`__.
By default, Nsight Systems shows each graph launch as one range. To see the
kernels inside the graph, add ``--cuda-graph-trace=node``. Node-level
tracing adds overhead to every kernel in the graph and can hide the benefit of
capture, so use it to see what runs inside the graph, not to measure how long
the step takes.

Nsight Compute
^^^^^^^^^^^^^^

Once you know which kernel dominates, use Nsight Compute to analyze it.
Profile only a few launches of that kernel, because Nsight Compute replays each
profiled kernel many times:

.. code-block:: console

    ncu -o h1_kernel --set full -k regex:<kernel_name> -c 5 \
        uv run -m newton.examples robot_h1 --viewer null --num-frames 20

Reading GPU performance counters can require administrator privileges. For
interpreting the report and correlating it with kernel source code, see Warp's
`Nsight Compute guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#nsight-compute-profiling>`__.

Memory
^^^^^^

:func:`warp.get_mempool_used_mem_high` reports the high-water mark of memory
used from a device's Warp memory pool. This excludes allocations outside that
pool. The process memory reported by ``nvidia-smi`` also includes reserved pool
memory and other CUDA allocations. See Warp's `memory management guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/memory_management.html#querying-memory-usage>`__.

.. _performance-graph-capture:

CUDA Graph Capture
------------------

A simulation frame consists of many small kernel launches: clearing forces,
detecting collisions, and one or more solver passes for every substep. For
small scenes, each kernel finishes quickly and the GPU spends much of the frame
waiting for the CPU to launch the next one.

A CUDA graph records a sequence of launches once and replays it with a single
call. With Warp, you record the frame inside :class:`warp.ScopedCapture` and
replay it with :func:`warp.capture_launch` (see Warp's `graph documentation
<https://nvidia.github.io/warp/stable/user_guide/runtime.html#graphs>`__).
This removes the per-launch CPU cost but does not make the kernels themselves
faster. The benefit is greatest when host
launch overhead dominates; workloads that already keep the GPU busy may gain
little.

Simulation Loop
^^^^^^^^^^^^^^^

The following class shows a frame that is captured once and replayed. It
allocates everything the frame uses, runs one frame without capture, and then
captures a full frame, including all substeps and collision detection:

.. testsetup:: performance-loop

    import numpy as np
    import warp as wp
    import newton

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.5)
    model = builder.finalize()

.. testcode:: performance-loop

    class Simulation:
        def __init__(self, model, sim_substeps, use_graph=True):
            self.sim_substeps = sim_substeps
            self.sim_dt = 1.0 / 60.0 / sim_substeps

            self.state_0 = model.state()
            self.state_1 = model.state()
            self.control = model.control()
            self.pipeline = newton.CollisionPipeline(model)
            self.contacts = self.pipeline.contacts()
            self.solver = newton.solvers.SolverSemiImplicit(model)

            self.graph = None
            if use_graph and model.device.is_cuda:
                self.simulate()  # warm-up
                with wp.ScopedCapture(device=model.device) as capture:
                    self.simulate()
                self.graph = capture.graph

        def simulate(self):
            for i in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.pipeline.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                if self.sim_substeps % 2 == 1 and i == self.sim_substeps - 1:
                    self.state_0.assign(self.state_1)
                else:
                    self.state_0, self.state_1 = self.state_1, self.state_0

        def step(self):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

Capturing a frame does not run it, so the warm-up frame is the first frame of
the captured simulation. With that in mind, the captured and eager versions
produce the same result:

.. testcode:: performance-loop

    num_frames = 20

    for substeps in (4, 5):
        eager = Simulation(model, sim_substeps=substeps, use_graph=False)
        for _ in range(num_frames):
            eager.step()

        captured = Simulation(model, sim_substeps=substeps)
        replay_frames = num_frames - 1 if captured.graph is not None else num_frames
        for _ in range(replay_frames):
            captured.step()

        np.testing.assert_allclose(
            eager.state_0.body_q.numpy(), captured.state_0.body_q.numpy(), atol=1e-5
        )
        np.testing.assert_allclose(
            eager.state_0.body_qd.numpy(), captured.state_0.body_qd.numpy(), atol=1e-5
        )

When validating your own captured loop, test both odd and even substep counts
and change controls between frames. Compare velocities and any relevant solver
history as well as positions, using tolerances appropriate to the scene. For
repeatable contact ordering, see :ref:`Deterministic Contacts`.

The uncaptured warm-up frame runs first-step work outside the graph. Some
solvers and controllers allocate buffers during their first step, and
initialization recorded during capture repeats on every replay. For example,
zeroing a lazily allocated history buffer during capture would discard its
warm-start data every frame. Because the warm-up advances the simulation, save
and restore the state, including any solver history, if the run must start from
its initial conditions.

The first :func:`warp.capture_launch` call takes longer than later ones,
because CUDA builds the executable graph at that point. Leave it out of
timings.

Capture the whole frame in one graph rather than one graph per substep, and
include collision detection, control updates, and any per-step metric kernels.
Kernels outside the graph still incur individual host launch costs.

The example uses CUDA graphs on CUDA devices and runs the frame directly on the
CPU. If the captured code allocates memory, capture also requires the device's
memory pool, which you can check with ``device.is_mempool_enabled`` (see Warp's
documentation on `graph allocations
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/memory_management.html#graph-allocations>`__).

.. _performance-odd-substeps:

Odd Substep Counts
^^^^^^^^^^^^^^^^^^

The example above treats the last substep differently when the substep count is
odd. A plain state swap at every substep would cause each replay to read an
outdated state.

The swap ``state_0, state_1 = state_1, state_0`` exchanges two Python names.
During capture, it runs once per substep, and the graph records which buffer
each solver step reads and writes. Call the two buffers A and B. With three
substeps, the recorded frame looks like this:

.. list-table::
   :header-rows: 1

   * - Substep
     - Reads
     - Writes
   * - 1
     - A
     - B
   * - 2
     - B
     - A
   * - 3
     - A
     - B

On replay, the graph writes the final result to B but starts the next frame by
reading A, which holds the result of substep 2. For a solver without other
persistent state, the simulation advances only two substeps per frame while
computing three. Solver history can introduce further inconsistencies. With one
substep, the input state never advances.

With an even substep count, the frame ends in A and the problem does not occur.
For odd counts, there are three options:

- Copy the result back to A at the end of the frame with
  :meth:`State.assign`, as the example does. This costs one state copy per
  frame and keeps ``state_0`` as the current state for rendering, sensors, and
  resets.
- Round the substep count up to an even number and recompute ``dt``.
- Capture two graphs, one starting from A and one from B, and replay the one
  that matches the current state. This avoids the copy but makes every reader
  of the state track which buffer is current. :ref:`stateful-actuators`
  shows this pattern.

Other double-buffered data in the graph, such as :ref:`actuator state
<stateful-actuators>`, behaves the same way. :class:`CollisionSubstepScheduler`
always swaps its two states and therefore requires an even substep count.

What a Graph Records
^^^^^^^^^^^^^^^^^^^^

A graph records memory addresses and launch parameters, not Python code. When
it replays, Python does not run. To update inputs between replays, write into
the captured arrays. Array addresses, launch dimensions, scalar arguments such
as ``dt``, and the outcome of any Python ``if`` stay as they were during
capture.

Replacing an array does not update the address recorded in the graph:

.. code-block:: python

    # Replacing the array leaves the graph reading the old buffer.
    control.joint_target_q = wp.array(targets, dtype=float)

    # Update the captured buffer in place instead.
    control.joint_target_q.assign(targets)

For a reusable graph:

- Keep every state, :class:`Contacts`, and solver used by the graph alive
  while you replay it. The graph refers to their memory directly.
- A counter that should advance from one replay to the next, such as a frame
  index into a trajectory, has to live in a device array and be incremented by
  a kernel inside the graph. A Python integer would be frozen at its value
  during capture.
- Settings that the solver evaluates in Python are frozen as well. Capture the
  graph again after changing solver update intervals, the collision schedule
  set with :meth:`~solvers.SolverBase.set_collision_frequency`, the ``dt``
  argument of :meth:`CollisionPipeline.collide`, the flag set by
  :meth:`~solvers.SolverVBD.set_rigid_history_update`, or the size of any
  buffer.

For updates scheduled by a Python step counter, capture a complete repeating
schedule. For example, the number of steps in a graph should be a multiple of
Featherstone's ``update_mass_matrix_interval``. Otherwise, replay repeats the
captured update pattern at a different frequency from eager execution.

For supported in-place model edits, such as changes to masses or gains, call
:meth:`~solvers.SolverBase.notify_model_changed` outside the graph with the
:class:`ModelFlags` that describe the change. Prefer the narrowest applicable
flags to avoid unnecessary updates. Setters such as :meth:`Model.set_gravity`
name the flag to use in their docstrings.

Whether an existing graph picks up the change depends on what the solver does
in response. If it only rewrites the contents of existing device arrays, the
graph reads the new values on its next replay. If the solver replaces buffers,
or decides on the host which kernels to launch, the graph keeps replaying what
it recorded and you need to capture it again. For example,
:class:`~solvers.SolverFeatherstone` reacts to inertial changes by recomputing
the mass matrix on its next :meth:`~solvers.SolverBase.step`, but that decision
is made in Python. With ``update_mass_matrix_interval`` greater than 1, a graph
captured before the change can keep using the old mass matrix on the substeps
where the update was skipped during capture. Use an interval of 1 or recapture
after notifying the solver. If a solver's documentation does not say that its
model updates are capture-safe, recapture after calling
:meth:`~solvers.SolverBase.notify_model_changed`.

Sensors, actuators, selection views, and some solvers have further capture
requirements, described on their own pages: :ref:`solver_observables`,
:ref:`actuators <stateful-actuators>`, :ref:`deformable-selection`, and
:doc:`/solvers/mujoco`.

.. _performance-host-sync:

Host Synchronization
--------------------

GPU operations are usually asynchronous: the CPU queues work and continues.
Reading device data on the host waits for the relevant GPU work to finish,
which can interrupt that overlap and leave gaps between launches. Host reads
also prevent CUDA graph capture. Warp's `concurrency guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/concurrency.html>`__
describes which operations run asynchronously and how to synchronize with as
little blocking as possible.

Common sources of synchronization are ``array.numpy()``, printing an array,
converting a PyTorch tensor to a Python number, and reading a contact count to
decide how many threads to launch. Host-side setters that accept NumPy arrays,
such as :meth:`Model.set_gravity`, copy data to the device and can read data
back; they are meant for setup or occasional changes, not for every step.

Move readbacks out of the simulation loop where possible. Accumulate metrics
and error sums on the device and copy them to the host once per episode. Launch
kernels over a buffer's full capacity and let each thread compare its index
with the count on the device. Collect peak counts or overflow flags on the
device and read them periodically or at the end of a run. To find reads you did
not expect, profile the run with Nsight Systems, which shows synchronization on
its timeline.

Render once per frame, after the simulation step, to avoid repeating viewer
work in every substep. Some viewers read data back to the host when logging. In
particular, :meth:`~viewer.ViewerBase.log_contacts` reads the contact count
whenever contacts are drawn. With many worlds, rendering only a few of them
with :meth:`~viewer.ViewerBase.set_visible_worlds`, or the examples'
``--max-worlds`` option, keeps the viewer from dominating the frame. To time
the simulation without rendering, use :class:`~viewer.ViewerNull` (``--viewer
null`` in the examples). See :ref:`choosing-a-viewer`.

.. _performance-buffer-capacity:

Buffer Capacity
---------------

Contacts and constraints are stored in buffers with a fixed capacity. Some
operations launch over the full capacity rather than the number of entries in
use, such as contact sorting with ``deterministic=True``, so an oversized
buffer costs time as well as memory. Insufficient capacity can drop contacts
or constraint rows and change the simulation.

Size the buffers from measurement. Run a worst-case scene at the world count
you intend to use, record the peak contact and constraint counts, and set the
capacities to that peak plus a margin.

The capacities of :class:`CollisionPipeline`, such as ``rigid_contact_max``,
``soft_contact_max``, and ``shape_pairs_max``, are totals over all worlds.
Automatic contact-capacity estimates are not worst-case bounds. In contrast,
the NxN and SAP broad phases default to a worst-case shape-pair bound summed
over worlds, including shared global shapes. This can reserve much more memory
than the scene needs. MuJoCo's contact and constraint capacities, by contrast,
are set per world (see :doc:`/solvers/mujoco`), and the solver can report
overflow per world (see :ref:`mujoco_overflow_observable`).

With ``verify_buffers=True``, the collision pipeline reports overflow. Keep it
enabled while sizing buffers. Once capacities have been validated, disabling
the check removes one diagnostic kernel per collision pass. Recheck capacities
when changing the scene, world count, or contact settings.

Collision Detection
-------------------

Reduce candidate pairs before changing collision accuracy. Use collision groups
and filters to exclude pairs that should not interact, and choose a broad phase
suited to the scene. Shared shapes (world ``-1``) collide with every world, so
their cost grows with the world count. See :ref:`Collision Pipeline Details`.

Request contact sensors and solver observables only when the application
needs their output. Simpler geometry and less frequent detection trade accuracy
for speed; see `Simulation Accuracy and Cost`_.

.. _performance-tradeoffs:

Simulation Accuracy and Cost
----------------------------

Changes to collision and solver settings can reduce work but affect the
simulated behavior. Evaluate them against the accuracy metric and tolerance
chosen in `Measuring Performance`_. :ref:`Simulation Tuning` and the linked
pages describe their effect on behavior.

Collision and solver rates
    Detecting collisions less often than the solver steps saves collision time
    but uses contacts that are older by up to the detection interval. See
    :ref:`collision-frequency-in-the-simulation-loop`.

Contact gap and tunneling
    A contact is found only when two shapes are within the contact gap at the
    time of detection. If a body moves farther than the gap between two
    detection passes, it can pass through thin geometry. A larger gap or
    velocity-dependent speculative contacts can reduce this risk, at the cost
    of more contacts; neither guarantees that tunneling is eliminated. Less
    frequent detection or larger time steps increase the motion between passes,
    so revisit the gap when changing either. See
    :ref:`Collisions`.

Collision geometry
    Primitives and convex hulls can be cheaper than detailed meshes, but an
    approximation changes the collision surface. Use
    :meth:`ModelBuilder.approximate_meshes` where that change is acceptable.
    For mesh-mesh contacts, attaching an SDF with :meth:`Mesh.build_sdf`
    usually speeds up collision detection; coarser SDFs use less memory but
    lose small features. See
    :ref:`Collision Pipeline Details`.

Contact model
    SDF queries and hydroelastic contact generation can add collision work.
    Hydroelastic contacts represent contact patches and pressure distributions;
    use them where the task needs that behavior. Enable expensive models only
    on the shapes that need them. See :ref:`Hydroelastic Contacts`.

Substeps, iterations, and tolerances
    Solver cost scales with substeps and iterations. Reduce iterations, or let
    the solver stop early through a convergence tolerance where it supports
    one, only while the residual or task metric stays within bounds that you
    can justify physically. Fewer substeps also make each step larger, which
    affects stability and contact. Equal products of substeps and iterations
    do not imply equal accuracy. See the tuning order in
    :ref:`Simulation Tuning` and :ref:`Tuning Solver Reference`.

Solver choice and solver options
    Solvers differ in cost and in the effects they model, and many expose
    options that trade accuracy for speed. For example,
    :class:`~solvers.SolverImplicitMPM` can transfer velocities with PIC, which
    is cheaper but more dissipative than the default APIC scheme. See the
    :doc:`solver overview </solvers/index>` and the solver's API reference.

Multiple Worlds
---------------

To simulate many copies of a scene, build one world and copy it with
:meth:`ModelBuilder.replicate`. Mesh approximations and SDFs are computed once
per source mesh and shared across worlds.

Measure both throughput (world-steps per second) and latency (time per step)
across several world counts. Batching can improve GPU utilization, but gains
eventually level off while memory use and latency continue to grow. Choose the
world count that meets the application's throughput, latency, and memory needs.

Batch per-world updates in device kernels rather than looping over worlds in
Python. After changing model parameters, call
:meth:`~solvers.SolverBase.notify_model_changed` with the relevant flags.
Partial state resets take a device mask rather than a list of indices; see
:ref:`Per-world reset masks`. Rather than lowering the timestep until a world
that exceeds physical limits (such as joint torque, contact force, or
penetration depth) stays stable, check those limits on the device and reset
the affected worlds.

A model and its solver run on a single device. To use several GPUs, run one
simulation per device, for example one process per GPU.
