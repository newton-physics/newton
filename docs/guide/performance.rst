.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _performance-guide:

Performance
===========

Most Newton simulations are slow for a small number of reasons: the CPU spends
more time launching kernels than the GPU spends running them, the frame stops
to copy data back to the host, collision detection does more work than the
scene needs, or buffers are sized far beyond what the simulation uses. This
page explains each of these and how to fix them.

Get the simulated behavior right before optimizing it (see
:ref:`Simulation Tuning`). An optimization should leave the results unchanged.
If a change makes the simulation faster by dropping contacts or skipping
substeps, it has changed the simulation, and the speedup does not count.

Newton runs on `NVIDIA Warp <https://nvidia.github.io/warp/>`__. If you write
your own kernels, Warp's `performance
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance.html>`__
and `profiling
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html>`__
guides apply to them.

.. important::

   Default capacities, solver options, and model flags change between
   releases. This page names them as examples; check the API reference of your
   installed version for the exact behavior.

Measuring
---------

Before comparing two configurations, make sure both produce valid results. The
state should contain no NaN or Inf values, contact and constraint counts should
stay below their capacities (see :ref:`performance-buffer-capacity`), and the
result should match a reference run within a tolerance you choose in advance.
A good reference is the same scene with more substeps or iterations, compared
using a physical quantity such as penetration depth or the trajectory of a body.

Then find out where the time goes. In practice, the largest cost is usually
one of the following, roughly from most to least common:

1. Kernel launch overhead, when the frame is not captured in a CUDA graph.
2. Host synchronization inside the frame.
3. Collision detection.
4. Solver substeps and iterations, multiplied by the number of worlds.
5. Buffers much larger than needed.
6. Rendering.

The first two are pure overhead and can be removed without touching the
physics, so start there. Change one setting at a time, and record the accuracy
metric next to each timing so you can tell a real speedup from a cheaper
simulation.

.. _performance-graph-capture:

CUDA Graph Capture
------------------

A simulation frame consists of many small kernel launches: clearing forces,
detecting collisions, and one or more solver passes for every substep. For
small scenes, each kernel finishes quickly and the GPU spends much of the
frame waiting for the CPU to launch the next one.

A CUDA graph records a sequence of launches once and replays it with a single
call. With Warp, you record the frame inside :class:`warp.ScopedCapture` and
replay it with :func:`warp.capture_launch`. This removes the per-launch CPU
cost but does not make the kernels themselves faster. Scenes with few bodies
and few worlds often run several times faster when captured; large scenes,
where a few kernels dominate the frame, gain little.

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

    eager = Simulation(model, sim_substeps=5, use_graph=False)
    for _ in range(num_frames):
        eager.step()

    captured = Simulation(model, sim_substeps=5)
    for _ in range(num_frames - 1 if captured.graph is not None else num_frames):
        captured.step()

    assert np.allclose(eager.state_0.body_q.numpy(), captured.state_0.body_q.numpy(), atol=1e-5)

A comparison like this catches most capture bugs and is worth keeping as a
test. Run it with both odd and even substep counts, change the controls
between frames, and compare velocities and solver history as well as
positions. Results agree to within ``float32`` tolerance, not bit for bit; for
bitwise repeatability, see :ref:`Deterministic Contacts`.

The uncaptured warm-up frame serves two purposes. Warp compiles and loads
kernels on their first launch, and the warm-up keeps that out of the graph.
Some solvers and controllers also allocate buffers during their first step. An
allocation made during capture becomes part of the graph and repeats on every
replay; for example, a buffer that is created and zeroed on first use would be
zeroed again every frame, which discards warm-start data. Because the warm-up
advances the simulation, save and restore the state, including any solver
history, if the run must start from its initial conditions.

The first :func:`warp.capture_launch` call takes longer than later ones,
because CUDA builds the executable graph at that point. Leave it out of
timings.

Capture the whole frame in one graph rather than one graph per substep, and
include collision detection, control updates, and any per-step metric kernels.
Any kernel left outside the graph still pays its launch cost every frame, and a
few of them can cancel most of the benefit.

Graph capture applies to CUDA devices; on the CPU, run the frame directly, as
the example does. If the captured code allocates memory, capture also requires
the device's memory pool, which you can check with
``device.is_mempool_enabled``.

.. _performance-odd-substeps:

Odd Substep Counts
^^^^^^^^^^^^^^^^^^

The example above treats the last substep differently when the substep count is
odd. Without that special case, the captured simulation silently runs in slow
motion.

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

After capture, the newest state is in B, but the graph always starts by
reading A, which holds the result of substep 2. Every replay therefore repeats
work from one substep back, and the simulation advances two substeps per frame
while computing three. Nothing fails, so the problem is easy to miss. Whether
it is visible also depends on the solver and scene, so a simulation that looks
correct is not proof that the loop is correct.

With an even substep count, the frame ends in A and the problem does not
occur. For odd counts, there are three options:

- Copy the result back to A at the end of the frame with
  :meth:`State.assign`, as the example does. This costs one state copy per
  frame and keeps ``state_0`` as the current state for rendering, sensors, and
  resets.
- Round the substep count up to an even number and recompute ``dt``.
- Capture two graphs, one starting from A and one from B, and replay the one
  that matches the current state. This avoids the copy but makes every reader
  of the state track which buffer is current. :ref:`stateful-actuators`
  shows this pattern.

Other double-buffered data in the graph, such as
:ref:`actuator state <stateful-actuators>`, behaves the same way. Some solvers
only support capture with an even substep count; their API reference says so.

What a Graph Records
^^^^^^^^^^^^^^^^^^^^

A graph records memory addresses and launch parameters, not Python code. When
it replays, Python does not run. Between replays, only the contents of the
captured arrays can change. Array addresses, launch dimensions, scalar
arguments such as ``dt``, and the outcome of any Python ``if`` stay as they were
during capture.

The most common mistake is replacing an array instead of writing into it:

.. code-block:: python

    # Wrong: the graph keeps reading the old array
    control.joint_target_q = wp.array(targets, dtype=float)

    # Right: copy the new values into the captured array
    control.joint_target_q.assign(targets)

The same reasoning leads to a few more rules:

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

Changing model parameters such as masses or gains does not require a new
capture, because the graph reads the model arrays at replay time. After
writing them, call :meth:`~solvers.SolverBase.notify_model_changed` outside
the graph with the :class:`ModelFlags` that describe the change, and keep
replaying. Setters such as :meth:`Model.set_gravity` name the flag to use in
their docstrings. Prefer the narrowest flags; ``ModelFlags.ALL`` makes the
solver recompute everything, which is expensive inside a loop.

Sensors, actuators, selection views, and some solvers have further capture
requirements, described on their own pages: :ref:`solver_observables`,
:ref:`actuators <stateful-actuators>`, :ref:`deformable-selection`, and
:doc:`/solvers/mujoco`.

.. _performance-host-sync:

Host Synchronization
--------------------

The CPU normally queues GPU work and moves on without waiting. Reading a
device array on the host breaks this: the CPU has to wait until the GPU has
finished everything queued before the read, and while it waits, it cannot
queue the next frame, so the GPU then sits idle. A single read per frame can
cost more than the simulation step itself in a small scene. Such reads also
cannot be captured in a graph.

Typical reads hidden in a simulation loop are ``array.numpy()``, printing an
array, converting a PyTorch tensor to a Python number, and reading a contact
count to decide how many threads to launch. Host-side setters that accept
NumPy arrays, such as :meth:`Model.set_gravity`, copy data to the device and
can read data back; they are meant for setup or occasional changes, not for
every step.

Most of these reads can be moved out of the frame. Accumulate metrics and
error sums on the device and copy them to the host once per episode. Launch
kernels over a buffer's full capacity and let each thread compare its index
with the count on the device. Check for buffer overflow at the end of a run or
every few hundred frames rather than every step. To find reads you did not
expect, profile the run with Nsight Systems, which shows synchronization on its
timeline.

Rendering is a common source of both readbacks and extra work. Render once per frame, after the simulation step, not inside the substep loop.
Some viewers read data back to the host when logging. In particular,
:meth:`~viewer.ViewerBase.log_contacts` reads the contact count whenever
contacts are drawn. With many worlds, rendering only a few of them with
:meth:`~viewer.ViewerBase.set_visible_worlds`, or the examples'
``--max-worlds`` option, keeps the viewer from dominating the frame. To time
the simulation without rendering, use :class:`~viewer.ViewerNull`
(``--viewer null`` in the examples). See :ref:`choosing-a-viewer`.

.. _performance-buffer-capacity:

Buffer Capacity
---------------

Contacts and constraints are stored in buffers with a fixed capacity. Many
operations on these buffers, such as sorting, clearing, and scanning, take time
proportional to the capacity rather than to the number of entries in use, so
an oversized buffer slows down every step. A buffer that is too small is worse:
contacts that do not fit are dropped, and the simulation changes.

Size the buffers from measurement. Run a worst-case scene at the world count
you intend to use, record the peak contact and constraint counts, and set the
capacities to that peak plus a margin.

The capacities of :class:`CollisionPipeline`, such as ``rigid_contact_max``,
``soft_contact_max``, and ``shape_pairs_max``, are totals over all worlds. When
you do not set them, Newton estimates them from the model; the estimates are
not upper bounds. The NxN and SAP broad phases by default reserve space for
every shape pair in each world, which is often far more than a scene needs.
MuJoCo's contact and constraint capacities, by contrast, are set per world
(see :doc:`/solvers/mujoco`), and the solver can report overflow per world
(see :ref:`mujoco_overflow_observable`).

With ``verify_buffers=True``, the collision pipeline reports overflow. Keep it
enabled while sizing the buffers and disable it afterwards to save the check.

Shapes with a nonzero ``gap`` generate speculative contacts before the shapes
touch. These take buffer space and collision time, so leave the gap at zero
for shapes that do not need early contacts.

Collision Detection
-------------------

Collision detection often costs as much as the solver, and making it cheaper
does not reduce accuracy the way fewer iterations would. The biggest savings
usually come from simpler geometry: collide with primitives or convex hulls
instead of render meshes (see :meth:`ModelBuilder.approximate_meshes`), and
build an SDF for meshes that collide with other meshes. Fewer candidate pairs
also help; the broad phase choice, separate worlds, collision groups, and
filters all reduce them, and global shapes (world ``-1``) are tested against
every world. :ref:`Collision Pipeline Details` covers these options.

Collision detection does not have to run every substep. Running it every few
substeps, or once per frame, is often enough when bodies move little between
substeps; compare the result with per-substep detection before relying on it.
See :ref:`collision-frequency-in-the-simulation-loop`.

Contact sensors and contact observables add work to every step, so request
them only when you use them.

Multiple Worlds
---------------

To simulate many copies of a scene, build one world and copy it with
:meth:`ModelBuilder.replicate`. Approximate meshes and build SDFs on the
original before replicating, so the work is done once.

Throughput, measured in world-steps per second, increases with the number of
worlds until the GPU is fully used. Beyond that point, adding worlds increases
the time per step and the memory use without increasing throughput. Measure a
few world counts to find where this happens on your hardware.

Update all worlds together: randomize or reset parameters in one kernel launch,
then call :meth:`~solvers.SolverBase.notify_model_changed` once. Partial resets
take a device mask rather than a list of indices; see
:ref:`Per-world reset masks`.

To measure memory, :func:`warp.get_mempool_used_mem_high` reports Warp's peak
allocation. ``nvidia-smi`` reports more, because it includes memory reserved by
the pool and used by other processes.

.. _performance-benchmarking:

Benchmarking
------------

The examples accept ``--benchmark``, which replaces the viewer with the null
viewer, warms up, and prints frames per second:

.. code-block:: console

    python -m newton.examples basic_shapes --benchmark 10

When timing your own code, capture and replay the graph once before starting
the timer, and synchronize the device before and after the timed replays.
Report the setup time, which includes building the model and compiling
kernels, separately from the time per step. :class:`~utils.EventTracer` times
instrumented solver functions with CUDA events. To track performance changes in
Newton itself, use the airspeed velocity suite described in the
:doc:`development guide </guide/development>`.

A timing is only comparable when the configuration behind it is known. Along
with the result, note the world count, substeps, iterations, collision
frequency, the hardware and software versions, and whether rendering and resets
were included, together with the accuracy check from `Measuring`_.
