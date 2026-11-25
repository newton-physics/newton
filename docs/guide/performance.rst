.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _performance-guide:

Performance
===========

This guide shows how to benchmark and profile Newton simulations, reduce GPU
launch overhead with CUDA graphs, and keep per-step data on the device. It
also covers sizing contact and constraint buffers, tuning collision and solver
work while checking accuracy, and scaling to multiple worlds.
For custom kernels, see Warp's `performance guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance.html>`__.

Start with a scene that represents your workload. Choose a quantity to
preserve, such as tracking error, penetration depth, or force balance, and
set an acceptable tolerance. For solver tuning, use a reference with enough
substeps or iterations for that quantity to converge. Compare changes over
the same amount of simulated time. Check for non-finite state and contact or
constraint overflow before accepting a speedup. See :ref:`Simulation Tuning`
for help choosing a reference.

.. _performance-benchmarking:

Benchmark the workload
----------------------

Run a headless example for a quick baseline. The benchmark warms up and reports
frames per second. This command stops at 10 measured seconds or 1000 total
frames, including warm-up, whichever comes first.

.. code-block:: console

    uv run -m newton.examples basic_shapes --benchmark 10 --num-frames 1000

The example benchmark synchronizes after every measured frame, so its FPS
includes per-frame synchronization and prevents overlap between successive
frames. Timing a batch with synchronization only at its boundaries can report
different throughput, especially for small workloads. Use the same
synchronization policy when comparing results.

For your own loop, warm up the simulation and replay any captured graph once
before timing. Synchronize the device before and after timing a batch of
frames. Report startup separately from steady-state stepping. Repeat the
measurement and report the spread. Change one setting at a time and record
both timing and accuracy.

See Warp's guides to `cold-start benchmarking
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#cold-start-benchmarking-approaches>`__
and `reducing compilation and startup time
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/reducing_compilation_and_startup_time.html>`__.

Record hardware and software versions, world count, timestep, substeps,
collision cadence, solver settings, and whether the run includes rendering,
sensors, controls, or resets. Report the accuracy metric alongside the timings.
To track regressions in Newton itself, use the ASV suite in the
:doc:`development guide </guide/development>`.

.. _performance-debug-settings:

Benchmark configuration
^^^^^^^^^^^^^^^^^^^^^^^

Disable Warp debug-mode compilation and CUDA error verification for timing.
See Warp's `debugging guide
<https://nvidia.github.io/warp/stable/user_guide/debugging.html>`__.

Enable ``ModelBuilder.finalize(requires_grad=True)`` when differentiating
through the simulation. It allocates gradient arrays and enables additional
differentiable contact data. For custom kernels that never need gradients,
set ``enable_backward=False`` per kernel or module. Leave the global
``warp.config.enable_backward`` enabled because some Newton features use
gradients during forward simulation. See Warp's `module settings
<https://nvidia.github.io/warp/stable/user_guide/configuration.html#module-settings>`__.

.. _performance-profiling:

Find the bottleneck
-------------------

Use Nsight Systems to see where the CPU and GPU spend time. Start with the
null viewer to isolate stepping. Enable the viewer when measuring rendering.

.. code-block:: console

    nsys profile -o h1 --trace=cuda,nvtx,osrt \
        uv run -m newton.examples robot_h1 --viewer null --num-frames 300

To record only the steps after warm-up, see Warp's guide to
`limiting the profiler capture range
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#limiting-the-profiler-capture-range>`__.
Inspect the steady-state GPU timeline and the CUDA API calls around gaps.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - What you see
     - What to check first
   * - GPU idle between many short kernels
     - Check for CPU launch overhead and :ref:`capture the frame <performance-graph-capture>`.
   * - CPU waits or device-to-host copies inside the step
     - Trace the calls to :ref:`host reads <performance-host-sync>`,
       such as contact-count reads and viewer logging.
   * - Collision or solver work dominates
     - :ref:`Check buffer capacity <performance-buffer-capacity>`, then
       :ref:`reduce collision or solver work <performance-tradeoffs>`.
   * - Throughput levels off as you add worlds
     - Compare :ref:`throughput, latency, and memory <performance-multiple-worlds>`
       across world counts.

Use :class:`warp.ScopedTimer` with ``use_nvtx=True`` to label timeline regions.
See Warp's `ScopedTimer documentation
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#scopedtimer>`__
for timing options.

Use graph-level tracing for whole-frame timing. Collect a separate trace with
``--cuda-graph-trace=node`` to rank individual kernels, since tracing each node
adds overhead. To measure regions within a captured frame, see Warp's guide to
`timing within CUDA graphs
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#timing-in-cuda-graphs>`__.

Once the timeline identifies an expensive kernel, use Nsight Compute to
investigate its compute and memory throughput. See Warp's `Nsight Compute guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#nsight-compute-profiling>`__.

.. _performance-graph-capture:

Capture a simulation frame
--------------------------

A CUDA graph replays a frame's kernel launches with one host call. Capture
the full frame with :class:`warp.ScopedCapture`, including collision, control
updates, and any device-side metrics that run each frame. Replay it with
:func:`warp.capture_launch`. The largest gains come when launch overhead
leaves the GPU idle. See Warp's `CUDA graph guide
<https://nvidia.github.io/warp/stable/user_guide/runtime.html#graphs>`__.

Allocate buffers and run an eager warm-up frame before capture so first-step
allocation and initialization happen once. The warm-up advances the
simulation. Account for it when comparing runs, or restore the initial
state, including solver history, if the run must start from the original
conditions. Keep the captured arrays, contacts, and solver alive across replays.

Replay uses the buffer addresses, scalar arguments, launch dimensions, and
Python decisions recorded during capture.

- Update controls in place, for example with
  ``control.joint_target_q.assign(targets)``.
- Put counters that advance each replay in device arrays and update them
  with a kernel. For a schedule driven by a Python counter, capture a full
  repeating cycle of updates.
- Recapture after changing a recorded value such as ``dt``, a launch dimension,
  a buffer address, or a Python-side collision schedule.
- After a supported in-place model edit, call
  :meth:`~solvers.SolverBase.notify_model_changed` outside the graph with the
  narrowest applicable :class:`ModelFlags`. Recapture unless the solver
  documents that the update is safe for the existing graph.

See the capture requirements for :ref:`solver_observables`,
:ref:`actuators <stateful-actuators>`, :ref:`deformable-selection`, and
:doc:`/solvers/mujoco`. For capture-time allocation and device memory pools,
see Warp's `graph allocation guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/memory_management.html#graph-allocations>`__.

For individual custom kernels launched repeatedly outside a graph, Warp's
`launch objects
<https://nvidia.github.io/warp/stable/user_guide/runtime.html#launch-objects>`__
can reduce Python launch overhead.

.. _performance-odd-substeps:

Odd substep counts
^^^^^^^^^^^^^^^^^^

With three substeps, the captured state sequence is A → B → A → B. The next
replay starts from A, which still holds the result of the second substep.
Copy the final state back to A inside the frame with :meth:`State.assign`.
With an even number of substeps the final state already lands in A. The same
rule applies to other double-buffered data, such as actuator state.
:ref:`stateful-actuators` shows how to alternate between two graphs to avoid
the copy.

Simulation loop
^^^^^^^^^^^^^^^

This example accepts an already finalized :class:`Model` and uses
:class:`~solvers.SolverSemiImplicit` to demonstrate the loop. It captures a
full frame on CUDA and steps eagerly on CPU. The final substep copies the
state back when the substep count is odd.

.. testsetup:: performance-loop

    import numpy as np
    import warp as wp
    import newton

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.8), wp.quat_identity()))
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

.. dropdown:: Validate eager and captured stepping

   Capture records the frame, and the warm-up already advanced the simulation
   by one frame. Account for that frame in the comparison below. The run
   continues until the sphere contacts the ground.

   .. testcode:: performance-loop

       num_frames = 20

       for substeps in (4, 5):
           eager = Simulation(model, sim_substeps=substeps, use_graph=False)
           for _ in range(num_frames):
               eager.step()
           assert eager.contacts.rigid_contact_count.numpy()[0] > 0

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

   When validating your own loop, change controls between replays and compare
   relevant solver history as well as positions and velocities. For
   repeatable contact ordering, see :ref:`Deterministic Contacts`.

.. _performance-host-sync:

Keep per-step data on the device
--------------------------------

Device-to-host reads in the step can stall work and prevent CUDA graph
capture. Look for ``array.numpy()``, PyTorch ``tensor.item()`` or
``tensor.cpu()``, and host reads of contact counts used to choose a launch
size. Launch over the buffer capacity and check the active count on the
device. Accumulate metrics and peak counts on the device, then read them
periodically or at the end of the run. Use host-side setters such as
:meth:`Model.set_gravity` for setup or occasional edits.

Render once per frame. Drawing contacts with
:meth:`~viewer.ViewerBase.log_contacts` reads the contact count on the host.
For many worlds, display a subset with
:meth:`~viewer.ViewerBase.set_visible_worlds`. See :ref:`choosing-a-viewer`.

See Warp's `concurrency guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/concurrency.html>`__
for asynchronous operations and its `synchronization guidance
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/concurrency.html#synchronization-guidance>`__
on overlapping transfers with computation.

.. _performance-buffer-capacity:

Size contact and constraint buffers
-----------------------------------

Use a peak-load scene at the intended world count. Run with enough capacity
to avoid overflow, then size each buffer from the measured peak plus a
margin. Repeat the check when the scene, world count, or contact settings
change. Oversized buffers can cost time as well as memory because operations
such as deterministic contact sorting can use the full capacity.

Keep ``verify_buffers=True`` while testing :class:`CollisionPipeline`
capacities. It checks rigid narrow-phase overflow, including intermediate
buffers. Check ``soft_contact_count`` against ``soft_contact_max`` separately:
ordinary particle soft contacts can overflow without a warning even with
verification enabled. Check solver constraints separately using the solver's
diagnostics. Confirm how each capacity is defined. For example,
:class:`CollisionPipeline` capacities are totals across worlds, while
:class:`~solvers.SolverMuJoCo` has per-world capacities.

To measure the memory cost of different capacities, see Warp's guide to
`measuring memory usage
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/memory_management.html#querying-memory-usage>`__.

.. _performance-tradeoffs:

Reduce collision and solver work
--------------------------------

Start by filtering pairs that should never interact, choosing a broad phase
suited to the scene, and requesting only the contact data the application
uses. Shared shapes (world ``-1``) can collide with every world. See the
:ref:`collision performance notes <Performance>` for pipeline-specific
options and their costs.

Use the baseline accuracy target to evaluate changes to the physics.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Change
     - What to recheck
   * - Detect collisions less often
     - Recheck missed contacts, tunneling, and penetration, especially for
       fast bodies and thin geometry. See
       :ref:`collision-frequency-in-the-simulation-loop`.
   * - Simplify geometry or contact models
     - Check contact location and force distribution. Use
       :meth:`ModelBuilder.approximate_meshes` or adjust SDF resolution where
       the task tolerates changes to the collision surface. See
       :ref:`Collision Pipeline Details` and :ref:`Hydroelastic Contacts`.
   * - Reduce solver effort
     - Check convergence or task error after changing iterations or
       tolerances. If you reduce substeps, also recheck stability and
       contacts at the larger timestep. See :ref:`Tuning Solver Reference`.
       For early exit in a captured custom loop, see Warp's `conditional execution
       <https://nvidia.github.io/warp/stable/user_guide/runtime.html#conditional-execution>`__.
   * - Enable sleeping
     - Measure the speedup in a representative scene and check that bodies
       wake when needed. See :ref:`MuJoCo sleeping <mujoco-sleeping>`.

For the order in which to change these settings, see :ref:`Simulation Tuning`.

.. _performance-multiple-worlds:

Scale to multiple worlds
------------------------

Build one world and copy it with :meth:`ModelBuilder.replicate`. Approximate
meshes and build any needed SDFs before replication so the copies share the
results. Batch per-world updates in device kernels and use
:ref:`Per-world reset masks` to reset selected worlds.

Measure world-steps per second, step latency, and peak memory at several
world counts. Choose a batch size that meets the application's throughput,
latency, and memory needs. A model and solver run on a single device. For
multiple GPUs, run one simulation per device.
