.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _performance-guide:

Performance
===========

Profile a representative Newton scene before changing execution settings or
tuning the physics. Choose an accuracy metric, such as tracking error,
penetration depth, or force balance, and set a tolerance before measuring.
For solver tuning, increase substeps or iterations until the metric
converges, then use that run as a reference. Compare runs over the same
simulated duration and check for non-finite state and buffer overflow before
accepting a speedup. See :ref:`Simulation Tuning` for help choosing a reference.

For custom kernels, see Warp's `performance guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance.html>`__.

.. _performance-benchmarking:

Benchmark the workload
----------------------

Run a headless example to establish a baseline. The command below reports
frames per second after warm-up. It stops after 10 measured seconds or 1000
total frames, including warm-up, whichever comes first.

.. code-block:: console

    uv run -m newton.examples basic_shapes --benchmark 10 --num-frames 1000

The example synchronizes after each measured frame. That adds a host wait
and prevents the host from submitting the next frame while the GPU finishes
the current one. Batch timing can give different throughput, especially
for small workloads. Use the same synchronization policy for each run.

For your own loop, warm up the simulation and replay any captured graph once
before timing. Synchronize the device before and after timing a batch of
frames. Report startup and steady-state timing separately. Repeat the
measurement and report how much it varies. Change one setting at a time.

See Warp's guides to `cold-start benchmarking
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#cold-start-benchmarking-approaches>`__
and `reducing compilation and startup time
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/reducing_compilation_and_startup_time.html>`__.

Record hardware and software versions, world count, timestep, substeps,
collision refresh frequency, solver settings, and whether the run includes
rendering, sensors, controls, or resets. Report the accuracy metric with the timings.
To track regressions in Newton itself, use the ASV suite in the
:doc:`development guide </guide/development>`.

.. _performance-debug-settings:

Benchmark configuration
^^^^^^^^^^^^^^^^^^^^^^^

Disable Warp debug-mode compilation and CUDA error verification for timing.
See Warp's `debugging guide
<https://nvidia.github.io/warp/stable/user_guide/debugging.html>`__.

Enable ``ModelBuilder.finalize(requires_grad=True)`` when differentiating
through the simulation. By default, :class:`CollisionPipeline` inherits this
setting for its soft contacts. For rigid-contact gradients, call
:func:`eval_rigid_contact_kinematics` explicitly and include that work in the
benchmark. See :ref:`Differentiable Contacts`. For custom kernels that never
need gradients, set ``enable_backward=False`` per kernel or module. Leave
the global ``warp.config.enable_backward`` enabled because some Newton
features use gradients during forward simulation. See Warp's `module settings
<https://nvidia.github.io/warp/stable/user_guide/configuration.html#module-settings>`__.

.. _performance-profiling:

Find the bottleneck
-------------------

Use Nsight Systems to see where the CPU and GPU spend time.
Pass ``--viewer null`` to measure stepping without rendering.

.. code-block:: console

    nsys profile -o h1 --trace=cuda,nvtx,osrt \
        uv run -m newton.examples robot_h1 --viewer null --num-frames 300

Inspect the gaps in the steady-state GPU timeline and the corresponding CUDA
API calls. To record only the steps after warm-up, see Warp's guide to
`limiting the profiler capture range
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#limiting-the-profiler-capture-range>`__.

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

Once the timeline identifies an expensive kernel, use Nsight Compute to profile
a matching CUDA graph node. Replace ``KERNEL_NAME`` with the name from the
Nsight Systems trace and adjust ``--launch-skip`` to sample a steady-state launch.

.. code-block:: console

    ncu --target-processes all --graph-profiling node -k 'regex:KERNEL_NAME' \
        --launch-skip 10 --launch-count 1 -o h1-kernel \
        uv run -m newton.examples robot_h1 --viewer null --num-frames 300

Open ``h1-kernel.ncu-rep`` and check the compute and memory throughput in
GPU Speed of Light Throughput to see which resource is closer to saturation.
See Warp's `Nsight Compute guide
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/profiling.html#nsight-compute-profiling>`__.

.. _performance-graph-capture:

Capture a simulation frame
--------------------------

A CUDA graph replays a frame's kernel launches with one host call. It helps
most when launch overhead leaves the GPU idle. Capture the full frame with
:class:`warp.ScopedCapture`, including collision detection, control updates,
and device-side metrics. Replay it with :func:`warp.capture_launch`.
See Warp's `CUDA graph guide
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
- For in-place model edits, use :meth:`Model.get_attribute_update_flags` to
  find the flags for :meth:`~solvers.SolverBase.notify_model_changed`. The
  experimental :meth:`~solvers.SolverBase.is_notify_model_changed_graphable`
  query checks whether the notification can run inside capture. If you update
  the model separately from stepping, recapture the stepping graph
  unless the solver documents that reuse is safe. See :ref:`Runtime model updates`.

For example, :class:`~solvers.SolverFeatherstone` can capture a step that skips
the mass matrix update when ``update_mass_matrix_interval`` is greater than 1.
Replay repeats that choice even after notification marks the matrix dirty.
For forward simulations with kinematic joints, an armature edit and notification
can also replace a captured buffer, so recapture even with an interval of 1.

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

Pass a finalized :class:`Model` to the loop below. It uses
:class:`~solvers.SolverSemiImplicit`, captures a full frame on CUDA, and steps
eagerly on CPU. With an odd substep count, it copies the final state back to
the input buffer.

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
            newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
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

When validating your own loop, compare eager stepping and graph replay over
the same simulated duration. Change controls between replays and compare
positions, velocities, and any relevant solver history. See
:ref:`Deterministic Contacts` for repeatable contact ordering.

.. dropdown:: Validate eager and captured stepping

   Capture records the frame. The eager warm-up advances the simulation by
   one frame, so count it toward the captured run's 20 frames. The test
   checks that the sphere reaches the ground before comparing the final states.

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

   A joint frame can place a body away from its default pose. Check that the
   loop starts from the pose defined by the joint before taking any steps.

   .. testcode:: performance-loop

       builder = newton.ModelBuilder()
       link = builder.add_link()
       builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
       joint = builder.add_joint_revolute(
           parent=-1,
           child=link,
           parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
       )
       builder.add_articulation([joint])
       articulated = Simulation(builder.finalize(), sim_substeps=4, use_graph=False)
       np.testing.assert_allclose(articulated.state_0.body_q.numpy()[link, :3], [0.0, 0.0, 1.0])

.. _performance-host-sync:

Keep per-step data on the device
--------------------------------

Device-to-host reads can stall stepping and prevent CUDA graph capture.
Look for calls such as ``array.numpy()``, PyTorch ``tensor.item()`` and
``tensor.cpu()``. If a kernel processes only active contacts, launch over
the buffer capacity and check the contact count on the device. Accumulate
metrics and peak counts on the device and read them periodically or at the
end of the run. Use host-side setters such as :meth:`Model.set_gravity` for
setup or occasional edits.

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

Measure peak buffer use at the intended world count in a scene that
exercises the expected load. Start with enough capacity to avoid overflow,
then size each buffer for the measured peak plus a margin. Repeat the check
when the scene, world count, or contact settings change. Oversized buffers
can cost time as well as memory. For example, deterministic contact sorting
can use the full capacity.

Keep ``verify_buffers=True`` while testing :class:`CollisionPipeline`
capacities. It checks rigid narrow-phase overflow, including intermediate
buffers. Check ``soft_contact_count`` against ``soft_contact_max`` separately:
ordinary particle soft contacts can overflow without a warning. Use the
solver's diagnostics to check constraint capacity.

:class:`CollisionPipeline` contact capacities are totals across worlds.
With :class:`~solvers.SolverMuJoCo` on the MuJoCo Warp backend, ``njmax``
limits constraints per world and ``naconmax`` is the shared runtime contact
capacity. Size shared buffers for the combined peak across worlds and per-world
limits for the busiest world. See :ref:`mujoco_overflow_observable`.

To measure the memory cost of different capacities, see Warp's guide to
`measuring memory usage
<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/memory_management.html#querying-memory-usage>`__.

.. _performance-tradeoffs:

Reduce collision and solver work
--------------------------------

Filter pairs that should never interact, choose a broad phase suited to the
scene, and request only the contact data the application uses. Shared shapes
(world ``-1``) can collide with every world. See the
:ref:`collision performance notes <Performance>` for pipeline options.

Use the baseline accuracy metric to evaluate each change.
:ref:`Adaptive collision detection <tuning-adaptive-collision-detection>`
can skip collision checks when motion allows while keeping the solver
timestep fixed. When tuning iterations or tolerances, use the
:ref:`per-world solver residual <solver_residual_observable>` available in
Kamino and VBD to check convergence on the device, including during graph
replay.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Change
     - What to recheck
   * - Detect collisions less often
     - Check for missed contacts, tunneling, and penetration, especially for
       fast bodies and thin geometry. See
       :ref:`collision-frequency-in-the-simulation-loop`.
   * - Simplify geometry or contact models
     - Check contact location and force distribution. Use
       :meth:`ModelBuilder.approximate_meshes` or adjust SDF resolution where
       the task tolerates changes to the collision surface. See
       :ref:`Collision Pipeline Details` and :ref:`Hydroelastic Contacts`.
   * - Reduce solver effort
     - Check convergence and task error after changing iterations or
       tolerances. If you reduce substeps, check stability and contacts at
       the larger timestep. See :ref:`Tuning Solver Reference`.
   * - Enable sleeping
     - Measure the speedup in a representative scene and check that bodies
       wake when needed. See :ref:`MuJoCo sleeping <mujoco-sleeping>`.

Set task-specific physical limits, such as actuator loads or impact forces the
real robot can withstand. End an episode when it exceeds those limits and
record it as a failure. A smaller timestep can improve numerical stability
without bringing the trajectory within the robot's operating limits.

For the order in which to change these settings, see :ref:`Simulation Tuning`.

.. _performance-multiple-worlds:

Scale to multiple worlds
------------------------

Build one world and copy it with :meth:`ModelBuilder.replicate`. Approximate
meshes and build any needed SDFs before replication so the copies share the
results. Batch per-world updates in device kernels and use
:ref:`Per-world reset masks` to reset selected worlds.

At several world counts, measure throughput in world-steps per second,
latency in milliseconds per frame, and peak memory. A world-step is one
physics substep in one world. For a fixed substep count and world count,
throughput is ``frames × substeps per frame × worlds / elapsed seconds``.
Choose a batch size that meets the application's throughput, latency, and
memory needs.

A model and solver run on a single device. For multiple GPUs, run one
simulation per device.
