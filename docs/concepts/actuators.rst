.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton.actuators

Actuators
=========

.. experimental::

   The actuator API may change without prior notice. Feedback is welcome —
   please file issues or discussion threads.

Actuators provide composable implementations that read physics simulation
state, compute effort, and **accumulate** (scatter-add) the effort into
control arrays for application to the simulation.  The caller must zero the
output array before stepping actuators each frame.  The simulator does not
need to be part of Newton: actuators can be used in any simulator that
provides the state arrays and applies the computed effort.

Each :class:`Actuator` instance is **vectorized**: a single actuator object
operates on a batch of DOF indices in global state and control arrays, allowing
efficient integration into RL workflows with many parallel environments.

The goal is to provide canonical actuator models with support for
**differentiability** and **graphable execution**.  Actuators are designed to be easy to
customize and extend for specific actuator models.

Architecture
------------

An actuator is composed of five building blocks, applied in this order:

.. code-block:: text

   Actuator
   ├── InputProcessor[]  (optional: transforms state and command inputs)
   │   ├── InputProcessorDelay       (delays command inputs by N actuator timesteps)
   │   └── InputProcessorEncoder     (position encoder, e.g. behind a backlash joint)
   ├── Drive      (control law that computes raw effort, or the motor command)
   ├── CommandClamping[]  (optional, with a motor: clamps the motor command)
   │   └── ClampingCurrentPWM       (current and PWM limits of a servo firmware)
   ├── Motor       (optional: turns the motor command into the effort of a motor)
   │   └── MotorDC                 (PWM-driven DC motor, optionally on a battery)
   └── Clamping[]  (clamps effort to model actuator limits)
       ├── ClampingMaxEffort        (±max_effort symmetric clamp)
       ├── ClampingDCMotor         (velocity-dependent saturation)
       └── ClampingPositionBased   (position-dependent lookup table)

**Input processor**
   Transforms the inputs to the drive: the simulation state (positions,
   velocities) and the control targets (positions, velocities, feedforward
   effort).  Examples include a command delay and a position encoder, e.g.
   behind a backlash joint.  Multiple input processors can be combined on a single
   actuator.  See the individual input processor class
   documentation for details.

**Drive**
   Computes raw actuator effort [N or N·m] from the simulation state and
   control targets.  This is the actuator's control law.  Examples include PD,
   PID and neural-network-based control.  See the individual drive class
   documentation for the control-law equations.

   In an actuator with a motor, the drive must compute the motor command
   instead, e.g. the voltage of the firmware of a voltage-controlled servo
   (:class:`DriveVoltageControlled`), and receives the supply voltage of the
   motor.  Drives that compute an effort, such as :class:`DrivePD`, cannot be
   combined with a motor.

**Command clamping**
   Clamps the motor command to model the limits of the motor's electronics,
   e.g. the current and PWM limits of a servo firmware
   (:class:`ClampingCurrentPWM`) on its voltage.  Requires a motor.  A command
   clamp may take parameters from the motor
   (:attr:`CommandClampingBase.MOTOR_PARAMS`), so they are authored once.

**Motor**
   Provides the supply voltage, and turns the motor command into the effort a
   motor produces at the joint velocity, e.g. with back-EMF.  A motor may draw
   from a battery (see :ref:`actuator-battery`).  At most one motor per
   actuator.

**Clamping**
   Clamps effort to model actuator limits.  Examples include saturation,
   back-EMF losses, performance envelopes and position-dependent effort
   limits.  Multiple clamping stages can be combined
   on a single actuator.  With a motor, it clamps the effort of the motor.

The per-step pipeline is:

.. code-block:: text

   Input processors → Supply voltage → Drive → Command clamping → Motor → Clamping → Scatter-add → Drive state update

Drives, clamping and command clamping objects, and input processors are
pluggable: implement the :class:`DriveBase`, :class:`ClampingBase`,
:class:`CommandClampingBase` or :class:`InputProcessorBase` class to add new
models.

.. deprecated:: 1.8

   ``Delay`` is renamed :class:`InputProcessorDelay`.  The ``delay``
   argument of :class:`Actuator`, the ``delay_steps`` argument of
   :meth:`~newton.ModelBuilder.add_actuator`, :attr:`Actuator.delay`,
   :attr:`Actuator.State.delay_state`, ``ModelBuilder.ActuatorEntry.delay_args``,
   ``ComponentKind.DELAY``, and the
   ``update_state(target_pos, target_vel, feedforward, pos_indices,
   vel_indices, current_state, next_state)`` form of the delay are
   deprecated.  Pass the delay in ``input_processors`` and read its state
   from :attr:`Actuator.State.input_processor_states`.

.. deprecated:: 1.6

   The actuator ``Controller*`` class names are retained as compatibility
   aliases for :class:`DriveBase`, :class:`DrivePD`, :class:`DrivePID`,
   :class:`DriveNeuralMLP`, and :class:`DriveNeuralLSTM`. The former
   ``controller`` constructor keywords and attributes are also deprecated;
   use ``drive``, ``drive_class``, ``drive_state``, and ``drive_kwargs``.
   The clamping base class is now :class:`ClampingBase`; :class:`Clamping`
   remains as a deprecated compatibility alias.

.. note::

   **Current limitations:** the first version does not include a transmission
   model (gear ratios / linkage transforms), supports only one DOF per
   actuator, and does not model
   actuator dynamics (inertia, friction, thermal effects).

Usage
-----

Actuators are registered during model construction with
:meth:`~newton.ModelBuilder.add_actuator` and are instantiated automatically
when the model is finalized:

.. testsetup:: actuator-usage

   import warp as wp
   import newton
   from newton.actuators import (
       Actuator, ClampingMaxEffort, DrivePD, InputProcessorDelay,
   )

   builder = newton.ModelBuilder()
   link = builder.add_link()
   joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
   builder.add_articulation([joint])
   dof_index = builder.joint_qd_start[joint]

.. testcode:: actuator-usage

   builder.add_actuator(
       DrivePD,
       index=dof_index,
       kp=100.0,
       kd=10.0,
       input_processors=[(InputProcessorDelay, {"delay_steps": 5})],
       clamping=[(ClampingMaxEffort, {"max_effort": 50.0})],
   )

   model = builder.finalize()

For manual construction (outside of :class:`~newton.ModelBuilder`), compose the
components directly:

.. testcode:: actuator-usage

   indices = wp.array([0], dtype=wp.uint32)
   kp = wp.array([100.0], dtype=wp.float32)
   kd = wp.array([10.0], dtype=wp.float32)
   max_e = wp.array([50.0], dtype=wp.float32)

   actuator = Actuator(
       indices,
       drive=DrivePD(kp=kp, kd=kd),
       clamping=[ClampingMaxEffort(max_effort=max_e)],
       control_target_pos_attr="joint_target_q",
       control_target_vel_attr="joint_target_qd",
   )

Step an actuator with the :class:`newton.State` and :class:`newton.Control`
of the model:

.. testcode:: actuator-usage

   state = model.state()
   control = model.control()
   control.joint_f.zero_()
   actuator.step(state, control, dt=0.01)

The simulation state and control objects do not need to be a full
:class:`newton.State` / :class:`newton.Control`: any objects exposing joint
positions, velocities, control targets and an output effort array can be
used.  The attribute names are set in the :class:`Actuator` constructor with
``state_pos_attr``, ``state_vel_attr``, ``control_target_pos_attr``,
``control_target_vel_attr``, ``control_feedforward_attr`` and
``control_output_attr``.  This makes actuators reusable from a custom
simulator or test harness:

.. testcode:: actuator-usage

   import types

   actuator = Actuator(
       indices,
       drive=DrivePD(kp=kp, kd=kd),
       clamping=[ClampingMaxEffort(max_effort=max_e)],
       state_pos_attr="q",
       state_vel_attr="qd",
       control_target_pos_attr="q_target",
       control_target_vel_attr="qd_target",
       control_feedforward_attr=None,
       control_output_attr="effort",
   )

   sim_state = types.SimpleNamespace(
       q=wp.array([0.0], dtype=wp.float32),
       qd=wp.array([0.0], dtype=wp.float32),
   )
   sim_control = types.SimpleNamespace(
       q_target=wp.array([1.0], dtype=wp.float32),
       qd_target=wp.array([0.0], dtype=wp.float32),
       effort=wp.zeros(1, dtype=wp.float32),
   )

   sim_control.effort.zero_()
   actuator.step(sim_state, sim_control, dt=0.01)


.. _stateful-actuators:

Stateful Actuators
------------------

An actuator whose drive or input processors keep internal state requires
explicit double-buffered state management.  Stateful drives are, e.g.,
:class:`DrivePID` with an integral accumulator, :class:`DriveNeuralGRU` with
hidden state, and :class:`DriveNeuralLSTM` with hidden/cell state; a stateful
input processor is, e.g., :class:`InputProcessorDelay`.  Create two state
objects with :meth:`Actuator.state` and swap them after each step:

.. testcode:: actuator-usage

   state_0 = model.actuators[0].state()
   state_1 = model.actuators[0].state()
   state = model.state()
   control = model.control()

   for step in range(3):
       control.joint_f.zero_()  # zero output before stepping actuators
       model.actuators[0].step(state, control, state_0, state_1, dt=0.01)
       state_0, state_1 = state_1, state_0

The swap rebinds Python names, while a CUDA graph records fixed buffer
addresses. A captured region with an odd number of actuator steps therefore
cannot carry its final state into the next replay through the ordinary swap. A
single-step region updates its destination buffer, but does not advance state
across replays. Even-length regions end with the original buffer orientation
and need no special handling.

For an odd-length region, choose between assigning the state at the region
boundary or alternating graphs for the two buffer orientations.

Boundary assignment
^^^^^^^^^^^^^^^^^^^

Call :meth:`Actuator.State.assign` in place of the final swap. This uses one
graph and copies the actuator state once at the boundary of each replay.

.. code-block:: python

   with wp.ScopedCapture() as capture:
       for i in range(steps):
           control.joint_f.zero_()
           model.actuators[0].step(state, control, state_0, state_1, dt=0.01)
           if steps % 2 == 1 and i == steps - 1:
               state_0.assign(state_1)
           else:
               state_0, state_1 = state_1, state_0

Alternating graphs
^^^^^^^^^^^^^^^^^^

Key captured graphs by the current state buffer and alternate between them.
This avoids the boundary copy and keeps at most two graphs.

.. code-block:: python

   graphs = {}
   for _ in range(replays):
       key = id(state_0)  # one entry per buffer orientation
       if key not in graphs:
           with wp.ScopedCapture() as capture:
               after = run_region(state_0, state_1)  # ordinary swapping inside
           graphs[key] = (capture.graph, after)
       graph, (state_0, state_1) = graphs[key]
       wp.capture_launch(graph)

Stateless actuators (e.g. a plain PD drive without delay) do not require
state objects — simply omit them:

.. testcode:: actuator-usage

   # Build a stateless actuator (no delay, stateless drive)
   b2 = newton.ModelBuilder()
   lk = b2.add_link()
   jt = b2.add_joint_revolute(parent=-1, child=lk, axis=newton.Axis.Z)
   b2.add_articulation([jt])
   b2.add_actuator(DrivePD, index=b2.joint_qd_start[jt], kp=50.0)
   m2 = b2.finalize()

   m2.actuators[0].step(m2.state(), m2.control())

.. _custom-drive-inputs:

Custom Drive Inputs
-------------------

A drive can read extra arrays from ``sim_control`` beyond the positions,
velocities and targets the actuator already reads, e.g. an estimated load.  It lists
the array names in :attr:`DriveBase.custom_inputs`; each array has one entry
per DOF.

For actuators created by :class:`~newton.ModelBuilder`, the declared arrays are
registered on :class:`~newton.Control` automatically.  The caller writes them
before each :meth:`Actuator.step`; registration does not fill or clear them.

Users of another simulation engine pass their own arrays instead: any object
or mapping that provides the same-named arrays works as ``sim_state`` and
``sim_control`` in :meth:`Actuator.step`.  To see which names an actuator
needs, use :meth:`Actuator.sim_state` and :meth:`Actuator.sim_control`: they
return empty containers with exactly the fields the actuator reads, including
all declared custom inputs, ready to be filled with the engine's arrays.

.. warning::

   ``sim_state`` holds references. The simulation loop swaps ``state_0`` and
   ``state_1`` each step, so a ``sim_state`` built once from ``state_0`` still
   points at that buffer after the swap and the actuator reads stale positions
   and velocities. Re-point its fields every step.

See :ref:`custom_attributes` for custom attributes and frequencies in general.

In the loop below the drive declares one array named ``extra_input``:

.. code-block:: python

   sim_state = actuator.sim_state()
   sim_control = actuator.sim_control()
   sim_control.joint_target_q = target_positions
   sim_control.joint_target_qd = target_velocities
   sim_control.joint_f = control.joint_f

   for _ in range(num_steps):
       sim_state.joint_q = state_0.joint_q
       sim_state.joint_qd = state_0.joint_qd
       sim_control.extra_input = compute_extra_input(model, state_0)

       control.joint_f.zero_()
       actuator.step(sim_state, sim_control, dt=dt)

       solver.step(state_0, state_1, control, contacts, dt)
       state_0, state_1 = state_1, state_0

.. deprecated:: 1.8

   Declaring :attr:`DriveBase.custom_inputs` as a tuple of names is
   deprecated; the names are read as joint-DOF inputs.

.. _neural-network-checkpoints:

Neural-Network Checkpoints
--------------------------

Neural-network drives support `ONNX <https://onnx.ai/>`__ (``.onnx``), which
Warp-NN runs with its own Warp kernels. :class:`DriveNeuralMLP` and
:class:`DriveNeuralLSTM` additionally support Torch checkpoints through the
Torch backend; :class:`DriveNeuralGRU` uses ONNX only.

For :class:`DriveNeuralMLP` and :class:`DriveNeuralLSTM`, prefer ONNX exports
with dynamic batch axes: axis 0 of the MLP input and axis 1 of the LSTM
input, hidden state, and cell state. Existing fixed-batch exports remain
supported by adapting these input declarations without changing the source
checkpoint. This does not rewrite batch sizes embedded in graph operations
such as ``Reshape``; re-export those models with dynamic batch axes when
running a different number of actuators.

Torch checkpoints must be pt2 archives saved with ``torch.export.save``.
Use the ``.pt2`` extension; ``.pt`` and ``.pth`` are also accepted for these
archives.
Checkpoint metadata (scales and network configuration) is stored as a JSON
extra file:

.. code-block:: python

   import json
   import torch

   exported = torch.export.export(net, example_inputs)
   metadata = {"effort_scale": 2.0, "num_layers": 2, "hidden_size": 8}
   torch.export.save(exported, "policy.pt2", extra_files={"metadata.json": json.dumps(metadata)})

USD neural actuators may reference a checkpoint through ``newton:modelPath``.
Importing a USD rejects Torch checkpoints (``.pt2``, ``.pt``, or ``.pth``)
by default because :meth:`~newton.ModelBuilder.finalize` would load them with
``torch.export.load``, which can execute code from an untrusted checkpoint.
For a trusted USD asset and checkpoint, pass
``allow_unsafe_torch_checkpoints=True`` to
:meth:`newton.ModelBuilder.add_usd`. Programmatic construction of a neural
drive from a checkpoint path remains the caller's explicit trust decision.

:class:`DriveNeuralLSTM` requires ``num_layers`` and ``hidden_size`` in
the metadata of both pt2 and ONNX checkpoints.

:class:`DriveNeuralGRU` requires ``input_columns``, ``normalization`` and
``sample_dt_s`` in its ONNX checkpoint metadata, and an optional
``custom_inputs`` that marks one column of ``input_columns`` as an array the
application supplies each step.

.. _effort-modes:

Effort Modes
------------

By default an actuator computes effort **explicitly**: the control law is
evaluated at the current state and held constant over the step (zero-order
hold). At stiff gains and large timesteps this can overshoot or go unstable.

The **implicit** effort mode instead solves the control law against the
predicted end-of-step state (a Stable-PD style solve). A key advantage of this
formulation over the explicit effort mode is that it stays stable at higher
gains.

That stability comes at a cost: the solve reaches it by applying less effort
than the control law nominally asks for. The trade-off is between stability at
large timesteps with the implicit mode, and fidelity to the requested gains at
small timesteps with the explicit mode.

The implicit effort mode necessarily requires the joint-space inverse mass
matrix. This is supplied by a :class:`~newton.actuators.JointSpaceResponse`, which
is refreshed once per step at the current pose:

.. code-block:: python

   from newton.actuators import JointSpaceResponse

   response = JointSpaceResponse(model)
   actuator.set_effort_mode_implicit(response=response)

   # Simulation loop
   response.refresh(sim_state)
   sim_control.joint_f.zero_()
   actuator.step(sim_state, sim_control, state_a, state_b, dt=0.01)
   solver.step(sim_state, next_sim_state, sim_control, contacts, dt=0.01)

By default the predicted state accounts only for the actuator's own impulse.
Gravity, Coriolis forces and every other force the step applies are absent
from it. The drive still pushes back once those forces have moved the joint,
but it cannot anticipate them. For a single joint under a constant load ``g``
with ``kd = kp dt``, this leaves a steady error of ``g / kp + 2 dt^2 A g``: the
first part shrinks as the gain grows, the second part does not. Pass the forces
as a *bias force* to include them in the prediction. The bias force uses the
sign convention of the manipulator equation ``tau = M qdd + C qd + g``: it is
``C qd + g``, indexed like ``joint_qd``, and the step itself applies its
negative. The predicted end-of-step velocity
then becomes ``qd + A (p - dt bias_force)`` instead of ``qd + A p``, where
``A`` is the inverse mass matrix (see below) and ``p`` the actuator's impulse.

:func:`~newton.eval_inverse_dynamics_passive` returns the two parts,
``gravity_force`` ``g(q)`` and ``coriolis_force`` ``C(q, qd) qd``, so the bias
force is their sum. The function reads ``body_q``, which must match
``joint_q``; call :func:`~newton.eval_fk` first if it may not.

.. code-block:: python

   @wp.kernel
   def bias_force_kernel(
       gravity_force: wp.array[float],
       coriolis_force: wp.array[float],
       bias_force: wp.array[float],
   ):
       i = wp.tid()
       bias_force[i] = gravity_force[i] + coriolis_force[i]


   dt = 0.01
   n = model.joint_dof_count
   gravity_force = wp.zeros(n, dtype=float, device=model.device)
   coriolis_force = wp.zeros(n, dtype=float, device=model.device)
   bias_force = wp.zeros(n, dtype=float, device=model.device)

   # Simulation loop
   response.refresh(sim_state)
   newton.eval_inverse_dynamics_passive(
       model, sim_state, gravity_force=gravity_force, coriolis_force=coriolis_force
   )
   wp.launch(
       bias_force_kernel,
       dim=n,
       inputs=[gravity_force, coriolis_force],
       outputs=[bias_force],
       device=model.device,
   )
   sim_control.joint_f.zero_()
   actuator.step(sim_state, sim_control, state_a, state_b, dt=dt, bias_force=bias_force)
   solver.step(sim_state, next_sim_state, sim_control, contacts, dt=dt)

These are the forces at the start of the step. A solver that holds them
constant over the step, such as :class:`~newton.solvers.SolverMuJoCo` with its
default ``implicitfast`` integrator, applies the same values, so the
prediction matches its step up to the approximations of the response described
below. This bias force does not include contacts, joint limits, joint damping,
other actuators, or joint force applied without the actuator, such as other
writes to ``joint_f``. Contact and limit forces depend on the actuator's own
effort and cannot be computed before it.
:func:`~newton.eval_inverse_dynamics_passive` allocates scratch memory on every
call, so capturing it in a CUDA graph needs the device's memory pool. The
explicit effort mode ignores the bias force.

In this mode :meth:`Actuator.step <newton.actuators.Actuator.step>` evaluates
the joint force for each actuated DOF. A force on one DOF changes the velocity
of every DOF in its articulation, so an articulation's actuated DOFs are solved
together as one coupled system.

The inverse mass matrix, called *the response* below, is computed for a whole
articulation. The actuator then reads only the entries for the DOFs it drives.
:class:`~newton.actuators.JointSpaceResponse` is responsible for providing that
matrix, and there are two ways to obtain it: compute it from scratch
(:meth:`JointSpaceResponse.refresh <newton.actuators.JointSpaceResponse.refresh>`), or
reuse what the solver already has (:meth:`JointSpaceResponse.refresh_from_solve
<newton.actuators.JointSpaceResponse.refresh_from_solve>`).

:meth:`~newton.actuators.JointSpaceResponse.refresh` builds the mass matrix itself,
from :func:`~newton.eval_mass_matrix` and joint armature. This comes with
approximations. First, joint damping, joint limits, friction, contacts and
constraint regularization are absent. All of those resist motion, so the
response comes out larger than anticipated. A larger response further divides
the control law and so yields a smaller effort than would have been evaluated
without the simplifications listed above. Second, kinematic loop closures are
also ignored.

The approximations inherent in :meth:`JointSpaceResponse.refresh
<newton.actuators.JointSpaceResponse.refresh>` may be avoided when working with a
solver that is able to evaluate the inverse mass matrix more directly, with the
exception of loop closure effects. To this end,
:meth:`JointSpaceResponse.refresh_from_solve
<newton.actuators.JointSpaceResponse.refresh_from_solve>` takes a callable that
computes ``x = M^-1 y``. The response recovers the matrix one column at a time,
by passing unit vectors through that callable. MuJoCo is currently the only
Newton solver that provides one. MuJoCo's free and ball joints use different
coordinates from Newton's, so pass the state as well; the response then
converts them at that pose.

.. code-block:: python

   def solve_inverse(x, y):
       # x = M^-1 y, using the factorization the solver already built
       mujoco_warp.solve_m(solver.mjw_model, solver.mjw_data, x, y)

   # Simulation loop, in place of response.refresh(sim_state)
   response.refresh_from_solve(solve_inverse, dof_map=solver.mjc_dof_to_newton_dof, state=sim_state)

Both refresh paths launch only kernels, so the actuator, the solver step and the
response update can be captured in one CUDA graph.

:meth:`~newton.actuators.Actuator.set_effort_mode_explicit` switches back to
explicit mode. :class:`~newton.actuators.Actuator.ImplicitOptions` sets the
solve's iteration count, line-search length and convergence tolerances.

The following built-in drives support the implicit mode:
:class:`~newton.actuators.DrivePD`,
:class:`~newton.actuators.DrivePID`,
:class:`~newton.actuators.DriveNeuralMLP` and
:class:`~newton.actuators.DriveNeuralLSTM`.
The neural drives enter the solve as a per-step linearization of the
network, whose slopes come from a Warp autodiff pass over the loaded network.

Limitations of the implicit mode.  In the cases marked *error*,
:meth:`~newton.actuators.Actuator.set_effort_mode_implicit` raises
``NotImplementedError`` instead of falling back silently:

* **Other drives** (error), e.g. :class:`~newton.actuators.DriveNeuralGRU`:
  they define no :attr:`~newton.actuators.DriveBase.evaluate_force` law for
  the solve.
* **Motors** (error): the solve does not model the motor stage, e.g.
  :class:`~newton.actuators.MotorDC`.
* **Torch neural checkpoints** (error): the linearization needs Warp autodiff
  over the network, which only the ONNX backend provides (see
  :ref:`neural-network-checkpoints`).
  :class:`~newton.actuators.DriveNeuralMLP` also needs a single-step input
  history (``input_idx == [0]``).
* **Differentiable actuators** (error): an actuator built with
  ``requires_grad=True`` cannot be solved implicitly, because the solve has no
  adjoint.
* **Custom inputs without** :meth:`~newton.actuators.DriveBase.prepare_implicit`
  (error): a drive that declares
  :attr:`~newton.actuators.DriveBase.custom_inputs` must pass them to the
  solve, or the solve would ignore them.
* **Backlash is not modeled** (warning): an actuator with an
  :class:`~newton.actuators.InputProcessorEncoder` is accepted with a
  :class:`~newton.exceptions.NewtonWarning`, because the result is only
  approximate.  The drive reads the encoder behind the output joint, while the
  solve predicts the end-of-step state from the response of the actuated joint
  alone.  While the backlash gap is open, the motor's effort does not reach
  the link, yet the solve assumes it does, so it applies less effort than
  needed to close the gap.

Differentiability and Graph Capture
-----------------------------------

Whether an actuator supports differentiability and CUDA graph capture depends on
its drive.  :class:`DrivePD` and :class:`DrivePID` are fully
graphable.  For neural-network drives it depends on the checkpoint
backend: ONNX checkpoints are graphable, while Torch checkpoints are not due
to framework interop overhead.  :meth:`Actuator.is_graphable` returns ``True``
when all components can be captured in a CUDA graph.

:meth:`Actuator.is_graphable` describes the components, not the captured region.
A stateful actuator also needs the region's state exchange to be graph-safe.  See
:ref:`stateful-actuators` for the two patterns that keep an odd-length region
correct. Torch-backed neural drives are not graphable and cannot be used in a
captured region.

Available Components
--------------------

Input processors
^^^^^^^^^^^^^^^^

* :class:`InputProcessorDelay` — circular-buffer delay for control targets (stateful).
* :class:`InputProcessorEncoder` — position encoder, e.g. behind a backlash
  joint (stateless).

Multiple input processors can be stacked on a single actuator; they are applied
in sequence.

Drives
^^^^^^

* :class:`DrivePD` — proportional-derivative control law (stateless).
* :class:`DrivePID` — proportional-integral-derivative control law
  (stateful: integral accumulator with anti-windup clamp).
* :class:`DriveNeuralGRU` — GRU neural-network drive
  (stateful: hidden state; explicit Warp-NN inference).
* :class:`DriveNeuralMLP` — MLP neural-network drive
  (stateful: position/velocity history buffers).
* :class:`DriveNeuralLSTM` — LSTM neural-network drive
  (stateful: hidden/cell state).
* :class:`DriveVoltageControlled` — firmware position controller of a
  voltage-controlled servo as in BAM, which outputs a voltage for a motor
  (stateless).

See the API documentation for each drive's control-law equations.

Clamping
^^^^^^^^

* :class:`ClampingMaxEffort` — symmetric clamp to ±max_effort per actuator.
* :class:`ClampingDCMotor` — velocity-dependent effort saturation using the DC
  motor effort-speed characteristic.
* :class:`ClampingPositionBased` — position-dependent effort limits via
  interpolated lookup table (e.g. for linkage-driven joints).

Multiple clamping objects can be stacked on a single actuator; they are applied
in sequence.

Command clamping
^^^^^^^^^^^^^^^^

* :class:`ClampingCurrentPWM` — current and PWM limits of the firmware of a
  servo as in BAM, on the voltage command of a :class:`MotorDC`, applied in
  this order.

Command clamps are passed in the same ``clamping`` list as the other clamps and
run in sequence before the motor.

Motors
^^^^^^

* :class:`MotorDC` — DC motor driven by a PWM voltage as in BAM (Better
  Actuator Models): takes a voltage command and models the back-EMF,
  optionally on a battery.  It models the back-EMF itself, so do not combine
  it with :class:`ClampingDCMotor`.

.. _actuator-battery:

Battery
-------

A battery powers the motors of one or more actuators.  Battery models subclass
:class:`BatteryBase`.  See the individual battery class documentation for the
model equations.

A battery reads one array and writes one array on ``control``, each with one
entry per actuator of the model
(:attr:`Model.AttributeFrequency.ACTUATOR <newton.Model.AttributeFrequency.ACTUATOR>`):

* input ``control.actuator_supply_current``: the current each motor draws
  from its battery, written by the motor in :meth:`Actuator.step`;
* output ``control.actuator_supply_voltage``: the supply voltage of each
  motor, read by the motor in the next :meth:`Actuator.step`.

* :meth:`~newton.ModelBuilder.add_actuator` returns the index of the actuator
  in per-actuator arrays.  :meth:`~newton.ModelBuilder.add_battery` takes a
  battery class, these indices and the battery parameters, and adds a battery
  for those actuators in the current world.
  :meth:`~newton.ModelBuilder.replicate` copies the batteries into every
  world.  :meth:`~newton.ModelBuilder.finalize` creates one object per battery
  class in ``model.batteries``, holding all batteries of that class.  Its
  parameter arrays have one entry per battery, and :attr:`BatteryBase.world`
  maps each battery to its world, e.g. to randomize the parameters per world.
  :attr:`BatteryBase.actuators` and :attr:`BatteryBase.actuator_start` give
  the actuators of each battery.
* :meth:`BatteryBase.step` updates the supply voltage from the current.  Call
  it once per step after the actuators.
* :meth:`BatteryBase.reset` resets all batteries or the batteries in the worlds
  of a mask.  Call it after resetting worlds.  As in :meth:`SolverBase.reset() <newton.solvers.SolverBase.reset>`,
  the mask has ``world_count + 1`` entries, and the last one selects batteries
  outside any world.

Below, the hip and knee share one battery and the ankle draws from another:

.. testcode:: actuator-battery

   import newton
   from newton.actuators import BatteryConstantResistance, ClampingCurrentPWM, DriveVoltageControlled, MotorDC

   builder = newton.ModelBuilder()
   actuators = []
   for name in ("hip", "knee", "ankle"):
       link = builder.add_link()
       builder.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
       joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z, label=name)
       builder.add_articulation([joint])
       actuator = builder.add_actuator(
           DriveVoltageControlled,
           index=builder.joint_qd_start[joint],
           kp=5.0,
           error_gain=0.1,
           clamping=[(ClampingCurrentPWM, {"max_current": 2.0, "max_pwm": 0.9})],
           motor=(MotorDC, {"kt": 0.5, "resistance": 2.0, "vin": 12.0, "supply_from_battery": True}),
       )
       actuators.append(actuator)
   builder.add_battery(BatteryConstantResistance, actuators[:2], nominal_voltage=12.0, resistance=0.2, min_voltage=9.0)
   builder.add_battery(BatteryConstantResistance, actuators[2:], nominal_voltage=12.0, resistance=0.2, min_voltage=9.0)
   model = builder.finalize()

   solver = newton.solvers.SolverFeatherstone(model)
   state_0, state_1 = model.state(), model.state()
   control = model.control()
   control.joint_target_q.fill_(1.0)
   dt = 0.01

   for _ in range(10):
       control.joint_f.zero_()
       for actuator in model.actuators:
           actuator.step(state_0, control, dt=dt)
       for battery in model.batteries:
           battery.step(control)
       solver.step(state_0, state_1, control, None, dt)
       state_0, state_1 = state_1, state_0

Customization
-------------

Any actuator can be assembled from the existing building blocks — mix and
match drives, clamping stages, input processors and motors to fit a specific use case.
When the built-in components are not sufficient, implement new ones by
subclassing :class:`DriveBase`, :class:`ClampingBase` or
:class:`InputProcessorBase`.

For example, a custom drive needs to implement
:meth:`~DriveBase.compute`, :meth:`~DriveBase.resolve_arguments`,
:meth:`~DriveBase.is_stateful`, and :meth:`~DriveBase.is_graphable`:

.. code-block:: python
   :caption: Skeleton — the ``compute`` body is omitted; see existing
             drives for complete examples.

   import warp as wp
   from newton.actuators import DriveBase

   class MyDrive(DriveBase):
       @classmethod
       def resolve_arguments(cls, args):
           return {"gain": args.get("gain", 1.0)}

       def __init__(self, gain: wp.array):
           self.gain = gain

       def is_stateful(self):
           return False

       def is_graphable(self):
           return True

       def compute(self, positions, velocities, target_pos, target_vel,
                   feedforward, pos_indices, vel_indices,
                   target_pos_indices, target_vel_indices,
                   forces, state, dt, device=None, *, custom_inputs=None):
           # Launch a Warp kernel that writes effort into `forces`
           ...

``resolve_arguments`` maps user-provided keyword arguments (from
:meth:`~newton.ModelBuilder.add_actuator` or USD schemas) to constructor
parameters, filling in defaults where needed.

A stateful custom drive also defines a dataclass subclass of
:class:`DriveBase.State` and implements :meth:`~DriveBase.State.reset`. The
default :meth:`Actuator.State.assign` behavior copies direct Warp array and
Torch tensor fields without replacing their storage. States with other field
types or nested storage implement ``assign()`` to define that copy.

A drive that needs additional per-step arrays declares them through
:attr:`DriveBase.custom_inputs`. They arrive in the ``custom_inputs`` argument
to :meth:`~DriveBase.compute` after being read from ``sim_control``; see
:ref:`custom-drive-inputs`.

A custom drive works in the explicit mode with the methods above. To also
support the implicit mode it provides three more things, because the solve
evaluates the control law inside its own kernel rather than calling
:meth:`~DriveBase.compute`:

* :attr:`~DriveBase.evaluate_force` — a ``@wp.func`` holding the control law.
  The solve calls it at the predicted state, so it must read every parameter it
  needs from one packed row rather than from the drive's own arrays.
* :meth:`~DriveBase.bind_params` — packs those parameters into an
  ``(num_actuators, P)`` array and re-points the drive's public arrays at
  its columns, so later writes stay visible to the solve.
* :meth:`~DriveBase.prepare_implicit` — optional, called once per step before
  the solve. Use it for parameters that depend on the current state, such as a
  PID integral term or a network linearization. Drives with fixed
  parameters do not override it.

.. code-block:: python
   :caption: Adding implicit support to ``MyDrive``.

   @wp.func
   def _my_force(q: wp.float64, qd: wp.float64, target_q: wp.float64,
                 target_qd: wp.float64, feedforward: wp.float64,
                 params: wp.array2d[float], i: wp.int32) -> wp.float64:
       return wp.float64(params[i, 0]) * (target_q - q)

   class MyDrive(DriveBase):
       evaluate_force = _my_force

       def bind_params(self):
           pack = wp.zeros((len(self.gain), 1), dtype=float, device=self.gain.device)
           pack[:, 0].assign(self.gain)
           self.gain = pack[:, 0]   # writes to self.gain now reach the solve
           return pack

Returning ``None`` from :meth:`~DriveBase.bind_params` declares that this
configuration cannot be solved implicitly.
:meth:`~newton.actuators.Actuator.set_effort_mode_implicit` then raises
``NotImplementedError`` rather than falling back silently, as it does for a
Torch-backed neural checkpoint. Leaving
:attr:`~DriveBase.evaluate_force` as ``None`` raises the same error.

Similarly, a custom clamping stage subclasses :class:`ClampingBase` and
implements :meth:`~ClampingBase.modify_forces` (which reads effort from a source
buffer and writes bounded effort to a destination buffer).  A clamp of the
motor command subclasses :class:`CommandClampingBase` instead, whose
:meth:`~CommandClampingBase.modify_forces` also receives the supply voltage.

See Also
--------

* :mod:`newton.actuators` — full API reference
* :meth:`newton.ModelBuilder.add_actuator` — registering actuators during
  model construction
