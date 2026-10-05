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
need to be part of Newton: actuators are designed to be reusable anywhere the
caller can provide state arrays and consume effort.

Each :class:`Actuator` instance is **vectorized**: a single actuator object
operates on a batch of DOF indices in global state and control arrays, allowing
efficient integration into RL workflows with many parallel environments.

The goal is to provide canonical actuator models with support for
**differentiability** and **graphable execution** where the underlying
drive implementation supports it.  Actuators are designed to be easy to
customize and extend for specific actuator models.

Architecture
------------

An actuator is composed from three building blocks, applied in this order:

.. code-block:: text

   Actuator
   ├── InputProcessor[]  (optional: transform state and command inputs)
   │   ├── InputProcessorDelay       (delays command inputs by N actuator timesteps)
   │   ├── InputProcessorRandomDelay (delays command inputs by a randomly redrawn lag)
   │   └── InputProcessorBacklash   (reads position on the link side of a backlash joint)
   ├── Drive      (control law that computes raw effort)
   └── Clamping[]  (clamps raw effort based on motor-limit modeling)
       ├── ClampingMaxEffort        (±max_effort symmetric clamp)
       ├── ClampingDCMotor         (velocity-dependent saturation)
       └── ClampingPositionBased   (position-dependent lookup table)

**Input processors**
   Optionally transform the inputs of the drive: the simulation state it
   reads (positions, velocities) and the commands (control targets and
   feedforward terms).  Processors run in list order; each one receives the
   inputs returned by the previous one.

   :class:`InputProcessorDelay` delays command inputs by *N* actuator timesteps, modeling
   communication or processing latency.  The delay always produces output;
   when the buffer is empty or a DOF has ``delay_steps == 0``, the current
   command inputs are used directly.  When underfilled, the lag is clamped
   to the available history so the oldest available entry is returned.

   :class:`InputProcessorRandomDelay` draws the lag of each DOF at random
   from ``[min_delay, max_delay]`` and redraws it during the episode, every
   ``update_period`` steps and with a ``hold_probability`` of keeping it.
   DOFs that name the same ``lag_joint`` share one lag, e.g. all servos of
   one robot.  Each reset starts a new random stream.

   :class:`InputProcessorBacklash` models an encoder on the link side of a
   backlash joint.  The backlash itself is a passive joint with a small
   motion range, placed in series between the actuated joint and the link;
   the solver handles the dead zone.  The drive reads the sum of the actuated
   and the backlash joint positions and velocities.

   In USD, processors keep the order of the actuator prim's ``apiSchemas``,
   and a joint parameter is a relationship, e.g. ``rel newton:backlashJoint``.

**Drive**
   Computes raw actuator effort [N or N·m] from the current simulator state
   and control targets.  This is the actuator's control law — for example PD,
   PID, or neural-network-based control.  See the individual drive class
   documentation for the control-law equations.

**Clamping**
   Clamps raw effort based on motor-limit modeling.  This applies
   post-drive output limits to the computed effort to model motor limits
   such as saturation, back-EMF losses, performance envelopes, or
   position-dependent effort limits.  Multiple clamping stages can be combined
   on a single actuator.

The per-step pipeline is:

.. code-block:: text

   Input processors (read, then write their next state) → Drive → Clamping → Scatter-add → Drive state update

Drives, clamping objects, and input processors are pluggable: implement the
:class:`DriveBase`, :class:`ClampingBase`, or :class:`InputProcessorBase`
class to add new models.

.. deprecated:: 1.7

   ``Delay`` is renamed :class:`InputProcessorDelay`.  The ``delay``
   argument of :class:`Actuator`, the ``delay_steps`` argument of
   :meth:`~newton.ModelBuilder.add_actuator`, :attr:`Actuator.delay`,
   :attr:`Actuator.State.delay_state`, and ``ComponentKind.DELAY`` are
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
   model (gear ratios / linkage transforms), supports only single-input
   single-output (SISO) actuators (one DOF per actuator), and does not model
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
       input_processors=[InputProcessorDelay(delay_steps=wp.array([5], dtype=wp.int32), max_delay=5)],
       clamping=[ClampingMaxEffort(max_effort=max_e)],
       control_target_pos_attr="joint_target_q",
       control_target_vel_attr="joint_target_qd",
   )

The simulator state and control objects do not need to be a full
:class:`newton.Model` / :class:`newton.Control` — any objects exposing
``joint_q``, ``joint_qd``, ``joint_target_q``, ``joint_target_qd``,
``joint_act`` (optional), and ``joint_f`` will do.  This makes actuators
reusable from a custom simulator or test harness:

.. testcode:: actuator-usage

   import types

   sim_state = types.SimpleNamespace(
       joint_q=wp.array([0.0], dtype=wp.float32),
       joint_qd=wp.array([0.0], dtype=wp.float32),
   )
   sim_control = types.SimpleNamespace(
       joint_target_q=wp.array([1.0], dtype=wp.float32),
       joint_target_qd=wp.array([0.0], dtype=wp.float32),
       joint_act=None,
       joint_f=wp.zeros(1, dtype=wp.float32),
   )

   state_a = actuator.state()
   state_b = actuator.state()
   sim_control.joint_f.zero_()
   actuator.step(sim_state, sim_control, state_a, state_b, dt=0.01)


.. _stateful-actuators:

Stateful Actuators
------------------

Drives that maintain internal state (e.g. :class:`DrivePID` with an
integral accumulator, or :class:`DriveNeuralLSTM` with hidden/cell state) and
actuators with stateful input processors (e.g. :class:`InputProcessorDelay`)
require explicit double-buffered state management.  Create two state objects with :meth:`Actuator.state` and swap them
after each step:

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

.. _neural-network-checkpoints:

Neural-Network Checkpoints
--------------------------

Neural-network drives (:class:`DriveNeuralMLP`,
:class:`DriveNeuralLSTM`) support two checkpoint backends. `ONNX
<https://onnx.ai/>`__ (``.onnx``) is an open format for trained networks, which
Warp-NN runs with its own Warp kernels. Torch checkpoints use the Torch backend
and require PyTorch.

Torch checkpoints are pt2 archives (``.pt2``) saved with ``torch.export.save``.
Checkpoint metadata (scales and network configuration) is stored as a JSON
extra file:

.. code-block:: python

   import json
   import torch

   exported = torch.export.export(net, example_inputs)
   metadata = {"effort_scale": 2.0, "num_layers": 2, "hidden_size": 8}
   torch.export.save(exported, "policy.pt2", extra_files={"metadata.json": json.dumps(metadata)})

:class:`DriveNeuralLSTM` requires ``num_layers`` and ``hidden_size`` in
the metadata of both pt2 and ONNX checkpoints.

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

The predicted state accounts only for the actuator's own impulse. Gravity, any
other applied force, other actuators driving the same articulation, and joint
drive applied without the actuator are all absent from it.

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
Newton solver that provides one.

.. code-block:: python

   def solve_inverse(x, y):
       # x = M^-1 y, using the factorization the solver already built
       mujoco_warp.solve_m(solver.mjw_model, solver.mjw_data, x, y)

   # Simulation loop, in place of response.refresh(sim_state)
   response.refresh_from_solve(solve_inverse, dof_map=solver.mjc_dof_to_newton_dof)

Both refresh paths launch only kernels, so the actuator, the solver step and the
response update can be captured in one CUDA graph.

:meth:`~newton.actuators.Actuator.set_effort_mode_explicit` switches back to
explicit mode. :class:`~newton.actuators.Actuator.ImplicitOptions` sets the
solve's iteration count and convergence tolerances.

The following built-in drives support the implicit mode (neural drives require
the ONNX backend):
:class:`~newton.actuators.DrivePD`,
:class:`~newton.actuators.DrivePID`,
:class:`~newton.actuators.DriveNeuralMLP` and
:class:`~newton.actuators.DriveNeuralLSTM`.

The neural drives enter the solve as a per-step linearization of the
network. Its slopes come from a Warp autodiff pass over the loaded network, so
only the ONNX backend supports them (see :ref:`neural-network-checkpoints`);
with a Torch checkpoint
:meth:`~newton.actuators.Actuator.set_effort_mode_implicit` raises
``NotImplementedError``. :class:`~newton.actuators.DriveNeuralMLP` also
needs a single-step input history (``input_idx == [0]``).

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
* :class:`InputProcessorRandomDelay` — delay with a randomly redrawn lag
  (stateful).
* :class:`InputProcessorBacklash` — position and velocity on the link side of
  a backlash joint (stateless).  :meth:`~newton.ModelBuilder.add_actuator`
  takes the backlash joint as a joint index or joint label in
  ``backlash_joint``.

Drives
^^^^^^

* :class:`DrivePD` — proportional-derivative control law (stateless).
* :class:`DrivePID` — proportional-integral-derivative control law
  (stateful: integral accumulator with anti-windup clamp).
* :class:`DriveNeuralMLP` — MLP neural-network drive
  (stateful: position/velocity history buffers).
* :class:`DriveNeuralLSTM` — LSTM neural-network drive
  (stateful: hidden/cell state).
* :class:`DriveBAM` — BAM servo model: firmware gain, PWM and current
  limits, DC motor with back-EMF (stateless).  BAM's gearbox friction is not
  included.

See the API documentation for each drive's control-law equations.

Battery
^^^^^^^

* :class:`Battery` — supply shared by the actuators of one or more DOFs.  Its
  voltage sags with the total motor torque drawn from it, down to a minimum.
  Pass it to :meth:`Actuator.step` and call :meth:`Battery.refresh` once per
  step before stepping the actuators.  Drives with
  :meth:`DriveBase.uses_battery` (e.g. :class:`DriveBAM`) read its voltage and
  add their motor torque; the sag of a step therefore comes from the previous
  step's torque of all actuators on that battery, also across actuator
  groups.  ``Battery.dof_battery`` maps each DOF to its battery.

Clamping
^^^^^^^^

* :class:`ClampingMaxEffort` — symmetric clamp to ±max_effort per actuator.
* :class:`ClampingDCMotor` — velocity-dependent effort saturation using the DC
  motor effort-speed characteristic.
* :class:`ClampingPositionBased` — position-dependent effort limits via
  interpolated lookup table (e.g. for linkage-driven joints).

Multiple clamping objects can be stacked on a single actuator; they are applied
in sequence.

Customization
-------------

Any actuator can be assembled from the existing building blocks — mix and
match drives, clamping stages, and input processors to fit a specific use case.
When the built-in components are not sufficient, implement new ones by
subclassing :class:`DriveBase`, :class:`ClampingBase`, or
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
                   forces, state, dt, device=None):
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
buffer and writes bounded effort to a destination buffer).

See Also
--------

* :mod:`newton.actuators` — full API reference
* :meth:`newton.ModelBuilder.add_actuator` — registering actuators during
  model construction
