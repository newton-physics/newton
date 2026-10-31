.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton.actuators

Actuators
=========

.. warning::

   The actuator API is **experimental** and may change in future releases.
   Feedback is welcome — please file issues or discussion threads.

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
controller implementation supports it.  Actuators are designed to be easy to
customize and extend for specific actuator models.

Architecture
------------

An actuator is composed from three building blocks, applied in this order:

.. code-block:: text

   Actuator
   ├── Delay       (optional: delays command inputs by N actuator timesteps)
   ├── Controller  (control law that computes raw effort)
   └── Clamping[]  (clamps raw effort based on motor-limit modeling)
       ├── ClampingMaxEffort        (±max_effort symmetric clamp)
       ├── ClampingDCMotor         (velocity-dependent saturation)
       └── ClampingPositionBased   (position-dependent lookup table)

**Delay**
   Optionally delays command inputs (control targets and feedforward terms)
   by *N* actuator timesteps before they reach the controller, modeling
   communication or processing latency.  The delay always produces output;
   when the buffer is empty or a DOF has ``delay_steps == 0``, the current
   command inputs are used directly.  When underfilled, the lag is clamped
   to the available history so the oldest available entry is returned.

**Controller**
   Computes raw actuator effort [N or N·m] from the current simulator state
   and control targets.  This is the actuator's control law — for example PD,
   PID, or neural-network-based control.  See the individual controller class
   documentation for the control-law equations.

**Clamping**
   Clamps raw effort based on motor-limit modeling.  This applies
   post-controller output limits to the computed effort to model motor limits
   such as saturation, back-EMF losses, performance envelopes, or
   position-dependent effort limits.  Multiple clamping stages can be combined
   on a single actuator.

The per-step pipeline is:

.. code-block:: text

   Delay read → Controller → Clamping → Scatter-add → State updates (controller + delay write)

Controllers and clamping objects are pluggable: implement the
:class:`Controller` or :class:`Clamping` base class to add new models.

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
       Actuator, ClampingMaxEffort, ControllerPD, Delay,
   )

   builder = newton.ModelBuilder()
   link = builder.add_link()
   joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
   builder.add_articulation([joint])
   dof_index = builder.joint_qd_start[joint]

.. testcode:: actuator-usage

   builder.add_actuator(
       ControllerPD,
       index=dof_index,
       kp=100.0,
       kd=10.0,
       delay_steps=5,
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
       controller=ControllerPD(kp=kp, kd=kd),
       delay=Delay(delay_steps=wp.array([5], dtype=wp.int32), max_delay=5),
       clamping=[ClampingMaxEffort(max_effort=max_e)],
   )


Stateful Actuators
------------------

Controllers that maintain internal state (e.g. :class:`ControllerPID` with an
integral accumulator, or :class:`ControllerNeuralLSTM` with hidden/cell state) and
actuators with a :class:`Delay` require explicit double-buffered state
management.  Create two state objects with :meth:`Actuator.state` and swap them
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

Stateless actuators (e.g. a plain PD controller without delay) do not require
state objects — simply omit them:

.. testcode:: actuator-usage

   # Build a stateless actuator (no delay, stateless controller)
   b2 = newton.ModelBuilder()
   lk = b2.add_link()
   jt = b2.add_joint_revolute(parent=-1, child=lk, axis=newton.Axis.Z)
   b2.add_articulation([jt])
   b2.add_actuator(ControllerPD, index=b2.joint_qd_start[jt], kp=50.0)
   m2 = b2.finalize()

   m2.actuators[0].step(m2.state(), m2.control())

Differentiability and Graph Capture
-----------------------------------

Whether an actuator supports differentiability and CUDA graph capture depends on
its controller.  :class:`ControllerPD` and :class:`ControllerPID` are fully
graphable.  Neural-network controllers (:class:`ControllerNeuralMLP`,
:class:`ControllerNeuralLSTM`) require PyTorch and are not graphable due to
framework interop overhead.

:meth:`Actuator.is_graphable` returns ``True`` when all components can be
captured in a CUDA graph.

Available Components
--------------------

Delay
^^^^^

* :class:`Delay` — circular-buffer delay for control targets (stateful).

Controllers
^^^^^^^^^^^

* :class:`ControllerPD` — proportional-derivative control law (stateless).
* :class:`ControllerPID` — proportional-integral-derivative control law
  (stateful: integral accumulator with anti-windup clamp).
* :class:`ControllerNeuralMLP` — MLP neural-network controller (requires
  PyTorch, stateful: position/velocity history buffers).
* :class:`ControllerNeuralLSTM` — LSTM neural-network controller (requires
  PyTorch, stateful: hidden/cell state).

See the API documentation for each controller's control-law equations.

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
match controllers, clamping stages, and delay to fit a specific use case.
When the built-in components are not sufficient, implement new ones by
subclassing :class:`Controller` or :class:`Clamping`.

For example, a custom controller needs to implement
:meth:`~Controller.compute`, :meth:`~Controller.resolve_arguments`,
:meth:`~Controller.is_stateful`, and :meth:`~Controller.is_graphable`:

.. code-block:: python
   :caption: Skeleton — the ``compute`` body is omitted; see existing
             controllers for complete examples.

   import warp as wp
   from newton.actuators import Controller

   class MyController(Controller):
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

Similarly, a custom clamping stage subclasses :class:`Clamping` and implements
:meth:`~Clamping.modify_forces` (which reads effort from a source buffer and writes bounded effort to a destination buffer).

See Also
--------

* :mod:`newton.actuators` — full API reference
* :meth:`newton.ModelBuilder.add_actuator` — registering actuators during
  model construction
