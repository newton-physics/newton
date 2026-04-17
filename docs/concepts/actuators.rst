.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton.actuators

Actuators
=========

Actuators provide composable implementations that read physics simulation
state, compute forces, and write the forces back to control arrays for
application to the simulation.  The simulator does not need to be part of
Newton: actuators are designed to be reusable anywhere the caller can provide
state arrays and consume forces.

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
   ├── Delay       (optional: delays control targets by N timesteps)
   ├── Controller  (control law that computes raw forces)
   └── Clamping[]  (clamps raw forces based on motor-limit modeling)
       ├── ClampingMaxForce        (±max_force box clamp)
       ├── ClampingDCMotor         (velocity-dependent saturation)
       └── ClampingPositionBased   (angle-dependent lookup table)

**Delay**
   Optionally delays the control targets (e.g. position or velocity) by *N*
   timesteps before they reach the controller, allowing the actuator to model
   communication or processing latency.  The delay always produces output;
   when the buffer is still filling, the lag is clamped to the available
   history so the most recent data is returned.

**Controller**
   Computes raw forces or torques from the current simulator state and control
   targets.  This is the actuator's control law — for example PD, PID, or
   neural-network-based control.  See the individual controller class
   documentation for the control-law equations.

**Clamping**
   Clamps raw forces based on motor-limit modeling.  This applies
   post-controller output limits to the computed forces or torques to model
   motor limits such as saturation, back-EMF losses, performance envelopes, or
   angle-dependent torque limits.  Multiple clamping stages can be combined on
   a single actuator.

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

.. code-block:: python

   import newton
   from newton.actuators import ClampingMaxForce, ControllerPD

   builder = newton.ModelBuilder()
   # ... add links, joints, articulations ...

   builder.add_actuator(
       ControllerPD,
       index=dof_index,
       kp=100.0,
       kd=10.0,
       delay=5,
       clamping=[(ClampingMaxForce, {"max_force": 50.0})],
   )

   model = builder.finalize()

For manual construction (outside of :class:`~newton.ModelBuilder`), compose the
components directly:

.. code-block:: python

   import warp as wp
   from newton.actuators import Actuator, ControllerPD, ClampingMaxForce, Delay

   indices = wp.array([0, 1, 2], dtype=wp.uint32, device="cuda:0")
   kp = wp.array([100.0, 100.0, 100.0], dtype=wp.float32, device="cuda:0")
   kd = wp.array([10.0, 10.0, 10.0], dtype=wp.float32, device="cuda:0")
   max_f = wp.array([50.0, 50.0, 50.0], dtype=wp.float32, device="cuda:0")

   actuator = Actuator(
       indices,
       controller=ControllerPD(kp=kp, kd=kd),
       delay=Delay(delay=wp.array([5, 5, 5], dtype=wp.int32, device="cuda:0"), max_delay=5),
       clamping=[ClampingMaxForce(max_force=max_f)],
   )

   # In the simulation loop:
   actuator.step(sim_state, sim_control, state_a, state_b, dt=0.01)


Stateful Actuators
------------------

Controllers that maintain internal state (e.g. :class:`ControllerPID` with an
integral accumulator, or :class:`ControllerNetLSTM` with hidden/cell state) and
actuators with a :class:`Delay` require explicit double-buffered state
management.  Create two state objects with :meth:`Actuator.state` and swap them
after each step:

.. code-block:: python

   state_a = actuator.state()
   state_b = actuator.state()

   for step in range(num_steps):
       actuator.step(sim_state, sim_control, state_a, state_b, dt=dt)
       state_a, state_b = state_b, state_a  # swap

Stateless actuators (e.g. a plain PD controller without delay) do not require
state objects — simply omit them:

.. code-block:: python

   actuator.step(sim_state, sim_control)

Differentiability and Graph Capture
-----------------------------------

Whether an actuator supports differentiability and CUDA graph capture depends on
its controller.  :class:`ControllerPD` and :class:`ControllerPID` are fully
graphable.  Neural-network controllers (:class:`ControllerNetMLP`,
:class:`ControllerNetLSTM`) require PyTorch and are not graphable due to
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
* :class:`ControllerNetMLP` — MLP neural-network controller (requires
  PyTorch, stateful: position/velocity history buffers).
* :class:`ControllerNetLSTM` — LSTM neural-network controller (requires
  PyTorch, stateful: hidden/cell state).

See the API documentation for each controller's control-law equations.

Clamping
^^^^^^^^

* :class:`ClampingMaxForce` — symmetric box clamp to ±max_force per actuator.
* :class:`ClampingDCMotor` — velocity-dependent torque saturation using the DC
  motor torque-speed characteristic.
* :class:`ClampingPositionBased` — angle-dependent torque limits via
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
:meth:`~Controller.compute` and :meth:`~Controller.resolve_arguments`:

.. code-block:: python

   import warp as wp
   from newton.actuators import Controller

   class MyController(Controller):
       @classmethod
       def resolve_arguments(cls, args):
           return {"gain": args.get("gain", 1.0)}

       def __init__(self, gain: wp.array):
           self.gain = gain

       def compute(self, positions, velocities, target_pos, target_vel,
                   feedforward, input_indices, target_indices, forces,
                   state, dt, device=None):
           # Launch a Warp kernel that writes into `forces`
           ...

``resolve_arguments`` maps user-provided keyword arguments (from
:meth:`~newton.ModelBuilder.add_actuator` or USD schemas) to constructor
parameters, filling in defaults where needed.

Similarly, a custom clamping stage subclasses :class:`Clamping` and implements
:meth:`~Clamping.modify_forces`.

See Also
--------

* :mod:`newton.actuators` — full API reference
* :meth:`newton.ModelBuilder.add_actuator` — registering actuators during
  model construction
