.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Actuators
=========

Actuators compute joint forces from simulation state and control targets.
They model real-world motor controllers — from simple PD loops to neural
networks — and are integrated directly into Newton's simulation pipeline.

Overview
--------

An actuator reads joint positions and velocities from the simulation
:class:`~newton.State`, reads target values from the :class:`~newton.Control`
object, and writes computed forces back into the control's ``joint_f`` array.

All actuators derive from :class:`~newton.actuators.Actuator` and share the
same lifecycle:

1. **Registration** -- call :meth:`~newton.ModelBuilder.add_actuator` once per
   controlled DOF while building the model.
2. **Instantiation** -- :meth:`~newton.ModelBuilder.finalize` groups compatible
   actuators and creates the instances stored in :attr:`~newton.Model.actuators`.
3. **Execution** -- call :meth:`~newton.actuators.Actuator.step` each timestep
   to apply forces.

Available Actuators
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Class
     - Stateful
     - Description
   * - :class:`~newton.actuators.ActuatorPD`
     - No
     - Proportional-derivative controller.
   * - :class:`~newton.actuators.ActuatorPID`
     - Yes
     - PID controller with integral anti-windup.
   * - :class:`~newton.actuators.ActuatorDCMotor`
     - No
     - PD with velocity-dependent torque saturation (DC motor model).
   * - :class:`~newton.actuators.ActuatorDelayedPD`
     - Yes
     - PD with configurable input delay via circular buffer.
   * - :class:`~newton.actuators.ActuatorRemotizedPD`
     - Yes
     - Delayed PD with angle-dependent torque limits from a lookup table.
   * - :class:`~newton.actuators.ActuatorNetMLP`
     - Yes
     - MLP neural-network controller (requires PyTorch).
   * - :class:`~newton.actuators.ActuatorNetLSTM`
     - Yes
     - LSTM neural-network controller (requires PyTorch).

Adding Actuators to a Model
---------------------------

Use :meth:`~newton.ModelBuilder.add_actuator` to attach an actuator to one or
more degrees of freedom::

    import newton
    from newton.actuators import ActuatorPD

    builder = newton.ModelBuilder()

    body = builder.add_body()
    joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
    builder.add_articulation([joint])

    dof = builder.joint_qd_start[joint]
    builder.add_actuator(ActuatorPD, input_indices=[dof], kp=100.0, kd=10.0)

    model = builder.finalize()

Multiple calls with the same actuator class and matching scalar parameters
(e.g. ``delay``) are merged into a single batched instance.  Different scalar
values create separate instances automatically.

Stepping Actuators
------------------

After finalizing the model, iterate over ``model.actuators`` each timestep::

    state = model.state()
    control = model.control()

    # Set targets on control
    # ...

    for actuator in model.actuators:
        actuator.step(state, control)

Stateful Actuators
^^^^^^^^^^^^^^^^^^

Stateful actuators (PID, delayed, neural-network) require double-buffered
state objects::

    from newton.actuators import ActuatorPID

    indices = wp.array([0, 1], dtype=wp.uint32)
    pid_actuator = ActuatorPID(
        input_indices=indices,
        output_indices=indices,
        kp=wp.array([100.0, 100.0], dtype=wp.float32),
        ki=wp.array([10.0, 10.0], dtype=wp.float32),
        kd=wp.array([5.0, 5.0], dtype=wp.float32),
        max_force=wp.array([50.0, 50.0], dtype=wp.float32),
        integral_max=wp.array([10.0, 10.0], dtype=wp.float32),
        constant_force=wp.array([0.0, 0.0], dtype=wp.float32),
    )

    # Check if actuator needs state management
    if pid_actuator.is_stateful():
        # Create double-buffered states
        state_a = pid_actuator.state()
        state_b = pid_actuator.state()

    # Simulation loop with state swapping
    current_state, next_state = state_a, state_b
    for step in range(num_steps):
        pid_actuator.step(sim_state, sim_control, current_state, next_state, dt=0.01)
        current_state, next_state = next_state, current_state  # Swap buffers

Scalar vs Per-Actuator Parameters
----------------------------------

Each actuator class declares which parameters are **scalar** (shared across
all DOFs in one instance) via the ``SCALAR_PARAMS`` class attribute.  For
example, :class:`~newton.actuators.ActuatorDelayedPD` declares
``SCALAR_PARAMS = {"delay"}``.

- **Per-actuator parameters** (``kp``, ``kd``, ``max_force``, …) can differ
  across DOFs and are stored as Warp arrays.
- **Scalar parameters** (``delay``, ``network_path``, …) must be identical
  for all DOFs in one instance.  Different values cause
  :meth:`~newton.ModelBuilder.add_actuator` to create separate groups.

USD Parsing
-----------

Actuator prims in USD stages are automatically parsed during
:func:`newton.usd.parse_usd`.  The parser inspects
``newton:actuator:*`` attributes to infer the actuator class and parameters.

See the :doc:`usd_parsing` guide for details on the USD schema.

API Reference
-------------

See :doc:`../api/newton_actuators` for the full API.
