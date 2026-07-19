.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Kamino
======

:class:`~newton.solvers.SolverKamino` simulates constrained rigid multi-body
systems in maximal coordinates. It is designed for mechanical assemblies with
kinematic loops, under- or overactuation, joint limits, hard frictional
contacts, and restitutive impacts.

Unlike the other maximal-coordinate solvers, Kamino focuses on constrained
rigid mechanical assemblies rather than particle or deformable simulation.
Kamino is currently in BETA 1, and Newton users are discouraged from depending
on it. Evaluate it only when kinematic loops and hard contact constraints are
primary requirements and an experimental solver is acceptable.

.. experimental::

   :class:`~newton.solvers.SolverKamino` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

See the :class:`~newton.solvers.SolverKamino` API reference for construction
and configuration details. Runnable workflows are available in the
`Kamino examples <https://github.com/newton-physics/newton/tree/main/newton/examples/kamino>`_.

External Actuator Jacobians
---------------------------

Kamino has an opt-in path for using supported actuator force
Jacobians in its implicit joint dynamics.  This is useful for stiff PD-like
external actuators: the actuator remains outside the solver, but Kamino can use
the actuator's local force derivatives to form a linearly implicit update.

There are three relevant actuation paths:

**Built-in Kamino implicit PD**
   Configure joint targets and built-in model gains
   (:attr:`newton.Model.joint_target_ke` /
   :attr:`newton.Model.joint_target_kd`) and do not call an external actuator.
   Kamino uses its existing implicit PD drive path.

**External explicit actuator PD**
   Call an external actuator normally.  The actuator writes only
   :attr:`newton.Control.joint_f`, and Kamino treats that effort as an explicit
   generalized joint force.  Set the corresponding built-in
   :attr:`newton.Model.joint_target_ke` and
   :attr:`newton.Model.joint_target_kd` values to zero to avoid double counting
   and make this path truly explicit rather than external PD plus built-in
   Kamino PD.

**External PD with actuator Jacobians**
   Call an external actuator with ``write_force_jacobians=True`` and enable
   :attr:`~newton.solvers.SolverKamino.Config.use_actuator_jacobians` in the
   Kamino config.  The actuator writes :attr:`newton.Control.joint_f` plus
   :attr:`newton.Control.joint_f_dq` /
   :attr:`newton.Control.joint_f_dqd`; Kamino uses those derivatives in its
   implicit joint dynamics.

Example setup:

.. code-block:: python

   config = newton.solvers.SolverKamino.Config.from_model(model)
   config.use_actuator_jacobians = True
   solver = newton.solvers.SolverKamino(model, config=config)

   control.clear(model)
   for actuator in model.actuators:
       actuator.step(
           state_0,
           control,
           dt=sim_dt,
           write_force_jacobians=config.use_actuator_jacobians,
       )

   solver.step(state_0, state_1, control, contacts, sim_dt)

Force Reconstruction
^^^^^^^^^^^^^^^^^^^^

External actuators still write the full current-state actuator effort
``tau_act`` into :attr:`newton.Control.joint_f`.  For DOFs with valid actuator
Jacobians, Kamino avoids using that same force twice.  Kamino subtracts the
PD-like part that it will reconstruct implicitly, leaving only the remaining
bias term:

.. math::

   \tau_\mathrm{ref}
   =
   \tau_\mathrm{act}
   -
   k_{p,\mathrm{jacob}}(q_\mathrm{target} - q)
   -
   k_{d,\mathrm{jacob}}(\dot{q}_\mathrm{target} - \dot{q})

Kamino then uses effective gains in its implicit drive expression:

.. math::

   \tau_\mathrm{drive}
   =
   \tau_\mathrm{ref}
   +
   k_{p,\mathrm{effective}}(q_\mathrm{target} - q)
   +
   k_{d,\mathrm{effective}}\dot{q}_\mathrm{target}.

The velocity-proportional part involving ``-kd * qdot`` enters implicitly
through Kamino's effective mass term:

.. math::

   m_\mathrm{eff}
   =
   a
   +
   h(b + k_{d,\mathrm{effective}})
   +
   h^2k_{p,\mathrm{effective}}.

For a pure PD actuator with no constant or feedforward term,
``tau_ref`` is zero.  For other supported controllers, or PD with feedforward,
``tau_ref`` is the residual term that makes Kamino's reconstructed force match
the actuator force at the current state.

Gain Sources
^^^^^^^^^^^^

The gain names describe where a value comes from and what Kamino actually uses
for a step:

``k_controller``
   Gain owned by one external controller, such as one
   :class:`newton.actuators.ControllerPD`.

``k_jacob``
   Gain inferred from
   :attr:`newton.Control.joint_f_dq` /
   :attr:`newton.Control.joint_f_dqd`.  If multiple actuators contribute to the
   same DOF, their forces and derivatives add before Kamino consumes them.

``k_model``
   Built-in model drive gain from :attr:`newton.Model.joint_target_ke` /
   :attr:`newton.Model.joint_target_kd`.

``k_effective``
   Gain Kamino uses for the current step after the Jacobian/fallback decision.

Fallback Behavior
^^^^^^^^^^^^^^^^^

If actuator Jacobians are not requested, Kamino behaves as before.  If
Jacobians are requested but the actuator cannot provide them, the actuator
warns once, leaves the Jacobian entries as ``NaN``, and Kamino falls back to
the explicit :attr:`newton.Control.joint_f` path for those DOFs.

As part of this same decision, Kamino sets ``k_effective`` for each DOF.  Valid
actuator Jacobians select ``k_jacob``.  Missing, unsupported, or unrequested
Jacobians select the original built-in drive gain ``k_model`` and use any
:attr:`newton.Control.joint_f` effort explicitly.

Kamino also warns if Jacobians were written but ``use_actuator_jacobians`` was
not enabled in solver config, which helps catch the common case where the actuator side was
configured but the solver side was not.

Current Limitations
^^^^^^^^^^^^^^^^^^^

The current Kamino path is intentionally narrow:

* It is opt-in through
  :attr:`~newton.solvers.SolverKamino.Config.use_actuator_jacobians`.
* Analytic Jacobians are currently provided by unclamped
  :class:`newton.actuators.ControllerPD` actuators.
* Jacobians are diagonal per joint DOF; coupled multi-DOF actuator Jacobians
  are not consumed yet.
* Kamino maps scalar joint DOFs through the model's joint start arrays instead
  of assuming :attr:`newton.State.joint_q` and :attr:`newton.State.joint_qd`
  share the same indexing.  This supports the scalar actuated joints commonly
  used on floating-base robots, but ball/free-base actuator Jacobians are not
  handled by this first path.
