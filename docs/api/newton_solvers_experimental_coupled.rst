.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers.experimental.coupled
===================================

Experimental coupled-solver framework.

.. experimental::

.. py:module:: newton.solvers.experimental.coupled
.. currentmodule:: newton.solvers.experimental.coupled

.. rubric:: Classes

.. autoclass:: CouplingInterface

.. autoclass:: ModelView

.. autoclass:: SolverCoupled

.. autoclass:: SolverCoupledADMM

.. autoclass:: SolverCoupledProxy


.. _coupled-admm-early-stopping:

ADMM early stopping
-------------------

Set ``Config.convergence`` to enable velocity and force checks:

.. code-block:: python

    coupling = SolverCoupledADMM.Config(
        iterations=20,
        convergence=SolverCoupledADMM.ConvergenceConfig(
            linear_velocity_tolerance=1.0e-5,   # m/s
            angular_velocity_tolerance=1.0e-5,  # rad/s
            force_tolerance=1.0e-3,             # N
            torque_tolerance=1.0e-3,            # N*m
            force_relative_tolerance=1.0e-3,
            check_interval=3,
        ),
    )

Every populated interface row must pass both criteria; angular rows use
angular velocity and torque tolerances. See :class:`SolverCoupledADMM.ConvergenceConfig`
for the residual definitions and defaults. The first check follows ``min_iterations``
complete iterations, then checks occur every ``check_interval`` and at the
``iterations`` cap. Increasing the interval reduces check overhead but can delay
stopping. The default ``convergence=None`` uses the fixed iteration count.

With positive ``gamma``, interface agreement alone is insufficient: participants
can move together while their relative velocity and coupling-force change are
zero. The proximal shift still depends on each participant's previous iterate.
Early stopping therefore also checks every velocity affected by that shift,
including contributions retained for candidate contacts with no populated rows.

Writing the velocity shift as

.. math::

    s^k = \frac{v_n + \alpha v^k}{1 + \alpha},

the additional criterion is

.. math::

    \max(1, \alpha)\,\lVert v^{k+1} - v^k\rVert \leq \epsilon_v.

This requires both the iterate increment and its proximal contribution
``alpha * (v_next - v_previous)`` to be small. Checking only the increment
could accept slow progress caused by a large proximal factor. The latter
quantity equals ``(1 + alpha) * (s_next - s_previous)`` and has velocity units;
for a particle it is the change in proximal momentum divided by the base mass.

For translation, ``alpha`` is the added proximal mass divided by the base mass.
For body rotation it is the analogous ratio using the scalar inertia in the
velocity shift. Generalized velocities use their dimensionless proximal
factors. Translation and rotation use the linear and angular velocity
tolerances, respectively; generalized velocities
are checked per DOF with the corresponding units. Zero-factor components are
not checked. These checks use the iterates before and after the complete
participant solve, including all its substeps, at the same check boundaries
as the interface criteria. They add no host readbacks or velocity snapshots.

The device arrays :attr:`SolverCoupledADMM.iteration_count` and
:attr:`SolverCoupledADMM.converged` report the last check. Convergence is global
across all worlds. Reaching the cap returns the last iterate; the criteria do
not bound integration error or guarantee nonlinear stability.

.. note::

    CUDA graph capture keeps stopping decisions on the device using conditional
    nodes (CUDA 12.4+). Participants must support conditional graph bodies, which
    prohibit allocations and host/device copies. XPBD scratch allocations and
    participant particle hash-grid rebuilds currently prevent this mode.
    Outside graph capture, CPU and CUDA use the same Warp checks with one scalar
    readback between checked blocks; this synchronizes CUDA execution.
