.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers.mujoco
=====================

MuJoCo solver and programmatic model-authoring helpers.

Use :class:`~newton.solvers.SolverMuJoCo` as the solver class. The module-level
helpers create MuJoCo-specific actuators, tendons, contact pairs, and equality
constraints on a :class:`~newton.ModelBuilder` without exposing custom-frequency
storage details.

Example::

    from newton.solvers import mujoco

    actuator = mujoco.add_actuator_dcmotor(
        builder,
        target=mujoco.ActuatorTarget.joint(joint),
        motorconst=(0.05, 0.05),
        resistance=2.0,
    )

.. note::

   This page documents helper functions exposed through the ``newton.solvers.mujoco`` attribute.
   Because ``newton.solvers`` is a module rather than a package, use
   ``from newton.solvers import mujoco`` instead of ``import newton.solvers.mujoco``.

.. currentmodule:: newton._src.solvers.mujoco

.. rubric:: Classes

.. autoclass:: ActuatorTarget

.. autoclass:: TendonWrapGeom

.. autoclass:: TendonWrapPulley

.. autoclass:: TendonWrapSite


.. rubric:: Functions

.. autofunction:: add_actuator_dcmotor

.. autofunction:: add_actuator_general

.. autofunction:: add_actuator_motor

.. autofunction:: add_actuator_position

.. autofunction:: add_actuator_velocity

.. autofunction:: add_contact_pair

.. autofunction:: add_equality_connect

.. autofunction:: add_equality_joint

.. autofunction:: add_equality_weld

.. autofunction:: add_tendon_fixed

.. autofunction:: add_tendon_spatial
