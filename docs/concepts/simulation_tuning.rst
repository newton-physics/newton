.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Simulation Tuning:

Simulation Tuning
=================

Physics tuning is a process of reducing one failure mode at a time. Start with
the smallest scene that reproduces the issue, verify the model scale and mass
properties, then tune the solver and material parameters in a fixed order.

This page is intentionally operational: it is written as a checklist for humans
and agents. Parameter names are Newton parameter names unless explicitly marked
as external references.

.. toctree::
   :maxdepth: 1

   simulation_tuning_solvers

Tuning Order
------------

Use this order for most rigid-body and articulation problems:

1. **Simplify the scene.** Test the robot, mechanism, gripper, or object in
   isolation before tuning the full environment.
2. **Validate the model.** Check SI units, shape dimensions, mass, inertia,
   joint axes, joint limits, and unintended self-collisions.
3. **Choose the contact representation.** Prefer primitives for speed. Use SDF
   or hydroelastic contacts when contact patch quality, force distribution, or
   non-convex geometry matters.
4. **Set the timestep.** Reduce the simulation ``dt`` or increase substeps
   before raising stiffness. A smaller ``dt`` is usually the most reliable
   stability improvement, but it is also expensive.
5. **Tune solver convergence.** If the selected solver exposes iterations or
   tolerances, increase iterations or tighten tolerances until constraint
   residuals stop improving enough to justify the cost.
6. **Tune contacts.** Adjust stiffness, damping, friction, contact margins,
   gaps, contact count, and collision refresh cadence.
7. **Tune joints and drives.** Use realistic drive stiffness and damping. Add target rate limits in
   control code. Use supported model features such as effort limits, armature, or joint
   friction only where the selected solver supports them.
8. **Optimize performance last.** Reduce collision frequency, contact count,
   solver iterations, or substeps only after the behavior is acceptable.

Do not hide model errors with extreme solver settings. Bad mass ratios,
incorrect inertia tensors, overlapping collision geometry, and over-stiff
drives usually remain unstable even with more iterations.

Symptom Table
-------------

.. list-table::
   :header-rows: 1
   :widths: 18 32 32 18

   * - Symptom
     - Try first
     - Then try
     - Main cost
   * - Persistent penetration
     - Reduce ``dt``; increase substeps; verify contact normals, margins, and
       collision geometry.
     - Increase solver iterations if available; increase contact stiffness
       within stability limits; use SDF or hydroelastic contacts for complex
       meshes.
     - Runtime
   * - Jitter or explosive motion
     - Lower contact or drive stiffness; add damping; check for overlapping
       shapes and unintended self-collision.
     - Clamp or rate-limit commands in control code; use supported effort
       limits or armature when physically justified; reduce mass and inertia
       ratios.
     - Fidelity
   * - Weak grasp or object slip
     - Check friction coefficients, contact locations, contact count, and
       gripper force limits.
     - Use richer contact geometry; raise solver convergence work if supported;
       tune torsional and rolling friction where relevant.
     - Runtime
   * - Slow or inaccurate drive tracking
     - Tune ``joint_target_ke`` and ``joint_target_kd``; clamp control
       effort in controller code or with MuJoCo effort limits where supported;
       avoid step changes in targets.
     - Add feed-forward control; reduce ``dt``; rate-limit targets; add
       armature where supported and physically justified.
     - Runtime or response speed
   * - Stack or mechanism drifts
     - Verify mass properties and joint frames; reduce ``dt``.
     - Increase solver-specific constraint work; reduce unsupported or
       over-constrained features; use a solver that supports the needed
       constraints.
     - Runtime
   * - Simulation is too slow
     - Reduce substeps, contact refresh rate, and expensive contact models.
     - Lower iterations if available; simplify collision geometry; reduce
       contact buffers or contact count when safe.
     - Accuracy

Agent Checklist
---------------

When asked to tune a Newton scene, follow this checklist:

1. Identify the active solver class and read its public constructor or
   configuration object, plus :ref:`Joint feature support` for public model
   attributes.
2. Verify that each proposed option is supported by that solver. Do not copy
   parameter names from another Newton solver, MuJoCo CPU, MuJoCo Warp, or
   Omniverse unless Newton exposes the same option on the active solver.
3. Record ``dt``, substeps, contact refresh cadence, solver parameters, contact
   material values, and drive gains before changing anything.
4. Reproduce the symptom in a minimal scene.
5. Change one category at a time: model, timestep, solver convergence,
   contacts, drives, then performance.
6. Prefer physically meaningful changes before solver brute force.
7. Do not invent cross-solver options. For example, ``noslip_iterations`` is
   not a Newton :class:`~newton.solvers.SolverMuJoCo` constructor option.
8. Keep the final recommendation solver-specific.

Further Reading
---------------

- `Omniverse Articulation and Robot Simulation Stability Guide <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html>`__
- `Omniverse Robotiq Gripper Joint Parameter Tuning Example <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html>`__
- `MuJoCo Modeling: Solver Parameters <https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters>`__
- `MuJoCo Modeling: Solver Settings <https://mujoco.readthedocs.io/en/stable/modeling.html#solver-settings>`__
