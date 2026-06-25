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
   simulation_tuning_mujoco

Diagnose Before Tuning
----------------------

Before changing any parameter, classify the problem. Most "soft contact"
symptoms are not contact problems:

- **Initialization / geometry:** initial penetration, collision-vs-visual mesh
  mismatch, wrong joint state.
- **Control:** a bad controller or IK target, step changes in drive targets.
- **Model:** missing joint friction, armature, or damping; a missing drive
  import; bad mass or inertia.
- **Capacity:** too few contact or constraint rows for the scene.
- **Contact / solver:** only after the above are ruled out.

Three principles guide every change:

1. **Rule out non-contact issues before tuning contact.** Do not hide a model
   or control problem by raising contact stiffness.
2. **Tune physical parameters before solver options.** Solver iterations,
   line-search iterations, and tolerances affect *convergence*; they are not a
   substitute for correct geometry, mass, drives, and contact parameters.
3. **Harder is not always more stable.** A harder contact (smaller penetration,
   closer to a hard constraint) can be *less* stable: excessive stiffness, high
   plateau impedance, or too large a timestep cause jitter, energy injection, or
   poor solver conditioning. Tune to task metrics, not to a single penetration
   number.

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
   tolerances, increase them in a bounded sweep (for example, double the
   iterations) and stop when an increase reduces the constraint residual by less
   than a small margin you set in advance (a few percent); further iterations
   then cost runtime without meaningful accuracy.
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

For symptoms that classify as **Control** or **Model** under "Diagnose Before
Tuning" above, resolve that category before contacts: steps 6–7 list contacts
before joints and drives only because that order suits contact-dominated
symptoms. For drive- or controller-dominated symptoms (such as poor tracking or
oscillation), tune joints and drives first and consult the Symptom Table.

Accept a parameter change only when the target task metric improves and no
guardrail regresses past a bound set in advance — NaN/Inf count, maximum
penetration or constraint residual, and runtime. Record those baselines before
the first change and change one knob at a time.

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

Going Deeper
------------

- :ref:`Tuning Solver Reference` — supported knobs per solver and sanity-check
  math.
- :ref:`Tuning MuJoCo` — the MuJoCo-Warp constraint model, ``ke``/``kd`` to
  ``solref``/``solimp`` mapping, and task templates.

Agent Checklist
---------------

Automated agents should follow the diagnostic order and per-solver verification
checklist in the ``simulation-tuning`` skill
(``.claude/skills/simulation-tuning/SKILL.md``), which routes to the relevant
section of this guide for parameter detail.

Further Reading
---------------

- `Omniverse Articulation and Robot Simulation Stability Guide <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html>`__
- `Omniverse Robotiq Gripper Joint Parameter Tuning Example <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html>`__
- `MuJoCo Modeling: Solver Parameters <https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters>`__
- `MuJoCo Modeling: Solver Settings <https://mujoco.readthedocs.io/en/stable/modeling.html#solver-settings>`__
