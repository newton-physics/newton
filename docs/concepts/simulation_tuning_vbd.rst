.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Tuning VBD:

VBD Tuning
==========

This page will explain how :class:`~newton.solvers.SolverVBD` reacts to its
solver and model parameters, what typically goes wrong and how to fix it, and
which parameter combinations are known to work; it is currently an
in-development outline rather than complete tuning guidance. See
:ref:`Simulation Tuning` for the diagnostic workflow, :ref:`Tuning Solver
Reference` for the full knob list, and :doc:`VBD </solvers/vbd>` for setup and
API explanations.

.. important::

   The specific values, mode names, and formulas on this page reflect the code
   at a point in time and can drift. Treat them as starting points and verify
   any you rely on against the cited source (for example
   :class:`~newton.solvers.SolverVBD` and its kernels). See
   :ref:`Simulation Tuning` for the full guidance.

.. note::

   This page is under active development. Each section below states its
   intended scope and links to existing references until the full content
   lands.

Mental Model
------------

*In development.* This section will explain, briefly and without paper-level
math, how the solver converges and why it fails:

- how VBD updates one block at a time — a cloth or soft-body vertex, or a
  rigid body — from its local system while its neighbors stay fixed within
  the update, and why that makes heavily deformed or stiffly coupled
  configurations the hard case;
- how joint and contact constraints are enforced inside the same VBD loop
  through an augmented-Lagrangian (ALM) formulation
  (compliant ALM by default; the legacy path is deprecated),
  and what roles ``rigid_avbd_alpha``, ``rigid_avbd_beta``, and
  ``rigid_avbd_gamma`` play;
- how the penetration-free Divide and Truncate (DAT) scheme bounds and
  truncates per-vertex motion against contacts, and what that implies for
  self-contact parameters.

For the underlying method, see the VBD paper cited in
:class:`~newton.solvers.SolverVBD`.

.. _vbd-rigid-cable-tuning:

Rigid Bodies and Cables
~~~~~~~~~~~~~~~~~~~~~~~

This section covers bodies integrated by :class:`~newton.solvers.SolverVBD`,
including capsule cables connected by rod joints.

For specific guidance, see :ref:`joint gains <vbd-rigid-joint-gains>`,
:ref:`cable materials <vbd-cable-inputs>`,
:ref:`rigid contacts <vbd-rigid-contacts>`,
:ref:`contact history <vbd-rigid-contact-history>`, and
:ref:`model changes and reset <vbd-rigid-model-updates>`.

See :class:`~newton.solvers.SolverVBD` and
:doc:`VBD </solvers/vbd>` for setup, stepping, scene-authored defaults, and
the experimental status of the solver. Check :ref:`Joint feature support` and
:ref:`Contact material support` before tuning an imported model; use
:ref:`Simulation Tuning` for the general timestep, iteration, and diagnostic
workflow.

VBD advances body poses and velocities in ``state.body_q`` and
``state.body_qd`` directly. :meth:`~newton.solvers.SolverVBD.step` does not
call :func:`eval_fk` or :func:`eval_ik`, and does not automatically update
``state.joint_q`` or ``state.joint_qd``. These helpers are supported for
articulations: call ``eval_fk`` explicitly to convert joint coordinates and
velocities to body state, or ``eval_ik`` to recover them from body state.

.. _vbd-rigid-compliance:

Compliant ALM uses finite stiffness, so some stretch, joint displacement, or
contact penetration can remain even after convergence. If deformation is
larger than expected, check loads, material values, rest geometry, attachments,
and contact setup. Also check timestep and iteration sensitivity rather than
relying only on higher stiffness; see :ref:`Simulation Tuning`.

Parameter Effects
-----------------

VBD-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~

*In development.* Effects and interactions of the constructor parameters, by
group: common (``iterations``, ``friction_epsilon``), particle (the
self-contact family, the self-contact slot of ``collision_frequency`` /
``collision_frequency_type``, ``dat_conservative_bound_relaxation``,
``particle_enable_tile_solve``, contact buffer sizes), and rigid constraints
(``rigid_compliant_alm`` mode selection, ``rigid_avbd_alpha``/``gamma``,
the deprecated ``rigid_avbd_beta`` ramping, penalty seeds and ceilings,
``rigid_contact_hard``, ``rigid_contact_history``).

Deprecated penalty-ramping controls do not affect compliant-ALM rigid constraints.

Until then, the authoritative parameter list with defaults is
:class:`~newton.solvers.SolverVBD`; the supported-knob summary lives in
:ref:`Tuning Solver Reference`. The legacy aliases
``particle_collision_detection_interval`` and
``particle_conservative_bound_relaxation`` remain supported but are
deprecated; new configurations should use ``collision_frequency`` /
``collision_frequency_type`` and ``dat_conservative_bound_relaxation``.

.. _vbd-rigid-joint-gains:

Rigid Constraints
`````````````````

Use ``rigid_compliant_alm=True`` and initially keep ``rigid_avbd_alpha`` and
``rigid_avbd_gamma`` at their defaults. Revalidate scenes when migrating from
legacy AVBD; the same settings can produce different behavior.

Non-rod structural stiffness and damping are set by ``rigid_joint_linear_ke``
/ ``rigid_joint_angular_ke`` and ``rigid_joint_linear_kd`` /
``rigid_joint_angular_kd``. Omitted arguments use the corresponding settings
stored in ``model.vbd`` when present; otherwise, stiffness defaults to
``1.0e5`` N/m and N·m/rad, and damping to ``0.0`` N·s/m and N·m·s/rad.
Changing drive gains does not change that structural response.

Tune these gains on an isolated joint under load. Choose stiffness for
acceptable displacement or rotation in constrained directions, then tune
damping for those directions from constraint-error decay after release.
Very large stiffness can slow convergence; excessive damping can slow
settling. Check timestep and iteration sensitivity when errors exceed
expectations; see :ref:`Simulation Tuning`. Rebuild the solver after changing
these construction-only settings.

Model Parameters
~~~~~~~~~~~~~~~~

*In development.* How the solver interprets material and scene parameters:
cloth membrane and bending stiffness and damping (``tri_ke``, ``tri_ka``,
``tri_kd``, ``edge_ke``, ``edge_kd``), soft-body volumetric material, contact
materials (``soft_contact_*`` and shape ``mu``/``ke``/``kd``), joint and rod
stiffness and damping, particle mass and radius, and the
``dt``–substeps–``iterations`` trade-off.

.. _vbd-cable-inputs:

Cable Material, Resolution, and Rest Shape
``````````````````````````````````````````

Create a :class:`Rod` and pass it to :meth:`ModelBuilder.add_rod`; each
segment becomes a capsule body connected to its neighbors by rod joints (see
:ref:`Rod joints`).

For resolution studies, material inputs or section rigidities let the builder
recompute joint stiffness from segment length while keeping the intended
material fixed. Specify either elastic material inputs (``youngs_modulus``
[Pa], ``radius`` [m], and exactly one of dimensionless ``poissons_ratio`` or
``shear_modulus`` [Pa]), or all four section rigidities: ``stretch_rigidity``
(:math:`EA`) and ``shear_rigidity`` (:math:`kGA`) [N], and ``bend_rigidity``
(:math:`EI`) and ``twist_rigidity`` (:math:`GJ`) [N·m²]. The two descriptions
are mutually exclusive.

The builder maps each rigidity to joint stiffness as :math:`R / L_{\mathrm{dual}}`,
where :math:`L_{\mathrm{dual}}` is the mean rest length of the two adjacent
segments. Explicit ``*_stiffness`` arguments to ``add_rod`` override that
mapping per mode: stretch/shear use [N/m], bend/twist use [N·m/rad]. Without
constitutive inputs on :class:`Rod` or explicit overrides, ``add_rod``
defaults stretch and shear stiffness to ``1.0e5`` N/m and bend and twist
stiffness to ``0.0`` N·m/rad. Direct stiffness and damping are applied
unchanged to every joint. Damping uses [N·s/m] or [N·m·s/rad]; it is not a
damping ratio and is not automatically length-normalized. See
:meth:`ModelBuilder.add_rod` for topology restrictions.

If using direct stiffness inputs for a resolution study, rescale them with
segment length to preserve the intended material. Refine segment count
separately from timestep and iterations: the new joint graph, masses,
inertias, and contact workload require another convergence check. Recheck the
damping transient at the final resolution too.

Initial geometry (``points``, ``quaternions``) and rest geometry
(``rest_points``, ``rest_quaternions``) are distinct; rest defaults to the
initial shape. For an ordered chain, ``rest_straight=True`` sets a straight,
untwisted rest shape with the initial segment lengths. Do not supply explicit
rest inputs with it. With nonzero bending stiffness, the cable then tends
toward straight. The default ``body_frame_origin="com"`` places the body origin
at the segment midpoint; use the corresponding local offset for an endpoint
attachment.

Before tuning stiffness, check the initial and rest orientations and how the
cable ends are attached. Fixing only an endpoint's position allows it to
rotate; fixing its orientation as well makes the attachment clamped.

Fit stretch, bend, and twist stiffness separately under static loads, then
tune damping from oscillation decay after release. Lower stretch stiffness
changes the material; use it only if the extra extension is acceptable.
Excessive damping can make motion sluggish.

Contact Handling
~~~~~~~~~~~~~~~~

*In development.* How the contact stack responds to tuning: the self-contact
detection family (radius, margin, detection interval, buffer sizes, and the
topological and rest-shape filters), the penetration-free Divide and Truncate
(DAT) truncation and ``particle_conservative_bound_relaxation``, and the
interplay of contact stiffness, damping, and friction. Coverage of Offset
Geometric Contact (OGC) will be added once it is available in Newton.

.. _vbd-rigid-contacts:

Rigid Contacts
``````````````

**Detection first.** Shape geometry, such as a capsule radius, describes the
body; margin shifts the contact surface; gap extends the detection distance.
See :ref:`margin and gap semantics <margin-gap-semantics>`. The builder's
``rigid_gap`` defaults to 0.1 m for shapes without an explicit gap, which can
be large beside a centimeter-scale cable. Set ``builder.default_shape_cfg.gap``
or a shape's ``ShapeConfig.gap`` before adding shapes, then check fast motion
between collision passes. Changing the builder default later does not update
existing shapes. Choose a gap that detects the required contacts reliably.
If those contacts exceed the available capacity, increase the relevant
:ref:`contact buffer <vbd-contact-buffers>`.

**Collision filtering.** Directly jointed body pairs are filtered by default
(``collision_filter_parent``), but other nearby cable segments can still
overlap, especially with short segments or thick capsules. Unfiltered overlaps
can produce unwanted contact forces from the first step. Inspect the initial
geometry and exclude intentionally overlapping neighboring segments whose
collision response is unnecessary throughout the expected motion, using
:meth:`ModelBuilder.add_shape_collision_filter_pair` before ``finalize()``.
Correct unintended initial overlaps, and preserve self-contact needed for
folding and tight bends.

Filtering additional neighbors along the cable can also reduce contact
processing and buffer demand.

**Normal support before friction.** Check penetration and normal-load support
before increasing ``mu``. VBD mixes body–body ``ke`` and ``kd`` with an
arithmetic mean and ``mu`` with a geometric mean, so one stiff shape can
dominate normal stiffness while a shape with ``mu=0`` makes its pairs
frictionless. Unexpected slip can also come from missing contacts or an
incomplete solve. Leave ``friction_epsilon`` at its default initially. It
controls how friction computations handle very small sliding speeds and is
specified in m/s. For compliant-ALM body–body contacts, it can affect
convergence at a finite iteration count; ``mu`` still sets the Coulomb
friction limit. For body–particle contacts, it smooths friction forces near
zero slip.

**Body coloring.** VBD updates bodies in the same color group in parallel.
Automatic coloring uses joint and tendon connections, but does not include
contacts. Contacting bodies may share a color, which can slow or prevent
contact convergence. For small scenes with predictable contacts, consider splitting
existing groups in ``builder.body_color_groups`` to separate frequently
contacting bodies. Make these edits after ``builder.color()`` and before
``builder.finalize()``; calling ``color()`` again replaces them.

Split existing groups without dropping or duplicating bodies; this preserves
the original joint/tendon separation. Custom groups are not fully validated.
More colors reduce parallelism, so compare convergence and runtime.

.. _vbd-rigid-contact-history:

Contact History
'''''''''''''''

Set ``contact_matching`` on :class:`CollisionPipeline` and
``rigid_contact_history`` on :class:`~newton.solvers.SolverVBD`.
Enabling history does not enable matching automatically.

Matching associates contacts across collision passes. VBD uses it for
between-step warm starts when history is enabled, and to preserve ALM state
during ``ITERATIONS`` re-detection by a solver-owned pipeline, even with
``rigid_contact_history=False``.

With history enabled, passing a pipeline with matching disabled to
``SolverVBD`` raises ``ValueError`` at construction. With externally supplied
contacts, VBD raises ``RuntimeError`` during rigid-contact refresh if the
buffer lacks match-index data or reports neither ``"latest"`` nor ``"sticky"``
as its matching mode.

With compliant ALM and ``rigid_contact_history=True``, the matching modes
behave as follows:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``contact_matching``
     - Behavior
   * - ``"disabled"``
     - Incompatible with history. Use ``rigid_contact_history=False``.
   * - ``"latest"``
     - Keep fresh geometry. Restore normal and projected tangential
       multipliers for valid matches.
   * - ``"sticky"``
     - Replay eligible contact geometry. Restore normal multipliers;
       tangential memory comes from the replayed material anchor.
       This matching mode is experimental.

For a baseline, set ``rigid_contact_history=False`` explicitly. Use
``contact_matching="disabled"`` unless rigid detection uses ``ITERATIONS``,
which requires ``"latest"`` or ``"sticky"`` matching.

Contact history can improve convergence and reduce jitter or unintended slip
for persistent contacts; it does not change the authored friction coefficient.
Compare latest matching plus history against the baseline, measuring drift,
penetration, and total collision plus solver time. Treat sticky matching as
a separate choice: test sliding, separation, and changing contact normals as
well as rest. Increasing ``contact_matching_pos_threshold`` allows matching
across larger position changes; decreasing
``contact_matching_normal_dot_threshold`` allows matching across larger
changes in contact normal. See :ref:`Contact Matching`. Neither mode
guarantees improvement or replaces convergence work.

Contact matching and solver history add computation and memory costs.
Contact matching alone does not guarantee a deterministic GPU rigid-contact
solve.

Warm starts require correspondence to the contact set actually solved in the
previous step. New or unmatched contacts start cold; this is normal.
Alternating contact buffers or colliding twice between steps can cold-start
contacts even with history enabled; see
:class:`~newton.solvers.SolverVBD` for the contact-generation contract.

When a substep reuses contacts without rerunning collision detection, call
:meth:`~newton.solvers.SolverVBD.set_rigid_history_update` with ``False`` to
skip rebuilding contact lists and reinitializing contact records. Reuse the
last refreshed contact buffers unchanged, with body mass and kinematic status
unchanged. Per-step contact/joint updates, solving, and enabled history
snapshots still run. Keep refresh enabled after detection or reset. The flag
resets to ``True`` after each step and is independent of
``rigid_contact_history``.

.. _vbd-contact-buffers:

Contact Capacity
''''''''''''''''

Global storage and per-body adjacency have independent capacities. Increasing
one does not increase the other. :class:`CollisionPipeline` exposes:

- ``rigid_contact_max``: global rigid contact storage.
- ``max_triangle_pairs``: intermediate triangle pairs for mesh/heightfield
  collision and global contact reduction.
- ``contact_reduction_hashtable_size_factor``: global contact-reduction
  hash-table capacity relative to ``max_triangle_pairs``. Increase it for
  hash-table fill/failure warnings.

VBD's ``rigid_body_contact_buffer_size`` (default 64) controls body–body
contact entries per rigid body. Overflow drops contributions from that body's
solve. Keep ``CollisionPipeline(verify_buffers=True)`` diagnostics enabled
while sizing collision buffers and check VBD's per-body overflow warnings
separately. Measure peak demand during impacts, dense packing, and resets,
then allocate headroom; a count at capacity may already be clipped. Check
filters and an unnecessarily large detection gap before allocating more memory.

For body–particle contacts, ``rigid_body_particle_contact_buffer_size``
(default 256) is a separate per-body budget. Its overflow falls back to scanning
all stored soft contacts, preserving reactions at extra cost. It cannot recover
contacts lost from the global ``CollisionPipeline.soft_contact_max`` storage.

Best Practices
--------------

What Can Go Wrong and How to Fix It
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*In development.* One entry per failure mode — symptom, cause, fix — topped
by a quick-lookup symptom table. Planned entries include cloth oscillation
that more iterations do not fix, outlier-vertex explosions, self-contact
tunneling, contact chattering and penetration, grasp slip versus finger
penetration, and joint drift.

Until then, use the generic Symptom Table in :ref:`Simulation Tuning`.

Rigid-Body and Cable Problems
`````````````````````````````

**Unexpected motion at startup.** Check initial overlaps, cable rest geometry,
and joint attachment frames. Adjust the initial geometry to remove unintended
overlaps. Exclude intentionally overlapping neighboring segments only if their
collision response is unnecessary throughout the expected motion. See
:ref:`cable inputs <vbd-cable-inputs>` and
:ref:`rigid contact setup <vbd-rigid-contacts>`.

**Excessive stretch or joint displacement.** Check loads, stiffness inputs,
rest geometry, and attachments. Also check timestep and iteration sensitivity
with those inputs held fixed. Finite stiffness permits some deformation;
see :ref:`rigid compliance <vbd-rigid-compliance>`,
:ref:`cable materials <vbd-cable-inputs>`, and
:ref:`non-rod joint gains <vbd-rigid-joint-gains>`.

**Excessive penetration, unexpected slip, or jitter.** Check contact detection,
including fast motion between collision passes, buffer overflow, and
normal-load support. For slip, also check both shapes' friction and the
joint/contact solve's convergence before increasing ``mu``. For persistent
contacts, compare :ref:`contact history <vbd-rigid-contact-history>` against
a baseline. See :ref:`rigid contact setup <vbd-rigid-contacts>` and
:ref:`contact capacity <vbd-contact-buffers>`.

**Unexpected behavior after edits or reset.** Check whether a model edit
requires :meth:`~newton.solvers.SolverVBD.notify_model_changed` or rebuilding
the solver. For resets, clear solver and collision-pipeline history and
regenerate contacts at the final pose. See
:ref:`model changes and reset <vbd-rigid-model-updates>`.

.. _vbd-rigid-model-updates:

Model Changes and Reset
~~~~~~~~~~~~~~~~~~~~~~~

Rod stiffness and damping can be updated during simulation. VBD caches the
rod gains and requires equal X/Y shear and X/Y bend coefficients on each rod
joint. After editing model properties, call
:meth:`~newton.solvers.SolverVBD.notify_model_changed` with the matching flag
before the next step:

- :attr:`ModelFlags.JOINT_DOF_PROPERTIES` for rod gains in
  ``model.joint_target_ke`` or ``model.joint_target_kd``.
- :attr:`ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES` for rod angular rest
  targets in ``model.joint_target_q``.
- :attr:`ModelFlags.JOINT_PROPERTIES` after toggling ``model.joint_enabled``.
- :attr:`ModelFlags.BODY_PROPERTIES` after changing kinematic flags.
- :attr:`ModelFlags.BODY_INERTIAL_PROPERTIES` after changing body mass or inertia.

When editing mass or inertia directly, keep ``model.body_inv_mass`` and
``model.body_inv_inertia`` consistent with them. The notification refreshes
solver caches; it does not recompute these model arrays.

Create a new ``SolverVBD`` after changing construction-only settings,
joint topology or DOF layout, or the non-rod reference configuration used
to initialize drive/limit angle offsets.

For an episode reset, call :meth:`~newton.solvers.SolverVBD.reset` to clear
rigid solver history and optionally restore selected state fields from the
model defaults. For a custom reset, assign ``state.body_q`` and
``state.body_qd`` and use ``flags=0`` to preserve them. Set velocities to zero
for a reset to rest. If authoring the reset in joint coordinates, run
:func:`eval_fk` after reset to populate the body poses and velocities. Finish
all state edits before the next step.

Also call :meth:`CollisionPipeline.reset_contact_matching` with the same
world mask to discard correspondence and sticky geometry. Regenerate contacts
at the final pose and leave rigid-history refresh enabled for the next
contact-bearing step. See the reset API for partial-world and coupled-solver
behavior.

Empirical Parameter Combinations That Work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*In development.* Known-good value tables per scenario family — cloth, soft
body, rods (cables), rigid contacts, mixed scenes — each linking to a
representative
runnable example from the `examples directory
<https://github.com/newton-physics/newton/tree/main/newton/examples>`_.

Cable Examples
``````````````

These examples demonstrate cable setup and behavior:

- :github:`Initial and rest shapes <newton/examples/cable/example_cable_initial_rest_pose.py>`.
- :github:`Twist propagation with different angular stiffnesses <newton/examples/cable/example_cable_twist.py>`.
- :github:`Cable wrapping and contact in a driven mechanism <newton/examples/cable/example_cable_cross_slide_table.py>`.
- :github:`Contact and settling in a dense cable pile <newton/examples/cable/example_cable_pile.py>`.

Run, for example, ``uv run -m newton.examples cable_initial_rest_pose``.
Example settings are specific to each scene; check convergence before reusing
them.

Rules of Thumb
~~~~~~~~~~~~~~

*In development.* Distilled starting values and tuning orderings, with the
reasoning that makes them portable across scenes.
