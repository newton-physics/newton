.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Tuning MuJoCo:

MuJoCo-Warp Contact Tuning
==========================

This page explains how :class:`~newton.solvers.SolverMuJoCo` interprets contact
and constraint parameters, so that :attr:`~Model.shape_material_ke` and
:attr:`~Model.shape_material_kd` can be tuned with intent. See
:ref:`Simulation Tuning` for the diagnostic workflow and
:ref:`Tuning Solver Reference` for the full knob list. For more details about
Newton-to-MuJoCo mappings, contact-pipeline behavior, and solver-option
resolution, see :doc:`MuJoCo Solver </solvers/mujoco>`.

.. important::

   The specific values, mode names, and formulas on this page reflect the code at
   a point in time and can drift. Treat them as starting points and verify any you
   rely on against the cited source (for example
   :class:`~newton.solvers.SolverMuJoCo` and its kernels). See
   :ref:`Simulation Tuning` for the full guidance.

Constraint Mental Model
-----------------------

.. note::

   This section condenses MuJoCo's own constraint model into the terms a Newton
   user tunes; it is not a new formulation. For the authoritative treatment, see
   the MuJoCo references on `constraint computation
   <https://mujoco.readthedocs.io/en/stable/computation/index.html#constraint-model>`__
   and `solver parameters
   <https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters>`__.

MuJoCo-style contact, limit, and equality constraints are not explicit
spring-damper penalties in world space. A more accurate picture is a *soft servo
in constraint space*. For one scalar constraint row with residual ``r``,
constraint-space velocity ``v``, impedance ``d`` (from ``solimp``), and the
constrained and unconstrained accelerations ``a`` and ``a0``:

.. math::

   a + d (b v + k r) = (1 - d) a_0

``solref`` sets *how* the constraint corrects error (``b``/``k``, i.e.
``timeconst``/``dampratio``); ``solimp`` sets *how much authority* it has
(impedance ``d(r)`` and regularization).

``solref`` Formats
------------------

- **Positive format** ``solref = (timeconst, dampratio)``: ``timeconst`` is how
  fast error is removed (smaller is harder, faster); ``dampratio`` is the
  damping ratio (below 1 rebounds, ~1 is near-critical, above 1 is sluggish and
  dissipative). Raising ``dampratio`` at fixed ``timeconst`` also changes the
  effective stiffness — compare damping at fixed ``ke`` instead.
- **Direct format** ``solref = (-stiffness, -damping)`` (both negative):
  directly specifies position-error stiffness and velocity-error damping.
  Clearer for system identification.

``solimp`` Impedance Curve
--------------------------

``solimp = (d0, dmax, width, midpoint, power)`` defines the impedance ``d(r)``;
``d`` near 1 is hard, near 0 is soft. It is not a second stiffness — it sets
regularization and the soft-to-hard transition.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Parameter
     - Meaning
     - Tuning intuition
   * - ``d0``
     - impedance near zero residual
     - raise to harden shallow contact; expect possible force jumps
   * - ``dmax``
     - plateau impedance at depth
     - raise to cut deep penetration; conditioning may worsen
   * - ``width``
     - residual scale of the ``d0``→``dmax`` transition
     - reduce to reach hard contact sooner
   * - ``midpoint``
     - inflection of the transition curve
     - controls whether hardening happens early or late
   * - ``power``
     - shape of the transition curve
     - controls smoothness; larger is not simply harder

At solver level, ``invweight`` is the row's inverse weight and ``ε`` prevents
division by zero. The effective regularization satisfies
``R_eff = max(invweight·(1-d)/d, ε)`` and ``efc_D = 1/R_eff``. A smaller
``R_eff`` (larger ``efc_D``) is a harder constraint row but can condition worse;
``efc_D`` is the inverse regularization, not the regularizer itself.

Reference Dynamics
------------------

For one constraint row, MuJoCo-Warp computes the reference acceleration as

.. math::

   a_{\mathrm{ref}} = -k_0\,d\,\mathtt{pos} - b_0\,\mathtt{vel}

Here ``pos`` and ``vel`` are the implementation names for the constraint
residual ``r`` and constraint-space velocity ``v`` introduced above, and ``d``
is the current impedance ``d(r)``. The gains depend on the active ``solref``
format:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Format
     - :math:`k_0`
     - :math:`b_0`
   * - Positive
     - :math:`\dfrac{1}{d_{\max}^2\,\mathtt{timeconst}^2\,\mathtt{dampratio}^2}`
     - :math:`\dfrac{2}{d_{\max}\,\mathtt{timeconst}}`
   * - Direct
     - :math:`\dfrac{\mathtt{stiffness}}{d_{\max}^2}`
     - :math:`\dfrac{\mathtt{damping}}{d_{\max}}`

The ``solimp`` plateau impedance ``dmax`` therefore normalizes both gains; raising
``dmax`` hardens the row but couples into ``k_0`` and ``b_0`` together.

Identify the Active Contact Mapping
-----------------------------------

Newton ``ke``/``kd`` arrays retain their documented units (``N/m`` and
``N·s/m``), but their realized meaning on the
:class:`~newton.solvers.SolverMuJoCo` path depends on the active mapping. The
legacy per-geometry conversion treats their numeric values as unit-mass
reference dynamics and does not preserve a physical force-space response across
masses. Existing force-space-mode contacts apply inverse-weight scaling before
the same conversion. Neither path is a Young's modulus or a world-space penalty
spring; see :ref:`mujoco-contact-solref-conversion`.

Before tuning ``ke`` and ``kd``, determine whether the contact uses authored raw
MuJoCo values, Newton's per-geometry conversion, or the per-contact force-space
path. The active interpretation depends on imported metadata, contact path, and
backend. See :ref:`shape-material-contact-stiffness-and-damping` for the mode
definitions and path conditions, and :ref:`mujoco-contact-solref-conversion` for
the exact conversion and ``refsafe`` behavior.

Authored raw ``solref`` retains native MuJoCo meaning. Existing force-space-mode
contacts aim to make normal-contact tuning more transferable across effective
masses, but the mode constants and direct symbolic selection are internal; do
not import them from ``newton._src`` or use a magic integer to opt in. Imported
metadata selects authored/default behavior automatically. The force-space mode
is documented here to interpret existing models and implementation behavior,
not as a supported user-selection workflow.

If an existing model uses force-space mode, evaluate damping and timestep safety
with :ref:`contact-stiffness-sanity-checks`. The mode does not add friction,
normal force, contacts, or controller effort. If ``refsafe`` limits the requested
positive-format response, reduce ``dt`` or increase substeps rather than
repeatedly raising the gains.

Make Harder vs. Make Stable
---------------------------

These two goals require different actions and have different costs. Choose the
goal that matches the actual failure, not the one that seems most intuitive.

**Making contact harder** (less penetration, faster correction):

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - Goal
     - Action
     - Cost
   * - Less penetration
     - Raise ``ke`` and retune ``kd`` for the desired damping ratio
     - Stability margin; may require smaller ``dt``
   * - Faster error correction
     - Raise the active mode's natural frequency; for raw positive ``solref``,
       reduce ``timeconst`` at fixed ``dampratio``
     - Stability margin; harder constraint rows
   * - Higher plateau impedance
     - Raise ``dmax`` in ``solimp``
     - Solver conditioning may worsen
   * - Sharper soft-to-hard transition
     - Reduce ``width`` in ``solimp``
     - Less cushioning; potential force jumps
   * - Finer timestep support for stiffness
     - Reduce ``dt`` or increase substeps
     - Runtime

**Making contact more stable** (reduce jitter, NaN, energy injection):

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - Goal
     - Action
     - Cost
   * - Less bounce without changing stiffness
     - Hold ``ke`` fixed; raise ``kd`` toward critical damping for the active
       ``solref_mode``
     - Less desired rebound; excessive ``kd`` can overdamp
   * - Eliminate NaN or energy injection
     - Reduce ``ke`` and ``dmax``; raise ``width``; reduce ``dt``
     - More penetration; runtime
   * - Reduce jitter at steady contact
     - Reduce ``dt``; increase substeps; improve collision geometry and
       body inertia
     - Runtime; setup effort
   * - Improve grasp stability
     - Verify friction, contact count, and normal force; check controller
       limits and drive gains
     - Setup effort
   * - Reduce oscillation at impact
     - Move ``dampratio`` toward 1 (raise it if below 1, lower it only if
       overdamped); measure energy per step before changing stiffness
     - Fidelity at impact

Hardness is mainly ``timeconst``/``ke`` and ``d(r)``; stability depends on
``timeconst``, ``ke``, ``kd``, ``d(r)``, ``dt``, solver, friction, cone,
geometry, mass/inertia, and controller.

Do not treat raising ``kd`` as equivalent to reducing an independently authored
``timeconst``. In Newton's positive conversion, raising ``kd`` also changes the
mapped ``dampratio``. For force-space contacts, choose damping using the
effective-mass check in :ref:`contact-stiffness-sanity-checks`; see
:ref:`mujoco-contact-solref-conversion` for the exact mapping.

.. _friction-cone-choice:

Friction Cone Choice
--------------------

:class:`~newton.solvers.SolverMuJoCo` exposes MuJoCo's elliptic and pyramidal
friction cones; inspect the constructor or resolved model option for the active
choice. MuJoCo describes elliptic cones as closer to physical friction and
better for suppressing slip, but more expensive. Pyramidal cones can improve
algorithm performance, though not necessarily for every model. If the elliptic
solve is too costly or does not converge within the available budget, compare
pyramidal with the timestep, solver settings, and contact parameters held fixed;
do not assume either cone is universally more stable.

Changing the cone changes the soft-contact model, not only the solver. See
MuJoCo's `solver-setting guidance
<https://mujoco.readthedocs.io/en/stable/modeling.html#solver-settings>`__,
`cone option reference
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-cone>`__, and
`friction-cone formulation
<https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones>`__.

Solver Options and Capacity
---------------------------

A few :class:`~newton.solvers.SolverMuJoCo` options dominate behavior in
practice:

- **Integrator.** Inspect the resolved integrator before tuning stiff joint
  drives. Compare alternatives only for a specific integration failure and
  expect the stable timestep to change.
- **Contact path.** Determine whether MuJoCo or Newton generates contacts and
  tune within that path rather than mixing assumptions from both. See
  :ref:`mujoco-collision-pipeline` for the exact selection behavior.
- **Contact margin and gap.** In Newton collision generation, ``margin`` sets
  the shifted contact surface and ``gap`` adds speculative detection distance.
  Positive gaps increase detected contacts before they become active and can
  therefore affect capacity and cost. See :ref:`mujoco-margin-gap-mapping` for
  exact forwarding, import, inactive-contact, and native-CCD behavior.
- **Armature as a stabilizer.** A small :attr:`~Model.joint_armature` on light,
  high-gain joints raises effective joint inertia and tames stiff drives on the
  MuJoCo path; justify the magnitude with actuator or gearbox data where
  possible.

``nconmax`` and ``njmax`` size the **per-world** contact and constraint buffers.
Set them for the busiest world, not the average: a buffer that fits a quiet
world can truncate contacts or constraints in a heavier one, while an oversized
buffer wastes GPU memory multiplied across every world. If left unset, they are
estimated from the initial state; monitor overflow counters or warnings and raise
the relevant buffer when needed. A positive gap can increase the number of
detected contacts even though contacts outside the margin remain inactive.
After changing gaps or upgrading margin/gap behavior, remeasure peak contacts,
constraints, and overflow in the busiest world before compensating with
stiffness or iterations; do not assume the previous run generated the same rows.
Large gaps and oversized buffers also increase work and memory.

In batched, many-world runs everything per step is multiplied by the world
count: total buffer memory scales with ``nconmax``/``njmax`` times the number of
worlds, and a parameter that is only marginally stable will diverge in *some*
worlds even if most are fine. Tune to the worst-case world and keep per-step
work (solver iterations, substeps, contact count) modest, since each multiplies
by the world count.

Task Templates
--------------

Each template below gives a goal and a sequence of parameter-direction steps.
The workflow logic applies to any solver; ``solimp``/``solref`` advice is
MuJoCo-specific. For which solvers support armature, effort limits, and joint
friction, see :ref:`Tuning Solver Reference`.

New Asset Import
~~~~~~~~~~~~~~~~

*Goal: verify stable simulation before adding performance requirements; catch
geometry, joint, and controller problems early.*

- Start with conservative contact gains and inspect the resolved ``solimp``;
  do not assume its current default is loose or firm without checking it.
- Inspect initial contacts — overlapping geometries at spawn cause immediate
  instability.
- Check joint parameters: ranges, damping, and effective inertia. Zero armature
  is valid; inspect it only when reflected actuator or gearbox inertia is
  expected, and flag zero or implausible effective inertia instead.
- Check drives: verify gains, effort limits, and target values are physically
  reasonable.
- Check model plausibility: confirm mass, inertia, and friction are physically
  reasonable.
- Check capacity: ensure contact/constraint row limits (``nconmax``, ``njmax``)
  and contact buffers are not overflowing or dropping contacts.
- Only harden contact (raise ``ke``/``kd``, tighten ``solimp``) once the asset
  simulates stably with gravity and light loading.

Tabletop Support / Pressing / Stacking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Goal: reduce penetration, keep support stable, and suppress bounce and chatter.*

- Choose ``ke`` and ``kd`` together using the active contact mapping. For
  force-space contacts, use :ref:`contact-stiffness-sanity-checks`; for the exact
  positive conversion, see :ref:`mujoco-contact-solref-conversion`.
- Raise ``dmax`` in ``solimp`` to cut deep penetration; raise ``d0`` only if
  shallow contact is also too soft.
- Increase substeps if the contact must be hard and the timestep cannot shrink.
- Verify the controller maintains a downward force; loss of support often
  traces to drive saturation, not contact stiffness.

Impact / Rebound
~~~~~~~~~~~~~~~~

*Goal: limit penetration on collision, preserve reasonable rebound, and maintain
energy and velocity transfer.*

- Raise stiffness (higher ``ke``, lower ``timeconst``) to limit penetration
  depth.
- If contact is overdamped, move ``dampratio`` toward 1 or reduce ``kd`` using
  the active mapping; overdamped contact absorbs energy that should transfer.
- Reduce ``dt`` or increase substeps — high stiffness is more stable at small
  timesteps.
- Judge contact quality by energy retention and rebound height, not penetration
  alone; excessive dissipation is as wrong as excessive bounce.

Grasping / Holding
~~~~~~~~~~~~~~~~~~

*Goal: prevent slipping, reduce stick-slip oscillation, and keep contact forces
stable across the grasp.*

- Check commanded and clamped gripping force first: insufficient available
  normal force cannot be replaced by friction or stiffness tuning.
- Then check friction: raise ``mu`` before touching stiffness.
- Then check contact stiffness: raise ``ke``/``kd`` to stiffen the contact
  patch if friction is adequate but the grasp deflects.
- Prefer an elliptic cone and tune ``impratio`` if stick-slip persists. Try a
  pyramidal cone if solver convergence or cost is the limiting issue, then
  revalidate the grasp; see :ref:`Friction Cone Choice <friction-cone-choice>`.
- Never use higher stiffness as a substitute for insufficient friction capacity;
  it increases constraint load without fixing the root cause.

Articulated Joints
~~~~~~~~~~~~~~~~~~

*Goal: doors, drawers, knobs, and switches stop naturally; joint limits do not
jitter; drives behave as intended.*

- Verify drive import: confirm gains, effort limits, and target mode match the
  intended behavior.
- Add joint friction (``Model.joint_friction``; MJCF ``frictionloss``) so joints
  resist motion without a drive. This is Coulomb friction loss, not viscous
  damping. On solvers without Coulomb friction, damping can slow motion but
  cannot reproduce static friction or hold a load at zero velocity.
- Add physically justified armature to low-inertia joints to damp high-frequency
  oscillation. Scale it relative to reflected inertia; its units are ``kg·m²``
  for revolute joints and ``kg`` for prismatic joints.
- Add passive damping (``Model.joint_damping``; MJCF ``damping``) to slow
  unwanted motion at zero command.
- Tune joint limit stiffness and damping separately from contact stiffness;
  limit jitter usually requires raising ``kd`` on the limit, not on the contact.
- Clip controller targets to the joint range; drives that demand positions beyond
  the limits fight the limit constraint and destabilize the joint.
