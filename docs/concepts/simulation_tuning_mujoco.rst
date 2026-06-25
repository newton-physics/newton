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
:ref:`Tuning Solver Reference` for the full knob list.

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
unconstrained acceleration ``a0``:

.. math::

   a + d\,(b\,v + k\,r) = (1 - d)\,a_0

``solref`` sets *how* the constraint corrects error (``b``/``k``, i.e.
``timeconst``/``dampratio``); ``solimp`` sets *how much authority* it has
(impedance ``d(r)`` and regularization). Therefore Newton ``ke``/``kd`` on the
:class:`~newton.solvers.SolverMuJoCo` path are **force-space gains feeding
constraint-space reference dynamics**, not a world-space Young's modulus or an
N/m material spring.

Reference Dynamics
------------------

For one constraint row, MuJoCo-Warp computes the reference acceleration as

.. math::

   a_{\mathrm{ref}} = -k_0\,d\,\mathtt{pos} - b_0\,\mathtt{vel}

where ``d`` is the current impedance ``d(r)`` and the gains depend on the active
``solref`` format:

- **Positive format**
  :math:`k_0 = \dfrac{1}{d_{\max}^2\,\mathtt{timeconst}^2\,\mathtt{dampratio}^2}`,
  :math:`b_0 = \dfrac{2}{d_{\max}\,\mathtt{timeconst}}`.
- **Direct format**
  :math:`k_0 = \dfrac{\mathtt{stiffness}}{d_{\max}^2}`,
  :math:`b_0 = \dfrac{\mathtt{damping}}{d_{\max}}`.

The ``solimp`` plateau impedance ``dmax`` therefore normalizes both gains; raising
``dmax`` hardens the row but couples into ``k_0`` and ``b_0`` together.

Mapping ``ke``/``kd`` to ``solref``
-----------------------------------

:class:`~newton.solvers.SolverMuJoCo` converts ``ke``/``kd`` to positive-format
``solref`` using the current impedance. With ``d_width`` the impedance width
factor and ``d_r`` the impedance at the current residual (the same impedance
``d(r)`` used in the constraint mental model above):

.. math::

   \mathtt{timeconst} = \frac{2}{k_d \cdot d_{\mathrm{width}}}, \qquad
   \mathtt{dampratio} = \frac{k_d}{2}\sqrt{\frac{d_r}{k_e}}

with inverse

.. math::

   k_e = \frac{d_r}{d_{\mathrm{width}}^2\,\mathtt{timeconst}^2\,\mathtt{dampratio}^2},
   \qquad k_d = \frac{2}{d_{\mathrm{width}} \cdot \mathtt{timeconst}}

In the impedance plateau (``d_width = 1``, ``d_r = 1``) this reduces to the
intuition ``timeconst ≈ 2 / kd`` and ``dampratio ≈ kd / (2√ke)`` — so ``√ke`` is
the nominal constraint-space natural frequency and ``kd / (2√ke)`` the damping
ratio. The real response also depends on ``solimp``, ``dmax``, the current
``d(r)``, constraint inverse inertia, the friction cone, solver convergence, and
the timestep. To compare damping at fixed stiffness, hold ``ke`` and set
``kd = 2·ζ·√ke``; do not raise ``ke`` or ``kd`` alone.

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

The solver-side regularization satisfies ``R_eff = max(invweight·(1-d)/d, ε)``
and ``efc_D = 1/R_eff``; smaller ``R_eff`` (larger ``efc_D``) is a harder
constraint row but can condition worse. ``efc_D`` is the inverse regularization,
not the regularizer itself.

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
     - Reduce ``timeconst`` (raise ``kd``); raise ``ke``
     - Stability margin; may require smaller ``dt``
   * - Faster error correction
     - Reduce ``timeconst`` (raise ``kd``)
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
     - Hold ``ke`` fixed; raise ``kd`` so ``kd = 2·ζ·√ke`` with ``ζ ≥ 1``
     - Slightly slower error correction
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
     - Lower ``dampratio`` toward 1; measure energy per step before
       changing stiffness
     - Fidelity at impact

Hardness is mainly ``timeconst``/``ke`` and ``d(r)``; stability depends on
``timeconst``, ``ke``, ``kd``, ``d(r)``, ``dt``, solver, friction, cone,
geometry, mass/inertia, and controller.

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

- Start with conservative contact: low ``ke``, default ``solimp`` (wide
  ``width``, low ``dmax``).
- Inspect initial contacts — overlapping geometries at spawn cause immediate
  instability.
- Check joint parameters: ranges, armature, and damping; flag joints with zero
  inertia or zero armature.
- Check drives: verify gains, effort limits, and target values are physically
  reasonable.
- Check capacity: confirm mass, inertia, and friction are plausible.
- Only harden contact (raise ``ke``/``kd``, tighten ``solimp``) once the asset
  simulates stably with gravity and light loading.

Tabletop Support / Pressing / Stacking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Goal: reduce penetration, keep support stable, and suppress bounce and chatter.*

- Use medium-to-high ``ke``; use medium-to-high ``kd`` (target ``kd ≈ 2·√ke``
  for near-critical contact).
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
- Lower damping (``dampratio`` toward 1, or lower ``kd``) to preserve rebound;
  overdamped contact absorbs energy that should transfer.
- Reduce ``dt`` or increase substeps — high stiffness is more stable at small
  timesteps.
- Judge contact quality by energy retention and rebound height, not penetration
  alone; excessive dissipation is as wrong as excessive bounce.

Grasping / Holding
~~~~~~~~~~~~~~~~~~

*Goal: prevent slipping, reduce stick-slip oscillation, and keep contact forces
stable across the grasp.*

- Check normal force first: insufficient normal force cannot be fixed by any
  friction or stiffness setting.
- Then check friction: raise ``mu`` before touching stiffness.
- Then check contact stiffness: raise ``ke``/``kd`` to stiffen the contact
  patch if friction is adequate but the grasp deflects.
- Tune ``impratio`` and the friction cone shape if stick-slip persists after
  normal force and friction are correct.
- Never use higher stiffness as a substitute for insufficient friction capacity;
  it increases constraint load without fixing the root cause.

Articulated Joints
~~~~~~~~~~~~~~~~~~

*Goal: doors, drawers, knobs, and switches stop naturally; joint limits do not
jitter; drives behave as intended.*

- Verify drive import: confirm gains, effort limits, and target mode match the
  intended behavior.
- Add joint friction (``jnt_frictionloss`` in MuJoCo; solver-equivalent damping
  on other solvers) so joints resist motion without a drive.
- Add armature to low-inertia joints to damp high-frequency oscillation; small
  values (0.01–0.1 kg·m²) are often sufficient.
- Add passive damping (``jnt_damping``) to prevent free-spinning at zero
  velocity.
- Tune joint limit stiffness and damping separately from contact stiffness;
  limit jitter usually requires raising ``kd`` on the limit, not on the contact.
- Clip controller targets to the joint range; drives that demand positions beyond
  the limits fight the limit constraint and destabilize the joint.
