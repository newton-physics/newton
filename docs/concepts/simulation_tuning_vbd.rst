.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Tuning VBD:

VBD Tuning
==========

This page explains how :class:`~newton.solvers.SolverVBD` reacts to its solver
and model parameters, what typically goes wrong and how to fix it, and which
parameter combinations are known to work. It covers particle systems: cloth
(triangle meshes) and soft bodies (tetrahedral meshes), their self-contact, and
their contact with rigid and static shapes, grippers included. Rods, cables and
rigid articulations are out of scope here; see :ref:`Tuning Solver Reference`
for their knobs and :doc:`VBD </solvers/vbd>` for setup and API explanations.
:ref:`Simulation Tuning` holds the diagnostic workflow that this page assumes.

.. important::

   The specific values, mode names, and formulas on this page reflect the code
   at a point in time and can drift. Treat them as starting points and verify
   any you rely on against the cited source (for example
   :class:`~newton.solvers.SolverVBD` and its kernels). See
   :ref:`Simulation Tuning` for the full guidance.

Mental Model
------------

VBD (Vertex Block Descent) treats every time step as one minimization and
solves it one small block at a time. Two kinds of blocks exist. A **particle**
:math:`i` has 3 degrees of freedom, its position :math:`\mathbf{x}_i`, and a
mass :math:`m_i`. A **rigid body** :math:`b` has 6 degrees of freedom, its pose
:math:`\mathbf{q}_b` made of a centre of mass :math:`\mathbf{c}_b` and a
rotation :math:`\mathbf{R}_b`, with a mass :math:`m_b` and an inertia tensor
:math:`\mathbf{I}_b`. The notation follows the VBD paper cited in
:class:`~newton.solvers.SolverVBD`: :math:`h` is the time step and
:math:`\mathbf{y}` the inertial prediction.

One step is one minimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The step finds the positions and poses that minimize

.. math::

   G = \sum_i \frac{m_i}{2h^2}\,\lVert \mathbf{x}_i - \mathbf{y}_i \rVert^2
     + \sum_b \left[ \frac{m_b}{2h^2}\,\lVert \mathbf{c}_b - \mathbf{c}_b^* \rVert^2
     + \frac{1}{2h^2}\,\boldsymbol{\theta}_b^{\mathsf T} \mathbf{I}_b\, \boldsymbol{\theta}_b \right]
     + \sum_e E_e .

The first sum is the particle inertia. Its target is the inertial prediction

.. math::

   \mathbf{y}_i = \mathbf{x}_i + h\left(\mathbf{v}_i + h\left(\mathbf{g} + \mathbf{f}^{\text{ext}}_i / m_i\right)\right),

computed by the ``forward_step`` kernel from the current velocity, gravity and
external forces. The second sum is the rigid inertia. Its target pose
:math:`(\mathbf{c}_b^*, \mathbf{R}_b^*)` comes from forward integration
(``forward_step_rigid_bodies``), and :math:`\boldsymbol{\theta}_b` is the
rotation vector from the current rotation to the target rotation
(``solve_rigid_body``). The last sum is every energy term in the scene. Each
term depends on a few blocks and on model parameters:

.. list-table::
   :header-rows: 1
   :widths: 22 22 30 26

   * - Energy term
     - Acts on
     - Model parameters
     - Kernel
   * - Membrane (stable Neo-Hookean per unit area)
     - the 3 particles of a triangle
     - ``tri_ke`` (:math:`\mu`), ``tri_ka`` (:math:`\lambda`), ``tri_kd``
     - ``evaluate_neo_hookean_membrane_force_hessian``
   * - Bending (dihedral angle hinge)
     - the 4 particles of an interior edge
     - ``edge_ke``, ``edge_kd``
     - ``evaluate_dihedral_angle_based_bending_force_hessian``
   * - Volumetric (stable Neo-Hookean)
     - the 4 particles of a tetrahedron
     - ``k_mu``, ``k_lambda``, ``k_damp``
     - ``evaluate_volumetric_neo_hookean_force_and_hessian``
   * - Self-contact, with friction and damping
     - a vertex and a triangle, or two edges
     - ``soft_contact_ke``, ``soft_contact_kd``, ``soft_contact_mu``
     - ``evaluate_self_contact_force_norm``, ``compute_friction``, ``damp_collision``
   * - Rigid-soft contact, with friction and damping
     - a particle, edge or face and one body or static shape
     - the mean of ``soft_contact_ke`` and the shape ``ke``; the geometric
       mean of the two ``mu`` values; ``particle_radius`` and the shape
       ``margin`` set the distance
     - ``init_body_particle_contacts``, ``_compute_body_particle_contact_force``
   * - Attachments and springs
     - a particle and a body, or two particles
     - attachment and spring stiffness and damping
     - ``accumulate_body_particle_attachment_force_and_hessian``,
       ``accumulate_spring_force_and_hessian``
   * - Joints and body-body contacts
     - one or two bodies
     - joint stiffness and damping, shape materials
     - compliant augmented-Lagrangian rows; see :doc:`VBD </solvers/vbd>`

Joints and body-body contacts use a compliant augmented-Lagrangian formulation.
Their tuning is not covered here; the backend guide explains it.

How it is solved
~~~~~~~~~~~~~~~~

The solver never forms the full system. In each iteration it visits every block
once, with the neighbours frozen at their current values. For a block it
assembles the gradient :math:`\mathbf{g}_i` and the Hessian :math:`\mathbf{H}_i`
of the terms that touch the block, and moves the block by one Newton step:

.. math::

   \Delta \mathbf{x}_i = -\mathbf{H}_i^{-1}\,\mathbf{g}_i, \qquad
   \mathbf{H}_i = \frac{m_i}{h^2}\,\mathbf{1} + \sum_{e \ni i} \frac{\partial^2 E_e}{\partial \mathbf{x}_i^2}.

For a particle this is a 3x3 solve (``solve_elasticity``). For a body it is a
6x6 solve with the block Hessian
:math:`\operatorname{diag}(m_b/h^2\,\mathbf{1}_3,\ \mathbf{I}_b/h^2)` plus the
energy terms, factored once per visit (``solve_rigid_body``, ``ldlt6_solve``).
Blocks are grouped into colours so that no two blocks of one colour share an
energy term. The update is Gauss-Seidel across colours and Jacobi inside a
colour. Rigid blocks are solved before particle blocks in every iteration.

The real call tree, with the control flow written out:

.. code-block:: text

   SolverVBD.step(state_in, state_out, control, contacts, dt)
     _initialize_rigid_bodies   -> forward_step_rigid_bodies   (predicted poses)
     _initialize_particles      -> forward_step                (predicted positions y)
     collision detection (default schedule: before and after this initialization)
     for iteration in range(iterations):
         if detection is due in this iteration (ITERATIONS schedule only):
             detect again, refresh the DAT reference
         _solve_rigid_body_iteration
             for colour in body colours:
                 accumulate attachments, body-particle contacts, body-body contacts
                 solve_rigid_body            (6x6 LDL^T per body)
             update dual variables of the constraint rows
         _solve_particle_iteration
             for colour in particle colours:
                 accumulate attachment, body-particle contact, spring and self-contact terms
                 solve_elasticity            (3x3 per particle: inertia + membrane + bending + volumetric)
                 DAT truncation of the accumulated displacement
     _finalize_rigid_bodies     -> update_body_velocity
     _finalize_particles        -> update_velocity             v = (x_new - x_old) / dt

Two facts follow directly. Information moves one ring of neighbours per
iteration, because a block only sees its neighbours' current values. And an
unfinished solve is not discarded: whatever position the last iteration leaves
becomes the new velocity through :math:`\mathbf{v} = (\mathbf{x}_{\text{new}} -
\mathbf{x}_{\text{old}})/h`.

Penetration-free truncation (DAT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contacts are penalty energies, so a large step could jump through a surface
between two detections. The Divide and Truncate scheme (DAT) prevents that.
After each detection, every vertex receives a **motion budget**: the relaxation
factor ``dat_conservative_bound_relaxation`` (default 0.85) times the nearest
detected distance among its own elements, capped at the detection query radius
(``compute_particle_conservative_bound``). Between two detections, the
displacement accumulated since the last detection may not exceed that budget;
the solver scales it down in every direction when it would
(``apply_conservative_bound_truncation``, isotropic truncation). For each
detected pair a division plane is placed between the two primitives, and each
side keeps its own side of the plane (``apply_planar_truncation_parallel_by_collision``).
The solver-level budget between detections is one half of the relaxation factor
times the query radius, one half per side. The same scheme can be enabled for
rigid-soft contacts with ``rigid_soft_enable_dat``.

The guarantee has a price: a vertex cannot move faster than about one motion
budget per detection cycle. This page calls the motion that the truncation
removes **DAT damping**. It is not related to any ``kd`` damping coefficient.

What follows for tuning
~~~~~~~~~~~~~~~~~~~~~~~

- Stiff terms dominate :math:`\mathbf{H}_i`. A contact that is far stiffer than
  the elasticity makes the local step serve the contact and neglect the rest.
  See :ref:`vbd-fail-contact-stiffness`.
- Information moves one ring per iteration. A fine mesh needs more iterations
  to reach the same state. See :ref:`vbd-fail-resolution`.
- An unfinished solve becomes velocity. Corrections that did not settle inside
  the step come back as motion in the next step.
- Truncation removes motion. A small query radius means a small budget and
  visible DAT damping. See :ref:`vbd-fail-dat-damping`.
- Float32 sets a floor on the time step. See :ref:`vbd-fail-small-dt`.

Parameter Effects
-----------------

VBD-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~

The constructor arguments of :class:`~newton.solvers.SolverVBD` that matter for
particle systems. Defaults are those of the constructor at the time of
writing; the class docstring is the source of truth.

.. list-table::
   :header-rows: 1
   :widths: 26 30 24 20

   * - Parameter
     - What it controls
     - Raise or enable when
     - Cost or risk
   * - ``iterations`` (10)
     - Block-descent sweeps per substep; how far information travels and how
       far the solve converges.
     - The object looks too soft or lags; a finer mesh; as a last resort for
       jitter.
     - Runtime grows linearly. Does not fix a stiffness-ratio problem.
   * - ``friction_epsilon`` (1e-2)
     - Relative velocity below which friction is smoothed; the smoothing
       length is ``friction_epsilon`` x ``dt``.
     - Friction chatters at rest.
     - Slow creep under tangential load when large.
   * - ``dat_conservative_bound_relaxation`` (0.85)
     - Fraction of the detected clearance a vertex may use between detections.
     - Rarely; DAT damping with a query radius you cannot raise.
     - Values near 1 leave no safety for detection error.
   * - ``particle_enable_self_contact`` (False)
     - Vertex-triangle and edge-edge self-contact with DAT.
     - Cloth folds or stacks; soft bodies touch themselves.
     - Detection cost; requires a non-``NONE`` self-contact schedule.
   * - ``particle_self_contact_margin`` (0.2)
     - Interaction distance: the surface offset at which self-contact forces
       start. Always pass it together with the gap.
     - Visible gaps between layers are too small; see
       :ref:`vbd-margin-gap-guide`.
     - Rest neighbours beyond the topological filter push each other when it
       exceeds their rest distance.
   * - ``particle_self_contact_gap`` (0)
     - Detection slack: pairs are found out to margin plus gap, and the motion
       budget scales with that sum.
     - Motion looks capped or sticky (DAT damping); fast relative motion.
     - More candidate pairs, larger buffers.
   * - ``particle_vertex_contact_buffer_size`` (32),
       ``particle_edge_contact_buffer_size`` (64)
     - Candidate pairs stored per vertex and per edge. Pairs beyond the
       capacity are dropped silently.
     - Many layers or a large gap; tunneling. See
       :ref:`vbd-fail-tunneling`.
     - Memory, detection time.
   * - ``particle_topological_contact_filter_threshold`` (2)
     - Ring distance under which candidate pairs are discarded.
     - A coarse mesh with a large interaction distance.
     - Values above 3 raise the cost significantly (docstring).
   * - ``particle_rest_shape_contact_exclusion_radius`` (0)
     - Pairs closer than this in the rest shape are never contacts.
     - Seams, layered cloth, or thin regions that are close by design.
     - Real contacts in those regions are missed too.
   * - ``particle_enable_tile_solve`` (True)
     - Tile-based elasticity kernel specialized at construction.
     - Keep on. Rebuild the solver after changing stiffness values.
     - None in normal use.
   * - ``collision_frequency``, ``collision_frequency_type``
       (``SOFT_SELF_CONTACT`` slot)
     - When self-contact detection runs. ``AUTO`` resolves to before and
       after the inertial step. ``ITERATIONS`` with frequency :math:`k` also
       detects before iterations :math:`k, 2k, \ldots`
     - Fast motion with a small gap; as a last resort for jitter.
     - One detection pass per refresh.
   * - ``rigid_soft_contact_use_log_barrier`` (False)
     - Rigid-soft normal law: quadratic penalty, or the self-contact barrier.
     - A gripper must never let a thin sheet through.
     - Stiffer local systems, more iterations.
   * - ``rigid_soft_enable_dat`` (False)
     - DAT truncation of rigid pose updates and particles against rigid-soft
       contact planes.
     - Thin cloth between fast rigid parts.
     - Rigid motion is capped by the rigid-soft query radius.
   * - ``rigid_body_particle_contact_buffer_size`` (256)
     - Rigid-soft contact records per body.
     - Dense cloth on a large body; grippers with many vertices in contact.
     - Memory.

Deprecated aliases remain for one release cycle: ``particle_self_contact_radius``
(legacy radius and margin pair), ``particle_collision_detection_interval``
(now the self-contact slot of the collision schedule) and
``particle_conservative_bound_relaxation`` (now
``dat_conservative_bound_relaxation``). New code should not use them. Passing a
margin without a gap also triggers the legacy meaning of the margin and a
deprecation warning, so always pass both.

Model Parameters
~~~~~~~~~~~~~~~~

Model parameters are set when the geometry is added, through
:meth:`~newton.ModelBuilder.add_cloth_mesh`,
:meth:`~newton.ModelBuilder.add_cloth_grid`,
:meth:`~newton.ModelBuilder.add_soft_mesh` and
:meth:`~newton.ModelBuilder.add_soft_grid`, or on the
:class:`~newton.Model` after :meth:`~newton.ModelBuilder.finalize`.

.. list-table::
   :header-rows: 1
   :widths: 24 30 24 22

   * - Parameter
     - What it controls
     - Raise when
     - Cost or risk
   * - ``tri_ke``, ``tri_ka`` [N/m]
     - Membrane stretch and area stiffness (:math:`\mu` and :math:`\lambda`
       per unit area of the stable Neo-Hookean membrane).
     - Cloth stretches visibly under its own weight.
     - Above about 1e6, slow convergence and jitter under contact; stiffening
       only the membrane moves deformation into bending (see
       :ref:`vbd-fail-membrane-only`).
   * - ``tri_kd``
     - Membrane damping.
     - In-plane ringing after impacts.
     - Overdamped, slow drape.
   * - ``edge_ke`` [N]
     - Bending stiffness of the dihedral hinge. The hinge energy is
       :math:`\tfrac{1}{2}\,\texttt{edge\_ke}\; L_{\text{rest}}\,(\theta - \theta_0)^2`.
     - Cloth folds too sharply or looks too limp.
     - Stiff hinges need more iterations; raise membrane and bending together
       for a stiffer object.
   * - ``edge_kd``
     - Bending damping.
     - Bending jitter; first knob of the jitter pipeline for cloth without
       contacts.
     - Slow unfolding.
   * - ``k_mu``, ``k_lambda`` [Pa]
     - Volumetric Neo-Hookean shear and volume stiffness.
     - A soft body sags or inverts too easily.
     - Same ceiling as the membrane; ``k_lambda`` far above ``k_mu`` makes a
       stiff local system.
   * - ``k_damp`` [Pa s]
     - Volumetric viscous damping.
     - Soft-body jitter; the soft-body knob of the jitter pipeline.
     - Slow recovery.
   * - ``soft_contact_ke`` [N/m]
     - Contact stiffness for self-contact; averaged with the shape ``ke`` for
       rigid-soft contact.
     - Penetration between layers or into shapes that the fix below does not
       cure.
     - Keep within about 10x the elastic stiffness; see
       :ref:`vbd-fail-contact-stiffness`.
   * - ``soft_contact_kd``
     - Contact damping on the gap rate, active only while approaching.
     - Contact jitter; second knob of the jitter pipeline.
     - Shrinks each correction by enlarging the local matrix; too much slows
       settling.
   * - ``soft_contact_mu``
     - Friction coefficient (geometric mean with the shape ``mu`` for
       rigid-soft contact).
     - Cloth slides off a gripper.
     - Sticking, slow sliding.
   * - ``particle_radius`` [m]
     - The particle's half of the rigid-soft interaction distance; also the
       resting offset from shapes.
     - Cloth sinks into shapes.
     - Visible floating; a large radius with a small mesh spacing distorts
       contact geometry.
   * - ``density``, ``mass``
     - Particle masses (per area for cloth, per volume for soft meshes).
     - Never as a fix; set it from the real object.
     - Too small a mass is an explosion cause; see
       :ref:`vbd-explosion-checklist`.
   * - shape ``ke``, ``kd``, ``mu``, ``margin``, ``gap``
       (:class:`~newton.ModelBuilder.ShapeConfig`)
     - The shape's half of the rigid-soft material and distance.
     - Set once per object; the shape ``margin`` adds to the interaction
       distance.
     - Shape values are averaged with the particle side, so one side cannot
       fully override the other.
   * - ``CollisionPipeline(soft_contact_gap=...)`` (0.01)
     - Rigid-soft detection slack, added on top of particle radius plus shape
       margin.
     - Fast relative motion between cloth and shapes.
     - More rigid-soft records; rigid-soft DAT budget grows with it.
   * - ``vel``
     - Initial velocity of the added geometry.
     - Never as a fix; a large initial velocity is an explosion cause.
     - See :ref:`vbd-explosion-checklist`.

How the energies use these values: the membrane energy is the stable
Neo-Hookean energy per unit area with :math:`\mu = \texttt{tri\_ke}` and
:math:`\lambda_{\text{NH}} = \texttt{tri\_ka} + \texttt{tri\_ke}`, so a Poisson
ratio of :math:`\nu` corresponds to :math:`\texttt{tri\_ka} =
2\nu\,\texttt{tri\_ke}/(1-\nu)` (see ``example_cloth_stiff_material_stretch``).
The bending energy is the hinge formula above, integrated along the rest edge
length. The volumetric energy is the stable
Neo-Hookean energy with ``k_mu`` and ``k_lambda`` in pascal, damped by
``k_damp``. Every contact energy is quadratic in the penetration beyond the
interaction distance; the self-contact law switches to a logarithmic barrier
inside half the interaction distance and to a quadratic extension below
10 micrometres, so that the surfaces never cross. Friction follows the IPC
form with a smoothing length of ``friction_epsilon`` x ``dt`` and a contact
normal held fixed inside the block solve. Contact damping acts on the gap rate,
only while the gap closes.

Time step, substeps and iterations
``````````````````````````````````

Most shipped examples render at 60 frames per second with 10 substeps, so the
solver runs at ``dt`` = 1/600 s, with 4 to 20 iterations. Start there with 10
iterations. A smaller ``dt`` improves stiffness and convergence, because every
substep adds a full set of iterations and a detection pass. It also costs the
same factor in runtime. Do not go much below 1/3000 s: positions are stored in
float32, and when the per-step update becomes small compared to the position
value, part of each update is rounded away (one float32 unit at 1 m is about
1.2e-7 m). Before that floor, reducing ``dt`` does more for stiffness than
adding iterations. See :ref:`vbd-stiffer-object` for the order of operations.

Contact Handling
~~~~~~~~~~~~~~~~

Four distances govern a contact. The **interaction distance** is where the
contact energy starts to act. The **detection slack** is the extra reach of
detection, so pairs are known before they interact. The **query radius** is
their sum. The **motion budget** is the DAT limit between two detections, and it
scales with the query radius.

.. figure:: ../images/vbd_tuning_contact_distances.svg
   :alt: Interaction distance, detection slack, query radius and motion budget for self-contact and rigid-soft contact
   :width: 95%
   :align: center

   Left: self-contact. ``particle_self_contact_margin`` is the interaction
   distance, ``particle_self_contact_gap`` the detection slack. Right:
   rigid-soft contact. The interaction distance is ``particle_radius`` plus the
   shape ``margin``; the pipeline's ``soft_contact_gap`` adds detection slack.

For self-contact the solver reads the two values from its constructor and
queries at ``particle_self_contact_margin + particle_self_contact_gap``. For
rigid-soft contact the force kernel uses ``particle_radius + margin`` of the
shape as the interaction distance, and the collision pipeline detects out to
``soft_contact_gap`` beyond that. The default self-contact schedule detects
before and after the inertial step of every substep, so the motion budget is
refreshed twice per substep.

**What DAT does to motion.** DAT keeps the simulation penetration-free between
detections by never letting a vertex use more than its budget. A vertex that
wants to move further is slowed down to the budget in every direction, not only
along the contact normal. A cloth that falls fast onto a shape, or a sheet
pressed between fingers, can therefore move slower than physics would predict,
hover at a fixed offset, or stop. This is DAT damping. The budget grows with the
query radius, so the cure is more room: a larger gap first, then a larger margin.
The pouch example in :ref:`vbd-fail-dat-damping` shows the extreme case where
the cycle amplitude of a jittering vertex follows the budget exactly.

.. _vbd-margin-gap-guide:

Setting margin and gap
``````````````````````

There is no single right value. The choice trades physical fidelity against
cost and DAT damping.

**Fidelity first.** The interaction distance is the resting separation you will
see. Two cloth layers rest one margin apart; a particle rests at its radius plus
the shape margin from a shape. Pick the interaction distance from the thickness
you want to show. Pairs more than two rings apart are contact candidates, so an interaction
distance larger than their rest distance creates contact forces at rest. Where
a mesh is close to itself by design, such as seams or layered
panels, use ``particle_rest_shape_contact_exclusion_radius`` instead of a
smaller margin.

.. list-table::
   :header-rows: 1
   :widths: 18 20 22 22 18

   * - Change
     - Cost
     - Separation
     - DAT damping
     - Other effects
   * - Raise the margin
     - More active pairs
     - Larger visible gaps
     - Less (larger budget)
     - Rest neighbours beyond the topological filter can push each other
   * - Raise the gap
     - More candidate pairs, larger buffers
     - None
     - Less (larger budget)
     - Dropped pairs when the buffers overflow
   * - Lower either
     - Less
     - Tighter contact
     - More
     - Tunneling risk when the budget is below the per-step motion

**Starting point and adjustments.** Take the values of the nearest example in
:ref:`vbd-empirical-table`; they span 1 mm to 40 mm depending on the scene
scale. Then adjust by symptom: a visible gap between layers means a smaller
margin; slow, hovering or locked motion means a larger gap first and a larger
margin second; dropped pairs (see :ref:`vbd-fail-tunneling`) mean larger
buffers. Keep the gap at about half to one margin unless speed asks for more.

**Stiffness, damping and friction together.** The contact stiffness sets how
far a contact penetrates under load and how stiff the local system becomes.
Keep it within about 10x the elastic stiffness of the material it presses on.
Contact damping only acts while the gap closes and mostly shrinks the per
iteration correction, so it is a second-order knob. Friction scales with the
normal force; a contact that is too soft also has too little friction, which
shows as slipping in grasps.

Best Practices
--------------

What Can Go Wrong and How to Fix It
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Symptom
     - Likely cause
     - Entry
   * - Cloth or a soft body jitters, more iterations do not help
     - Contact stiffness far above the elastic stiffness
     - :ref:`vbd-fail-contact-stiffness`
   * - Raising ``tri_ke`` does not make the object stiffer
     - Deformation moves into bending
     - :ref:`vbd-fail-membrane-only`
   * - A fine mesh is soft, laggy or slow to settle
     - One ring of information per iteration
     - :ref:`vbd-fail-resolution`
   * - Motion is slower than expected, hovers or locks near contact
     - DAT damping, small query radius
     - :ref:`vbd-fail-dat-damping`
   * - Explosion or NaN at a few vertices
     - Bad elements in the asset
     - :ref:`vbd-fail-asset`
   * - A smaller ``dt`` made things worse
     - Compute budget, float32 precision
     - :ref:`vbd-fail-small-dt`
   * - Cloth passes through itself
     - Candidate buffers overflow, or an exclusion filter
     - :ref:`vbd-fail-tunneling`

.. _vbd-jitter-pipeline:

Jitter: where to look first
```````````````````````````

Work through these steps in order and stop at the first one that fixes it.

1. If the scene has self-contacts, compare the contact stiffness with the
   elastic stiffness. Keep ``soft_contact_ke`` within about 10x of
   ``tri_ke`` for cloth or ``k_mu`` for soft bodies. This is the stiffness
   ratio entry below.
2. If the jitter stays, tune the self-contact damping, ``soft_contact_kd``.
3. If it still stays, or the scene has no contacts, raise the elastic damping:
   ``edge_kd`` first for cloth, ``k_damp`` for soft bodies. If the scene has
   articulations, also raise ``joint_target_kd``; see
   :ref:`Tuning Solver Reference`.
4. Last resort, because both cost compute: more ``iterations``, or more
   frequent collision detection through the ``ITERATIONS`` schedule of the
   self-contact slot.

.. _vbd-explosion-checklist:

Explosion: what to check, in this order
```````````````````````````````````````

1. **Asset quality.** Add the mesh with ``validate_mesh=True`` and read the
   warnings. See :ref:`vbd-fail-asset`.
2. **Density too small?** A light vertex next to a stiff element or contact
   receives a large acceleration. Check ``density`` or ``mass`` against the
   real object; do not lower them to speed things up.
3. **Stiffness too high?** Compare the elastic and contact stiffness with the
   examples in :ref:`vbd-empirical-table`. Above about 1e6 expect trouble.
4. **Initially penetrating?** Inspect the start pose with the
   :ref:`viewer debugging tools <debugging-with-viewers>`. Layers that start
   inside the interaction distance receive a large separating push in the
   first steps.
5. **Initial stretch or velocity too high?** Compare the start positions with
   the rest shape, and check ``vel`` and any prescribed motion in the first
   frame.

.. _vbd-fail-contact-stiffness:

Contact stiffness too high compared to the elasticity
``````````````````````````````````````````````````````

**Symptom.** Cloth layers or a grasped sheet jitter at a few vertices. The
motion is fast, local, and does not settle when you add iterations.

**Cause.** The stiffness ratio: the contact stiffness is too high compared to
the elasticity.

**Fix.** Bring ``soft_contact_ke`` to within about 10x of the membrane
stiffness (``tri_ke``) or the volumetric stiffness (``k_mu``), then check that
grasps still hold. Lower contact stiffness also means lower friction force, so
a grasp can slip; in the pouch scene below, lowering the contact stiffness to
2x the membrane stiffness lost the grasp. The rule is a starting point, not a
universal one.

.. figure:: ../images/vbd_tuning_stack_contact_stiffness.png
   :alt: Five stacked cloth layers at the same time with contact stiffness 30,000 N/m (wavy) and 4,000 N/m (flat)
   :width: 95%
   :align: center

   Five stacked cloth layers after 6.5 s of rest, same time in both runs.
   Membrane stiffness 2,000 N/m, self-contact margin 8 mm, gap 8 mm,
   ``dt`` = 1/300 s, 20 iterations. Left: contact stiffness 30,000 N/m.
   Right: 4,000 N/m. The scene is a test scene, not a shipped example. Newton
   commit 353e0ea1.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Local motion of free vertices, 5 to 50 Hz band, 2 to 7 s
     - Contact stiffness 30,000 N/m
     - Contact stiffness 4,000 N/m
   * - Mean amplitude (RMS over vertices) [mm]
     - 0.93
     - 0.016
   * - 95th percentile vertex [mm]
     - 1.93
     - 0.020
   * - Worst vertex [mm]
     - 4.79
     - 0.024

"Local motion" is the motion of a vertex relative to the mean of its mesh
neighbours, so a rigid shift of the whole stack does not count.

More iterations did not remove this kind of jitter. On a pinched pouch
(membrane 3,000 N/m, contact 50,000 N/m, ``dt`` = 1/300 s) the mean mesh speed
over 0.1 s was 0.123 m/s at 20 iterations, 0.023 m/s at 40, 0.141 m/s at 100
and 0.205 m/s at 200.

**Reproduce.** A similar setup is ``example_vbd_soft_rigid_contact`` with
``soft_contact_ke`` raised from 1e5 to 1e6 while ``cloth_tri_ke`` stays at 1e5.

.. _vbd-fail-membrane-only:

Stiffening only the membrane does not stiffen the object
`````````````````````````````````````````````````````````

**Symptom.** You raise ``tri_ke`` and ``tri_ka`` by orders of magnitude and the
object still wobbles; at the highest values it buckles and wrinkles more.

**Cause.** The stiffness ratio between the membrane and the bending terms. Once
the membrane is near inextensible, the deformation energy moves into bending,
which stayed soft.

**Fix.** Scale ``edge_ke`` together with ``tri_ke`` and ``tri_ka``. Stiffening
both gives a monotonic stiffening with diminishing returns past the baseline.

.. figure:: ../images/vbd_tuning_bag_stretch_sweep.png
   :alt: Final frames of a pinned bag at five membrane stiffness values from 1e3 to 1e7
   :width: 100%
   :align: center

   A bag with rigid contents, pinned at the rim, rim driven sideways at
   0.85 Hz, after 6 s. Only the membrane stiffness changes, ``edge_ke`` stays
   at 200 N. ``dt`` = 1/600 s, 10 iterations. The x100 bag buckles at the
   bottom. The scene is a test scene, not a shipped example.

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Multiplier
     - Stretch score, membrane only
     - Bend score, membrane only
     - Stretch score, both scaled
     - Bend score, both scaled
   * - x0.01
     - 0.114
     - 0.012
     - 0.012
     - 0.033
   * - x0.1
     - 0.015
     - 0.008
     - 0.007
     - 0.018
   * - x1
     - 0.006
     - 0.015
     - 0.006
     - 0.015
   * - x10
     - 0.006
     - 0.045
     - 0.006
     - 0.015
   * - x100
     - 0.006
     - 0.179
     - 0.006
     - 0.015

Baseline x1: ``tri_ke`` 1e5, ``tri_ka`` 2e4, ``edge_ke`` 200. Scores are mean
deformation measures over the last second of each run; lower means stiffer.
Scaling only the membrane leaves the stretch score flat from x1 on and raises
the bend score twelve-fold. Scaling both keeps every score flat.

**Reproduce.** A similar setup is ``example_cloth_hanging`` with ``tri_ke`` and
``tri_ka`` multiplied by 100 and ``edge_ke`` unchanged, compared with all
three multiplied.

.. _vbd-fail-resolution:

A fine mesh converges slowly
````````````````````````````

**Symptom.** After refining the mesh, the object looks softer, lags behind
contacts, or needs many more frames to settle, with unchanged material values.

**Cause.** Information moves one ring per iteration. A disturbance crosses
:math:`N` rings in no fewer than :math:`N` iterations, so at fixed iterations a
finer mesh is less converged at the end of every step.

**Fix.** Use the lowest resolution that expresses the deformation you need.
When you refine, raise ``iterations`` in proportion to the number of rings
across the object.

.. _vbd-fail-dat-damping:

DAT damping: slow, hovering or locked motion
````````````````````````````````````````````

**Symptom.** Near contacts, cloth falls slower than gravity, hovers at a fixed
offset, or a pinched region stops moving. In the worst case a vertex oscillates
with a fixed amplitude.

**Cause.** The motion budget between detections is too small for the motion the
scene asks for. The budget is the relaxation factor times the detected
clearance, capped at the query radius, so a small margin and gap cap the speed.

**Fix.** Raise ``particle_self_contact_gap`` first, then
``particle_self_contact_margin``, following :ref:`vbd-margin-gap-guide`. For
rigid-soft contact raise the pipeline's ``soft_contact_gap``. More frequent
detection also refreshes the budget more often, at a cost.

.. figure:: ../images/vbd_tuning_pouch_fast_cycles.png
   :alt: Position and velocity traces of three pouch vertices showing two- and four-substep cycles
   :width: 100%
   :align: center

   Three vertices of a pouch held by a gripper, position (left) and velocity
   (right) at every substep, ``dt`` = 1/300 s. Margin 2 mm, gap 1.5 mm,
   relaxation 0.85, so the budget is 1.49 mm. Vertex 589 alternates every
   substep: peak to peak 0.74 mm, half the budget, at 150 Hz. Vertices 560 and
   559 alternate every two substeps: peak to peak 1.49 mm, one budget, at
   75 Hz. The scene is a test scene, not a shipped example. Newton commit
   353e0ea1.

The amplitudes follow the budget exactly, which identifies the truncation as
the limiter. Giving the vertices more room, a larger query radius, changes the
amplitude; adding iterations does not.

.. _vbd-fail-asset:

Bad asset
`````````

**Symptom.** NaN or an explosion at a few vertices from the first frames,
jitter at rest in one region, or warnings when the mesh is added.

**Cause.** Degenerate or sliver elements make the local solve ill-conditioned.

**Fix.** Run the built-in checker, then remesh. The checker is opt-in:

.. code-block:: python

   builder.add_cloth_mesh(
       pos=pos, rot=rot, scale=1.0, vel=wp.vec3(0.0),
       vertices=vertices, indices=indices, density=0.1,
       validate_mesh=True, label="shirt",
   )
   builder.add_soft_mesh(
       pos=pos, rot=rot, scale=1.0, vel=wp.vec3(0.0),
       vertices=vertices, indices=tets, density=100.0,
       k_mu=1.0e5, k_lambda=1.0e5, k_damp=1.0,
       validate_mesh=True, label="sponge",
   )

It never raises; it warns and adds the mesh anyway. The same checks are
available directly as :func:`newton.utils.validate_triangle_mesh` and
``newton.utils.validate_tet_mesh``. Thresholds at the time of writing:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Mesh
     - Check
     - Threshold
   * - Triangles
     - Small area
     - below 1 mm² (1e-6 m²)
   * - Triangles
     - Sliver, longest edge over shortest altitude
     - above 20
   * - Triangles
     - Minimum interior angle
     - below 5°
   * - Triangles
     - Non-manifold edge (always checked by the adjacency builder)
     - edge shared by more than two faces
   * - Tetrahedra
     - Inverted element
     - signed volume below 0
   * - Tetrahedra
     - Small volume
     - below 1 mm³ (1e-9 m³)
   * - Tetrahedra
     - Sliver, shape quality :math:`\eta` (1 for a regular tetrahedron)
     - below 0.01
   * - Tetrahedra
     - Non-manifold face
     - face shared by more than two tetrahedra

A warning looks like this (from a real asset):

.. code-block:: text

   UserWarning: Mesh quality warning [bag] (873 vertices, 1544 triangles):
     - 4 sliver triangle(s) with aspect ratio > 20 (worst: 18.9)
     - 1 triangle(s) with minimum angle < 5° (smallest: 4.0°)
   Consider remeshing the input geometry.

Remesh with an isotropic explicit remesher, for example MeshLab's
*Remeshing: Isotropic Explicit Remeshing*, with a target edge length that is
uniform over the asset and larger than the interaction distance. Then add the
mesh with ``validate_mesh=True`` again. Remeshing at a coarse target can still
leave slivers; the checker tells you.

.. _vbd-fail-small-dt:

A smaller time step made things worse
`````````````````````````````````````

**Symptom.** After halving ``dt`` the simulation is slower, and either nothing
improved or the object drifts and loses detail over long runs.

**Cause.** Two effects. First, a smaller ``dt`` costs more substeps per second,
each with collision detection; at a fixed compute budget it takes iterations
away from each substep. Second, positions are float32. A small ``dt`` gives a
small per-step update compared to the position value, so part of each update is
rounded away, and the loss accumulates over time.

**Fix.** Start at ``dt`` = 1/600 s with 10 iterations. Reduce ``dt`` while it
visibly helps, but not below about 1/3000 s. Beyond that, spend the budget on
iterations, or move the scene closer to the origin so that positions are small
numbers.

.. _vbd-fail-tunneling:

Self-contact tunneling
``````````````````````

**Symptom.** Cloth passes through itself, or one layer of a stack drops
through another, although self-contact is enabled.

**Cause.** The candidate buffers overflow, or a filter excludes the pair. Each
vertex stores at most ``particle_vertex_contact_buffer_size`` vertex-triangle
candidates (default 32) and each edge at most
``particle_edge_contact_buffer_size`` edge-edge candidates (default 64). Pairs
beyond the capacity are dropped silently, and a dropped pair has no force and
no truncation. The topological filter and the rest-shape exclusion radius also
remove pairs, by design.

**Fix.** Raise the two buffer sizes, and lower the gap if it is far above the
margin. The detector records the number of pairs found per vertex and per edge,
including the dropped ones, next to the buffer sizes. A count above the buffer
size is a measured overflow. The result struct is reachable as
``solver.trimesh_collision_info`` (fields ``vertex_colliding_triangles_count``,
``vertex_colliding_triangles_buffer_sizes``, ``edge_colliding_edges_count`` and
``edge_colliding_edges_buffer_sizes``); it is an experimental type and can
change. If a filter is the cause, lower
``particle_rest_shape_contact_exclusion_radius`` or check the external filter
maps you passed.

.. _vbd-empirical-table:

Empirical Parameter Combinations That Work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start from an example, not from material data. Stiffness values derived from
measured material properties (Young's modulus, Poisson ratio, thickness) are
usually far too high for VBD to converge at practical iteration counts. The
values below are the ones each shipped example runs with at Newton commit
cced8ced1; open the example file for the complete configuration. The
"resembles" column is a visual resemblance at the example's scale, not a
material measurement.

.. list-table::
   :header-rows: 1
   :widths: 14 30 22 14 20

   * - Scenario
     - Material values
     - Contact values
     - dt, iterations
     - Resembles
   * - Cloth drape
       (`cloth_hanging <https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_hanging.py>`__)
     - grid cells 0.1 m, 0.1 kg per particle; ``tri_ke`` 1e3, ``tri_ka`` 1e3,
       ``tri_kd`` 1e2, ``edge_ke`` 10, ``edge_kd`` 0
     - radius 0.05, margin 0.02, gap 0.01; ``soft_contact_ke`` 1e2, ``kd``
       1e2, ``mu`` 1
     - 1/600, 10
     - heavy cotton sheet at a coarse scale
   * - Cloth with a robot
       (`cloth_franka <https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_franka.py>`__,
       scene in centimetres)
     - density 0.02; ``tri_ke`` 1e4, ``tri_ka`` 1e4, ``tri_kd`` 1.5e-2,
       ``edge_ke`` 5, ``edge_kd`` 0.5
     - radius 0.8, pipeline gap 0.8, self margin 0.2, gap 0;
       ``soft_contact_ke`` 1e4, ``kd`` 10; robot shapes ``ke`` 5e4, ``kd``
       50, ``mu`` 1.5
     - 1/600, 5
     - cotton cloth handled by a gripper
   * - Stiff sheets
       (`cloth_poker_cards <https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_poker_cards.py>`__)
     - 1.8 g per card over 35 particles; ``tri_ke`` 1e4, ``tri_ka`` 1e4,
       ``tri_kd`` 1, ``edge_ke`` 1e2, ``edge_kd`` 1
     - radius 3 mm, margin 1 mm, gap 0.5 mm, pipeline gap 5 mm
     - 1/1200, 10
     - playing cards, cardboard
   * - Inextensible sheet
       (`cloth_stiff_material_hanging <https://github.com/newton-physics/newton/blob/main/newton/examples/vbd/example_cloth_stiff_material_hanging.py>`__)
     - density 0.2; ``tri_ke`` 1e8, ``tri_ka`` 1e8, ``tri_kd`` 0, ``edge_ke``
       10, ``edge_kd`` 0
     - margin 2 mm, gap 1.5 mm
     - 1/600, 10
     - a plastic membrane that drapes but does not stretch
   * - Soft-body beams
       (`softbody_hanging <https://github.com/newton-physics/newton/blob/main/newton/examples/softbody/example_softbody_hanging.py>`__)
     - cells 0.1 m, density 1e3; ``k_mu`` 1e5, ``k_lambda`` 1e5, ``k_damp``
       10 to 1e4
     - none
     - 1/600, 10
     - rubber to foam beams, depending on ``k_damp``
   * - Soft box with ribbon
       (`softbody_gift <https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_softbody_gift.py>`__)
     - box density 100, ``k_mu`` 1e5, ``k_lambda`` 1e5, ``k_damp`` 1; ribbon
       density 0.02, ``tri_ke`` 1e5, ``tri_ka`` 1e5, ``tri_kd`` 1,
       ``edge_ke`` 0.01, ``edge_kd`` 1e-4
     - margin 0.04, gap 0.02
     - 1/600, 15
     - foam box, thin ribbon
   * - Soft-body grasp
       (`softbody_franka <https://github.com/newton-physics/newton/blob/main/newton/examples/softbody/example_softbody_franka.py>`__)
     - density 100; ``k_mu`` 1e6, ``k_lambda`` 1e6, ``k_damp`` 1
     - radius 5 mm, margin 3 mm, gap 2 mm, pipeline gap 10 mm;
       ``soft_contact_ke`` 2e6, ``kd`` 0.2
     - 1/600, 5
     - a firm rubber block in a gripper
   * - Thin bags with contents
       (`vbd_soft_rigid_contact <https://github.com/newton-physics/newton/blob/main/newton/examples/vbd/example_vbd_soft_rigid_contact.py>`__)
     - density 0.08; ``tri_ke`` 1e5, ``tri_ka`` 1e5, ``tri_kd`` 1e2,
       ``edge_ke`` 50, ``edge_kd`` 0.5
     - radius 3 mm, margin 3 mm, gap 3 mm, pipeline gap 10 mm, shape margin
       5 mm; ``soft_contact_ke`` 1e5, ``kd`` 1e2, ``mu`` 0.8; shapes ``ke``
       1e5, ``kd`` 10, ``mu`` 0.5; filter threshold 3
     - 1/300, 15
     - plastic bags holding rigid objects
   * - Tarp under fast impacts
       (`vbd_dat_rigid_soft <https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_vbd_dat_rigid_soft.py>`__)
     - 1.6 m sheet, 40 cells, 0.6 kg; ``tri_ke`` 4e3, ``tri_ka`` 4e3,
       ``tri_kd`` 0.2, ``edge_ke`` 1
     - radius 6 mm, margin 12 mm, gap 40 mm, pipeline gap 60 mm; shapes
       ``ke`` 3e6, ``kd`` 1e3; ``soft_contact_ke`` 2e4, ``kd`` 1e-4, ``mu``
       0.6; rigid-soft DAT on
     - 1/240, 8
     - a tarp struck by thrown objects
   * - Gripper on a soft pad
       (`vbd_gripper_soft_grid <https://github.com/newton-physics/newton/blob/main/newton/examples/vbd/example_vbd_gripper_soft_grid.py>`__)
     - 0.1 kg pad; ``tri_ke`` 5e2, ``tri_kd`` 0.1
     - radius 10 mm, pipeline gap 10 mm; ``soft_contact_ke`` 1e3, ``kd`` 10,
       ``mu`` 1
     - 1/1200, 5
     - a foam pad pinched and lifted

Rules of Thumb
~~~~~~~~~~~~~~

- Use the lowest resolution that expresses the deformation you need. Raise it
  only when the degrees of freedom are not enough.
- Use a uniform triangulation with good element quality. Check it with
  ``validate_mesh=True``.
- Start at ``dt`` = 1/600 s and 10 iterations, and adjust from there. If you
  start from a shipped example, keep its values.
- Keep a copy of the last working configuration and change one parameter at a
  time, so you know what each change did and can go back. The
  :ref:`Simulation Tuning` page explains the one-knob rule.
- Start from the paradigm example closest to your task: ``cloth_hanging`` for
  cloth simulation, ``cloth_franka`` for cloth manipulation,
  ``softbody_hanging`` and ``softbody_franka`` for soft bodies,
  ``vbd_gripper_soft_grid`` for grippers.

.. _vbd-stiffer-object:

Getting a stiffer object
````````````````````````

Do not blindly crank up stiffness. Above about 1e6, expect slow convergence and
jitter under contact and fast motion; a static drape can still work at higher
values, as ``cloth_stiff_material_hanging`` shows at 1e8. If your stiffness is
already very high, raising it further is unlikely to help, and it creates more
stiffness-ratio artifacts. Look at these instead, in order:

1. **Resolution.** Is such a high resolution necessary? Reducing the resolution
   immediately improves convergence and so increases the effective stiffness.
2. **Stiffness ratio.** One stiffness far above the others impedes the overall
   convergence and makes some of the elasticity look less stiff. The bag sweep
   in :ref:`vbd-fail-membrane-only` shows membrane stiffening that only makes
   the bag wrinkle more.
3. **Time step and iterations.** Reducing ``dt`` always helps, but not below
   about 1/3000 s. More iterations also help. Before ``dt`` gets too small,
   reducing ``dt`` improves stiffness more than adding iterations.
