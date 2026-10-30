.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

VBD
===

:class:`~newton.solvers.SolverVBD` is a unified implicit solver based on
Vertex Block Descent (VBD): cloth, soft bodies, rigid bodies (joints,
contacts, rods), and their interactions are all solved in one VBD loop, with
joint and contact constraints enforced through an augmented-Lagrangian (ALM)
extension of the method. The compliant ALM constraint path is enabled
by default (``rigid_compliant_alm=True``). Pass ``rigid_compliant_alm=False``
explicitly to retain the deprecated legacy path during its migration period.

.. experimental::

   :class:`~newton.solvers.SolverVBD` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

This page is the VBD backend guide: when to choose the solver, how to set it
up correctly, and where to find runnable examples. For symptom-driven
diagnosis and parameter tuning, see :ref:`Tuning VBD` and the
:ref:`Simulation Tuning` landing page.

.. note::

   This guide is under active development. Each section below states its
   intended scope and links to existing references until the full content
   lands.

When to Choose VBD
------------------

*In development.* This section will position VBD against the other Newton
solvers for cloth, soft-body, rigid, and mixed scenes, and state its current
limitations.

Until then, start from the :ref:`Supported Features` matrix and the
:ref:`Joint feature support` tables in the :doc:`solver overview
</solvers/index>`. The underlying methods are described in the references
cited by :class:`~newton.solvers.SolverVBD`:

- Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block
  Descent. *ACM Trans. Graph.* 43, 4. https://doi.org/10.1145/3658179
- Chris Giles, Elie Diaz, and Cem Yuksel. 2025. Augmented Vertex Block
  Descent. *ACM Trans. Graph.* 44, 4. https://doi.org/10.1145/3731195
  (the deprecated legacy rigid constraint path)

.. _vbd-how-it-works:

How VBD Works
-------------

VBD (Vertex Block Descent) treats every time step as one minimization and
solves it one small block at a time. Two kinds of blocks exist. A **particle**
:math:`i` has 3 degrees of freedom, its position :math:`\mathbf{x}_i`, and a
mass :math:`m_i`. A **rigid body** :math:`b` has 6 degrees of freedom, its pose
made of a centre of mass :math:`\mathbf{c}_b` and a rotation
:math:`\mathbf{R}_b`, with a mass :math:`m_b` and an inertia tensor
:math:`\mathbf{I}_b`. The notation follows the VBD paper cited above:
:math:`h` is the time step and :math:`\mathbf{y}` the inertial prediction.

.. _vbd-step-objective:

Step objective
~~~~~~~~~~~~~~

Every time step is one minimization, solved one block at a time. A particle
:math:`i` has 3 degrees of freedom, its position :math:`\mathbf{x}_i`, and a
mass :math:`m_i`. A rigid body :math:`b` has 6 degrees of freedom, its pose
made of a centre of mass :math:`\mathbf{c}_b` and a rotation
:math:`\mathbf{R}_b`, with a mass :math:`m_b` and an inertia tensor
:math:`\mathbf{I}_b`. With :math:`h` the time step, the step finds the
positions and poses that minimize

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
(``solve_rigid_body``). The last sum is every energy term in the scene:
membrane, bending and volumetric elasticity, self-contact and rigid-soft
contact with friction and damping, attachments and springs, and the compliant
augmented-Lagrangian rows of joints and body-body contacts.

.. _vbd-how-solved:

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

.. _vbd-energy-terms:

The energy terms
~~~~~~~~~~~~~~~~

Each energy term depends on a few blocks and on model parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 20 32 28

   * - Energy term
     - Acts on
     - Model parameters
     - Kernel
   * - Membrane
     - the 3 particles of a triangle
     - ``tri_ke`` (:math:`\mu`), ``tri_ka`` (:math:`\lambda`), ``tri_kd``
     - ``evaluate_neo_hookean_membrane_force_hessian``
   * - Bending
     - the 4 particles of an interior edge
     - ``edge_ke``, ``edge_kd``
     - ``evaluate_dihedral_angle_based_bending_force_hessian``
   * - Volumetric
     - the 4 particles of a tetrahedron
     - ``k_mu``, ``k_lambda``, ``k_damp``
     - ``evaluate_volumetric_neo_hookean_force_and_hessian``
   * - Self-contact, with friction and damping
     - a vertex and a triangle, or two edges
     - ``soft_contact_ke``, ``soft_contact_kd``, ``soft_contact_mu``
     - ``evaluate_self_contact_force_norm``, ``compute_friction``, ``damp_collision``
   * - Rigid-soft contact, with friction and damping
     - a particle, edge or face and one body or static shape
     - ``soft_contact_ke``, ``soft_contact_kd``, ``soft_contact_mu``, averaged
       with the shape ``ke``, ``kd``, ``mu``; ``particle_radius`` and the shape
       ``margin``
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

**Elasticity** is three terms. The membrane term resists stretch and shear of a
triangle. With :math:`\mathbf{F}` the in-plane deformation gradient,
:math:`I_C = \lVert \mathbf{F} \rVert^2` and :math:`J_s` the area ratio, the
energy per unit rest area is the stable Neo-Hookean membrane energy

.. math::

   \psi_{\text{m}} = \frac{\mu}{2}\,(I_C - 2) + \frac{\lambda_{\text{NH}}}{2}\,(J_s - \alpha)^2,
   \qquad \alpha = 1 + \frac{\mu}{\lambda_{\text{NH}}},

with :math:`\mu = \texttt{tri\_ke}` and :math:`\lambda_{\text{NH}} =
\texttt{tri\_ka} + \texttt{tri\_ke}`; the triangle energy is :math:`A_0\,\psi_{\text{m}}`.
``tri_kd`` damps the rate of the metric :math:`\mathbf{F}^{\mathsf T}\mathbf{F}`,
so rigid rotations are not damped. The bending term resists a change of the
dihedral angle :math:`\theta` across an interior edge of rest length
:math:`L_0`:

.. math::

   E_{\text{b}} = \tfrac{1}{2}\,\texttt{edge\_ke}\;L_0\,(\theta - \theta_0)^2,

and ``edge_kd`` damps the angle rate with the force
:math:`\mathbf{f}_d = -\texttt{edge\_kd}\,L_0\,\dot\theta\,\partial\theta/\partial\mathbf{x}`,
where :math:`\dot\theta` is the change of :math:`\theta` over the step divided
by :math:`h`. The volumetric term is the stable
Neo-Hookean energy of a tetrahedron of rest volume :math:`V_0`,

.. math::

   \psi_{\text{v}} = \frac{\mu}{2}\,(I_C - 3) + \frac{\lambda_{\text{NH}}}{2}\,(J - \alpha)^2,
   \qquad J = \det \mathbf{F},

with :math:`\mu = \texttt{k\_mu}`, :math:`\lambda_{\text{NH}} = \texttt{k\_lambda} +
\texttt{k\_mu}`, the same :math:`\alpha`, energy :math:`V_0\,\psi_{\text{v}}`,
and ``k_damp`` damping the rate of the metric :math:`\mathbf{F}^{\mathsf T}\mathbf{F}`
like ``tri_kd``. In both Neo-Hookean terms the user values are
Lamé parameters; the kernels convert them to the Neo-Hookean parameters, which
is why :math:`\lambda_{\text{NH}}` carries the extra :math:`\mu`.

**Contact** is a penalty on the distance :math:`d` between the two primitives of
a pair, acting inside the interaction distance :math:`r`:

.. math::

   E_{\text{c}} = \frac{k}{2}\,(r - d)^2 \quad \text{for } \tfrac{r}{2} \le d < r,

with :math:`k = \texttt{soft\_contact\_ke}`. For self-contact the law switches
to a logarithmic barrier below :math:`d = r/2`, with gradient
:math:`-k\,(r/2)^2 / d`, and to a quadratic extension below 10 micrometres, so
two surfaces never cross. For rigid-soft contact :math:`r` is ``particle_radius``
plus the shape ``margin``, :math:`k` is the mean of ``soft_contact_ke`` and the
shape ``ke``, and the law stays quadratic unless
``rigid_soft_contact_use_log_barrier`` is set. **Friction** follows the IPC
form, :math:`\mathbf{f} = -\mu_f\, f_n\, \mathbf{T}\, f_1(\lVert\mathbf{u}\rVert)\,
\hat{\mathbf{u}}`, where :math:`f_n` is the normal force, :math:`\mathbf{u}` the
tangential displacement over the step, and :math:`f_1(x) = 2x/\epsilon_u - x^2/\epsilon_u^2`
ramps smoothly to one at the smoothing length :math:`\epsilon_u = \texttt{friction\_epsilon}\, h`
and stays one beyond it; the contact normal is held fixed inside the block
solve. **Contact damping** acts on the gap rate
:math:`\dot d`, only while the gap closes: :math:`\mathbf{f}_d = -\texttt{soft\_contact\_kd}\;
\dot d\,\mathbf{n}`. For rigid-soft contact the friction coefficient is the
geometric mean of ``soft_contact_mu`` and the shape ``mu``, and the damping the
mean of the two ``kd`` values.

.. _vbd-dat:

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
detected vertex-triangle or edge-edge pair a division plane is placed between
the two primitives, and each side keeps its own side of the plane
(``apply_planar_truncation_parallel_by_collision``). The solver-level budget
between detections is one half of the relaxation factor times the query radius,
one half per side. The same scheme can be enabled for rigid-soft contacts with
``rigid_soft_enable_dat``. :ref:`vbd-contact-handling` illustrates the planes.

The guarantee has a price: a vertex cannot move faster than about one motion
budget per detection cycle. The motion that the truncation removes is called
**DAT damping** here and in :ref:`Tuning VBD`. It is not related to any ``kd``
damping coefficient.

API Explanations
----------------

Construction and Stepping
~~~~~~~~~~~~~~~~~~~~~~~~~

*In development.* This section will cover the canonical setup and step loop:
coloring via :meth:`~newton.ModelBuilder.color` (required for particles and
for rigid bodies integrated by VBD), the recommended
``CollisionPipeline`` → ``Contacts`` → ``SolverVBD`` construction order, and
the CUDA graph capture constraints around contact state pre-allocation.

Until then, see the :class:`~newton.solvers.SolverVBD` API reference, which
documents the constructor and includes a minimal working simulation loop.

Model Inputs
~~~~~~~~~~~~

*In development.* This section will explain what the solver consumes from
:class:`~newton.ModelBuilder` for each system: triangle meshes for cloth,
tetrahedral meshes for soft bodies, rigid bodies with the supported joint
types (including rods), the default compliant ALM constraint mode
(``rigid_compliant_alm=True``) versus the deprecated legacy path, and the deprecated
legacy-path controls — ``model.vbd.joint_is_hard`` and
:meth:`~newton.solvers.SolverVBD.set_joint_constraint_mode` (deprecated since
1.6; no solver-mode effect under compliant ALM).

Scene-authored settings
^^^^^^^^^^^^^^^^^^^^^^^

Call :meth:`~newton.solvers.SolverVBD.register_custom_attributes` on the
:class:`~newton.ModelBuilder` before :meth:`~newton.ModelBuilder.add_usd` to read
the ``NewtonVbdSceneAPI`` settings from a USD ``PhysicsScene``. They are stored in
``model.vbd`` and used as :class:`~newton.solvers.SolverVBD` defaults:

.. list-table::
   :header-rows: 1

   * - USD attribute
     - ``model.vbd`` attribute
     - Constructor argument
     - Default
   * - ``newton:vbd:rigid:contactHistory``
     - ``rigid_contact_history``
     - ``rigid_contact_history``
     - ``False``
   * - ``newton:vbd:rigid:jointLinearStiffness``
     - ``rigid_joint_linear_ke``
     - ``rigid_joint_linear_ke``
     - ``1.0e5`` N/m
   * - ``newton:vbd:rigid:jointAngularStiffness``
     - ``rigid_joint_angular_ke``
     - ``rigid_joint_angular_ke``
     - ``1.0e5`` N·m/rad
   * - ``newton:vbd:rigid:jointLinearDamping``
     - ``rigid_joint_linear_kd``
     - ``rigid_joint_linear_kd``
     - ``0.0`` N·s/m
   * - ``newton:vbd:rigid:jointAngularDamping``
     - ``rigid_joint_angular_kd``
     - ``rigid_joint_angular_kd``
     - ``0.0`` N·m·s/rad

Precedence is explicit constructor argument, then ``model.vbd.*``, then the
built-in default. Scene-authored stiffness and damping must be finite and
non-negative, otherwise the constructor raises ``ValueError``. A scene-authored
``contactHistory`` is ignored when VBD does not integrate rigid bodies (no
bodies, or ``integrate_with_external_rigid_solver=True``). When VBD integrates
bodies and is given a collision pipeline, the pipeline needs ``contact_matching``
other than ``"disabled"``; pass ``rigid_contact_history=False`` to override the
scene value.
Scene values apply in both constraint modes: they are not skipped when the
deprecated ``rigid_compliant_alm=False`` path is selected, even though the
schema scopes them to the compliant ALM path.

Coupling Modes
~~~~~~~~~~~~~~

*In development.* This section will cover how deformables and rigid bodies
interact inside the solver (no special coupling setup is needed) and one-way
coupling with an external rigid-body solver
(``integrate_with_external_rigid_solver=True``).

Until then, see :doc:`Coupled Solvers </concepts/coupling>`.

Contact Handling
~~~~~~~~~~~~~~~~

*In development.* This section will explain the contact stack: particle
self-contact detection (vertex–triangle and edge–edge), body–particle
contacts, friction, and the penetration-free Divide and Truncate (DAT)
scheme that truncates particle motion against contact bounds. Coverage of
Offset Geometric Contact (OGC) will be added once it is available in
Newton.

The contact methods are described in:

- Anka H. Chen, Jerry Hsu, Youssef Ayman, and Miles Macklin. 2026. Divide
  and Truncate: A Penetration and Inversion Free Framework for Coupled
  Multi-physics System. In *ACM SIGGRAPH 2026 Conference Papers*.
  https://doi.org/10.1145/3799902.3811143
- Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem
  Yuksel. 2025. Offset Geometric Contact. *ACM Trans. Graph.* 44, 4.
  https://doi.org/10.1145/3731205

Key Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

*In development.* A lean orientation over the three constructor parameter
groups — common, ``particle_*``, and ``rigid_*`` — cross-linked to
:class:`~newton.solvers.SolverVBD` for defaults and to :ref:`Tuning VBD` for
effects and recommended values.

Examples
--------

*In development.* This section will walk through a small, representative set
of examples per system rather than the full list.

Runnable examples using :class:`~newton.solvers.SolverVBD` on GitHub:

- `Cloth examples <https://github.com/newton-physics/newton/tree/main/newton/examples/cloth>`_
- `Soft-body examples <https://github.com/newton-physics/newton/tree/main/newton/examples/softbody>`_
- `Cable examples <https://github.com/newton-physics/newton/tree/main/newton/examples/cable>`_
- `VBD-specific examples <https://github.com/newton-physics/newton/tree/main/newton/examples/vbd>`_
  (rigid and soft contacts, grippers, stiff materials)
- `Multiphysics examples <https://github.com/newton-physics/newton/tree/main/newton/examples/multiphysics>`_
  (coupling with MuJoCo, XPBD, and MPM)
