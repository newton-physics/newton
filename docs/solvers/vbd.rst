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

Start from the :ref:`Supported Features` matrix and the
:ref:`Joint feature support` tables in the :doc:`solver overview
</solvers/index>`. The underlying methods are described in the references
cited by :class:`~newton.solvers.SolverVBD`:

- Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block
  Descent. *ACM Trans. Graph.* 43, 4. https://doi.org/10.1145/3658179
- Chris Giles, Elie Diaz, and Cem Yuksel. 2025. Augmented Vertex Block
  Descent. *ACM Trans. Graph.* 44, 4. https://doi.org/10.1145/3731195
  (the deprecated legacy rigid constraint path)
- Anka H. Chen, Jerry Hsu, Youssef Ayman, and Miles Macklin. 2026. Divide
  and Truncate: A Penetration and Inversion Free Framework for Coupled
  Multi-physics System. In *ACM SIGGRAPH 2026 Conference Papers*.
  https://doi.org/10.1145/3799902.3811143 (penetration-free truncation, DAT)
- Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem
  Yuksel. 2025. Offset Geometric Contact. *ACM Trans. Graph.* 44, 4.
  https://doi.org/10.1145/3731205 (feature regions of full-surface mesh
  contact, OGC)

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
                 DAT: planar truncation per pair, then the radius cap
     _finalize_rigid_bodies     -> update_body_velocity
     _finalize_particles        -> update_velocity             v = (x_new - x_old) / dt

Two facts follow directly. Information moves one ring of neighbours per
iteration, because a block only sees its neighbours' current values. And an
unfinished solve is not discarded: whatever position the last iteration leaves
becomes the new velocity through :math:`\mathbf{v} = (\mathbf{x}_{\text{new}} -
\mathbf{x}_{\text{old}})/h`.

.. _vbd-rigid-joints:

Rigid bodies and joints
~~~~~~~~~~~~~~~~~~~~~~~

A rigid body :math:`b` is one block with 6 degrees of freedom: the position
:math:`\mathbf{c}_b` of its centre of mass and a rotation vector about it. With
the target pose :math:`(\mathbf{c}_b^*, \mathbf{R}_b^*)` from forward
integration and :math:`\boldsymbol{\theta}_b` the body-frame rotation vector
from the current rotation :math:`\mathbf{R}_b` to :math:`\mathbf{R}_b^*`, the
inertia term of the step objective contributes

.. math::

   \mathbf{f}^{\text{in}}_b = \frac{m_b}{h^2}\,(\mathbf{c}_b^* - \mathbf{c}_b),
   \qquad
   \boldsymbol{\tau}^{\text{in}}_b = \frac{1}{h^2}\,\mathbf{R}_b\,\mathbf{I}_b\,\boldsymbol{\theta}_b,
   \qquad
   \mathbf{H}^{\text{in}}_{ll} = \frac{m_b}{h^2}\,\mathbf{1},
   \qquad
   \mathbf{H}^{\text{in}}_{aa} = \frac{1}{h^2}\,\mathbf{R}_b\,\mathbf{I}_b\,\mathbf{R}_b^{\mathsf T}.

Every term that touches the body (joints, body-body contacts, rigid-soft
contacts, attachments, tendons) adds its force, its torque and the three
:math:`3 \times 3` blocks of the Hessian. The block then solves one
:math:`6 \times 6` system by :math:`LDL^{\mathsf T}` (``ldlt6_solve``) and
applies the increments (``solve_rigid_body``; the rotation increment acts in
world space):

.. math::

   \begin{bmatrix} \mathbf{H}_{ll} & \mathbf{H}_{al}^{\mathsf T} \\ \mathbf{H}_{al} & \mathbf{H}_{aa} \end{bmatrix}
   \begin{bmatrix} \Delta\mathbf{c} \\ \Delta\boldsymbol{\theta} \end{bmatrix}
   =
   \begin{bmatrix} \mathbf{f} \\ \boldsymbol{\tau} \end{bmatrix},
   \qquad
   \mathbf{c}_b \leftarrow \mathbf{c}_b + \Delta\mathbf{c},
   \qquad
   \mathbf{R}_b \leftarrow \exp(\Delta\boldsymbol{\theta})\,\mathbf{R}_b .

Joints and body-body contacts are constraint rows between two bodies, handled
as compliant augmented-Lagrangian (ALM) rows. A row has a residual :math:`C`,
a material stiffness :math:`k`, an internal metric :math:`\rho` that the solver
sets once per step, and a multiplier :math:`\lambda`. Its force along the row
and the dual update after every iteration are

.. math::

   f = k_{\text{eff}}\,C + s\,\lambda,
   \qquad
   k_{\text{eff}} = \frac{k\,\rho}{k + \rho},
   \qquad
   s = \frac{k}{k + \rho},
   \qquad
   \lambda \leftarrow s\,(\lambda + \rho\,C),

and :math:`k_{\text{eff}}` enters the Hessian blocks through the row's
Jacobian. A bilateral row keeps the full force. A unilateral row keeps only
the admissible sign: for contact :math:`f_n = \max(f, 0)` and
:math:`\lambda_n \leftarrow \max(\,\cdot\,, 0)`. Joint multipliers persist
across steps, scaled by ``rigid_avbd_gamma`` (0.999); body-body contacts
restart theirs at zero unless ``rigid_contact_history=True``.

*Joints* (``evaluate_joint_force_hessian``). The structural rows of a joint
between parent :math:`p` and child :math:`c` are the offset of the two joint
anchors and the rotation away from the rest relative orientation, each
projected onto the locked axes of the joint type
(``compute_kappa_and_jacobian``, ``build_joint_projectors``):

.. math::

   \mathbf{C}_{\text{lin}} = \mathbf{P}_{\text{lin}}\,(\mathbf{x}_c - \mathbf{x}_p),
   \qquad
   \mathbf{C}_{\text{ang}} = \mathbf{P}_{\text{ang}}\,\boldsymbol{\kappa}(\mathbf{R}_p, \mathbf{R}_c),
   \qquad
   \mathbf{P} = \mathbf{1} - \sum_{\text{free axes}} \mathbf{a}\,\mathbf{a}^{\mathsf T},

so BALL keeps the three linear rows; REVOLUTE the three linear rows and two
angular rows; PRISMATIC two linear rows and the three angular rows; FIXED all
six; D6 whatever its free axes leave; FREE adds no row. The material stiffness
is ``rigid_joint_linear_ke`` or ``rigid_joint_angular_ke``, and the metric is
:math:`\rho = \max(\text{support}, 9k)`, chosen so that :math:`k_{\text{eff}}
\ge 0.9\,k` in every local solve (``_bilateral_auto_rho``);
``rigid_joint_linear_kd`` and ``rigid_joint_angular_kd`` add absolute damping.
Drives are bilateral rows on the free axes of REVOLUTE, PRISMATIC and D6
joints, with the spring and the damper folded into one row
(``_spring_damper_step``, ``_evaluate_drive_axis``):

.. math::

   K = k_t + \frac{d_t}{h},
   \qquad
   e = \frac{k_t\,(q - q^*) + d_t\,(\dot q - \dot q^*)}{K},
   \qquad
   f_d = K_{\text{eff}}\,e + s\,\lambda_d,

with :math:`k_t = \mathtt{joint\_target\_ke}`, :math:`d_t =
\mathtt{joint\_target\_kd}`, and :math:`K_{\text{eff}}, s` formed from
:math:`K` and the axis metric as above. Limits are unilateral rows at the
nearer bound with ``joint_limit_ke`` and ``joint_limit_kd``; only forces into
the bound are kept, and the multiplier is projected onto its half-line
(``_evaluate_limit_axis``, ``_update_limit_lambda``). ROD joints use four rows,
stretch, shear, bend and twist, with ``joint_target_ke`` and
``joint_target_kd`` as their stiffness and damping. DISTANCE joints are not
supported.

*Body-body contact* is the unilateral row of the same family: :math:`C` is the
penetration of the two effective surfaces, :math:`k` the pair average of the
shape stiffness, :math:`\rho_n` the pair's inverse Delassus support over
:math:`h^2`, :math:`f_n = \max(k_{\text{eff}}\,C + s\,\lambda_n,\ 0)`, and
friction is a Coulomb projection carrying :math:`\boldsymbol{\lambda}_t`. The
complete law, with the material mixing and the damping, is in
:ref:`vbd-energy-terms`.

The deprecated legacy path (``rigid_compliant_alm=False``) replaces the
finite-material rows with ramped penalties and hard or soft slots.

.. _vbd-energy-terms:

The energy terms
~~~~~~~~~~~~~~~~

Each energy term depends on a few blocks and on model parameters:

.. list-table::
   :class: wrap-literals
   :header-rows: 1
   :widths: 22 22 28 28

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
     - a vertex and a triangle, or two edges, of one particle system
     - ``soft_contact_ke``, ``soft_contact_kd``, ``soft_contact_mu``,
       ``particle_self_contact_margin``
     - ``evaluate_self_contact_force_norm``, ``compute_friction``, ``damp_collision``
   * - Rigid-soft contact, with friction and damping
     - a particle, or a point on a soft edge or face, and one shape
     - ``soft_contact_ke``, ``soft_contact_kd``, ``soft_contact_mu`` averaged
       with the shape ``ke``, ``kd``, ``mu``; ``particle_radius`` and the shape
       ``margin``
     - ``_compute_body_particle_contact_force``,
       ``compute_projected_isotropic_friction``
   * - Body-body contact, with friction and damping
     - two shapes, on two rigid bodies or on a rigid body and a static shape
     - shape ``ke``, ``kd``, ``mu`` averaged over the pair; shape radius and
       ``margin``
     - ``evaluate_rigid_contact_from_collision``, ``_compliant_contact_dual_step``
   * - Attachments and springs
     - a particle and a body, or two particles
     - attachment and spring stiffness and damping
     - ``accumulate_body_particle_attachment_force_and_hessian``,
       ``accumulate_spring_force_and_hessian``
   * - Joints, drives and limits
     - one or two bodies
     - joint stiffness and damping
     - compliant augmented-Lagrangian rows, ``update_duals_joint``

**Elasticity** is three terms. The membrane term resists stretch and shear of a
triangle. With :math:`\mathbf{F}` the in-plane deformation gradient,
:math:`I_C = \lVert \mathbf{F} \rVert^2` and :math:`J_s` the area ratio, the
energy per unit rest area is the stable Neo-Hookean membrane energy

.. math::

   \psi_{\text{m}} = \frac{\mu}{2}\,(I_C - 2) + \frac{\lambda_{\text{NH}}}{2}\,(J_s - \alpha)^2,
   \qquad \alpha = 1 + \frac{\mu}{\lambda_{\text{NH}}},

with :math:`\mu = \mathtt{tri\_ke}` and :math:`\lambda_{\text{NH}} =
\mathtt{tri\_ka} + \mathtt{tri\_ke}`; the triangle energy is :math:`A_0\,\psi_{\text{m}}`.
``tri_kd`` damps the rate of the metric :math:`\mathbf{F}^{\mathsf T}\mathbf{F}`,
so rigid rotations are not damped. The bending term resists a change of the
dihedral angle :math:`\theta` across an interior edge of rest length
:math:`L_0`:

.. math::

   E_{\text{b}} = \tfrac{1}{2}\,\mathtt{edge\_ke}\;L_0\,(\theta - \theta_0)^2,

and ``edge_kd`` damps the angle rate with the force
:math:`\mathbf{f}_d = -\mathtt{edge\_kd}\,L_0\,\dot\theta\,\partial\theta/\partial\mathbf{x}`,
where :math:`\dot\theta` is the change of :math:`\theta` over the step divided
by :math:`h`. The volumetric term is the stable
Neo-Hookean energy of a tetrahedron of rest volume :math:`V_0`,

.. math::

   \psi_{\text{v}} = \frac{\mu}{2}\,(I_C - 3) + \frac{\lambda_{\text{NH}}}{2}\,(J - \alpha)^2,
   \qquad J = \det \mathbf{F},

with :math:`\mu = \mathtt{k\_mu}`, :math:`\lambda_{\text{NH}} = \mathtt{k\_lambda} +
\mathtt{k\_mu}`, the same :math:`\alpha`, energy :math:`V_0\,\psi_{\text{v}}`,
and ``k_damp`` damping the rate of the metric :math:`\mathbf{F}^{\mathsf T}\mathbf{F}`
like ``tri_kd``. In both Neo-Hookean terms the user values are
Lamé parameters; the kernels convert them to the Neo-Hookean parameters, which
is why :math:`\lambda_{\text{NH}}` carries the extra :math:`\mu`.

**Contact** is three different terms, one per contact kind. They share the
friction smoothing length :math:`\epsilon_u = \mathtt{friction\_epsilon}\, h`
and nothing else: each kind has its own distance, its own stiffness and its
own law.

*Self-contact* acts between a vertex and a triangle, or between two edges, of
one particle system (``accumulate_self_contact_force_and_hessian``). The
distance :math:`d` is the closest-point distance of the pair, the interaction
distance is :math:`r = \mathtt{particle\_self\_contact\_margin}` and the
stiffness is :math:`k = \mathtt{soft\_contact\_ke}`. The energy
(``evaluate_self_contact_force_norm``) is a quadratic penalty that turns into
a logarithmic barrier at half the interaction distance:

.. math::

   E_{\text{sc}}(d) =
   \begin{cases}
   \tfrac{k}{2}\,(r - d)^2 & \tfrac{r}{2} \le d < r,\\[3pt]
   -k\,(\tfrac{r}{2})^2 \ln d + \text{const} & d_{\min} < d < \tfrac{r}{2},\\[3pt]
   \text{second-order expansion of the barrier at } d_{\min} & d \le d_{\min},
   \end{cases}
   \qquad d_{\min} = \min(10\,\mu\text{m},\ \tfrac{r}{4}).

The gradient in the barrier region is :math:`-k\,(r/2)^2/d`, so the force
grows like :math:`1/d` as the gap closes; below :math:`d_{\min}` it continues
linearly, so the force stays finite. The pair force is distributed to the
four vertices by the barycentric weights of the two closest points. Friction
follows the IPC form (``compute_friction``). With :math:`f_n = -\partial
E_{\text{sc}}/\partial d` the normal force, :math:`\mathbf{u}` the tangential
relative displacement of the two closest points over the step and
:math:`\mathbf{T}` the tangent basis,

.. math::

   \mathbf{f}_{\text{fr}} = -\mu\, f_n\, \mathbf{T}\,
   \frac{f_1(\lVert\mathbf{u}\rVert)}{\lVert\mathbf{u}\rVert}\,\mathbf{u},
   \qquad
   f_1(x) = \begin{cases} 2x/\epsilon_u - x^2/\epsilon_u^2 & x < \epsilon_u,\\
   1 & x \ge \epsilon_u, \end{cases}

with :math:`\mu = \mathtt{soft\_contact\_mu}`. Unlike IPC, the contact normal
is held constant inside the block solve; the kernel notes that this improves
stability. Damping (``damp_collision``) acts on the gap rate :math:`\dot d =
\mathbf{n}\cdot\Delta\mathbf{x}_{\text{rel}}/h`, only while the gap closes:
:math:`\mathbf{f}_d = -\mathtt{soft\_contact\_kd}\;\dot d\,\mathbf{n}`. A
separating pair is not damped.

*Rigid-soft contact* acts between a soft contact point and one shape of a
rigid or static body. The point is a particle, or, with full-surface mesh
contact (``enable_rigid_soft_full_surface_contact``), a barycentric point on a
soft edge or face; the records come from the collision pipeline. The
distance is :math:`d = \mathbf{n}\cdot(\mathbf{x} - \mathbf{b})`, with
:math:`\mathbf{b}` the shape's contact point carried along with its body and
:math:`\mathbf{n}` the stored contact normal. The interaction distance is
:math:`r = \mathtt{particle\_radius}` plus the shape ``margin``; an edge or
face point uses the largest radius among its corners. The material is mixed
per contact from the two sides (``init_body_particle_contacts``):
:math:`k = \tfrac{1}{2}(\mathtt{soft\_contact\_ke} + \mathtt{ke}_{\text{shape}})`,
:math:`k_d = \tfrac{1}{2}(\mathtt{soft\_contact\_kd} + \mathtt{kd}_{\text{shape}})`
and :math:`\mu = \sqrt{\mathtt{soft\_contact\_mu}\;\mathtt{mu}_{\text{shape}}}`.
The normal law (``_evaluate_rigid_soft_contact_force_norm``) is the plain
quadratic penalty :math:`E = \tfrac{k}{2}(r - d)^2` for :math:`d < r`;
``rigid_soft_contact_use_log_barrier=True`` switches it to the self-contact
barrier law. Damping and friction have the same form as for self-contact,
written in world space (``_compute_body_particle_contact_force``,
``compute_projected_isotropic_friction``): the damping force
:math:`-k_d\,\dot d\,\mathbf{n}` acts only while the point approaches the
shape, and the friction force is :math:`-\mu\, f_n\,
f_1(\lVert\mathbf{u}_t\rVert)\,\hat{\mathbf{u}}_t`, with :math:`\mathbf{u}_t`
the tangential slip of the point relative to the shape surface over the step.
The force is distributed to the particles by their barycentric weights, and
the equal and opposite force with its torque acts on the body
(``accumulate_body_particle_contacts_per_body``). With full-surface mesh
contact, every detected edge and face record is kept for DAT, but only the
records that pass the feature-region test of Offset Geometric Contact produce
force (``mesh_contact_valid``).

*Body-body contact* acts between two shapes, on two rigid bodies or on a
rigid body and a static shape. It is a constraint row of the rigid solve, not
a penalty energy. The gap is measured between the two effective surfaces:
with :math:`\mathbf{p}_0, \mathbf{p}_1` the support points carried along with
their bodies and :math:`t_0, t_1` the effective thicknesses (shape radius plus
``margin``, stored by the collision pipeline as ``rigid_contact_margin0`` and
``rigid_contact_margin1``), the penetration is

.. math::

   C = -\big(\mathbf{n}\cdot(\mathbf{p}_1 - \mathbf{p}_0) - t_0 - t_1\big).

The material is the pair average of the shape values:
:math:`k = \tfrac{1}{2}(\mathtt{ke}_0 + \mathtt{ke}_1)`,
:math:`k_d = \tfrac{1}{2}(\mathtt{kd}_0 + \mathtt{kd}_1)` and
:math:`\mu = \sqrt{\mathtt{mu}_0\,\mathtt{mu}_1}`. On the default compliant
ALM path (``rigid_compliant_alm=True``) the normal force is

.. math::

   f_n = \max\!\big(k_{\text{eff}}\,C + s\,\lambda_n,\ 0\big),
   \qquad k_{\text{eff}} = \frac{k\,\rho_n}{k + \rho_n},
   \qquad s = \frac{k}{k + \rho_n},

where :math:`\lambda_n` is the multiplier of the row and :math:`\rho_n` is an
internal metric that the solver sets every step from the pair's inverse
Delassus support over :math:`h^2` plus structural stiffness
(``_contact_conditioning_scale``); the user does not set :math:`\rho_n`. The
force acts along :math:`\mathbf{n}` at the support points with the matching
torques, and the row adds :math:`k_{\text{eff}}\,\mathbf{n}\mathbf{n}^{\mathsf T}`
to the body Hessians (``evaluate_rigid_contact_from_collision``). After every
iteration the multiplier is advanced by the relaxed ascent
:math:`\lambda_n \leftarrow \max\big(s\,(\lambda_n + \rho_n C),\ 0\big)`
(``update_duals_body_body_contacts``, ``_compliant_contact_dual_step``), so a
resting contact carries its load in :math:`\lambda_n` instead of in
penetration. Friction is a Coulomb bound, not a smooth law: the tangential
force :math:`\rho_t\,\mathbf{u}_t + \boldsymbol{\lambda}_t`, with
:math:`\mathbf{u}_t` the tangential offset between the two anchored surface
points and :math:`\rho_t` the tangent metric, is projected onto the disk
:math:`\lVert\mathbf{f}_t\rVert \le \mu f_n` (``_project_coulomb_tangent``),
and :math:`\boldsymbol{\lambda}_t` is advanced and projected onto the disk of
radius :math:`\mu\lambda_n` in the same dual step. Damping is
:math:`-k_d\,(\mathbf{n}\cdot\mathbf{v}_{\text{rel}})\,\mathbf{n}` while the
bodies approach and the row carries load. The multipliers start from zero in
every step unless ``rigid_contact_history=True`` warm-starts them from the
matched contacts of the previous step. The legacy AVBD path
(``rigid_compliant_alm=False``, deprecated) ramps a penalty stiffness from
``rigid_contact_k_start`` by ``rigid_avbd_beta`` per iteration up to the pair
stiffness, and its soft mode uses the IPC friction law instead of the Coulomb
projection.

The underlying methods are described in the references cited by
:class:`~newton.solvers.SolverVBD`: VBD for the particle blocks, AVBD for the
rigid rows, Divide and Truncate (DAT) for penetration-free contact and Offset
Geometric Contact (OGC) for the feature regions of mesh contact. The two
contact papers are also listed under Contact Handling below.

.. _vbd-dat:

Penetration-free truncation (DAT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Self-contact and rigid-soft contact are penalty energies, so a large step
could jump through a surface between two detections. The Divide and Truncate
scheme (DAT) prevents that. It works on the displacement each vertex has
accumulated since the last detection and has two distinct parts
(``_penetration_free_truncation``).

**Planar truncation** acts per detected vertex-triangle or edge-edge pair
(``apply_planar_truncation_parallel_by_collision``). At the detection-time
positions a certified separating plane is found and placed between the two
primitives; it splits the gap in proportion to how far each side wants to move
toward the other, and each side keeps at least 5 percent of the gap
(``place_dat_division_plane``). Each vertex of the pair receives a factor
:math:`t \le 1` so that its straight trajectory stays on its own side of the
plane, scaled by ``dat_conservative_bound_relaxation``; motion away from the
plane or along it passes untouched (``planar_truncation_t``). A vertex in
several pairs keeps the smallest factor. A pair whose primitives already
intersect at detection time has no strict separator and is left unconstrained
in that pass.

**Radius truncation** acts on every particle (``apply_truncation_ts``). After
the planar factor is applied, the length of the accumulated displacement is
capped at one half of ``dat_conservative_bound_relaxation`` times the query
radius, isotropically. Two primitives farther apart than the query radius at
detection therefore cannot meet before the next detection.

**Rigid-soft DAT** (``rigid_soft_enable_dat``, off by default) applies the
same two parts to contacts between soft geometry and shapes
(``_rigid_penetration_free_truncation``). One thread per contact record builds
the division plane from the detection-time configuration and truncates both
sides against it (``apply_rigid_soft_truncation``). The soft vertices of the
record move on straight rays, as in self-contact. The vertices of the rigid
primitive move on curved trajectories because the body rotates, so their
crossing times are found by sampling and bisection, with an interval bound on
the signed-distance derivative for rows that start inside the plane's band;
``rigid_soft_dat_use_interval_arithmetic`` additionally certifies the sampled
prefix with interval arithmetic (experimental). Full-surface mesh contacts
name the exact rigid triangle, edge or vertex through
``Contacts.soft_contact_rigid_indices``; analytic SDF contacts and
penetration-recovery contacts truncate only their stored rigid surface point.
Each body then scales its whole pose update, translation and rotation about
the centre of mass, by the smallest factor of its contacts
(``apply_body_truncation_ts``). The radius truncation caps the motion of every
point of the body, bounded by the translation plus the rotation angle times
the body's bounding radius, at one half of the relaxation factor times the
smallest rigid-soft query radius (pipeline gap plus soft feature radius plus
shape margin); that budget is fixed at construction. The pass runs after the
forward step and after every body colour of every iteration, and the particle
pass re-enforces the same planes after the particles move. Kinematic bodies
move outside the solver and are not truncated. A pair for which no separating
plane can be certified freezes. :ref:`vbd-contact-handling` describes the
parts for tuning and illustrates the planes.

The guarantee has a price: a vertex cannot move faster than about one motion
budget per detection cycle. The motion that the truncation removes is called
**DAT damping** here and in :ref:`Tuning VBD`. It is not related to any ``kd``
damping coefficient.

API Explanations
----------------

Construction and Stepping
~~~~~~~~~~~~~~~~~~~~~~~~~

Colour the model before you finalize it. :meth:`~newton.ModelBuilder.color`
fills :attr:`~newton.Model.particle_color_groups` and
:attr:`~newton.Model.body_color_groups`; the solver raises at construction when
particles are present without particle colours, or when rigid bodies are
integrated by VBD without body colours. Pass ``include_bending=True`` when the
model has bending edges. Then build the collision pipeline, its contacts buffer
and the solver, in that order:

.. code-block:: python

   builder.color()
   model = builder.finalize()

   collision_pipeline = newton.CollisionPipeline(model)
   contacts = collision_pipeline.contacts()
   solver = newton.solvers.SolverVBD(model, iterations=10)

   state_in, state_out = model.state(), model.state()
   control = model.control()
   for _ in range(num_steps):
       collision_pipeline.collide(state_in, contacts)
       solver.step(state_in, state_out, control, contacts, dt)
       state_in, state_out = state_out, state_in

For rigid joints the solver captures the structural rest pose from
``model.body_q`` at construction, so the body transforms must match the joint
angles when the solver is built; ROD rest bend and twist come from
``model.joint_target_q``.

With a solver-owned pipeline, ``SolverVBD(model, collision_pipeline=...)``, the
solver allocates its own contacts buffer, seeds the pipeline's self-contact
configuration from the ``particle_self_contact_*`` arguments, runs rigid
detection itself on the schedule of ``collision_frequency_type``, and ``step``
takes ``contacts=None``. Rigid-soft DAT requires this form. The self-contact
slot works in both forms; :ref:`vbd-jitter-pipeline` shows the arguments that
put it on the ``ITERATIONS`` schedule.

For CUDA graph capture keep the same order, pipeline, contacts, solver, all
before capture. Body-body contact state is pre-sized from the pipeline's
published rigid contact maximum and body-particle contact state from the
particle-shape pair count; both grow from ``Contacts`` on the first step.
During capture that growth needs Warp's stream-ordered memory pool, otherwise
the solver raises and asks you to pre-size. ``rigid_contact_history=True``
allocates its warm-start buffers before capture in every case.

Start at ``dt`` = 1/600 s with 10 iterations and adjust following
:ref:`vbd-rules-of-thumb`. The :ref:`Tuning VBD` parameter table lists every
constructor argument that matters for particle systems, with its effect and
cost.

Model Inputs
~~~~~~~~~~~~

The solver reads three systems from the :class:`~newton.Model`:

- **Cloth**: particles with triangles and bending edges, from
  :meth:`~newton.ModelBuilder.add_cloth_mesh`,
  :meth:`~newton.ModelBuilder.add_cloth_grid`, or
  :meth:`~newton.ModelBuilder.add_triangle` and
  :meth:`~newton.ModelBuilder.add_edge`. Material: ``tri_ke``, ``tri_ka`` and
  ``tri_kd`` per triangle, ``edge_ke`` and ``edge_kd`` per edge, and
  ``particle_radius`` per particle. ``validate_mesh=True`` runs the
  element-quality checker.
- **Soft bodies**: particles with tetrahedra, from
  :meth:`~newton.ModelBuilder.add_soft_mesh`,
  :meth:`~newton.ModelBuilder.add_soft_grid` or
  :meth:`~newton.ModelBuilder.add_tetrahedron`. Material: ``k_mu``,
  ``k_lambda`` and ``k_damp`` per tetrahedron. The surface triangles take part
  in self-contact and rigid-soft contact.
- **Rigid bodies**: :meth:`~newton.ModelBuilder.add_body` with shapes, and
  joints of type BALL, FIXED, FREE, REVOLUTE, PRISMATIC, D6 and ROD
  (:meth:`~newton.ModelBuilder.add_joint_rod` for cables). DISTANCE joints are
  not supported. Shape materials ``ke``, ``kd``, ``mu`` and ``margin`` come
  from :class:`~newton.ModelBuilder.ShapeConfig`. Drives and limits read
  ``joint_target_ke``, ``joint_target_kd``, ``joint_limit_lower``,
  ``joint_limit_upper``, ``joint_limit_ke`` and ``joint_limit_kd``.

Springs (:meth:`~newton.ModelBuilder.add_spring`) and particle-body
attachments (:meth:`~newton.ModelBuilder.add_attachment_particle_body`) connect
the systems. The contact material between particles and shapes mixes
``soft_contact_ke``, ``soft_contact_kd`` and ``soft_contact_mu`` on the Model
with the shape values (:ref:`vbd-energy-terms`). The Model Parameters table of
:ref:`Tuning VBD` gives the effect and typical values of each of these.

Rigid constraint rows use the compliant ALM formulation by default
(``rigid_compliant_alm=True``): the authored stiffness is the physical
compliance and the solver picks the metric :math:`\rho` itself
(:ref:`vbd-rigid-joints`). The legacy AVBD path (``rigid_compliant_alm=False``)
is deprecated since 1.6, together with its controls: the per-joint
``vbd:joint_is_hard`` custom attribute (``model.vbd.joint_is_hard``, declared
by :meth:`~newton.solvers.SolverVBD.register_custom_attributes`) and
:meth:`~newton.solvers.SolverVBD.set_joint_constraint_mode`. Under compliant
ALM they have no solver-mode effect.

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

Inside one solver no coupling setup is needed. Rigid blocks and particle
blocks are solved in the same iteration loop, rigid blocks first, and
rigid-soft contacts, attachments and springs connect them as ordinary energy
terms (:ref:`vbd-how-solved`). Cloth between the fingers of a VBD-integrated
gripper is the plain case.

With ``integrate_with_external_rigid_solver=True`` the rigid bodies belong to
another solver. SolverVBD then skips the rigid solve, reads the body poses the
external solver wrote into the output state, and applies rigid-soft contact to
the particles only (one-way coupling). Attachments are not supported in this
mode. For two-way coupling through proxies, including contact forces reported
back to the source body, see :doc:`Coupled Solvers </concepts/coupling>`.

Contact Handling
~~~~~~~~~~~~~~~~

The contact stack has three layers.

**Detection.** Self-contact pairs come from the triangle-mesh self-contact
detector: vertex-triangle and edge-edge candidates within
``particle_self_contact_margin + particle_self_contact_gap``, stored in fixed
buffers of ``particle_vertex_contact_buffer_size`` and
``particle_edge_contact_buffer_size`` entries, and filtered by topological
distance (``particle_topological_contact_filter_threshold``) and rest-shape
distance (``particle_rest_shape_contact_exclusion_radius``). It runs on the
schedule of the ``SOFT_SELF_CONTACT`` collision slot, by default before and
after the inertial step of every substep. Rigid-soft records (particles, and
with ``enable_rigid_soft_full_surface_contact`` also soft edges and faces) and
body-body records come from the :class:`~newton.CollisionPipeline`, which
detects out to its ``soft_contact_gap`` beyond ``particle_radius`` plus the
shape ``margin`` and carries the shape thicknesses of body-body pairs. With a
solver-owned pipeline the rigid slot follows ``collision_frequency_type`` as
well.

**Forces.** Each contact kind has its own law with friction and damping;
self-contact, rigid-soft contact and body-body contact are given in
:ref:`vbd-energy-terms`. Full-surface mesh records produce force only when they
pass the feature-region test of Offset Geometric Contact
(``mesh_contact_valid``); every record is kept for truncation.

**Truncation.** Divide and Truncate keeps self-contact, and with
``rigid_soft_enable_dat`` also rigid-soft contact, penetration-free between
detections, through the planar and the radius truncation of :ref:`vbd-dat`.

How to set the distances, and what they cost, is on the tuning page:
:ref:`vbd-contact-handling` and :ref:`vbd-margin-gap-guide`. The contact
methods are described in:

- Anka H. Chen, Jerry Hsu, Youssef Ayman, and Miles Macklin. 2026. Divide
  and Truncate: A Penetration and Inversion Free Framework for Coupled
  Multi-physics System. In *ACM SIGGRAPH 2026 Conference Papers*.
  https://doi.org/10.1145/3799902.3811143
- Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem
  Yuksel. 2025. Offset Geometric Contact. *ACM Trans. Graph.* 44, 4.
  https://doi.org/10.1145/3731205

Key Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

The constructor has three groups. Common: ``iterations``, ``friction_epsilon``,
``dat_conservative_bound_relaxation``,
``integrate_with_external_rigid_solver``, and ``collision_pipeline`` with
``collision_frequency_type`` and ``collision_frequency``. ``particle_*``:
self-contact on or off, margin and gap, the candidate buffer sizes, the
topological and rest-shape filters, and the tile solve. ``rigid_*``: the
compliant ALM switch with its stabilization and retention factors, body-body
contact history, the per-body contact list capacities, the rigid-soft barrier
law and DAT, and the structural joint stiffness and damping. Defaults and exact
semantics are in :class:`~newton.solvers.SolverVBD`; effects, symptoms and
recommended values for particle systems are in the :ref:`Tuning VBD`
parameter table.

Examples
--------

:ref:`vbd-empirical-table` on the tuning page lists one shipped example per
scenario, from cloth drape to a gripper on a soft pad, with the material,
contact and time-step values each one runs with.

Runnable examples using :class:`~newton.solvers.SolverVBD` on GitHub:

- `Cloth examples <https://github.com/newton-physics/newton/tree/main/newton/examples/cloth>`_
- `Soft-body examples <https://github.com/newton-physics/newton/tree/main/newton/examples/softbody>`_
- `Cable examples <https://github.com/newton-physics/newton/tree/main/newton/examples/cable>`_
- `VBD-specific examples <https://github.com/newton-physics/newton/tree/main/newton/examples/vbd>`_
  (rigid and soft contacts, grippers, stiff materials)
- `Multiphysics examples <https://github.com/newton-physics/newton/tree/main/newton/examples/multiphysics>`_
  (coupling with MuJoCo, XPBD, and MPM)
