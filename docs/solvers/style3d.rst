.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Style3D
=======

:class:`~newton.solvers.SolverStyle3D` solves cloth with projective dynamics
and free rigid bodies with affine body dynamics. Cloth and rigid bodies can
share one implicit solve, including two-way particle-shape contact response.
The solver runs on CPU and CUDA. Joint constraints and edge or triangle soft
contacts are not supported.

Selecting dynamics
------------------

Choose which parts of the model to advance when constructing the solver:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``SolverStyle3D.Dynamics``
     - Behavior
   * - ``DEFORMABLE`` (default)
     - Advance deformable particles, including cloth. Rigid bodies are
       prescribed colliders and receive no contact response from this solve.
   * - ``AFFINE_BODY``
     - Advance free rigid bodies, including rigid-rigid contacts.
   * - ``ALL``
     - Advance particles and affine bodies together, including two-way
       particle-rigid contact response.
   * - ``NONE``
     - Advance neither particles nor bodies.

For cloth, call :meth:`~newton.solvers.SolverStyle3D.register_custom_attributes`
before adding cloth to the builder with
:func:`~newton.solvers.style3d.add_cloth_grid` or
:func:`~newton.solvers.style3d.add_cloth_mesh`. Rigid-only models do not require
Style3D custom attributes.

For example, create a falling rigid sphere and advance one substep:

.. code-block:: python

   import newton
   import warp as wp
   from newton.solvers import SolverStyle3D

   builder = newton.ModelBuilder()
   body = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
   builder.add_shape_sphere(body, radius=0.1)
   builder.add_ground_plane()
   model = builder.finalize()

   solver = SolverStyle3D(
       model,
       config=SolverStyle3D.Config(iterations=10, linear_iterations=10),
       dynamics=SolverStyle3D.Dynamics.AFFINE_BODY,
   )
   state_in, state_out = model.state(), model.state()
   control = model.control()
   collision = newton.CollisionPipeline(model)
   contacts = collision.contacts()

   state_in.clear_forces()
   collision.collide(state_in, contacts)
   solver.step(state_in, state_out, control, contacts, 1.0 / 120.0)

For a model containing both Style3D cloth and rigid bodies, select
``dynamics=SolverStyle3D.Dynamics.ALL``. Generate contacts before each substep
and swap the input and output states after stepping.

For ``Dynamics.ALL``, increase ``Config.linear_iterations`` as needed for
coupled particle-rigid contacts. The default 10 PCG iterations may be
insufficient during impacts and can cause artificial momentum gain. Check
convergence for the scene by increasing this count until the contact response
stabilizes.

Bodies with positive mass and :attr:`~newton.BodyFlags.DYNAMIC` are advanced.
With ``Dynamics.AFFINE_BODY`` or ``Dynamics.ALL``, kinematic body poses and
velocities are copied unchanged from ``state_in`` to ``state_out``, including
zero-mass kinematic bodies. Other zero-mass body outputs remain caller-owned.
``Dynamics.DEFORMABLE`` and ``Dynamics.NONE`` leave all body outputs untouched.
Changes to model particle and body flags are read at the start of each step.

.. _style3d-externally-driven-bodies:

Externally driven bodies
------------------------

With ``Dynamics.AFFINE_BODY`` or ``Dynamics.ALL``, a
:attr:`~newton.BodyFlags.KINEMATIC` body participates in contact prediction
regardless of mass. Its input pose and velocity pass through unchanged to
``state_out``; the solver does not advance its output pose from its velocity.
The external driver supplies the prescribed pose and velocity in ``state_in``
before each step, after any state-buffer swap.

For consistent contact prediction, the supplied velocity must make the predictor
reproduce the body's externally prescribed end-of-substep pose, within
numerical precision. This requirement applies to both translation and
rotation, using the same ``dt`` as :meth:`~newton.solvers.SolverStyle3D.step`.

The kinematic predictor uses only ``state_in.body_q`` and ``state_in.body_qd``.
It ignores gravity, applied forces and torques, and gyroscopic acceleration.
For input COM linear velocity :math:`v`, its predicted center of mass is

.. math::

   \hat c = c + v\,dt.

Rotation uses the input world-space angular velocity :math:`\omega` in a
normalized Euler quaternion update, without angular damping:

.. math::

   \hat q = \mathrm{normalize}\left(q + \frac{dt}{2}\,(\omega, 0)\,q\right).

The input velocities must match these integration rules when representing a
prescribed trajectory; they must not compensate for gravity or applied forces.
Dynamic bodies retain semi-implicit rigid-body prediction with gravity, forces,
torques, and the gyroscopic term.

Contacts use this predicted pose to compute separation, damping, and friction
against the externally driven body. This prediction is internal to the contact
solve; it does not replace the prescribed input pose copied to ``state_out``.
A mismatch between the prediction and the externally driven trajectory makes
the contact response inconsistent with the body's actual motion. Zero-mass bodies
without :attr:`~newton.BodyFlags.KINEMATIC` use their input pose as a fixed
contact boundary for the substep.

Affine rigidity
---------------

Each body uses its world-space center of mass and three affine basis vectors
as internal coordinates. The affine mass matrix uses the body mass and the
COM mass second moment :math:`S`, related to the COM inertia tensor :math:`I`
by

.. math::

   S = \tfrac{1}{2}\mathrm{tr}(I) I_3 - I,
   \qquad \mathrm{tr}(I) = 2\,\mathrm{tr}(S),

where :math:`I_3` is the identity matrix. Rigid poses and velocities are
reconstructed after the solve.

The affine matrix :math:`A` is kept close to a proper rotation by the penalty

.. math::

   E(A) = \frac{k}{2}\min_{R\in SO(3)}\lVert A-R\rVert_F^2,
   \qquad
   k = r\frac{\mathrm{tr}(S)}{3\,dt^2}
     = r\frac{\mathrm{tr}(I)}{6\,dt^2}.

Here :math:`r` is ``Config.affine_rigidity_ratio``, a dimensionless ratio
relative to the mean directional inertia. Its default is ``100``; zero
disables the penalty. :math:`dt` is the solver substep duration in seconds,
the tensors have units of :math:`\mathrm{kg}\,\mathrm{m}^2`, and :math:`k`
has units of joules.
Increasing the ratio resists affine deformation more strongly, but does not
increase contact stiffness or make the constraint exact at finite iterations.
The solve uses a positive-semidefinite normal-space tangent that is exact at
a rotation and approximate away from SO(3).

``Config.iterations`` controls nonlinear iterations and
``Config.linear_iterations`` controls PCG iterations within each nonlinear
iteration. Edit ``solver.config`` between steps to change these settings.
Captured CUDA graphs must be recaptured after configuration changes.

Contact parameters
------------------

With affine body dynamics enabled, contacts use a normal penalty,
closing-only damping, and Coulomb-limited viscous friction:

- Rigid-rigid contacts use the shapes' ``ke``, ``kd``, and ``kf`` values,
  combined by arithmetic mean, and ``mu`` combined by geometric mean.
  Positive per-contact stiffness and damping overrides take precedence;
  a positive per-contact friction value scales the combined coefficient.
- Particle-rigid contacts use :attr:`~newton.Model.soft_contact_ke`,
  :attr:`~newton.Model.soft_contact_kd`, and
  :attr:`~newton.Model.soft_contact_kf`. The Coulomb coefficient is the
  geometric mean of :attr:`~newton.Model.soft_contact_mu` and the shape's
  ``mu``.

The contact candidates and normals supplied through :class:`~newton.Contacts`
remain fixed within each solver step. Contact forces are reevaluated during
nonlinear iterations; the contact tangent remains fixed during each PCG
solve. Finite contact stiffness permits penetration. Tune the contact
material separately from ``affine_rigidity_ratio``.

Examples
--------

The ``basic_shapes``, ``basic_dzhanibekov``, ``balance_bird``,
``domino_spiral``, and ``pyramid`` examples accept ``--solver style3d``:

.. code-block:: console

   uv run -m newton.examples basic_shapes --solver style3d
   uv run -m newton.examples basic_dzhanibekov --solver style3d
   uv run -m newton.examples balance_bird --solver style3d
   uv run -m newton.examples domino_spiral --solver style3d
   uv run -m newton.examples pyramid --solver style3d

The coupled ``style3d_balls_dropped_into_bag`` example drops five rigid balls
into an open cloth bag suspended by two fixed handles at 60 FPS with two
substeps per frame:

.. code-block:: console

   uv run -m newton.examples style3d_balls_dropped_into_bag

Tetrahedral elasticity
---------------------

Style3D supports experimental co-rotational tetrahedral elasticity with a
volume-preserving projection. Build tetrahedra with
:meth:`~newton.ModelBuilder.add_tetrahedron`, :meth:`~newton.ModelBuilder.add_soft_grid`, or
:meth:`~newton.ModelBuilder.add_soft_mesh`; the solver consumes the standard
``tet_indices``, ``tet_poses``, and ``tet_materials`` model arrays. The existing surface collision path requires surface triangles and edges,
along with
:meth:`~newton.solvers.SolverStyle3D.register_custom_attributes` and the corresponding cloth
attributes. Models containing only tetrahedra without surface topology are
not supported by this integration. Disable surface stiffness when only volumetric elasticity is desired.

The elastic energy per element is
``V * mu * ||F - R||^2 + 1.5 * V * lambda * ||F - Q||^2``, where ``R`` is
a proper rotation and ``Q`` has determinant one. The volume projection uses
an approximate local minimizer and retains its previous feasible target when
that gives a smaller residual. This is a projective co-rotational model,
not the full stable Neo-Hookean constitutive law.

The volume projection retains history in the solver, outside
:class:`~newton.State`. After restoring particle positions and velocities,
call :meth:`solver.reset(state) <newton.solvers.SolverStyle3D.reset>` to clear
this history before stepping again. This restarts projection from identity,
matching a newly constructed solver at the restored state. It does not restore
the projection history of an arbitrary saved frame or guarantee exact
continuation of that frame's trajectory. Optional ``world_mask`` and ``flags``
arguments select which worlds and particle state changes reset the history.
The example browser's Reset button recreates the example and its solver.

The volume term uses the tetrahedron's material Lamé parameter ``lambda``
directly. Rest tetrahedra must
have positive orientation, and material coefficients must be finite and
nonnegative. The tetrahedral damping coefficient is currently unused.

USD-authored volumes use the existing deformable-volume importer and its
standard tetrahedral model arrays; no additional material schema is introduced.
Contact handling continues
to use surface triangles, edges, and particle contacts; tetrahedra do not
implicitly generate a collision surface.
