.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Articulations:

Articulations
=============

Articulations are a way to represent a collection of rigid bodies that are connected by joints.

.. _Articulation parameterization:

Generalized and maximal coordinates
-----------------------------------

There are two types of parameterizations to describe the configuration of an articulation:
generalized coordinates and maximal coordinates.

Generalized (sometimes also called "reduced") coordinates describe an articulation in terms of its joint positions and velocities.
For example, a double-pendulum articulation has two revolute joints, so its generalized state consists of two joint angles in :attr:`newton.State.joint_q` and two corresponding joint velocities in :attr:`newton.State.joint_qd`.
See the table below for the number of generalized coordinates for each joint type.
For a floating-base articulation (one connected to the world by a free joint), the generalized coordinates also include the base link pose: a 3D position and an XYZW quaternion.

Maximal coordinates describe the configuration of an articulation in terms of the body link positions and velocities.
Each rigid body's pose is represented by 7 parameters (3D position and XYZW quaternion) in :attr:`newton.State.body_q`,
and its velocity by 6 parameters (3D linear and 3D angular) in :attr:`newton.State.body_qd`.

To convert between these two representations, we use forward and inverse kinematics:
forward kinematics (:func:`newton.eval_fk`) converts generalized coordinates to maximal coordinates, and inverse kinematics (:func:`newton.eval_ik`) converts maximal coordinates to generalized coordinates.

Newton supports both parameterizations, and each solver chooses which one it treats as the primary articulation state representation.
For example, :class:`~newton.solvers.SolverMuJoCo` and :class:`~newton.solvers.SolverFeatherstone`
use generalized coordinates, while :class:`~newton.solvers.SolverXPBD`,
:class:`~newton.solvers.SolverSemiImplicit`, and :class:`~newton.solvers.SolverVBD`
use maximal coordinates.
Note that collision detection, e.g., via :meth:`newton.Model.collide` requires the maximal coordinates to be current in the state.

To showcase how an articulation state is initialized using reduced coordinates, let's consider an example where we create an articulation with a single revolute joint and initialize
its joint angle to 0.5 and joint velocity to 10.0:

.. testcode::

  builder = newton.ModelBuilder()
  # create an articulation with a single revolute joint
  body = builder.add_link()
  builder.add_shape_box(body)  # add a shape to the body to add some inertia
  joint = builder.add_joint_revolute(parent=-1, child=body, axis=wp.vec3(0.0, 0.0, 1.0))  # add a revolute joint to the body
  builder.add_articulation([joint])  # create articulation from the joint
  builder.joint_q[-1] = 0.5
  builder.joint_qd[-1] = 10.0

  model = builder.finalize()
  state = model.state()

  # The generalized coordinates have been initialized by the revolute joint:
  assert all(state.joint_q.numpy() == [0.5])
  assert all(state.joint_qd.numpy() == [10.0])

While the generalized coordinates have been initialized by the values we set through the :attr:`newton.ModelBuilder.joint_q` and :attr:`newton.ModelBuilder.joint_qd` definitions,
the body poses (maximal coordinates) are still initialized by the identity transform (since we did not provide a ``xform`` argument to the :meth:`newton.ModelBuilder.add_link` call, it defaults to the identity transform).
This is not a problem for generalized-coordinate solvers, as they do not use the body poses (maximal coordinates) to represent the state of the articulation but only the generalized coordinates.

In order to update the body poses (maximal coordinates), we need to use the forward kinematics function :func:`newton.eval_fk`:

.. testcode::

  newton.eval_fk(model, state.joint_q, state.joint_qd, state)
  
Now, the body poses (maximal coordinates) have been updated by the forward kinematics and a maximal-coordinate solver can simulate the scene starting from these initial conditions.
As mentioned above, this call is not needed for generalized-coordinate solvers.

When declaring an articulation using the :class:`~newton.ModelBuilder`, the rigid body poses (maximal coordinates :attr:`newton.State.body_q`) are initialized by the ``xform`` argument:

.. testcode::

  builder = newton.ModelBuilder()
  tf = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi))
  body = builder.add_body(xform=tf)
  builder.add_shape_box(body)  # add a shape to the body to add some inertia

  model = builder.finalize()
  state = model.state()

  # The body poses (maximal coordinates) are initialized by the xform argument:
  assert all(state.body_q.numpy()[0] == [*tf])

  # Note: add_body() automatically creates a free joint, so generalized coordinates exist:
  assert len(state.joint_q) == 7  # 7 DOF for a free joint (3 position + 4 quaternion)

In this setup, we have a body with a box shape that both maximal-coordinate and generalized-coordinate solvers can simulate.
Since :meth:`~newton.ModelBuilder.add_body` automatically adds a free joint, the body already has the necessary degrees of freedom in generalized coordinates (:attr:`newton.State.joint_q`).

.. testcode::

  builder = newton.ModelBuilder()
  tf = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi))
  body = builder.add_link(xform=tf)
  builder.add_shape_box(body)  # add a shape to the body to add some inertia
  joint = builder.add_joint_free(body)  # add a free joint to connect the body to the world
  builder.add_articulation([joint])  # create articulation from the joint
  # The free joint's coordinates (joint_q) are initialized by its child body's pose,
  # so we do not need to specify them here
  # builder.joint_q[-7:] = *tf

  model = builder.finalize()
  state = model.state()

  # The body poses (maximal coordinates) are initialized by the xform argument:
  assert all(state.body_q.numpy()[0] == [*tf])

  # Now, the generalized coordinates are initialized by the free joint:
  assert len(state.joint_q) == 7
  assert all(state.joint_q.numpy() == [*tf])

This scene can now be simulated by both maximal-coordinate and generalized-coordinate solvers.

.. _Kinematic links:

Kinematic links and bodies
--------------------------

Newton distinguishes three motion modes for rigid bodies:

**Static**
  Does not move. Typical examples are world-attached shapes or links attached to world with a fixed joint.

**Kinematic**
  Moves only from user-prescribed state updates. It can have joint DOFs (free, revolute, etc.), but external forces do not accelerate it.

**Dynamic**
  Moves from forces, constraints, and contacts during solver integration.

Kinematic bodies are created through the ``is_kinematic=True`` flag on :meth:`~newton.ModelBuilder.add_link`
or :meth:`~newton.ModelBuilder.add_body`. Only root links (joint parent ``-1``) may be kinematic.
Setting a non-root link to kinematic raises a :class:`ValueError` during articulation construction.

Common combinations
^^^^^^^^^^^^^^^^^^^

The following patterns are valid and commonly used:

1. **Kinematic free-base body**: ``add_body(is_kinematic=True)`` (free joint root).
2. **Kinematic articulated root**: root link is kinematic and attached to world with a non-fixed joint
   (for example revolute), with dynamic descendants.
3. **Static fixed-root body**: root link is kinematic and attached to world with a fixed joint.
   This has zero joint DOFs and behaves as static.

.. testcode:: articulation-kinematic-combinations

   builder = newton.ModelBuilder()

   # 1) Kinematic free-base body (add_body creates free joint + articulation)
   kinematic_free = builder.add_body(is_kinematic=True, mass=1.0)

   # 2) Kinematic revolute root with a dynamic child
   root = builder.add_link(is_kinematic=True, mass=1.0)
   child = builder.add_link(mass=1.0)
   j_root = builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Z)
   j_child = builder.add_joint_revolute(parent=root, child=child, axis=newton.Axis.Z)
   builder.add_articulation([j_root, j_child])

   # 3) Static fixed-root body (zero joint DOFs)
   static_root = builder.add_link(is_kinematic=True, mass=1.0)
   j_static = builder.add_joint_fixed(parent=-1, child=static_root)
   builder.add_articulation([j_static])

   model = builder.finalize()

.. list-table:: Static vs kinematic vs dynamic bodies/links
   :header-rows: 1
   :widths: 22 26 26 26

   * - Property
     - Static
     - Kinematic
     - Dynamic
   * - Typical definition
     - World-attached shape, or root link fixed to world
     - ``is_kinematic=True`` on a root link/body with free/revolute/etc. joint
     - Default link/body (no kinematic flag)
   * - Joint DOFs
     - 0 for fixed-root links
     - Joint-dependent (free/revolute/D6/etc.)
     - Joint-dependent
   * - Position/velocity state
     - Constant (not integrated)
     - User-prescribed ``q``/``qd`` (or ``body_q``/``body_qd`` depending on solver coordinates)
     - Integrated by solver from dynamics
   * - Response to applied force/torque
     - No acceleration
     - No acceleration (force-immune for own motion)
     - Accelerates according to dynamics
   * - Collision/contact participation
     - Yes (acts as obstacle/support)
     - Yes (can push dynamic bodies while following prescribed motion)
     - Yes
   * - Mass/inertia (see :ref:`Mass and Inertia`)
     - Not used for motion when fixed
     - Preserved for body properties and future dynamic switching
     - Fully used by dynamics
   * - Mass matrix / constraint role
     - No active DOFs when fixed to world
     - Solver-dependent infinite-mass approximation along kinematic DOFs
     - Standard articulated mass matrix
   * - Typical applications
     - Environment geometry, fixtures
     - Conveyors, robot bases on trajectories, scripted mechanism roots
     - Physically simulated robots and objects

Velocity consistency for prescribed motion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For prescribed motion, it is up to the user to keep position and velocity updates consistent across time.
In particular, ``qd`` should be consistent with the finite-differenced motion implied by ``q``.
For scalar coordinates, this is the familiar ``q_next = q + qd * dt`` relation; quaternion-based coordinates
(for example FREE/BALL joint) require manifold-consistent quaternion integration instead of direct addition.

When writing kinematic state values:

- For generalized-coordinate workflows, write :attr:`newton.State.joint_q` and :attr:`newton.State.joint_qd`,
  then call :func:`newton.eval_fk` so maximal coordinates (for collisions and body-space consumers) are current.
- For maximal-coordinate workflows, write :attr:`newton.State.body_q` and :attr:`newton.State.body_qd` directly.

Rigid-body solver behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^

The rigid-body solvers (:class:`~newton.solvers.SolverMuJoCo`,
:class:`~newton.solvers.SolverFeatherstone`, :class:`~newton.solvers.SolverXPBD`,
:class:`~newton.solvers.SolverSemiImplicit`, :class:`~newton.solvers.SolverVBD`)
support the same user-facing kinematic authoring model:

- Kinematic links keep their declared joint type (free/revolute/etc.).
- A kinematic root attached to world by a fixed joint remains fixed (zero DOFs).
- Kinematic links participate in collisions/contacts and can impart motion to dynamic bodies.
- Applied forces do not drive kinematic motion; motion is user-prescribed.

Implementation details differ by coordinate formulation:

- Generalized-coordinate solvers (:class:`~newton.solvers.SolverMuJoCo`,
  :class:`~newton.solvers.SolverFeatherstone`) treat kinematic motion through prescribed joint state.
- Maximal-coordinate solvers (:class:`~newton.solvers.SolverXPBD`,
  :class:`~newton.solvers.SolverSemiImplicit`, :class:`~newton.solvers.SolverVBD`)
  use prescribed body transforms/twists.
- Contact handling of kinematic bodies is not identical across the solvers. :class:`~newton.solvers.SolverXPBD`,
  :class:`~newton.solvers.SolverVBD`, :class:`~newton.solvers.SolverMuJoCo`, and
  :class:`~newton.solvers.SolverFeatherstone` treat kinematic bodies like
  infinite-mass colliders for contact response, while
  :class:`~newton.solvers.SolverSemiImplicit` currently preserves prescribed state but
  does not zero inverse mass/inertia inside its contact solver. Contacts against
  kinematic bodies can therefore be softer under SemiImplicit.

In :class:`~newton.solvers.SolverMuJoCo`, kinematic DOFs are regularized with a
large internal armature value; see :ref:`Kinematic Links and Fixed Roots <mujoco-kinematic-links-and-fixed-roots>` for details.

.. _Joint types:

Joint types
-----------

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 0

   * - Joint Type
     - Description
     - Coordinates in ``joint_q``
     - DOFs in ``joint_qd``
   * - ``JointType.PRISMATIC``
     - Prismatic (slider) joint with 1 linear degree of freedom
     - 1
     - 1
   * - ``JointType.REVOLUTE``
     - Revolute (hinge) joint with 1 angular degree of freedom
     - 1
     - 1
   * - ``JointType.BALL``
     - Ball (spherical) joint with quaternion state representation
     - 4
     - 3
   * - ``JointType.FIXED``
     - Fixed (static) joint with no degrees of freedom
     - 0
     - 0
   * - ``JointType.FREE``
     - Free (floating) joint with 6 degrees of freedom in velocity space
     - 7 (3D position + 4D quaternion)
     - 6 (see :ref:`Twist conventions in Newton <Twist conventions>`)
   * - ``JointType.DISTANCE``
     - Distance joint that keeps two bodies at a distance within its joint limits
     - 7
     - 6
   * - ``JointType.D6``
     - Generic D6 joint with up to 3 translational and 3 rotational degrees of freedom
     - up to 6
     - up to 6
   * - ``JointType.CABLE``
     - Cable joint with 1 linear (stretch/shear) and 1 angular (bend/twist) degree of freedom
     - 2
     - 2

D6 joints are the most general joint type in Newton and can be used to represent any combination of translational and rotational degrees of freedom.
Prismatic, revolute, planar, and universal joints can be seen as special cases of the D6 joint.

Definition of ``joint_q``
^^^^^^^^^^^^^^^^^^^^^^^^^

The :attr:`newton.Model.joint_q` array stores the default generalized joint positions
for all joints in the model and is used to initialize :attr:`newton.State.joint_q`.
Both arrays share the same per-joint layout.
For scalar-coordinate joints (for example this D6 joint), the positional coordinates can be queried as follows:

.. testsetup:: articulation-joint-layout

    builder = newton.ModelBuilder()
    body = builder.add_link()
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_d6(
        parent=-1,
        child=body,
        linear_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, limit_lower=-0.5, limit_upper=0.5)],
        angular_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, limit_lower=-1.0, limit_upper=1.0)],
    )
    builder.add_articulation([joint])

    model = builder.finalize()
    state = model.state()
    control = model.control()
    joint_id = 0

    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_q = state.joint_q.numpy()
    joint_qd = state.joint_qd.numpy()
    joint_dof_dim = model.joint_dof_dim.numpy()
    joint_axis = model.joint_axis.numpy()
    joint_limit_lower = model.joint_limit_lower.numpy()
    joint_target_pos = control.joint_target_pos.numpy()
    joint_f = control.joint_f.numpy()

.. testcode:: articulation-joint-layout

    q_start = joint_q_start[joint_id]
    coord_count = joint_dof_dim[joint_id, 0] + joint_dof_dim[joint_id, 1]
    # now the positional coordinates can be queried as follows:
    q = joint_q[q_start : q_start + coord_count]
    q0 = q[0]
    q1 = q[1]

Definition of ``joint_qd``
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :attr:`newton.Model.joint_qd` array stores the default generalized joint velocities
for all joints in the model and is used to initialize :attr:`newton.State.joint_qd`.
The generalized joint forces at :attr:`newton.Control.joint_f` use the same DOF order.

Several other arrays also use this same DOF-ordered layout, indexed from
:attr:`newton.Model.joint_qd_start` rather than :attr:`newton.Model.joint_q_start`.
This includes :attr:`newton.Model.joint_axis`, joint limits and other per-DOF
properties defined via :class:`newton.ModelBuilder.JointDofConfig`, and the
position targets at :attr:`newton.Control.joint_target_pos`.

For every joint, these per-DOF arrays are stored consecutively, with linear DOFs
first and angular DOFs second. Use :attr:`newton.Model.joint_dof_dim` to query
how many of each a joint has.

The velocity DOFs for each joint can be queried as follows:

.. testcode:: articulation-joint-layout

    qd_start = joint_qd_start[joint_id]
    dof_count = joint_dof_dim[joint_id, 0] + joint_dof_dim[joint_id, 1]
    # now the velocity DOFs can be queried as follows:
    qd = joint_qd[qd_start : qd_start + dof_count]
    qd0 = qd[0]
    qd1 = qd[1]
    # the generalized joint forces can be queried as follows:
    f = joint_f[qd_start : qd_start + dof_count]
    f0 = f[0]
    f1 = f[1]

The same start index can be used to query other per-DOF arrays for that joint:

.. testcode:: articulation-joint-layout

    num_linear_dofs = joint_dof_dim[joint_id, 0]
    num_angular_dofs = joint_dof_dim[joint_id, 1]
    # all per-DOF arrays for this joint start at this index:
    dof_start = joint_qd_start[joint_id]
    # the axis vector for the first linear DOF
    first_lin_axis = joint_axis[dof_start]
    # the position target for this linear DOF
    first_lin_target = joint_target_pos[dof_start]
    # the joint limit of this linear DOF
    first_lin_limit = joint_limit_lower[dof_start]
    # the axis vector for the first angular DOF comes after all linear DOFs
    first_ang_axis = joint_axis[dof_start + num_linear_dofs]
    # the position target for this angular DOF
    first_ang_target = joint_target_pos[dof_start + num_linear_dofs]
    # the joint limit of this angular DOF
    first_ang_limit = joint_limit_lower[dof_start + num_linear_dofs]

    assert (num_linear_dofs, num_angular_dofs) == (1, 1)
    assert np.allclose(first_lin_axis, [1.0, 0.0, 0.0])
    assert np.allclose(first_ang_axis, [0.0, 0.0, 1.0])
    assert np.allclose([first_lin_limit, first_ang_limit], [-0.5, -1.0])


Common articulation workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Center ``joint_q`` between joint limits with Warp kernels
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Joint limits are stored in DOF order (``joint_qd`` layout), while ``joint_q`` stores generalized
joint coordinates (which may include quaternion coordinates for free/ball joints).
The pattern below sets each scalar coordinate to the midpoint between its lower and upper limits.

A robust pattern is:

1. Loop over joints.
2. Use ``Model.joint_qd_start`` to find the first DOF index for each joint.
3. Use ``Model.joint_dof_dim`` to get the number of linear and angular DOFs for that joint.
4. Use ``Model.joint_q_start`` to find where that joint starts in ``State.joint_q``.
5. Center only scalar coordinates (for example, revolute/prismatic axes) and skip quaternion joints.

.. testsetup:: articulation-center-joint-q

    builder = newton.ModelBuilder()
    parent = builder.add_link()
    child = builder.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(parent, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(child, hx=0.1, hy=0.1, hz=0.1)
    j0 = builder.add_joint_revolute(
        parent=-1,
        child=parent,
        axis=wp.vec3(0.0, 0.0, 1.0),
        limit_lower=-1.0,
        limit_upper=1.0,
    )
    j1 = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        limit_lower=0.0,
        limit_upper=2.0,
    )
    builder.add_articulation([j0, j1])

    model = builder.finalize()
    state = model.state()

.. testcode:: articulation-center-joint-q

    @wp.kernel
    def center_joint_q_from_limits(
        joint_q_start: wp.array(dtype=wp.int32),
        joint_qd_start: wp.array(dtype=wp.int32),
        joint_dof_dim: wp.array2d(dtype=wp.int32),
        joint_type: wp.array(dtype=wp.int32),
        joint_limit_lower: wp.array(dtype=float),
        joint_limit_upper: wp.array(dtype=float),
        joint_q: wp.array(dtype=float),
    ):
        joint_id = wp.tid()

        # First DOF index for this joint in qd-order arrays (limits/axes/forces)
        qd_begin = joint_qd_start[joint_id]
        dof_count = joint_dof_dim[joint_id, 0] + joint_dof_dim[joint_id, 1]

        # Start index for this joint in generalized coordinates q
        q_begin = joint_q_start[joint_id]

        # Skip free/ball joints because their q entries include quaternion coordinates.
        jt = joint_type[joint_id]
        if (
            jt == newton.JointType.FREE
            or jt == newton.JointType.BALL
            or jt == newton.JointType.DISTANCE
        ):
            return

        # For scalar joints, q coordinates align with this joint's total DOF count.
        for local_dof in range(dof_count):
            qd_idx = qd_begin + local_dof
            q_idx = q_begin + local_dof

            lower = joint_limit_lower[qd_idx]
            upper = joint_limit_upper[qd_idx]
            if wp.isfinite(lower) and wp.isfinite(upper):
                joint_q[q_idx] = 0.5 * (lower + upper)


    # Launch over all joints in the model
    wp.launch(
        kernel=center_joint_q_from_limits,
        dim=model.joint_count,
        inputs=[
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_dof_dim,
            model.joint_type,
            model.joint_limit_lower,
            model.joint_limit_upper,
            state.joint_q,
        ],
    )

    # Recompute transforms after editing generalized coordinates
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

ArticulationView: selection interface for RL and batched control
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

:class:`newton.selection.ArticulationView` is the high-level interface for selecting a subset
of articulations and accessing their joints/links/DOFs with stable tensor shapes. This is
especially useful in RL pipelines where the same observation/action logic is applied to many
parallel environments.

Construct a view by matching articulation keys with a pattern and optional filters:

.. testsetup:: articulation-view

    builder = newton.ModelBuilder()
    for i in range(2):
        root = builder.add_link(
            xform=wp.transform(wp.vec3(float(i) * 2.0, 0.0, 0.0), wp.quat_identity())
        )
        tip = builder.add_link(
            xform=wp.transform(wp.vec3(float(i) * 2.0 + 1.0, 0.0, 0.0), wp.quat_identity())
        )
        builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(tip, hx=0.1, hy=0.1, hz=0.1)
        j_root = builder.add_joint_free(parent=-1, child=root)
        j_tip = builder.add_joint_revolute(
            parent=root,
            child=tip,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
        builder.add_articulation([j_root, j_tip], label=f"robot_{i}")

    model = builder.finalize()
    state = model.state()

.. testcode:: articulation-view

    # select all articulations whose key starts with "robot"
    view = newton.selection.ArticulationView(model, pattern="robot*")
    assert view.count == 2

    # select only scalar-joint articulations (exclude quaternion-root joint types)
    scalar_view = newton.selection.ArticulationView(
        model,
        pattern="robot*",
        include_joint_types=[newton.JointType.PRISMATIC, newton.JointType.REVOLUTE],
        exclude_joint_types=[newton.JointType.FREE, newton.JointType.BALL],
    )
    assert scalar_view.get_dof_positions(state).shape == (1, 2, 1)

Use views to read/write batched state slices (joint positions/velocities, root transforms,
link transforms) without manual index bookkeeping.

Move articulations in world space
"""""""""""""""""""""""""""""""""

Use :meth:`newton.selection.ArticulationView.set_root_transforms` to move selected articulations:

.. testcode:: articulation-view

    view = newton.selection.ArticulationView(model, pattern="robot*")
    root_tf = view.get_root_transforms(state).numpy()

    # shift +0.2 m along world x for all selected articulations
    root_tf[..., 0] += 0.2
    view.set_root_transforms(state, root_tf)

    # recompute link transforms from generalized coordinates
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    assert np.allclose(view.get_root_transforms(state).numpy()[0, :, 0], [0.2, 2.2])

For floating-base articulations (root joint type ``FREE`` or ``DISTANCE``), this updates
the root coordinates in ``joint_q``.
For non-floating-base articulations (for example ``FIXED`` or a world-attached
``REVOLUTE`` root), ``set_root_transforms()`` moves the articulation by writing
``Model.joint_X_p`` because there is no root pose stored in state coordinates.

Use ``ArticulationView`` to inspect and modify selected articulations
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

``ArticulationView`` provides stable, per-articulation access to links, joints, DOFs, and attributes:

.. testcode:: articulation-view

    view = newton.selection.ArticulationView(model, pattern="robot*")
    scalar_view = newton.selection.ArticulationView(
        model,
        pattern="robot*",
        include_joint_types=[newton.JointType.PRISMATIC, newton.JointType.REVOLUTE],
        exclude_joint_types=[newton.JointType.FREE, newton.JointType.BALL],
    )

    # inspect
    q = scalar_view.get_dof_positions(state)     # shape [world_count, articulation_count, dof_count]
    qd = scalar_view.get_dof_velocities(state)   # shape [world_count, articulation_count, dof_count]
    link_q = view.get_link_transforms(state)     # shape [world_count, articulation_count, link_count]
    assert q.shape == (1, 2, 1)
    assert qd.shape == (1, 2, 1)
    assert link_q.shape == (1, 2, 2)

    # edit selected articulation values in-place
    q_np = q.numpy()
    q_np[..., 0] = 0.0
    scalar_view.set_dof_positions(state, q_np)
    assert np.allclose(scalar_view.get_dof_positions(state).numpy()[0, :, 0], 0.0)

    # if model attributes are edited through the view, notify the solver afterwards
    # solver.notify_model_changed()


.. _FK-IK:

Forward / Inverse Kinematics
----------------------------

Articulated rigid-body mechanisms are kinematically described by the joints that connect the bodies as well as the
relative transform from the parent and child body to the respective anchor frames of the joint in the parent and child body:

.. image:: /_static/joint_transforms.png
   :width: 400
   :align: center

.. list-table:: Variable names in the articulation kernels
   :widths: 10 90
   :header-rows: 1

   * - Symbol
     - Description
   * - x_wp
     - World transform of the parent body (stored at :attr:`State.body_q`)
   * - x_wc
     - World transform of the child body (stored at :attr:`State.body_q`)
   * - x_pj
     - Transform from the parent body to the joint parent anchor frame (defined by :attr:`Model.joint_X_p`)
   * - x_cj
     - Transform from the child body to the joint child anchor frame (defined by :attr:`Model.joint_X_c`)
   * - x_j
     - Joint transform from the joint parent anchor frame to the joint child anchor frame

In the forward kinematics, the joint transform is determined by the joint coordinates (generalized joint positions :attr:`State.joint_q` and velocities :attr:`State.joint_qd`).
Given the parent body's world transform :math:`x_{wp}` and the joint transform :math:`x_{j}`, the child body's world transform :math:`x_{wc}` is computed as:

.. math::
   x_{wc} = x_{wp} \cdot x_{pj} \cdot x_{j} \cdot x_{cj}^{-1}.


.. autofunction:: newton.eval_fk
   :noindex:

.. autofunction:: newton.eval_ik
   :noindex:


.. _Orphan joints:

Orphan joints
-------------

An **orphan joint** is a joint that is not part of any articulation. This situation can arise when:

* The USD asset does not define a ``PhysicsArticulationRootAPI`` on any prim, so no articulations are discovered during parsing.
* A joint connects two bodies that are not under any ``PhysicsArticulationRootAPI`` prim, even though other articulations exist in the scene.

When orphan joints are detected during USD parsing (:meth:`~newton.ModelBuilder.add_usd`), Newton issues a warning that lists the affected joint paths.

**Validation and finalization**

By default, :meth:`~newton.ModelBuilder.finalize` validates that every joint belongs to an articulation and raises a :class:`ValueError` if orphan joints are found.
To proceed with orphan joints, skip this validation:

.. testsetup:: articulation-orphan-joints

   builder = newton.ModelBuilder()
   body = builder.add_link()
   builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
   builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)

.. testcode:: articulation-orphan-joints

   model = builder.finalize(skip_validation_joints=True)

**Solver compatibility**

Only maximal-coordinate solvers (:class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverSemiImplicit`) support orphan joints.
Generalized-coordinate solvers (:class:`~newton.solvers.SolverFeatherstone`, :class:`~newton.solvers.SolverMuJoCo`) require every joint to belong to an articulation.
