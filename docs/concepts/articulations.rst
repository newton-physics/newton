.. _Articulations:

Articulations
=============

Articulations are a way to represent a collection of rigid bodies that are connected by joints.

.. _Articulation parameterization:

Generalized and maximal coordinates
-----------------------------------

There are two types of parameterizations to describe the configuration of an articulation:
generalized coordinates and maximal coordinates.

Generalized (sometimes also called "reduced") coordinates describe the configuration of an articulation in terms of joint positions and velocities.
For example, a double-pendulum articulation (two links serially attached to world by revolute joints) has two generalized coordinates corresponding to the two joint angles (:attr:`newton.State.joint_q`) and joint velocities (:attr:`newton.State.joint_qd`).
See the table below for the number of generalized coordinates for each joint type.
Note that for a floating-base articulation (which is connected to the world by a free joint), the generalized coordinates include the maximal coordinates of the base link, i.e. the 3D position and 4D orientation of the base link.

Maximal coordinates describe the configuration of an articulation in terms of the body link positions and velocities.
Each rigid body's pose is represented by 7 parameters (3D position and XYZW quaternion) in :attr:`newton.State.body_q`,
and its velocity by 6 parameters (3D linear and 3D angular) in :attr:`newton.State.body_qd`.

To convert between these two representations we use forward and inverse kinematics:
forward kinematics (:func:`newton.eval_fk`) converts generalized coordinates to maximal coordinates, and inverse kinematics (:func:`newton.eval_ik`) converts maximal coordinates to generalized coordinates.

In Newton, we support both parameterizations and it is up to the solver which one to use to read and write the configuration.
For example, :class:`~newton.solvers.SolverMuJoCo` and :class:`~newton.solvers.SolverFeatherstone` use generalized coordinates, while :class:`~newton.solvers.SolverXPBD` and :class:`~newton.solvers.SolverSemiImplicit` use maximal coordinates.
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

.. code-block:: python

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

.. _Joint types:

Joint types
-----------

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 0

   * - Joint Type
     - Description
     - DOFs in ``joint_q``
     - DOFs in ``joint_qd``
     - DOFs in ``joint_axis``
   * - ``JointType.PRISMATIC``
     - Prismatic (slider) joint with 1 linear degree of freedom
     - 1
     - 1
     - 1
   * - ``JointType.REVOLUTE``
     - Revolute (hinge) joint with 1 angular degree of freedom
     - 1
     - 1
     - 1
   * - ``JointType.BALL``
     - Ball (spherical) joint with quaternion state representation
     - 4
     - 3
     - 3
   * - ``JointType.FIXED``
     - Fixed (static) joint with no degrees of freedom
     - 0
     - 0
     - 0
   * - ``JointType.FREE``
     - Free (floating) joint with 6 degrees of freedom in velocity space
     - 7 (3D position + 4D quaternion)
     - 6 (see :ref:`Twist conventions in Newton <Twist conventions>`)
     - 0
   * - ``JointType.DISTANCE``
     - Distance joint that keeps two bodies at a distance within its joint limits
     - 7
     - 6
     - 1
   * - ``JointType.D6``
     - Generic D6 joint with up to 3 translational and 3 rotational degrees of freedom
     - up to 6
     - up to 6
     - up to 6
   * - ``JointType.CABLE``
     - Cable joint with 1 linear (stretch/shear) and 1 angular (bend/twist) degree of freedom
     - 2
     - 2
     - 2

D6 joints are the most general joint type in Newton and can be used to represent any combination of translational and rotational degrees of freedom.
Prismatic, revolute, planar, and universal joints can be seen as special cases of the D6 joint.

Definition of ``joint_q``
^^^^^^^^^^^^^^^^^^^^^^^^^

The :attr:`newton.Model.joint_q` array stores the generalized joint positions for all joints in the model.
The positional dofs for each joint can be queried as follows:

.. code-block:: python

    q_start = Model.joint_q_start[joint_id]
    q_end = Model.joint_q_start[joint_id + 1]
    # now the positional dofs can be queried as follows:
    q0 = State.joint_q[q_start]
    q1 = State.joint_q[q_start + 1]
    ...

Definition of ``joint_qd``
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :attr:`newton.Model.joint_qd` array stores the generalized joint velocities for all joints in the model.
The generalized joint forces at :attr:`newton.Control.joint_f` are stored in the same order.

The velocity dofs for each joint can be queried as follows:

.. code-block:: python

    qd_start = Model.joint_qd_start[joint_id]
    qd_end = Model.joint_qd_start[joint_id + 1]
    # now the velocity dofs can be queried as follows:
    qd0 = State.joint_qd[qd_start]
    qd1 = State.joint_qd[qd_start + 1]
    ...
    # the generalized joint forces can be queried as follows:
    f0 = Control.joint_f[qd_start]
    f1 = Control.joint_f[qd_start + 1]
    ...

Axis-related quantities
^^^^^^^^^^^^^^^^^^^^^^^

Axis-related quantities include the definition of the joint axis in :attr:`newton.Model.joint_axis` and other properties
defined via :class:`newton.ModelBuilder.JointDofConfig`. The joint targets in :attr:`newton.Control.joint_target` are also
stored in the same per-axis order.

The :attr:`newton.Model.joint_dof_dim` array can be used to query the number of linear and angular dofs.
All axis-related quantities are stored in consecutive order for every joint. First, the linear dofs are stored, followed by the angular dofs.
The indexing of the linear and angular degrees of freedom for a joint at a given ``joint_index`` is as follows:

.. code-block:: python

    num_linear_dofs = Model.joint_dof_dim[joint_index, 0]
    num_angular_dofs = Model.joint_dof_dim[joint_index, 1]
    # the joint axes for each joint start at this index:
    axis_start = Model.joint_qd_start[joint_id]
    # the first linear 3D axis
    first_lin_axis = Model.joint_axis[axis_start]
    # the joint target for this linear dof
    first_lin_target = Control.joint_target[axis_start]
    # the joint limit of this linear dof
    first_lin_limit = Model.joint_limit_lower[axis_start]
    # the first angular 3D axis is therefore
    first_ang_axis = Model.joint_axis[axis_start + num_linear_dofs]
    # the joint target for this angular dof
    first_ang_target = Control.joint_target[axis_start + num_linear_dofs]
    # the joint limit of this angular dof
    first_ang_limit = Model.joint_limit_lower[axis_start + num_linear_dofs]


.. _FK-IK:

Forward / Inverse Kinematics
----------------------------

Articulated rigid-body mechanisms are kinematically described by the joints that connect the bodies as well as the
relative transform from the parent and child body to the respective anchor frames of the joint in the parent and child body:

.. image:: /_static/joint_transforms.png
   :width: 400
   :align: center

.. list-table:: Variable names in the kernels from :mod:`newton.core.articulation`
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

When orphan joints are detected during USD parsing (:meth:`~newton.ModelBuilder.add_usd`), Newton issues a warning with the affected joint paths.

**Validation and finalization**

By default, :meth:`~newton.ModelBuilder.finalize` validates that every joint belongs to an articulation and raises a :class:`ValueError` if orphan joints are found.
To proceed with orphan joints, skip this validation:

.. code-block:: python

   model = builder.finalize(skip_validation_joints=True)

**Solver compatibility**

Only maximal-coordinate solvers (:class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverSemiImplicit`) support orphan joints.
Generalized-coordinate solvers (:class:`~newton.solvers.SolverFeatherstone`, :class:`~newton.solvers.SolverMuJoCo`) require every joint to belong to an articulation.
