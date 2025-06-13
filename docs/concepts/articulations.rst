Articulations
=============

Articulations are a way to represent a collection of rigid bodies that are connected by joints.

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
   * - ``JOINT_PRISMATIC``
     - Prismatic (slider) joint with 1 linear degree of freedom
     - 1
     - 1
     - 1
   * - ``JOINT_REVOLUTE``
     - Revolute (hinge) joint with 1 angular degree of freedom
     - 1
     - 1
     - 1
   * - ``JOINT_BALL``
     - Ball (spherical) joint with quaternion state representation
     - 4
     - 3
     - 3
   * - ``JOINT_FIXED``
     - Fixed (static) joint with no degrees of freedom
     - 0
     - 0
     - 0
   * - ``JOINT_FREE``
     - Free (floating) joint with 6 degrees of freedom in velocity space
     - 7 (3D position + 4D quaternion)
     - 6 (see :ref:`Twist conventions in Newton <Twist conventions>`)
     - 0
   * - ``JOINT_COMPOUND``
     - Compound joint with 3 rotational degrees of freedom
     - 3
     - 3
     - 3
   * - ``JOINT_UNIVERSAL``
     - Universal joint with 2 rotational degrees of freedom
     - 2
     - 2
     - 2
   * - ``JOINT_DISTANCE``
     - Distance joint that keeps two bodies at a distance within its joint limits
     - 7
     - 6
     - 1
   * - ``JOINT_D6``
     - Generic D6 joint with up to 3 translational and 3 rotational degrees of freedom
     - up to 6
     - up to 6
     - up to 6

D6 joints are the most general joint type in Newton and can be used to represent any combination of translational and rotational degrees of freedom.
Prismatic, revolute, planar, universal, and compound joints can be seen as special cases of the D6 joint.

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

.. autofunction:: newton.eval_ik
