``warp.sim`` Migration Guide
============================

This guide is designed for users seeking to migrate their applications from ``warp.sim`` to Newton.


Solvers
-------

+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| **warp.sim**                                                                 | **Newton**                                                                          |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
|:class:`warp.sim.FeatherstoneIntegrator`                                      |:class:`newton.solvers.FeatherstoneSolver`                                           |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
|:class:`warp.sim.SemiImplicitIntegrator`                                      |:class:`newton.solvers.SemiImplicitSolver`                                           |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
|:class:`warp.sim.VBDIntegrator`                                               |:class:`newton.solvers.VBDSolver`                                                    |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
|:class:`warp.sim.XPBDIntegrator`                                              |:class:`newton.solvers.XPBDSolver`                                                   |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| ``integrator.simulate(self.model, self.state0, self.state1, self.dt, None)`` | ``solver.step(self.state0, self.state1, self.control, None, self.dt)``              |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+

Importers
---------

+-----------------------------------------------+----------------------------------------------------+
| **warp.sim**                                  | **Newton**                                         |
+-----------------------------------------------+----------------------------------------------------+
|:func:`warp.sim.parse_urdf`                    |:func:`newton.utils.parse_urdf`                     |
+-----------------------------------------------+----------------------------------------------------+
|:func:`warp.sim.parse_mjcf`                    |:func:`newton.utils.parse_mjcf`                     |
+-----------------------------------------------+----------------------------------------------------+
|:func:`warp.sim.parse_usd`                     |:func:`newton.utils.parse_usd`                      |
+-----------------------------------------------+----------------------------------------------------+
|:func:`warp.sim.resolve_usd_from_url`          |:func:`newton.utils.import_usd.resolve_usd_from_url`|
+-----------------------------------------------+----------------------------------------------------+

The joint-specific arguments to the importers have been removed.
Instead, you can set the default joint properties on a :class:`newton.ModelBuilder` instance in the :attr:`newton.ModelBuilder.default_joint_cfg` attribute.
For example, ``limit_lower`` is now defined using ``builder.default_joint_cfg.limit_lower``, where ``builder`` is an instance of :class:`newton.ModelBuilder`.

Similarly, the shape contact parameters have been removed from the importers.
Instead, you can set the default contact parameters on a :class:`newton.ModelBuilder` instance in the :attr:`newton.ModelBuilder.default_shape_cfg` object before loading the asset.
For example, ``ke`` is now defined using ``builder.default_shape_cfg.ke``, where ``builder`` is an instance of :class:`newton.ModelBuilder`.

The MJCF and URDF importers both have an ``up_axis`` argument that defaults to +Z.
All importers will rotate the asset now to match the builder's ``up_axis`` (instead of overwriting the ``up_axis`` in the builder, as was the case previously for the USD importer).

The MJCF importer from Warp sim only uses the ``geom_density`` defined in the MJCF for sphere and box shapes but ignores these definitions for other shape types (which will receive the default density specified by the ``density`` argument to ``wp.sim.parse_mjcf``). The Newton MJCF importer now considers the ``geom_density`` for all shape types. This change may yield to different simulation results and may require tuning contact and other simulation parameters to achieve similar results in Newton compared to Warp sim.


``Model``
---------

:attr:`newton.ShapeGeometry.is_solid` now is of dtype ``bool`` instead of ``wp.uint8``.

Forward and Inverse Kinematics
------------------------------

The signatures of the :func:`newton.eval_fk` and :func:`newton.eval_ik` functions have been slightly modified to make the mask argument optional:

+--------------------------------------------------------+------------------------------------------------------------------------+
| **warp.sim**                                           | **Newton**                                                             |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``eval_fk(model, joint_q, joint_qd, mask, state)``     | ``eval_fk(model, joint_q, joint_qd, state, mask=None)``                |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``eval_ik(model, state, joint_q, joint_qd)``           | ``eval_ik(model, state, joint_q, joint_qd, mask=None)``                |
+--------------------------------------------------------+------------------------------------------------------------------------+

The ``Model.ground`` attribute and the special ground collision handling have been removed. Instead, you need to manually add a ground plane via :meth:`newton.ModelBuilder.add_ground_plane`.

The attributes related to joint axes now have the same dimension as the joint dofs, which is :attr:`newton.Model.joint_dof_count`.
The ``Model.joint_axis`` attribute has been removed since it now equals :attr:`newton.Model.joint_qd_start`.

``Control``
-----------

The :class:`newton.Control` class now has a :attr:`newton.Control.joint_f` attribute which encodes the generalized force (torque) input to the joints.
In order to match the MuJoCo convention, :attr:`~newton.Control.joint_f` now includes the dofs of the free joints as well, so its dimension is :attr:`newton.Model.joint_dof_count`.
The control mode ``JOINT_MODE_FORCE`` has been removed, since it is now realized by setting :attr:`Control.joint_f` instead of ``joint_act``.
To disable joint target control for a dof, use ``JOINT_MODE_NONE``.

The :class:`newton.Control` class now has a :attr:`newton.Control.joint_target` attribute (in place of the previous ``joint_act`` attribute) that encodes either the position or the velocity target for the control,
depending on the control mode selected for the joint dof.
Using joints with zero stiffness (:class:`newton.ModelBuilder.JointDofConfig.target_ke`) and damping (:class:`newton.ModelBuilder.JointDofConfig.target_kd`) will disable the target control.

.. note::

    :attr:`newton.Control.joint_target` is likely a temporary attribute and may be removed in a future release in favor of a more general actuation interface.


``ModelBuilder``
----------------

The default up axis of the builder is now Z instead of Y.

Analogously, the geometry types plane, capsule, cylinder, and cone now have their up axis set to the Z axis instead of Y by default.

+--------------------------------------------------------+------------------------------------------------------------------------+
| **warp.sim**                                           | **Newton**                                                             |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_body(origin=..., m=...)``           | ``ModelBuilder.add_body(xform=..., mass=...)``                         |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder._add_shape()``                          | :func:`newton.ModelBuilder.add_shape`                                  |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_shape_*(pos=..., rot=...)``         | ``ModelBuilder.add_shape_*(xform=...)``                                |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_shape_*(..., ke=..., ka=..., ...)`` | ``ModelBuilder.add_shape_*(cfg=ShapeConfig(ke=..., ka=..., ...))``     |
|                                                        | see :class:`newton.ModelBuilder.ShapeConfig`                           |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_joint_*(..., target=...)``          | ``ModelBuilder.add_joint_*(..., action=...)``                          |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder(up_vector=(0, 1, 0))``                  | ``ModelBuilder(up_axis="Y")`` or ``ModelBuilder(up_axis=Axis.Y)``      |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``JointAxis``                                          | :class:`newton.ModelBuilder.JointDofConfig`                            |
+--------------------------------------------------------+------------------------------------------------------------------------+

It is now possible to set the up axis of the builder using the :attr:`~newton.ModelBuilder.up_axis` attribute,
which can be defined from any value compatible with the :obj:`~newton.core.types.AxisType` alias.
:attr:`newton.ModelBuilder.up_vector` is now a read-only property computed from :attr:`newton.ModelBuilder.up_axis`.

The ``ModelBuilder.add_joint_*()`` functions now use ``None`` as default args values to be filled in by the ``ModelBuilder.default_joint_*`` attributes.

The ``ModelBuilder.add_joint*()`` methods no longer accept ``linear_compliance`` and ``angular_compliance`` arguments
and the ``Model`` no longer stores them as attributes.
Instead, you can pass them as arguments to the :class:`newton.solvers.XPBDSolver` constructor. Note that now these values
apply to all joints and cannot be set individually per joint anymore. So far we have not found applications that require
per-joint compliance settings and have decided to remove this feature for memory efficiency.

The :meth:`newton.ModelBuilder.add_joint_free()` method now initializes the positional dofs of the free joint with the child body's transform (``body_q``).

The universal and compound joints have been removed in favor of the more general D6 joint.


Renderers
---------

+-----------------------------------------------+----------------------------------------------+
| **warp.sim**                                  | **Newton**                                   |
+-----------------------------------------------+----------------------------------------------+
|``warp.sim.render.SimRenderer``                |:class:`newton.utils.SimRenderer`             |
+-----------------------------------------------+----------------------------------------------+
|:attr:`warp.sim.render.SimRendererUsd`         |:class:`newton.utils.SimRendererUsd`          |
+-----------------------------------------------+----------------------------------------------+
|:attr:`warp.sim.render.SimRendererOpenGL`      |:class:`newton.utils.SimRendererOpenGL`       |
+-----------------------------------------------+----------------------------------------------+
