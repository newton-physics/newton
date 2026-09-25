.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

``warp.sim`` Migration Guide
============================

This guide is designed for users seeking to migrate their applications from ``warp.sim`` to Newton.


Solvers
-------

+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| **warp.sim**                                                                 | **Newton**                                                                          |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
|:class:`warp.sim.FeatherstoneIntegrator`                                      |:class:`newton.solvers.SolverFeatherstone`                                           |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
|:class:`warp.sim.SemiImplicitIntegrator`                                      |:class:`newton.solvers.SolverSemiImplicit`                                           |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
|:class:`warp.sim.VBDIntegrator`                                               |:class:`newton.solvers.SolverVBD`                                                    |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
|:class:`warp.sim.XPBDIntegrator`                                              |:class:`newton.solvers.SolverXPBD`                                                   |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| ``integrator.simulate(self.model, self.state0, self.state1, self.dt, None)`` | ``solver.step(self.state0, self.state1, self.control, None, self.dt)``              |
+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+

Importers
---------

+-----------------------------------------------+---------------------------------------------------------+
| **warp.sim**                                  | **Newton**                                              |
+-----------------------------------------------+---------------------------------------------------------+
|:func:`warp.sim.parse_urdf`                    |:meth:`newton.ModelBuilder.add_urdf`                     |
+-----------------------------------------------+---------------------------------------------------------+
|:func:`warp.sim.parse_mjcf`                    |:meth:`newton.ModelBuilder.add_mjcf`                     |
+-----------------------------------------------+---------------------------------------------------------+
|:func:`warp.sim.parse_usd`                     |:meth:`newton.ModelBuilder.add_usd`                      |
+-----------------------------------------------+---------------------------------------------------------+

The joint-specific arguments to the importers have been removed.
Instead, you can set the default joint properties on a :class:`newton.ModelBuilder` instance in the :attr:`newton.ModelBuilder.default_joint_cfg` attribute.
For example, ``limit_lower`` is now defined using ``builder.default_joint_cfg.limit_lower``, where ``builder`` is an instance of :class:`newton.ModelBuilder`.

Similarly, the shape contact parameters have been removed from the importers.
Instead, you can set the default contact parameters on a :class:`newton.ModelBuilder` instance in the :attr:`newton.ModelBuilder.default_shape_cfg` object before loading the asset.
For example, ``ke`` is now defined using ``builder.default_shape_cfg.ke``, where ``builder`` is an instance of :class:`newton.ModelBuilder`.

The MJCF and URDF importers both have an ``up_axis`` argument that defaults to +Z.
All importers will rotate the asset now to match the builder's ``up_axis`` (instead of overwriting the ``up_axis`` in the builder, as was the case previously for the USD importer).

:meth:`newton.ModelBuilder.add_usd` accepts both file paths and URLs directly, so a separate
``resolve_usd_from_url()`` helper is usually unnecessary when migrating from ``warp.sim``.

The MJCF importer from Warp sim only uses the ``geom_density`` defined in the MJCF for sphere and box shapes but ignores these definitions for other shape types (which will receive the default density specified by the ``density`` argument to ``wp.sim.parse_mjcf``). The Newton MJCF importer now considers the ``geom_density`` for all shape types. This change may yield to different simulation results and may require tuning contact and other simulation parameters to achieve similar results in Newton compared to Warp sim.


``Model``
---------

:attr:`newton.Model.shape_is_solid` is now of dtype ``bool`` instead of ``wp.uint8``.

The ``Model.ground`` attribute and the special ground collision handling have been removed. Instead, you need to manually add a ground plane via :meth:`newton.ModelBuilder.add_ground_plane`.

Newton's public ``spatial_vector`` arrays now use ``(linear, angular)`` ordering.
For example, :attr:`newton.State.body_qd` stores ``(lin_vel, ang_vel)``, whereas
``warp.sim`` followed Warp's native ``(ang_vel, lin_vel)`` convention. See
:ref:`Twist conventions`.

The attributes related to joint axes now have the same dimension as the joint DOFs, which is
:attr:`newton.Model.joint_dof_count`. :attr:`newton.Model.joint_axis` remains available and is
indexed per DOF; use :attr:`newton.Model.joint_qd_start` and :attr:`newton.Model.joint_dof_dim`
to locate a joint's slice in the per-DOF arrays.

For free and D6 joints, Newton stores linear DOFs before angular DOFs in per-axis arrays. In
particular, floating-base slices of :attr:`newton.State.joint_qd`, :attr:`newton.Control.joint_f`,
:attr:`newton.Control.joint_target_pos`, and :attr:`newton.Control.joint_target_vel` use
``(lin_vel, ang_vel)`` ordering, whereas ``warp.sim`` used ``(ang_vel, lin_vel)``.

+------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| **warp.sim**                                                     | **Newton**                                                                                                            |
+------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Model.shape_geo_src``                                          | :attr:`Model.shape_source`                                                                                            |
+------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Model.shape_geo``                                              | Removed ``ShapeGeometry`` struct                                                                                      |
+------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Model.shape_geo.type``, ``Model.shape_geo.scale``, etc.        | :attr:`Model.shape_type`, :attr:`Model.shape_scale`, etc.                                                             |
+------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Model.shape_geo.source``                                       | :attr:`Model.shape_source_ptr`                                                                                        |
+------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Model.shape_materials``                                        | Removed ``ShapeMaterial`` struct                                                                                      |
+------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Model.shape_materials.ke``, ``Model.shape_materials.kd``, etc. | :attr:`Model.shape_material_ke`, :attr:`Model.shape_material_kd`, etc.                                                |
+------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Model.rigid_contact_torsional_friction``                       | :attr:`Model.shape_material_mu_torsional` (now per-shape array)                                                       |
|                                                                  |                                                                                                                       |
|                                                                  | Note: these coefficients are now interpreted as absolute values rather than being scaled by the friction coefficient. |
+------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``Model.rigid_contact_rolling_friction``                         | :attr:`Model.shape_material_mu_rolling` (now per-shape array)                                                         |
|                                                                  |                                                                                                                       |
|                                                                  | Note: these coefficients are now interpreted as absolute values rather than being scaled by the friction coefficient. |
+------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------+

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

``Control``
-----------

The :class:`newton.Control` interface is split by responsibility:
:attr:`newton.Control.joint_target_pos` and :attr:`newton.Control.joint_target_vel` store per-DOF
position and velocity targets, :attr:`newton.Control.joint_act` stores feedforward actuator input,
and :attr:`newton.Control.joint_f` stores generalized forces/torques. Unlike ``warp.sim``,
``joint_act`` is no longer the target array.

In order to match the MuJoCo convention, :attr:`~newton.Control.joint_f` includes the DOFs of the
free joints as well, so its dimension is :attr:`newton.Model.joint_dof_count`.

``JointMode`` has been replaced by :class:`newton.JointTargetMode`. Direct force control
corresponds to :attr:`newton.JointTargetMode.EFFORT` together with
:attr:`newton.Control.joint_f`, while simultaneous position and velocity target control uses
:attr:`newton.JointTargetMode.POSITION_VELOCITY` together with
:attr:`newton.Control.joint_target_pos` and :attr:`newton.Control.joint_target_vel`.


``ModelBuilder``
----------------

The default up axis of the builder is now Z instead of Y.

Analogously, the geometry types plane, capsule, cylinder, and cone now have their up axis set to the Z axis instead of Y by default.

+--------------------------------------------------------+------------------------------------------------------------------------+
| **warp.sim**                                           | **Newton**                                                             |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_body(origin=..., m=...)``           | ``ModelBuilder.add_body(xform=..., mass=...)``                         |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder._add_shape()``                          | :meth:`newton.ModelBuilder.add_shape`                                  |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_shape_*(pos=..., rot=...)``         | ``ModelBuilder.add_shape_*(xform=...)``                                |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_shape_*(..., ke=..., ka=..., ...)`` | ``ModelBuilder.add_shape_*(cfg=ShapeConfig(ke=..., ka=..., ...))``     |
|                                                        | see :class:`newton.ModelBuilder.ShapeConfig`                           |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_joint_*(..., target=...)``          | ``ModelBuilder.add_joint_*(..., target_pos=..., target_vel=...)``      |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder(up_vector=(0, 1, 0))``                  | ``ModelBuilder(up_axis="Y")`` or ``ModelBuilder(up_axis=Axis.Y)``      |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``JointAxis``                                          | :class:`newton.ModelBuilder.JointDofConfig`                            |
+--------------------------------------------------------+------------------------------------------------------------------------+

It is now possible to set the up axis of the builder using the :attr:`~newton.ModelBuilder.up_axis` attribute,
which can be defined from any value compatible with the :obj:`~newton.core.types.AxisType` alias.
:attr:`newton.ModelBuilder.up_vector` is now a read-only property computed from :attr:`newton.ModelBuilder.up_axis`.

The ``ModelBuilder.add_joint_*()`` functions now use ``None`` defaults that are filled in from
the fields of :attr:`newton.ModelBuilder.default_joint_cfg`.

Newton uses ``world_count`` throughout the public API (for example in
:meth:`newton.ModelBuilder.replicate` and :attr:`newton.Model.world_count`); older ``num_envs``
terminology is obsolete.

The ``ModelBuilder.add_joint*()`` methods no longer accept ``linear_compliance`` and ``angular_compliance`` arguments
and the ``Model`` no longer stores them as attributes.
Instead, you can pass them as arguments to the :class:`newton.solvers.SolverXPBD` constructor. Note that now these values
apply to all joints and cannot be set individually per joint anymore. So far we have not found applications that require
per-joint compliance settings and have decided to remove this feature for memory efficiency.

The :meth:`newton.ModelBuilder.add_joint_free()` method now initializes the positional dofs of the free joint with the child body's transform (``body_q``).

The universal and compound joints have been removed in favor of the more general D6 joint.


Collisions
----------

+-----------------------------------------------+--------------------------------------------------------------+
| **warp.sim**                                  | **Newton**                                                   |
+-----------------------------------------------+--------------------------------------------------------------+
| ``contacts = model.collide(state)``           | ``contacts = model.collide(state)``                          |
+-----------------------------------------------+--------------------------------------------------------------+

:meth:`~newton.Model.collide` allocates and returns a contacts buffer when ``contacts`` is omitted.
For more control, create a :class:`~newton.CollisionPipeline` directly.


Renderers
---------

+-----------------------------------------------+----------------------------------------------+
| **warp.sim**                                  | **Newton**                                   |
+-----------------------------------------------+----------------------------------------------+
|:class:`warp.sim.render.UsdRenderer`           |:class:`newton.viewer.ViewerUSD`              |
+-----------------------------------------------+----------------------------------------------+
|:class:`warp.sim.render.OpenGLRenderer`        |:class:`newton.viewer.ViewerGL`               |
+-----------------------------------------------+----------------------------------------------+
