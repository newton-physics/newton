.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

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

USD cables now use ``physics:restShapePoints`` and ``physics:restNormals`` as their simulated
rest shape. Previously, rest points affected only material gains and rest normals were ignored.
With both rest fields absent, the cable uses its initial geometry as its rest shape.
Review assets that relied on rest fields being ignored: remove unwanted rest fields, or set
them to the initial points and normals, to start relaxed. Separately, welded cable graphs now
honor authored initial normals even without rest fields; earlier versions ignored those normals.
Rest lengths now determine capsule size and mass volume weights. Explicit segment masses keep
their authored values. Attachments follow native rod material endpoints, which can differ from
the initial USD points when initial and rest segment lengths differ; see :ref:`usd_parsing`
for the supported subset.
These changes affect the experimental deformable importer, so they do not require a deprecation
period under Newton's :doc:`compatibility policy <guide/compatibility>`.

The MJCF importer from Warp sim only uses the ``geom_density`` defined in the MJCF for sphere and box shapes but ignores these definitions for other shape types (which will receive the default density specified by the ``density`` argument to ``wp.sim.parse_mjcf``). The Newton MJCF importer now considers the ``geom_density`` for all shape types. This change may yield to different simulation results and may require tuning contact and other simulation parameters to achieve similar results in Newton compared to Warp sim.


``Model``
---------

:attr:`newton.Model.shape_is_solid` is now of dtype ``bool`` instead of ``wp.uint8``.

The ``Model.ground`` attribute and the special ground collision handling have been removed. Instead, you need to manually add a ground plane via :meth:`newton.ModelBuilder.add_ground_plane`.

Newton's public ``spatial_vector`` arrays now use ``(linear, angular)`` ordering.
For example, :attr:`newton.State.body_qd` stores ``(lin_vel, ang_vel)``, whereas
``warp.sim`` followed Warp's native ``(ang_vel, lin_vel)`` convention. See
:ref:`Twist conventions`.

For rigid bodies, the linear component is the world-frame velocity of the body's
center of mass (COM). If you need the body-origin velocity instead, recover it
as ``v_origin = v_com - omega x r_com_world``.

The attributes related to joint axes now have the same dimension as the joint DOFs, which is
:attr:`newton.Model.joint_dof_count`. :attr:`newton.Model.joint_axis` remains available and is
indexed per DOF; use :attr:`newton.Model.joint_qd_start` and :attr:`newton.Model.joint_dof_dim`
to locate a joint's slice in the per-DOF arrays.

For free and D6 joints, Newton stores linear DOFs before angular DOFs in per-axis arrays. In
particular, floating-base slices of :attr:`newton.State.joint_qd`, :attr:`newton.Control.joint_f`,
and :attr:`newton.Control.joint_target_qd` use
``(linear, angular)`` ordering, whereas ``warp.sim`` used ``(ang_vel, lin_vel)``.
:attr:`newton.Control.joint_target_q` keeps that ordering but follows :attr:`newton.State.joint_q`:
a free joint occupies 7 coordinates ``(position, quaternion)``. See
:ref:`joint-target-layout` for migration guidance.
For public ``FREE``, ``DISTANCE``, and ``ROD`` joints, :attr:`newton.State.joint_qd`
stores the child-COM twist in the joint parent frame, while
:attr:`newton.Control.joint_f` stores the world-frame COM wrench
``(f_world, tau_com_world)``.

Rod joints now use seven ``joint_q`` coordinates for relative position and
orientation and six ``joint_qd`` entries for relative linear and angular
velocity. Their material-axis order is ``[shear_x, shear_y, stretch_z,
bend_x, bend_y, twist_z]``. Migrate old ``[stretch, shear, bend, twist]``
offsets ``[0, 1, 2, 3]`` to ``[2, (0, 1), (3, 4), 5]``, and use
:attr:`newton.Model.joint_q_start` and
:attr:`newton.Model.joint_qd_start` to locate each Rod slice.

Rod angular rest is stored in :attr:`newton.Model.joint_target_q`; after
editing it, call :meth:`newton.solvers.SolverVBD.notify_model_changed` with
:attr:`newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES`. Editing
:attr:`newton.Model.body_q` no longer changes angular rest.

Code that wrote the old bend offset ``qd_start + 2`` now silently sets stretch.
The shear/bend isotropy check cannot detect every outdated offset; audit
direct Rod ``joint_target_ke``/``joint_target_kd`` indexing.

:func:`newton.eval_fk` now reconstructs Rod body state from ``joint_q`` and
``joint_qd`` like free joints.
:class:`newton.solvers.SolverVBD` advances Rod bodies in maximal coordinates
without updating these arrays. Before calling ``eval_fk`` during the simulation,
exclude the rod articulations with ``mask``/``indices`` or use
:func:`newton.eval_ik` to update the coordinate arrays passed to ``eval_fk``.

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
:attr:`newton.Control.joint_target_q` and :attr:`newton.Control.joint_target_qd` store joint
position and velocity targets, :attr:`newton.Control.joint_act` stores feedforward actuator input,
and :attr:`newton.Control.joint_f` stores generalized forces/torques. Unlike ``warp.sim``,
``joint_act`` is no longer the target array.

In order to match the MuJoCo convention, :attr:`~newton.Control.joint_f` includes the DOFs of
``FREE`` and ``DISTANCE`` joints as well, so its dimension is :attr:`newton.Model.joint_dof_count`.

``JointMode`` has been replaced by :class:`newton.JointTargetMode`. Direct force control
corresponds to :attr:`newton.JointTargetMode.EFFORT` together with
:attr:`newton.Control.joint_f`, while simultaneous position and velocity target control uses
:attr:`newton.JointTargetMode.POSITION_VELOCITY` together with
:attr:`newton.Control.joint_target_q` and :attr:`newton.Control.joint_target_qd`.

.. _joint-target-layout:

Joint-target layout
~~~~~~~~~~~~~~~~~~~

In ``warp.sim`` the target array was DOF-shaped like ``joint_qd``. Newton stores position
targets in the coordinate layout of :attr:`~newton.State.joint_q` instead, since that is what a
position semantically is: the two layouts diverge whenever an articulation contains a free,
distance, or rod joint (7 coords vs. 6 DOFs) or ball joint (4 coords vs. 3 DOFs), and under the
DOF layout every actuated DOF downstream of such a joint is indexed with the wrong stride.

Position targets always use the coordinate layout:

.. code-block:: python

   import newton

   builder = newton.ModelBuilder()
   # ... build articulation ...
   model = builder.finalize()
   # model.joint_target_q  has shape (joint_coord_count,)  — matches joint_q
   # model.joint_target_qd has shape (joint_dof_count,)    — matches joint_qd

Under the coordinate layout, ball and free joint position targets are quaternions (plus a
translation for free joints), matching ``joint_q``. Migration notes coming from ``warp.sim``:

- Replace writes to the ``warp.sim`` target arrays with
  :attr:`Control.joint_target_q` / :attr:`Control.joint_target_qd`. Configure per-axis initial
  targets via :attr:`ModelBuilder.JointDofConfig.target_pos` /
  :attr:`~ModelBuilder.JointDofConfig.target_vel` before calling ``add_joint*()``, or write to
  :attr:`ModelBuilder.joint_target_q` / :attr:`~ModelBuilder.joint_target_qd` directly.
- When indexing ``joint_target_q`` from user code, use :attr:`Model.joint_target_q_start` (which
  aliases :attr:`Model.joint_q_start`). Velocity targets continue to use
  :attr:`Model.joint_qd_start`.
- When constructing an :class:`Actuator` for the standard ``joint_target_q`` array,
  ``target_pos_indices`` defaults to ``pos_indices`` and can be omitted. Keep an explicit
  ``target_pos_indices`` mapping for custom target buffers that use different indices.

Newton 1.8 removes ``newton.use_coord_layout_targets`` and the legacy DOF-shaped position-target
layout, deprecated in 1.5. Remove assignments to the flag and reads of
``Model.use_coord_layout_targets``. Replace DOF-based position-target indices with coordinate
indices, and replace raw Euler-angle targets for ball/free/distance/rod joints with unit
quaternions. ``Model.joint_target_q_start`` remains an alias for ``Model.joint_q_start``.


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
which can be defined from any value compatible with the :obj:`~newton.AxisType` alias.
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
| ``contacts = model.collide(state)``           | ``pipeline.collide(state, contacts)``                        |
+-----------------------------------------------+--------------------------------------------------------------+

Create a :class:`~newton.CollisionPipeline` and its contacts buffer before stepping::

    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

The compatibility helpers :meth:`~newton.Model.contacts` and
:meth:`~newton.Model.collide` are deprecated in Newton 1.5.


Renderers
---------

+-----------------------------------------------+----------------------------------------------+
| **warp.sim**                                  | **Newton**                                   |
+-----------------------------------------------+----------------------------------------------+
|:class:`warp.sim.render.UsdRenderer`           |:class:`newton.viewer.ViewerUSD`              |
+-----------------------------------------------+----------------------------------------------+
|:class:`warp.sim.render.OpenGLRenderer`        |:class:`newton.viewer.ViewerGL`               |
+-----------------------------------------------+----------------------------------------------+
