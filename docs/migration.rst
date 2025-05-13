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
| ``integrator.simulate(self.model, self.state0, self.state1, self.dt, None)`` | ``solver.step(self.model, self.state0, self.state1, self.control, None, self.dt)``  |
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
Instead, you can set the default joint properties on the :class:`newton.ModelBuilder` class.
For example, ``limit_lower`` is now defined using :attr:`ModelBuilder.default_joint_limit_lower`.

Similarly, the shape contact parameters have been removed from the importers.
Instead, you can set the default contact parameters on the :attr:`newton.ModelBuilder.default_shape_cfg` object before loading the asset.
For example, ``ke`` is now defined using :attr:`ModelBuilder.default_shape_cfg.ke`.


``Model``
---------

``ModelShapeGeometry.is_solid`` now is of dtype ``bool`` instead of ``wp.uint8``.


``ModelBuilder``
----------------

+--------------------------------------------------------+------------------------------------------------------------------------+
| **warp.sim**                                           | **Newton**                                                             |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_body(origin=..., m=...)``           | ``ModelBuilder.add_body(xform=..., mass=...)``                         |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder._add_shape()``                          | :func:`ModelBuilder.add_shape`                                         |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_shape_*(pos=..., rot=...)``         | ``ModelBuilder.add_shape_*(xform=...)``                                |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_shape_*(..., ke=..., ka=..., ...)`` | ``ModelBuilder.add_shape_*(cfg=ShapeProperties(ke=..., ka=..., ...))`` |
+--------------------------------------------------------+------------------------------------------------------------------------+
| ``ModelBuilder.add_joint_*(..., target=...)``          | ``ModelBuilder.add_joint_*(..., action=...)``                          |
+--------------------------------------------------------+------------------------------------------------------------------------+

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
