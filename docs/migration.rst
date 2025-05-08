``warp.sim`` Migration Guide
============================

This guide is designed for users seeking to migrate their applications from ``warp.sim`` to Newton.


Solvers
-------

+------------------------------------------------------------------------------+-------------------------------------------------------------------------------------+
| **warp.sim**                                                                 | **Newton**                                                                          |
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


``ModelBuilder``
----------------

+-----------------------------------------------+----------------------------------------------+
| **warp.sim**                                  | **Newton**                                   |
+-----------------------------------------------+----------------------------------------------+
|``ModelBuilder.add_body(origin=..., m=...)``   |``ModelBuilder.add_body(xform=..., mass=...)``|
+-----------------------------------------------+----------------------------------------------+
|``ModelBuilder._add_shape()``                  |:func:`ModelBuilder.add_shape`                |
+-----------------------------------------------+----------------------------------------------+
|``ModelBuilder.add_shape_*(pos=..., rot=...)`` |``ModelBuilder.add_shape_*(xform=...)``       |
+-----------------------------------------------+----------------------------------------------+

The ``ModelBuilder.add_joint*()`` methods no longer accept ``linear_compliance`` and ``angular_compliance`` arguments
and the ``Model`` no longer stores them as attributes.
Instead, you can pass them as arguments to the :class:`newton.XPBDSolver` constructor. Note that now these values
apply to all joints and cannot be set individually per joint anymore. We have not found applications that require
per-joint compliance settings and have decided to remove this feature for memory efficiency.

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
