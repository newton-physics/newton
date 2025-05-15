API Reference
=============

Core Data Structures
--------------------

.. autoclass:: newton.Model

.. autoclass:: newton.State

.. autoclass:: newton.Control

.. autoclass:: newton.ModelBuilder

.. autoclass:: newton.core.builder.JointDofConfig

.. autoclass:: newton.core.builder.ShapeConfig

Solvers
-------

.. autoclass:: newton.solvers.SolverBase
    
.. autoclass:: newton.solvers.XPBDSolver

.. autoclass:: newton.solvers.VBDSolver

.. autoclass:: newton.solvers.MuJoCoSolver

Importers
---------

Newton supports the loading of simulation models from URDF, MuJoCo (MJCF), and USD Physics files.

.. autofunction:: newton.utils.parse_urdf

.. autofunction:: newton.utils.parse_mjcf

.. autofunction:: newton.utils.parse_usd

.. autofunction:: newton.utils.import_usd.resolve_usd_from_url

Collision Detection
-------------------

.. _Joint types:

Joint Types
-----------

.. data:: JOINT_PRISMATIC

    Prismatic (slider) joint

.. data:: JOINT_REVOLUTE

    Revolute (hinge) joint

.. data:: JOINT_BALL

    Ball (spherical) joint with quaternion state representation

.. data:: JOINT_FIXED

    Fixed (static) joint

.. data:: JOINT_FREE

    Free (floating) joint

.. data:: JOINT_COMPOUND

    Compound joint with 3 rotational degrees of freedom

.. data:: JOINT_UNIVERSAL

    Universal joint with 2 rotational degrees of freedom

.. data:: JOINT_DISTANCE

    Distance joint that keeps two bodies at a distance within its joint limits (only supported in :class:`XPBDIntegrator` at the moment)

.. data:: JOINT_D6

    Generic D6 joint with up to 3 translational and 3 rotational degrees of freedom

.. _Joint modes:

Joint Control Modes
-------------------

Joint modes control whether the respective :attr:`newton.Control.joint_target` input is a joint position or velocity target.

.. data:: JOINT_MODE_TARGET_POSITION

    The control input is the target position :math:`\mathbf{q}_{\text{target}}` which is achieved via PD control of torque :math:`\tau` where the proportional and derivative gains are set by :attr:`Model.joint_target_ke` and :attr:`Model.joint_target_kd`:

    .. math::

        \tau = k_e (\mathbf{q}_{\text{target}} - \mathbf{q}) - k_d \mathbf{\dot{q}}

.. data:: JOINT_MODE_TARGET_VELOCITY
   
    The control input is the target velocity :math:`\mathbf{\dot{q}}_{\text{target}}` which is achieved via a controller of torque :math:`\tau` that brings the velocity at the joint axis to the target through proportional gain :attr:`Model.joint_target_ke`: 
    
    .. math::

        \tau = k_e (\mathbf{\dot{q}}_{\text{target}} - \mathbf{\dot{q}})    

Renderers
---------

Based on the renderers from :mod:`warp.render`, the :class:`newton.utils.SimRendererUsd` (which equals :class:`newton.utils.SimRenderer`) and
:class:`newton.utils.SimRendererOpenGL` classes from :mod:`newton.utils.render` are derived to populate the renderers directly from
:class:`newton.ModelBuilder` scenes and update them from :class:`newton.State` objects.

.. autoclass:: newton.utils.SimRendererUsd
    :members:

.. autoclass:: newton.utils.SimRendererOpenGL
    :members:
