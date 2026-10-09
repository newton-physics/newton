.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _asset-migration-1-7:

Asset Migration for Newton 1.7
==============================

Newton 1.7 corrects several asset-import interpretations that can change the
behavior of scenes tuned with Newton 1.6. The options below preserve specific
previous interpretations while an application is migrated. They default to
``False`` and apply only to the asset loaded by that call; they do not modify
the source asset or builder defaults. They do not restore all Newton 1.6
behavior or guarantee identical simulation trajectories across releases.

URDF Joint Damping
------------------

In Newton 1.7, :meth:`newton.ModelBuilder.add_urdf` imports
``<dynamics damping="...">`` into passive :attr:`newton.Model.joint_damping`.
Drive damping, :attr:`newton.Model.joint_target_kd`, comes independently from
``builder.default_joint_cfg.target_kd``. In Newton 1.6, the authored value
instead supplied drive damping, so controllers tuned with that mapping can
behave differently even when the URDF loads successfully.

To preserve the previous damping mapping:

.. code-block:: python

    import newton

    builder = newton.ModelBuilder()
    builder.add_urdf("robot.urdf", legacy_joint_damping=True)

This restores each imported joint's authored drive damping, including distinct
values on different joints and explicit zeros. If damping is absent, drive
damping uses ``builder.default_joint_cfg.target_kd``. Passive damping retains
the old defaults: ``builder.default_joint_cfg.damping`` for revolute and
prismatic joints, and zero for planar joints. Copying the authored value into
both damping fields would add damping twice in solvers that consume both.

To migrate to the corrected mapping, omit the option and configure the
controller's drive gains explicitly. A builder default can set a common drive
gain, but it cannot reproduce different authored gains on different joints.
Set per-joint gains before :meth:`~newton.ModelBuilder.finalize` when needed:

.. code-block:: python

    builder = newton.ModelBuilder()
    builder.add_urdf("robot.urdf")
    joint = builder.joint_label.index("robot/shoulder")  # Use the imported label.
    dof = builder.joint_qd_start[joint]
    builder.joint_target_kd[dof] = 5.0

.. note::

   :class:`newton.solvers.SolverXPBD` and :class:`newton.solvers.SolverVBD`
   do not currently consume passive ``joint_damping``. For those solvers,
   use ``legacy_joint_damping=True`` to retain the former imported damping,
   or explicitly configure drive damping. Passive damping and drive damping
   have different semantics; equal coefficients need not produce equal
   behavior, especially with nonzero velocity targets.

USD Joint-State Angular Velocities
----------------------------------

Newton 1.7 converts resolved ``state:angular:physics:velocity`` and native D6
``state:rotX:physics:velocity``, ``rotY``, and ``rotZ`` values from degrees/s
to radians/s. This covers revolute joints and revolute joints merged into D6
joints, as well as native D6 angular axes. A value of ``90`` therefore initializes
``joint_qd`` to approximately ``1.5708``, rather than ``90`` as in Newton 1.6.
State attributes must be recognized by the selected schema resolver.

For assets or application code that rely on the old interpretation:

.. code-block:: python

    import newton.usd as usd

    builder = newton.ModelBuilder()
    result = builder.add_usd(
        "robot.usda",
        schema_resolvers=[usd.SchemaResolverPhysx()],
        legacy_angular_velocity_units=True,
    )

With the option enabled, resolved joint-state angular velocity values are
copied without angle conversion. Rigid-body ``physics:angularVelocity``
continues to convert degrees/s to radians/s. Linear joint velocities, angular
positions, drive targets, and velocity limits keep their usual interpretation.

To migrate, remove application code that manually converted these imported
joint-state velocities to radians/s. If the asset instead stored radians/s in
these degree-based attributes to compensate for the old importer, multiply
those authored values by ``180 / pi`` and remove the compatibility option.
For example, an old compensated value of ``1.5708`` should become approximately
``90``. Do not apply this conversion again to assets already authored in degrees/s.

USD MuJoCo Spring References
----------------------------

Newton 1.7 interprets revolute ``mjc:springref`` using the PhysicsScene's
``mjc:compiler:angle`` setting: ``"degree"`` or ``"radian"``. An unauthored
setting, including a stage without a PhysicsScene, defaults to degrees.
Prismatic spring references remain lengths. Newton 1.6 copied spring-reference
values directly into the model without angle conversion.

To retain that previous spring-reference interpretation:

.. code-block:: python

    from newton.solvers import SolverMuJoCo

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    result = builder.add_usd("robot.usda", legacy_springref_units=True)

This option affects ``mjc:springref`` only, including references on revolute
joints merged into D6 joints. It does not change ``mjc:ref`` conversion, the
authored-relative Newton joint-coordinate convention, or MJCF import.

To migrate an asset that stored radians under a degree compiler setting,
multiply each affected revolute ``mjc:springref`` by ``180 / pi``. For example,
``0.5`` radians becomes approximately ``28.6479`` degrees. Alternatively,
author ``mjc:compiler:angle = "radian"`` only if all attributes governed by
that setting, including ``mjc:ref``, use radians. Changing that setting solely
for spring references can misinterpret other joint attributes.

Checking a Migration
--------------------

Compare each asset's imported damping, initial joint velocities, and spring
references with its Newton 1.6 values. Then compare representative simulation
trajectories with the same solver and controller. Disable compatibility
options individually as assets and application compensations are updated.
