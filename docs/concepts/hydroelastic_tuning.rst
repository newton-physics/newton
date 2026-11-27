.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Tuning Hydroelastic Contacts:

Hydroelastic Contacts Tuning
============================

This guide shows how to enable hydroelastic contacts for robotic manipulation
and choose starting values for stiffness and friction.

Enabling Hydroelastic Contacts
------------------------------

Both shapes in a pair need SDF data and must have ``is_hydroelastic=True``.
For mesh shapes, use closed, volumetric meshes and build their SDFs on CUDA
before finalizing the model. The example uses two overlapping copies of the
same mesh, so they can share one SDF.

.. code-block:: python

    import newton
    import warp as wp

    with wp.ScopedDevice("cuda:0"):
        mesh = newton.Mesh.create_box(0.02, duplicate_vertices=False)
        mesh.build_sdf(max_resolution=64, margin=0.005)
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(is_hydroelastic=True, kh=2.0e9, gap=0.0)
        builder.add_shape_mesh(-1, mesh=mesh, cfg=cfg)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.039), wp.quat_identity()))
        builder.add_shape_mesh(body, mesh=mesh, cfg=cfg)
        model = builder.finalize()

Set ``gap`` explicitly because an omitted value inherits the builder's
``rigid_gap`` (0.1 m by default). If you use a wider gap or a positive shape
margin, make sure the construction padding (``margin`` in
:meth:`~Mesh.build_sdf`) covers their sum. For primitive shapes, configure
``sdf_max_resolution`` or ``sdf_target_voxel_size`` on
:class:`~ModelBuilder.ShapeConfig`. For scaled meshes, bake non-unit scale into
the SDF. See :ref:`Hydroelastic Contacts` for the full requirements.

Choosing the Stiffness
----------------------

With the default linear law, the pressure inside each shape is its
hydroelastic stiffness ``kh`` times the depth below its margin-adjusted surface.
Pressure is in N/m\ :sup:`2`, so ``kh`` is in N/m\ :sup:`3`. At the
pressure-balanced contact surface, the two material slopes combine in series:

.. math::

   k_{\mathrm{pair}} = \frac{k_a k_b}{k_a + k_b}.

Equal per-shape values give half that value for the pair. When one surface is
much stiffer than the other, the pair slope is close to the softer surface's
``kh``.

For a roughly flat pad or surface layer, estimate each shape's ``kh`` from the
change in normal pressure divided by the corresponding change in compression
depth. For example, if a backed gripper pad compresses by 0.5 mm under 0.1 MPa
of pressure with a similar contact footprint, start with ``kh = 2e8`` N/m\ :sup:`3`
for the pad. Repeat for the object surface if it has appreciable compliance.
For a uniform layer of thickness ``t`` under small, approximately uniaxial
compression, ``E / t`` is a starting estimate, where ``E`` is the Young's
modulus in Pa. Layer geometry and backing affect compression, so prefer
measurements under a representative load when available.

You can also start from a desired normal force per contact patch. Estimate
its loaded area ``A`` and the total overlap depth ``d`` acceptable at that
force. For a roughly flat patch with nearly uniform pressure:

.. math::

   F \approx k_{\mathrm{pair}} A d.

For example, 50 N over 1 cm\ :sup:`2` at 0.5 mm overlap gives
``k_pair = 1e9`` N/m\ :sup:`3`. If the two surfaces have similar compliance,
start with ``kh = 2e9`` N/m\ :sup:`3` on each. Check the actual force and
overlap with your solver, timestep, and contact geometry. The estimate
describes the collision pressure law. The solver determines how the
generated contact stiffness contributes to the simulated response.
For example, use ``use_mujoco_contacts=False`` with :class:`~solvers.SolverMuJoCo`
to use Newton's contacts.

If the contact compresses too much under load, raise ``kh`` on the more
compliant side first. If it oscillates, compare a lower stiffness or smaller
timestep under the same load. To move the active contact surface itself, use
:attr:`~ModelBuilder.ShapeConfig.margin`. Positive values expand it and
negative values move it inward.

Tuning Friction
---------------

Start with a friction coefficient ``mu`` for each participating shape based
on the surface pair. Test for slip under the normal load and tangential
motion expected in the task. If the contact reaches its friction limit too
early, check the normal load and adjust ``mu`` within a plausible range.
Increasing ``mu`` raises the available tangential force. The tangential
response below that limit depends on the solver and its friction settings.
See :doc:`Solver Tuning <simulation_tuning_solvers>`.

Inspecting Contacts
-------------------

Enable contact-surface output when constructing the collision pipeline.
After collision detection, retrieve the generated surface to inspect patch
location and coverage.

.. code-block:: python

    from newton.geometry import HydroelasticSDF

    pipeline = newton.CollisionPipeline(
        model, sdf_hydroelastic_config=HydroelasticSDF.Config(output_contact_surface=True)
    )
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    surface = pipeline.hydroelastic_sdf.get_contact_surface()

If the patch flickers or looks coarse, inspect the mesh and compare a finer
SDF grid using ``target_voxel_size`` or ``max_resolution``. Check that
``narrow_band_range`` covers the distances of interest. If Newton reports
hydroelastic buffer overflows, increase ``HydroelasticSDF.Config.buffer_fraction``
to give contact generation more capacity. See :ref:`Hydroelastic Contacts`
for buffer limits and :github:`newton/examples/robot/example_robot_panda_hydro.py`
for an example that displays the contact surface.
