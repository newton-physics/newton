.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

.. _Tuning Hydroelastic Contacts:

Hydroelastic Contacts Tuning
===========================

This page describes how to turn on and tune Newton's hydroelastic contact
feature for a robotic manipulation task, such as choosing starting values for
the contact stiffness from the materials of the gripper and the manipulated
object, or how to tune friction.

.. important::

   The specific parameter names, ranges, and formulas on this page reflect the
   code at a point in time and can drift. Treat them as starting points and
   verify any you rely on against the cited source (see
   :class:`~newton.geometry.types.ShapeConfig` and
   :class:`~newton.geometry.types.SDF`).

Enabling Hydroelastic Contacts
------------------------------

Hydroelastic contacts require two steps:

1. Build a signed distance field on the colliding mesh before finalizing the
   model:

   .. code-block:: python

      mesh.build_sdf()

2. Mark the participating shapes as hydroelastic. For example:

   .. code-block:: python

      builder.add_shape(mesh, config=ShapeConfig(is_hydroelastic=True, kh=1e9))

   In :meth:`~ModelBuilder.finalize`, ``Model.shape_material_kh`` is built
   automatically from the per-shape ``kh`` values, and shapes flagged
   ``is_hydroelastic`` take part in the hydroelastic collision pipeline.

Choosing the Stiffness ``kh``
-----------------------------

The linear hydroelastic law is ``pressure = -kh * signed_depth``, where
``signed_depth`` is the penetration depth. ``kh`` has units of force per
volume of depth (N/m\ :sup:`3`), so it must scale with how much force the
gripper and object can deliver.

Starting-point heuristic:

- Estimate the volume ``V`` that the objects displace when pressed together
  (roughly: ``base_area * expected_penetration`` for a flat-on-flat contact).
- Estimate the force ``F`` the mechanisms can exert around the contact
  (gripper opening force, or the resting weight of the manipulated part
  plus the acceleration budget you care about).
- Set ``kh`` so that ``F ≈ kh * depth`` for a depth you consider shallow,
  e.g. ``depth ≈ 0.01 m`` for a fingered grip:

  .. code-block:: python

      kh = F / 0.01

  With ``F = 500 N`` this starts at ``kh = 5e4``, then adjust per-shape
  through :attr:`~Model.shape_material_kh`.

Rules of thumb when tuning:

- **Too soft (spongy)**: raise ``kh`` (raises pressure for the same depth).
- **Too stiff (objects bounce)**: lower ``kh``, and pair with a small
  negative :attr:`~newton.geometry.types.SDF.shape_margin`, which shrinks the
  effective penetration depth.
- **Uneven contacts across pairs**: ``kh_a`` and ``kh_b`` combine as an
  average, so an object much softer than the gripper drags the pair down;
  give the stiffer side the larger ``kh``.

Friction on Hydroelastic Contacts
---------------------------------

Hydroelastic contacts report contact points and areas, so the usual
friction pipeline applies. When a hydroelastic contact feels slip-prone:

- Increase the shape friction coefficient on both sides of the pair; the
  hydroelastic pressure term mainly controls normal force, and a higher
  tangential stiffness then follows from the friction model.
- If contacts are noisy, narrow the work range used by the contact sort
  keys or reduce the aggregate buffer factors on
  :class:`~newton.geometry.sdf_hydroelastic.HydroelasticSDF` instead of
  lowering ``kh`` blindly.

Useful Diagnostics
------------------

When a hydroelastic contact feels wrong, compare before/after changes with:

- ``Model.shape_material_kh``: the effective stiffness per shape.
- ``ShapeConfig.margin`` / :meth:`~newton.geometry.types.SDF.build_sdf`
  (``narrow_band_range``, ``shape_margin``): how far the SDF is sampled
  around the surface.
- Hydroelastic visual surfaces: set ``visualize_contact_surfaces=True`` on
  the :class:`~newton.geometry.sdf_hydroelastic.HydroelasticSDF` configuration
  to inspect the generated contact surface directly.

Related Pages
-------------

- :doc:`Collisions and Contacts </concepts/collisions>`
- :doc:`Tuning Solver Reference <simulation_tuning_solvers>`
