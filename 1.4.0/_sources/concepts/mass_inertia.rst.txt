.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _Mass and Inertia:

Mass and Inertia
================

Every dynamic rigid body in Newton needs positive mass and a physically
meaningful inertia tensor for stable simulation.  Newton provides three ways to
assign these properties:

1. **Direct specification** on the body via :meth:`~newton.ModelBuilder.add_link`.
2. **Density-based inference** from collision shapes added with
   :class:`~newton.ModelBuilder.ShapeConfig`.
3. **File import** (USD, MJCF, URDF), where mass properties are parsed from
   the source format and mapped to Newton's internal representation.

For the distinction between static, kinematic, and dynamic bodies see
:ref:`Articulations`.


Best practices
--------------

.. tip::

   **Dynamic bodies should have positive mass.**
   If a body has no shapes with density, set ``mass`` and ``inertia``
   explicitly on :meth:`~newton.ModelBuilder.add_link`.

.. tip::

   **Use** ``is_kinematic=True`` **for prescribed motion** — do not rely on
   zero mass to make a body immovable.  See :ref:`Articulations` for details.

.. tip::

   **Prefer density-based inference** when possible.  Letting Newton compute
   mass and inertia from shape geometry keeps mass properties consistent with
   collision geometry and avoids manual bookkeeping.

.. tip::

   **Use** ``lock_inertia=True`` **to protect hand-specified mass properties**
   from subsequent shape additions.  This is the mechanism the MJCF importer
   uses when an ``<inertial>`` element is present.

.. tip::

   **Check finalize warnings.**  Set ``validate_inertia_detailed=True`` on
   :class:`~newton.ModelBuilder` during development to get per-body warnings
   for any mass or inertia values that were corrected.


.. _Specifying mass and inertia:

Specifying mass and inertia
---------------------------

Direct specification on the body
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~newton.ModelBuilder.add_link` accepts the following inertial
parameters:

- ``mass`` — body mass [kg].  Defaults to ``0.0``.
- ``inertia`` — 3x3 inertia tensor [kg m\ :sup:`2`] relative to the center
  of mass.  Defaults to the zero matrix.
- ``com`` — center-of-mass offset [m] from the body origin.  Defaults to the
  origin.
- ``armature`` — artificial scalar inertia [kg m\ :sup:`2`] added to the
  diagonal of the inertia tensor.  Useful for regularization.

These values serve as the initial mass properties of the body.  If shapes with
positive density are subsequently added, their contributions are accumulated on
top of these initial values (see below).

Automatic inference from shape density
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a shape is added via one of the ``add_shape_*()`` methods with
:attr:`ShapeConfig.density <newton.ModelBuilder.ShapeConfig.density>` > 0, Newton
automatically computes mass, center of mass, and inertia tensor from the shape
geometry and accumulates the result onto the parent body.

The accumulation follows three steps:

1. **Mass**: the shape's mass is added to the body's total mass.
2. **Center of mass**: the body COM is recomputed as the mass-weighted average
   of the existing body COM and the shape's COM (transformed to body frame).
3. **Inertia**: both the existing body inertia and the shape inertia are
   shifted to the new COM using the parallel-axis theorem (Steiner's theorem),
   then summed.

This means multiple shapes on the same body compose additively.

.. testcode:: mass-accumulation

   import numpy as np
   import warp as wp
   import newton

   builder = newton.ModelBuilder()

   # Body with initial mass 2.0 kg
   body = builder.add_link(mass=2.0)
   # Shape adds mass from density: a 1m-radius sphere at 1000 kg/m^3
   builder.add_shape_sphere(body, radius=1.0, cfg=builder.ShapeConfig(density=1000.0))

   sphere_mass = 4.0 / 3.0 * np.pi * 1000.0  # ~4189 kg
   assert abs(builder.body_mass[body] - (2.0 + sphere_mass)) < 1.0

Special cases:

- **Planes and heightfields** never contribute mass, regardless of density.
- **Sites** enforce ``density=0`` and never contribute mass.
- ``lock_inertia=True`` on :meth:`~newton.ModelBuilder.add_link` prevents
  subsequent shape additions from modifying the body's mass, COM, or inertia.
- **Hollow shapes** (``ShapeConfig.is_solid=False``) compute shell inertia
  by subtracting the inner volume's contribution from the outer, using
  :attr:`ShapeConfig.margin <newton.ModelBuilder.ShapeConfig.margin>` as shell thickness.


.. _Mass resolution during file import:

Mass resolution during file import
-----------------------------------

When importing from USD, MJCF, or URDF, each format has its own rules for how
mass properties are authored, inferred, or overridden.  Regardless of format,
shape-derived contributions all flow through the same accumulation logic
described above.

The common pattern across formats is:

- **Explicit inertial data takes precedence** when present (``UsdPhysics.MassAPI``
  for USD, ``<inertial>`` for MJCF/URDF).
- **Shape-based inference** is the fallback when explicit data is missing —
  mass and inertia are computed from collision geometry and density.
- Both MJCF and URDF importers accept ``ignore_inertial_definitions=True`` to
  skip explicit inertial data and always infer from shapes.

For format-specific details, see:

- **USD**: :ref:`usd_parsing` — covers the ``MassAPI`` precedence cascade,
  ``ComputeMassProperties`` callback, and collider density priority.
- **MJCF**: :meth:`~newton.ModelBuilder.add_mjcf` — follows MuJoCo semantics
  where ``<inertial>`` fully overrides geom-derived mass (via
  ``lock_inertia``), and geoms contribute via density when ``<inertial>`` is
  absent.
- **URDF**: :meth:`~newton.ModelBuilder.add_urdf` — uses ``<inertial>``
  directly when present; falls back to collision geometry with default density
  otherwise.  By default visual shapes do not contribute mass; however,
  ``parse_visuals_as_colliders=True`` promotes visual geometry into the
  collider set, making it mass-contributing at ``default_shape_density``.


.. _Validation and correction at finalize:

Validation and correction at finalize
--------------------------------------

During :meth:`~newton.ModelBuilder.finalize`, Newton validates and optionally
corrects mass and inertia properties for all bodies.

Compiler settings
^^^^^^^^^^^^^^^^^

The following attributes on :class:`~newton.ModelBuilder` control validation
behavior:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Setting
     - Default
     - Description
   * - :attr:`~newton.ModelBuilder.balance_inertia`
     - ``True``
     - Fix triangle-inequality violations on principal moments by shifting
       eigenvalues uniformly.
   * - :attr:`~newton.ModelBuilder.bound_mass`
     - ``None``
     - Minimum mass value.  If set, clamps small masses up to this value.
   * - :attr:`~newton.ModelBuilder.bound_inertia`
     - ``None``
     - Minimum inertia eigenvalue.  If set, ensures all principal moments are
       at least this value.
   * - :attr:`~newton.ModelBuilder.validate_inertia_detailed`
     - ``False``
     - When ``True``, uses CPU validation with per-body warnings.  When
       ``False``, uses a fast GPU kernel that returns only the count of
       corrected bodies.

Checks performed
^^^^^^^^^^^^^^^^

The detailed (``validate_inertia_detailed=True``) and fast (default) validation
paths apply the same conceptual checks, but the detailed path emits per-body
warnings.  Work is underway to unify the two implementations.

The following checks are applied in order:

1. **Negative mass** — set to zero.
2. **Small positive mass** below ``bound_mass`` — clamped (if ``bound_mass``
   is set).
3. **Zero mass with non-zero inertia** — inertia zeroed.
4. **Inertia symmetry** — enforced via :math:`(I + I^T) / 2`.
5. **Positive definiteness** — negative eigenvalues adjusted.
6. **Eigenvalue bounds** — all eigenvalues clamped to at least
   ``bound_inertia`` (if set).
7. **Triangle inequality** — principal moments must satisfy
   :math:`I_1 + I_2 \geq I_3`.  If violated and ``balance_inertia`` is
   ``True``, eigenvalues are shifted uniformly to satisfy the inequality.

.. note::

   :meth:`~newton.ModelBuilder.collapse_fixed_joints` merges mass and inertia
   across collapsed bodies *before* validation runs.


.. _Shape inertia reference:

Shape inertia reference
-----------------------

The table below summarizes the mass formula for each shape type when density
is positive.  For the full inertia tensor expressions, see
:func:`~newton.geometry.compute_inertia_shape`.

.. list-table::
   :header-rows: 1
   :widths: 18 35 47

   * - Shape
     - Mass
     - Notes
   * - Sphere
     - :math:`\tfrac{4}{3} \pi r^3 \rho`
     - Hollow: shell inertia (outer minus inner).
   * - Box
     - :math:`8\, h_x h_y h_z\, \rho`
     - Half-extents.
   * - Capsule
     - hemisphere caps + cylinder
     - :math:`(\tfrac{4}{3}\pi r^3 + 2\pi r^2 h)\,\rho`
   * - Cylinder
     - :math:`\pi r^2 h\, \rho`
     -
   * - Cone
     - :math:`\tfrac{1}{3} \pi r^2 h\, \rho`
     - Center of mass offset from base.
   * - Ellipsoid
     - :math:`\tfrac{4}{3} \pi a\,b\,c\, \rho`
     - Semi-axes.
   * - Mesh
     - Integrated from triangles
     - Cached on :class:`~newton.Mesh` when available; recomputed from
       vertices otherwise.  Supports solid and hollow.
   * - Plane
     - Always 0
     - Regardless of density.
   * - Heightfield
     - Always 0
     - Regardless of density.

Hollow shapes (``ShapeConfig.is_solid=False``) compute shell inertia by
subtracting the inner volume's contribution, using
:attr:`ShapeConfig.margin <newton.ModelBuilder.ShapeConfig.margin>` as shell thickness.
