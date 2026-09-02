<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Massless Routed Tendons

Newton's tendon model represents a massless cable routed through points on
rigid bodies. It complements Newton's rod-based cable model: rods discretize
the cable itself into massive bodies, while a tendon stores only its route and
applies forces to the bodies along that route.

The route geometry follows Müller et al., *Cable Joints* (SCA 2018), and the
Newton tendon model supports finite capstan friction at circular rollers and
pinholes. Both {class}`~newton.solvers.SolverXPBD` and
{class}`~newton.solvers.SolverVBD` consume the same tendon model and mutable
routing state.

## Model construction

Call {meth}`~newton.ModelBuilder.add_tendon` before adding an ordered sequence
of links with {meth}`~newton.ModelBuilder.add_tendon_link`. Every link after
the first creates the segment ending at that link, so its `compliance`,
`damping`, and `rest_length` arguments describe that segment.

```python
import newton

builder = newton.ModelBuilder()

# Create anchor_body, guide_body, and payload_body first.

builder.add_tendon()
builder.add_tendon_link(
    body=anchor_body,
    link_type=newton.TendonLinkType.ATTACHMENT,
    offset=(0.0, 0.0, 0.0),
)
builder.add_tendon_link(
    body=guide_body,
    link_type=newton.TendonLinkType.PINHOLE,
    offset=(0.0, 0.0, 0.0),
    mu=0.1,
    compliance=1.0e-6,
    damping=0.1,
)
builder.add_tendon_link(
    body=payload_body,
    link_type=newton.TendonLinkType.ATTACHMENT,
    offset=(0.0, 0.0, 0.0),
    compliance=1.0e-6,
    damping=0.1,
)

# VBD requires tendon-connected bodies to receive different graph colors.
builder.color()
model = builder.finalize()
```

A negative `rest_length` measures the initial segment rest length from the
authored body poses. Explicit non-negative values use SI units of metres.

## Link types

| Type | Behavior |
| --- | --- |
| `ATTACHMENT` | Fixes the cable to a body-local point. |
| `PINHOLE` | Routes the cable through a body-local point. Material can transfer between the adjacent spans according to the local bend angle and friction coefficient. |
| `ROLLING` | Routes the cable around a circular roller. Tangent points and wrap length are updated from the current body poses. |

For a rolling link, `offset` is the roller center, `axis` is the roller axis,
`radius` is its contact radius, and `orientation` selects the winding side
(`1` or `-1`). The friction coefficient `mu` limits the adjacent tension
ratio using the capstan relation

```text
T_tight / T_slack <= exp(mu * theta)
```

where `theta` is the current wrap angle. A zero coefficient permits free slip;
larger values increasingly resist material transfer.

## Dynamic routing

Setting `dynamic=True` on an internal rolling link lets the solver activate or
bypass that roller from the accepted route geometry at the start of each time
step. `orientation` determines which signed side of the neighboring bypass
span engages the roller. `tendon_activation_tol` on the solver adds a
radius-relative activation gap to prevent state chattering near tangency.

Dynamic routing currently supports isolated rolling candidates only.
Consecutive dynamic rolling links, endpoint candidates, and route reordering
are not supported.

## Solver controls

Both tendon-capable solvers expose these controls:

- `tendon_max_sweeps`: maximum capstan material-relaxation sweeps per solver
  iteration.
- `tendon_settle_tol`: relative tension-change tolerance for stopping those
  sweeps early.
- `tendon_activation_tol`: radius-relative activation gap for dynamic
  rollers. The default is `2.0e-3`.

XPBD solves tendon stretch and rolling slip as constraint rows. VBD evaluates
the same routed material state as body force and positive-semidefinite local
Hessian contributions. VBD approximates zero segment compliance with
`1.0e-8 m/N`.

XPBD applies `joint_linear_relaxation` to tendon stretch and rolling-slip
corrections. As with other parallel XPBD constraints, the value may need
adjustment for strongly coupled routes.

The solver-owned arrays `tendon_seg_material_tension` and
`tendon_seg_damping_tension` report the constitutive and signed damping
components after a step. The instantaneous unilateral axial tension is
`max(material_tension + damping_tension, 0)`. VBD does not solve XPBD multipliers, so
`tendon_seg_lambda` is zero on the VBD path.

## Current limitations

- Tendons are open ordered routes; branching and cyclic routes are not
  represented.
- Rolling links use circular profiles.
- Segment compliance applies to the straight free spans. Wrapped arc length is
  included in material transport but is not independently stretchable.
- Free-span rest lengths are limited to at least `1.0e-6 m` during material
  transfer and dynamic-route transitions.
- Fixed rolling routes must remain within the supported winding range. A route
  that crosses through a zero/negative wrap emits a diagnostic and is not
  physically valid.
- Dynamic candidates cannot change their authored order or attach between
  non-neighboring links.
- Model-builder merging and replication reject tendon-bearing source builders;
  tendon data is also not imported from USD/MuJoCo.
- Tendon actuation is not part of {class}`~newton.Control`, and tendon
  diagnostics are solver-owned rather than part of {class}`~newton.State`.
- The tendon path is not differentiable.

## References

- Müller, M., Chentanez, N., Jeschke, S., and Macklin, M. [*Cable
  Joints*](https://doi.org/10.1111/cgf.13507). Computer Graphics Forum 37(8),
  2018.
