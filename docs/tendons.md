<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Massless Routed Tendons

```{experimental}
The massless-tendon API (`TendonGuide`, `TendonGuideType`, `TendonGuideFlags`,
`ModelBuilder.add_tendon`, the `Model.tendon_*` arrays, and the XPBD/VBD
`tendon_*` parameters and diagnostics) may change without a deprecation period.
Native routed-tendon USD authoring and MuJoCo lowering below are design sketches,
not implemented APIs. Existing MuJoCo-provider tendon import is supported separately.
```

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

Pass a complete ordered list of {class}`~newton.TendonGuide` entries to
{meth}`~newton.ModelBuilder.add_tendon`. A guide describes the cable's interaction
with a body; it is not an articulation link. Every guide after the first owns
the incoming straight segment, so its `compliance`, `damping`, and `rest_length`
describe that segment. Those three values on the first guide are ignored.

The builder validates the entire route before changing any tendon arrays.
At least two guides are required, all attached to bodies in the same world.
Routes are open and do not implicitly form loops. Internal anchors are allowed;
they prevent material transfer between their incoming and outgoing spans.

```python
import newton

builder = newton.ModelBuilder()

anchor_body = builder.add_body(mass=0.0)
guide_body = builder.add_body(mass=0.0)
payload_body = builder.add_body(mass=1.0)

tendon = builder.add_tendon(
    [
        newton.TendonGuide(body=anchor_body, offset=(-1.0, 0.0, 0.0)),
        newton.TendonGuide(
            body=guide_body,
            guide_type=newton.TendonGuideType.PINHOLE,
            offset=(0.0, 0.0, 1.0),
            mu=0.1,
            compliance=1.0e-6,
            damping=0.1,
        ),
        newton.TendonGuide(
            body=payload_body,
            offset=(1.0, 0.0, 0.0),
            compliance=1.0e-6,
            damping=0.1,
        ),
    ]
)

# VBD requires tendon-connected bodies to receive different graph colors.
builder.color()
model = builder.finalize()
```

A negative `rest_length` measures the initial segment rest length from the
authored body poses. Explicit non-negative values use SI units of metres.
The returned index addresses `model.tendon_start`: a tendon's guides occupy
`[tendon_start[tendon], tendon_start[tendon + 1])` after finalization. Its
segment range starts at `tendon_start[tendon] - tendon` and has one fewer entry
than its guide range.

`add_builder`, `add_world`, and `replicate` copy guide and segment data and remap
body indices. Local offsets and axes follow the copied body transforms;
fixed-joint collapse transforms them into the retained body's frame.

## Guide types

| Type | Behavior |
| --- | --- |
| `ANCHOR` | Fixes the cable to a body-local point. |
| `PINHOLE` | Routes the cable through a body-local point. Material can transfer between the adjacent spans according to the local bend angle and friction coefficient. |
| `ROLLER` | Routes the cable around a circular roller. Tangent points and wrap length are updated from the current body poses. |

For a roller guide, `offset` is the roller center, `axis` is the roller axis,
`radius` is its contact radius, and `orientation` selects the winding side
(`1` or `-1`). The friction coefficient `mu` limits the adjacent tension
ratio using the capstan relation

```text
T_tight / T_slack <= exp(mu * theta)
```

where `theta` is the current wrap angle. A zero coefficient permits free slip;
larger values increasingly resist material transfer.

## Dynamic routing

Setting `dynamic=True` on an internal roller guide lets the solver activate or
bypass that roller from the accepted route geometry at the start of each time
step. `orientation` determines which signed side of the neighboring bypass
span engages the roller. `tendon_activation_tol` on the solver adds a
radius-relative activation gap to prevent state chattering near tangency.

Dynamic routing currently supports isolated rolling candidates only.
Consecutive dynamic roller guides, endpoint candidates, and route reordering
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

### Reading tension (experimental diagnostics)

The solver-owned arrays `tendon_seg_material_tension` and
`tendon_seg_damping_tension` report the constitutive and signed damping
components after a step. The instantaneous unilateral axial tension is
`max(material_tension + damping_tension, 0)`. VBD does not solve XPBD multipliers, so
`tendon_seg_lambda` is zero on the VBD path.

These arrays are debugging/analysis outputs, **not a stable solver-neutral
observation API** on {class}`~newton.State` or {class}`~newton.Model`. Read them
after `step`; they are not prescribed forces or actuator inputs. For a dynamic
route, use `solver.tendon_seg_active` to filter bypassed segment slots and
`solver.tendon_seg_active_guide_l/r` to identify the current span endpoints.
Calling `.numpy()` synchronizes the device and belongs outside captured steps.

## Choosing rods or tendons

Use a massless tendon for a prescribed route through rigid-body guides when
cable inertia, bending stiffness, and free cable contact are negligible. Use
{class}`~newton.Rod` with {meth}`~newton.ModelBuilder.add_rod` when the cable's own
shape and inertia matter, including bending, twisting, sagging, and contact along
its length. A tendon does not exert a bending force while slack or create new
guides when it touches arbitrary geometry. A roller's radius describes its
routing geometry, not the cable thickness.

## Native USD authoring sketch (not implemented)

Keep one stable prim per tendon, for example `/World/Tendons/DriveCable`, so
future actuators and observations can target the tendon itself. Give each guide
its own child prim; this also permits multiple distinct guides on the same body.
An ordered relationship on the tendon prim refers to these guide prims, never
to transient integer body indices. A possible property layout is:

| Owner | Proposed data |
| --- | --- |
| Tendon prim | Ordered `newton:tendon:guides` relationship to guide prims. |
| Guide prim | `body` relationship; type token `anchor`, `pinhole`, or `roller`; body-local offset and axis. |
| Roller guide | Radius, orientation, friction coefficient, dynamic flag; optional source-geometry relationship. |
| Pinhole guide | Friction coefficient. |
| Guide after the first | Incoming span rest length, compliance, and damping, matching `TendonGuide`. |

Lengths and material units need conversion from stage units at import. A geometry
relationship would identify the source shape; it must not silently imply support
for arbitrary collider profiles. Import should resolve body relationships, preserve
authored guide order, and call `add_tendon` once with the complete validated route.
Neither the property names nor a new schema are registered by this implementation.

Newton's MuJoCo provider already imports both fixed and spatial `MjcTendon`
prims into provider-specific `model.mujoco.tendon_*` data. Register
{meth}`~newton.solvers.SolverMuJoCo.register_custom_attributes` before
{meth}`~newton.ModelBuilder.add_usd` to enable this path:

- Fixed tendons use `mjc:path`, optional `mjc:path:indices`, and `mjc:path:coef`
  for weighted joint coordinates.
- Spatial tendons (also the default when `mjc:type` is unauthored) use an ordered,
  optionally indexed path through sites and sphere/cylinder wrapping shapes.
  Pulley branches and wrapping side sites are supported.
- MuJoCo actuator targets referring to tendon prims resolve to those imported
  provider tendons, not to native routed tendons.

See {ref}`MuJoCo tendons in the USD importer <usd-mujoco-tendons>`
for the spatial path attributes. This representation must remain supported. A
future native schema should share compatible tendon identity and authoring
concepts without interpreting joint coefficients as geometric guides or
discarding provider-specific semantics. Importing a MuJoCo tendon does not
create a native `TendonGuide` route for XPBD or VBD.

## MuJoCo compatibility

There is currently **no automatic lowering** between native `Model.tendon_*`
data and `model.mujoco.tendon_*`. Existing MJCF/provider tendons remain a separate
path; choosing `SolverMuJoCo` does not translate a native routed tendon.

The following is a prospective mapping, not a claim of identical solver results:

| Native feature | Potential MuJoCo mapping | Fidelity |
| --- | --- | --- |
| End anchors and frictionless pinholes | Spatial-tendon sites | Exact point geometry. |
| Circular roller in a fixed plane | Cylinder wrapping geom, with side selection | Conditional geometric match; routing conventions need validation. |
| Per-span compliance and damping | Whole-tendon stiffness, damping, and spring length | Approximation; no general one-to-one material-state mapping. |
| Guide capstan friction and rolling material transport | No equivalent in tendon `frictionloss` | Unsupported as an exact mapping. |
| Dynamic routing and activation gap | Optional geom wrapping | Different switching rules; not exact. |
| Internal anchors | Separate tendon paths | Requires splitting the native route. |
| Fixed joint-coordinate tendons or pulley divisors | Existing MuJoCo provider path | Outside this native guide model. |

MuJoCo's spatial routes use sites and sphere/cylinder wrapping geoms; successive
wrapping geoms require intervening sites. Its pulley element branches a path—it
is not a native circular roller. See the [MuJoCo tendon reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#tendon)
and [wrapping overview](https://mujoco.readthedocs.io/en/stable/modeling.html#muscles).
A future converter should start with the compatible frictionless geometry subset
and reject unsupported semantics rather than silently discard them.

## Current limitations

- Tendons are open ordered routes; branching and cyclic routes are not
  represented.
- Roller guides use circular profiles.
- Segment compliance applies to the straight free spans. Wrapped arc length is
  included in material transport but is not independently stretchable.
- Free-span rest lengths are limited to at least `1.0e-6 m` during material
  transfer and dynamic-route transitions.
- Fixed roller wraps must remain in `[0, pi]`. Zero wrap is a valid tangent
  boundary; crossing below zero or beyond pi is unsupported and emits a
  diagnostic from the accepted pose. The diagnostic does not repair the route.
- Dynamic candidates cannot change their authored order or attach between
  non-neighboring guides.
- Native tendon data is not imported from USD/MuJoCo.
- Tendon actuation is not part of {class}`~newton.Control`, and tendon
  diagnostics are solver-owned rather than part of {class}`~newton.State`.
- The tendon path is not differentiable.

## References

- Müller, M., Chentanez, N., Jeschke, S., and Macklin, M. [*Cable
  Joints*](https://doi.org/10.1111/cgf.13507). Computer Graphics Forum 37(8),
  2018.
