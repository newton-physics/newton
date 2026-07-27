# Plan: Hydroelastic Margin and Gap Contact Bands

## Goal

Implement newton-physics/newton#2947 so hydroelastic collision follows
Newton's existing margin-and-gap semantics:

```python
margin_sum = margin_a + margin_b
gap_sum = gap_a + gap_b
effective_separation = real_surface_separation - margin_sum
```

| Effective separation | Result |
| --- | --- |
| `< 0` | Stiffness-bearing hydroelastic contact |
| `0 <= separation <= gap_sum` | Speculative contact candidate |
| `> gap_sum` | No contact |

The lower boundary is inclusive in the speculative band. The outer boundary
is also inclusive, matching Newton's existing gap checks.

A speculative contact must:

- survive narrow phase and appear in the public contact output;
- carry nonnegative margin-relative separation;
- appear in optional hydroelastic contact-surface output;
- produce no response for stationary or separating motion;
- retain the constraint information a compatible solver needs for
  anticipatory response during closing motion.

Speculative means detected but not currently active. The contact still needs a
well-defined stiffness for a solver that activates it later. In particular,
Newton's MuJoCo Warp conversion treats a positive per-contact stiffness as an
override, while zero leaves the geom-level `solref` fallback in place. Using
zero would therefore change the material model at the speculative-to-active
transition rather than simply mean "no force."

Call this value **activation stiffness**, not speculative force or speculative
patch stiffness. It is derived from one selected raw face and carried by its
speculative representative. It is used only if a solver advances that cached
contact into penetration before collision geometry is rebuilt. If collision
geometry is rebuilt before activation, the normal penetrating path replaces
it and this value is never used to produce force.

## Plain implementation contract

The implementation should remain a short data flow:

1. Sample both SDFs in world-distance units.
2. Subtract each shape's margin.
3. Sum the adjusted values to select the contact band.
4. Export the band using the contact-buffer rules below.
5. Preserve the band through contact reduction and existing solver consumers.

Use the same names and definitions in kernels, tests, and documentation. Avoid
epsilon areas, magic stiffness values, hidden sentinels, and duplicated
threshold calculations.

### Pairwise separation

Do not classify a hydroelastic face using twice one shape's SDF depth. With
unequal `kh`, the iso-pressure surface is not halfway between the two shapes.

At each sampled point:

```python
effective_sdf_a = raw_sdf_a_world - margin_a
effective_sdf_b = raw_sdf_b_world - margin_b
pair_effective_separation = effective_sdf_a + effective_sdf_b
```

Use `pair_effective_separation` for:

- contact-band classification;
- the outer gap check;
- speculative contact distance;
- band visualization.

Continue to evaluate each shape's pressure from its own `effective_sdf`.
Interpolate both adjusted SDF values onto each extracted face, then form their
sum. Never reconstruct the pair value as `2 * effective_sdf_a` or
`2 * effective_sdf_b`.

### Contact-buffer fields

Use the sign of the existing contact distance to represent the band. Do not
add a public speculative flag unless a concrete supported consumer cannot use
the sign.

| Field | Penetrating | Speculative |
| --- | --- | --- |
| Band test | `pair_effective_separation < 0` | `pair_effective_separation >= 0` |
| `contact_distance` | Current penetrating solver distance | `pair_effective_separation` |
| Writer `margin_a`, `margin_b` | `0.0`, `0.0` | `0.0`, `0.0` |
| Current pressure force | Included | Zero |
| Contact stiffness | Current penetrating value | Local activation stiffness |

Hydroelastic distance is already margin-relative at export. Passing shape
margins to `write_contact()` would shift the points and subtract the margins
again.

For one raw face under the default linear pressure law, use:

```python
combined_material_slope = (kh_a * kh_b) / (kh_a + kh_b)
activation_stiffness = margin_contact_area * combined_material_slope
```

This coefficient parameterizes possible activation; it does not create force
while contact distance is nonnegative. The deprecated
`margin_contact_area` remains effective for one compatibility window rather
than becoming a no-op in the same release. This value is not the stiffness of
the whole speculative patch and must not be summed or redistributed by
penetrating force/wrench matching.

For a custom pressure callback, the existing callback API does not expose a
boundary tangent. Do not pretend that its declared `kh` is the derivative of
an arbitrary callback. Use the combined declared `kh` only as an explicitly
documented activation regularization until a future API can provide a tangent.
This PR does not claim physical force matching for that custom-callback
transition.

Keep damping and friction metadata, but consumers must apply them only after
normal activation.

### MuJoCo Warp behavior

Newton's MuJoCo converter can copy a speculative contact into MuJoCo Warp's
contact array. It reconstructs the real-surface distance and supplies
`includemargin = margin_a + margin_b`, so MuJoCo Warp evaluates:

```python
constraint_position = contact_dist - includemargin
```

MuJoCo Warp creates a contact constraint only when this value is negative. A
speculative contact with nonnegative margin-relative separation can therefore
exist in the contact array, but it has no `efc` constraint and produces no
force. Current MuJoCo Warp does not use that positive-distance contact for
anticipatory response.

The stiffness is still relevant to continuity across activation:

- when per-contact stiffness is positive, Newton converts it to the contact's
  MuJoCo `solref`;
- when it is zero, MuJoCo retains the geom-level material `solref`;
- after the distance becomes negative, the selected `solref` controls the
  active constraint.

For that reason, speculative hydroelastic contacts should carry the
activation stiffness defined above, even though their current force is
zero. This is a transition regularization for a validated cached
representative, not an aggregate speculative-patch model. Add an integration
test that observes the contact in MuJoCo Warp's contact array while
speculative, confirms it has no constraint address or force, then crosses the
margin and confirms the per-contact stiffness-derived `solref` is used.

## Relationship to #3503 and a future gradient-aware mode

#2947 owns pair separation and contact-band selection. #3503 owns alternative
penetrating stiffness and solver-distance calculations.

This PR must not:

- sample new SDF gradients;
- combine SDF-gradient magnitudes;
- derive stiffness or solver distance from SDF gradients;
- change contact normals using a new gradient rule;
- add a public gradient-mode selector.

Keep the current, non-gradient-aware penetrating calculation behind a small
internal seam:

```python
if pair_effective_separation > gap_sum:
    discard_contact()
elif pair_effective_separation >= 0.0:
    export_speculative_contact(pair_effective_separation)
else:
    stiffness, solver_distance = linearize_hydroelastic_face_current(...)
    export_penetrating_contact(stiffness, solver_distance)
```

`linearize_hydroelastic_face_current()` must not require new SDF-gradient
data. A later PR can add a configuration mode and a second implementation,
such as `linearize_hydroelastic_face_gradient_aware(...)`, without changing
margin adjustment, band selection, speculative export, or reduction rules.

## Non-goals

This work does not:

- implement #3503's alternative penetrating linearization;
- implement an SDF-gradient-aware mode;
- make the experimental hydroelastic API stable;
- change margin or gap behavior for non-hydroelastic contacts, importers,
  inertia, or hollow shapes;
- make gap a compliant thickness, adhesion distance, or source of stationary
  separation force;
- require every solver to support the same anticipatory policy;
- implement continuous collision detection or time-of-impact calculation;
- replace marching cubes or the contact-reduction algorithm;
- broadly retune SDF resolution or pressure laws;
- remove `HydroelasticSDF.Config.margin_contact_area`; it remains available
  and effective during the deprecation period;
- guarantee identical contact count, order, or tessellation;
- add end-to-end hydroelastic support to the private SAP implementation.

Small extraction and reduction changes required to retain speculative faces
are in scope.

## Implementation steps

### 1. Add focused tests first

Add `unittest` coverage to `newton/tests/test_hydroelastic.py`.

Test an isolated band-classification helper at the exact inclusive
boundaries. For end-to-end texture-SDF tests, use just-inside and just-outside
positions separated by more than the interpolation tolerance. Tie tolerances
to voxel size.

Cover:

- penetrating, touching, speculative, outer-boundary, and no-contact states;
- reduced and unreduced output;
- asymmetric margins and gaps;
- unequal `kh`;
- both shape orderings;
- primitive-generated and attached mesh SDFs;
- uniform non-unit scale and a scale-baked nonuniform mesh;
- `gap=None` inheriting `builder.rigid_gap`;
- stationary, separating, and rapidly closing velocities where supported.

Add adversarial reduction cases in which speculative and penetrating faces
share a reduction bin. Verify that speculative output retains the selected raw
face's position, normal, pair separation, geometric area, and activation
stiffness while penetrating force and wrench matching remain unchanged.

Directly assert that:

- pair separation uses both adjusted SDF values;
- speculative `contact_distance` is the pair separation;
- exported writer margins are zero;
- default-law activation stiffness is `margin_contact_area` times the combined
  material slope;
- speculative faces contribute no pressure force;
- damping and friction remain inactive before normal activation;
- reduced and unreduced paths keep the same band;
- speculative reduction retains raw speculative positions and normals, so
  reduction cannot move a speculative representative across either boundary;
- normal matching and synthetic-anchor generation remain penetrating-only;
- penetrating force/wrench values never classify or assign depth to a
  speculative representative;
- activation stiffness remains associated with the selected raw contact and
  is not summed or redistributed as a speculative patch stiffness;
- MuJoCo Warp stores the speculative contact but does not allocate an active
  constraint or produce force;
- MuJoCo Warp uses the per-contact stiffness-derived `solref` after the contact
  crosses into the active band, rather than silently falling back to geom
  material.

Confirm the focused regression fails on the implementation before the fix.
Every new test method must have the required triple-double-quoted docstring.

### 2. Add a verification-only visual example

Use Newton's
[shape-configuration documentation](https://newton-physics.github.io/newton/stable/concepts/collisions.html#shape-configuration)
and
[margin-and-gap illustration](https://newton-physics.github.io/newton/stable/_images/margin_and_gap.svg)
as the visual reference.

Add:

```text
newton/examples/contacts/example_hydroelastic_margin_gap.py
```

The example contains one dynamic hydroelastic sphere above one static
hydroelastic box. Give each shape an explicit, visually large margin and gap,
for example:

```python
margin = 0.10
gap = 0.15
```

Use primitive-generated SDFs with adequate resolution and narrow-band range.
Leave `sdf_padding` unset so the example verifies automatic `margin + gap`
coverage.

Use these phases:

1. Place the sphere above the outer boundary and show no contact.
2. Move it into the speculative band and show the exported candidate.
3. Stop overriding the pose and take a stationary solver step, verifying no
   separating response.
4. Optionally move it upward to visualize a separating speculative candidate.
5. Release it under gravity and observe the transition to active support.

Display viewer-only guides for:

- real geometry;
- the margin-shifted support boundary;
- the outer margin-plus-gap boundary;
- the current contact band and contact surface.

When the viewer supports it, display `real_surface_separation`,
`margin_sum`, `gap_sum`, and `pair_effective_separation`.

The example must:

- use public Newton APIs only;
- follow the `Example` class format;
- implement `test_post_step()` or `test_final()`;
- remain directly runnable;
- be registered in `newton/tests/test_examples.py`;
- not be added to the README gallery;
- not add a gallery screenshot.

Its automated check should observe the three bands in order and verify that
the final real-surface separation is near `margin_sum`, not
`margin_sum + gap_sum`. Do not assert one exact marching-cubes contact count.

### 3. Cover the complete SDF envelope

For automatically generated hydroelastic SDFs:

```python
required_sdf_padding = shape_margin + resolved_shape_gap
```

Apply this to primitive, deferred mesh, cache-key, and secondary SDF
construction paths.

Resolve `gap=None` before SDF construction. It inherits
`builder.rigid_gap`, currently `0.1`, and therefore does not mean zero.

For an explicit `ShapeConfig.sdf_padding` smaller than the required envelope,
prefer an actionable validation error over silently missing contacts.

For an attached mesh SDF, validate its construction padding. If necessary, add
internal host metadata rather than inferring padding from runtime texture
bounds. Tell users how to rebuild it with `Mesh.build_sdf(margin=...)`.

Preserve or enforce the existing scale-baked invariant. Never subtract a
world-space margin from an unscaled local-space SDF value. Reject unsupported
unbaked nonuniform scale with a clear error.

### 4. Carry margins and pair separation through contact extraction

Pass per-shape margins explicitly through:

- hydroelastic launch;
- subblock and voxel refinement;
- marching-cubes corner evaluation;
- face extraction;
- reduced and unreduced export.

At every stage, retain both adjusted SDF values and their pair sum. Keep
one-sided values for pressure evaluation and the pair sum for band selection.

Make pruning conservative over the full voxel or subblock. It must not discard
a valid pair merely because one shape does not individually own a fixed
fraction of `gap_sum`.

### 5. Keep speculative geometry separate from pressure area

Refactor face extraction so these are separate values:

- geometric triangle area;
- penetrating portion of the triangle;
- pair effective separation;
- contact band.

A speculative triangle may have zero penetrating area and still be valid.
Do not keep it alive with an epsilon area.

Write speculative faces to the ordinary contact buffer and optional
`HydroelasticSDF.get_contact_surface()` output. Visual output should carry
nonnegative margin-relative separation.

### 6. Preserve the band through reduction

Do not let speculative candidates contribute current force, moment, or
penetrating tangent-stiffness to reduction totals.

Their current force and wrench are both zero, so ordinary force matching and
wrench matching contain no meaningful information for relocating them. Keep
the implementation simple: a reduced speculative contact must retain the
selected raw face's position, normal, pair separation, and activation
stiffness. Normal matching and synthetic center-of-pressure anchors apply only
to penetrating contacts.

Use a disjoint reduction-key namespace for speculative candidates. This keeps
their geometric selection bounded without letting them compete for
penetrating winner slots or contribute to penetrating aggregates.

This stronger invariant makes it impossible for reduction to move a
speculative representative into the margin regime. Do not clamp its
separation, copy an aggregate penetrating depth onto it, rotate its normal, or
manufacture speculative force. If a future change wants to relocate or rotate
speculative representatives, it must first add a final two-SDF band check and
tests proving the representative still satisfies
`0.0 <= pair_effective_separation <= gap_sum`.

When speculative and penetrating faces coexist, account for their bands
independently. They may share geometric clustering machinery, but speculative
faces must not change the penetrating aggregate force, wrench, selected
penetrating contacts, or matched penetrating stiffness.

This deliberately does not preserve the stiffness or future wrench of the
entire discarded speculative patch. Such a wrench is not a current physical
quantity. The activation stiffness is only a short-lived transition
parameter until collision geometry is rebuilt.

Reduction must not cause a contact to change bands within one collision pass.
Later body motion may legitimately move a cached speculative contact into the
penetrating regime; that is activation due to motion, not a reduction-created
force.

Do not store new SDF-gradient data in reduction buffers. Keep band data
independent so a later gradient-aware mode can replace the penetrating
linearization without changing reduction classification.

### 7. Verify current solver consumers

Inspect:

- semi-implicit and Featherstone contact kernels;
- XPBD;
- MuJoCo contact conversion;
- coupled solvers that copy rigid-contact arrays.

Record a support matrix showing:

- whether each consumer reads per-contact hydroelastic stiffness;
- whether positive distance keeps the contact inactive;
- whether it supports within-step speculative activation;
- which automated test covers it.

For every current hydroelastic consumer, verify stationary and separating
speculative contacts produce no response. Test anticipatory closing response
only for consumers that already support it.

For MuJoCo Warp, record that the current behavior is detected-but-inactive,
not anticipatory. Test all three stages:

1. positive margin-relative distance: present in the contact array, absent
   from `efc`, zero force;
2. zero distance: still inactive because MuJoCo Warp uses a strict negative
   activation check;
3. negative distance: active constraint using the per-contact hydroelastic
   stiffness conversion.

The private `sap_warp` implementation is a semantics reference only. Its
contact Jacobian currently rejects hydroelastic arrays, so production SAP
integration is outside this PR.

### 8. Preserve compatibility and document the change

Do not claim `margin=0.0` alone restores old behavior. A nonzero gap still
enables speculative candidates.

The closest parameter-only legacy geometric configuration is:

```python
shape_cfg.margin = 0.0
shape_cfg.gap = 0.0
```

This preserves the original geometric pressure surface and disables the
speculative band. Call it legacy geometric behavior, not bit-for-bit
compatibility.

Before publishing the guidance, compare pre-change and post-change
penetrating cases for:

- total normal force;
- contact-surface position;
- simulation outcome;
- absence of speculative output when both values are zero.

Explicitly warn that `gap=None` may inherit a nonzero `builder.rigid_gap`.

Keep the public `margin_contact_area` field for configuration compatibility,
and deprecate it without changing its behavior in the same release. Warn when
a caller changes it from its legacy default; warning for every default
configuration would be noisy and would not show that the caller relied on the
setting.

Update collision and hydroelastic documentation with:

```python
pair_effective_separation = (
    raw_sdf_a_world - margin_a
    + raw_sdf_b_world - margin_b
)
has_contact = pair_effective_separation <= gap_a + gap_b
is_penetrating = pair_effective_separation < 0.0
```

Explain that speculative `contact_distance` is geometric and
margin-relative, while penetrating contacts retain the current solver
distance. Add a user-facing `[Unreleased]` changelog entry. Run
`docs/generate_api.py` only if a public symbol is added.

### 9. Check capacity and cost

Speculative faces can increase contact counts. Record before-and-after:

- raw hydroelastic face count;
- reduced contact count;
- contact and reduction buffer usage;
- collision time for the focused boxes and nut-and-bolt example.

No buffer may silently overflow. If growth is excessive, improve conservative
pruning or capacity estimation without changing the band contract.

## Copy-paste verification commands

Run from the repository root.

```bash
uv sync --extra dev --extra examples
```

```bash
uv run --extra dev -m newton.tests -k test_hydroelastic_margin_gap_bands
```

```bash
uv run --extra dev --extra examples -m newton.tests -k test_contacts.example_hydroelastic_margin_gap
```

```bash
uv run --extra examples -m newton.examples hydroelastic_margin_gap --device cuda:0 --viewer null --test --quiet --num-frames 240
```

```bash
uv run --extra dev -m newton.tests -k test_hydroelastic
```

```bash
uv run --extra dev -m newton.tests -k test_sdf
```

```bash
uv run --extra dev --extra examples -m newton.tests -k test_contacts.example_nut_bolt_hydro
```

```bash
uv run --extra dev --extra examples -m newton.tests -k test_robot.example_robot_panda_hydro
```

```bash
uv run --extra examples -m newton.examples nut_bolt_hydro --device cuda:0 --viewer null --test --quiet --num-frames 120 --world-count 1
```

```bash
uv run --extra examples -m newton.examples robot_panda_hydro --device cuda:0 --viewer null --test --quiet --num-frames 720 --world-count 1 --scene pen
```

```bash
uv run --extra examples -m newton.examples robot_panda_hydro --device cuda:0 --viewer null --test --quiet --num-frames 720 --world-count 1 --scene cube
```

```bash
uv run --extra dev -m newton.tests
```

```bash
uvx pre-commit run -a
```

## Required visual checks

Run the dedicated verification scene:

```bash
uv run --extra examples -m newton.examples hydroelastic_margin_gap --device cuda:0 --viewer gl --num-frames 240
```

Confirm:

- no contact outside the outer boundary;
- a visible speculative candidate inside the gap band;
- visible real-geometry separation during speculation;
- no separating response while stationary or moving apart;
- active support at `margin_sum`;
- final separation is not `margin_sum + gap_sum`;
- guides match the published margin-and-gap illustration;
- transitions do not flicker.

Run the nut-and-bolt scene:

```bash
uv run --extra examples -m newton.examples nut_bolt_hydro --device cuda:0 --viewer gl --num-frames 600 --world-count 1
```

Confirm stable thread engagement, rotation, descent, and speculative-to-active
transitions without explosive motion.

Run both Panda scenes:

```bash
uv run --extra examples -m newton.examples robot_panda_hydro --device cuda:0 --viewer gl --num-frames 720 --world-count 1 --scene pen
```

```bash
uv run --extra examples -m newton.examples robot_panda_hydro --device cuda:0 --viewer gl --num-frames 720 --world-count 1 --scene cube
```

Confirm stable grasping, placement, and contact-surface behavior. Record the
device and visual results in the PR verification notes.

## Definition of done

The work is complete when:

- focused band tests fail before the fix and pass afterward;
- exact helper boundaries and tolerance-aware SDF boundaries are covered;
- pair separation uses both adjusted world-space SDF values;
- reduced and unreduced output follows the same bands;
- speculative reduction retains the selected raw speculative position, normal,
  separation and activation stiffness;
- normal matching and synthetic anchors apply only to penetrating contacts;
- reduction never clamps speculative separation or exports it with an
  aggregate penetrating depth;
- speculative reduction does not change penetrating force/wrench matching or
  redistribute activation stiffness as a speculative patch stiffness;
- speculative contacts appear in public and optional surface output;
- stationary and separating speculative contacts produce no response;
- existing speculative-aware consumers retain closing-motion information;
- MuJoCo Warp speculative contacts remain detected but outside `efc`, then use
  their per-contact stiffness-derived `solref` after activation;
- default inherited gap and scaled-shape cases behave as documented;
- legacy geometric behavior passes the before-and-after comparison;
- the verification example observes all three bands and receives visual
  signoff;
- nut-and-bolt and both Panda scenes pass headless and visual checks;
- raw/reduced counts, buffer use, and timing are recorded without overflow;
- no SDF-gradient-aware behavior or public gradient selector is added;
- the internal current-linearization seam can accept a future gradient-aware
  implementation without changing margin/gap classification;
- the focused, hydroelastic, SDF, example, and complete Newton suites pass;
- `uvx pre-commit run -a` passes;
- documentation and changelog use the plain definitions above.

## Current reproduction

The current checkout was tested with two boxes using:

```python
margin_a = margin_b = 0.01
gap_a = gap_b = 0.005
```

It produced contacts only for real overlap. Positive real-surface separations
of `0.005`, `0.015`, and `0.025` meters produced no hydroelastic contacts,
although the first two lie inside the intended inflated or speculative
envelopes. Preserve this as the pre-fix failing reproduction.
