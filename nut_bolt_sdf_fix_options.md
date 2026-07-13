# Nut/Bolt SDF Fix Options

## Problem

The `nut_bolt_sdf` example regressed for both XPBD and MuJoCo. Compared with
Newton 1.3.0, the nuts rotate and descend more slowly and the manual test
reported jitter while the nuts are partially engaged. The MuJoCo path can also
emit `nefc overflow` when its per-world constraint capacity is too small.

The released baseline is tag `v1.3.0`, commit
`ce11136b3a28390944f7fe5a32801b31d8aa5670`, running Warp 1.15.0. The
immediate pre-contact-reduction comparison point is commit
`9aa625aca2fa36088dc96c1c267df96b04b2c507`, the parent of `7b630e09`
(`Contact reduction improvements (#3184)`).

The rotational-speed metric is mean absolute angular velocity about the bolt
axis while each nut descends from `z=0.060 m` to `z=0.050 m`. Runs use 1,000
frames at 120 Hz. Unless noted otherwise, the comparison uses four worlds.

## Baselines

| Configuration | XPBD [rad/s] | MuJoCo [rad/s] |
| --- | ---: | ---: |
| Newton 1.3.0, stock | 6.745 | 6.681 |
| Newton 1.3.0, canonical odd-substep state handling | approximately 6.75 | 5.767 |
| Current main, stock | 3.229 | 3.454 |
| Parent of #3184 with current dependencies | 6.765 | 6.677 |

The raw Newton 1.3.0 MuJoCo result partly depends on the old odd-substep state
buffer behavior. The canonicalized 1.3.0 result is the appropriate target when
evaluating current main without reinstating that bug.

## Option 1: Tune public gaps per solver

Use solver-specific public `ModelBuilder.ShapeConfig.gap` values and pass the
same value to public `Mesh.build_sdf(margin=...)`:

| Solver | Per-shape gap | 100-world speed [rad/s] | Completion | Peak `nefc` |
| --- | ---: | ---: | ---: | ---: |
| XPBD | 0.3 mm | 6.623 | 100/100 | N/A |
| MuJoCo | 2.0 mm | 6.000 | 100/100 | 447 |

For MuJoCo, use `njmax=600`. No overflow was observed.

Advantages:

- Uses public APIs only.
- Localized to the example.
- Low regression risk for unrelated SDF users.

Disadvantages:

- Requires solver-specific values; no shared gap matched both solvers.
- MuJoCo is sensitive around 2 mm.
- Tunes around the global contact-selection behavior rather than addressing it.
- The MuJoCo result depends on keeping shape gap and SDF cooking margin
  consistent. XPBD is primarily sensitive to the runtime shape gap.

Rejected public settings:

- Zero gap caused ejections and NaNs.
- The existing 5 mm per-shape gap is a combined 10 mm pair gap and produced the
  slow result.
- Friction set to zero did not recover the baseline behavior.

## Option 2: Add a thin tolerance to the new selector's inner tier

Keep the two-spatial-depth selector introduced by #3184, but change its inner
threshold from:

```python
margin_sum
```

to:

```python
margin_sum + wp.static(BETA_THRESHOLD)
```

`BETA_THRESHOLD` already exists in the global reducer and is 0.1 mm. The outer
threshold remains `margin_sum + gap_sum`. This fixed-tolerance form is useful
for establishing the mechanism, but the scale tests below show that it should
not be the final implementation.

Threshold sweep with the original 5 mm shape gap:

| Added inner tolerance | XPBD [rad/s] | MuJoCo [rad/s] | MuJoCo peak `nefc` |
| ---: | ---: | ---: | ---: |
| 0 mm, current behavior | 3.23 | 3.45 | 516 |
| 0.05 mm | 6.85 | 5.52 | 351 |
| 0.10 mm | 6.85 | 5.62 | 342 |
| 0.20 mm | 6.93 | 5.89 | 342 |

At 100 worlds, the 0.1 mm candidate produced:

- XPBD: 6.836 rad/s, 100/100 completed.
- MuJoCo: 5.630 rad/s, 100/100 completed.
- MuJoCo peak `nefc`: 360 with `njmax=600`; no overflow.

Advantages:

- Addresses the regression in the shared contact-selection behavior.
- Retains the new inner/outer selector instead of reverting #3184.
- Reuses an existing reducer tolerance rather than introducing a new constant.
- Allows the example to retain one material configuration for both solvers.

Risks and open questions:

- Changes internal behavior for all globally reduced mesh-SDF contacts.
- A fixed 0.1 mm tolerance may be scale-sensitive.
- Existing unit tests cover selector ordering but do not exercise the nut/bolt
  manifold or verify behavior around the inner threshold.
- The exact mechanism needs confirmation: the current hypothesis is that
  near-surface candidates alternate between inner and outer priority when the
  boundary is exactly zero, degrading manifold coherence.

Focused tests already run against the temporary 0.1 mm candidate:

- `test_sdf_contact`: 1/1 passed.
- `test_contact_reduction_global`: 46/46 passed on CPU and CUDA.

### Option 2 investigation results

The two-tier selector added by #3184 has the following contract:

- Contacts below `margin_sum` are **inner**. They compete for directional,
  max-depth, and voxel slots and always outrank outer contacts.
- Contacts from `margin_sum` to `margin_sum + gap_sum` are **outer**. They can
  provide directional fallbacks but cannot displace inner contacts.
- Contacts outside the combined gap are rejected.

Adding a tolerance therefore changes semantics: it classifies a thin shell
outside the exact margin as inner. The question is whether that shell
represents numerical uncertainty or merely broadens the contact model.

#### Contact-manifold evidence

A 600-frame, four-world XPBD diagnostic enabled Newton's geometric
frame-to-frame contact matcher and reconstructed the selected contact
distances. Results were:

| Metric | Exact zero boundary | Fixed 0.1 mm boundary |
| --- | ---: | ---: |
| Relevant contacts per frame, mean | 465.9 | 553.0 |
| Frame-to-frame matched fraction | 31.1% | 42.9% |
| Mean matched depth change | 34.1 µm | 13.2 µm |
| 95th-percentile matched depth change | 146.7 µm | 46.6 µm |
| Crossings of the active selector boundary | 2.12% | 0.20% |

With the tolerance, 50.6% of selected contacts lay in the narrow positive
`[0, 0.1 mm)` shell. If those contacts were still classified against zero,
12.9% of matched contacts would cross tiers. This supports the hypothesis that
the exact zero boundary is inside the narrow phase's numerical uncertainty and
causes unstable tier membership. The result is not just an increase in
far-gap directional contacts.

The MuJoCo manifold did not match reliably under the diagnostic matcher, so
its matching percentage is not used as evidence. Its dynamic speed recovery,
constraint-count reduction, and agreement with XPBD remain independent
evidence.

#### Scale sensitivity of a fixed tolerance

The geometry, runtime gaps, ground offset, and SDF cooking envelope were scaled
together. At half scale:

| Inner tolerance | Matched fraction | Selector-boundary crossings |
| --- | ---: | ---: |
| Fixed 0.1 mm | 46.9% | 4.75% |
| Scale-relative 0.05 mm | 50.1% | 0.64% |

At twice the scale, fixed 0.1 mm and scale-relative 0.2 mm were materially
similar in this diagnostic. The half-scale result is enough to reject a fixed
world-space constant as the general rule. Both half-scale candidates recovered
similar XPBD rotational speed, so this distinction is about manifold coherence
and generality rather than tuning the example's scalar metric.

#### SDF-resolution-relative tolerance

The nut/bolt texture SDF has world-space voxel widths of approximately
0.087-0.146 mm and voxel radii of 0.075-0.127 mm. The successful 0.1 mm
constant is therefore approximately one SDF voxel radius.

A refined candidate uses the SDF's existing discretization scale:

```python
inner_tolerance = texture_sdf.voxel_radius * min_sdf_scale
inner_spatial_depth = margin_sum + wp.min(inner_tolerance, gap_sum)
outer_spatial_depth = margin_sum + gap_sum
```

The explicit `gap_sum` clamp preserves the selector invariant
`inner_spatial_depth <= outer_spatial_depth`. If the positive gap is smaller
than the SDF uncertainty, the two tiers coincide, which is preferable to
pretending the narrower distinction is numerically resolvable.

For BVH-backed SDF and heightfield paths, the current refined candidate adds no
voxel-derived tolerance. Those paths need separate evidence before assigning a
different numerical scale.

Four-world results for the voxel-relative candidate were:

| Mode | XPBD [rad/s] | MuJoCo [rad/s] | MuJoCo peak `nefc` |
| --- | ---: | ---: | ---: |
| Default reduction | 6.801 | 5.668 | 354 |
| Deterministic reduction | 6.888 | 5.657 | 348 |

At 100 worlds:

| Solver | Mean speed [rad/s] | Completed nuts | Peak `nefc` |
| --- | ---: | ---: | ---: |
| XPBD | 6.834 | 100/100 | N/A |
| MuJoCo, `njmax=600` | 5.629 | 100/100 | 360 |

No `nefc` overflow occurred. The MuJoCo result is within about 2.4% of the
canonical Newton 1.3.0 target, and XPBD is within about 1.3% of its target.

Nonzero-margin probing with a 0.5 mm margin per shape did not reveal a
selector-specific regression: approximately 96.5% of selected contacts were
already inner before adding a tolerance. That large margin changed the
nut/bolt physical fit enough that both versions eventually lost engagement, so
it is not a valid behavioral acceptance case. A small synthetic selector test
should cover the margin boundary before implementation.

Validation of the voxel-relative candidate, now implemented locally on branch
`mzamoramora/fix-sdf-inner-depth`:

- `test_sdf_contact`: 1/1 passed on CUDA.
- `test_contact_reduction_global`: 46/46 passed on CPU and CUDA.
- `contacts_rj45_plug`: 300-frame headless positive-gap mesh-SDF stability
  smoke passed. The stock example cannot run directly with `ViewerNull`
  because it assumes interactive picking attributes, so the diagnostic
  supplied neutral UI state without changing its physics path.

## Option 3: Restore the pre-#3184 centered reducer

Temporarily replacing only the two-spatial-depth selector call with the old
centered reducer on current main produced approximately 6.87 rad/s for XPBD
and 5.72 rad/s for MuJoCo. This proves that contact selection is responsible
for most of the regression.

This is not the preferred fix because it discards the intended inner/outer
selection behavior from #3184 and may reintroduce issues that motivated that
change.

## Option 4: Raise MuJoCo constraint capacity

Set MuJoCo `njmax=600`. This is independent of the motion-quality fix:

- It prevents `nefc overflow` in the tested 100-world configurations.
- It has no effect on XPBD.
- It does not recover rotational speed or manifold quality by itself.

## Options to reject

- Restore legacy odd-substep state swapping: this reinstates a state-buffer
  bug and only makes the raw MuJoCo speed look closer to 1.3.0.
- Set gap to zero: this caused severe instability and NaNs.
- Tune friction alone: this did not materially improve behavior.
- Use one shared public gap: tested values could not match both solvers.

## Current recommendation

For an example-local release fix, option 1 together with `njmax=600` and a
quantitative example regression test remains the lowest-risk change.

For a root-cause Newton fix, prefer the voxel-relative form of option 2 over the
fixed `BETA_THRESHOLD` form. The evidence now covers threshold semantics,
contact-set stability, geometry scale, zero and nonzero margins, deterministic
and non-deterministic reduction, both solvers, 100 worlds, and one additional
positive-gap mesh-SDF example. Before merging, add focused tests for the
voxel-relative threshold and its `gap_sum` clamp, then run the broader example
suite.

## Option 2 investigation plan

Completed investigation items:

1. Defined selector invariants at the inner/outer boundaries.
2. Measured selected-contact stability and counts for the nut/bolt pair.
3. Tested object scale and shape margin; selected a voxel-relative tolerance.
4. Ran focused mesh-SDF/contact-reduction tests and an RJ45 smoke test.
5. Re-ran four-world, deterministic, and 100-world nut/bolt measurements for
   both solvers.

Remaining implementation work:

1. Add a focused unit test that distinguishes an exact zero boundary from a
   voxel-relative inner tolerance.
2. Test that the tolerance is clamped to `gap_sum` and remains zero for paths
   without texture-SDF resolution metadata.
3. Add a quantitative `nut_bolt_sdf` example regression test.
4. Run the broader example suite before changing the shared selector.
