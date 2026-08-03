# LIMX Edge-Face Recovery Stiffness Design

## Goal

Make edge-face untangling materially stronger than vertex-face and edge-edge
proximity support. By default, EF recovery uses three times the shared VF/EE
stiffness.

## API behavior

`ConstraintSelfCollision(..., stiffness=k)` sets VF and EE stiffness to `k`.
When `untangle_stiffness` is omitted, EF stiffness becomes `3 * k`. An explicit
positive `untangle_stiffness` continues to override the default without a
minimum-ratio restriction.

The twist example passes `untangle_stiffness=3.0e4` explicitly beside its
`stiffness=1.0e4` setting so the intended relationship remains visible during
scene tuning.

## Validation

Add a CUDA unit test that constructs `ConstraintSelfCollision` with an omitted
EF override and verifies the default ratio. Preserve the existing validation
and add an assertion that an explicit positive EF override remains unchanged.
Run the focused LIMX self-collision tests and a CUDA headless twist smoke test.

## Compatibility

The constructor signature is unchanged. Only the omitted-argument default
behavior changes; callers that explicitly set `untangle_stiffness` retain their
current behavior.
