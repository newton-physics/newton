# Contact input correction status

## Decision

**The contact contradiction is not resolved. Default identification must remain
blocked.** The GPU implementation has been tested independently, not used to fit
away the remaining measurement/geometry mismatch.

## Verified correction

`cartesian/prepare_visual3d.py` previously encoded the static ankle-to-metatarsal
endpoint in the measured marker-pitch frame while ground-referenced foot states
used the selected shoe-reference pitch. The endpoint is now expressed in the
state's actual reference frame. Legacy shank/marker conventions remain unchanged.

The checked F01 static reconstructions previously had roughly 21–22 mm vertical
endpoint error. Regression tests reconstruct the original measured vector to
roundoff. This corrects endpoint geometry **only**: mass, bias, ankle position,
shoe column positions, measured force, and contact remain unchanged. Existing
bundles are not silently rewritten; regeneration is required for the correction.

## Why that does not fix contact

Fresh current-code tracking-plan reconstructions exactly matched the checked
cached plans. For FR3_1_train_000 and FR3_2_train_000, all nominal shoe bottoms
remain above the plane under force above 50 N for 75.50 and 81.63 milliseconds.
Maximum loaded clearance is 25.08 and 22.60 millimetres. The contact law correctly
returns no support when separated. Disabling that rule would invent a force.

Checks that do not explain away the mismatch:

- Heel-cluster rigid transport still leaves loaded clearances of roughly
  29–36 millimetres in the inspected variants.
- Changing 3D elevation to sagittal heading changes clearance by at most about
  one millimetre in those cases.
- Removing the extra Hogan smoothing still leaves roughly 20–24 millimetres.
- Tracking-plan IK already places the ankle within about 0.12 millimetres of its
  chosen target in the inspected cases. The generative direct-coordinate path
  intentionally does not use this IK, so its additional fixed-length mismatch
  is now checked independently using actual model FK.

For FR3_1_eval_000, at 0.25 seconds into the window, measured force is about
801 N; the heel-cluster centroid is 85 mm above the laboratory origin and the
toe marker is 24 mm above it. At 0.30 seconds, heel height is 158 mm while the
toe remains at 21 mm, with about 177 N still measured. The single rigid shoe
transform does not establish the same support configuration. Marker height is
not outsole height; subtracting an arbitrary constant is not a calibration.

These observations motivate an articulated forefoot and an independently
calibrated worn-shoe mapping, but they do not uniquely determine a joint axis,
marker offset, shoe bending law, or partition of ground loads.

## Acquisition and registration uncertainties

- The saved `data/F01/FullBuild.v3s` specifies 6 Hz analog filtering, but it was
  edited after the export; the user reports the export used 20 Hz. The exported
  FP1 Fz agrees with 20 Hz, not 6 Hz. Its median peak loading rate is about
  12.9 stance-peaks/s (17.5 kN/s at ~1355 N, 546 stances), and 50 N-to-half-peak
  rise is 43–45 ms. A two-pass 6 Hz zero-phase Butterworth gives 8–10 peaks/s
  and 64–74 ms rise for plausible stances (11.4 peaks/s absolute bound for any
  input bounded by the peak). A 20 Hz filter on an impact-free
  ~250 ms stance gives 12.3 peaks/s and 44 ms. The export is clipped at 0 N, so
  high-frequency spectra are not discriminating
  (`outputs/impedance_instron/force_filter_check_20261008/`). COP is not
  used as a calibration or gating target because treadmill low-force COP is unreliable.
- That script supplies session-specific plate calibration and corners at
  z = -27 mm. This is not justification to translate the model by 27 mm: the
  actual belt surface, laboratory origin, and model floor must be reconciled.
- The saved exporter does not produce all the processed targets, joint centers,
  joint angles, and clocks in the current dataset; the exact export/workspace
  history remains unverified.
- The artifact contains a left Instron last; the selected gait is right-foot.
  A projected sagittal proxy may be useful, but is not measured subject-specific
  registration.
- Static loading supplies one vertical seating condition, not independent
  longitudinal alignment or a dynamic forefoot mapping.

The repository has older unnumbered FR3 C3Ds, but their correspondence to the
numbered F01 source trials has not been established. They must not silently
replace the sources used by the current exports.

## Required to unblock the physical fix

1. The actual numbered F01 static/metabolic C3Ds or the Visual3D workspace and
   complete processing/export pipeline used to create these files, with plate
   calibration, common clocks, analog filtering, and belt-surface origin.
2. The worn shoe's size/side and independently measured correspondence between
   ankle, heel, metatarsal/toe markers, and shoe geometry. A static pose alone
   cannot identify all these mappings.
3. Validate loaded clearance and COP footprint using those mappings; if the
   rigid foot remains incompatible, add a measured/calibrated forefoot degree
   of freedom rather than fitting a time-varying registration to measured force.

No gains or material parameters should be fitted to compensate for these
unknowns. The updated check examines both measured-ankle placement and the
actual model-FK placement. The complete recheck is saved at
`outputs/impedance_instron/contact_recheck_20261007/summary.json`:
**110 references inspected, 110 incompatible**.

## Leg-coordinate IK and COP (2026-10-08)

Generative targets now re-solve hip and knee to the measured ankle. Peak ankle
FK error fell from 26–49 mm to below 5.5 mm. COP is now reported only. With
the original mount (`contact_recheck_nocop_20261008`), all 110 still fail on
loaded clearance: peak 8.3–20.0 mm (median 14.2), 6–12 frames per stance,
median 450 N at the worst frame. Three also touch the ground at the
prediction origin. A constant-registration search was tried and abandoned:
no single rigid placement passed the clearance check.

## Independent GPU verification

GPU rollout and GPU objective tests cover dynamic contact, candidate-specific
bounds, reset/permutation/isolation, exact mixed timesteps, failures, target
isolation, aggregate-only downloads, memory-bounded streaming, and end-to-end
CPU/GPU fit parity. The contact gate is tested to fail **before** GPU evaluator
construction. Real-data GPU comparisons here were frozen rollouts, not training.

The current CLI defaults to CUDA; `--device cpu` retains the CPU reference.
This prepares the compute backend while accurately leaving the physics task
blocked rather than declaring an unsupported contact correction.
