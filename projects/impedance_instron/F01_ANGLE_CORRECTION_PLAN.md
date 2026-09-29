# F01 virtual-foot conversion and refit plan

Date: 2026-09-23

Status: completed; all conversion, qualification, refit, testing, and reporting criteria verified.

## Objective

Produce a complete F01 right-leg contact simulation and report in which measured
segment orientation, solver coordinates, contact geometry, and rendered shoe
orientation agree throughout the expanded pre-contact and post-contact window.
Keep the existing foot-to-shoe registration fixed. Report angles in degrees.

## Findings that govern the correction

- [Simple_model.mdh](../../data/F01/Simple_model.mdh) defines
  `Right_Virtual_Foot` from floor-projected ankle and forefoot landmarks. These
  landmarks are calibration-only. Dynamic tracking uses
  `RCAL1+RCAL2+RCAL3+RLMAL+RMMAL+RMT1H+RMT5H+RTOE`.
- The same model tracks `RSK` with `RSH1+RSH2+RSH3+RSH4` and defines its
  calibration geometry using knee and ankle landmarks.
- [FullBuild.v3s](../../data/F01/FullBuild.v3s) computes `RVirualFootAngle`
  using `JOINT_ANGLE`, segment `Right_Virtual_Foot`, reference `RSK`.
  ASCII exports also use the spelling `RVirtualFootAngle`.
- HAS describes a virtual-foot frame calibrated against the floor and then
  tracked with the foot. Its angle relative to the shank represents ankle
  motion. With a flat foot it relates to shank inclination, but it is not a
  fixed-ground measurement throughout stance. See
  [Foot and Ankle Angles](https://has-motion.com/wiki/doku.php?id=visual3d:tutorials:kinematics_and_kinetics:foot_and_ankle_angles).
- Three-dimensional joint angles describe relative rotations. Reconstruct
  rotations with the verified axis order and signs before projecting into the
  simulation plane; scalar addition of exported components is insufficient in
  general. See
  [Joint Angle](https://has-motion.com/wiki/doku.php?id=visual3d:documentation:visual3d_signal_types:link_model_based_data_type:joint_angle).

The latest `outputs/impedance_instron/f01_right_ground/` run directly treated
the exported virtual-foot X angle as shoe-to-ground pitch. That interpretation
is invalid. Preserve its artifacts for comparison and identify it as superseded
in the new report. Its optimizer and algebra checks do not establish anatomical
correctness.

## Constraints

- Use the right leg, right contact, and original point/analog clocks.
- Retain the user's right-shoe asset
  `DigitalInstron/puma-fast-r-nitro-elite-3-3d-internal-wt-LR.obj` and the existing
  calibrated digital-shoe bundle. Record hashes and any stale left-side labels.
- Freeze the existing mount translation and static shoe pitch from the selected
  baseline. No registration search, vertical-placement variants, or angle
  offsets fitted to make a screenshot look correct.
- Keep declared provisional inertial assumptions visible. Do not present them
  as subject-specific measurements.
- Prioritize a complete run with contact. Record limit and refinement failures
  separately; do not expand this task into limit tuning.
- Do not force touchdown to 8 degrees: that expectation must first be assigned
  to the correct physical angle and checked against the source data.

## 1. Freeze and audit inputs

- [x] Create a fresh output directory, proposed:
  `outputs/impedance_instron/f01_right_frame_corrected/`.
- [x] Save source, model, processing-script, shoe, registration, and runtime
  hashes alongside the exact reproduction commands.
- [x] Recheck FR3_3 and the existing source-time window of approximately
  0.140–0.490 s. Retain its expanded margins if the native force trace confirms
  the intended right contact, previously approximately 0.186–0.445 s at 50 N.
- [x] Verify contact side using COP proximity to both feet during loaded stance;
  do not rely on a trial-wide side label when successive contacts alternate.
- [x] Audit right hip, knee, ankle, virtual-foot, marker, force, COP, and moment
  channels, units, signs, and timestamps. Distinguish force-plate measurements
  from any exported joint kinetics and record which are actually used in fitting.
- [x] Preserve native force samples and document any interpolation onto solver
  time. Apply treadmill translation consistently; translation must not change
  reconstructed orientations.

Deliverable: input audit with side evidence, source-time bounds, contact events,
channel mapping, assumptions, and immutable registration values.

## 2. Establish the actual segment frames

- [x] Read the full model/export settings, including segment axis modifications,
  active angle sequence, sign changes, filtering, and tracking defaults.
  Commented settings must not be treated as active overrides.
- [x] Define an explicit coordinate table: lab axes, simulation forward/up axes,
  shank frame, virtual-foot frame, carrier frame, and shoe frame. Specify the
  direction of every rotation matrix and the meaning of zero for each angle.
- [x] Prefer existing exported segment rotation matrices if available. Otherwise
  reconstruct calibrated frames from the saved static landmarks and dynamic
  tracking markers using the MDH definitions and proper rigid rotations.
- [x] Use all model-listed tracking markers for the model reconstruction.
  Record missing markers, conditioning, rigid-fit residuals, and tracking method.
  Use a heel-only reconstruction as a sensitivity check, not an assertion that
  it exactly reproduces Visual3D's tracking algorithm.
- [x] Verify reconstruction against all three exported virtual-foot components
  where possible, including static posture and non-planar dynamic frames.
- [x] If Visual3D defaults or tracking behavior cannot be reproduced, state the
  discrepancy and retain a clearly labeled provisional marker reconstruction.
  Specify a fresh segment-orientation export for exact verification; do not
  claim exact Visual3D equivalence without evidence.

For matrices mapping segment-local vectors into lab coordinates, use the
explicit convention `R_relative = R_shank.T @ R_virtual_foot`. Confirm that the
export's decomposition convention matches this definition before using it.
Transport the calibrated forward axis with the full rotation, then compute
ground pitch from its forward and up components in the simulation plane.

Deliverable: frame/convention audit and full-window trajectories for shank
inclination, relative virtual-foot angle, and reconstructed foot ground pitch.

## 3. Correct preparation and fitting coordinates

Primary files:

- `cartesian/prepare_visual3d.py` and `cartesian/visual3d.py`
- `cartesian/data.py`
- `cartesian/fit.py`
- `cartesian/gpu/objective.py` and `cartesian/gpu/adjoint_objective.py`
- `cartesian/run.py` and `cartesian/gpu/engine.py`

Tasks:

- [x] Replace the F01 raw-virtual-X-to-ground interpretation with the audited
  orientation reconstruction. Prevent the known F01 relative signal from being
  silently declared absolute ground pitch. Preserve public compatibility through
  explicit validation/deprecation where necessary.
- [x] Store raw Visual3D joint angles separately from reconstructed ground pitch
  and solver ankle coordinates, with source/convention metadata for each.
- [x] Resolve any mismatch between the projected shank direction and the shank
  implied by the current thigh-plus-exported-knee mapping. Derive the planar
  chain consistently and quantify differences from the 3D clinical angles.
- [x] Invert the existing carrier/shoe transformation to obtain solver ankle
  coordinates from reconstructed ground pitch, applying the fixed shoe-frame
  transform exactly once. Verify the current relationship in mechanics and
  rendering before relying on it:

  `shoe_pitch = q_thigh + q_knee + q_ankle + pi/2 - shoe_static_pitch`.

- [x] Generate angular velocities after conversion and continuous-angle
  unwrapping. Validate shapes, units, finite values, and frame metadata.
- [x] Keep CPU objective, GPU objective, and adjoint gradients consistent with
  the selected measured quantity. Fit independently reconstructed ground pitch
  when appropriate; retain the raw relative ankle signal as a separate audit.

Deliverable: corrected reference bundle and a deterministic preparation command.

## 4. Prove conversion correctness before training

Use focused `unittest` cases through `uv`; add no dependencies.

- [x] Flat foot with inclined shank: changing relative ankle angle must not
  spuriously rotate a foot whose world orientation stays fixed.
- [x] Rotate shank and foot together: relative rotation stays fixed while ground
  pitch changes correctly.
- [x] Verify neutral calibration, known toe-up/toe-down rotations, and a 3D case
  containing yaw/roll to expose scalar-angle composition errors.
- [x] Verify fixed registration is applied once and translation does not alter
  orientation. Exercise the actual contact and renderer transformations.
- [x] Check CPU/GPU loss agreement and relevant adjoint gradients against finite
  differences for the corrected mapping.
- [x] Demonstrate the semantic regression fails under the old direct-angle
  mapping and passes after correction. Existing algebra-only tests are insufficient.
- [x] Replay measured kinematics without fitting at touchdown, 25%, 50%, 75%,
  toe-off, and the reported problematic frame near source time 0.370 s.
- [x] Compare reconstructed axes, model-tracked markers, the heel-cluster check,
  contact geometry, and rendered axes throughout the window. Report errors and
  tracking uncertainty; do not hide disagreements with an added pitch offset.

Gate: synthetic rotations and transform round trips must pass numerical
tolerances established in the tests. Measured-data disagreement must be
quantified and either resolved or explicitly identified as a provisional
reconstruction limitation before its fit is interpreted.

## 5. Refit and replay the complete contact

- [x] Freeze runtime code and source hashes before qualification and training.
- [x] Build a fresh baseline from the corrected reference and fixed shoe bundle;
  regenerate controls/initial conditions rather than resuming the invalid fit.
- [x] Run CPU/CUDA and contact-replay qualification required by the current
  pipeline, including the mixed-world check where required.
- [x] Run the full 200-iteration fit used for the previous comparison, recording
  optimizer configuration, initial/final losses, and all completion statuses.
- [x] Replay the entire expanded window and compare measured versus simulated
  force, contact onset/release, knee motion, and foot ground pitch.
- [x] Run the existing timestep refinement check once. Report a limit-related
  failure separately from conversion validation and run completion.

Deliverable: complete reproducible run, saved trajectories, contact/spring replay,
fit metrics, and an explicit distinction between completion and acceptance.

## 6. Publish the corrected report

Primary files: `cartesian/report.py`, `VISUAL3D_INPUTS.md`, and new run artifacts.

- [x] Label the three angles separately: shank inclination, virtual-foot angle
  relative to shank, and shoe pitch relative to ground. Show degrees throughout.
- [x] Display source time, window-relative time, and percentage of measured
  contact. Identify true midstance from contact events.
- [x] Render frame axes with labels identifying whether they represent the
  virtual-foot reference or intrinsic shoe geometry. Compute displayed pitch
  from the same transformed axis used to validate the geometry.
- [x] Provide measured-kinematics and fitted-motion views, with touchdown,
  midstance, toe-off, and the previously problematic frame as saved comparisons.
- [x] Include input provenance, right-contact evidence, unchanged registration,
  angle-reconstruction evidence, fit/contact errors, and provisional assumptions.
- [x] Explain why the earlier +24-degree toe-up rendering was invalid and link
  the superseded run without presenting its fit score as physical validation.

Final artifacts:

- `REPORT.md` and interactive `fit/report.html`
- Input/frame audit and conversion-regression evidence
- Full-window angle and force plots; key-frame images
- Corrected reference, fit parameters, replay data, and reproduction commands

## Completion criteria

- [x] The selected contact and all leg-specific inputs are verified as right-side.
- [x] Relative joint angle is never mislabeled or directly consumed as ground pitch.
- [x] Model frame conversion is supported by source definitions and independent
  checks, with any reconstruction approximation clearly stated.
- [x] Reference, solver, contact, and renderer transformations agree through the
  complete window, including the reported problematic frame.
- [x] Shoe asset and registration match the frozen baseline.
- [x] A full refit and contact replay are delivered with honest completion,
  acceptance, and refinement results.
- [x] The report uses degrees and permits inspection of both measured and fitted
  orientations at the same source time.
