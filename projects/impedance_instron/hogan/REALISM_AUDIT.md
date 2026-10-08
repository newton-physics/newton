# Hogan simulation realism audit

Date: 2026-10-07. Scope: current Hogan source, its Cartesian data/shoe
dependencies, the digital-shoe material/contact implementation, all 110 cached
mechanical-run plans, saved learning summaries, and targeted live rollouts.
Existing uncommitted implementation changes were left untouched. No controller
was retrained and no physical parameters were changed on disk.

## Conclusion

The main obstacle is not insufficient gain-search effort. The registered rigid
foot/shoe cannot reproduce the supplied motion and measured contact together.
The optimizer trades motion error against force error to compensate. A large
external pelvis torque, stance-specific inverse dynamics, and unqualified
contact assumptions further prevent interpreting a good fit as a predictive
human/shoe simulation.

The incompatibility is verified. Its upstream allocation among registration,
foot/forefoot representation, and measurement synchronization remains unresolved;
it would be premature to fix it by changing shoe stiffness or translating the
motion until the force curve looks right.

## 1. Reference motion and contact geometry are incompatible

For every cached plan, the reference ankle and absolute foot angle were used
to transform **all** nominal shoe-bottom points into world coordinates. With
the current ground-plane contact law, if their minimum height is positive,
the shoe cannot supply normal ground force.

- **110/110 stances** have positive clearance while measured vertical force
  exceeds 50 N.
- This conflict occupies **80.08 ms per window on average** (range
  74.00–87.50 ms).
- The largest measured force during these separated intervals averages
  **935.44 N** across stances (range 789.61–1024.65 N).
- At measured toe-off, the minimum nominal bottom height averages **23.24 mm
  above ground** (range 20.36–26.86 mm).

Independent direct shoe replays, before running any impedance controller:

| Cached reference | Contact above 50 N | Shoe-replay contact above 50 N | Replay Fz RMSE |
|---|---:|---:|---:|
| FR3_1_eval_000 | 253.66 ms | 154.82 ms | 572.73 N |
| FR3_2_eval_000 | 255.87 ms | 147.75 ms | 603.65 N |

In these same stances, measured COP lies outside even the full projected
shoe-bottom x-range for **19.27% / 17.59%** of samples above 200 N. Maximum
excess is **36.52 / 28.00 mm**. The actually contacting subset is smaller,
so using the whole shoe is a conservative check.

The conflict also exists before Hogan's 12 Hz filtering: using the prepared
ankle and foot pitch directly gives separated-shoe measured-force peaks of
**866.74 / 929.46 N**. Filtering is therefore not its sole origin.

Relevant implementation:

- `../cartesian/prepare_visual3d.py`: the selected dataset uses `sole_markers`;
  heel-to-metatarsal elevation supplies one rigid foot/shoe pitch.
- `registration.py:register_static_height`: one vertical offset from a slow
  static press; no dynamic contact/COP feasibility gate.
- `plan.py:build_plan` and `mechanics.py:Chain.reach`: preserve the measured
  ankle and absolute foot angle while adjusting hip/joint coordinates.
- `../cartesian/shoe.py:Shoe`: one rigid fullfoot-last carrier; no articulated
  toes, longitudinal foot/plate bending, or separate forefoot contact body.
- `../../digital_shoe/runtime.py`: separated nominal bottoms carry no external
  ground force, even with nonzero Maxwell history.

Missing forefoot articulation is a plausible contributor during push-off, but
the audit does not establish that it alone explains the discrepancy. Check the
ankle mount, size/shape, foot pitch, loaded COP, and force/marker clock together.
The shoe artifact identifies a left shoe; the motion dataset selects the right
foot. This is an additional unqualified transfer, not proof of the main cause.

### Force preparation check

For one held-out stance in each trial, saved GRF agrees **exactly** with the
current native Visual3D force exports. All checked FORCE/COFP source hashes
match the dataset manifest. The discrepancy is not caused by an extra Hogan
GRF filter or corrupted saved force arrays.

The manifest's COP-proximity side-assignment intervals average 190.05 ms,
whereas force-threshold contact averages 255.56 ms. These are different event
definitions: the former is a confident side-assignment interval, not a validated
physical toe-off. Do not truncate the force to that interval as an assumed fix.
Original acquisition/filtering/synchronization still needs independent review.

## 2. The floating body receives a substantial artificial torque

`mechanics.py:RestOfBody` rigidly combines trunk, arms, head, and opposite leg.
For this dataset it is **56.426 kg of the 70 kg subject**, with assumed COM
offset 0.19 m and radius of gyration 0.35 m. The leg profile explicitly records
provisional transfer from S001. Pelvis tilt is also used to orient this entire
lump, after a 4 Hz filter.

The pelvis angle is an absolute floating-base coordinate. A generalized actuator
on it is an **external torque on the modeled system**, unlike an internal hip,
knee, or ankle actuator with equal and opposite body reactions.

Across the 110 plans:

- Mean residual pelvis-moment RMS: **106.98 N·m**.
- Mean per-stance peak: **227.09 N·m**; largest peak: **803.45 N·m**.
- By comparison, mean residual x/z force RMS is only **1.05 / 2.29 N**.

The force residual is small partly by construction: `build_plan` integrates
measured GRF into COM motion. That does not remove the moment inconsistency.
`learn.py` optimizes pelvis K/D alongside joint gains, and `Batch` requires the
residual feedforward to remain enabled.

Live mechanical `final_mean` rollouts on FR3_1_eval_000 and FR3_2_eval_000 used
**97.54 / 97.28 N·m RMS** total pelvis torque. Disabling only residual
feedforward still left **50.03 / 55.70 N·m RMS** of pelvis feedback torque.
Removing both residual feedforward and pelvis K/D gave zero pelvis torque and
both windows still completed, but pelvis tracking RMS grew to **0.0540 / 0.0435
rad**. No refit was performed. This isolates how the reported orientation
accuracy is assisted; it does not imply the model inevitably falls without it.

The appropriate remedy is to reconcile external moment balance and represent
the relevant rest-of-body motion, not silently increase pelvis gains or simply
delete the residual without revalidating the model.

## 3. Held-out evaluation is reference-conditioned, not autonomous prediction

`learn.py:load_stances` builds a separate `Plan` for **every** stance, including
evaluation stances. Each plan uses that stance's complete future kinematics,
measured GRF, and measured COP to construct its inverse-dynamics feedforward.
Only the gain schedule is shared.

Consequently the split tests whether shared gains work on unseen,
measurement-conditioned tracking problems. It does not test prediction from
initial conditions alone, nor independent prediction of a new shoe's GRF.
This is a legitimate tracking experiment but a materially narrower claim.

Targets are modified too: the GRF-consistent hip differs from the smoothed
marker hip by mean RMS **8.81 / 6.32 mm** in x/z, with per-stance peak
adjustments averaging **30.07 / 19.35 mm**. IK replaces joint angles. Reported
tracking errors are against this adjusted plan, not the original measurements.
These are separate error quantities and should both be reported, not added.

## 4. Mechanical phase introduces inconsistent trajectory derivatives

`rollout.py` and `batch.py` retime reference position and look up the original
reference velocity and feedforward at that retimed position. They do not
transform velocity/acceleration by the phase rate or recompute inverse dynamics.

For a retimed reference `q_ref(s(t))`, trajectory-consistent derivatives are
`q_s * s_dot` and `q_ss * s_dot**2 + q_s * s_ddot`. Merely sampling original
`v_ref(s)` and `tau_ff(s)` is not the inverse dynamics of that retimed motion.
Alternatively this could be specified as a phase-indexed velocity policy, but
then it should not be described as consistent tracking of one retimed trajectory.

The never-decreasing clamp also does not guarantee continuity. In a live
FR3_1_eval_000 rollout, the reference clock jumped **8.56 ms in one approximately
0.125 ms simulation step** around touchdown. Even excluding rapid jumps,
reference velocity disagreed with the numerical derivative of the retimed
position by **0.70 rad/s knee and 0.68 rad/s ankle RMS** during contact.

Use a bounded, continuous phase transition and explicitly choose/validate the
reference-velocity and feedforward semantics. Multiplying feedforward torque
by a phase factor is not a correct general remedy.

## 5. The objective permits mechanically unrealistic solutions

`learn.py:stance_loss` scores adjusted-position/angle RMSE, GRF RMSE, and,
in the current version, peak Fz error. It does not directly constrain:

- External residual torque or angular-momentum balance.
- Physiological torque, torque-rate, power, activation delay, or joint range.
- COP progression, slip, impact loading rate, or contact duration.
- Whole-cycle periodicity or multi-step stability.

Gain-size and second-difference penalties are not physiological constraints.
Positive K/D and smooth PCHIP interpolation do not establish passive behavior
for moving targets, changing stiffness, and inverse-dynamics actuation.
`Config` explicitly calls its limits numerical screens, not physiological
acceptance limits. Thus `completed` means neither realistic nor validated.

Saved held-out `final_mean` results:

| Quantity | Smooth/time run | Mechanical run |
|---|---:|---:|
| Fz RMSE | 112.80 N | 130.03 N |
| Mean absolute peak Fz error | 105.01 N | 26.92 N |
| Contact above 50 N | 221.87 ms | 223.41 ms |
| Reference contact span | 255.83 ms | 255.83 ms |
| Hip-z tracking RMS | 19.54 mm | 17.58 mm |
| Knee tracking RMS | 0.01659 rad | 0.05580 rad |
| Ankle tracking RMS | 0.06443 rad | 0.07968 rad |

Mechanical-run touchdown is 13.39 ms late and toe-off 18.98 ms early on average.
It fits the peak better without resolving contact duration or overall waveform.
**These are not a controlled phase-only comparison:** the saved smooth run has
no peak-force objective term, while the mechanical run does. Their loss values
must not be compared as if the objective were unchanged.

## 6. Contact calibration does not yet certify gait transfer

The artifact explicitly limits validation to the tested intact-shoe geometry
and conditions, not new rates, temperatures, impacts, or shoes. `Shoe` declares
mu = 0.8 and derives tangential stiffness from foam modulus/area/thickness;
normal Instron loading does not identify outsole shear behavior. A rigid last
on a massless column foundation also does not independently represent foot or
plate bending.

**Do not diagnose zero damping from `normal_damping=0`:** the normal constitutive
law already contains Maxwell viscoelasticity. This artifact has equilibrium
fraction 0.6953 and relaxation time approximately 5.0004 ms. Adding an arbitrary
normal dashpot could double-count dissipation rather than fix the cause.

## 7. Secondary implementation and reproducibility issues

- `Batch` uses each plan's own timestep for rigid dynamics but the first plan's
  timestep for all shoe updates and contact-duration conversion. Current
  timestep spread is only **0.0367%**, so this is real but not an explanation
  for the much larger contact mismatch. Mixed-dt batches need a regression test.
- `load_stances` trusts existing plan caches solely by filename. Changing
  filters, registration, source data, dt, or model parameters can reuse an old
  plan without a fingerprint check. Empty registration summaries in the
  inspected runs reflect cache reuse; stale physics was not proven here.
- `__main__.py` accepts mechanical phase but constructs a time-based `Impedance`
  without `phase_gains`; `simulate` rejects it with an explicit ValueError.
  The learning/report path uses `PhaseSchedule` and works. This is CLI wiring,
  not the reason the saved mechanical learning run is unrealistic.
- The roughness penalty uses unscaled second differences on uneven phase
  knots. It regularizes table entries, not physical curvature or torque slew.
- The report's sample reproduction command omits phase and other nondefault
  settings, so it cannot fully reproduce all inspected runs.

## Checks that passed / explanations ruled out

- All **14** tests in `newton.tests.test_impedance_hogan` passed. They cover
  synthetic mechanics, plan, impedance, adaptation, and static rollout behavior,
  not the real-data compatibility problem above.
- No editor errors were reported for the Hogan folder.
- One live mechanical CPU/GPU comparison had maximum terminal-coordinate
  difference **2.54e-9** and Fz-RMSE difference **8.27e-6 N**.
- Frozen-plan half-step refinement on that stance changed Fz by **1.12 N RMS**
  and **2.46 N maximum**, much less than the force-fit discrepancy. This is
  targeted evidence, not qualification of all stances or all settings.
- Contact moment-arm coupling is present: direct point-force projection and
  ankle-wrench projection agreed within **7.11e-14** in a constructed check.
  Adding another COP/COM moment would double-count it.
- Learned report rollouts use the same PCHIP `PhaseSchedule` as the batch;
  the separate linear time-based `Impedance` is not an interpolation mismatch
  on that executed path.
- Scoring on the measurement clock while control uses mechanical phase is not
  automatically a bug. It preserves timing error; scoring only on warped phase
  could conceal the contact-timing failure.

## Recommended order of work

1. **Gate input feasibility before optimization.** Check ground clearance,
   measured load, contact/COP footprint, and acquisition synchronization
   together; display native measurements beside the adjusted plan. Resolve
   registration/foot representation instead of fitting away contradictions.
2. **Close the physical balance.** Quantify permissible residuals and validate
   the rest-of-body COM, inertia, and angular momentum. Add the minimum
   required articulated body/forefoot mechanics, not more virtual support.
3. **Make phase and trajectory semantics consistent.** Add touchdown/toe-off
   continuity and retimed-derivative tests; retain both clock- and phase-based
   diagnostic errors.
4. **Add physical acceptance criteria.** Require contact duration, COP/slip,
   residual loads, torque/power and impulse balance alongside RMSE. Calibrate
   limits from the available subject/experimental evidence rather than choosing
   thresholds just to pass the current fit.
5. **Separate tracking from prediction.** Freeze training-derived control and
   input processing before testing new stances/materials without their measured
   GRF in feedforward. Carry state across steps if sustained running is the goal.
6. **Then retrain and qualify.** Fingerprint plans/configuration, fix mixed-dt
   handling, test CPU/GPU and refinement across representative cases, and compare
   optimizers/phase modes under identical objectives.

Priority: resolve item 1 before another long gain search. At present better
optimization mostly selects a different compromise between incompatible targets.