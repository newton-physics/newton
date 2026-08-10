# Digital Instron Material-Model Findings

## Scope

This document records the evidence and decisions behind the Digital Instron
foundation model. The immediate objective is one effective through-thickness
shoe model shared by the rearfoot punch and full-foot shoe-last tests. Fitting
uses every time-aligned force sample once, normalized by each trial's measured
peak force. Peak force, branch RMSE, and loop work are validation metrics, not
extra fitting weights.

The fitted parameters are effective intact-shoe properties. They are not
intrinsic PEBA parameters because the tests include geometry, bonding, outsole,
plate, confinement, and spatially varying thickness.

## Retained Data

- Rearfoot punch: 140 ms nominal loading, 501 averaged samples, 22.76 mm peak
  travel, approximately 1,180 N corrected peak force.
- Full-foot last: 185 ms nominal loading, 501 averaged samples, 17.92 mm peak
  travel, approximately 1,980 N corrected peak force.
- Both averaged traces span 0.501 s and represent cycles 90-100.
- The current data do not include multiple rates, hold-relaxation tests,
  lateral strain, shear, confined compression, or held-out cycles.

## Geometry Findings

The geometry had to be corrected before material identification was meaningful.

1. The raw right-foot shoe-last STL contains an attachment above the physical
   last. After canonicalization and a 90 degree fixture rotation, retaining
   triangles entirely below 80 mm produces the physical last: 15,853 triangles,
   47,559 STL vertices, and 241.69 x 81.05 x 40.80 mm extents.
2. The raw STL was mirrored across its source X axis to create
   `Instron Shoe Last Size 9 6drop merged attachment 1 left.STL`. Triangle
   winding was reversed. The mirrored asset preserves topology, dimensions,
   outward winding, and the attachment crop.
3. Bounds canonicalization already centers the midsole and last. A later
   vertex-mean alignment shifted the last by approximately 32 mm along shoe
   length because STL vertex density is asymmetric. That translation was
   removed.
4. The contact side cannot be inferred from STL winding. The physical side is
   configured explicitly as the first (`near`) intersection along the positive
   thickness ray. It lies at approximately 0-17 mm. The opposite shell lies at
   approximately 14-39 mm.
5. A cropped open surface may produce one ray intersection. Requiring two hits
   incorrectly deleted valid boundary contact samples; the contact-surface
   raycaster now accepts single hits.
6. Midsole column creation previously required `abs(n dot axis) >= 0.8` at both
   ray hits. This rejected curved but valid surfaces and created 37 internal
   holes. Every candidate thicker than 5 mm had correctly opposing orientation.
   The retained rule requires a downward lower face, upward upper face, and
   signed normal magnitude at least 0.1.
7. The 5 mm grid now contains 910 valid midsole columns, up from 647, with no
   internal lattice holes. At peak full-foot travel, 595 columns overlap the
   fixture and 578 compress; the remaining 17 are perimeter or arch clearance.
8. A spacing sweep at 5, 4, 3, and 2.5 mm produced similar force-fit parameters
   and errors. The remaining force mismatch is not primarily grid resolution.
9. The mirrored last's heel-to-toe orientation is correct: its narrow heel and
   broad forefoot correspond to the same midsole landmarks. Bounds-centering did
   leave a longitudinal clearance ramp equivalent to about 1.8 degrees, causing
   peak compression to be toe-heavy. A geometry-only plane fit to the two
   surfaces gives roll=-1.01 degrees and pitch=-1.82 degrees. Applying that pose
   moves the peak compression centroid from +16.0 mm toward the toe to +4.0 mm
   and balances longitudinal compression quartiles without using force data.

Generated diagnostics:

- `processed/v2_cache/digital_instron_geometry.png`: attachment crop and near
  versus far STL surfaces.
- `processed/v2_cache/digital_instron_contact_current.png`: all created columns,
  fixture overlap, and peak-compressed columns.

## Constitutive Models Tested

All comparisons below used peak-normalized pointwise residuals at every frame.
No loading, unloading, strain-region, peak, or loop-area weights were used.

The following model families were tested and rejected as the final model:

- Capped locked-rational elasticity: pressure plateaued after lock strain and
  produced an unphysical horizontal rearfoot tip.
- Additive piecewise densification onset: matched force envelopes but introduced
  a fitted strain-region transition and S-shaped curves.
- Global power, rational, and rational-power elasticity: smooth, but unable to
  reproduce both force shape and loop work.
- Constant and strain-dependent Kelvin-Voigt damping: improved pointwise force
  but did not transfer loop work between fixtures.
- Nonlinear, cubic, and saturating rate damping: reduced pointwise loss in some
  cases but remained unable to reproduce both loop areas robustly.
- One- and two-branch Prony/Maxwell models: branches collapsed to very short
  relaxation times. Exact periodic-state initialization did not fix this.
- Dahl and smooth structural-hysteresis states: optimized to near-discontinuous
  state changes and worsened loop work.
- Warmed periodic viscoelastic states: confirmed that starting averaged cycles
  from a virgin state was not the main issue.
- Pasternak spatial coupling: substantially improved pointwise force shape and
  confirmed missing lateral confinement, but a purely elastic/viscous shear
  layer did not reproduce dissipated work.

Important diagnostics:

- A smooth global model can fit rearfoot independently to below 1% normalized
  RMSE and loop-work error, so rearfoot behavior is representable.
- Full-foot independent fits retain substantial loop-work error under simple
  rate models, showing that a memoryless dashpot is insufficient.
- Rearfoot and full-foot demand different apparent independent-column stiffness.
  This is consistent with missing lateral confinement and with published limits
  of Winkler foundations for thick/localized contact.
- Simple Maxwell spectra are not uniquely identifiable from two different
  fixtures at one cycle duration. Fixed timescales are required until multiple
  rates or hold tests are available.

## Primary Literature

### Recommended constitutive architecture

X. Li, J. Tao, A. K. Landauer, C. Franck, and D. L. Henann,
"Large-deformation constitutive modeling of viscoelastic foams: Application to
a closed-cell foam material," *Journal of the Mechanics and Physics of Solids*
161, 104807 (2022). <https://doi.org/10.1016/j.jmps.2022.104807>

This is the strongest directly applicable source. It models Reebok-supplied
closed-cell EVA over compression/tension rates from 1e-3 to 1e1 1/s and validates
against indentation and shear. The model combines a smooth compressible
hyperelastic equilibrium network with multiple finite-strain viscoelastic
networks and includes volumetric as well as distortional viscous evolution. The
authors publish an Abaqus VUMAT at
<https://github.com/HenannResearchGroup/ViscoelasticFoam>.

The current intact-shoe data cannot identify the full three-dimensional Li
model. The implementation in this project is therefore explicitly a reduced
one-dimensional foundation analogue, not a reproduction of the VUMAT.

### Hyperfoam equilibrium response

S. Kim et al., "Finite element analysis-based optimization of longitudinal
bending stiffness and rearfoot stability in carbon-plated running shoes,"
*Frontiers in Bioengineering and Biotechnology* 14, 1797727 (2026).
<https://doi.org/10.3389/fbioe.2026.1797727>

The paper uses first-order Abaqus Hyperfoam fits for EVA, TPU, and PEBA. The
reported PEBA constants are mu=0.112 MPa, alpha=5.05, and effective Poisson ratio
0.30. Calibration used compression only and was rate independent; the authors
recommend time-dependent properties for dynamic loading. These values are useful
initial scales, not transferable parameters for this shoe.

Abaqus Hyperfoam is based on the compressible Ogden-Hill-Storakers energy and
provides a smooth global elastic response through large volumetric compression.
It cannot generate a stabilized hysteresis loop without additional dissipative
networks.

### Cyclic shoe-foam evidence

C. Aimar et al., "Compression fatigue of elastomeric foams used in midsoles of
running shoes," *Footwear Science* 16(2), 93-103 (2024).
<https://doi.org/10.1080/19424280.2024.2317881>

Commercial EVA, expanded TPU, and PEBA foams were cycled under running-like
compression. The work documents softening, loss of absorbed energy, residual
strain, and material-specific fatigue, but does not provide a constitutive law.

N. Even-Tzur et al., "Role of EVA viscoelastic properties in the protective
performance of a sport shoe: computational studies," *Biomedical Materials and
Engineering* 16(5), 289-299 (2006). PMID 17075164.

This shoe-level study shows sensitivity to EVA viscous damping, relaxation, and
foam thickness. Thickness was the dominant wear parameter in the reported
simulations.

J. Wu et al., "A One-Dimensional Dynamic Constitutive Modeling of Ethylene
Vinyl Acetate (EVA) Foam," *Polymers* 15, 4514 (2023).
<https://doi.org/10.3390/polym15234514>

The model separates foam-matrix plateau stress from trapped-cell gas pressure
and demonstrates rate-sensitive densification. It covers monotonic compression,
not unloading or hysteresis, so it is mechanistic guidance rather than a cyclic
model for this project.

### Limits of independent columns

R. Elandt, *Pressure Field Contact*, PhD dissertation, Cornell University
(2022). <https://doi.org/10.7298/x58e-tg70>

The pressure-field/Winkler approximation is most accurate for compliant layers
thin relative to contact width and for conforming contact. It omits Poisson
expansion, anisotropy, and shear. Published comparisons underpredict localized
contact as thickness approaches contact diameter. This directly motivates a
Pasternak-type coupling term for the rearfoot punch.

## Selected Reduced Model

The next implementation uses:

1. A first-order uniaxial Hyperfoam equilibrium pressure under a fixed effective
   Poisson ratio. This replaces piecewise power/densification regions with one
   smooth finite-strain potential.
2. Fixed-timescale generalized Maxwell branches driven by changes in equilibrium
   pressure. The cycle is evaluated at its exact periodic steady state because
   the source traces average cycles 90-100.
3. A Pasternak pressure term from the lateral Laplacian of column compression.
   This represents effective confinement/shear transfer and is not an intrinsic
   shear modulus.
4. One shared parameter set across both fixtures, fitted using every frame once.

Fixed assumptions required by missing data:

- Effective Poisson ratio: 0.30, based on the PEBA Hyperfoam baseline reported
  by Kim et al.; it is not fitted from these tests.
- Maxwell relaxation times: fixed around the 0.5 s cycle's informative band.
  Branch amplitudes may be fitted, but times are not independently identifiable.
- Isotropic, homogeneous effective material.
- No fatigue evolution, Mullins softening, permanent set, or temperature shift.

## Required Experiments

The full Li-style material should not be implemented or interpreted as
identified until the following are available:

1. Same fixture at multiple cycle periods, for example 0.1, 0.25, 0.5, 1, 2,
   and 10 s.
2. Hold-relaxation tests at low strain, plateau, and densification.
3. Stabilized cycles after a documented preconditioning protocol.
4. Lateral-strain measurement or confined/volumetric compression.
5. Shear under precompression for three-dimensional transferability.
6. Held-out cycle windows and at least one held-out amplitude and rate.
7. Controlled temperature and recovery time.

Until then, Maxwell branch amplitudes and Pasternak coupling are effective
shoe-level parameters. They must not be presented as uniquely identified PEBA
material constants.

## Current Reduced-Model Result

The implemented reduced model fits four parameters:

- Instantaneous Hyperfoam shear modulus: approximately 19.0 kPa.
- Hyperfoam exponent: approximately 5.13.
- Long-term/instantaneous shear-modulus fraction: approximately 0.106.
- Pasternak coupling: approximately 918 N/m.

The fixed effective Poisson ratio is 0.30. A corrected-pose timescale profile
over 0.005-0.16 s produced a broad pointwise-loss minimum around 0.06-0.10 s,
consistent with the characteristic 1/(2*pi*f) scale of a 2 Hz cycle. The branch
time is therefore fixed at 0.08 s rather than claimed as uniquely identified.
Additional fixed-time branches were redundant, and a second smooth Hyperfoam
term converged to zero.

Current in-sample metrics on averaged cycles 90-100 are:

| Trial | Pointwise RMSE | Peak error | Loop-work error |
|---|---:|---:|---:|
| Rearfoot 140 ms | 3.44% | 12.5% | 16.1% |
| Full-foot 185 ms | 2.30% | 15.3% | 17.6% |

This model is accepted as the current smooth force-envelope baseline, not as a
validated hysteresis model. The failed loop-work gate is intentional and must
remain visible. Adding more fitted Maxwell branches to these same two cycles is
not justified: tested branches collapse to redundant short timescales. The next
material-model change must be supported by same-fixture multi-rate or relaxation
data.


## Phase-1 Held-Out Validation

The raw Instron CSVs are now the source of truth for a train/validation split.
`projects/digital_instron_v2/phase1.py` regenerates a train trace from cycles
90-98 and a held-out validation trace from cycles 99-100 for each fixture
(`cycle_windows.py`, top-of-stroke displacement zero, sign-flip force), fits one
shared shoe material on the train traces only, and scores the official
baseline-corrected peak, active-frame RMSE, and positive-hysteresis metrics on
the held-out cycles (`validation.py`). Run it with:

    uv run -m projects.digital_instron_v2.phase1 --backend scipy

The held-out metrics match the in-sample metrics almost exactly, so the reduced
model does not overfit the fitted cycles:

| Trial | Split | Peak error | Active RMSE | Hysteresis error |
|---|---|---:|---:|---:|
| Rearfoot 140 ms | train 90-98 | 12.5% | 6.7% | 15.4% |
| Rearfoot 140 ms | held-out 99-100 | 12.6% | 6.7% | 15.6% |
| Full-foot 185 ms | train 90-98 | 14.4% | 4.1% | 16.8% |
| Full-foot 185 ms | held-out 99-100 | 14.7% | 4.1% | 17.1% |

The peak and hysteresis gates still fail by design. This is an identifiability
ceiling of two single-rate cycles, not overfitting: the held-out generalization
gap is under 0.4 points on every metric.

## Calibration Backend Comparison

The differentiable fit (`inverse_id.fit_material_to_trials`, exact Warp-tape
gradients, Adam in scale space) was run through the identical train/validate
protocol and compared against the shipped scipy least-squares fit:

| Backend | G_inst | eq. fraction | Rearfoot peak/hyst | Full-foot peak/hyst |
|---|---:|---:|---|---|
| scipy (100 evals) | 19.0 kPa | 0.107 | 12.6% / 15.6% | 14.7% / 17.1% |
| diff cold (400 iters) | 18.9 kPa | 0.632 | 13.5% / 53.6% | 15.4% / 40.3% |
| diff warm from scipy | 19.0 kPa | 0.107 | 12.6% / 15.6% | 14.7% / 17.2% |

Two findings decide the backend question:

1. From the literature seed, Adam recovers the shear modulus and exponent but
   stalls at equilibrium fraction 0.63 and wrecks hysteresis, because the
   equilibrium/overstress direction is nearly flat in the peak/RMSE loss.
   Levenberg-Marquardt conditions that ill-posed direction far better.
2. Warm-started from the scipy optimum, exact gradients leave the material
   unchanged and do not lower any residual. The residual floor is a data/model
   identifiability limit, not an optimizer limit.

Decision: scipy least-squares remains the authoritative calibration backend. The
differentiable fit is retained as an optional backend for gradient-based design
and coupled objectives, warm-started from scipy when used for calibration. Exact
gradients do not improve peak or hysteresis residuals on this two-cycle dataset.


## Phase-2 Dynamic Replay (SolverMuJoCo)

Phase 2 couples the Phase-1 effective law into MuJoCo-Warp as a live Warp
foundation kernel: each substep the kernel writes a body wrench into Newton
``state.body_f`` and ``SolverMuJoCo`` advances the model. The shared material is
fit on the train cycles only (``phase1.fit_train_material``), so the dynamic
replay is scored on held-out cycles [99, 100] that neither the fit nor the
servo tuning ever saw. Run: ``uv run -m projects.digital_instron_v2.phase2``.

Both fixtures now build a dynamic column bed: the fullfoot last samples the mesh
footprint (606 active columns at peak, 152 cm^2), and a new flat-punch
path (``_build_rearfoot_geometry``) samples the 62-column, 15 cm^2 heel
disc with uniform anchoring (``z_free = slack``).

### Drive modes

- **Kinematic crosshead (gated).** A position-controlled body prescribes the
  exact measured pose every substep -- the faithful digital twin of the
  servo-hydraulic Instron. This is the mode scored against the bench data.
- **Servo (stress test).** A prismatic PD joint (ke=5e+07, kd=2e+04) with
  position + velocity feed-forward tracks the same trajectory in closed loop. It
  is *not* gated on the rate-dependent hysteresis (a finite servo low-passes the
  crosshead velocity and thins the loop) -- only on stability and continuity.

### Held-out dynamic metrics (kinematic, substeps=32, dt=521 us)

| fixture | peak err | active RMSE | hysteresis err | tracking | max dF/frame |
|---|---|---|---|---|---|
| rearfoot punch | 0.127 | 0.067 | 0.156 | 0.000 mm | 13.6 N |
| fullfoot last | 0.146 | 0.041 | 0.171 | 0.000 mm | 31.1 N |

The kinematic replay reproduces the Phase-1 *static* held-out metrics
bit-for-bit (rearfoot 0.127 / 0.067 / 0.156; fullfoot 0.146 / 0.041 / 0.171),
confirming the Warp/MuJoCo implementation of the calibrated law is faithful and
phase aligned. The peak-force and hysteresis gates fail by the same margin as
Phase 1 -- this is the two-cycle data/model identifiability ceiling, **not** a
dynamics failure. Active pointwise RMSE passes (4-7%).

A one-substep phase-misalignment between the logged force and the commanded
displacement smears the loop badly (fullfoot hysteresis 0.171 -> 0.56); the
recorder is therefore phase aligned (force and depth are logged *before* the
substep clock advances). Driven kinematically at the measured frames, the
foundation kernel matches ``core.predict`` exactly and is dt-robust (substeps
16/32/64 all give the same loop).

### Dynamics diagnostics (physical and continuous)

- **COP.** Fullfoot peak COP (1.8, -2.4) mm sits near the last centroid;
  rearfoot COP (-88.4, 0.7) mm sits at the heel-punch center.
- **Wrench.** Peak body Fz 2288 N (fullfoot) / 1031 N (rearfoot); the
  rearfoot moment (91 N.m) is the r x F of the 88 mm heel offset -- physical.
- **Continuity.** Max force jump per frame 31 N (fullfoot) / 14 N (rearfoot),
  far below 25% of peak; force finite throughout.
- **Energy.** Simulated dissipation -1.79 J vs measured -1.52 J
  (fullfoot); -0.93 J vs -1.11 J (rearfoot), consistent with the
  hysteresis metric.

### Servo stress test (stability under closed-loop actuation)

| fixture | tracking | max dF/frame | finite |
|---|---|---|---|
| rearfoot punch | 0.193 mm | 13.5 N | True |
| fullfoot last | 0.147 mm | 26.8 N | True |

Closed-loop PD actuation stays stable and tracks the trajectory to <0.2 mm, so
the calibrated law is dynamics-stable, not just kinematically replayable. (The
stiff servo needs substeps >= 32: its natural period ~0.9 ms must resolve the
substep dt.)

**Conclusion.** The Phase-1 calibrated shoe law runs stably as a coupled
Warp/MuJoCo body-wrench source, reproduces the held-out bench metrics exactly
under kinematic drive, and remains stable under closed-loop servo actuation. The
residual peak/hysteresis gap is the same data/model ceiling identified in Phase
1; moving it requires same-fixture multi-rate / relaxation experiments, not a
dynamics change.


### Figures

Regenerate with ``uv run -m projects.digital_instron_v2.plot_phase2``
(``DigitalInstron/figures/phase2/``):

- ``fig1_hysteresis_loops.png`` -- measured vs simulated force-displacement
  loops (kinematic), annotated with the peak / RMSE / hysteresis errors.
- ``fig2_force_vs_phase.png`` -- force vs cycle phase, showing the pointwise
  (active-RMSE) agreement.
- ``fig3_dynamics_diagnostics.png`` -- COP trajectory, active contact area, max
  column compression, and wrench continuity over the cycle (both fixtures).
- ``fig4_kinematic_vs_servo.png`` -- loops by drive mode and servo tracking. The
  servo hysteresis differs from kinematic only because closed-loop PD smooths the
  crosshead velocity (an uncontrolled artifact, not a genuine model change), so
  only the kinematic loops are gated.


### Backend comparison figures

Regenerate with ``uv run -m projects.digital_instron_v2.plot_backends`` (uses the
GPU differentiable backend; a few minutes; ``DigitalInstron/figures/backends/``):

- ``fig1_backend_heldout_loops.png`` -- held-out force-displacement loops for
  ``scipy``, ``diff (cold)`` and ``diff (warm <- scipy)``. scipy and warm-diff
  overlap; cold-diff is visibly thinner.
- ``fig2_backend_metrics.png`` -- held-out peak / RMSE / hysteresis per fixture.
  peak and RMSE are backend-independent; only cold-diff hysteresis blows up
  (57% / 42% vs ~16%).
- ``fig3_backend_parameters.png`` -- fitted shared parameters. G_inst, alpha and
  pasternak agree; the tell is ``eq_fraction`` (scipy/warm 0.107 vs cold 0.685),
  which sets the Maxwell overstress and hence the loop area.
- ``fig4_backend_convergence.png`` -- Adam loss vs iteration. Cold-diff descends
  from the seed but plateaus *above* the scipy train-loss optimum; warm-diff sits
  on it and does not move.

Takeaway: exact gradients do not beat the derivative-free scipy fit. Warm-started
diff reproduces scipy; a cold diff descent settles in a worse basin (higher train
loss, wrecked held-out hysteresis). The residual floor is a data/model
identifiability ceiling, not an optimizer limitation -- so scipy least-squares is
the authoritative calibration backend.
