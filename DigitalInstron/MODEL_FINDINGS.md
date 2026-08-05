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
