# Digital Instron v2 — Geometry & Material Foundation Report

*How the midsole column bed and its constitutive law are built, and why this is
the right reduction for a differentiable, contact-ready digital shoe.*

## 1. The modeling problem

A running-shoe midsole is a thick block of nonlinear, rate-dependent,
spatially non-uniform foam. A first-principles finite-element hyper-viscoelastic
solve of it is far too slow to sit inside a contact-rich, real-time, and
*differentiable* simulation loop (a running stride, a design optimization, an
inverse-ID problem). We therefore reduce the midsole to an **elastic foundation**:
a bed of independent through-thickness nonlinear viscoelastic columns with a weak
lateral shear coupling. This is the classical **Winkler/Pasternak** reduction for
a compliant layer squeezed between two effectively rigid surfaces (crosshead/last
above, ground/load-cell below) — exactly the Instron and ground-contact regime.

Two things are built:

1. **Geometry** — turn the real midsole mesh into a column bed (this section is
   fixture-specific: a flat heel punch vs. a curved full-foot last).
2. **Material law** — one shared 4-parameter constitutive law each column obeys
   (this is fixture-*independent*).

The central design claim: **the geometry carries the fixture, the material carries
the foam.** Keeping those two concerns separate is what makes the model both
identifiable and transferable.

## 2. Geometry: from a triangle mesh to a column bed

### 2.1 Canonicalizing the mesh (`geometry.load_mesh`)

The raw midsole `.obj` is loaded at mm scale (×0.001 → metres) and put in a
repeatable frame with no manual alignment:

- **Axis identification by extent.** Thickness = smallest-extent axis, length =
  largest-extent axis, width = the remaining one. Vertices are reordered to
  `(length, width, thickness)`.
- **Footprint centering** in-plane and **base drop** so the foam bottom sits at
  `z = 0`. Optional roll/pitch/yaw and a height crop are supported.

This makes the pipeline export-agnostic and deterministic — a fresh checkout
samples the identical bed.

### 2.2 Column sampling (`geometry.build_column_grid`)

A uniform **5 mm** grid (`coarse_spacing_m = 0.005`) tiles the footprint bounding
box. At each grid point a vertical ray is cast through the mesh
(`geometry.raycast`), returning the first (bottom) and last (top) intersections:

- **Rest length** `slack = top − bottom` per column.
- A **normal threshold** keeps only clean through-columns (bottom face pointing
  down, top face pointing up), rejecting side walls and grazing hits.
- Columns thinner than 5 mm are dropped.

Each surviving column gets a **tributary area** = spacing² = 25 mm².

The result is the actual midsole thickness field, not an idealized slab:

| fixture | columns | spacing | area/col | rest thickness (min–max) |
|---|---|---|---|---|
| fullfoot last | 611 | 5 mm | 25.0 mm² | 7.4 – 43.8 mm |
| rearfoot punch | 62 | 5 mm | 24.5 mm² | 25.4 – 43.6 mm |

The thin forefoot / thick heel stiffness contrast therefore **emerges from the
sampled geometry** — it is never tuned.

### 2.3 Fixture indenters

The indenter geometry — not the material — is what makes the two bench tests
different:

- **Full-foot last** (`build_foundation_geometry`): the shoe-last mesh is loaded,
  posed, and its underside is raycast onto the column grid to give every column an
  initial clearance under the crosshead (`surface_m`, `gap0_m`). Because the last
  is curved, columns engage **progressively** as it descends — this is precisely
  what produces the ramp-in of active contact area and the migrating center of
  pressure seen in the Phase-2 diagnostics.
- **Rearfoot punch** (`_build_rearfoot_geometry`): a rigid, flat **22 mm-radius**
  circular punch. The punch center is placed 20 % of the way from the heel along
  the long axis (`rearfoot_length_fraction = 0.2`, `geometry.rearfoot_center`), the
  62 disc columns are selected, and every column top is anchored at its own rest
  height (`z_free = slack`). A carrier descent `d` then compresses **every** disc
  column by exactly `d` — the uniform-compression condition of a rigid flat platen.

### 2.4 The Pasternak neighbor stencil (`_neighbor_indices`)

Each column stores its four in-plane lattice neighbors (`±u, ±v`), tagged:

- `≥ 0` — an active neighbor index,
- `−1` — a **footprint boundary** (natural, zero-gradient),
- `−2` — an **interior gap** cell (contributes no coupling).

This table *is* the discrete 5 mm Laplacian stencil used for the Pasternak shear
term (§3c). Crucially it is the **same operator** used during calibration, so the
fitted material and the simulated bed are bit-consistent.

### Why this geometry approach is a good idea

- **Real part geometry, no idealization.** Heel-vs-forefoot behavior falls out of
  the sampled thickness profile and footprint shape, not from fudge factors.
- **Deterministic & reproducible.** Extent-based canonicalization is export-
  agnostic; the bed regenerates identically from a clean checkout.
- **Embarrassingly parallel.** One thread per column; ~600 columns evaluate in
  microseconds on the GPU — thousands of substeps per second.
- **One bed, two consumers.** The same sampled bed and neighbor stencil feed both
  the NumPy calibration (`core.predict`) and the Warp dynamic kernel, so *what you
  fit is exactly what you simulate* (Phase 2 confirmed bit-for-bit agreement).
- **Fixtures via geometry alone.** Flat punch vs. curved last differ only in the
  indenter; the material is untouched — a genuine cross-fixture test of one law.

## 3. The material foundation law

Every column obeys the same law, built from three physically-motivated pieces
that share **four** parameters. Per column, the through-thickness engineering
strain is `ε = compression / rest_length` (clamped ≥ 0) and the stretch is
`λ = 1 − ε`, floored at `stretch_floor = 0.05` (a densification limit — foam
cannot compress to zero thickness).

### (a) Equilibrium pressure — first-order Ogden–Hill *Hyperfoam*

```
p_eq(ε) = (2 · G_eq) / (α · λ) · [ (λ^(1−2ν))^(−αβ) − λ^α ],
          β = ν / (1 − 2ν),   G_eq = G_inst · eq_fraction,   ν = 0.30 (fixed)
```

This is the standard rubber-/foam **hyperfoam** strain-energy first term
specialized to confined uniaxial compression. With just a modulus `G` and an
exponent `α` it reproduces the characteristic **J-shaped** foam response — a soft
initial "toe" that stiffens sharply into densification. Poisson's ratio `ν` is
fixed at a physically sensible weakly-compressible value (0.30), removing a
parameter that a single test cannot identify.

### (b) Rate-dependent overstress — one generalized-Maxwell branch

The **instantaneous** stiffness is higher than the equilibrium stiffness by the
factor `G_inst / G_eq = 1 / eq_fraction`. That extra "overstress" `q` relaxes with
a fixed time constant `τ = 0.08 s`:

```
dq/dt = −q/τ + overstress · dp_eq/dt,     overstress = (1 − eq_fraction)/eq_fraction
```

integrated **exactly** per substep:

```
decay = exp(−dt/τ)
q ← decay·q + overstress · [ τ(1−decay)/dt ] · (p_eq − p_eq_prev)
p_total = p_eq + q
```

This single branch is the minimal model that **opens the hysteresis loop**: on
loading, `q` rides the pressure up; on unloading it relaxes down, so the unloading
branch sits below the loading branch and the enclosed area is the dissipated
energy. The discretization is **rate-consistent (dt-robust — verified over
substeps 16/32/64)**, so the *same* law gives the correct loop at the slow bench
cycle rate and at fast running-stride rates.

### (c) Lateral coupling — Pasternak shear layer

Pure Winkler springs are decoupled and cannot spread load. A **Pasternak** shear
layer adds a curvature term so a column feels its neighbors:

```
p_eff = p_total − k_p · ∇²(compression)
```

where `∇²` is the 5 mm 4-neighbor Laplacian from the stencil of §2.4 (with a
natural zero-gradient footprint edge). `k_p` (N/m) smooths the pressure field and
models the in-plane shear that a single independent column cannot — essential for
a curved last to distribute pressure realistically rather than punching through.

### Force assembly

Per column: `f_n = max(p_eff, 0) · area`. Each column force is accumulated into a
rigid-body **wrench** with moment `r × f` about the carrier COM, giving the total
normal force, the **center of pressure** (from the pressure-weighted moment), and
the full 6-DOF wrench written into `newton.State.body_f`. The dynamic scenarios
optionally layer Kelvin–Voigt normal damping and an anchored-bristle (elasto-
plastic) Coulomb friction on top; the Instron leaves those at zero so the fitted
loop is reproduced exactly.

### The parameter set (4 fitted, shared across both fixtures)

| symbol | meaning | value | data signature |
|---|---|---|---|
| `G_inst` | instantaneous shear modulus | ≈ 19.0 kPa | initial loading slope |
| `α` | Hyperfoam stiffening exponent | ≈ 5.13 | densification knee |
| `eq_fraction` | equilibrium/instantaneous ratio | ≈ 0.106 | loop area (→ overstress ≈ 8.4×) |
| `k_p` | Pasternak lateral coupling | ≈ 918 N/m | spatial pressure spread |

Fixed (not fitted): `ν = 0.30`, `τ = 0.08 s`, `stretch_floor = 0.05`.

## 4. Calibrating the law: derivative-free and differentiable backends

The four parameters are recovered by **inverse identification**: drive the column
bed through the measured compression cycle and descend a loss that matches the
predicted reaction force to the bench force-displacement loop. Two backends share
the *identical* train (cycles 90-98) / held-out (cycles 99-100) protocol, so exact
gradients can be compared head-to-head against a derivative-free reference
(`phase1.py`, `--compare`).

### (a) `scipy` -- derivative-free least-squares (authoritative)

`core.fit_material` runs a bounded least-squares fit (SciPy) of the four
parameters against the per-trial, peak-normalized force residual of the analytic
NumPy `core.predict` model. It needs no gradients, is robust from a cold manifest
seed, and is the **authoritative** calibration used everywhere downstream (Phase-2
dynamics, all figures). Fitted values: `G_inst = 19.0 kPa`, `alpha = 5.13`,
`eq_fraction = 0.107`, `k_p = 918 N/m`.

### (b) `diff` -- differentiable Adam with exact gradients

`inverse_id.fit_material_to_trials` records the whole compression cycle -- *including
the generalized-Maxwell loading/unloading hysteresis* -- on a `warp.Tape` and
back-propagates the same peak-normalized force loss to get the exact gradient
w.r.t. all four parameters in one backward pass, then takes Adam steps in a
dimensionless `scale x reference` space (the parameters span very different
magnitudes). It runs on the **differentiable sibling** force model
(`dynamics_diff.DifferentiableMidsoleFoundation`), which is the same Hyperfoam-
Maxwell-Pasternak law reshaped to be autodiff-safe: a write-once viscoelastic
recurrence (no in-place `q_state` aliasing across substeps) and a smooth,
cone-respecting Coulomb friction (`warp.smooth_normalize`) in place of the forward
model's integer stick/slip flag. The material lives in a length-4 `requires_grad`
array, so any simulation objective exposes `material_params.grad` directly.

### (c) What the comparison shows

Running both backends through the identical protocol (figures under
`DigitalInstron/figures/backends/`) yields a clear result:

- **scipy and warm-started diff coincide.** Seeding the differentiable Adam fit at
  the scipy optimum reproduces it bit-for-bit (`eq_fraction 0.107`, held-out
  hysteresis 16% / 17%) and the loss does not move -- scipy already sits at the
  optimum the gradient points to.
- **A cold diff descent settles in a worse basin.** Started from the manifest
  seed, Adam converges but to `eq_fraction = 0.685` (far less Maxwell overstress ->
  a too-thin loop), giving a **higher** train loss (0.96 vs 0.86) and held-out
  hysteresis errors of **57% / 42%**. Peak and RMSE stay backend-independent; only
  the loop area is spoiled.

The conclusion is that **exact gradients do not beat the derivative-free fit on
this static loop**: the residual floor is a *data/model identifiability ceiling*,
not an optimizer limitation. `eq_fraction` (the equilibrium/instantaneous ratio,
hence the hysteresis) is the weakly-identified direction that a single-rate loop
cannot pin down tightly, so a far cold start can trade it off against the other
parameters at nearly-equal peak/RMSE.

### (d) Why keep the differentiable backend

Its payoff is not out-fitting scipy on the static loop -- it is enabling
gradient-based tasks the derivative-free fit *cannot* do:

- **friction identification**: the Coulomb coefficient `mu` sits in its own length-1
  `requires_grad` array, so a lateral-/shear-force target identifies friction by
  gradient descent (`scenarios_diff.DifferentiableSlide`);
- **gradients through the full dynamic loop**: contact, PD actuation, and smooth
  friction of a mass-carrying stride (`scenarios_diff.DifferentiableAttached`,
  integrated by `SolverSemiImplicit`), which the quasi-static `core.predict` cannot
  reach;
- **design / inverse problems** that need `d(objective)/d(material)`.

The gradient is *exact while contact is continuous* and a valid subgradient at
column make/break events (which a finite difference will not match at the event).
So the practical policy is: **scipy is authoritative for the static material fit;
the differentiable backend is used for gradient tasks and is best warm-started
from the scipy optimum.**

## 5. Why this is a good idea

1. **Physically interpretable & minimal.** Four parameters, each with a distinct
   physical role *and* a distinct fingerprint in the data (slope, knee, loop area,
   spread). This is the opposite of a black box or a many-parameter FE model whose
   parameters trade off against one another.
2. **Right physics for the regime.** Confined through-thickness compression of a
   compliant layer between rigid platens genuinely *is* a Winkler/Pasternak
   problem; Hyperfoam is the accepted foam constitutive family; one Maxwell branch
   is the minimal rate-dependent hysteresis model. The reduction is principled,
   not a fit of convenience.
3. **Identifiable from cheap bench data.** The whole law calibrates from a single
   displacement-controlled force–displacement loop and **generalizes to held-out
   cycles** (< 0.4-point metric gap). The parameters are separable enough that
   `scipy` least-squares recovers them from a cold start.
4. **Shared law, fixture-specific geometry.** One material set reproduces *both* a
   flat rearfoot punch and a curved full-foot last, purely through geometry. That
   the same four numbers explain both fixtures is strong evidence the law captures
   the **material**, not the test rig.
5. **GPU-native, real-time, differentiable.** One Warp thread per column; the bed
   evaluates in microseconds and couples straight into MuJoCo-Warp via
   `state.body_f`. Every formula is smooth and analytic, so the identical law also
   supports gradient-based design and inverse identification (the Phase-1
   differentiable backend).
6. **Calibration/simulation consistency.** The same strain definition, Hyperfoam
   and Maxwell formulas, and neighbor stencil drive both the NumPy fit and the Warp
   kernel — Phase 2 verified they agree to the bit. There is no "sim-to-fit" gap.
7. **Extensible to full contact.** The bed already carries optional normal damping
   and Coulomb foam-shear friction, so the calibrated normal law drops into 3D
   ground contact (settle, stride) with no recalibration.

### Honest limitations

- **Winkler/Pasternak** ignores full 3D stress coupling and plate bending — fine
  for a compliant layer between rigid platens, weaker for extreme shear or thick-
  plate bending.
- **One Maxwell branch with fixed `τ`** captures a single dominant relaxation
  time. The measured loop has a nearly strain-*independent* gap (structural-
  damping-like) that one branch under-fits, which is the source of the residual
  ~16 % hysteresis ceiling. A multi-branch Prony series or an added rate-
  independent damping term would help — but needs multi-rate / relaxation bench
  data to identify.
- **`ν` and `τ` are fixed** because a single-rate test cannot separate them; moving
  them requires same-fixture multi-rate experiments.

---

*Sources: `projects/digital_instron_v2/geometry.py` (mesh → columns),
`projects/digital_instron_v2/dynamics.py` (Warp foundation kernels, geometry
builders), `projects/digital_instron_v2/core.py` (`Material`, Hyperfoam, periodic
Maxwell, `predict`). Fitted values from the train-only calibration
(`digital_instron_material.json`).*
