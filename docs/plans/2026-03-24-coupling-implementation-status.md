# Newton Coupling SDD — Implementation Status

**Last updated:** 2026-05-12
**Branch:** `gdaviet/coupled-solver-framework`
**SDD:** [Newton Coupling SDD (WIP)](https://docs.google.com/document/d/16Bhp9Fw4lsLdMt3AhJnVR2V_iVcvBw8Xc2zmMC9L2IQ/edit?tab=t.0)

## Summary

The Python coupling framework is implemented as an experimental solver layer.
The branch intentionally does not update the README example gallery yet, and the
temporary coupling screenshots have been removed from `docs/images/examples`.

The implementation now has three layers:

- `SolverCoupled`: shared base class for model views, state distribution,
  reconciliation, per-entry substeps, effective-mass fallbacks, and hook dispatch.
- `SolverProxyCoupled`: lagged-impulse proxy coupling for body and particle
  proxies, including virtual inertia, contact preparation, velocity rewind, and
  force harvesting.
- `SolverAdmmCoupled`: linearized ADMM coupling for model-derived joints,
  custom body-particle attachments, and internally detected contacts.

## Implemented Components

| Component | File | Status |
|-----------|------|--------|
| `ModelView` | `newton/_src/solvers/coupled/model_view.py` | Working. Attribute-delegating model proxy with view-local body/particle ownership, proxy flags, full body inertia overrides, particle mass overrides, and per-view collision flags. |
| `SolverCoupled` | `newton/_src/solvers/coupled/solver_coupled.py` | Working. Owns entry construction, model-view customization, state copy/reconcile, per-entry substeps, force-buffer injection, effective-mass fallback queries, and state-update notifications. |
| `SolverProxyCoupled` | `newton/_src/solvers/coupled/solver_proxy_coupled.py` | Working. Contains the lagged proxy-specific loop extracted from `SolverCoupled`. Proxy mode currently supports at most two solver entries by design. |
| `SolverAdmmCoupled` | `newton/_src/solvers/coupled/solver_admm_coupled.py` | Working prototype. Supports cross-solver model joints, custom body-particle attachments, rigid/particle contacts, Coulomb friction, and graph-capturable fixed-iteration solves. |
| Proxy kernels | `newton/_src/solvers/coupled/proxy_utils.py` | Working. Body/particle proxy state sync, velocity rewind, smooth body teleport support, and fallback momentum harvesting. |
| ADMM kernels | `newton/_src/solvers/coupled/admm_utils.py` | Working. Constraint/contact relative velocities, local updates, dual updates, force splats, Coulomb projection, joint box friction, and particle-particle hash-grid contact generation. |
| Public API | `newton/solvers.py` | Working. Exports `ModelView`, `SolverCoupled`, `SolverProxyCoupled`, `SolverAdmmCoupled`, and contact-stream helpers. No single-pair wrapper classes are exported. |

## Coupling Hooks

`SolverBase.CouplingHooks` now describes only operations where a solver may need
custom behavior. Missing entries use the coupler fallback, `CUSTOM` calls the
solver hook, and `UNSUPPORTED` rejects the coupling mode.

| Hook | Default behavior | Current custom users |
|------|------------------|----------------------|
| `NOTIFY_INPUT_STATE_UPDATE` | No-op after public `State` arrays are written. | VBD uses it to update `body_q_prev` / smooth proxy teleports and repeated coupling iterations. MPM uses it to keep collider point/velocity caches in sync. |
| `BODY_PROXY_REWIND_VELOCITY` | Subtract previous lagged feedback, public `body_f`, and gravity from destination proxy body velocity. | MPM collider body proxies. |
| `PARTICLE_PROXY_REWIND_VELOCITY` | Subtract previous lagged feedback, public `particle_f`, and gravity from destination proxy particle velocity. | MPM transfer-active and deformable-collider proxy particles. |
| `BODY_PROXY_HARVEST` | Estimate feedback from destination proxy body momentum change. | VBD reduces final rigid and body-particle contact forces; MPM harvests collider grid impulses. |
| `PARTICLE_PROXY_HARVEST` | Estimate feedback from destination proxy particle momentum change. | MPM harvests transfer/grid feedback for proxy particles. |
| `PROXY_CONTACT_PREPARE` | Use supplied contacts with generic proxy filtering. | VBD filters proxy contacts and controls rigid history updates on contact-refresh cadence. |
| `EFFECTIVE_MASS_DIAGONAL` | Use model mass/inertia scalar fallback. | MuJoCo can provide articulated effective mass on GPU. |
| `EFFECTIVE_MASS_BLOCK` | Use model mass and full body inertia fallback. | MuJoCo can provide articulated block inertia on GPU. |

Force injection and model-view inertia/mass overrides are no longer hooks.
Couplers write `state.body_f`, `state.particle_f`, and `control.joint_f`
directly, mutate `ModelView` mass/inertia arrays for virtual/proximal inertia,
then use the normal `notify_model_changed()` path when a constructed solver must
refresh private model caches.

## Changes to Existing Solvers

**SolverBase.** The common solver base now defines the experimental coupling
interface: endpoint kinds, input-state update flags, hook identifiers,
default/custom/unsupported capability reporting, effective-mass query hooks,
proxy rewind/harvest hooks, and proxy contact preparation. These additions keep
`step()` source-compatible; coupled wrappers only call a custom hook when the
solver advertises one, otherwise they use the public model/state fallback.

**VBD.** VBD advertises custom proxy contact preparation, body-proxy harvesting,
and input-state update handling. The input-state hook keeps `body_q_prev`
consistent when the coupler synchronizes proxy poses or restarts a coupling
iteration, so repeated proxy/ADMM passes are not treated as physical teleports.
The proxy contact hook filters invalid proxy contacts and controls rigid-history
refresh cadence, while the body harvest hook reduces final rigid-rigid and
body-particle contact forces onto proxy bodies instead of relying on brittle
momentum harvesting. A particle-only CUDA guard was also added so VBD does not
read triangle adjacency when no triangle topology is present.

**Implicit MPM.** Implicit MPM gained `collider_particle_ids` support so mesh
colliders can be tied to finite-mass model particles and coupled as deformable
proxy colliders. It now maintains separate transfer and material particle flags:
pure proxy particles can participate in P2G/G2P transfers, while proxy material
contribution and stress terms are suppressed through `material_particle_flags`.
The solver advertises custom body/particle proxy rewind and harvest hooks to
sync collider caches, subtract lagged gravity/feedback, and harvest grid or
collider impulses back into proxy force buffers.

**XPBD.** XPBD's particle contact kernels now understand proxy particles and
proxy bodies. Owned particles can collide with destination proxy particles, but
proxy-proxy, proxy-static, and proxy-particle-versus-proxy-body contacts are
filtered so the destination solve does not create feedback between two proxy
endpoints or against immovable particles. This is the main solver-side change
needed by the XPBD/VBD particle-proxy example.

**MuJoCo.** MuJoCo advertises custom diagonal and block effective-mass hooks on
the GPU path. Those hooks query MuJoCo Warp's `body_invweight0` data and map it
back to Newton body ids, giving proxy virtual inertia and ADMM endpoint weights
an articulated effective-mass estimate instead of a raw body mass fallback.
The normal MuJoCo step and force-input path remain unchanged.

**SemiImplicit.** SemiImplicit now reserves and rebuilds the model particle
hash-grid before particle contact evaluation when particle contacts are active,
and particle contact forces accumulate into the existing `particle_f` buffer
instead of overwriting it. That preserves externally injected or coupled
particle forces when SemiImplicit is used in coupled scenes.

**Kamino.** Kamino model construction now accepts `ModelView` instances by
duck-typing the required model fields instead of requiring an actual
`newton.Model` instance. This lets coupled examples pass a rigid-only Kamino
view that hides unsupported soft topology while still using the normal Kamino
solver construction path.

## Proxy Coupling Status

| Feature | Status |
|---------|--------|
| Body proxies | Working for rigid-to-VBD, rigid-to-XPBD, rigid-to-MPM, and cable robot proxy scenes. Virtual body inertia uses full mass/inertia blocks from `EFFECTIVE_MASS_BLOCK` when available. |
| Particle proxies | Working for XPBD/VBD and XPBD/MPM particle transfer scenes. Pure MPM proxy particles remain transfer-active but are excluded from material transfer through `material_particle_flags`. |
| Deformable MPM colliders | Working for VBD/MPM soft bodies. Triangle-owned particles use the collider path rather than pure transfer particles. |
| Contact preparation | Proxy-local collision pipelines can be supplied per proxy with an optional collide interval. A `None` factory means the outer contact set is passed through. |
| Iteration count | Working for lagged proxy relaxation. Repeated iterations notify solvers with `ITERATION_RESTART` so private histories can be realigned. |
| Harvesting | Solver-specific contact/impulse harvesting is preferred where available; the fallback remains momentum-based and is most useful for simple particle proxy cases. |

## ADMM Coupling Status

| Feature | Status |
|---------|--------|
| Cross-solver model joints | Working for `BALL`, `FIXED`, and `REVOLUTE` joints. Revolute joints preserve the hinge axis and can add per-axis box friction. |
| Body-particle attachments | Working through `coupling:body_particle_attachment` custom model attributes with stiffness and damping. |
| Contacts | Working for internally detected rigid-rigid, rigid-particle, and particle-particle contacts. The local contact update solves Coulomb friction with a maximum-dissipation projection rather than cone complementarity. |
| Particle-particle detection | Working through a private hash-grid stream structurally similar to `newton.Contacts`; it is not a public contact API yet. |
| Effective mass | ADMM uses diagonal endpoint effective mass hooks when available and model fallbacks otherwise. |
| Graph capture | Examples use fixed iteration counts and graph-capture their frame loops on CUDA. |

## Examples

The new examples are under `newton/examples/multiphysics` and are deliberately
standalone. Shared example-only helpers have been removed.

| Example | Purpose |
|---------|---------|
| `example_mujoco_vbd_coupled_solver.py` | Rigid/VBD body proxy coupling with optional MuJoCo or Kamino rigid solver. |
| `example_mujoco_xpbd_coupled_solver.py` | Rigid/XPBD body proxy coupling with optional MuJoCo or Kamino rigid solver. |
| `example_mujoco_mpm_coupled_solver.py` | Rigid/MPM proxy-collider coupling with optional MuJoCo or Kamino rigid solver. |
| `example_xpbd_vbd_coupled_solver.py` | Particle proxy coupling between XPBD particles and VBD cloth. |
| `example_xpbd_mpm_coupled_solver.py` | Pure particle proxy transfer coupling between XPBD particles and MPM material. |
| `example_vbd_mpm_coupled_solver.py` | Deformable VBD collider coupled to an MPM bed. |
| `example_cable_robot_proxy_coupled_solver.py` | Port of the cable robot two-way proxy scheme to `SolverProxyCoupled`. |
| `example_admm_contact_solver.py` | ADMM rigid/particle contact coupling with collision detection. |
| `example_admm_rigid_contact_solver.py` | ADMM rigid-rigid frictional contact, including stick-to-slide behavior. |
| `example_mujoco_vbd_admm_solver.py` | MuJoCo/Kamino rigid bodies coupled to VBD cloth and payloads through ADMM joints/attachments. |
| `example_kamino_mujoco_admm_solver.py` | Four-bar linkage split across Kamino and MuJoCo via ADMM revolute joints. |
| `example_proxy_coupling_convergence.py` | Synthetic proxy fixed-point convergence and virtual-inertia sweep. |

## Tests

| Test | Coverage |
|------|----------|
| `newton/_src/solvers/coupled/test_coupled.py` | ModelView overlays, proxy body/particle mappings, virtual inertia, hook dispatch, proxy iteration behavior, XPBD proxy filtering, and force-buffer notification. |
| `newton/_src/solvers/coupled/test_admm_coupled.py` | ADMM attachments, joint discovery, contact detection, Coulomb friction, particle-particle hash-grid contacts, external-force passthrough, effective-mass hooks, and repeated-iteration notifications. |
| `newton/_src/solvers/implicit_mpm/test_proxy_particles.py` | MPM transfer-active proxy particles, material inactivity, deformable collider proxies, and gravity-subtracted feedback. |
| `newton/tests/test_implicit_mpm.py` | Public MPM regression coverage for proxy material flags. |

## Known Limitations

1. Proxy stability remains tuning-sensitive. Virtual inertia scale, contact
   stiffness, solver iterations, and lagged versus staggered mode can strongly
   affect damping and convergence.
2. Momentum harvesting is only a fallback. Hard or solver-private contact modes
   should expose custom contact/impulse harvesting where possible.
3. ADMM contact detection is internal. It does not yet consume arbitrary caller
   `newton.Contacts` rows as a public coupling stream.
4. ADMM joint support is still narrow. `BALL`, `FIXED`, and `REVOLUTE` are
   implemented; prismatic, distance, and D6 rows remain future work.
5. USD ownership and automatic coupled-solver construction are not implemented.

## Next Steps

1. Decide whether and how to expose a public contact streaming API for ADMM
   contacts and, potentially, future proxy harvesters.
2. Add a principled tuning guide for proxy virtual inertia and ADMM `rho` /
   `gamma` / Baumgarte parameters.
3. Extend ADMM joint rows to prismatic, distance, and D6 constraints.
4. Benchmark proxy versus ADMM coupling on the same reference problems.
5. Revisit README/example-gallery exposure once the experimental API settles.
