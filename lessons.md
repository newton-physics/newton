# Project Lessons

## 2026-08-12 — Use one Newton solve for the first ABD bunny drop

- Context: Fixing the runtime configuration for the first frictional affine-body bunny-to-ground example.
- Mistake: Proposed four Newton iterations with 32 PCG iterations instead of preserving the user's preferred LIMX convergence schedule.
- Rule: Configure this ABD bunny drop with exactly one Newton iteration and 50 PCG iterations per frame. Do not retune those counts unless the user explicitly requests it.

## 2026-08-12 — Keep ABD ground contact penalty-based

- Context: Designing the first frictional affine-body bunny drop after consulting libuipc's affine contact implementation.
- Mistake: Proposed adopting libuipc's IPC barrier even though the requested Newton/LIMX collision direction remains penalty contact.
- Rule: Implement ABD normal contact with the project's penalty formulation. Use libuipc only as a reference for the `A+t` material-point Jacobian, the lifts `J^T g` and `J^T H J`, and smooth lagged friction unless the user explicitly requests a barrier formulation.

## 2026-08-11 — Keep affine and particle solver blocks distinct

- Context: Designing pure-Newton affine-body dynamics integration with deformable LIMX particles.
- Mistake: Proposed storing each 12-DOF affine body as four ordinary 3-DOF CSR rows, forcing the preconditioner to reconstruct a logical 12x12 body block from sixteen 3x3 blocks.
- Rule: Represent deformable particles with native 3x3 blocks and affine bodies with native 12x12 blocks. Couple the two spaces through explicit mixed or matrix-free contact operators, and apply native 3x3 and 12x12 block preconditioners within one global conjugate-gradient solve.

## 2026-08-11 — Distinguish unsigned closest-point VF from oriented projected VF

- Context: Designing an experiment for VF/EE feature-boundary continuity and possible PE/PP coverage.
- Mistake: Claimed that current VF implicitly covers PE/PP from the unsigned closest-point branch without first checking that oriented surface collision uses projected barycentrics and rejects projections outside the triangle interior.
- Rule: Before claiming VF covers boundary PE/PP features, identify the active `use_outward_normals` mode. Treat unsigned closest-point VF and oriented projected VF separately, and verify the exact scene configuration before reasoning about feature coverage or duplicate contacts.

## 2026-08-11 — Prove EE-derived features by exact VF membership

- Context: Experimenting whether endpoint EE contacts are redundant with VF-generated PE/PP contacts.
- Mistake: Used aggregate counts and the existence of a VF contact against any triangle incident to the target edge as the coverage criterion, which does not prove that VF generated the same PP or PE feature as EE.
- Rule: Classify both EE and VF contacts into canonical PP/PE keys and test the exact set inclusion `EE-derived PP/PE keys ⊆ VF-derived PP/PE keys`. Use counts only to summarize the result, never as the proof of equivalence.

## 2026-08-11 — Keep LIMX EE contact interior-only

- Context: Comparing the twist scene with and without endpoint-derived EE PE/PP contacts after confirming unsigned VF already evaluates triangle boundary features.
- Mistake: Retained clamped EE endpoint responses alongside VF boundary responses, creating redundant PE/PP constraints and roughly tripling the twist scene's EE count without improving its RMS motion.
- Rule: LIMX EE proximity contact must require both closest parameters strictly inside their segments. Delegate PE/PP response to unsigned VF, and keep a regression test that rejects endpoint EE while retaining strict interior EE.

## 2026-07-29 — Keep the primary checkout aligned with the user's daily branch

- Context: Configuring this Newton fork as the editable source for Isaac Sim 6.0.1.
- Mistake: Kept `main` at `/home/limx/github/newton` and placed the actual development branch in a hidden `.worktrees/` path, adding unnecessary friction to the user's normal solver workflow.
- Rule: When the user only needs one compatibility branch for daily development, make the canonical repository path check out that branch. Keep other branches available remotely or as Git refs unless the user explicitly asks for simultaneous local checkouts.

## 2026-07-29 — Distinguish integration mechanics from solver design

- Context: The user asked how CUDA code is embedded into Newton.
- Mistake: Redirected the discussion into choosing solver algorithms and joint scope instead of explaining the concrete CUDA/Warp integration boundary.
- Rule: When asked how native or GPU code plugs into an existing framework, first explain the exact module, kernel, launch, state, and export path. Discuss numerical-method design only if the user asks for it afterward.

## 2026-07-29 — Separate vector atomic execution from atomicity guarantees

- Context: Discussing CUDA `atomicAdd(float4*)` on RTX 50-series hardware versus Warp's component-wise vector atomic implementation.
- Mistake: Asserted equivalence from component-wise semantics without first verifying the architecture-specific CUDA implementation and documented guarantees.
- Rule: For GPU intrinsics, distinguish API semantics, compiler lowering, generated instructions, and hardware transaction width. Verify the installed toolkit and NVIDIA's architecture documentation before claiming two implementations are equivalent.

## 2026-07-29 — Preserve the requested Isaac Sim GUI context

- Context: The user asked whether there was a cloth demo after discussing Isaac Sim integration.
- Mistake: Launched Newton's standalone OpenGL cloth example instead of checking for a demo inside the full Isaac Sim GUI.
- Rule: When the active workflow is Isaac Sim and the user asks to run or show a demo, default to the Isaac Sim GUI context. Clearly distinguish standalone Newton examples from Isaac Sim-integrated examples before launching anything.

## 2026-07-30 — Start the cloth solver with a Warp MVP, not a CUDA implementation

- Context: Designing the user's first custom cloth solver in the Newton fork.
- Mistake: Continued to frame the design around the earlier CUDA requirement after the user narrowed the goal to a minimal two-corner-pinned falling cloth and no longer wanted to start in CUDA.
- Rule: Build the first cloth solver as the smallest testable Warp implementation using Newton's existing model/state interface. Treat handwritten PTX or another native backend only as a later, measured replacement for proven hotspot kernels.

## 2026-07-30 — Use the exact example command named by the user as the scene source

- Context: Selecting the baseline scene for the user's custom cloth solver.
- Mistake: Inferred that the user meant an internal two-particle sagging test after inspecting pin indices, even though the intended runnable scene was Newton's `python -m newton.examples cloth_hanging` example.
- Rule: When the user identifies a runnable example command, treat that exact example as the authoritative scene source. Preserve its scene construction and runtime behavior unless the user explicitly asks to simplify it.

## 2026-07-30 — Validate the Newton cloth scene in Isaac Sim before developing a custom solver

- Context: Planning the first custom cloth solver milestone and its visualization path.
- Mistake: Continued designing a custom solver harness before establishing the requested Newton-to-Isaac-Lab-to-Isaac-Sim scene path.
- Rule: For this cloth project, first reproduce Newton's `cloth_hanging` scene through the Isaac Lab Newton backend in the full Isaac Sim GUI. Do not start the custom solver until that integration baseline is working and visually verified.

## 2026-08-03 — Place LIMX inside Newton's solver package

- Context: Implementing the first constraint-based LIMX cloth solver.
- Mistake: Interpreted the requested LIMX folder as a repository-root experimental package and started creating `limx/` outside Newton's solver hierarchy.
- Rule: Implement LIMX alongside Newton's existing solvers under the solver package hierarchy, expose it through Newton's solver module as appropriate, and keep its examples and tests integrated with Newton rather than creating a repository-root package.

## 2026-08-03 — Distinguish explicit damping from integrator dissipation

- Context: Evaluating whether the LIMX hanging cloth should keep swinging after setting `velocity_damping=1.0`.
- Mistake: Treated removal of the explicit velocity multiplier as if it made the full implicit-Euler/projective-dynamics update energy-conserving, without measuring total mechanical energy or accounting for stiff penalty-anchor modes.
- Rule: When conservative motion is expected, measure kinetic, gravitational, elastic, and anchor energy over time. Distinguish explicit damping from algorithmic damping in the time integrator, and do not describe a solver as undamped or energy-preserving solely because its velocity multiplier is one.

## 2026-08-03 — Zero bending stiffness permits folding

- Context: Interpreting the curled free edge in the LIMX mass-spring cloth example, which contains only anchor and triangle-edge distance constraints.
- Mistake: Did not make explicit that removing bending energy eliminates resistance to changes in adjacent-triangle dihedral angles; it does not preserve a flat sheet and can expose mesh-scale buckling under compression.
- Rule: When diagnosing cloth shape, distinguish absence of bending force from absence of bending deformation. Verify the active constraint set, edge-length error, compression, and triangle-normal/dihedral variation before attributing curls to an unintended bending term.

## 2026-08-03 — Audit PD algebra before tuning iteration counts

- Context: The LIMX mass-spring scene appeared to require 64 nonlinear PD iterations before showing normal pendulum motion.
- Mistake: Treated the high iteration count as a tuning issue before explicitly auditing the assembled Hessian, its stiffness-to-inertia scale, and the PCG initial guess against the standard local-global PD equations.
- Rule: Before increasing PD iterations, write down the exact global matrix and right-hand side produced by the code, list every diagonal contribution, verify whether PCG is zero-started or warm-started, and compare convergence on a minimal mode against the reference formulation.

## 2026-08-03 — Keep Newton and PD formulations conceptually separate

- Context: The LIMX cloth solver was initially built around projective-dynamics local/global iterations, but the requested large-time-step solver was changed to full Newton assembly.
- Mistake: Continued treating the constant PD surrogate Hessian as the core solver even after its transverse curvature was shown to dominate convergence.
- Rule: Implement LIMX Newton iterations by evaluating elastic forces and exact Hessians at the current iterate, using the predicted position only in the inertial residual. Do not retain local projections or describe the Newton path as a variation of PD.

## 2026-08-03 — Keep the LIMX demo physics and rendering one-to-one

- Context: Choosing how often to render the LIMX cloth example after increasing its physics time step to 0.01 s.
- Mistake: Proposed grouping two physics steps into one rendered frame after the user briefly considered rendering every other step.
- Rule: For the LIMX visualization example, render after every physics step unless the user explicitly requests frame decimation. With a 0.01 s physics step, use one substep per 0.01 s frame so simulation and visualization remain one-to-one.

## 2026-08-03 — Keep LIMX parameter-tuning loops immediate

- Context: The user requested only changing the LIMX example time step and Newton/PCG iteration counts so they could inspect the motion.
- Mistake: Turned a small parameter adjustment into a long design, planning, full-regression, and branch-finishing workflow before showing the requested dynamics.
- Rule: For LIMX example-only parameter tuning, make the smallest requested edit and launch the visualization immediately after a focused smoke test. Do not run unrelated full-suite or branch-integration workflows unless the user requests them; keep deeper regression work separate from the interactive tuning loop.

## 2026-08-03 — State the LIMX PCG initial guess explicitly during convergence diagnosis

- Context: The one-Newton, 50-PCG LIMX cloth motion appeared unconverged and the user asked what PCG used as its initial solution.
- Mistake: Discussed iteration tuning without keeping the exact PCG initialization visible in the convergence analysis.
- Rule: When diagnosing LIMX convergence, first report whether PCG is zero-started or warm-started, identify the array used for `x0`, and distinguish the linear increment guess from the nonlinear position iterate. Test alternative initial guesses only after confirming the assembled residual and operator correspond to the same current Newton iterate.

## 2026-08-03 — Warm-start LIMX PCG from the previous frame

- Context: The user explicitly chose the prior frame's PCG solution as the next frame's initial solution after observing poor apparent convergence with a zero start.
- Mistake: Kept resetting the Newton increment to zero on every linear solve even though the requested interactive cloth path advances through nearby frame-to-frame systems.
- Rule: Preserve the final PCG increment from the preceding frame and use it as `x0` for the first linear solve of the next frame, recomputing `r0 = b - A*x0`. Initialize the persistent increment to zero for the first frame; if multiple Newton iterations are enabled, use zero-started corrections after the first solve within that frame.

## 2026-08-03 — Use dihedral-angle bending for LIMX

- Context: Extending the LIMX anisotropic membrane cloth with bending resistance.
- Mistake: Assumed the new constraint should copy Style3D's quadratic Crouzeix-Raviart isometric bending energy before confirming the requested bending model.
- Rule: Implement LIMX bending as a four-particle dihedral-angle energy, using `/home/limx/github/Ai-Phyiscs` as the reference for its force and Hessian treatment. Do not substitute Style3D's quadratic isometric bending model unless the user explicitly requests it.

## 2026-08-03 — Verify LIMX bending on GPU first

- Context: Defining the validation path for the LIMX dihedral-bending feature.
- Mistake: Included routine CPU example runs even though the active solver workflow and target hardware are CUDA, adding avoidable turnaround time.
- Rule: Run LIMX bending tests and example validation on CUDA by default. Use CPU only as a diagnostic fallback after a GPU failure or when isolating a kernel or numerical issue.

## 2026-08-03 — Verify the LIMX viewer controls before handing off a window

- Context: Launching the interactive CUDA cloth visualization for the user to inspect.
- Mistake: Reported the window as ready after confirming only that the render process stayed alive, despite its log warning that the ImGui backend was unavailable and therefore the control buttons were missing.
- Rule: Before handing off an interactive LIMX window, verify both rendering and GUI-control imports in the exact runtime environment, treat viewer dependency warnings as a failed launch, and restart only after the button panel is confirmed available.

## 2026-08-03 — Verify proximity support before blaming EF contact chatter

- Context: Diagnosing a localized oscillation in the LIMX cloth-twist self-collision scene.
- Mistake: Attributed repeated edge-face crossings directly to EF untangle constraints appearing and disappearing without first proving why the surrounding vertex-face and edge-edge proximity contacts failed to support the separated surfaces.
- Rule: When an untangled cloth pair intersects again, trace the exact primitive pair across frames. Verify current-versus-predicted detection timing, activation distance, VF/EE feature coverage, normal orientation, and the achieved EF separation before concluding that EF persistence is the root cause.

## 2026-08-03 — Verify LIMX contact coefficients before expanding the model

- Context: Discussing whether recurring twist-scene intersections required CCD/barrier contact instead of penalty contact.
- Mistake: Proposed broadening the collision formulation before first answering the narrower question of whether EE stiffness matched VF stiffness.
- Rule: When LIMX contact behavior is attributed to unequal VF/EE stiffness, inspect the actual force, Hessian-vector, diagonal, and example configuration first. Report an already-shared coefficient explicitly rather than proposing a wider collision redesign.

## 2026-08-03 — Keep LIMX EF recovery stronger than proximity contact

- Context: Tuning self-collision recovery after confirming VF and EE already share one stiffness.
- Mistake: Left EF untangle stiffness equal to VF/EE even though crossing recovery must be materially stronger than proximity support in this solver.
- Rule: Configure LIMX EF untangle stiffness to at least three times the VF/EE stiffness. Keep the relationship explicit in defaults and examples so later tuning does not silently collapse them back to one coefficient.

## 2026-08-03 — Do not use the full LIMX test module for GPU-only checks

- Context: Verifying the EF untangle stiffness default after the user required CUDA-first validation and no routine CPU runs.
- Mistake: Ran the whole `test_solver_limx.py` module, whose mixed test classes compile and execute several CPU kernels even though the changed self-collision and twist tests were already covered on CUDA.
- Rule: For GPU-only LIMX changes, select the exact CUDA-decorated tests with both `-p test_solver_limx.py` and narrow `-k` patterns. Do not run the complete module unless the user explicitly requests mixed CPU/GPU regression coverage or CUDA debugging requires it.

## 2026-08-04 — Keep the LIMX T-shirt bending visually compliant

- Context: Reviewing the first T-shirt-on-table visualization with rest-shape dihedral bending.
- Mistake: Used bending stiffness `0.01`, which preserved too much of the garment's curved USD rest shape and made the cloth look overly rigid on impact.
- Rule: Use dihedral bending stiffness `1.0e-4` in `cloth_limx_tshirt_table`. For visual parameter corrections, change the requested scene value, run the focused CUDA smoke test, and relaunch immediately before broader documentation or regression work.

## 2026-08-04 — Audit collision support before tuning T-shirt contact

- Context: The low-bending T-shirt-on-table scene visibly jittered under collision, raising questions about thickness, stiffness, and vertex-face feature coverage.
- Mistake: Treated the earlier settling threshold as sufficient evidence that the collision parameters and activation geometry were appropriate for the more compliant garment.
- Rule: Before tuning LIMX garment collision, compare its actual thickness/stiffness scaling and VF feature test against Style3D, measure active VF/EE/EF/table contacts and force support on the failing scene, and state explicitly whether projected points outside triangle interiors are delegated to EE rather than handled by VF.

## 2026-08-04 — Do not equate mean settling with visual stillness

- Context: The adaptive-contact T-shirt passed a 3000-frame regression based on mean particle speed, but the user still observed persistent local jitter in the interactive window.
- Mistake: Treated a global mean-speed threshold as proof that the garment had visually settled, allowing a small set of continuously oscillating particles or churning contacts to be averaged away.
- Rule: For LIMX cloth settling, measure final-window per-particle RMS/maximum speed, the count of persistently active particles, and frame-to-frame VF/EE/EF/table contact churn in addition to the global mean. Do not report visual stillness until those localized temporal metrics and the interactive scene agree.
## 2026-08-04 — Prefer established IPC degeneracy handling before custom collision ramps

- Context: LIMX EE self-collision stabilization for nearly parallel edges.
- Mistake: Implemented and tuned a custom angle/depth response before verifying the standard IPC edge-edge mollifier and its reference implementation.
- Rule: For established contact degeneracies, inspect the primary paper and authoritative source first, reproduce its invariant and threshold construction in tests, then justify any project-specific deviation explicitly.

## 2026-08-04 — Keep EE distance classification unchanged for this mollifier

- Context: Replacing LIMX's custom near-parallel EE response with the standard IPC mollifier.
- Mistake: Included IPC's extreme-parallel PP/PE distance fallback in the proposed scope even though the requested change is only the EE mollifier.
- Rule: Apply the IPC EE mollifier to the existing EE distance and closest-point formulation; do not add PP/PE degeneration unless the user explicitly requests it.

## 2026-08-04 — Tune LIMX scene framing feedback immediately

- Context: The first three-T-shirt box visualization was stable, but the user found the box visually too large relative to the garments.
- Mistake: The initial box footprint prioritized generous containment over the requested compact pile-up stress test.
- Rule: For visual scene-size feedback, update the contact planes, matching render geometry, and containment assertions together; run only the focused CUDA smoke test and relaunch the viewer promptly.

## 2026-08-04 — Diagnose sleeve-local jitter before retuning cloth globally

- Context: In the compact three-T-shirt box scene, the user observed overly strong bending and severe oscillation localized at sleeve openings despite the finite 300-frame rollout passing.
- Mistake: The containment regression did not distinguish excessive bending response from local self-contact churn around open boundary loops.
- Rule: For localized garment jitter, identify the affected mesh region and correlate its per-particle velocity with VF/EE/EF contact activation before changing global stiffness. Tune bending separately from the verified collision root cause, and validate localized temporal metrics on CUDA.

## 2026-08-04 — Classify sleeve EE churn by topology before adding damping

- Context: The settled three-T-shirt scene showed persistent sleeve-opening jitter, initially attributed broadly to undamped EE contact churn.
- Mistake: Aggregated all sleeve EE contacts before distinguishing one-ring same-garment pairs from nonlocal self-contact and inter-garment contact.
- Rule: Before adding global self-contact damping for localized garment jitter, classify active and churning EE pairs by topology locality and mollifier state. Treat dominant one-ring activation separately so a local thickness artifact does not cause broad physical-model changes.

## 2026-08-04 — Ignore one-ring EE pairs in LIMX self-collision

- Context: Settled sleeve jitter was traced to same-garment one-ring EE pairs grazing the collision-thickness threshold.
- Mistake: Retained Style3D's local half-edge-length thickness clamp even though these intrinsic neighbors dominated contact churn in the target garment scene.
- Rule: LIMX self-collision must skip topology-local one-ring EE pairs entirely. Continue handling nonlocal same-garment and inter-garment EE pairs, and keep the near-parallel mollifier only for those retained contacts.

## 2026-08-04 — Preserve one-ring EE support while diagnosing its chatter

- Context: Disabling topology-local one-ring EE pairs reduced contact churn numerically but made the compact three-T-shirt result visibly worse.
- Mistake: Equated the dominant churning contact class with the contact class that should be removed, without testing the structural support those contacts provide against local foldover.
- Rule: Keep Style3D's half-local-edge-length clamp for one-ring EE pairs. Diagnose chatter at persistent pair level—distance, activation threshold, closest parameters, normal continuity, adaptive stiffness, and force sign—before changing their response or topology filter.
## 2026-08-04 — Keep contact-type test dimensions synchronized

- Context: Adding experimental PE contacts to LIMX self-collision expanded overflow tracking from three buffers to four.
- Mistake: Updated the accumulated overflow vector and samples but left the expected zero vector at length three, causing a shape-only test failure.
- Rule: Whenever a contact type is added or removed, update buffer construction, runtime accumulation, and all expected count/overflow vector dimensions in the same patch before running the focused example test.
## 2026-08-04 — Derive surface PE/PP from filtered 3D parent stencils

- Context: An experimental LIMX kernel independently queried every cloth vertex against every nearby edge to fill EE endpoint regions.
- Mistake: This created thousands of PE constraints, including topology-local pairs that were not admitted by a valid surface collision parent stencil.
- Rule: For 3D triangle-surface self-collision, follow IPC's candidate structure: broad-phase FV and EE only, filter topology at the parent stencil, then evaluate PT/EE distance types that may reduce to PE or PP. Reserve independent EV/VV candidates for genuinely codimensional primitives.

## 2026-08-04 — Reject PE/PP as the primary sleeve-jitter cause

- Context: Filtered EE-parent PE/PP contacts with rest-distance clamping were tested in the settled three-T-shirt scene.
- Mistake: Treated the observed EE-to-PE feature transition as the likely root cause before verifying that filling the feature region actually reduced late-time motion.
- Rule: Keep the LIMX cloth experiment on strict VF/EE after the PE/PP variant failed to settle. Diagnose persistent jitter from contact-pair activation, normal continuity, adaptive stiffness, and force variation using matched CUDA rollouts before proposing another response change.

## 2026-08-05 — Bound cloth collision thickness by geometry

- Context: Reasoning about whether the 6 mm LIMX self-collision thickness should settle on the diagnosed sleeve patch.
- Mistake: Suggested that an arbitrary thickness should settle whenever the solver is sufficiently robust, without first requiring the thickness to lie in a geometrically admissible range; the infinite-thickness counterexample shows that finite mesh features and anchors can make the resulting contact targets mutually incompatible.
- Rule: Before judging solver stability, derive and enforce an admissible cloth collision-thickness range from physical thickness and local mesh feature size. Never assume every positive thickness admits a static equilibrium; treat an oversized thickness as an invalid geometric configuration rather than only a solver-convergence problem.

## 2026-08-05 — Prefer visual cloth validation over the full suite

- Context: The user asked to see whether the geometry-aware LIMX collision thickness fixes visible cloth jitter.
- Mistake: Ran the entire 5,496-test project suite even though focused LIMX tests had already passed and a rendered example was the requested evidence.
- Rule: For interactive cloth-stability work, first render and present one representative example plus its focused regression result. Run the full project suite only when the user requests release-level verification or before an integration action that requires it.

## 2026-08-05 — Execute explicit visualization variants directly

- Context: The user explicitly requested changing the three-T-shirt box scene to one T-shirt to isolate the source of visible jitter.
- Mistake: Paused for another approval after the requested variant, preserved parameters, and intended visual comparison were already unambiguous.
- Rule: When an interactive visualization request specifies the changed variable and comparison goal, make the minimal reversible configuration change and launch it directly. Do not ask for redundant approval; only stop for choices that materially change the experiment.

## 2026-08-05 — Verify the intended collision mode in visual comparisons

- Context: The one-T-shirt box variant still jittered after geometry-aware collision radii had stabilized the isolated EE patch.
- Mistake: Presented the one-vs-three garment run as evidence about the new thickness scheme without first checking that the box example actually passed `geometry_radius_scale`; it was still using uniform 6 mm self-collision thickness.
- Rule: Before launching a visual A/B collision experiment, inspect and report the exact active collision constructor parameters. Never attribute the result to a feature that the scene has not enabled.

## 2026-08-05 — Preserve the requested VF/EE rigid-cloth direction

- Context: Discussing how the Franka mesh should collide with LIMX cloth after the user explicitly chose vertex-face and edge-edge contact.
- Mistake: Replaced the requested VF/EE design with Newton's existing SDF rigid-soft path instead of answering whether reusable open-source VF/EE code had been found.
- Rule: Treat an explicitly selected collision representation as a fixed requirement. Investigate and cite its concrete implementation source before recommending a different representation; distinguish existing local code, externally reusable code, and a proposed extension.

## 2026-08-06 — Keep the Franka gripper above the table throughout its trajectory

- Context: The interactive cloth-grasp scene visibly drove the Franka gripper through the tabletop.
- Mistake: Validated cloth CCD activation without validating the complete commanded gripper geometry against the table over the full approach, pinch, lift, and release trajectory.
- Rule: Before presenting a robot-cloth trajectory, compute or test the minimum table clearance of every gripper collision primitive at every motion phase. Reject or clamp any target pose and interpolated segment that places the gripper at or below the tabletop, independently of cloth-contact behavior.

## 2026-08-09 — Treat tetrahedral ARAP as a Newton constraint

- Context: Designing a traditional Newton FEM path with tetrahedral ARAP energy and future Affine Body Dynamics coupling.
- Mistake: Proposed a separate FEM Newton solver even though the existing particle Newton solver already owns inertia, global assembly, and PCG, and ARAP only adds a four-particle energy stencil.
- Rule: Implement tetrahedral ARAP as a four-particle static constraint in the existing Newton solver. Reuse its global Newton assembly and linear solve; introduce a new solver only when genuinely different degrees of freedom, such as affine-body 12-DOF blocks, require a broader shared system.

## 2026-08-09 — Treat “开始” as authorization to implement

- Context: The bunny table-drop design was already approved and the user said “开始”.
- Mistake: Continued producing an implementation-plan document without creating the requested example code, leaving the user unable to find any implementation.
- Rule: After an approved design, interpret “开始” as authorization to implement immediately. Keep required planning brief and internal to the workflow, then create the requested code before reporting progress.

## 2026-08-09 — Resolve “刚才实现的代码” against completed work

- Context: While preparing the bunny follow-up, the user asked where the code implemented “刚才” was located.
- Mistake: Assumed the user meant the not-yet-created bunny example instead of the already-completed tetrahedral ARAP implementation.
- Rule: When the user asks where recently implemented code is, first map the reference to completed commits and report its branch/worktree and exact paths. Do not reinterpret it as the current follow-up feature when that feature has not been implemented yet.

## 2026-08-10 — Restrict tetrahedral soft-body collision to the boundary

- Context: Building VF/EE collision for eight volumetric ARAP bunnies.
- Mistake: Treated `ConstraintSelfCollision` as surface-only because its faces and edges come from the boundary triangle mesh, without noticing that its VF detector still launched over every tetrahedral particle, including interior vertices. Static-plane contact likewise considered every volume particle.
- Rule: For tetrahedral soft bodies, derive collision candidate vertices from the unique indices of the boundary triangle mesh. Use only those boundary vertices for VF and external plane/rigid contact; internal tetrahedral nodes participate through elasticity, never direct collision detection.

## 2026-08-10 — Keep the requested collision feature set to VF/EE

- Context: Stabilizing surface collision for the eight-bunny volumetric ARAP scene.
- Mistake: Kept edge-face (EF) intersection recovery enabled even though the intended collision formulation is limited to vertex-face and edge-edge pairs.
- Rule: For this LIMX collision path, detect and assemble only VF and EE contacts. Do not generate EF recovery contacts or include EF forces, Hessian-vector products, or diagonal blocks unless the user explicitly requests EF again.

## 2026-08-10 — Validate pile dynamics, not only numerical stability

- Context: The eight-bunny VF/EE scene remained stable but its upper layer did not visibly slide down after the walls were hidden.
- Mistake: Aligned every upper bunny almost exactly above a lower bunny and validated only positive volume, containment, and contact overflow. With friction `0.05`, the small lateral perturbations were dissipated and the scene settled into four stable columns.
- Rule: When a collision demo is intended to show pile rearrangement, verify per-object center-of-mass motion and friction sensitivity. Avoid vertically aligned initial layers unless stable stacking is the behavior being tested.

## 2026-08-10 — Measure friction thresholds before recommending coefficients

- Context: Diagnosing why the upper bunnies remained stacked in the VF/EE pile scene.
- Mistake: Recommended reducing self-collision friction from `0.05` to `0.01` after testing only the endpoints `0.05` and `0.0`.
- Rule: Do not infer a useful intermediate friction value from endpoint behavior. Run the proposed coefficient first and compare per-object center-of-mass trajectories; this scene remains stacked at `0.01` and only the tested frictionless case clearly slides.

## 2026-08-10 — Use the explicitly requested frictionless baseline

- Context: After the `0.01` bunny self-friction experiment still produced stable stacking, the user specified zero friction so the pile should collapse to the floor.
- Mistake: Proposed and launched an intermediate coefficient even though the already measured frictionless endpoint was the only configuration that clearly produced sliding.
- Rule: For the current bunny-pile experiment, set VF/EE self-collision friction exactly to `0`; keep box-plane friction separate unless the user asks to change it.

## 2026-08-10 — Preserve oriented tetrahedral boundary normals

- Context: Cross-bunny collision used boundary triangles extracted from positive-volume tetrahedra, but treated them like an unoriented cloth mesh.
- Mistake: VF detection replaced the oriented signed distance with its absolute value and flipped the response toward whichever side currently contained the vertex. Once a vertex crossed inside another bunny, this pushed it farther inward. EE likewise used only the closest-point separation vector and ignored incident outward face normals.
- Rule: Preserve the outward winding of tetrahedral boundary faces. For cross-component volume contact, use signed VF penetration and the target face's outward normal consistently in force and Hessian assembly; use incident outward face normals to disambiguate EE response. Keep component identity explicit, because the same one-sided half-space rule is not valid for arbitrary same-component self-collision, and retain CCD to prevent contacts from crossing beyond the activation band.

## 2026-08-10 — Defer CCD for oriented volume contact

- Context: Scoping the correction for tetrahedral bunny surface collision after confirming that the current VF/EE response discards outward normals.
- Mistake: Included CCD in the immediate correction even though the user only wants the discrete oriented response fixed first.
- Rule: Implement component-aware outward-normal VF/EE response within the existing discrete 3 mm candidate band first. Do not add swept collision detection, trajectory tests, or CCD infrastructure until the user explicitly requests that separate phase.

## 2026-08-10 — Use signed response for same-body volume self-contact

- Context: Designing oriented VF/EE collision for the tetrahedral bunny surfaces without CCD.
- Mistake: Proposed preserving the old unsigned formula for contacts within the same bunny while using signed outward normals only across different bunnies.
- Rule: Apply the oriented signed VF/EE formulation uniformly to all tetrahedral boundary contacts, including same-bunny self-collision. Do not branch the response formula on connected-component identity; component labels may remain diagnostic only.

## 2026-08-10 — Keep signed VF inside the discrete detection band

- Context: Enabling outward-normal signed VF on the eight-bunny volume scene.
- Mistake: Treated every `s < thickness` candidate from an expanded triangle AABB as active. Oblique triangle bounds admitted vertices much deeper than the 3 mm narrow phase, producing 15 mm contact depths and immediate tetrahedron inversion.
- Rule: Preserve `abs(signed_distance) < effective_thickness` as the discrete VF narrow band. Use the sign only for the outward direction and `effective_thickness - signed_distance` response within that band.

## 2026-08-10 — Measure topology-local contacts beyond one ring

- Context: Filtering false same-bunny contacts for signed VF/EE volume collision.
- Mistake: Assumed one-ring VF and adjacent-opposite EE filtering was sufficient because it removed every rest-state VF and most rest-state EE candidates. During deformation, all destabilizing same-bunny contacts were still within surface graph distance three, and the legacy local contacts had also been accidentally stiffening the soft body.
- Rule: Diagnose dynamic contact graph distances, not only rest-state counts. For this oriented volume path, exclude VF/EE pairs through three surface-graph rings and validate material stiffness again after removing artificial local-contact stiffness.

## 2026-08-10 — Keep permanent collision exclusions strictly incident

- Context: Revisiting the three-ring exclusion used by outward-normal signed VF/EE volume collision.
- Mistake: Turned a scene-specific stability observation into a permanent three-ring topology exclusion, which can suppress real local self-contact and allow undetected penetration.
- Rule: Permanently exclude only incident primitives: shared features and one-ring adjacency. Keep graph-distance-two and farther VF/EE pairs eligible for collision. Do not introduce a tetrahedron determinant barrier unless the user requests it separately.

## 2026-08-10 — Do not classify graph one-ring pairs as IPC-incident

- Context: Comparing the oriented tetrahedral VF/EE topology filter with libuipc's IPC candidate filtering.
- Mistake: Called graph one-ring adjacency “incident” and permanently excluded nonintersecting-index VF/EE pairs that IPC retains.
- Rule: Match IPC's strict primitive-incidence test: reject VF only when the vertex is one of the face vertices, and reject EE only when the edges share an endpoint. Keep graph-adjacent pairs with disjoint vertex sets eligible for contact.

## 2026-08-10 — Do not broaden Style3D EE locality ahead of EF-derived candidates

- Context: Estimating LIMX collision thickness while the future collision pipeline will derive VF/EE feature cases from edge-face parent candidates.
- Mistake: Proposed broadening Style3D's adjacent-opposite EE special case to every graph-one-ring edge pair based on two close bunny pairs, even though the planned EF parent stencil will naturally unify those local feature cases.
- Rule: Keep the current Style3D-local EE predicate for this experiment and do not redesign topology classification around graph hop distance. Estimate a feasible thickness interval instead: a discrete-detection lower bound and a geometry-based upper bound that prevents the nominal band from reaching two-ring pairs.

## 2026-08-10 — Estimate collision thickness from only the geometric upper bound

- Context: Choosing the automatic nominal VF/EE thickness while one-ring pairs receive special treatment and future EF parent candidates will unify feature cases.
- Mistake: Added a velocity-dependent discrete-detection lower bound and proposed selecting from a feasible interval, despite the requested policy being a simple geometry-only cap.
- Rule: For this experiment, compute the nominal thickness as `min(eta * two_ring_upper_bound, 0.005 m)` with `eta = 0.8`. Do not add a motion-derived lower bound, CCD criterion, or graph-one-ring redesign to this estimator.

## 2026-08-10 — Use the canonical dev branch without worktrees

- Context: Finishing the automatic LIMX collision-thickness change in a hidden linked worktree.
- Mistake: Continued using a feature worktree and offered local merge or agent-created PR flows, adding friction to the user's preferred daily workflow.
- Rule: For Newton, keep only `main` and `dev` as the normal working branches. Develop, commit, and push from the canonical repository path on `dev`; the user opens the PR from `dev` into `main`. Do not create or use worktrees unless the user explicitly requests one.

## 2026-08-10 — Treat visual twist behavior as the acceptance criterion

- Context: Testing automatic LIMX collision thickness in the cloth twist scene against the user's more stable Chysx reference.
- Mistake: Treated finite state and a passing 600-frame rollout as sufficient evidence even though the visible twist behavior was wrong and the scene had not been matched to the reference implementation.
- Rule: For twist stability comparisons, visually validate the motion and first match the reference scene's mesh, material, boundary drive, contact parameters, timestep, and solver iterations. Do not attribute a difference to collision thickness while the compared scene configurations are unmatched.

## 2026-08-10 — Preserve EF untangling when evaluating twist stability

- Context: Comparing LIMX twist collision settings against the Chysx reference after observing severe motion with strong EF recovery.
- Mistake: Recommended disabling EF because the velocity metrics settled, without checking whether the cloth had already self-intersected and remained topologically tangled.
- Rule: A twist result is acceptable only if it both settles and remains penetration-free. Keep EF recovery available for already-crossed edge-face pairs; diagnose and tune its direction, persistence, depth, stiffness, and Hessian instead of declaring a penetrated but motionless state stable.

## 2026-08-10 — Run the actual Chysx reference before tuning Newton twist

- Context: Retuning LIMX twist from source comments and headless EF/velocity metrics while the user expected the visibly stable Ai-Physics Chysx result.
- Mistake: Continued proposing Newton parameter combinations without first building and running the reference environment; the resulting scene still looked bad despite favorable aggregate metrics.
- Rule: Stop Newton twist tuning until the Ai-Physics Chysx reference is built and visually reproduced. Use its actual executable scene, mesh, drive, material, collision, timestep, and solver configuration as the comparison baseline; do not substitute inferred settings or headless settling metrics for the reference animation.

## 2026-08-11 — Attribute the twist difference to the scene before topology-local EE

- Context: Comparing the stable `newton-ChysX` twist animation with the coarse LIMX twist scene.
- Mistake: Continued presenting ChysX's explicitly enumerated adjacent EE candidates as a possible stability explanation after the measured scene mismatch was already dominant.
- Rule: For this comparison, treat scene configuration as the working cause. First align mesh resolution and dimensions, boundary drive, mass, material, collision thickness and stiffness, timestep, and solver iterations. Do not change or attribute behavior to topology-local EE candidate generation unless a later matched-scene A/B test specifically isolates it.

## 2026-08-11 — Validate the post-drive hold phase in twist scenes

- Context: Declaring the aligned LIMX twist scene stable from a 1,000-frame rollout whose boundary targets were still rotating at the final frame.
- Mistake: Reported final-frame EF and finite-state metrics as evidence for settling even though the simulation never entered a fixed-target hold phase; the visible violent jitter appeared only after rotation stopped.
- Rule: Twist stability requires an explicit post-drive hold interval. Record VF, EE, EF, overflow, kinetic speed, and contact depth across both the driven and fixed-target phases; never infer settling from a rollout that ends while the anchors are still moving.

## 2026-08-11 — Isolate collision handling before changing twist material energy

- Context: Diagnosing post-drive LIMX twist jitter after confirming intermittent EF recovery contacts.
- Mistake: Treated ChysX's rotated-axis material energy as a likely cause and combined a material replacement with an EF Hessian change in one A/B, so the observed improvement could not identify the collision-side cause the user wanted investigated.
- Rule: Keep the existing LIMX material energy fixed for this diagnosis. Isolate collision variables one at a time—EF Hessian coupling, recovery thickness/target, contact persistence, direction, and stiffness—and attribute improvement only to a collision-only A/B.

## 2026-08-11 — Do not present diagonal EF Hessians as inherently more stable

- Context: Explaining the diagonal-only five-particle EF Hessian used during the LIMX twist investigation.
- Mistake: Described dropping every cross-particle EF Hessian block as definitively more stable, even though the observed twist behavior did not establish that claim and the user wanted the physically stronger five-point coupling retained.
- Rule: Treat diagonal-only EF assembly as an experimental approximation, not a stability guarantee. For the current Newton LIMX path, keep the full coupled rank-one EF Hessian unless a controlled A/B with the target scene demonstrates a reason to approximate it.

## 2026-08-11 — Preserve the authored twist drive while changing collision

- Context: Aligning and debugging the LIMX twist scene against the ChysX collision implementation.
- Mistake: Changed the scene's rotation speed and total twist angle together with mesh and collision settings, making the visible motion differ independently of the collision algorithm.
- Rule: Restore and preserve the twist example's authored drive duration and total angle when modifying self-collision. Collision experiments may change contact behavior, but must not change rotation speed, rotation duration, or final twist angle unless the user explicitly requests it.

## 2026-08-11 — Use newton-ChysX as the complete twist-scene authority

- Context: The user clarified the intended meaning of preserving the twist speed and angle after an attempted restoration of the old LIMX drive.
- Mistake: Restored LIMX's former four-second smoothstep drive instead of treating `newton-ChysX/newton/examples/chysx/cloth/example_chysx_twist.py` as the requested scene reference.
- Rule: For the current twist comparison, copy the newton-ChysX drive exactly: constant `1 rad/s`, `rot_end_time = 25 s`, and the reference default frame count. Keep any early-stop/hold schedule in diagnostic scripts only, not in the example.

## 2026-08-11 — Reject automatic cloth thickness that activates the flat rest mesh

- Context: Switching the aligned LIMX twist scene from its manual 1.212 mm thickness to the geometry-based automatic formula returned the 5 mm cap.
- Mistake: Reported the estimated value without first checking that a flat, penetration-free rest mesh generated zero nonincident VF/EE contacts under the exact runtime topology filters.
- Rule: Validate every automatic cloth thickness against the real VF/EE detector in the flat rest pose. Reject or reduce any estimate that activates topology-local rest pairs; the geometric bound and runtime candidate/filter definitions must cover the same primitive classes.

## 2026-08-11 — Evaluate automatic thickness after one-ring special handling

- Context: The 5 mm automatic twist thickness activated flat-rest VF pairs that should have been governed by the cloth's topology-local thickness rule.
- Mistake: Evaluated the nominal automatic thickness independently of the one-ring VF/EE special thickness, so the estimator's exclusions and the runtime detector used different effective contact bands.
- Rule: Compute and validate automatic cloth thickness with topology-local VF/EE special handling enabled. The final effective per-pair thickness—not the nominal scalar alone—must produce zero VF/EE contacts in the penetration-free rest state.

## 2026-08-11 — Do not recompute automatic rest-geometry thickness on viewer reset

- Context: Enabling automatic thickness in the 20,000-particle twist scene made the interactive Reset action visibly stall.
- Mistake: Put the Python/CPU two-ring geometry scan directly in `ConstraintSelfCollision` construction, so every viewer reset recomputed a rest-topology quantity that is unchanged.
- Rule: Measure constructor cost when enabling geometry-derived parameters in interactive examples. Cache or precompute rest-topology thickness data so Reset does not rerun an expensive invariant calculation.

## 2026-08-11 — Compute automatic collision-thickness bounds in parallel on GPU

- Context: The 20,000-particle twist mesh spent most of Reset in millions of serial Python VF/EE distance evaluations.
- Mistake: Started optimizing scalar CPU arithmetic and caching instead of moving the naturally independent candidate-distance evaluations to the GPU.
- Rule: Build fixed two-ring VF/EE candidate stencils once, evaluate their rest distances in parallel on the model device, and obtain the upper bound with a GPU minimum reduction/atomic reduction. Do not retain a serial Python distance loop as the production path for large meshes.
