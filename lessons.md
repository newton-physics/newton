# Project Lessons

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
