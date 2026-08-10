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
