# Changelog

## [Unreleased]

### Added

- Add 3D texture-based SDF, replacing NanoVDB volumes in the mesh-mesh collision pipeline for improved performance and CPU compatibility.
- Add `--benchmark [SECONDS]` flag to examples for headless FPS measurement with warmup
- Interactive example browser in the GL viewer with tree-view navigation and switch/reset support
- Add `TetMesh` class and USD loading API for tetrahedral mesh geometry
- Support kinematic bodies in VBD solver
- Add reduced elastic body links with floating-frame modal coordinates, VBD support for revolute-joint attachments, and deformed Viewer mesh rendering
- Add rotational joint coupling for reduced elastic links in `SolverVBD`, so a fixed, prismatic, or revolute attachment transmits a clamp moment into modal bending via per-endpoint angular mode shapes (`ModalBasis.sample_psi`); the coupling is bidirectional, so the beam's modal twist reacts back on the attached rigid body
- Add angular mode shape (`ModalBasis.sample_psi`) support across the modal generators: `ModalGeneratorBeam` populates it analytically in closed form, `ModalGeneratorSampled` accepts `sample_psi` directly, and `ModalGeneratorPOD` and `ModalGeneratorFEM` can estimate it from the displacement-gradient curl via `derive_psi` (off by default), so any reduced elastic basis can drive joint rotational coupling
- Add sampled `ModalBasis` and modal generators for reusable reduced elastic mode construction
- Add `ModalGeneratorFEM` for matrix-based reduced elastic modes from nodal mass, stiffness, and damping matrices
- Add `ModalGeneratorCraigBampton` for retaining named six-DOF interfaces and fixed-interface modes from vendor-neutral reduced mass, stiffness, damping, and recovery arrays
- Add `ModalBasis` floating-frame inertia coupling integrals (`sample_mass`, `mode_coupling_linear`, `mode_coupling_angular`, `mode_coupling_centrifugal`, `mode_coupling_coriolis`), computed exactly by `ModalGeneratorFEM` and by lumped quadrature from per-sample masses; `mode_mass` is now optional and derived from `sample_mass` when omitted
- Add floating-frame gravity and translational-acceleration coupling for reduced elastic modes in `SolverVBD` (modal force `S·(g − a)`), so a basis with per-sample masses sags under gravity and is excited by base motion
- Add floating-frame rotational coupling for reduced elastic modes in `SolverVBD` (Euler, centrifugal, and Coriolis modal forces), so a basis with per-sample masses responds to angular acceleration and steady rotation of its frame
- Add the reduced elastic modal-to-frame inertia coupling in `SolverVBD` (the back-reaction force and matching moment), so a free-floating elastic body's modal vibration conserves linear and angular momentum
- Add the reduced elastic modal-to-frame velocity reactions in `SolverVBD`, so a free-floating elastic body that rotates while it deforms conserves momentum
- Add a reduced elastic gravity coupling example contrasting a coupled and an uncoupled cantilever under self-weight
- Add a reduced elastic base excitation example contrasting a coupled and an uncoupled cantilever on a vertically oscillating base
- Add a reduced elastic base rotation example demonstrating the floating-frame rotational (Euler) coupling and its invariance to reference-frame placement
- Add a reduced elastic centrifugal example where a spinning hub stretches a coupled beam through an axial mode while an uncoupled beam stays rigid
- Add a reduced elastic Coriolis example where a spinning hub turns a coupled beam's plucked bending mode toward the perpendicular direction while an uncoupled beam keeps swinging in its original plane
- Add a reduced elastic frame coupling example where a coupled free-floating beam recoils against its plucked bump mode to keep its center of mass fixed while an uncoupled beam lets the center of mass slosh
- Add a reduced elastic angular frame coupling example where a coupled free-floating beam's plucked antisymmetric bending mode rotates the frame to conserve angular momentum while an uncoupled beam's frame stays fixed
- Add a reduced elastic clamp moment example where a beam clamped between a fixed wall and a spinning drum twists through the joint rotational coupling while an uncoupled beam stays flat
- Add a reduced elastic cantilever vibration example with finite modal mass dynamics
- Add a reduced elastic prismatic compression example with Poisson bulging and parent rotation validation
- Add reduced elastic crank-slider, Watt linkage, and bell-crank examples with analytic geometry checks
- Add reduced elastic gravity examples for a tip-weight cantilever and a suspended vertical bar
- Add a reduced elastic matrix ROM bracket example driven through rigid fixed-joint interfaces
- Add a flexible dipper arm example with a prismatic actuator loop and suspended rigid payload
- Add a UR10 reduced elastic car-panel handling example with offline-decimated STL mesh loading support
- Add reduced elastic surface contacts in VBD, including separate wall-pad, two-gripper pickup, and scraper contact examples
- Add a reduced elastic monobloc chair stick-slip example using a cached CC0 Poly Haven asset
- Add a fixed joint stiffness mode to `SolverVBD` for non-adaptive joint penalty solves
- Add optional per-vertex elastic displacement coloring in ViewerGL
- Add brick stacking example
- Add box pyramid example and ASV benchmark for dense convex-on-convex contacts

### Changed

- Tune the reduced elastic chair friction for a stable stick-slip transition with complete contact assembly
- Tune reduced elastic torsion and cantilever vibration examples for smoother and faster motion
- Tune reduced elastic mechanism and gravity examples for clearer visible deformation and camera framing
- Increase elastic box render tessellation and make the torsion fixture a held 90 degree shaft release with exact surface modal samples
- Unify heightfield and mesh collision pipeline paths; the separate `heightfield_midphase_kernel` and `shape_pairs_heightfield` buffer are removed in favor of the shared mesh midphase
- Replace per-shape `Model.shape_heightfield_data` / `Model.heightfield_elevation_data` with compact `Model.shape_heightfield_index` / `Model.heightfield_data` / `Model.heightfield_elevations`, matching the SDF indirection pattern
- Standardize `rigid_contact_normal` to point from shape 0 toward shape 1 (A-to-B), matching the documented convention. Consumers that previously negated the normal on read (XPBD, VBD, MuJoCo, Kamino) no longer need to.
- Replace `Model.sdf_data` / `sdf_volume` / `sdf_coarse_volume` with texture-based equivalents (`texture_sdf_data`, `texture_sdf_coarse_textures`, `texture_sdf_subgrid_textures`)
- Render inertia boxes as wireframe lines instead of solid boxes in the GL viewer to avoid occluding objects

### Deprecated

### Removed

- Remove `Model.elastic_endpoint_joint`, `Model.elastic_endpoint_side` and `Model.elastic_endpoint_body`, which no longer have any consumer. Use `Model.joint_parent_elastic_endpoint` and `Model.joint_child_elastic_endpoint` to map a joint to its endpoint index and side

### Fixed

- Solve each reduced elastic body's floating frame and modal coordinates in one
  coupled VBD block so joint response no longer depends on slow frame/modal
  fixed-point convergence.
- Keep rigid-frame and modal contact Hessians consistent when a per-body contact
  list overflows, and apply tiled elastic updates from tile-local solve results.
- Resolve USD asset references recursively in `resolve_usd_from_url` so nested stages are fully downloaded
- Unify CPU and GPU inertia validation to produce identical results for zero-mass bodies with `bound_mass`, singular inertia, non-symmetric tensors, and triangle-inequality boundary cases
- Fix `UnboundLocalError` crash in detailed inertia validation when eigenvalue decomposition encounters NaN/Inf input
- Handle NaN/Inf mass and inertia deterministically in both validation paths (zero out mass and inertia)
- Update `ModelBuilder` internal state after fast-path (GPU kernel) inertia validation so it matches the returned `Model`
- Fix MJCF mesh scale resolution to use the mesh asset's own class rather than the geom's default class, avoiding incorrect vertex scaling for models like Robotiq 2F-85 V4
- Fix viewer crash with `imgui_bundle>=1.92.6` when editing colors by normalizing `color_edit3` input/output in `_edit_color3`
- Show prismatic joints in the GL viewer when "Show Joints" is enabled
- Fix body `gravcomp` not being written to the MuJoCo spec, causing it to be absent from XML saved via `save_to_mjcf`
- Fix `eq_solimp` not being written to the MuJoCo spec for equality constraints, causing it to be absent from XML saved via `save_to_mjcf`
- Fix WELD equality constraint quaternion written in xyzw format instead of MuJoCo's wxyz format in the spec, causing incorrect orientation in XML saved via `save_to_mjcf`
- Fix loop joint coordinate mapping in the MuJoCo solver so joints after a loop joint read/write at correct qpos/qvel offsets
- Fix viewer crash when contact buffer overflows by clamping contact count to buffer size
- Decompose loop joint constraints by DOF type (WELD for fixed, CONNECT-pair for revolute, single CONNECT for ball) instead of always emitting 2x CONNECT
- Render reduced elastic shape meshes double-sided so thin imported shells remain visible in ViewerGL
- Fix reduced elastic angular joint coupling for rotation-only constraints, modal endpoint damping, finite multi-axis rotations, and revolute drives and limits

## [1.0.0] - 2026-03-10

Initial public release.
