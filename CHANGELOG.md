# Changelog

## [Unreleased]

### Added

- Add composable actuator subsystem with pluggable `Controller` (`ControllerPD`, `ControllerPID`, `ControllerNeuralMLP`, `ControllerNeuralLSTM`), `Clamping` (`ClampingMaxEffort`, `ClampingDCMotor`, `ClampingPositionBased`), and `Delay` components; supports per-DOF delays, CUDA graph capture, and masked environment reset
- Add heatmap rendering for scalar arrays logged through `ViewerGL.log_array()`
- Add Blender-style orbit, pan, and dolly controls to the GL viewer using middle-mouse drag combinations
- Add `SolverXPBD.update_contacts()` to populate `contacts.force` with per-contact spatial forces (linear force and torque) derived from XPBD constraint impulses
- Add `body_parent_f` extended state attribute support to `SolverXPBD` so it populates per-body incoming joint wrenches in world frame at the body's COM (matches `SolverMuJoCo`'s convention; values are approximate due to XPBD's relaxation and non-momentum-conserving nature)
- Add `body_parent_f` extended state attribute support to `SolverFeatherstone` populated directly from the RNEA backward pass (per-body net spatial wrench translated to the body's COM, matching the `SolverMuJoCo` convention)
- Add public `newton.geometry.build_bvh_shape()`, `build_bvh_particle()`, `refit_bvh_shape()`, and `refit_bvh_particle()` helpers for managing model BVHs
- Raise process priority automatically in `--benchmark` mode for more stable measurements; add `--realtime` for maximum priority.
- Import per-shape authored color from USD stages into `ModelBuilder.shape_color`
- Add `TRIANGLE_PRISM` support-function type for heightfield triangles, extruding 1 m along the heightfield's local -Z so GJK/MPR naturally resolves shapes on the back side
- Add `ViewerGL.log_scalar()` for live scalar time-series plots in the viewer
- Add `Mesh.is_watertight` property (cached) that reports whether every geometric edge is shared by exactly two triangles
- Add `deterministic` flag to `CollisionPipeline` and `NarrowPhase` for GPU-thread-scheduling-independent contact ordering via radix sort and deterministic fingerprint tiebreaking in contact reduction
- Add fast parity-based SDF construction path for watertight meshes in `SDF.create_from_mesh`, using `wp.mesh_query_point_sign_parity` instead of winding numbers; selected via the new `sign_method` argument (`"auto"` — the default — picks parity when `Mesh.is_watertight` is true, or `"parity"` / `"winding"` to force either strategy)
- Add `Viewer.log_image()` for displaying single or batched images in `ViewerGL`; other backends inherit a no-op. Also add `SensorTiledCamera.utils.to_rgba_from_color()`, `to_rgba_from_normal()`, `to_rgba_from_depth()`, and `to_rgba_from_shape_index()` (hash palette or caller-provided RGB lookup) adapters producing output consumable by `log_image()`.
- Add on-disk caching of cooked texture-based SDFs via the new `cache_dir` argument on `SDF.create_from_mesh` and `Mesh.build_sdf`. Cached entries are content-addressed by mesh and build parameters, written atomically as a single uncompressed `.npz`, and versioned via `CACHE_FORMAT_VERSION` so format changes invalidate stale caches transparently
- Enable CPU execution of the collision pipeline, including mesh–mesh and mesh–heightfield SDF contacts and contact reduction (`reduce_contacts`) that were previously CUDA-only, by replacing the CUDA `__shared__` fast paths in `sdf_contact.py`, `multicontact.py`, and `collision_core.py` with portable `wp.tile_stack` / `wp.tile_mesh_query_aabb` primitives. CPU runs now execute the same kernels as CUDA; the previous `"NarrowPhase running on CPU: mesh-mesh contacts will be skipped"` warning is no longer emitted.
- Add `ViewerBase.log_arrows()` for arrow rendering (wide line + arrowhead) in the GL viewer with a dedicated geometry shader
- Add frame-to-frame contact matching via `CollisionPipeline(contact_matching=...)` with modes `"latest"` (populates `contacts.rigid_contact_match_index`) and `"sticky"` (experimental; additionally replays previous-frame contact geometry on matched contacts — the sticky update strategy may change without warning). Optional `contact_report=True` exposes new/broken contact index lists on `Contacts`.
- Add VBD rigid-contact warm-starting via `rigid_contact_history`, backed by `Contacts.rigid_contact_match_index` from `CollisionPipeline(contact_matching="latest")`.
- Add VBD hard/soft controls for body-body contacts and structural joint slots, including `rigid_contact_hard`, `SolverVBD.set_joint_constraint_mode()`, and `SolverVBD.JointSlot`
- Add AVBD contact/joint alpha overrides and linear/angular beta overrides to `SolverVBD` for stabilization and penalty-ramping control
- Add `enable_multiccd` parameter to `SolverMuJoCo` for multi-CCD contact generation (up to 4 contact points per geom pair)
- Warn when `SolverMuJoCo` detects installed `mujoco` or `mujoco-warp` versions that do not satisfy `pyproject.toml` requirements
- Support `<joint type="ball"/>` in the MJCF importer, and preserve authored damping, stiffness, and frictionloss when exporting ball joints to MuJoCo specs (previously silently dropped)
- Add `ViewerViser.log_scalar()` for live scalar time-series plots via uPlot
- Honor `UsdGeomImageable` visibility (including inherited `invisible`) on USD prims imported via `ModelBuilder.add_usd()`; visual shapes, gaussian splats, and collider shapes are imported with `ShapeFlags.VISIBLE` cleared when the prim is effectively invisible, while collision behavior is preserved
- Add import of `UsdGeom.TetMesh` prims as soft meshes through `ModelBuilder.add_usd()`
- Add site-targeted actuator support to MuJoCo solver
- Add USD parsing support for equality constraints based on the `MjcEquality` schema
- Add more solver options to implicit MPM: `gs-soa` (or `gauss-seidel-soa`) for improved memory coalescing, `gs-batched` (or `gauss-seidel-batched`) merging GS colors with Jacobi-style mass-split parallelism, plus `cr` (Conjugate Residual) and `gmres` linear solver options.
- Add frame-by-frame step support to `ViewerGL`: press `.` while paused to advance one simulation frame
- Add ViewerBase.should_step() — call once per frame to determine whether the simulation loop should advance; returns True when not paused.
- Add Kamino-specific simulation examples in `newton/examples/kamino`
- Add per-mesh `color` override to `ViewerBase.log_mesh()` for tinting individual meshes without authoring per-vertex colors
- Add per-mesh `roughness` and `metallic` PBR overrides to `ViewerBase.log_mesh()`

### Changed

- Use pre-computed local AABB for `CONVEX_MESH` shapes in `compute_shape_aabbs`, avoiding a per-frame support-function AABB computation
- Build mesh SDFs via the texture-based sparse path only; sample via `SDF.texture_data` instead of `SDF.sparse_volume` / `SDF.coarse_volume`.
- Change implicit MPM default `solver` from `"gs"` to `"auto"`, which selects `"gs"` for trilinear bases and `"gs-batched"` for higher-order ones. Set `solver="gs"` explicitly to restore the previous behavior.
- Change `SolverImplicitMPM.Config.solver` warmstart syntax from `+`-separated strings to ordered sequences; use `solver=("cg", "gauss-seidel")` instead of `solver="cg+gauss-seidel"`.
- Change implicit MPM default `collider_basis` from `"Q1"` to `"S2"` for improved contact quality; set `collider_basis="Q1"` explicitly to restore the previous behavior.
- Change GL viewer scroll to dolly toward the orbit pivot; use Ctrl+scroll for FOV zoom
- Render all GL viewer lines (joints, contacts, wireframes) as geometry-shader quads instead of ``GL_LINES`` for uniform width across zoom levels and non-square viewports
- Adjust grouping of `reset`, `step`, and `pause` controls so they appear together
- Bump `Pillow` floor to `>=11.3.0`
- Bump `jupyterlab` lower bound to `>=4.5.7` to pick up the fix for CVE-2026-40171
- Replace `ModelBuilder.add_actuator(actuator_class, input_indices=..., output_indices=..., **kwargs)` with `ModelBuilder.add_actuator(controller_class, index=..., clamping=[...], delay_steps=..., pos_index=..., **ctrl_kwargs)` where each call registers a single DOF
- Change `ArticulationView.get_actuator_parameter(actuator, name)` and `set_actuator_parameter(actuator, name, values)` to require a `component` argument identifying the owning `Controller`, `Clamping`, or `Delay` instance: `get_actuator_parameter(actuator, actuator.controller, "kp")`
- Update default environment map texture in GL viewer (source: https://polyhaven.com/a/brown_photostudio_02)
- Remove the implicit-MPM rasterized collider's reliance on Warp's `warp.fem` module (behavior unchanged)
- Replace the StVK VBD triangle membrane material with the stable Neo-Hookean form (Smith et al. 2018, adapted to 2D shells). The upstream two-constraint Rayleigh damping model is preserved unchanged
- Bump `mujoco` and `mujoco-warp` dependencies to `~=3.8.0` (`mujoco-warp` requires `>=3.8.0.1`)
- Bump `GitPython` lower bound to `>=3.1.47` to pick up the fix for GHSA-x2qx-6953-8485 (`multi_options` argument injection in `Repo.clone_from`)
- Bump `open3d` floor to `>=0.19.0`
- Bump `meshio` floor to `>=5.3.5`; `5.3.0` calls `np.string_` which was removed in NumPy 2.0
- Restrict `usd-core` to `<26.5` to avoid deprecation warnings introduced in 26.5
- Require explicit `SensorTiledCamera` BVH lifecycle management instead of implicit camera maintenance: call `newton.geometry.build_bvh_shape()` / `build_bvh_particle()` once after setup, then `refit_bvh_shape()` / `refit_bvh_particle()` before rendering frames that change geometry
- Increase conveyor rail roughness in `example_basic_conveyor` to reduce mirror-like reflections
- Migrate all raycast logic to `geometry.raycast`, all raycast functions now return distance and normal information
- Disable process reuse in the test runner on multi-GPU systems to prevent CUDA errors from cascading across test suites, keeping process reuse enabled on single-GPU systems for faster throughput
- Default `python -m newton.examples` with no argument to launch `basic_pendulum`; use `--list` to print available examples
- Reduce default `stretch_stiffness` from `1.0e9` to `1.0e5` in `add_joint_cable()`, `add_rod()`, and `add_rod_graph()`
- Treat `stretch_stiffness` and `bend_stiffness` in `add_rod()` and `add_rod_graph()` as direct per-joint stiffness values, matching `add_joint_cable()` and other joint stiffness APIs
- VBD solver uses augmented-Lagrangian hard constraints for body-body contacts by default (`rigid_contact_hard=True`)
- Reduce collision-pipeline overhead in `SolverMuJoCo` via incremental contact conversion when the contact set is unchanged (~6× speedup on `example_robot_anymal_d` with 4096 worlds)

### Deprecated

- Deprecate `SensorRaycast` in favor of `SensorTiledCamera`; migrate to `SensorTiledCamera.utils.compute_pinhole_camera_rays()` and `create_depth_image_output()` for single-camera depth rendering — see the `SensorRaycast` class docstring for a complete migration example
- Deprecate and ignore `rigid_enable_dahl_friction` in `SolverVBD`; Dahl friction is now auto-detected from model attributes (`model.vbd.dahl_eps_max` / `model.vbd.dahl_tau`)
- Deprecate `newton-actuators` package dependency; all actuator functionality is now built into `newton.actuators`. The dependency is kept for backward compatibility and will be removed in a future release; migrate imports from `newton_actuators` to `newton.actuators`

### Fixed

- Fix `remesh_convex_hull` raising `QhullError` on degenerate (coincident, collinear, or coplanar) point clouds; it now returns a zero-volume fallback mesh with a `UserWarning`, raises `ValueError` on empty input, and retries Qhull with `QJ` joggle as a last resort on the 3D path
- Fix `ViewerGL` Step button remaining clickable while the simulation is running; the button is now greyed out when not paused
- Fix the example viewer's Reset button discarding user-provided CLI options (e.g. `--world-count`) and rebuilding the example with parser defaults instead
- Fix `ModelBuilder.finalize()` crashing with 3+ articulations after `collapse_fixed_joints()` reordered `articulation_start` and dropped per-articulation metadata
- Fix Sphinx docs builds to auto-discover bundled ``pypandoc_binary`` pandoc so notebook tutorials build without manual PATH configuration
- Fix `SolverStyle3D` initialization to precompute its fixed PD matrix from the finalized model
- Fix connect constraint anchor computation to account for joint reference positions when `SolverMuJoCo` is the chosen solver.
- Fix joint-synthesized CONNECT constraint anchors not updating when `dof_ref` or `joint_X_p` changes at runtime via `notify_model_changed()`
- Fix WELD constraint data corruption when a model contains both FIXED and revolute/ball loop joints
- Fix `SolverMuJoCo` passing non-zero geom/pair margins to `mujoco_warp.put_model()`, which fails when NATIVECCD is enabled. Margins are forced to zero when MuJoCo handles collisions (`use_mujoco_contacts=True`); the Newton collision pipeline (`use_mujoco_contacts=False`) is unchanged
- Fix GPU illegal-memory-access in `SolverMuJoCo` Newton-contacts fast path when `notify_model_changed(BODY_INERTIAL_PROPERTIES | JOINT_DOF_PROPERTIES | MODEL_PROPERTIES)` was called between substeps (e.g. mass randomization in IsaacLab), or when the bound `Contacts` instance / MJWarp `naconmax` changed without invalidating the cached `tid_to_cid` mapping. The fast path is now invalidated on any property notify that affects cached MJWarp contact fields, and bounds-checks `cid` against `naconmax` defensively
- Fix `State.assign` not copying namespaced extended and custom state attributes 
- Fix mesh-convex back-face contacts generating inverted normals that trap shapes inside meshes and cause solver divergence (NaN)
- Fix finite plane geometry 2x too large in collision, bounding sphere, and raytrace sensor
- Fix MPR convergence failure on large and extreme-aspect-ratio mesh triangles by projecting the starting point onto the triangle nearest the convex center
- Fix O(W²·S²) memory explosion in `CollisionPipeline` shape-pair buffer allocation for NXN and SAP broad phase modes by computing per-world pair counts instead of a global N²
- Fix `SensorRaycast` ignoring `PLANE` geometry
- Fix VRAM leak when resetting examples that allocate large GPU state (e.g. `diffsim_bear`)
- Fix `SensorRaycast` and viewer picking ignoring `HFIELD` (heightfield) geometry
- Fix `SensorTiledCamera` textured albedo output rendering flat colors when color and normal outputs are disabled
- Fix `contacts_rj45_plug` example crashing on reset
- Fix `SolverMuJoCo` dependency version-mismatch warning being silently skipped when Newton is installed from a wheel
- Fix `ViewerGL.log_image()` windows persisting across example-browser switches and failing to re-open on re-entry after manual close, by clearing the image logger in `ViewerGL.clear_model()`
- Fix `ModelBuilder.add_shape_heightfield` `scale` being ignored by narrow-phase collision and raycast
- Fix `collision_filter_parent` silently ignored on joints to world (`parent=-1`); now honored for all `add_joint_*` methods, with `add_joint_fixed(parent=-1, ...)` defaulting to filter child shapes against world-static shapes
- Fix multi-world `qfrc_actuator` conversion using the wrong body center of mass for worlds with `worldid > 0`
- Fix `SolverMuJoCo.__init__` time scaling with `world_count × actuators_per_world` instead of `actuators_per_world` by vectorizing the template-world filter for site-targeted actuators
- Fix compressed tets in `evaluate_volumetric_neo_hookean_force_and_hessian` producing an indefinite Hessian by clamping the cofactor-derivative coefficient to `max(0, s)`, removing a contribution that could corrupt the VBD inner solve
- Fix SDF hydroelastic broadphase scatter kernel using a grid-stride loop with binary search instead of per-pair thread launch
- Fix box support-map sign flips from quaternion rotation noise (~1e-14) producing invalid GJK/MPR contacts for face-touching boxes with non-trivial base rotations
- Fix USD import of multi-DOF joints from MuJoCo-converted assets where multiple revolute joints between the same two bodies caused false cycle detection; merge them into D6 joints with correct DOF label mapping for MjcActuator target resolution
- Fix USD `MjcActuator` import so position and velocity actuators populate Newton's joint target arrays and can be driven via `Control.joint_target_pos` / `Control.joint_target_vel`
- Fix MJCF importer creating finite planes from MuJoCo visual half-sizes instead of infinite planes
- Fix USD custom-frequency parsing to respect `ModelBuilder.add_usd(root_path=...)`, avoiding rows from sibling subtrees
- Fix USD import of joint limit stiffness/damping from `MjcJointAPI`: `SchemaResolverMjc` now reads the schema-correct `mjc:solreflimit` attribute instead of the generic `mjc:solref`, which was never authored on joints
- Fix MJCF importer in `compiler.angle="degree"` mode: (1) stop multiplying joint `damping`/`stiffness` by `180/π` (MuJoCo stores these in `N·m·s/rad` and `N·m/rad` regardless of `angle`); (2) stop `deg2rad`-scaling the default `±MAXVAL` sentinel for joints without an explicit `range=`, which was turning unlimited hinges into bounded joints with `~1.75e8 rad` range
- Fix MJCF importer ignoring explicit `mass=` on visual geoms loaded via `parse_visuals=True`; authored visual-only mass now contributes to body mass and inertia like visual-only density already does
- Fix ViewerViser mesh popping artifacts caused by viser's automatic LOD simplification creating holes in complex geometry
- Fix ViewerViser notebook recording playback to load the matching browser client from the installed `viser` package and bind the playback HTTP server to loopback only
- Fix rendering of planes in ViewerViser as finite grids of line segments to prevent flickering artifacts
- Fix degenerate zero-area triangles in SDF marching-cubes isosurface extraction by clamping edge interpolation away from cube corners and guarding against near-zero cross products
- Fix multi-world coordinate conversion using the wrong body center of mass for replicated worlds
- Fix MJCF importer ignoring `<default><equality/></default>` attribute defaults (e.g. `solref`, `solimp`) for `<connect>`/`<weld>`/`<joint>` equality constraints
- Remove incorrect body-level `mjc:damping` -> `rigid_body_linear_damping` mapping from `SchemaResolverMjc`; `mjc:damping` is defined on `MjcJointAPI`, not on bodies
- Fix `target_voxel_size` being silently ignored on the texture-SDF path of `SDF.create_from_mesh()` and on the primitive-mesh path in `ModelBuilder`; the requested voxel resolution is now honored end-to-end and matches the sparse-SDF path
- Fix material-combination inconsistency in the Newton-to-`mujoco-warp` contact converter so combined friction / solref / solimp values match native MuJoCo
- Fix `eq_objtype` mismatch for joint equality and mimic constraints in `SolverMuJoCo` so compiled models match native MuJoCo XML behavior
- Fix implicit-MPM rheology solver launch-dim handling under `warp-lang` 1.13's templated `launch_bounds` (formerly produced out-of-bounds reads)
- Fix `SolverKamino.reset` clobbering `q_j_p`, `q_j`, and `dq_j` for worlds outside `world_mask` when `joint_q`/`joint_u` targets were provided. The previous unmasked writes broke TWOPI revolute-joint angle unwrapping after partial-mask resets.

## [1.1.0] - 2026-04-13

### Added

- Add repeatable `--warp-config KEY=VALUE` CLI option for overriding `warp.config` attributes when running examples
- Add 3D texture-based SDF, replacing NanoVDB volumes in the mesh-mesh collision pipeline for improved performance and CPU compatibility.
- Parse URDF joint `limit effort="..."` values and propagate them to imported revolute and prismatic joint `effort_limit` settings
- Add `--benchmark [SECONDS]` flag to examples for headless FPS measurement with warmup
- Interactive example browser in the GL viewer with tree-view navigation and switch/reset support
- Add `TetMesh` class and USD loading API for tetrahedral mesh geometry
- Support kinematic bodies in VBD solver
- Add brick stacking example
- Add box pyramid example and ASV benchmark for dense convex-on-convex contacts
- Add plotting example showing how to access and visualize per-step simulation diagnostics
- Add `exposure` property to GL renderer
- Add `snap_to` argument to `ViewerGL.log_gizmo()` to snap gizmos to a target world transform when the user releases them
- Expose `gizmo_is_using` attribute to detect whether a gizmo is actively being dragged
- Add per-axis gizmo filtering via `translate`/`rotate` parameters on `log_gizmo`
- Add conceptual overview and MuJoCo Warp integration primer to collision documentation
- Add configurable velocity basis for implicit MPM (`velocity_basis`, default `"Q1"`) with GIMP quadrature option (`integration_scheme="gimp"`)
- Add plastic viscosity, dilatancy, hardening and softening rate as per-particle MPM material properties (`mpm:viscosity`, `mpm:dilatancy`, `mpm:hardening_rate`, `mpm:softening_rate`)
- Add MPM beam twist, snow ball, and viscous coiling examples
- Add support for textures in `SensorTiledCamera` via `Config.enable_textures`
- Add `enable_ambient_lighting` and `enable_particles` options to `SensorTiledCamera.Config`
- Add `SensorTiledCamera.utils.convert_ray_depth_to_forward_depth()` to convert ray-distance depth to forward (planar) depth
- Add `newton.geometry.compute_offset_mesh()` for extracting offset surface meshes from any collision shape, and a viewer toggle to visualize gap + margin wireframes in the GL viewer
- Add differentiable rigid contacts (experimental) with respect to body poses via `CollisionPipeline` when `requires_grad=True`
- Add per-shape display colors via `ModelBuilder.shape_color`, `Model.shape_color`, and `color=` on `ModelBuilder.add_shape_*`; mesh shapes fall back to `Mesh.color` when available and viewers honor runtime `Model.shape_color` updates
- Add `ModelBuilder.inertia_tolerance` to configure the eigenvalue positivity and triangle inequality threshold used during inertia correction in `finalize()`
- Add `ViewerBase.set_visible_worlds()` for runtime control of which worlds are rendered, replacing the static `max_worlds` parameter
- Add `compute_normals` and `compute_uvs` optional arguments to `Mesh.create_heightfield()` and `Mesh.create_terrain()`
- Pin `newton-assets` and `mujoco_menagerie` downloads to specific commit SHAs for reproducible builds (`NEWTON_ASSETS_REF`, `MENAGERIE_REF`)
- Add `ref` parameter to `download_asset()` to allow overriding the pinned revision
- Add `total_force_friction` and `force_matrix_friction` to `SensorContact` for tangential (friction) force decomposition
- Add Gaussian Splat geometry support via `ModelBuilder.add_shape_gaussian()` and USD import
- Add configurable Gaussian sorting modes to `SensorTiledCamera`
- Add automatic box, sphere, and capsule shape fitting for convex meshes during MJCF import
- Add color and texture reading to `usd.utils.get_mesh()`
- Export `ViewerBase` from `newton.viewer` public API
- Add `custom_attributes` argument to `ModelBuilder.add_shape_convex_hull()`
- Add RJ45 plug-socket insertion example with SDF contacts, latch joint, and interactive gizmo

### Changed

- Require `mujoco ~=3.6.0` and `mujoco-warp ~=3.6.0` (previously 3.5.x)
- Replace `plyfile` dependency with `open3d` for mesh I/O. Users who depended on `plyfile` transitively should install it separately.
- Switch Python build backend from `hatchling` to `uv_build`
- Switch mesh-SDF collision from triangle-based gradient descent to edge-based Brent's method to reduce contact jitter
- Unify heightfield and mesh collision pipeline paths; the separate `heightfield_midphase_kernel` and `shape_pairs_heightfield` buffer are removed in favor of the shared mesh midphase
- Replace per-shape `Model.shape_heightfield_data` / `Model.heightfield_elevation_data` with compact `Model.shape_heightfield_index` / `Model.heightfield_data` / `Model.heightfield_elevations`, matching the SDF indirection pattern. Use `Model.heightfield_data` indexed via `Model.shape_heightfield_index` instead.
- Standardize `rigid_contact_normal` to point from shape 0 toward shape 1 (A-to-B), matching the documented convention. Consumers that previously negated the normal on read (XPBD, VBD, MuJoCo, Kamino) no longer need to.
- Replace `Model.sdf_data` / `sdf_volume` / `sdf_coarse_volume` with texture-based equivalents (`texture_sdf_data`, `texture_sdf_coarse_textures`, `texture_sdf_subgrid_textures`). Use `Model.texture_sdf_data`, `texture_sdf_coarse_textures`, and `texture_sdf_subgrid_textures` instead.
- Render inertia boxes as wireframe lines instead of solid boxes in the GL viewer to avoid occluding objects
- Make contact reduction normal binning configurable (polyhedron, scan directions, voxel budget) via constants in ``contact_reduction.py``
- Upgrade GL viewer lighting from Blinn-Phong to Cook-Torrance PBR with GGX distribution, Schlick-GGX geometry, Fresnel-weighted ambient, and ACES filmic tone mapping
- Change implicit MPM residual computation to consider both infinity and l2 norm
- Change implicit MPM hardening law from exponential to hyperbolic sine (`sinh(-h * log(Jp))`), no longer scales elastic modulus
- Change implicit MPM collider velocity mode names: `"forward"` / `"backward"` replace `"instantaneous"` / `"finite_difference"`. Old names are no longer accepted.
- Simplify `SensorContact` force output: add `total_force` (aggregate per sensing object) and `force_matrix` (per-counterpart breakdown, `None` when no counterparts)
- Add `sensing_obj_idx` (`list[int]`), `counterpart_indices` (`list[list[int]]`), `sensing_obj_type`, and `counterpart_type` attributes. Rename `include_total` to `measure_total`
- Replace verbose Apache 2.0 boilerplate with two-line SPDX-only license headers across all source and documentation files
- Improve wrench preservation in hydroelastic contacts with contact reduction.
- Show Newton deprecation warnings during example runs started via `python -m newton.examples ...` or `python -m newton.examples.<category>.<module>`; pass `-W ignore::DeprecationWarning` if you need the previous quiet behavior.
- Reorder `ModelBuilder.add_shape_gaussian()` parameters so `xform` precedes `gaussian`, in line with other `add_shape_*` methods. Callers using positional arguments should switch to keyword form (`gaussian=..., xform=...`); passing a `Gaussian` as the second positional argument still works but emits a `DeprecationWarning`
- Rename `ModelBuilder.add_shape_ellipsoid()` parameters `a`, `b`, `c` to `rx`, `ry`, `rz`. Old names are still accepted as keyword arguments but emit a `DeprecationWarning`
- Rename `collide_plane_cylinder()` parameter `cylinder_center` to `cylinder_pos` for consistency with other collide functions. The old name is no longer accepted.
- Add optional `state` parameter to `SolverBase.update_contacts()` to align the base-class signature with Kamino and MuJoCo solvers
- Use `Literal` types for `SolverImplicitMPM.Config` string fields with fixed option sets (`solver`, `warmstart_mode`, `collider_velocity_mode`, `grid_type`, `transfer_scheme`, `integration_scheme`)
- Migrate `wp.array(dtype=X)` type annotations to `wp.array[X]` bracket syntax (Warp 1.12+).
- Align articulated `State.body_qd` / FK / IK / Jacobian / mass-matrix linear velocity with COM-referenced motion. If you were comparing `body_qd[:3]` against finite-differenced body-origin motion, recover origin velocity via `v_origin = v_com - omega x r_com_world`. Descendant `FREE` / `DISTANCE` `joint_qd` remains parent-frame and `joint_f` remains a world-frame COM wrench.

### Deprecated

- Deprecate `ModelBuilder.default_body_armature`, the `armature` argument on `ModelBuilder.add_link()` / `ModelBuilder.add_body()`, and USD-authored body armature via `newton:armature` in favor of adding any isotropic artificial inertia directly to `inertia`
- Deprecate `SensorContact.net_force` in favor of `SensorContact.total_force` and `SensorContact.force_matrix`
- Deprecate `SensorContact(include_total=...)` in favor of `SensorContact(measure_total=...)`
- Deprecate `SensorContact.sensing_objs` in favor of `SensorContact.sensing_obj_idx`
- Deprecate `SensorContact.counterparts` and `SensorContact.reading_indices` in favor of `SensorContact.counterpart_indices`
- Deprecate `SensorContact.shape` (use `total_force.shape` and `force_matrix.shape` instead) 
- Deprecate `SensorTiledCamera.render_context`; prefer `SensorTiledCamera.utils` and `SensorTiledCamera.render_config`.
- Deprecate `SensorTiledCamera.RenderContext`; use `SensorTiledCamera.RenderConfig` for config types and `SensorTiledCamera.render_config` / `SensorTiledCamera.utils` for runtime access.
- Deprecate `SensorTiledCamera.Config`; prefer `SensorTiledCamera.RenderConfig` and `SensorTiledCamera.utils`.
- Deprecate `max_worlds` parameter of `ViewerBase.set_model()` in favor of `ViewerBase.set_visible_worlds()`
- Deprecate `Viewer.update_shape_colors()` in favor of writing directly to `Model.shape_color`
- Deprecate `ModelBuilder.add_shape_ellipsoid()` parameters `a`, `b`, `c` in favor of `rx`, `ry`, `rz`
- Deprecate passing a `Gaussian` as the second positional argument to `ModelBuilder.add_shape_gaussian()`; use the `gaussian=` keyword argument instead
- Deprecate `SensorTiledCamera.utils.assign_random_colors_per_world()` and `assign_random_colors_per_shape()` in favor of per-shape colors via `ModelBuilder.add_shape_*(color=...)`

### Removed

- Remove `Heightfield.finalize()` and stop storing raw pointers for heightfields in `Model.shape_source_ptr`; heightfield collision data is accessed via `Model.shape_heightfield_index` / `Model.heightfield_data` / `Model.heightfield_elevations`
- Remove `robot_humanoid` example in favor of `basic_plotting` which uses the same humanoid model with diagnostics visualization

### Fixed

- Fix GL viewer crash when enabling "Gap + Margin" for soft-body-only states with no rigid body transforms
- Fix inertia validation spuriously inflating small but physically valid eigenvalues for lightweight components (< ~50 g) by using a relative threshold instead of an absolute 1e-6 cutoff
- Restore keyboard camera movement while hovering gizmos so keyboard controls remain active when the pointer is over gizmos
- Resolve USD asset references recursively in `resolve_usd_from_url` so nested stages are fully downloaded
- Unify CPU and GPU inertia validation to produce identical results for zero-mass bodies with `bound_mass`, singular inertia, non-symmetric tensors, and triangle-inequality boundary cases
- Fix `UnboundLocalError` crash in detailed inertia validation when eigenvalue decomposition encounters NaN/Inf input
- Handle NaN/Inf mass and inertia deterministically in both validation paths (zero out mass and inertia)
- Update `ModelBuilder` internal state after fast-path (GPU kernel) inertia validation so it matches the returned `Model`
- Fix MJCF mesh scale resolution to use the mesh asset's own class rather than the geom's default class, avoiding incorrect vertex scaling for models like Robotiq 2F-85 V4
- Fix articulated bodies drifting laterally on the ground in XPBD solver by solving rigid contacts before joints
- Fix `hide_collision_shapes=True` not hiding collision meshes that have bound PBR materials
- Filter inactive particles in viewer so only particles with `ParticleFlags.ACTIVE` are rendered
- Fix concurrent asset download races on Windows by using content-addressed cache directories
- Fix body `gravcomp` not being written to the MuJoCo spec, causing it to be absent from XML saved via `save_to_mjcf`
- Fix `compute_world_offsets` grid ordering to match terrain grid row-major order so replicated world indices align with terrain block indices
- Fix `eq_solimp` not being written to the MuJoCo spec for equality constraints, causing it to be absent from XML saved via `save_to_mjcf`
- Fix WELD equality constraint quaternion written in xyzw format instead of MuJoCo's wxyz format in the spec, causing incorrect orientation in XML saved via `save_to_mjcf`
- Fix `update_contacts` not populating `rigid_contact_point0`/`rigid_contact_point1` when using `use_mujoco_contacts=True`
- Fix MPR anti-flicker inflate biasing contact distances and witness points for convex-convex pairs, causing phantom overlap in stacking scenarios
- Fix VSync toggle having no effect in `ViewerGL` on Windows 8+ due to a pyglet bug where `DwmFlush()` is never called when `_always_dwm` is True
- Fix loop joint coordinate mapping in the MuJoCo solver so joints after a loop joint read/write at correct qpos/qvel offsets
- Fix viewer crash when contact buffer overflows by clamping contact count to buffer size
- Decompose loop joint constraints by DOF type (WELD for fixed, CONNECT-pair for revolute, single CONNECT for ball) instead of always emitting 2x CONNECT
- Fix inertia box wireframe rotation for isotropic and axisymmetric bodies in viewer
- Implicit MPM solver now uses `mass=0` for kinematic particles instead of `ACTIVE` flag
- Suppress macOS OpenGL warning about unloadable textures by binding a 1x1 white fallback texture when no albedo or environment texture is set
- Fix MuJoCo solver freeze when immovable bodies (kinematic, static, or fixed-root) generate contacts with degenerate invweight
- Fix forward-kinematics child-origin linear velocity for articulated translated joints
- Fix `ModelBuilder.approximate_meshes()` to handle the duplication of per-shape custom attributes that results from convex decomposition
- Fix `get_tetmesh()` winding order for left-handed USD meshes
- Fix contact force conversion in `SolverMuJoCo` to include friction (tangential) components
- Fix URDF inertial parameters parsing in parse_urdf so inertia tensor is correctly calculated as R@I@R.T
- Fix Poisson surface reconstruction segfault under parallel test execution by defaulting to single-threaded Open3D Poisson (`n_threads=1`)
- Fix overly conservative broadphase AABB for mesh shapes by using the pre-computed local AABB with a rotated-box transform instead of a bounding-sphere fallback, eliminating false contacts between distant meshes
- Fix heightfield bounding-sphere radius underestimating Z extent for asymmetric height ranges (e.g. `min_z=0, max_z=10`)
- Fix VBD self-contact barrier C2 discontinuity at `d = tau` caused by a factor-of-two error in the log-barrier coefficient
- Fix fast inertia validation treating near-symmetric tensors within `np.allclose()` default tolerances as corrections, aligning CPU and GPU validation warnings
- Fix URDF joint dynamics friction import so specified friction values are preserved during simulation
- Fix `requires_grad` not being preserved in `ArticulationView` attribute getters, breaking gradient propagation through selection queries
- Fix duplicate Reset button in brick stacking example when using the example browser
- Cap `cbor2` dependency to `<6` to prevent recorder test failures caused by breaking deserialization changes in cbor2 6.0
- Clamp viewer picking force to prevent explosion when picking light objects near stiff contacts, configurable via `pick_max_acceleration` parameter on the `Picking` class (default 5g of effective articulation mass)
- Fix `cloth_franka` example Jacobian broken by COM-referenced `body_qd` convention change; adjust robot base height, gripper orientations, and grasp targets for improved reachability (a follow-up PR will migrate the example to `newton.ik`)

## [1.0.0] - 2026-03-10

Initial public release.
