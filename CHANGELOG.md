# Changelog

## [Unreleased]

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
- Add `compute_normals` and `compute_uvs` optional arguments to `Mesh.create_heightfield()` and `Mesh.create_terrain()`
- Add RJ45 plug-socket insertion example with SDF contacts, latch joint, and interactive gizmo
- Add `TRIANGLE_PRISM` support-function type for heightfield triangles, extruding 1 m along the heightfield's local -Z so GJK/MPR naturally resolves shapes on the back side
- Add `ViewerGL.log_scalar()` for live scalar time-series plots in the viewer

### Changed

- Switch mesh-SDF collision from triangle-based gradient descent to edge-based Brent's method to reduce contact jitter
- Unify heightfield and mesh collision pipeline paths; the separate `heightfield_midphase_kernel` and `shape_pairs_heightfield` buffer are removed in favor of the shared mesh midphase
- Replace per-shape `Model.shape_heightfield_data` / `Model.heightfield_elevation_data` with compact `Model.shape_heightfield_index` / `Model.heightfield_data` / `Model.heightfield_elevations`, matching the SDF indirection pattern
- Standardize `rigid_contact_normal` to point from shape 0 toward shape 1 (A-to-B), matching the documented convention. Consumers that previously negated the normal on read (XPBD, VBD, MuJoCo, Kamino) no longer need to.
- Replace `Model.sdf_data` / `sdf_volume` / `sdf_coarse_volume` with texture-based equivalents (`texture_sdf_data`, `texture_sdf_coarse_textures`, `texture_sdf_subgrid_textures`)
- Render inertia boxes as wireframe lines instead of solid boxes in the GL viewer to avoid occluding objects
- Make contact reduction normal binning configurable (polyhedron, scan directions, voxel budget) via constants in ``contact_reduction.py``
- Upgrade GL viewer lighting from Blinn-Phong to Cook-Torrance PBR with GGX distribution, Schlick-GGX geometry, Fresnel-weighted ambient, and ACES filmic tone mapping
- Change implicit MPM residual computation to consider both infinity and l2 norm
- Change implicit MPM hardening law from exponential to hyperbolic sine (`sinh(-h * log(Jp))`), no longer scales elastic modulus
- Change implicit MPM collider velocity mode names: `"forward"` / `"backward"` replace `"instantaneous"` / `"finite_difference"`
- Simplify `SensorContact` force output: add `total_force` (aggregate per sensing object) and `force_matrix` (per-counterpart breakdown, `None` when no counterparts)
- Add `sensing_obj_idx` (`list[int]`), `counterpart_indices` (`list[list[int]]`), `sensing_obj_type`, and `counterpart_type` attributes. Rename `include_total` to `measure_total`
- Replace verbose Apache 2.0 boilerplate with two-line SPDX-only license headers across all source and documentation files
- Add `custom_attributes` argument to `ModelBuilder.add_shape_convex_hull()`
- Improve wrench preservation in hydroelastic contacts with contact reduction.
- Show Newton deprecation warnings during example runs started via `python -m newton.examples ...` or `python -m newton.examples.<category>.<module>`; pass `-W ignore::DeprecationWarning` if you need the previous quiet behavior.
- Reorder `ModelBuilder.add_shape_gaussian()` parameters so `xform` precedes `gaussian`, in line with other `add_shape_*` methods. Callers using positional arguments should switch to keyword form (`gaussian=..., xform=...`); passing a `Gaussian` as the second positional argument still works but emits a `DeprecationWarning`
- Rename `ModelBuilder.add_shape_ellipsoid()` parameters `a`, `b`, `c` to `rx`, `ry`, `rz`. Old names are still accepted as keyword arguments but emit a `DeprecationWarning`
- Rename `collide_plane_cylinder()` parameter `cylinder_center` to `cylinder_pos` for consistency with other collide functions
- Add optional `state` parameter to `SolverBase.update_contacts()` to align the base-class signature with Kamino and MuJoCo solvers
- Use `Literal` types for `SolverImplicitMPM.Config` string fields with fixed option sets (`solver`, `warmstart_mode`, `collider_velocity_mode`, `grid_type`, `transfer_scheme`, `integration_scheme`)
- Migrate `wp.array(dtype=X)` type annotations to `wp.array[X]` bracket syntax (Warp 1.12+).
- Align articulated `State.body_qd` / FK / IK / Jacobian / mass-matrix linear velocity with COM-referenced motion. If you were comparing `body_qd[:3]` against finite-differenced body-origin motion, recover origin velocity via `v_origin = v_com - omega x r_com_world`. Descendant `FREE` / `DISTANCE` `joint_qd` remains parent-frame and `joint_f` remains a world-frame COM wrench.
- Pin `mujoco` and `mujoco-warp` dependencies to `~=3.6.0`

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
- Fix viewer crash with `imgui_bundle>=1.92.6` when editing colors by normalizing `color_edit3` input/output in `_edit_color3`
- Fix `hide_collision_shapes=True` not hiding collision meshes that have bound PBR materials
- Filter inactive particles in viewer so only particles with `ParticleFlags.ACTIVE` are rendered
- Fix concurrent asset download races on Windows by using content-addressed cache directories
- Show prismatic joints in the GL viewer when "Show Joints" is enabled
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
- Fix connect constraint anchor computation to account for joint reference positions when `SolverMuJoCo` is the chosen solver.
- Fix forward-kinematics child-origin linear velocity for articulated translated joints
- Fix URDF joint dynamics friction import so specified friction values are preserved during simulation
- Fix duplicate Reset button in brick stacking example when using the example browser
- Fix mesh-convex back-face contacts generating inverted normals that trap shapes inside meshes and cause solver divergence (NaN)
- Fix MPR convergence failure on large and extreme-aspect-ratio mesh triangles by projecting the starting point onto the triangle nearest the convex center
- Cap `cbor2` dependency to `<6` to prevent recorder test failures caused by breaking deserialization changes in cbor2 6.0
- Fix O(W²·S²) memory explosion in `CollisionPipeline` shape-pair buffer allocation for NXN and SAP broad phase modes by computing per-world pair counts instead of a global N²
- Clamp viewer picking force to prevent explosion when picking light objects near stiff contacts, configurable via `pick_max_acceleration` parameter on the `Picking` class (default 5g of effective articulation mass)
- Fix `cloth_franka` example Jacobian broken by COM-referenced `body_qd` convention change; adjust robot base height, gripper orientations, and grasp targets for improved reachability (a follow-up PR will migrate the example to `newton.ik`)
- Fix `SensorRaycast` ignoring `PLANE` geometry

## [1.0.0] - 2026-03-10

Initial public release.
