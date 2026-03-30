# Changelog

## [Unreleased]

### Added

- Add 3D texture-based SDF, replacing NanoVDB volumes in the mesh-mesh collision pipeline for improved performance and CPU compatibility.
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
- Add differentiable rigid contacts (experimental) with respect to body poses via `CollisionPipeline` when `requires_grad=True`
- Add per-shape display colors via `ModelBuilder.shape_color`, `Model.shape_color`, and `color=` on `ModelBuilder.add_shape_*`; mesh shapes fall back to `Mesh.color` when available and viewers honor runtime `Model.shape_color` updates
- Add `ModelBuilder.inertia_tolerance` to configure the eigenvalue positivity and triangle inequality threshold used during inertia correction in `finalize()`

### Changed

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

### Deprecated

- Deprecate `SensorContact.net_force` in favor of `SensorContact.total_force` and `SensorContact.force_matrix`
- Deprecate `SensorContact(include_total=...)` in favor of `SensorContact(measure_total=...)`
- Deprecate `SensorContact.sensing_objs` in favor of `SensorContact.sensing_obj_idx`
- Deprecate `SensorContact.counterparts` and `SensorContact.reading_indices` in favor of `SensorContact.counterpart_indices`
- Deprecate `SensorContact.shape` (use `total_force.shape` and `force_matrix.shape` instead) 
- Deprecate `SensorTiledCamera.render_context`; prefer `SensorTiledCamera.utils` and `SensorTiledCamera.render_config`.
- Deprecate `SensorTiledCamera.RenderContext`; use `SensorTiledCamera.RenderConfig` for config types and `SensorTiledCamera.render_config` / `SensorTiledCamera.utils` for runtime access.
- Deprecate `SensorTiledCamera.Config`; prefer `SensorTiledCamera.RenderConfig` and `SensorTiledCamera.utils`.
- Deprecate `Viewer.update_shape_colors()` in favor of writing directly to `Model.shape_color`

### Removed

- Remove `robot_humanoid` example in favor of `basic_plotting` which uses the same humanoid model with diagnostics visualization

### Fixed

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
- Fix concurrent asset download races on Windows by using content-addressed cache directories
- Show prismatic joints in the GL viewer when "Show Joints" is enabled
- Fix body `gravcomp` not being written to the MuJoCo spec, causing it to be absent from XML saved via `save_to_mjcf`
- Fix `eq_solimp` not being written to the MuJoCo spec for equality constraints, causing it to be absent from XML saved via `save_to_mjcf`
- Fix WELD equality constraint quaternion written in xyzw format instead of MuJoCo's wxyz format in the spec, causing incorrect orientation in XML saved via `save_to_mjcf`
- Fix `update_contacts` not populating `rigid_contact_point0`/`rigid_contact_point1` when using `use_mujoco_contacts=True`
- Fix loop joint coordinate mapping in the MuJoCo solver so joints after a loop joint read/write at correct qpos/qvel offsets
- Fix viewer crash when contact buffer overflows by clamping contact count to buffer size
- Decompose loop joint constraints by DOF type (WELD for fixed, CONNECT-pair for revolute, single CONNECT for ball) instead of always emitting 2x CONNECT
- Fix inertia box wireframe rotation for isotropic and axisymmetric bodies in viewer
- Implicit MPM solver now uses `mass=0` for kinematic particles instead of `ACTIVE` flag
- Fix MuJoCo solver freeze when immovable bodies (kinematic, static, or fixed-root) generate contacts with degenerate invweight
- Fix forward-kinematics child-origin linear velocity for articulated translated joints
- Fix `ModelBuilder.approximate_meshes()` to handle the duplication of per-shape custom attributes that results from convex decomposition
- Fix `get_tetmesh()` winding order for left-handed USD meshes
- Fix contact force conversion in `SolverMuJoCo` to include friction (tangential) components
- Fix URDF inertial parameters parsing in parse_urdf so inertia tensor is correctly calculated as R@I@R.T

## [1.0.0] - 2026-03-10

Initial public release.
