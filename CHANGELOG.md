# Changelog

## [Unreleased]

### Added

- Interactive example browser in the GL viewer with tree-view navigation and switch/reset support
- Add `TetMesh` class and USD loading API for tetrahedral mesh geometry
- Support kinematic bodies in VBD solver
- Add brick stacking example
- Add box pyramid example and ASV benchmark for dense convex-on-convex contacts

### Changed

### Deprecated

### Removed

### Fixed

- Fix MJCF mesh scale resolution to use the mesh asset's own class rather than the geom's default class, avoiding incorrect vertex scaling for models like Robotiq 2F-85 V4
- Fix viewer crash with `imgui_bundle>=1.92.6` when editing colors by normalizing `color_edit3` input/output in `_edit_color3`
- Show prismatic joints in the GL viewer when "Show Joints" is enabled
- Fix body `gravcomp` not being written to the MuJoCo spec, causing it to be absent from XML saved via `save_to_mjcf`
- Fix `eq_solimp` not being written to the MuJoCo spec for equality constraints, causing it to be absent from XML saved via `save_to_mjcf`
- Fix WELD equality constraint quaternion written in xyzw format instead of MuJoCo's wxyz format in the spec, causing incorrect orientation in XML saved via `save_to_mjcf`

## [1.0.0] - 2026-03-10

Initial public release.
