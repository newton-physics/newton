# Changelog

## [Unreleased]

### Added

- Interactive example browser in the GL viewer with tree-view navigation and switch/reset support
- Add box pyramid example and ASV benchmark for dense convex-on-convex contacts

### Changed

### Deprecated

### Removed

### Fixed

- Fix viewer crash with `imgui_bundle>=1.92.6` when editing colors by normalizing `color_edit3` input/output in `_edit_color3`
- Show prismatic joints in the GL viewer when "Show Joints" is enabled
- Fix body `gravcomp` not being written to the MuJoCo spec, causing it to be absent from XML saved via `save_to_mjcf`
- Fix WELD equality constraint quaternion written in xyzw format instead of MuJoCo's wxyz format in the spec, causing incorrect orientation in XML saved via `save_to_mjcf`

## [1.0.0] - YYYY-MM-DD

Initial public release.
