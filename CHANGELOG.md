# Changelog

## [Unreleased]

### Added

- Interactive example browser in the GL viewer with tree-view navigation and switch/reset support

### Changed

### Deprecated

### Removed

### Fixed

- Fix viewer crash with `imgui_bundle>=1.92.6` when editing colors by normalizing `color_edit3` input/output in `_edit_color3`
- Fix body `gravcomp` not being written to the MuJoCo spec, causing it to be absent from XML saved via `save_to_mjcf`

## [1.0.0] - YYYY-MM-DD

Initial public release.
