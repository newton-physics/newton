# Digital Instron Project

This package contains the material-identification workflow for fitting Newton
pressure-field shoe contact against physical Instron traces.

Run the project CLI directly:

```bash
uv run --extra examples -m projects.digital_instron.cli calibrate --viewer null
```

To rank material candidates by the foam-hysteresis goal loop, enable the goal
objective:

```bash
uv run --extra examples -m projects.digital_instron.cli calibrate --viewer null --calibration-objective goal
```

The legacy example command remains available as a wrapper:

```bash
uv run --extra examples -m newton.examples digital_instron --mode calibrate --viewer null
```

The calibration writes a reusable material JSON and a full shoe pressure-field
bundle. The bundle schema includes geometry, fixture, material, calibration,
ProtoMotions, and `state_model` sections. The default state model is
`memoryless_v1`, and `--pressure-memory` opts into the experimental
`max_compression_memory_v1` model for unloading diagnostics.
