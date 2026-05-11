# Digital Instron

This folder contains the first calibration harness for matching physical Instron shoe tests with Newton pressure-field contact.

## Current Decisions

- Fullfoot indenter: `Instron Shoe Last Size 9 6drop merged attachment 1.STL`.
- Rearfoot indenter: rigid flat-ended circular punch with radius `45 mm`.
- Midsole geometry: `puma-fast-r-nitro-elite-3-3d-internal-wt-LR.obj`.
- Units: source meshes are treated as millimeters and scaled by `0.001`.
- Coordinate inference: longest mesh axis is shoe length, shortest axis is vertical thickness.
- Fullfoot indenter pose: the STL is canonicalized, then rotated by `--fullfoot-rotation-deg` before SDF generation. The current default `90 0 0` turns the long missile axis vertical for this STL.
- Fullfoot start pose: `--fullfoot-start-clearance` raises the indenter above the midsole at zero displacement. The auto-contact search rejects offsets that exceed `--fullfoot-initial-force-max` at the first frame.
- Pressure profile: `--pressure-profile poisson` uses the default interior pressure field. `--pressure-profile layer` uses a one-sided local-Z foam-column field with `--layer-eta-lock`, `--layer-densification-power`, `--layer-cubic`, and `--layer-quintic`.
- Rearfoot location: heel-side 20 percent region, implemented as the heel-side 30 percent x location with lateral center. Use `--heel-side min|max` if the first run hits the toe instead.
- Physical target trace: averaged cycles `90-100`, compressive displacement and force positive.
- Fit score: equal weighting of force-displacement loop RMSE and peak-force error.

## Commands

Preprocess the physical traces:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra examples -m newton.examples digital_instron --mode preprocess --viewer null
```

Run the rearfoot pressure-field sweep on a CUDA-capable machine:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra examples -m newton.examples digital_instron --mode run --test-case rearfoot --viewer null --kh 6.085e6
```

Run the rearfoot sweep with the current phenomenological loop-shape, stiffness-scale, and hysteresis fits:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra examples -m newton.examples digital_instron --mode run --test-case rearfoot --viewer null --kh 6.085e6 --fit-elastic-force-power --fit-elastic-force-scale --fit-hysteresis-damping
```

Run a shared rearfoot plus fullfoot material calibration:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra examples -m projects.digital_instron.cli calibrate --viewer null --sdf-resolution 64 --kh 6.085e6 --fit-elastic-force-power --fit-elastic-force-scale --fit-hysteresis-damping --fullfoot-contact-search-max 0.20 --fullfoot-contact-search-steps 101 --fullfoot-rotation-deg 90 0 0
```

To fit damping against per-sample force error instead of total hysteresis energy:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra examples -m projects.digital_instron.cli calibrate --viewer null --sdf-resolution 64 --kh 6.085e6 --fit-elastic-force-power --fit-elastic-force-scale --fit-hysteresis-damping --fit-damping-mode per-step --fullfoot-contact-search-max 0.20 --fullfoot-contact-search-steps 101 --fullfoot-rotation-deg 90 0 0
```

Run the same calibration with the foam-layer pressure field:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra examples -m projects.digital_instron.cli calibrate --viewer null --sdf-resolution 64 --kh 6.085e6 --pressure-profile layer --layer-eta-lock 0.65 --layer-densification-power 0.0 --fit-elastic-force-power --fit-elastic-force-scale --fit-scale-min 0.05 --fit-scale-max 5.0 --fit-scale-steps 201 --fit-hysteresis-damping --fit-damping-mode per-step --fullfoot-contact-search-max 0.20 --fullfoot-contact-search-steps 101 --fullfoot-rotation-deg 90 0 0
```

This writes:

- `processed/pressure_field_material.json`: reusable shared material parameters.
- `processed/shoe_pressure_field_bundle.json`: ProtoMotions-oriented shoe asset bundle with geometry, fixture, material, calibration, and state-model sections.
- `processed/digital_instron_hysteresis.png`: physical-vs-digital force-displacement loops.
- `processed/digital_instron_calibration.summary.json`: combined calibration summary.
- `processed/rearfoot_sim_pressure_field.csv` and `processed/fullfoot_sim_pressure_field.csv`: per-trial simulation traces.

View the prescribed Instron compression and contact geometry:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra examples -m newton.examples digital_instron --mode view --test-case fullfoot --viewer gl --sdf-resolution 64 --kh 6.085e6 --fullfoot-contact-search-max 0.20 --fullfoot-contact-search-steps 101 --fullfoot-rotation-deg 90 0 0
```

For a browser viewer:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra examples -m newton.examples digital_instron --mode view --test-case fullfoot --viewer viser --sdf-resolution 64 --kh 6.085e6 --fullfoot-contact-search-max 0.20 --fullfoot-contact-search-steps 101 --fullfoot-rotation-deg 90 0 0
```

Run the fullfoot pressure-field sweep:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra examples -m newton.examples digital_instron --mode run --test-case fullfoot --viewer null
```

## Notes

The current pressure-field model is memoryless. It can fit an elastic force-displacement curve with `kh`, `elastic_force_power`, and `elastic_force_scale`, but true foam hysteresis will require a material-state or dissipative pressure-field extension after this baseline exposes the error.
