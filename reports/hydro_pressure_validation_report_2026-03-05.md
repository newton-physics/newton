# Hydro Pressure Validation Report

Date: 2026-03-05
Device: `cuda:0` (NVIDIA GeForce RTX 4070)
Repository: `D:\newton`

## Scope

Validated pressure-field contact behavior for:

- Pressure increases with penetration depth
- Pressure/stiffness increases with `k_hydro` at fixed penetration
- Nonlinear pressure field modulation affects contact response while sliding across a surface
- Plane/heightfield pressure workflow fallback routes to non-hydro path

## Test Run

Command:

```bash
uv run python -m unittest -v \
  newton.tests.test_hydroelastic.TestHydroelastic.test_pressure_workflow_depth_proxy_increases_with_penetration_for_all_shapes_cuda_0 \
  newton.tests.test_hydroelastic.TestHydroelastic.test_pressure_workflow_contact_stiffness_increases_with_k_hydro_for_all_shapes_cuda_0 \
  newton.tests.test_hydroelastic.TestHydroelastic.test_pressure_workflow_nonlinear_axis_sine_modulates_contact_depth_for_box_and_mesh_cuda_0 \
  newton.tests.test_hydroelastic.TestHydroelastic.test_pressure_field_falls_back_to_nonhydro_for_plane_and_heightfield_cuda_0
```

Result: `OK` (`4` tests, `0` failures, `0` errors).

## Quantitative Results

### 1) Penetration Depth Monotonicity (pressure workflow)

Metric: sum of penetrating contact-surface depth magnitudes for each shape pair.
Penetration samples: `[0.015, 0.025, 0.035]` m.

| Shape | Depth Sum Series | Strictly Increasing |
|---|---:|---:|
| sphere | 0.4357, 1.7536, 2.7031 | Yes |
| ellipsoid | 0.5227, 1.4395, 2.7117 | Yes |
| box | 8.6191, 13.8415, 18.4414 | Yes |
| capsule | 0.2394, 0.6563, 1.2150 | Yes |
| cylinder | 1.0632, 7.9224, 12.4276 | Yes |
| cone | 0.0011, 0.0379, 0.0807 | Yes |
| mesh | 3.7451, 5.9225, 9.3158 | Yes |

Summary: `7/7` shapes monotonic.

### 2) `k_hydro` Monotonicity (pressure workflow)

Metric: summed per-contact stiffness for shape pair at fixed penetration (`0.025` m).
`k_hydro` samples: `[8e7, 2e8, 6e8]`.

| Shape | Stiffness Sum Series | Strictly Increasing |
|---|---:|---:|
| sphere | 1.1627e8, 2.9068e8, 8.7204e8 | Yes |
| ellipsoid | 7.9503e7, 1.9876e8, 5.9628e8 | Yes |
| box | 1.1075e8, 2.7687e8, 8.3061e8 | Yes |
| capsule | 5.3755e7, 1.3439e8, 4.0316e8 | Yes |
| cylinder | 2.3130e8, 5.7826e8, 1.7348e9 | Yes |
| cone | 1.6010e7, 4.0024e7, 1.2007e8 | Yes |
| mesh | 1.1412e8, 2.8530e8, 8.5590e8 | Yes |

Summary: `7/7` shapes monotonic.

### 3) Nonlinear Modulation Under Tangential Motion

Shapes: box, mesh
Conditions: fixed penetration (`0.025` m), x-axis sweep across contact patch, sine modulation amplitude `(0.6, 0, 0)`.
Validation thresholds: `relative_spread > 0.05` and harmonic-fit `R^2 > 0.5`.

| Shape | Relative Spread | Harmonic Fit R² | Pass |
|---|---:|---:|---:|
| box | 0.1584 | 0.6351 | Yes |
| mesh | 0.1392 | 0.7830 | Yes |

Summary: `2/2` nonlinear modulation checks pass.

### 4) Fallback Path Validation

- Plane + pressure workflow: routed to non-hydro path with rigid contacts present.
- Heightfield + pressure workflow: routed to non-hydro path with rigid contacts present.

Result: pass.

## Conclusion

All targeted validation checks match the expected behavior:

- Pressure response increases with penetration depth
- Pressure/stiffness response increases with `k_hydro`
- Nonlinear pressure fields modulate contact response while sliding
- Unsupported terrain geometries fall back correctly to non-hydro contact workflow
