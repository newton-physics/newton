# Digital Instron

This project identifies one nonlinear, rate-dependent through-thickness material from
the rearfoot punch and full-foot shoe-last Instron trials.

The full-foot analysis uses the mirrored left-foot asset
`Instron Shoe Last Size 9 6drop merged attachment 1 left.STL`; the original
right-foot STL is retained unchanged as its source.

## Model

The repaired midsole mesh is sampled on a 5 mm column grid. Each column uses its
raycast thickness as its rest length. The punch or last shortens the columns and
the model integrates pressure over their finite areas.

The reduced literature-backed foundation has four fitted parameters:

- instantaneous first-order Hyperfoam shear modulus [Pa]
- Hyperfoam exponent
- long-term/instantaneous shear-modulus fraction
- Pasternak lateral coupling [N/m]

The effective Poisson ratio is fixed at `0.30` and the Maxwell relaxation time
at `0.08 s`; the current two single-period fixture tests do not identify these
independently. Maxwell state is initialized at the exact periodic fixed point of
the measured cycle because the traces average cycles 90-100.

`projects/digital_instron_v2/core.py` is the only constitutive implementation.
Fitting calls the same `predict()` function used for evaluation. Released columns
carry no tension.

## Command

Fit all four parameters to both trials:

```powershell
uv run --extra dev -m projects.digital_instron_v2.workflow `
  --manifest DigitalInstron/manifest_v2.json --evaluations 100
```

The fit writes the authoritative material artifact to:

`DigitalInstron/processed/v2_cache/digital_instron_material.json`

The optimizer minimizes peak-normalized pointwise force residuals. Peak force,
loading and unloading RMSE, and hysteresis work are reported separately as
validation metrics; they do not receive extra fit weight.

With `--plots`, accepted optimizer iterations are written to the material JSON
and plotted in `processed/v2_cache/digital_instron_convergence.png`. The figure
shows total and per-fixture pointwise MSE plus every material parameter.

`processed/v2_cache/digital_instron_contact_current.png` separates geometry
masks: gray points are created midsole columns, black outlines are columns under
the fixture, and colored points are compressed at peak travel. The current 5 mm
grid contains 910 midsole columns; the full-foot fixture overlaps 595 and
compresses 578 at peak. The remaining 17 outlined cells are perimeter or arch
clearance, not missing columns.

## Source layout

- `manifest_v2.json`: averaged traces, geometry, and initial values
- `projects/digital_instron_v2/core.py`: material prediction, fitting, and metrics
- `projects/digital_instron_v2/geometry.py`: material-column geometry
- `projects/digital_instron_v2/workflow.py`: fitting command
- `MODEL_FINDINGS.md`: literature review, rejected models, assumptions, and the
  additional experiments required for a full finite-strain network model
