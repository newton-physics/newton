# Digital Instron v2

This folder now hosts the experimental v2 restart for shoe Instron material
identification. The deleted `projects.digital_instron` CLI and old example
wrapper are intentionally not the contract for this workflow.

## v2 Boundary

- Trial setup lives in `manifest_v2.json`.
- Reusable code lives under `projects/digital_instron_v2`.
- The first runnable surface is a script/notebook workflow:

```bash
UV_CACHE_DIR=/tmp/uv-cache WARP_CACHE_PATH=/tmp/warp-cache uv run --extra dev -m projects.digital_instron_v2.workflow --manifest DigitalInstron/manifest_v2.json --step qc
```

After QC writes frame configs, run the first force-model smoke:

```bash
UV_CACHE_DIR=/tmp/uv-cache WARP_CACHE_PATH=/tmp/warp-cache uv run --extra dev -m projects.digital_instron_v2.workflow --manifest DigitalInstron/manifest_v2.json --step fit-smoke
```

Run the first autodiff material fit:

```bash
UV_CACHE_DIR=/tmp/uv-cache WARP_CACHE_PATH=/tmp/warp-cache uv run --extra dev -m projects.digital_instron_v2.workflow --manifest DigitalInstron/manifest_v2.json --step fit-autodiff --autodiff-iterations 25 --autodiff-sample-count 8
```

This writes `processed/v2_cache/digital_instron_v2_autodiff_fit.json`
with the fitted material and per-iteration loss/gradient history.

To inspect orientation, ray casting, and the current 1D spring response:

```bash
UV_CACHE_DIR=/tmp/uv-cache WARP_CACHE_PATH=/tmp/warp-cache uv run --extra dev -m projects.digital_instron_v2.workflow --manifest DigitalInstron/manifest_v2.json --step visualize
```

That writes:

- `processed/v2_cache/digital_instron_v2_mesh_orientation.png`
- `processed/v2_cache/digital_instron_v2_raycast_grid.png`
- `processed/v2_cache/digital_instron_v2_spring_response.png`
- `processed/v2_cache/digital_instron_v2_spring_snapshot.png`
- `processed/v2_cache/digital_instron_v2_spring_grid.csv`
- `processed/v2_cache/digital_instron_v2_spring_grid.npz`
- `processed/v2_cache/digital_instron_v2_visualization.summary.json`

## Current v2 Choices

- Train on all `include_in_fit` trials declared in the manifest.
- Repair the midsole mesh with Newton's Poisson remesh path before fitting and
  cache the repaired OBJ plus QC JSON under `processed/v2_cache`.
- Fail before fitting when CSV frame columns are missing, frame spans are
  implausible, or the conditioned midsole thickness is outside the manifest QC
  bounds.
- Start with a uniform 5 mm circular punch grid and keep the 3 mm grid as a
  later convergence check for localized punch trials.
- Place the rearfoot punch from the mesh footprint bounds with a 45 mm diameter:
  local rearfoot lateral midpoint and `rearfoot_length_fraction` from the
  configured `rearfoot_heel_side`.
- Use a full-footprint raycast grid for the midsole foundation. Each valid
  ray's top-to-bottom thickness is the slack length of that 1D spring.
- Fit the vertical differentiable Warp foundation first: locked Ogden-style
  compression plus compression-weighted viscous damping, with shared material
  parameters across trials.
- Keep MuJoCo out of training. The validation adapter applies learned foundation
  wrenches through Newton `state.body_f`, which `SolverMuJoCo` maps to MuJoCo
  `xfrc_applied`.

Full real-data calibration is deliberately manual/regression output for now:
plots and JSON summaries should be checked in only when they represent a
specific calibration run worth preserving.
