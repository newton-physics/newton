<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: Apache-2.0
-->

# Digital Instron v2

Identify one shoe-level effective viscoelastic midsole model from intact Digital
Instron bench tests, then exercise that calibrated model in live Newton
rigid-body physics.

## Calibration (`workflow.py`)

Fit the shared Hyperfoam-Maxwell-Pasternak column model to the rearfoot and
fullfoot bench cycles:

```bash
uv run -m projects.digital_instron_v2.workflow --manifest DigitalInstron/manifest_v2.json
```

The fitted parameters are cached at
`DigitalInstron/processed/v2_cache/digital_instron_material.json` and consumed by
the dynamic example below.

## Dynamic midsole example (`example.py`)

`dynamics.py` turns the calibrated column bed into a live Warp force model: each
substep every column reads its carrier-body pose, computes its through-thickness
compression, evaluates the Hyperfoam equilibrium pressure with a real-time
generalized-Maxwell overstress branch and Pasternak lateral coupling, and
accumulates the resulting wrench into `newton.State.body_f`. Four scenarios
share the same foundation:

```bash
# Displacement-controlled digital Instron: squish the midsole between a
# shoe-last crosshead and the ground plane; record the hysteresis loop.
uv run -m projects.digital_instron_v2.example --mode instron

# Free, massive midsole resting in stable equilibrium on the foundation;
# a lateral load is resisted by Coulomb foam-shear friction.
uv run -m projects.digital_instron_v2.example --mode settle

# Synthetic running stride that rolls a foot heel-to-toe over the foundation,
# producing a ground-reaction force profile and a migrating center of pressure.
uv run -m projects.digital_instron_v2.example --mode stride

# Fully dynamic, foot-mounted shoe with mass and inertia. A damped bilateral
# "upper" keeps the midsole coupled to the foot for the whole stride, so the
# shoe presses the foam into the ground in stance and the entire bed lifts clear
# with the foot in flight; the stance/flight ground reaction is recorded.
uv run -m projects.digital_instron_v2.example --mode attached
```

Add `--viewer null --num-frames N --test` to run headlessly and audit the
recorded response, or `--viewer gl` for the interactive viewer (the midsole
renders as a live bed of compression-coloured foam columns/springs that sink and
redden under load).

The `attached` mode is launch-overhead-bound (hundreds of tiny per-substep kernel
launches, not compute), so its whole 128-substep frame is captured into a single
CUDA graph and replayed once per frame — about a 7x speedup, making it faster than
real time. The foot trajectory is precomputed once into device arrays so the loop
stays fully on the GPU. Pass `--eager` to disable graph capture for debugging.

## Tests

```bash
uv run --extra dev -m unittest newton.tests.test_digital_instron_core
uv run --extra dev -m unittest newton.tests.test_digital_instron_project
uv run --extra dev -m unittest newton.tests.test_digital_instron_dynamics
```

`test_digital_instron_dynamics` verifies that the live per-substep force
integration reproduces the calibrated `core.predict` model to float precision,
that the Pasternak neighbour table matches the calibration Laplacian operator,
and that each example scenario passes its physical audit.
