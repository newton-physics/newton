# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Headless validation run for the dish-rack scene's per-material contacts.

Runs N_STEPS of the scene and reports adaptive-dt and step-doubling-error
statistics, so changes to per-material ke/kd/mu can be sanity-checked without
spinning up the GL viewer.
"""

from __future__ import annotations

import numpy as np

from scripts.scenes.dish_rack import DT_OUTER, build_model_randomized, make_solver

N_STEPS = 200
SEED = 42
NUM_WORLDS = 1


def main() -> None:
    model = build_model_randomized(NUM_WORLDS, seed=SEED)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = make_solver(model)

    dts, errs, accepts = [], [], []
    for _ in range(N_STEPS):
        state_0, state_1 = solver.step_dt(DT_OUTER, state_0, state_1, control)
        dts.append(float(solver.dt.numpy()[0]))
        errs.append(float(solver.last_error.numpy()[0]))
        accepts.append(bool(solver.accepted.numpy()[0]))

    dts = np.asarray(dts)
    errs = np.asarray(errs)
    accepts = np.asarray(accepts)
    rejects = int((~accepts).sum())

    print(
        f"dt_mean={dts.mean():.2e}  dt_min={dts.min():.2e}  dt_max={dts.max():.2e}  | "
        f"err_mean={errs.mean():.2e}  err_max={errs.max():.2e}  | "
        f"rejects={rejects}/{N_STEPS}"
    )


if __name__ == "__main__":
    main()
