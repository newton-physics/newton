# FR3_2 peak-to-peak window experiment

This bundle contains the FR3_2 measured reference re-windowed to source time
0.030–0.380 s, its historical four-channel rollout, and the fitted six-channel
controller with its profile and metrics. It allows inspection of the new
measured movement window alongside both controller results.

- `reference.npz`: 71 prepared motion samples and 353 native force samples.
- `controller.npz`: 12-point, four-channel equilibrium trajectory coefficients
  and duration for the historical fit.
- `trace.npz`: historical 5,600-step four-channel rollout, sampled at 62.5 us.
- `profile.json`: leg masses, centers of mass, sagittal inertias, gains, and
  equilibrium bounds.
- `summary.json`: historical four-channel fit settings, metrics, objective
  components, and diagnostics.
- `controller_ankle_xy.npz`: promoted 12-point, six-channel hip XY, knee/ankle
  angle, and ankle XY controller.
- `controller_ankle_xy_seed.npz`: six-channel starting controller before the
  GPU refit.
- `profile_ankle_xy.json`: six-channel profile, including independent ankle
  Cartesian gains.
- `fit_summary_ankle_xy.json`: measured-fit metrics, refinement, and rollout
  qualification for the promoted controller.

The earlier four-channel search ran 200 iterations from the saved FR3_2
controller. It reduced the objective from 34.82 to 4.64, but did not meet
measured acceptance limits. Its saved trace and summary are retained for
comparison; they do not describe the promoted six-channel controller.

The digital shoe contact-history artifact is not included.

## Promoted ankle-position refit

The six-channel GPU refit started from `controller_ankle_xy_seed.npz` and
completed 200 iterations on 192 CUDA worlds. Its loss fell from 24.03 to 0.520.
The fitted controller meets the numerical measured-fit criteria; half-timestep
refinement also passed. Hip flight gating was disabled, and the ankle position
controller remained active throughout flight. Ankle stiffness is 3,000 N/m
and damping is 40 N s/m per axis, independent of hip gains.

The fit's refined rollout still hit the existing passive shoe compression cap
on 3,828 steps. Numerical acceptance does not validate the inherited shoe
interface or passive cap behavior. The full shoe contact-history artifact and
fitted rollout are not included in this sample; `fit_summary_ankle_xy.json`
records the refit metrics and limits.
