# FR3_2 peak-to-peak window experiment

This bundle contains the FR3_2 measured reference re-windowed to source time
0.030–0.380 s, the resulting full-refit equilibrium trajectory, and its saved
rollout. It allows inspection of the new measured movement window alongside
the controller response.

- `reference.npz`: 71 prepared motion samples and 353 native force samples.
- `controller.npz`: 12-point, four-channel equilibrium trajectory coefficients
  and duration.
- `trace.npz`: 5,600-step rollout, sampled at 62.5 us.
- `profile.json`: leg masses, centers of mass, sagittal inertias, gains, and
  equilibrium bounds.
- `summary.json`: fit settings, metrics, objective components, and diagnostics.

The search ran 200 iterations from the saved FR3_2 controller. It reduced the
objective from 34.82 to 4.64, but the final hip RMSE was 26.2 / 30.6 mm and
force RMSE was 159.6 / 142.0 N; the fit did not meet measured acceptance
limits. Its rollout reached the passive compression cap. Treat this as an
exploratory sample, not a recommended controller or evidence that the altered
window improves the fit. Hip velocity appears in the saved rollout diagnostics
only; it is not a fitted loss block.

The digital shoe contact-history artifact is not included.
