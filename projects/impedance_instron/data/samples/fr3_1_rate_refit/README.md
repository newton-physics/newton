# FR3_1 recovered rate-refit movement

This sample restores the frozen movement bundle recovered from the removed Git
stash. It pairs the controller and rollout with the corresponding prepared
FR3_1 reference and leg profile.

- `reference.npz`: prepared movement and native force targets.
- `controller.npz`: 12-point, four-channel equilibrium trajectory coefficients
  and duration.
- `trace.npz`: saved 5,280-step rollout, sampled at approximately 62.5 us.
- `profile.json`: segment masses, centers of mass, sagittal inertias, gains,
  and controller bounds.
- `fit_summary.json`, `run.json`, `objective.json`: original search and
  rollout provenance, scores, and experimental objective details.

The fit ended at its iteration budget and is not marked converged. This is a
reproducible frozen example, not an accepted or physiologically validated
controller. The digital shoe contact-history artifact is not included.
