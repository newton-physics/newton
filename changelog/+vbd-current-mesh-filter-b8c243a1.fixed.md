Evaluate full-surface mesh contact eligibility at current particle and body positions in VBD force and Hessian evaluation, penalty updates, force reporting, and coupled proxy feedback. Retain detected candidates so contacts can become eligible again without another collision detection.

Preserve body reactions and coupling feedback when the per-body soft-contact list overflows by scanning the complete stored contact buffer for that body.

Coupled particle solvers now reject unsupported full-surface input consistently with standalone solvers, instead of silently reducing it to particle contacts. Coupling copies contact ownership and metadata; consuming solvers determine force eligibility.

Preserve source soft-contact overflow status through coupling and clear stale mesh tags when a buffer is reused for particle-only detection.
