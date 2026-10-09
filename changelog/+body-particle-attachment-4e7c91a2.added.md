Add `ModelBuilder.add_attachment_body_particle()`, a compliant attachment between a
rigid body and a deformable particle:

  - `SolverVBD` applies the attachment whenever it integrates both endpoints, with
    equal-and-opposite forces on the particle and the rigid body.
  - `SolverCoupledADMM` couples the same attachment rows when the endpoints belong to
    different solver entries.
  - The rigid endpoint may be any body, including a cable capsule created by
    `ModelBuilder.add_rod()`.
