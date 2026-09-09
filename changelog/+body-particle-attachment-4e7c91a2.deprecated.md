Deprecate `SolverCoupledADMM.add_body_particle_attachment()` in favor of
`ModelBuilder.add_attachment_body_particle()`. The deprecated helper keeps writing its
legacy `coupling:body_particle_attachment` custom rows, which `SolverCoupledADMM` still
consumes but `SolverVBD` cannot see. Attachments authored on the builder are applied by
both solvers.
