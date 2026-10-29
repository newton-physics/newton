Add optional dimensionless `rho`, `gamma`, and `baumgarte` overrides alongside each ADMM constraint definition: on `SolverCoupledADMM.ContactPair` for contacts, through `coupling:attachment_particle_body_rho`, `coupling:attachment_particle_body_gamma`, and `coupling:attachment_particle_body_baumgarte` custom attributes on particle-body attachments, and through `coupling:joint_rho`, `coupling:joint_gamma`, and `coupling:joint_baumgarte` custom attributes on individual joints.

Existing configurations require no changes. Omitted overrides inherit the
corresponding global `Config` values. To tune a contact pair, use
`ContactPair("rigid", "cloth", rho=0.4, gamma=0.2)`. Register joint and
attachment attributes with `SolverCoupledADMM.register_custom_attributes(builder)`
and pass them in the `custom_attributes` argument of the joint or of
`ModelBuilder.add_attachment_particle_body()`. Rows authored under the legacy
`coupling:body_particle_attachment` custom frequency read their overrides from
attributes with the `coupling:body_particle_attachment_` prefix. Custom
attributes use `-1` for inheritance; `ContactPair` uses `None`.
Explicit zero `gamma` or `baumgarte` disables that term for the constraint.
Contact parameters affect only contacts, and the iteration count remains
global. When migrating values from the previous timestep-dependent
formulation, apply the dimensionless-parameter migration rules first.
