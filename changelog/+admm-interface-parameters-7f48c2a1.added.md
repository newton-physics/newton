Add optional dimensionless `rho`, `gamma`, and `baumgarte` overrides alongside each ADMM constraint definition: on `SolverCoupledADMM.ContactPair` for contacts, in `add_body_particle_attachment()` for body-particle attachments, and through `coupling:joint_rho`, `coupling:joint_gamma`, and `coupling:joint_baumgarte` custom attributes on individual joints.

Existing configurations require no changes. Omitted overrides inherit the
corresponding global `Config` values. To tune a contact pair, use
`ContactPair("rigid", "cloth", rho=0.4, gamma=0.2)`; to tune an attachment,
pass these keywords to `add_body_particle_attachment()`. Register joint
attributes with `SolverCoupledADMM.register_custom_attributes(builder)` and
pass them in the joint's existing `custom_attributes` argument. Attachment
attributes use the `coupling:body_particle_attachment_` prefix. Custom
attributes use `-1` for inheritance; Python override arguments use `None`.
Explicit zero `gamma` or `baumgarte` disables that term for the constraint.
Contact parameters affect only contacts, and the iteration count remains
global. When migrating values from the previous timestep-dependent
formulation, apply the dimensionless-parameter migration rules first.
