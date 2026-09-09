Make `SolverCoupledADMM` apply timestep conversions for physical attachment stiffness, damping, and dry-friction limits while keeping ADMM penalty and proximal parameters dimensionless.

Migrate configurations tuned with the previous formulation at a fixed coupled
substep duration `h` (in seconds) using `rho_new = h * rho_old`,
`gamma_new = gamma_old / h`, and `stiffness_new = stiffness_old / h`.
Apply the stiffness conversion to translational and angular model-joint
stiffnesses and body-particle attachment stiffnesses. Keep damping,
dry-friction force/torque limits, `baumgarte`, and iteration counts unchanged.
For example, at `h = 1/120`, old values `rho=60`, `gamma=0.001`, and
`stiffness=1000` become `rho=0.5`, `gamma=0.12`, and `stiffness=120000`.
These conversions preserve the previous equations at the reference timestep;
use the resulting dimensionless parameters and physical stiffness/damping
values when changing timesteps instead of repeating the conversion each step.
Solver steps now require a strictly positive timestep.
