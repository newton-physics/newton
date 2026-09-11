Change the `SolverCoupledADMM` defaults to `rho=0.5` and `gamma=0.1` to improve convergence for stiff attachments with modest proximal inertia.

Specify `rho` and `gamma` explicitly to override the new tuning. To migrate
the previous defaults (`rho=1.0`, `gamma=0.0`) while preserving their behavior
at a fixed coupled substep duration `h`, use `rho=h` and `gamma=0.0`, together
with the stiffness conversion described in the dimensionless-parameter
migration note. Copying the old numerical value `rho=1.0` does not preserve
the previous penalty strength after the timestep normalization.
