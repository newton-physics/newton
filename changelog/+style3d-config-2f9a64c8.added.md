Add public `SolverStyle3D.Config` to group solver options and update them at
runtime; changes to `solver.config` take effect on the next step. Preserve the
original keyword arguments and defaults, and pass either
`config=SolverStyle3D.Config(...)` or the original options.
