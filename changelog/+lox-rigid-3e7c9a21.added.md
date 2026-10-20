Add the experimental LOX rigid-body dynamics backend to `SolverKamino`, selected with
`dynamics_solver="lox"` and configured through `SolverKamino.Config.lox`. Unlike the
PADMM and DVI backends, LOX advances kinematic, proxy, and massless bodies with their
velocity in `state_in.body_qd`, and it keeps speculative contacts regardless of
`cull_speculative_contacts`.
