Add the experimental LOX rigid-body dynamics backend to `SolverKamino`, selected with
`dynamics_solver="lox"` and configured through `SolverKamino.Config.lox`. Unlike the
PADMM and DVI backends, LOX advances kinematic bodies, and massless bodies whose only
joints are free joints to the world, with their velocity in `state_in.body_qd`, moves the
bodies welded to kinematic bodies rigidly with them, and keeps speculative contacts
regardless of `cull_speculative_contacts`.
