Reject `SolverStyle3D.Config.iterations` values below 1 at the start of each
step, before modifying either state. This prevents zero or negative iteration
counts from leaving output positions stale and computing velocities from them.
