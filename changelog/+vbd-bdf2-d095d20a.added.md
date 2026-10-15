Add an experimental BDF2 rigid time integrator to ``SolverVBD`` (``rigid_time_integration="bdf2"``) with a per-world
BDF1 fallback around contact, joint-limit and force events (``rigid_bdf2_theta_policy``).
