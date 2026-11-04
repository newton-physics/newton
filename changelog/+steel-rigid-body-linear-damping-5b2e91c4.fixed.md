Clamp the angular damping scale of `SolverSteel` free roots at 0, as PhysX does, so a `steel:rigid_body_angular_damping` above `1 / dt` stops the spin instead of reversing and growing it.
