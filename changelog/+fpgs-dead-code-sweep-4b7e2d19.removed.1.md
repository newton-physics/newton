Remove `SolverFeatherPGS`'s `friction_smoothing` parameter; the friction projection is an exact Coulomb-cone projection and never read it. Stop passing it.
