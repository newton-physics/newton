Fix implicit actuation returning the explicit effort when the warm-start residual is already below `residual_tol`, and not converging when full Newton steps jump between opposite clamp limits.
