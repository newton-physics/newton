Speed up the operational-space controllers' mass matrix inversions by keeping each Cholesky factor thread-local and solving one column of the inverse per thread.
