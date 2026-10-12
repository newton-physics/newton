Fix `SolverImplicitMPM` crashing on CPU, and reading out of bounds on CUDA, when setting up the `gs-soa` and `gs-batched` rheology solvers on sparse grids with a positive `max_active_cell_count`.
