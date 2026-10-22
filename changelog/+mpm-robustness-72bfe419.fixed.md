Fix `SolverImplicitMPM` robustness near material and collider boundaries and in Gauss-Seidel solves:

  - Retain strain modes with positive mass but cancelled integrals, and average material parameters with nonnegative weights.
  - Compute mesh collider distance signs from angle-weighted pseudonormals, rejecting sliver faces and falling back to the closest face when opposing faces cancel.
  - Keep the velocity, elastic strain, `Jp`, and stress of particles without a grid cell, such as inactive particles on rebuildable sparse grids, instead of resetting them to zero. Particles outside a fixed grid no longer extrapolate boundary-cell weights into NaN, and inactive particles inside grid cells keep their velocity.
  - Support capped particle strain bases (`picN`) on dense and fixed grids, which assumed one strain node per particle when assigning node volumes and transferring point warm starts.
  - Fix GIMP particle-domain integration and support-cell activation, including sparse grids with zero padding, and preserve per-overlap stress warm starts across cell crossings.
  - Run the full `max_iterations` budget without CUDA graphs, for example on CPU, where budgets below one 25- or 50-iteration batch previously ran no iterations and ignored contacts.
  - Fix Gauss-Seidel races on shared velocity nodes with particle (`pic*`) strain bases or `separate_worlds=True` on sparse grids, sweep identical worlds in the same order, update a cell's strain modes sequentially in `gs-soa`, and keep stress reads frozen within each `gs-batched` batch.
