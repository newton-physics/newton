Add `SolverStyle3D.Config` to group solver options with the existing defaults;
pass `config=SolverStyle3D.Config(...)` at construction and edit public
`solver.config` between steps. Changes take effect on the next `step()` call;
recapture CUDA graphs after configuration changes.
