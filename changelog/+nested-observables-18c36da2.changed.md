Nest the experimental observable container under `SolverBase.Observables` and
allocate each solver's `Observables` subclass directly, removing the standalone
`SolverObservables` export and `OBSERVABLES_TYPE` hook. Rename
`SolverObservableFlags` to `SolverObservableKind` and
`SolverMuJoCo.ObservableFlags` to `SolverMuJoCo.ObservableKind`. Use `kind=` in
field declarations, `kinds` in requests and selections,
`SUPPORTED_OBSERVABLE_KINDS` / `supported_observable_kinds` for solver
capabilities, and `solver_observable_kinds` for sensor requirements.
