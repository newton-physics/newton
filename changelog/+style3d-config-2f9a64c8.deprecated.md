Deprecate the `SolverStyle3D` constructor options `iterations`,
`linear_iterations`, `drag_spring_stiff`, and `enable_mouse_dragging`, and
the instance attributes `nonlinear_iterations`, `linear_iterations`,
`drag_spring_stiff`, and `enable_mouse_dragging`, as of Newton 1.8. They remain
functional during the deprecation period and emit `NewtonDeprecationWarning`.
Pass options through `config=SolverStyle3D.Config(...)` and access
`solver.config` fields instead; `solver.nonlinear_iterations` becomes
`solver.config.iterations`.
