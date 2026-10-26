Extend `newton.solvers.SolverVBD` rigid-soft DAT to complete mesh VT/TV/EE
primitives. Full-surface mesh collision always retains complete VT/TV/EE query
coverage; VBD selects force-eligible records at the current configuration using
OGC's wireframe edge region (Equation 14) for edge-edge contacts. If contact overflow
is reported, stop the simulation and increase contact capacity; DAT cannot
protect against missing contacts.

Preserve short nonzero edges in full-surface rigid-soft mesh closest-point queries.
