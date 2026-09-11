Rewrite triangle-mesh self-contact storage: detection now appends contact pairs
to one shared array per family (vertex-triangle and edge-edge) instead of
fixed-size per-element buffers, and the per-element query tables become exact
CSR rows over those arrays. `SolverVBD`'s `particle_vertex_contact_buffer_size`
and `particle_edge_contact_buffer_size` keep their names but now mean the
average contact budget per element (array capacity = budget x element count),
with defaults lowered from 32/64 to 8/16; a locally dense fold can no longer
overflow a private per-element budget, and on pool overflow the solver warns
and grows the arrays automatically outside CUDA graph capture
(`SolverVBD.check_and_grow_self_contact_buffers()`). Self-contact force
accumulation and the planar truncation guard run one thread per stored contact,
which reduces self-contact memory several-fold and speeds up most self-contact
demos. Warp's deterministic-atomics mode no longer covers the self-contact
force scatter (its record bound relied on the fixed per-element rows), so
self-contact forces are not bitwise reproducible run to run even under
`deterministic=...`; all other deterministic-atomics coverage is unchanged, and
a reproducible fixed-order summation is planned as a follow-up. The
`CollisionPipeline`/`Contacts` self-contact `*_buffer_pre_alloc` parameters
adopt the same average-per-element semantics and 8/16 defaults; standalone
pipeline users (without `SolverVBD`) can poll
`TriMeshCollisionDetector.check_self_contact_overflow()` after detection, as
nothing grows the arrays automatically on that path.
