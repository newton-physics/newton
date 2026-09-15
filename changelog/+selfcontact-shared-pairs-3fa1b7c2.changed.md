Rewrite triangle-mesh self-contact storage: result rows keep the historical
interleaved layout but are exact-length now, backed by an internal append log,
so memory scales with actual contacts. `TriMeshCollisionInfo` drops the
`*_buffer_sizes` fields and gains `global_pair_counts`; `SolverVBD`'s contact-buffer
knobs and the pipeline/`Contacts` `*_pre_alloc` parameters now mean an average
budget per element, with defaults lowered from 32/64 to 8/16. On overflow
excess contacts are dropped and flagged; call
`TriMeshCollisionDetector.check_and_grow_collision_buffers()` (or the
`SolverVBD.check_and_grow_self_contact_buffers()` wrapper) between steps to
report and grow the storage in place. Self-contact force sums are no longer
covered by Warp's deterministic-atomics mode, and
`TriMeshCollisionDetector(sort_contact_rows=True)` optionally sorts each row
into a canonical order. Most self-contact demos run faster.
