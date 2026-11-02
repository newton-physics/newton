Rewrite triangle-mesh self-contact storage: the result rows keep the same
interleaved (element, counterpart) layout as before but are exact-length now,
backed by an internal append log; storage is pooled across elements and
sized by a per-element average that grows on demand with peak contact
demand, rather than a fixed cap per element. `TriMeshCollisionInfo` drops the
`*_buffer_sizes` fields and gains `global_pair_counts`; `SolverVBD`'s contact-buffer
knobs and the pipeline/`Contacts` `*_pre_alloc` parameters now size one
global buffer per family (capacity = value x element count, shared by all
primitives); the defaults keep their previous values. On overflow
excess contacts are dropped and flagged; call
`TriMeshCollisionDetector.check_and_grow_collision_buffers()` (or the
`SolverVBD.check_and_grow_self_contact_buffers()` /
`CollisionPipeline.check_and_grow_soft_self_contact_buffers()` wrappers)
between steps to report and grow the storage in place. Contacts buffers
allocated before a grow are resized automatically at their next use, and
`SolverVBD.particle_self_contact_info()` returns the live self-contact
results for either construction path while keeping the solver's launch
size in sync.
`TriMeshCollisionDetector(sort_contact_rows=True)` optionally sorts each row
into a canonical order. Most self-contact demos run faster.
