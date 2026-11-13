Rewrite triangle-mesh self-contact storage: the result rows keep the same
interleaved (element, counterpart) layout as before but are exact-length now,
backed by an internal append log; storage is pooled across elements and
sized by a per-element average that grows on demand with peak contact
demand, rather than a fixed cap per element. `TriMeshCollisionInfo` drops the
`*_buffer_sizes` fields and gains `global_pair_counts`; its `*_count` fields now hold
the stored (clamped) counts, so read overflow from `global_pair_counts` instead of
comparing a count against its capacity. `SolverVBD`'s contact-buffer
knobs and the pipeline/`Contacts` `*_pre_alloc` parameters now size one
global buffer per family (capacity = value x element count, shared by all
primitives); the defaults keep their previous values. On overflow
excess contacts are dropped and flagged; call
`SolverVBD.check_and_grow_self_contact_buffers()` or
`CollisionPipeline.check_and_grow_soft_self_contact_buffers()` between steps
to report and grow the storage in place, or `SolverVBD.check_self_contact_overflow()` /
`CollisionPipeline.check_soft_self_contact_overflow()` to read the per-family
pair demand and overflow flags without growing. Contacts buffers
allocated before a grow are resized automatically at their next use, and
`SolverVBD.particle_self_contact_info()` returns the live self-contact
results for either construction path while keeping the solver's launch
size in sync. In Warp's deterministic modes the solver derives the
per-thread record bound of its self-contact kernels from the pooled pair
capacity and the launch size and recomputes it when the storage grows.
Most self-contact demos run faster end to end;
detection alone costs about 4-14% more than before on the recorded example
scenes and 1.2-1.8x on uniformly dense synthetic two-layer meshes, where the
row build adds up to 43% of the pass. The append log behind the rows is
internal scratch of 12 bytes per pair slot (16 with the triangle-side reverse
rows), sized by the larger of the two family capacities and kept across
detections.
