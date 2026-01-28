# Merge Plan for divide_and_truncate_with_rbd Branch

## Three Stages

### Stage 1: Soft Body Solver + Unified Solver Path

**Includes:**
- All changes on particle_vbd_kernels
- All changes to tet and graph coloring related editing in builder.py
- All changes to solver_vbd.py
- Examples:
  - example_softbody_hanging.py → soft_body example
  - falling_gift.py → multi-physics demos
  - example_softbody_dropping_to_cloth.py → multi-physics demos

**Excludes:**
- Remove the changes to trimesh_collision_detector.py

**Modifications:**
- Only keep truncation mode 0 (old truncation method)
- Remove parameters that let users choose truncation modes

**Process:**
- Analyze the changes
- If other unisolatable changes are found, check with user whether to add them or not
- Create a new branch called `vbd_softbody` based off main
- Only carry requested changes to that branch, don't contaminate it

### Stage 2: New Truncation
(To be detailed later)

### Stage 3: CCD Infrastructure + Resizable Buffers
(To be detailed later)

---

## Current Focus
Working on Stage 1
