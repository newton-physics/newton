Speed up mesh-mesh SDF and hydroelastic SDF contact generation without changing the produced contacts:

  - Hydroelastic octree refinement scans only the active records instead of the full worst-case buffers, and the marching-cubes kernel reads voxel corners with fewer texture lookups.
  - The mesh-SDF edge kernels use 128-thread blocks with overlapped pre-prune hashtable probes, and the contact reducer clears its active entries with more threads.

No migration is needed.
