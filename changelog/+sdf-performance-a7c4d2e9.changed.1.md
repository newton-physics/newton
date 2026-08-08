Cache packed mesh-SDF collision edge centers and half-vectors and specialize their contact kernels, removing repeated mesh-index and vertex fetches without changing contact generation.
