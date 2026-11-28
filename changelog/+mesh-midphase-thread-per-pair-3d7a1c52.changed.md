Speed up the mesh midphase for scenes with many mesh-convex pairs by querying each pair's mesh BVH in its own thread; the contacts are unchanged.
