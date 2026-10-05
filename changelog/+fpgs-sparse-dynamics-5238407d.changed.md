Use topology-derived sparse mass factors and contact rows for eligible branched
FeatherPGS articulations in matrix-free CUDA solves, including contacts between
articulations, independent rigid bodies, and prescribed supports, without changing
solver budgets. Existing configurations need no changes; unsupported shapes and
features retain their existing representation.
Reuse contact geometry and packed factor loads across normal and tangent rows,
with warp-local Jacobians and support-only traversal instead of global staging arrays.
