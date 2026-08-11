Import USD spheres whose scale is uniform to within floating-point round-off
without warning that non-uniform sphere scaling is unsupported. Scales reach the
importer through a single-precision transform decomposition, so an exactly
uniform scale composed through a nested transform chain could arrive with its
components a few ULP apart and trip an exact equality check. The warning now
also names the prim it applies to.
