Fix implicit MPM degenerate quantities near material and collider boundaries:

  - Strain modes with positive mass but cancelled integrals are retained, and material parameters are averaged with nonnegative weights.
  - Mesh collider distance signs use angle-weighted pseudonormals, reject sliver faces, and fall back to the closest face when opposing faces cancel.
