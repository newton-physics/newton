Use Warp's standard tile matrix multiplication for Kamino's blocked Cholesky
transpose updates. This improves CUDA DR Legs performance at the default solve
block size with libmathdx 0.4.2 or newer and MathDx GEMM enabled. Older Warp builds
can run these solves more slowly.
