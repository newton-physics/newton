Improve DR Legs performance at the default solve block size by using Warp's
standard tile matrix multiplication for Kamino's blocked Cholesky solves when
libmathdx 0.4.2 or newer is available and `wp.config.enable_mathdx_gemm` is enabled.
Retain the native fallback otherwise; no API changes are required.
