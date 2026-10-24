# LDL validation

Final broad regression run: **pending**. Earlier broad runs exposed the issues below and were stopped while they were fixed; they are not counted as passing runs.

## Numerical checks completed

- Five LDL numerical tests pass on CPU and CUDA with MECA and RCM. Coverage includes independent unit-lower L / explicit D reconstruction, sparse fill, heterogeneous matrices, multiple right-hand sides, in-place solves, masked failed-pivot retries, CUDA graph replay, and changing active dimensions.
- The dense capacity test reserves 1,537 rows while solving systems of 0, 7, 11, and 13 active rows. Dense PADMM now uses quadratic factor storage and an implicit complete block pattern, avoiding cubic symbolic allocation.
- The G1 velocity-predicate regression failed before the fix and passes afterward: a moving body must be rejected and a stationary body accepted.
- The earlier explicit-D regression failed against the original Cholesky implementation. The ordering regression failed before the ordering option/property was implemented.

## Broad run

```bash
uv run --no-sync --extra dev -m newton.tests \
  -k kamino -k basic.example_basic_joints -k basic.example_basic_urdf \
  --strict-warnings --no-cache-clear -j 4 -v
```

The selection includes Kamino unit tests, physics verification, contacts, solver integration, and Kamino-enabled examples. Both orderings are explicit variants for all six native Kamino example modules and seven shared joint/robot modules: pendulum, joints, URDF, ANYmal D, Allegro hand, H1, and G1. DR Legs exercises both DVI and dense PADMM. Other Kamino-enabled example tests retain their existing configurations, which use the MECA default where applicable.

## Issues discovered and addressed

1. G1's multiline lambda was extracted without its velocity comparison by Warp. Keep the comparison on one source line and test it with known moving/stationary bodies.
2. G1's speculative-contact culling caused intermittent ankle velocity spikes in the 256-world LDL-RCM run. Retaining near-touching foot contacts, as H1 already does, passed all six diagnostic checkpoints from frames 500 through 1,000. All twelve G1 benchmark cases are rerun with this setting for every backend.
3. Dense PADMM's reserved contact capacity created a complete sparse graph and cubic symbolic metadata. Replace that representation with implicit complete-pattern storage and native tiled LDL factorization of active rows. Sparse DVI still uses the joint graph and sparse factors.
4. A new dense retry regression caught a CUDA shift applied once per lane rather than once per diagonal entry. Build the regularization tile separately and add it cooperatively.

The ill-conditioned DR Legs frozen matrix remains a numerical limitation: float32 forward errors are substantial for both LDL and LLT despite small backward errors. The benchmark report includes the unmodified error measurements.
