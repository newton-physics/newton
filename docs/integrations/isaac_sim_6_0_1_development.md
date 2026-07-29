# Isaac Sim 6.0.1 Development

This branch is the Newton 1.2.1 source checkout used by the local Isaac Sim 6.0.1 installation. Keep solver changes on `vegtsunami/isaacsim-6.0.1`; the fork's `main` branch tracks newer Newton development and is not dependency-compatible with this Isaac Sim release.

## Start Isaac Sim

Run the full Newton GUI with this checkout as the active Newton source:

```bash
./tools/isaac_sim_6_0_1/launch.sh
```

The launcher performs an external import preflight and enables the repository-owned `newton.dev.source` Kit extension. That extension loads the checkout before Isaac Sim's bundled Newton prebundle. The launcher does not modify files under `/home/limx/apps/isaacsim-6.0.1`.

To use another Isaac Sim 6.0.1 installation, set `ISAAC_SIM_ROOT`:

```bash
ISAAC_SIM_ROOT=/path/to/isaacsim-6.0.1 ./tools/isaac_sim_6_0_1/launch.sh
```

Additional arguments are forwarded to the stock `isaac-sim.newton.sh` launcher.

## Run the Verified Demo

```bash
./tools/isaac_sim_6_0_1/launch.sh \
  --exec "$PWD/tools/isaac_sim_6_0_1/demo_rigid_bodies.py"
```

Successful startup prints values like:

```text
NEWTON_DEV source=/path/to/this/checkout
NEWTON_DEV newton_version=1.2.1
NEWTON_DEV warp_version=1.13.0
NEWTON_DEV runtime_source=/path/to/this/checkout/newton/__init__.py
NEWTON_DEV active_engine=newton device=cuda:0
```

The GUI scene contains a lit ground, four colored cubes, and one colored sphere. The bodies fall and collide using Newton on CUDA.

## Develop a Solver

Newton solver implementations are under `newton/_src/solvers/`. Restart Isaac Sim after editing solver Python or Warp kernel code. Warp recompiles kernels whose source changed and otherwise reuses its cache.

Before adding a custom solver to Isaac Sim, implement and test it through Newton's public `SolverBase` contract. Isaac Sim 6.0.1's Newton integration currently selects only `SolverMuJoCo` and `SolverXPBD`, so a new solver also needs a focused integration change to the solver factory after its Newton implementation is stable.

## Check the Checkout

```bash
git branch --show-current
git remote -v
```

Expected branch and remotes:

```text
vegtsunami/isaacsim-6.0.1
origin   https://github.com/little-veg/newton.git
upstream https://github.com/newton-physics/newton.git
```

## Synchronize Upstream Deliberately

```bash
git fetch upstream --tags
git log --oneline v1.2.1..upstream/main
```

Do not merge `upstream/main` or the fork's `main` wholesale into this branch. Current upstream development requires newer Warp, MuJoCo Warp, schemas, and Isaac integration code. Review and cherry-pick compatible fixes individually.

## Troubleshooting

If startup reports that Newton came from outside the expected repository, inspect the `NEWTON_DEV source` and `NEWTON_DEV runtime_source` lines. Always start Isaac Sim through `tools/isaac_sim_6_0_1/launch.sh` when testing fork changes.

If the launcher cannot find Warp 1.13 or the Newton prebundle, confirm that `ISAAC_SIM_ROOT` points to the complete Isaac Sim 6.0.1 installation.

Warnings about the CPU power profile, IOMMU, Omniverse Cache, or Fabric interface versions are emitted by the stock Isaac Sim installation and do not indicate that source selection failed.

## Roll Back to Bundled Newton

Launch the stock application directly:

```bash
/home/limx/apps/isaacsim-6.0.1/isaac-sim.newton.sh
```

Because the development workflow changes no Isaac Sim installation files, rollback needs no package restoration.
