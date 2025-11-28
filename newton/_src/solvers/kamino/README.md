# Kamino

<div style="background-color:#fff3cd; border:1px solid #ffecb5; padding:0.75em 1em; border-radius:6px;">
<strong>⚠️ Disclaimer ⚠️ </strong>

This project is currently in ALPHA development phase.

At present time we discourage users of Newton from depending on it.

Similarly, and due to limited bandwidth of the development team, we will NOT be accepting contributions from the community.

A stable BETA version is planned for release during the spring of 2026.
</div>

## Introduction

Kamino is a solver back-end for simulating arbitrary mechanical assemblies that may feature kinematic loops.

It currently supports:
- Constrained rigid multi-body systems with arbitrary joint topologies (i.e. a kinematic tree is not assumed)
- A large set of common and advanced bilateral joint constraints
- Unilateral joint-limit and contact constraints with spatial friction and restitutive impacts
- Fully configurable constraint stabilization that can be specified per constraint subset
- Both hard and soft constraints enforced via an advanced Proximal-ADMM forward dynamics solver

Kamino is being developed and maintained by [Disney Research](https://www.disneyresearch.com/) in collaboration with [NVIDIA](https://www.nvidia.com/) and [Google DeepMind](https://deepmind.google/).


## Getting Started

### Installing

For plain users, the official [installation instructions](https://newton-physics.github.io/newton/guide/installation.html) of Newton are recommended.

For developers, please refer to the [Development](#development) section below.


### Running Examples

A set of examples is provided in `newton/_src/solvers/kamino/examples`.

These can be run directly as stand-alone scripts, for example:
```bash
cd newton/_src/solvers/kamino/examples
python sim/example_sim_cartpole.py
```

All examples will eventually be migrated, and integrated, into the main set of examples provided in Newton located in `newton/examples`.


### Running Unit Tests

```bash
cd newton/_src/solvers/kamino/tests
python -m unittest discover -s . -p 'test_*.py'
```

Please refer to the [Unit Tests](./tests/README.md) `README.md` for further instructions for how to run unit tests using IDEs such as VS Code.

All tests will eventually be migrated, and integrated, into the main set of unit-tests provided in Newton located in `newton/tests`.

## Development

Development of Kamino requires the installation of [Newton](https://github.com/newton-physics/newton) from source and the latest version of [Warp](https://github.com/NVIDIA/warp) either through nightly builds or also from source.

The first step involves setting-up a python environment.

The simplest is to create a new `virtualenv`. Alternatively one could
follow the [instructions](https://newton-physics.github.io/newton/guide/installation.html) from Newton.

### Virtual environments using `pyenv`
Because we're working on a fork of the main Newton repository, it can be useful to create two `virtualenv|conda|uv` environments.

Using `pyenv` and `virtualenv` it would look something like this:
- one for development of Kamino within our fork
```bash
pyenv virtualenv kamino-dev
pyenv activate kamino-dev
pip install -U pip
```
- and a second for modifying Newton code outside of Kamino that must go through pull requests directly to the main repo:
```bash
pyenv virtualenv newton-dev
pyenv activate newton-dev
pip install -U pip
```

A similar setup can be achieved via `conda|uv`. We've used the `*-dev` suffix to denote environments were the packages will be installed from source, while this can be omitted when creating environments to test installations when installing from `pip` wheels.


### APT (Only Required for Linux)
On Linux platforms, e.g. Ubuntu, the following base APT packages must be installed:
```bash
sudo apt-get update
sudo apt-get install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev
```

### MuJoCo
The first `pip` package to install is MuJoCo:
```bash
pip install mujoco --pre -f https://py.mujoco.org/
```

**NOTE**:
This must be installed first as it will pull the required version of foundational dependencies such as `numpy`.


### Warp
Nightly builds of Warp can be installed using:
```bash
pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
```

Alternatively, Warp can be installed from source (recommended) using:
```bash
git clone git@github.com:NVIDIA/warp.git
cd warp
python build_lib.py
pip install -e .[dev,extras,docs]
```

**NOTE**:
Many new features and fixes in Warp that are requested by Newton developers come quite often, so keeping up to date with Warp `main` can prove useful.


### MuJoCo Warp
MuJoCo Warp (a.k.a. MJWarp) can be installed from source using:
```bash
pip install git+https://github.com/google-deepmind/mujoco_warp.git@main
```

For development purposes, it can also be installed explicitly with optional dependencies from source using:
```bash
git clone git@github.com:google-deepmind/mujoco_warp.git
cd mujoco_warp
pip install -e .[dev,cuda]
```


### Newton
Newton needs to be installed from source for Kamino development using:
```bash
git clone git@github.com:vastsoun/newton.git
cd newton
pip install -e .[dev,data,docs]
```

**WARNING**:
Note that currently the development branch where Kamino is being worked on is `dev/kamino`. All PR's within the fork must be issued to that instead of `main` and take care to not to accidentally issue them to `main` of the upstream Newton repository.

## Further Reading

The following [technical report](https://arxiv.org/abs/2504.19771) provides an in-depth description of the problem formulation and algorithms used within the solver:
```bibtex
@article{tsounis:2025,
      title={On Solving the Dynamics of Constrained Rigid Multi-Body Systems with Kinematic Loops},
      author={Vassilios Tsounis and Ruben Grandia and Moritz Bächer},
      year={2025},
      eprint={2504.19771},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2504.19771},
}
```

----