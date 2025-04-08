# Newton

**⚠️ Alpha Stage Software ⚠️**

**This project is in active alpha development.** This means the API is unstable, features may be added or removed, and breaking changes are likely to occur frequently and without notice as the design is refined.

Newton is a GPU-accelerated physics simulation engine built upon [NVIDIA Warp](https://github.com/NVIDIA/warp), specifically targeting roboticists and simulation researchers.
It extends and generalizes Warp's existing `warp.sim` module, integrating [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) as a primary backend.
Newton emphasizes GPU-based computation, differentiability, and user-defined extensibility, facilitating rapid iteration and scalable robotics simulation.

Newton is maintained by [Disney Research](https://www.disneyresearch.com/), [Google DeepMind](https://deepmind.google/), and [NVIDIA](https://www.nvidia.com/).

## Development

Although not required, [uv](https://docs.astral.sh/uv/) is recommended for setting up a development environment.

Running a basic example:

```bash
uv run newton/examples/example_quadruped.py
```

Running all tests:

```bash
uv run -m newton.tests
```

Running all tests with code coverage:

```bash
uv run -m newton.tests --coverage --coverage-html htmlcov
xdg-open htmlcov/index.html
```

Creating a wheel:

```bash
uv build --wheel
```

[Ruff](https://docs.astral.sh/ruff/) is used for Python linting and code formatting.
[pre-commit](https://pre-commit.com/) can be used to ensure that local code complies with Newton's checks.
From the top of the repository, run:

```bash
uvx pre-commit run -a
```

To automatically run pre-commit hooks with `git commit`:

```bash
uvx pre-commit install
```
