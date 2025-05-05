Development Guide
=================

This document is a guide for developers who want to contribute to the project.

Environment setup
-----------------

Clone the repository
^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    git clone git@github.com:newton-physics/newton.git
    cd newton

Using uv
^^^^^^^^

`uv <https://docs.astral.sh/uv/>`_ is a Python package and project manager.

Install uv:

.. code-block:: console

    # On macOS and Linux.
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # On Windows.
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Run basic examples:

.. code-block:: console

    # An example with basic dependencies
    uv run newton/examples/example_quadruped.py

    # An example that requires extras
    uv run --all-extras newton/examples/example_humanoid.py

Running all tests with all extras installed:

.. code-block:: console

    uv run --all-extras -m newton.tests

Running all tests with code coverage on Ubuntu:

.. code-block:: console

    uv run --all-extras -m newton.tests --coverage --coverage-html htmlcov
    xdg-open htmlcov/index.html

Creating a wheel:

.. code-block:: console

    uv build --wheel

Updating all dependencies in the project `lockfile <https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile>`__,
remember to commit ``uv.lock`` after running:

.. code-block:: console

    uv lock -U

Using venv
^^^^^^^^^^

These instructions are meant for users who wish to set up a development environment using `venv <https://docs.python.org/3/library/venv.html>`__
or Conda (e.g. from `Miniforge <https://github.com/conda-forge/miniforge>`__).

.. code-block:: console

    python -m venv .venv

    # On macOS and Linux.
    source .venv/bin/activate
    
    # On Windows (console).
    .venv\Scripts\activate.bat

    # On Windows (PowerShell).
    .venv\Scripts\Activate.ps1

Installing dependencies including optional ones:

.. code-block:: console

    python -m pip install mujoco --pre -f https://py.mujoco.org/
    python -m pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
    python -m pip install git+https://github.com/google-deepmind/mujoco_warp.git@main
    python -m pip install -e .[dev]

Run basic examples:

.. code-block:: console

    # An example with basic dependencies
    python newton/examples/example_quadruped.py

    # An example that requires extras
    python newton/examples/example_humanoid.py

Running all tests with all extras installed:

.. code-block:: console

    python -m newton.tests

Code formatting and linting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Ruff <https://docs.astral.sh/ruff/>`_ is used for Python linting and code formatting.
`pre-commit <https://pre-commit.com/>`_ can be used to ensure that local code complies with Newton's checks.
From the top of the repository, run:

.. code-block:: console

    # With uv installed
    uvx pre-commit run -a

    # With venv
    python -m pip install pre-commit
    pre-commit run -a

To automatically run pre-commit hooks with `git commit`:

.. code-block:: console

    # With uv installed
    uvx pre-commit install

    # With venv
    pre-commit install

Building the documentation
--------------------------

Here are a few ways to build the documentation.

Using uv
^^^^^^^^

.. code-block:: console

    rm -rf docs/_build
    uv run --extra docs sphinx-build -b html docs docs/_build/html

    # Alternatively using the Makefile
    uv sync --extra docs
    source .venv/bin/activate
    cd docs
    make clean
    make html

Using venv
^^^^^^^^^^

.. code-block:: console

    python -m pip install -e .[docs]
    python -m sphinx -b html docs docs/_build/html

    # Alternatively using the Makefile
    cd docs
    make clean
    make html
