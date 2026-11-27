.. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

Installation
============

This guide covers the recommended way to install Newton from PyPI. For
installing from source or using ``uv``, see the :doc:`development` guide.

.. _system-requirements:

System Requirements
-------------------

Minimum Requirements
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 30 45
   :header-rows: 1

   * - Requirement
     - Minimum
     - Notes
   * - Python
     - 3.10
     - 3.11+ recommended
   * - OS
     - Linux (x86-64, aarch64), Windows (x86-64), or macOS (CPU only)
     - macOS has no GPU acceleration

GPU requirements depend on the installed `Warp
<https://nvidia.github.io/warp/stable/user_guide/compatibility.html>`__ wheel:

.. list-table::
   :widths: 20 25 25 30
   :header-rows: 1

   * - Warp wheel
     - NVIDIA GPU
     - NVIDIA driver
     - CUDA support
   * - 1.18+ (fresh installs)
     - 1.18: compute capability 7.5+ (Turing)
     - 1.18: R580 or newer
     - Standard 1.18 wheel uses CUDA 13.4; check Warp's requirements for later releases
   * - 1.17
     - Compute capability 5.2+ (Maxwell) on x86-64; see the ARM64 limitation below
     - 545 or newer (Newton 1.7 requirement)
     - Standard wheel uses CUDA 12.9

Newton 1.7 deprecates CUDA 12 GPU support, while retaining it with a Warp
wheel built with CUDA 12 throughout the 1.7 release line. To use the standard
Warp 1.17 wheel on an R545–R579 driver or an older supported GPU, install both
packages in one request::

    python -m pip install "newton" "warp-lang==1.17.0"

Keep the Warp pin in your application's requirements or constraints file for
later installs. A local CUDA Toolkit is not needed for standard Warp wheels.
See :ref:`cuda-compatibility` for version-specific notes.

Platform-Specific Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Linux aarch64 (ARM64)**

The standard Warp 1.17 wheel omits ``sm_52``, ``sm_60``, ``sm_61``, and
``sm_70`` targets on Linux ARM64. The installation command above does not
enable these GPUs. They need a Warp wheel built with an earlier CUDA 12
toolkit; see `Warp's CUDA 12.9 ARM64 limitation
<https://nvidia.github.io/warp/v1.17/user_guide/installation.html#cuda-12-9-limitation-on-linux-arm-platforms>`__.

On ARM64 Linux, the ``importers`` extra requires GLIBC 2.35 or newer because
`usd-exchange <https://pypi.org/project/usd-exchange/>`__ publishes its Linux
ARM64 wheels for ``manylinux_2_35``. This also applies to extras that include
``importers``, such as ``examples`` and ``dev``. Distributions with an older
GLIBC, including RHEL 9 with GLIBC 2.34, can install the base Newton package but
cannot install these extras from the published wheels.

Installing the ``examples`` extra on ARM64 Linux systems such as NVIDIA Jetson
Thor and DGX Spark also requires X11 development libraries to build
``imgui_bundle`` from source:

.. code-block:: console

    sudo apt-get update
    sudo apt-get install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev

For tested configurations and CUDA version-specific notes, see
:doc:`compatibility`.

Installing Newton
-----------------

Basic installation:

.. code-block:: console

    pip install newton

Install with the ``examples`` extra to run the built-in examples (includes simulation and visualization dependencies):

.. code-block:: console

    pip install "newton[examples]"

We recommend installing Newton inside a virtual environment to avoid conflicts
with other packages:

.. tab-set::
    :sync-group: os

    .. tab-item:: macOS / Linux
        :sync: linux

        .. code-block:: console

            python -m venv .venv
            source .venv/bin/activate
            pip install "newton[examples]"

    .. tab-item:: Windows (console)
        :sync: windows

        .. code-block:: console

            python -m venv .venv
            .venv\Scripts\activate.bat
            pip install "newton[examples]"

    .. tab-item:: Windows (PowerShell)
        :sync: windows-ps

        .. code-block:: console

            python -m venv .venv
            .venv\Scripts\Activate.ps1
            pip install "newton[examples]"

.. note::

    Users on Python 3.10 may experience issues when installing ``imgui_bundle`` (a dependency of the
    ``examples`` extra). If you encounter installation errors, we recommend upgrading to a later
    Python version, or follow the :doc:`development` guide to install Newton using ``uv``.

.. _running-examples:

Running Examples
^^^^^^^^^^^^^^^^

After installing Newton with the ``examples`` extra, launch the default
``basic_pendulum`` example — you can browse other examples from the side panel:

.. code-block:: console

    python -m newton.examples

Run an example that performs RL policy inference. The ``examples`` extra
includes ``newton[onnx]``, which installs Warp-NN's ONNX runtime and the ONNX
parser:

.. code-block:: console

    pip install "newton[examples]"
    python -m newton.examples robot_anymal_c_walk

See a list of all available examples (also browsable from the viewer's side panel):

.. code-block:: console

    python -m newton.examples --list

Quick Start
^^^^^^^^^^^

After installing Newton with the base package, you can build models, create
solvers, and run simulations directly from Python. This example uses only the
required dependencies installed by ``pip install newton``:

.. code-block:: python

    import warp as wp
    import newton

    # Build a model
    builder = newton.ModelBuilder()
    body = builder.add_body(
        xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_sphere(body, radius=0.25)
    builder.add_ground_plane()
    model = builder.finalize()

    # Create a solver and allocate state
    solver = newton.solvers.SolverXPBD(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Step the simulation
    for step in range(120):
        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0

The following workflow uses :class:`~newton.solvers.SolverMuJoCo`, so install
the optional simulation dependencies first:

.. code-block:: console

    pip install "newton[sim]"

Then build a robot template, replicate it across many worlds, and step them all
simultaneously on the GPU:

.. code-block:: python

    # Build a single robot template
    template = newton.ModelBuilder()
    template.add_mjcf("humanoid.xml")

    # Replicate into parallel worlds
    builder = newton.ModelBuilder()
    builder.replicate(template, world_count=1024)
    builder.add_ground_plane()
    model = builder.finalize()

    # The solver steps all 1024 worlds in parallel
    solver = newton.solvers.SolverMuJoCo(model)

See the :doc:`MuJoCo solver guide </solvers/mujoco>` for solver-specific
details, the :doc:`/guide/overview` for Newton's core workflow, and
:doc:`/lab/isaac-lab` for Isaac Lab integration details.

.. _extra-dependencies:

Extra Dependencies
------------------

Newton's only mandatory dependency is `NVIDIA Warp <https://github.com/NVIDIA/warp>`_.
Additional optional dependency sets are defined in ``pyproject.toml``:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Set
     - Purpose
   * - ``sim``
     - Simulation dependencies, including MuJoCo
   * - ``importers``
     - Asset import and mesh processing dependencies
   * - ``remesh``
     - Remeshing dependencies (Open3D, pyfqmr) for :func:`newton.utils.remesh_mesh`
   * - ``onnx``
     - Warp-NN ONNX runtime dependencies for neural actuators and RL policy examples
   * - ``examples``
     - Dependencies for running examples, including visualization and ONNX policy inference (includes ``sim`` + ``importers`` + ``onnx``)
   * - ``torch-cu12``
     - PyTorch (CUDA 12.8+) for workflows that explicitly need PyTorch, such as
       training or running Torch ``.pt2`` / ``.pt`` / ``.pth`` policies (includes
       ``examples``). On an R545–R579 driver, install with
       ``python -m pip install "newton[torch-cu12]" "warp-lang==1.17.0"``; the extra
       does not select Warp 1.17 by itself.
   * - ``torch-cu13``
     - PyTorch (CUDA 13) for workflows that explicitly need PyTorch, such as training or running Torch ``.pt2`` / ``.pt`` / ``.pth`` policies (includes ``examples``)
   * - ``notebook``
     - Jupyter notebook support with Rerun visualization (includes ``examples``)
   * - ``dev``
     - Dependencies for development and testing (includes ``examples``)
   * - ``docs``
     - Dependencies for building the documentation

Some extras transitively include others. For example, ``examples`` pulls in
``sim``, ``importers``, and ``onnx``, and ``dev`` pulls in ``examples``. You only
need to install the most specific set for your use case.

Next Steps
----------

- Run ``python -m newton.examples --list`` to see all available examples and check out the :doc:`visualization` guide to learn how to interact with the example simulations.
- See the :doc:`compatibility` guide for Newton's supported platforms, versioning scheme, and deprecation policy.
- Check out the :doc:`development` guide to learn how to contribute to Newton, or how to use alternative installation methods.
