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
     - macOS has no GPU acceleration; see :ref:`cpu-limitations` below
   * - NVIDIA GPU
     - Compute capability 5.0+ (Maxwell)
     - Any GeForce GTX 9xx or newer
   * - NVIDIA Driver
     - 545 or newer (CUDA 12)
     - 550 or newer (CUDA 12.4) recommended for best performance
   * - CUDA
     - 12, 13
     - No local CUDA Toolkit required; `Warp <https://github.com/NVIDIA/warp>`__ bundles its own runtime

CUDA Compatibility
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - CUDA Version
     - Notes
   * - 12.3+
     - Required for reliable CUDA graph capture
   * - 12.4+
     - Recommended for best performance
   * - 13
     - Supported

Tested Configurations
^^^^^^^^^^^^^^^^^^^^^

Newton releases are tested on the following configurations:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Component
     - Configuration
   * - OS
     - Ubuntu 22.04/24.04 (x86-64 + ARM64), Windows, macOS (CPU only)
   * - GPU
     - NVIDIA Ada Lovelace, Blackwell
   * - Python
     - 3.10, 3.12, 3.14 (import-only)
   * - CUDA
     - 12, 13

Platform-Specific Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Linux aarch64 (ARM64)**

On ARM64 Linux systems (such as NVIDIA Jetson Thor and DGX Spark), installing the ``examples`` extras currently requires
X11 development libraries to build ``imgui_bundle`` from source:

.. code-block:: console

    sudo apt-get update
    sudo apt-get install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev

.. _cpu-limitations:

CPU-Only Limitations
^^^^^^^^^^^^^^^^^^^^

Newton can run on CPU (including macOS), but the following features require an
NVIDIA GPU and are unavailable in CPU-only mode:

- **SDF collision** — signed-distance-field computation requires CUDA
  (``wp.Volume`` is GPU-only).
- **Mesh-mesh contacts** — SDF-based mesh-mesh collision is silently skipped on CPU.
- **Hydroelastic contacts** — depends on the SDF system.
- **Tiled camera sensor** — GPU-accelerated raytraced rendering.
- **Implicit MPM solver** — designed for GPU execution with CUDA graph support.
- **Tile-based VBD solve** — uses GPU tile API; gracefully disabled on CPU.

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

Run an example that runs RL policy inference. Choose the extra matching your
NVIDIA driver's CUDA support (``torch-cu12`` for CUDA 12.x, ``torch-cu13`` for
CUDA 13.x) and the corresponding pytorch wheel (e.g, ``128`` for CUDA 12.8); run ``nvidia-smi``
to check the supported CUDA version (shown in the top-right corner of the output):

.. code-block:: console

    pip install newton[torch-cu12] --extra-index-url https://download.pytorch.org/whl/cu128
    python -m newton.examples robot_anymal_c_walk

.. note::

    The ``torch-cu12`` extra installs PyTorch built against CUDA 12.8. If your
    driver only supports CUDA 12.4 or 12.5 (check with ``nvidia-smi``), you
    need to install PyTorch 2.6.0 manually instead:

    .. code-block:: console

        pip install "newton[examples]"
        pip install torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124

See a list of all available examples (also browsable from the viewer's side panel):

.. code-block:: console

    python -m newton.examples --list

Quick Start
^^^^^^^^^^^

After installing Newton, you can build
models, create solvers, and run simulations directly from Python. A typical
workflow looks like this:

.. code-block:: python

    import warp as wp
    import newton

    # Build a model
    builder = newton.ModelBuilder()
    builder.add_mjcf("robot.xml")        # or add_urdf() / add_usd()
    builder.add_ground_plane()
    model = builder.finalize()

    # Create a solver and allocate state
    solver = newton.solvers.SolverMuJoCo(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Step the simulation
    for step in range(1000):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0 / 4.0)
        state_0, state_1 = state_1, state_0

For robot-learning workflows with parallel environments (as used by
`Isaac Lab <https://isaac-sim.github.io/IsaacLab/>`_), you can replicate a
robot template across many worlds and step them all simultaneously on the GPU:

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

See the :doc:`/guide/overview` guide and :doc:`/integrations/isaac-lab`
for more details.

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
   * - ``examples``
     - Dependencies for running examples, including visualization (includes ``sim`` + ``importers``)
   * - ``torch-cu12``
     - PyTorch (CUDA 12.8+) for running RL policy examples (includes ``examples``); see :ref:`note above <running-examples>` for CUDA 12.4–12.5
   * - ``torch-cu13``
     - PyTorch (CUDA 13) for running RL policy examples (includes ``examples``)
   * - ``notebook``
     - Jupyter notebook support with Rerun visualization (includes ``examples``)
   * - ``dev``
     - Dependencies for development and testing (includes ``examples``)
   * - ``docs``
     - Dependencies for building the documentation

Some extras transitively include others. For example, ``examples`` pulls in both
``sim`` and ``importers``, and ``dev`` pulls in ``examples``. You only need to
install the most specific set for your use case.

.. _versioning:

Versioning
----------

Newton currently uses the following versioning scheme. This may evolve
depending on the needs of the project and its users.

Newton uses a **major.minor.micro** versioning scheme, similar to
`Python itself <https://devguide.python.org/developer-workflow/development-cycle/#devcycle>`__:

* New **major** versions are reserved for major reworks of Newton causing
  disruptive incompatibility (or reaching the 1.0 milestone).
* New **minor** versions are feature releases with a new set of features.
  May contain deprecations, breaking changes, and removals.
* New **micro** versions are bug-fix releases. In principle, there are no
  new features. The first release of a new minor version always includes
  the micro version (e.g., ``1.1.0``), though informal references may
  shorten it (e.g., "Newton 1.1").

Prerelease Versions
^^^^^^^^^^^^^^^^^^^

In addition to stable releases, Newton uses the following prerelease
version formats:

* **Development builds** (``major.minor.micro.dev0``): The version string
  used in the source code on the main branch between stable releases
  (e.g., ``1.1.0.dev0``).
* **Release candidates** (``major.minor.microrcN``): Pre-release versions
  for QA testing before a stable release, starting with ``rc1`` and
  incrementing (e.g., ``1.1.0rc1``). Usually not published to PyPI.

Prerelease versions should be considered unstable and are not subject
to the same compatibility guarantees as stable releases.

Next Steps
----------

- Run ``python -m newton.examples --list`` to see all available examples and check out the :doc:`visualization` guide to learn how to interact with the example simulations.
- Check out the :doc:`development` guide to learn how to contribute to Newton, or how to use alternative installation methods.
