Installation
============

This guide will help you install Newton and set up your Python environment.

System Requirements
-------------------

- Python 3.10 or higher
- Windows or Linux on x86-64 architecture (Linux aarch64 is supported but not as thoroughly tested)
- NVIDIA GPU with compute capability >= 5.0 (Maxwell) and driver 545 or newer (see note below)

A local installation of the `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`__ is not required for Newton.

**Note:**
    - NVIDIA GPU driver 545+ is required for Warp kernel compilation *during* CUDA graph capture. Some examples using graph capture may fail with older drivers.
    - Unless otherwise specified, Newton's system requirements are identical to NVIDIA's `Warp <https://developer.nvidia.com/warp>`__ requirements.

Platform-Specific Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Linux aarch64 (ARM64)**

On ARM64 Linux systems (such as NVIDIA Jetson Thor and DGX Spark), installing the ``examples`` extras currently requires
X11 development libraries to build ``imgui_bundle`` from source:

.. code-block:: console

    sudo apt-get update
    sudo apt-get install -y libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev

Extra Dependencies
------------------

Newton's only mandatory dependency is `NVIDIA Warp <https://github.com/NVIDIA/warp>`_. Additional dependency sets are defined in the `pyproject.toml <https://github.com/newton-physics/newton/blob/main/pyproject.toml>`_ file:

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
     - Remeshing dependencies (Open3D, pyfqmr) for :class:`~newton.SurfaceReconstructor`
   * - ``examples``
     - Dependencies for running examples, including visualization (includes ``sim`` + ``importers``)
   * - ``torch-cu12``
     - PyTorch (CUDA 12) needed *in addition* to ``examples`` to run RL policy examples
   * - ``torch-cu13``
     - PyTorch (CUDA 13) needed *in addition* to ``examples`` to run RL policy examples
   * - ``notebook``
     - Jupyter notebook support with Rerun visualization (includes ``examples``)
   * - ``dev``
     - Dependencies for development and testing (includes ``examples``)
   * - ``docs``
     - Dependencies for building the documentation

Some extras transitively include others. For example, ``examples`` pulls in both
``sim`` and ``importers``, and ``dev`` pulls in ``examples``. You only need to
install the most specific set for your use case.

Installing from PyPI (Recommended)
----------------------------------

Basic installation:

.. code-block:: console

    pip install newton

Install with extras for running examples (includes simulation and visualization dependencies):

.. code-block:: console

    pip install "newton[examples]"

Install only simulation dependencies (without visualization):

.. code-block:: console

    pip install "newton[sim]"

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

Running Examples
^^^^^^^^^^^^^^^^

After installing Newton with the ``examples`` extra, run an example with:

.. code-block:: console

    python -m newton.examples basic_pendulum

Run an example that inferences an RL policy (requires ``torch-cu12`` or ``torch-cu13``):

.. code-block:: console

    pip install "newton[torch-cu12]"
    python -m newton.examples robot_anymal_c_walk

See a list of all available examples:

.. code-block:: console

    python -m newton.examples

Quick Start
^^^^^^^^^^^

After installing Newton, you can build models, create solvers, and run
simulations directly from Python. A typical workflow looks like this:

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

See the :doc:`/guide/key-concepts` guide and :doc:`/integrations/isaac-lab`
for more details.

Installing from Source
----------------------

Install from source if you want access to the full repository, tests, and the ``uv.lock`` lockfile for reproducible environments. This is recommended for developers and contributors.

Clone the Repository
^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    git clone git@github.com:newton-physics/newton.git
    cd newton

Method 1: Using uv (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install `uv <https://docs.astral.sh/uv/>`_:

.. tab-set::
    :sync-group: os

    .. tab-item:: macOS / Linux
        :sync: linux

        .. code-block:: console

            curl -LsSf https://astral.sh/uv/install.sh | sh

    .. tab-item:: Windows
        :sync: windows

        .. code-block:: console

            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

See also instructions on updating packages in the uv lockfile in the :doc:`development`.

Running Newton with uv
""""""""""""""""""""""

Run an example with minimal dependencies:

.. code-block:: console

    uv run -m newton.examples basic_pendulum --viewer null

Run an example with additional dependencies:

.. code-block:: console

    uv run --extra examples -m newton.examples robot_humanoid --world-count 16

Run an example that inferences an RL policy:

.. code-block:: console

    uv run --extra examples --extra torch-cu12 -m newton.examples robot_anymal_c_walk

See a list of all available examples with:

.. code-block:: console

    uv run -m newton.examples

Method 2: Using a Virtual Environment Setup by uv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`uv <https://docs.astral.sh/uv/>`_ can also be used to setup a virtual environment based on the `uv.lock <https://github.com/newton-physics/newton/blob/main/uv.lock>`_ file. You can setup a virtual environment with all ``examples`` dependencies by running:

.. code-block:: console

    uv venv
    uv sync --extra examples

Then you can activate the virtual environment and run an example using the virtual environment's Python:

.. tab-set::
    :sync-group: os

    .. tab-item:: macOS / Linux
        :sync: linux

        .. code-block:: console

            source .venv/bin/activate
            python newton/examples/robot/example_robot_humanoid.py

    .. tab-item:: Windows (console)
        :sync: windows

        .. code-block:: console

            .venv\Scripts\activate.bat
            python newton/examples/robot/example_robot_humanoid.py

    .. tab-item:: Windows (PowerShell)
        :sync: windows-ps

        .. code-block:: console

            .venv\Scripts\Activate.ps1
            python newton/examples/robot/example_robot_humanoid.py

Method 3: Manual Setup Using Pip in a Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These instructions are meant for users who wish to set up a development environment using `venv <https://docs.python.org/3/library/venv.html>`__
or Conda (e.g. from `Miniforge <https://github.com/conda-forge/miniforge>`__).

.. tab-set::
    :sync-group: os

    .. tab-item:: macOS / Linux
        :sync: linux

        .. code-block:: console

            python -m venv .venv
            source .venv/bin/activate

    .. tab-item:: Windows (console)
        :sync: windows

        .. code-block:: console

            python -m venv .venv
            .venv\Scripts\activate.bat

    .. tab-item:: Windows (PowerShell)
        :sync: windows-ps

        .. code-block:: console

            python -m venv .venv
            .venv\Scripts\Activate.ps1

Installing dependencies including optional development dependencies:

.. code-block:: console

    python -m pip install mujoco
    python -m pip install mujoco-warp
    python -m pip install warp-lang --pre -U -f https://pypi.nvidia.com/warp-lang/
    python -m pip install -e .[dev]

Test the installation by running an example:

.. code-block:: console

    python newton/examples/robot/example_robot_humanoid.py

Next Steps
----------

- Run ``python -m newton.examples`` to see all available examples and check out the :doc:`visualization` guide to learn how to interact with the example simulations.
- Check out the :doc:`development` guide to learn how to contribute to Newton.
