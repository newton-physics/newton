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

1. Clone the repository
-----------------------

.. code-block:: console

    git clone git@github.com:newton-physics/newton.git
    cd newton

2. Python Environment Setup
---------------------------

We recommend using the `uv <https://docs.astral.sh/uv/>`_ Python package and project manager. It will automatically setup a version-locked Python environment based on the `uv.lock <https://github.com/newton-physics/newton/blob/main/uv.lock>`_ file that the Newton team maintains.

.. note::
    During the alpha development phase, we recommend using uv. When Newton is stabilized and regularly publishing to PyPI we will update this guide to make the pip install approach the recommended method.

Extra Dependencies
^^^^^^^^^^^^^^^^^^

Newton's only mandatory dependency is `NVIDIA Warp <https://github.com/NVIDIA/warp>`_. We define additional dependency sets in the `pyproject.toml <https://github.com/newton-physics/newton/blob/main/pyproject.toml>`_ file. The sets are:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Set
     - Purpose
   * - ``sim``
     - Simulation dependencies, including MuJoCo
   * - ``importers``
     - Asset import and mesh processing dependencies
   * - ``examples``
     - Dependencies for running examples, including visualization
   * - ``torch-cu12``
     - PyTorch dependency needed *in addition* to ``examples`` dependencies to run examples that inference RL-trained control policies
   * - ``dev``
     - Dependencies for development and testing
   * - ``docs``
     - Dependencies for building the documentation

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

    uv run --extra examples -m newton.examples robot_humanoid --num-worlds 16

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

- Explore more examples in the ``newton/examples/`` directory and checkout the :doc:`visualization` guide to learn how to interact with the examples simulation.
- Check out the :doc:`development` guide to learn how to contribute to Newton.
