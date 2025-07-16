Quickstart
==========

Requirements
------------

- Python 3.9 or higher
- NVIDIA GPU with driver 545 or newer (see note below)
- NVIDIA Warp from nightly build (see ``pyproject.toml`` for the current minimum version)

A local installation of the `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`__ is not required for Newton.
See the :doc:`../development-guide` for more details on how to set up a development environment and install
nightly builds.

**Note:** NVIDIA GPU driver 545+ is required for Warp kernel compilation *during* CUDA graph capture. 
Some examples using graph capture may fail with older drivers.

Real-time Rendering
-------------------

Newton provides a simple OpenGL renderer for visualizing the simulation.
The renderer requires pyglet (version >= 2.0) to be installed.

.. code-block:: python

    renderer = newton.utils.SimRendererOpenGL(model=model, path="Newton Simulator", scaling=2.0)

    # at every frame:
    renderer.begin_frame(sim_time)
    renderer.render(state)
    renderer.end_frame()

    # pause the simulation (blocks the control flow):
    renderer.pause = True

Keyboard shortcuts when working with the OpenGLRenderer (aka newton.utils.SimRendererOpenGL):

.. list-table:: Keyboard Shortcuts
    :header-rows: 1

    * - Key(s)
      - Description
    * - ``W``, ``A``, ``S``, ``D`` (or arrow keys) + mouse drag
      - Move the camera like in a FPS game
    * - ``X``
      - Toggle wireframe rendering
    * - ``B``
      - Toggle backface culling
    * - ``C``
      - Toggle the visibility of the coordinate system axes
    * - ``G``
      - Toggle ground grid
    * - ``T``
      - Toggle depth rendering
    * - ``I``
      - Toggle info text in the top left corner
    * - ``SPACE``
      - Pause/continue the simulation
    * - ``TAB``
      - Skip rendering to continue the simulation in the background (can speed up RL training while running the renderer)

USD Rendering
-------------

Instead of rendering in real-time, you can also render the simulation as a time-sampled USD stage to be visualized in Omniverse.

.. code-block:: python

    renderer = newton.utils.SimRenderer(model=model, path="simulation.usd", scaling=2.0)

    # at every frame:
    renderer.begin_frame(sim_time)
    renderer.render(state)
    renderer.end_frame()

    # to save the USD stage:
    renderer.save()

Example: Creating a particle chain
----------------------------------

.. testcode::

    import newton

    builder = newton.ModelBuilder()

    # anchor point (zero mass)
    builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

    # build chain
    for i in range(1, 10):
        builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
        builder.add_spring(i - 1, i, 1.0e3, 0.0, 0)

    model = builder.finalize("cpu")

    print(f"{model.spring_indices.numpy()=}")
    print(f"{model.particle_count=}")
    print(f"{model.particle_mass.numpy()=}")

.. testoutput::

    model.spring_indices.numpy()=array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9], dtype=int32)
    model.particle_count=10
    model.particle_mass.numpy()=array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32)
