Visualization
=============

OpenGL Viewer
-------------

Newton provides a simple OpenGL renderer for visualizing the simulation.
The renderer requires pyglet (version >= 2.0) to be installed.

.. code-block:: python

    renderer = newton.viewer.RendererOpenGL(model=model, path="Newton Simulator", scaling=2.0)

    # at every frame:
    renderer.begin_frame(sim_time)
    renderer.render(state)
    renderer.end_frame()

    # pause the simulation (blocks the control flow):
    renderer.pause = True

Keyboard shortcuts when working with the OpenGLRenderer (aka newton.viewer.RendererOpenGL):

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

Rendering to USD
----------------

Instead of rendering in real-time, you can also render the simulation as a time-sampled USD stage to be visualized in Omniverse.

.. code-block:: python

    renderer = newton.utils.SimRenderer(model=model, path="simulation.usd", scaling=2.0)

    # at every frame:
    renderer.begin_frame(sim_time)
    renderer.render(state)
    renderer.end_frame()

    # to save the USD stage:
    renderer.save()
