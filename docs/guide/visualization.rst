Visualization
=============

OpenGL Viewer
-------------

Newton provides a simple OpenGL viewer for visualizing the simulation.
The viewer requires pyglet (version >= 2.1.6) and imgui_bundle (version >= 1.92.0) to be installed.

.. code-block:: python

    viewer = newton.viewer.ViewerGL()

    viewer.set_model(model)

    # at every frame:
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()

    # pause the simulation (blocks the control flow):
    viewer.pause = True

Keyboard shortcuts when working with the OpenGL Viewer (aka newton.viewer.ViewerGL):

.. list-table:: Keyboard Shortcuts
    :header-rows: 1

    * - Key(s)
      - Description
    * - ``W``, ``A``, ``S``, ``D`` (or arrow keys) + mouse drag
      - Move the camera like in a FPS game
    * - ``H``
      - Toggle Sidebar
    * - ``SPACE``
      - Pause/continue the simulation
    * - ``Right Click``
      - Pick objects

Rendering to USD
----------------

Instead of rendering in real-time, you can also render the simulation as a time-sampled USD stage to be visualized in Omniverse.

.. code-block:: python

    viewer = newton.viewer.ViewerUSD(output_path="simulation.usd")

    # at every frame:
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()
