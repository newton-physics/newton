Visualization
=============

Newton provides multiple viewer backends for different visualization needs, from real-time rendering to offline recording and external integrations.

Real-time Viewers
-----------------

OpenGL Viewer
~~~~~~~~~~~~~

Newton provides a simple OpenGL viewer for interactive real-time visualization of simulations.
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

Recording and Offline Viewers
-----------------------------

Recording to File (ViewerFile)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ViewerFile backend records simulation data to JSON or binary files for later replay or analysis. 
This is useful for capturing simulations for debugging, sharing results, or post-processing.

.. code-block:: python

    # Record to binary format (more efficient)
    viewer = newton.viewer.ViewerFile("simulation.bin", auto_save=True, save_interval=100)
    
    # Or record to JSON format (human-readable)
    viewer = newton.viewer.ViewerFile("simulation.json")

    viewer.set_model(model)

    # at every frame:
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()

    # Close to save the recording
    viewer.close()

Key parameters:

- ``output_path``: Path to the output file (.json for JSON format, .bin for binary CBOR2 format)
- ``auto_save``: If True, automatically save periodically during recording (default: True)
- ``save_interval``: Number of frames between auto-saves when auto_save=True (default: 100)

Rendering to USD
~~~~~~~~~~~~~~~~

Instead of rendering in real-time, you can also render the simulation as a time-sampled USD stage to be visualized in Omniverse or other USD-compatible tools.

.. code-block:: python

    viewer = newton.viewer.ViewerUSD(output_path="simulation.usd", fps=60, up_axis="Z")

    viewer.set_model(model)

    # at every frame:
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()

    # Save and close the USD file
    viewer.close()

External Integrations
---------------------

Rerun Viewer
~~~~~~~~~~~~

The ViewerRerun backend integrates with the `rerun <https://rerun.io>`_ visualization library, 
enabling real-time or offline visualization with advanced features like time scrubbing and data inspection.

**Installation**: Requires the rerun-sdk package:

.. code-block:: bash

    pip install rerun-sdk

**Usage**:

.. code-block:: python

    viewer = newton.viewer.ViewerRerun(
        server=True,                    # Start in server mode
        address="127.0.0.1:9876",      # Server address
        launch_viewer=True,            # Auto-launch web viewer
        app_id="newton-simulation"     # Application identifier
    )

    viewer.set_model(model)

    # at every frame:
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()

The rerun viewer provides a web-based interface with features like:

- Time scrubbing and playback controls
- 3D scene navigation
- Data inspection and filtering
- Recording and export capabilities

Utility Viewers
---------------

Null Viewer
~~~~~~~~~~~

The ViewerNull provides a no-operation viewer for headless environments or automated testing where visualization is not required.
It simply counts frames and provides stub implementations for all viewer methods.

.. code-block:: python

    # Run for 1000 frames without visualization
    viewer = newton.viewer.ViewerNull(num_frames=1000)

    viewer.set_model(model)

    while viewer.is_running():
        viewer.begin_frame(sim_time)
        viewer.log_state(state)
        viewer.end_frame()

This is particularly useful for:

- Performance benchmarking without rendering overhead
- Automated testing in CI/CD pipelines
- Running simulations on headless servers
- Batch processing of simulations

Choosing the Right Viewer
-------------------------

.. list-table:: Viewer Comparison
    :header-rows: 1

    * - Viewer
      - Use Case
      - Output
      - Dependencies
    * - ViewerGL
      - Interactive development and debugging
      - Real-time display
      - pyglet, imgui_bundle
    * - ViewerFile
      - Recording for replay/sharing
      - .json or .bin files
      - None
    * - ViewerUSD
      - Integration with 3D pipelines
      - .usd files
      - usd-core
    * - ViewerRerun
      - Advanced visualization and analysis
      - Web interface
      - rerun-sdk
    * - ViewerNull
      - Headless/automated environments
      - None
      - None