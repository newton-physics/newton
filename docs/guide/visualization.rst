Visualization
=============

Newton provides multiple viewer backends for different visualization needs, from real-time rendering to offline recording and external integrations.

Real-time Viewers
-----------------

OpenGL Viewer
~~~~~~~~~~~~~

Newton provides :class:`~newton.viewer.ViewerGL`, a simple OpenGL viewer for interactive real-time visualization of simulations.
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

**Troubleshooting:**

If you encounter an OpenGL context error on Linux with Wayland:

.. code-block:: text

    OpenGL.error.Error: Attempt to retrieve context when no valid context

Set the PyOpenGL platform before running:

.. code-block:: bash

    export PYOPENGL_PLATFORM=glx

This is a known issue when running OpenGL applications on Wayland display servers.

Recording and Offline Viewers
-----------------------------

Recording to File (ViewerFile)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~newton.viewer.ViewerFile` backend records simulation data to JSON or binary files for later replay or analysis. 
This is useful for capturing simulations for debugging, sharing results, or post-processing.

**File formats:**

- ``.json``: Human-readable JSON format (no additional dependencies)
- ``.bin``: Binary CBOR2 format (more efficient, requires ``cbor2`` package)

To use binary format, install the optional dependency:

.. code-block:: bash

    pip install cbor2

**Recording a simulation:**

.. code-block:: python

    # Record to binary format (more efficient, requires cbor2)
    viewer = newton.viewer.ViewerFile("simulation.bin", auto_save=True, save_interval=100)
    
    # Or record to JSON format (human-readable, no extra dependencies)
    viewer = newton.viewer.ViewerFile("simulation.json")

    viewer.set_model(model)

    # at every frame:
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()

    # Close to save the recording
    viewer.close()

**Loading and playing back recordings:**

.. code-block:: python

    # Load a recording for playback
    recorder = newton.utils.RecorderModelAndState()
    recorder.load_from_file("simulation.bin")

    # Create model and state for playback
    model = newton.Model()
    state = newton.State()

    # Restore the model from the recording
    recorder.playback_model(model)

    # Playback a specific frame (e.g., frame 10)
    recorder.playback(state, frame_index=10)

    # Use with any viewer to visualize
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.log_state(state)

For a complete example with UI controls for playback, see ``newton/examples/example_replay_viewer.py``.

Key parameters:

- ``output_path``: Path to the output file (format determined by extension: .json or .bin)
- ``auto_save``: If True, automatically save periodically during recording (default: ``True``)
- ``save_interval``: Number of frames between auto-saves when auto_save=True (default: ``100``)

Rendering to USD
~~~~~~~~~~~~~~~~

Instead of rendering in real-time, you can also render the simulation as a time-sampled USD stage to be visualized in Omniverse or other USD-compatible tools using the :class:`~newton.viewer.ViewerUSD` backend.

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

The :class:`~newton.viewer.ViewerRerun` backend integrates with the `rerun <https://rerun.io>`_ visualization library, 
enabling real-time or offline visualization with advanced features like time scrubbing and data inspection.

**Installation**: Requires the rerun-sdk package:

.. code-block:: bash

    pip install rerun-sdk

**Usage**:

.. code-block:: python

    viewer = newton.viewer.ViewerRerun(
        server=True,                   # Start in server mode
        address="127.0.0.1:9876",      # Server address
        launch_viewer=True,            # Auto-launch web viewer
        app_id="newton-simulation"     # Application identifier
    )

    viewer.set_model(model)

    # at every frame:
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()

By default, the viewer will run without keeping historical state data in the viewer to keep the memory usage constant when sending transform updates via :meth:`ViewerRerun.log_state`.
This is useful for visualizing long and complex simulations that would quickly fill up the web viewer's memory if the historical data was kept.
If you want to keep the historical state data in the viewer, you can set the ``keep_historical_data`` flag to ``True``.

The rerun viewer provides a web-based interface with features like:

- Time scrubbing and playback controls
- 3D scene navigation
- Data inspection and filtering
- Recording and export capabilities

**Jupyter notebook support**

The ViewerRerun backend automatically detects if it is running inside a Jupyter notebook environment and automatically generates an output widget for the viewer
during the construction of :class:`~newton.viewer.ViewerRerun`.

The rerun SDK provides a Jupyter notebook extension that allows you to visualize rerun data in a Jupyter notebook.

You can use ``uv`` to start Jupyter lab with the required dependencies (or install the extension manually with ``pip install rerun-sdk[notebook]``):

.. code-block:: bash

  uv run --extra notebook jupyter lab

Then, you can use the rerun SDK in a Jupyter notebook by importing the :mod:`rerun` module and creating a viewer instance.

.. code-block:: python

  viewer = newton.viewer.ViewerRerun(keep_historical_data=True)
  viewer.set_model(model)

  for frame in trange(500):
      wp.capture_launch(graph)

      viewer.begin_frame(sim_time)
      viewer.log_state(state)
      viewer.log_contacts(contacts, state)
      viewer.end_frame()

      sim_time += frame_dt
  
  # The history of states will be available in the viewer to scrub through
  # the simulation timeline.

Utility Viewers
---------------

Null Viewer
~~~~~~~~~~~

The :class:`~newton.viewer.ViewerNull` provides a no-operation viewer for headless environments or automated testing where visualization is not required.
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
    * - :class:`~newton.viewer.ViewerGL`
      - Interactive development and debugging
      - Real-time display
      - pyglet, imgui_bundle
    * - :class:`~newton.viewer.ViewerFile`
      - Recording for replay/sharing
      - .json or .bin files
      - None
    * - :class:`~newton.viewer.ViewerUSD`
      - Integration with 3D pipelines
      - .usd files
      - usd-core
    * - :class:`~newton.viewer.ViewerRerun`
      - Advanced visualization and analysis
      - Web interface
      - rerun-sdk
    * - :class:`~newton.viewer.ViewerNull`
      - Headless/automated environments
      - None
      - None