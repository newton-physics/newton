Visualization
=============

Newton provides multiple viewer backends for different visualization needs, from real-time rendering to offline recording and external integrations.

All viewer backends share a common interface with ``set_model()``, ``begin_frame()``, ``log_state()``, and ``end_frame()`` methods.

**Limiting rendered worlds**: When training with many parallel environments, rendering all worlds can impact performance. 
All viewers support the ``max_worlds`` parameter to limit visualization to a subset of environments:

.. code-block:: python

    # Only render the first 4 environments
    viewer.set_model(model, max_worlds=4)

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

Use :class:`~newton.viewer.ViewerFile` to load a recording, then restore the model and state for a given frame. Use :class:`~newton.viewer.ViewerGL` (or another rendering viewer) to visualize.

.. code-block:: python

    # Load a recording for playback
    viewer_file = newton.viewer.ViewerFile("simulation.bin")
    viewer_file.load_recording()

    # Restore the model and state from the recording
    model = newton.Model()
    viewer_file.load_model(model)

    state = model.state()
    viewer_file.load_state(state, frame_id=10)  # frame index in [0, get_frame_count())

    # Use ViewerGL for visualization
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.log_state(state)

For a complete example with UI controls for scrubbing and playback, see ``newton/examples/basic/example_replay_viewer.py``.

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

    # Default usage: spawns a local viewer
    viewer = newton.viewer.ViewerRerun(
        app_id="newton-simulation" # Optional application identifier to differentiate multiple parallel instances.
    )

    # Or specify a custom server address for remote viewing
    viewer = newton.viewer.ViewerRerun(
        address="rerun+http://127.0.0.1:9876/proxy", # Optional server address to connect to a remote rerun server.
        app_id="newton-simulation"
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

  frame_dt = 1 / 60.0
  sim_time = 0.0

  for frame in range(500):
      # simulate, step the solver, etc.
      solver.step(...)

      # visualize
      viewer.begin_frame(sim_time)
      viewer.log_state(state)
      viewer.end_frame()

      sim_time += frame_dt

  viewer.show_notebook()  # or simply `viewer` to display the viewer in the notebook
  
.. image:: /images/rerun_notebook_example.png
   :width: 1000
   :align: left

The history of states will be available in the viewer to scrub through the simulation timeline.

Viser Viewer
~~~~~~~~~~~~

The :class:`~newton.viewer.ViewerViser` backend integrates with the `viser <https://viser.studio>`_ visualization library,
providing web-based 3D visualization that works in any browser and has native Jupyter notebook support.

**Installation**: Requires the viser package:

.. code-block:: bash

    pip install viser

**Usage**:

.. code-block:: python

    # Default usage: starts a web server on port 8080
    viewer = newton.viewer.ViewerViser(port=8080)

    # Open http://localhost:8080 in your browser to view the simulation

    viewer.set_model(model)

    # at every frame:
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()

    # Close the viewer when done
    viewer.close()

Key parameters:

- ``port``: Port number for the web server (default: ``8080``)
- ``label``: Optional label for the browser window title
- ``verbose``: If True, print the server URL when starting (default: ``True``)
- ``share``: If True, create a publicly accessible URL via viser's share feature
- ``record_to_viser``: Path to record the visualization to a ``.viser`` file for later playback

**Recording and playback**

ViewerViser can record simulations to ``.viser`` files for later playback:

.. code-block:: python

    # Record to a .viser file
    viewer = newton.viewer.ViewerViser(record_to_viser="my_simulation.viser")

    viewer.set_model(model)

    # Run simulation...
    for frame in range(500):
        viewer.begin_frame(sim_time)
        viewer.log_state(state)
        viewer.end_frame()
        sim_time += frame_dt

    # Save the recording
    viewer.save_recording()

The recorded ``.viser`` file can be played back using the viser HTML player.

**Jupyter notebook support**

ViewerViser has native Jupyter notebook integration. When recording is enabled, calling ``show_notebook()`` 
will display an embedded player with timeline controls:

.. code-block:: python

    viewer = newton.viewer.ViewerViser(record_to_viser="simulation.viser")
    viewer.set_model(model)

    # Run simulation...
    for frame in range(500):
        viewer.begin_frame(sim_time)
        viewer.log_state(state)
        viewer.end_frame()
        sim_time += frame_dt

    # Display in notebook with timeline controls
    viewer.show_notebook()  # or simply `viewer` at the end of a cell

When no recording is active, ``show_notebook()`` displays the live server in an IFrame.

The viser viewer provides features like:

- Real-time 3D visualization in any web browser
- Interactive camera controls (pan, zoom, orbit)
- GPU-accelerated batched mesh rendering
- Recording and playback capabilities
- Public URL sharing via viser's share feature

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
    * - :class:`~newton.viewer.ViewerViser`
      - Browser-based visualization and Jupyter notebooks
      - Web interface, .viser files
      - viser
    * - :class:`~newton.viewer.ViewerNull`
      - Headless/automated environments
      - None
      - None