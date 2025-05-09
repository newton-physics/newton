Key Concepts
============

The Newton physics engine consists of several components that work together to simulate the physics of a scene. The main components are:

.. list-table:: Simulation Components
    :header-rows: 1

    * - Component
      - Responsibility
      - Type
    * - :class:`newton.Model`
      - Definition of simulation elements (bodies, joints, geometry, etc.)
      - Input
    * - :class:`newton.State`
      - Time-varying variables (joint positions, velocities)
      - Input/Output
    * - :class:`newton.Control`
      - Actuator inputs (joint & body forces)
      - Input
    * - :class:`newton.Contact`
      - Per-step collision information
      - Input/Output
    * - ``newton.solvers.*``
      - Advance simulation forward (state_in → state_out)
      - Compute

Simulation Loop
---------------

At every simulation step, the :meth:`newton.solvers.SolverBase.step` method takes the current state of the simulation and computes the next state based on the model, control inputs, and contact information.
The solver is responsible for integrating the equations of motion and updating the state of the simulation.

.. mermaid::
    
    flowchart TD

    Model(["Model"]) 
    State_in(["State in"])
    Ctrl(["Control"]) 
    Contact(["Contact"])

    Model --> Solver
    State_in --> Solver(["Solver.step()"])
    Ctrl --> Solver
    Contact --> Solver
    Solver --> State_out(["State out"])

.. code-block:: python

    import newton

    # Create a model
    builder = newton.ModelBuilder()
    # Add bodies, joints, and other elements to the model using builder methods

    model = builder.finalize()

    # Create a state
    state = model.state()

    # Create a control
    control = newton.Control(model)

    # Create a contact object
    contact = newton.Contact(model)

    # Create a solver
    solver = newton.MuJoCoSolver(model)

    # Run the simulation step
    for _ in range(num_frames):
        state_0.clear_forces()
        solver.step(model, state_0, state_1, control, contact, sim_dt)
        state_0, state_1 = state_1, state_0

Model Creation
--------------

To ease the creation of models, the :class:`newton.ModelBuilder` class provides a convenient interface for building models programmatically.
Several importers are provided to parse simulation assets, including USD, URDF, and MJCF files.

Once the scene is set up, :meth:`newton.ModelBuilder.finalize` is called to create the model, which will initialize the
Warp arrays and other data structures needed for the simulation on the desired device.

The :class:`newton.State` is constructed from the model via the :meth:`newton.Model.state` method.

.. mermaid::

    flowchart LR
    subgraph importers["Importing assets"]
        USD
        URDF
        MJCF
    end
        ModelBuilder(["ModelBuilder"])
    
        USD -- parse_usd() --> ModelBuilder
        URDF -- parse_usd() --> ModelBuilder
        MJCF -- parse_usd() --> ModelBuilder
        ModelBuilder -- add_builder() --> ModelBuilder
        ModelBuilder -- finalize() --> Model(["Model"])

Solvers
-------

.. autoclasstree:: newton.solvers

