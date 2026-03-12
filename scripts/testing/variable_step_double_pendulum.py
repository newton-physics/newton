import numpy as np
import warp as wp
import newton
import newton.examples


# Number of simulations to run in parallel
num_worlds = 4

# We'll use different timesteps for each world
min_dt = 0.001
max_dt = 0.01
dts = np.linspace(min_dt, max_dt, num_worlds).astype(np.float32)
dts = wp.array(dts, dtype=wp.float32)

# Create the model
cartpole = newton.ModelBuilder()
newton.solvers.SolverVariableStepMuJoCo.register_custom_attributes(cartpole)
cartpole.default_shape_cfg.density = 100.0
cartpole.default_joint_cfg.armature = 0.1
cartpole.default_body_armature = 0.1

cartpole.add_usd(
    newton.examples.get_asset("cartpole.usda"),
    enable_self_collisions=False,
    collapse_fixed_joints=True,
)
cartpole.joint_q[-3:] = [0.0, 0.3, 0.0]  # initial joint positions

builder = newton.ModelBuilder()
builder.replicate(cartpole, num_worlds, spacing=(1.0, 2.0, 0.0))
model = builder.finalize()

# Create the solver
solver = newton.solvers.SolverVariableStepMuJoCo(model)
state_0 = model.state()  # at the beginning of a simulation step
state_1 = model.state()  # at the end of a simulation step
control = model.control()

# Start the visualizer
viewer = newton.viewer.ViewerGL(headless=False)
viewer.set_model(model)

# Simulation loop
t = 0.0
while viewer.is_running():
    # TODO: should we be using wp.ScopedCapture and capture_launch here?
    with wp.ScopedTimer("simulate", active=False):
        # Apply forces from the visualizer (e.g., right click and drag).
        state_0.clear_forces()
        viewer.apply_forces(state_0)

        # Perform a simulation physics step (contacts = None).
        solver.step(state_0, state_1, control, None, dts)
        state_0, state_1 = state_1, state_0

        # This timestep is obviously nonsense: each world has a different
        # timestep, so some will appear to have slower physics.
        t += max_dt

    # Render at every step for now
    with wp.ScopedTimer("render", active=False):
        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.end_frame()