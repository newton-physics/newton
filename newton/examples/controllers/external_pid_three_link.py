# THIS IS NOT A REAL EXAMPLE!!!!!
# TEMPORARY FILE USED FOR INITIAL TESTS,
# WILL BE DELETED.


from contextlib import nullcontext
from types import SimpleNamespace

import warp as wp

import newton
from newton.controllers import ControlLawPID, Controller

# BUILD UP THE ARM MODEL.
with nullcontext():
    # first, build up the actual model we will be using.
    # lets make it a 3DOF manipulator with no joint limits!
    arm_model_builder = newton.ModelBuilder()

    # make the links:
    link0 = arm_model_builder.add_link(mass=1.0)
    link1 = arm_model_builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(1.0, 1.0, 1.0)))
    link2 = arm_model_builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(1.0, 1.0, 1.0)))
    link3 = arm_model_builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(1.0, 1.0, 1.0)))

    box_length = 1.0
    box_width = 0.05

    # add a tool frame.
    tool_site = arm_model_builder.add_site(
        link3,
        label="tool",
        xform=wp.transform(p=(box_length / 2, 0.0, 0.0)),
        type=newton.GeoType.SPHERE,  # already the default; included for clarity
        scale=(0.05, 0.05, 0.05),  # how big to render
        visible=True,  # turn on rendering
    )

    # give all of the links the same simple box geometry:
    for link_i in [link1, link2, link3]:
        arm_model_builder.add_shape_box(
            link_i,
            hx=0.5 * box_length,
            hy=0.5 * box_width,
            hz=0.5 * box_width,
        )

    # make the joints:
    armature = 0.05
    ke = 300.0
    kd = 160.0

    j0 = arm_model_builder.add_joint_fixed(-1, link0)

    j1 = arm_model_builder.add_joint_revolute(
        link0,
        link1,
        parent_xform=wp.transform(p=(box_length / 2, 0.0, 0.0)),
        child_xform=wp.transform(p=(-box_length / 2, 0.0, 0.0)),
        armature=armature,
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_ke=0.0,
        target_kd=0.0,
    )

    j2 = arm_model_builder.add_joint_revolute(
        link1,
        link2,
        parent_xform=wp.transform(p=(box_length / 2, 0.0, 0.0)),
        child_xform=wp.transform(p=(-box_length / 2, 0.0, 0.0)),
        armature=armature,
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_ke=0.0,
        target_kd=0.0,
    )

    j3 = arm_model_builder.add_joint_revolute(
        link2,
        link3,
        parent_xform=wp.transform(p=(box_length / 2, 0.0, 0.0)),
        child_xform=wp.transform(p=(-box_length / 2, 0.0, 0.0)),
        armature=armature,
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_ke=0.0,
        target_kd=0.0,
    )

    arm_model_builder.add_articulation(joints=[j0, j1, j2, j3], label="arm")


# basic simulation function:
def simulate():
    global state_0, state_1, controller_state_0, controller_state_1

    for _ in range(sim_substeps):
        state_0.clear_forces()
        arm_model.collide(state_0, contacts)

        # Rebind live read ports so the PID sees the latest joint_q / joint_qd
        # regardless of how the substep swap left state_0.
        pid_input.joint_q = state_0.joint_q
        pid_input.joint_qd = state_0.joint_qd
        # step the controllers:
        controller.step(
            pid_input,
            pid_output,
            controller_state_0,
            controller_state_1,
            dt=sim_dt,
        )

        # step the world:
        solver.step(
            state_in=state_0,
            state_out=state_1,
            control=control,
            contacts=contacts,
            dt=sim_dt,
        )

        controller_state_0, controller_state_1 = controller_state_1, controller_state_0
        state_0, state_1 = state_1, state_0


if __name__ == "__main__":
    arm_model = arm_model_builder.finalize()

    arm_view = newton.selection.ArticulationView(model=arm_model, pattern="arm")

    # newton objects for simulation:
    state_0 = arm_model.state()
    state_1 = arm_model.state()
    control = arm_model.control()
    contacts = arm_model.contacts()

    # set some random joint target:
    control.joint_target_q = wp.array(
        [0.0, wp.pi / 2, wp.pi / 2], dtype=wp.float32, device=control.joint_target_q.device
    )

    # Read ports: joint_q / joint_qd from the simulation state, gains and
    # setpoint references freshly allocated. Write port: control.joint_f.
    pid_input = SimpleNamespace(
        joint_q=state_0.joint_q,
        joint_qd=state_0.joint_qd,
        setpoint=control.joint_target_q,
        setpoint_rate=control.joint_target_qd,
        kp=wp.full(3, value=ke),
        ki=wp.zeros(3),
        kd=wp.full(3, value=kd),
        integral_max=wp.full(3, value=10.0),
    )
    pid_output = SimpleNamespace(joint_f=control.joint_f)

    # create an external PD controller to use:
    pid_indices = wp.array(arm_model.joint_target_q_start[j1:-1], dtype=wp.uint32)
    control_law = ControlLawPID(
        label="arm_pid",
        measurement=("joint_q", pid_indices),
        measurement_rate=("joint_qd", pid_indices),
        setpoint=("setpoint", pid_indices),
        setpoint_rate=("setpoint_rate", pid_indices),
        kp=("kp", pid_indices),
        ki=("ki", pid_indices),
        kd=("kd", pid_indices),
        integral_max=("integral_max", pid_indices),
        output=("joint_f", pid_indices),
    )

    controller = Controller([control_law])
    controller_state_0 = controller.state()
    controller_state_1 = controller.state()

    ## Setting up the solver:
    solver = newton.solvers.SolverMuJoCo(
        model=arm_model,
    )

    fps = 100
    frame_dt = 1.0 / fps
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps

    # capture a graph:
    graph = None
    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as capture:
            simulate()
        graph = capture.graph

    # create our viewer:
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(arm_model)

    # run the simulation:
    num_frames = 500
    sim_time = 0.0
    for _ in range(num_frames):
        if graph:
            wp.capture_launch(graph)
        else:
            simulate()

        # log the current state to the viewer:
        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.log_contacts(
            contacts=contacts,
            state=state_0,
        )
        viewer.end_frame()

        sim_time += frame_dt
