# THIS IS NOT A REAL EXAMPLE!!!!!
# TEMPORARY FILE USED FOR INITIAL TESTS,
# WILL BE DELETED.

from contextlib import nullcontext
from types import SimpleNamespace

import warp as wp

import newton
from newton.controllers import ControlLawDifferentialIK, Controller

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
        target_ke=ke,
        target_kd=kd,
    )

    j2 = arm_model_builder.add_joint_revolute(
        link1,
        link2,
        parent_xform=wp.transform(p=(box_length / 2, 0.0, 0.0)),
        child_xform=wp.transform(p=(-box_length / 2, 0.0, 0.0)),
        armature=armature,
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_ke=ke,
        target_kd=kd,
    )

    j3 = arm_model_builder.add_joint_revolute(
        link2,
        link3,
        parent_xform=wp.transform(p=(box_length / 2, 0.0, 0.0)),
        child_xform=wp.transform(p=(-box_length / 2, 0.0, 0.0)),
        armature=armature,
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_ke=ke,
        target_kd=kd,
    )

    arm_model_builder.add_articulation(joints=[j0, j1, j2, j3], label="arm")


# basic simulation function:
def simulate():
    global state_0, state_1, controller_state_0, controller_state_1

    for _ in range(sim_substeps):
        state_0.clear_forces()
        arm_model.collide(state_0, contacts)

        # Rebind live read ports so the controller sees the latest
        # joint_q / joint_qd regardless of substep-swap parity.
        diff_ik_input.joint_q = state_0.joint_q
        diff_ik_input.joint_qd = state_0.joint_qd
        # step the controller:
        controller.step(
            diff_ik_input,
            diff_ik_output,
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

    # set the initial state of the robot such that it is not in a singularity:
    state_0.joint_q = wp.array([0.1, 0.1, 0.1], dtype=wp.float32)

    # Read ports: joint_q / joint_qd from the simulation state, plus targets
    # and gain/damping. Write ports: control.joint_target_q / joint_target_qd.
    diff_ik_input = SimpleNamespace(
        joint_q=state_0.joint_q,
        joint_qd=state_0.joint_qd,
        target_pos=wp.array([wp.vec3f(2.0, 0.0, 0.0)], dtype=wp.vec3f),
        target_quat=wp.array([wp.quat_identity()], dtype=wp.quatf),
        damping=wp.array([1e-8], dtype=wp.float32),
        gain=wp.array([300.0], dtype=wp.float32),
    )
    diff_ik_output = SimpleNamespace(
        joint_target_q=control.joint_target_q,
        joint_target_qd=control.joint_target_qd,
    )

    # create an external differential IK to control the robot to the desired pose.
    # TODO: IS THERE A MORE ERGONOMIC WAY TO GRAB THESE INDICES?
    dof_indices = wp.array(arm_model.joint_target_q_start[j1:-1], dtype=wp.uint32)
    robot_indices = wp.array([0], dtype=wp.uint32)
    control_law = ControlLawDifferentialIK(
        label="arm_ik",
        model_builder=arm_model_builder,
        site="tool",
        measurement=("joint_q", dof_indices),
        measurement_rate=("joint_qd", dof_indices),
        target_pos=("target_pos", robot_indices),
        target_quat=("target_quat", robot_indices),
        damping=("damping", robot_indices),
        gain=("gain", robot_indices),
        output_q=("joint_target_q", dof_indices),
        output_qd=("joint_target_qd", dof_indices),
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
    num_frames = 10000
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
