import warp as wp
import newton

from contextlib import nullcontext

from newton.controllers import ControllerPID, ControlGroup

# BUILD UP THE ARM MODEL.
with nullcontext():
    # first, build up the actual model we will be using.
    # lets make it a 3DOF manipulator with no joint limits!
    arm_model_builder = newton.ModelBuilder()

    # make the links:
    link0 = arm_model_builder.add_link(
        mass=1.0)
    link1 = arm_model_builder.add_link(
        mass=1.0,
        inertia=wp.diag(wp.vec3(1., 1., 1.))
    )
    link2 = arm_model_builder.add_link(
        mass=1.0,
        inertia=wp.diag(wp.vec3(1., 1., 1.))
    )
    link3 = arm_model_builder.add_link(
        mass=1.0,
        inertia=wp.diag(wp.vec3(1., 1., 1.))
    )

    box_length = 1.0
    box_width = 0.05

    # add a tool frame.
    arm_model_builder.add_site(
        link3, 
        label="tool",
        xform=wp.transform(p=(box_length, 0., 0.))
    )

    # give all of the links the same simple box geometry:
    for link_i in [link1, link2, link3]:
        arm_model_builder.add_shape_box(
            link_i,
            hx=0.5*box_length,
            hy=0.5*box_width,
            hz=0.5*box_width,
        )

    # make the joints:
    armature = 0.05
    ke = 300.
    kd = 160.

    j0 = arm_model_builder.add_joint_fixed(-1, link0)

    j1 = arm_model_builder.add_joint_revolute(
        link0, 
        link1,
        parent_xform=wp.transform(p=(box_length/2, 0., 0.)),
        child_xform=wp.transform(p=(-box_length/2, 0., 0.)),
        armature=armature,
        axis=wp.vec3(0., 0., 1.),
        target_ke=ke,
        target_kd=kd,
    )

    j2 = arm_model_builder.add_joint_revolute(
        link1,
        link2,
        parent_xform=wp.transform(p=(box_length/2, 0., 0.)),
        child_xform=wp.transform(p=(-box_length/2, 0., 0.)),
        armature=armature,
        axis=wp.vec3(0., 0., 1.),
        target_ke=ke,
        target_kd=kd,
    )

    j3 = arm_model_builder.add_joint_revolute(
        link2, 
        link3,
        parent_xform=wp.transform(p=(box_length/2, 0., 0.)),
        child_xform=wp.transform(p=(-box_length/2, 0., 0.)),
        armature=armature,
        axis=wp.vec3(0., 0., 1.),
        target_ke=ke,
        target_kd=kd,
    )

    arm_model_builder.add_articulation(joints=[j0, j1, j2, j3], label="arm")

