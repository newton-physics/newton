"""Quick visual test for loading the Apptronik Apollo robot with MuJoCo solver."""

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        # Use the robot file directly (scene.xml uses <include> which isn't supported)
        mjcf_path = "/home/adenzler/git/mujoco_menagerie/apptronik_apollo/apptronik_apollo.xml"

        builder.add_mjcf(
            mjcf_path,
            xform=wp.transform(wp.vec3(0, 0, 1.0)),  # Raise robot above ground
            parse_sites=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )

        builder.add_ground_plane()

        # Set some joint stiffness/damping for PD control
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 200
            builder.joint_target_kd[i] = 10

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=200,
            nconmax=100,
            use_mujoco_contacts=True,  # Use MuJoCo's collision handling
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Evaluate forward kinematics
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pass


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example, args)
