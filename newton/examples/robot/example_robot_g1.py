# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Robot G1

Shows how to set up a simulation of a G1 robot articulation
from a USD stage using newton.ModelBuilder.add_usd().

Success criteria:
- G1 humanoids begin upright slightly above the ground, then drop onto their feet and remain
  supported; the displayed world count matches --world-count.
- Right-drag individual robots: robots in separate worlds do not collide with one another but
  interact with their ground plane.
- Without perturbations, robots remain supported and settle without persistent visible jitter.

Runs:
- `python -m newton.examples robot_g1 --world-count 3`
  - Three G1 robots appear.
- `python -m newton.examples robot_g1 --world-count 2`
  - Two G1 robots appear; use this scene for the shared viewer checklist.
- `python -m newton.examples robot_g1 --world-count 16`
"""

import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.examples.kamino.utils.ldl import add_ldl_arguments, configure_ldl


class Example:
    def __init__(self, viewer, args):
        """Build the G1 robot simulation with the requested solver options."""
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4 if args.solver == "kamino" else 6
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_type = args.solver
        if self.solver_type != "kamino":
            configure_ldl(None, args)

        self.viewer = viewer

        g1 = newton.ModelBuilder()
        if self.solver_type == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(g1)
        else:
            newton.solvers.SolverMuJoCo.register_custom_attributes(g1)
        g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        g1.default_shape_cfg.ke = 1.0e3
        g1.default_shape_cfg.kd = 2.0e2
        g1.default_shape_cfg.kf = 1.0e3
        g1.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_g1")

        g1.add_usd(
            str(asset_path / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
            xform=wp.transform(wp.vec3(0, 0, 0.2)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
            # Use the same Newton joint drives for both solvers.
            ignore_paths=[".*/Physics/.*_joint$"],
        )

        for i in range(6, g1.joint_dof_count):
            g1.joint_target_ke[i] = 500.0
            g1.joint_target_kd[i] = 10.0
            g1.joint_target_mode[i] = int(JointTargetMode.POSITION)

        # approximate meshes for faster collision detection
        g1.approximate_meshes("bounding_box")

        builder = newton.ModelBuilder()
        builder.replicate(g1, self.world_count)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 2.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()
        use_mujoco_contacts = self.solver_type == "mujoco" and args.use_mujoco_contacts
        if self.solver_type == "kamino":
            solver_config = newton.solvers.SolverKamino.Config.from_model(
                self.model, dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True
            )
            # Preserve foot support when nearly touching contacts cross zero separation.
            solver_config.dynamics.cull_speculative_contacts = False
            solver_config.dvi.max_alternating_iterations = 8
            solver_config.dvi.bilateral_solve_interval = 8
            solver_config.dvi.bilateral_solver_type = args.bilateral_solver_type or "LLTBRCM"
            configure_ldl(solver_config, args)
            solver_config.dvi.omega = 1.2
            solver_config.dvi.contact_warmstart_method = (
                "key_and_position_with_net_force_backup_and_tangential_net_force"
            )
            solver_config.dvi.use_schur_complement = True
            self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)
        else:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_cpu=False,
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=150,
                cone="elliptic",
                impratio=100,
                iterations=100,
                ls_iterations=50,
                use_mujoco_contacts=use_mujoco_contacts,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Evaluate forward kinematics for collision detection
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.use_mujoco_contacts = use_mujoco_contacts
        if use_mujoco_contacts:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model, rigid_contact_max=self.solver.get_max_contact_count(), soft_contact_max=0
            )
            self.contacts = self.collision_pipeline.contacts()
        else:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
        self.solver_observables = self.solver.observables(
            contacts=self.contacts, kinds={self.solver.ObservableKind.CONTACT_F}
        )

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        if not self.use_mujoco_contacts:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
                observables=self.solver_observables if substep == self.sim_substeps - 1 else None,
            )

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0, observables=self.solver_observables)
        self.viewer.end_frame()

    def test_final(self):
        """Verify one actuator per joint and a settled robot pose."""
        if self.solver_type == "mujoco":
            assert self.solver.mj_model.nu == self.model.joint_dof_count // self.world_count - 6, (
                "Each G1 joint should have exactly one MuJoCo actuator"
            )
            assert (self.solver.mjc_actuator_ctrl_source.numpy() == self.solver.CtrlSource.JOINT_TARGET).all(), (
                "G1 actuators should use Newton joint targets"
            )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        # Keep the comparison on the lambda's source line for Warp extraction.
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all body velocities are small",
            lambda q, qd: max(abs(qd)) < 0.015,
        )

    @staticmethod
    def create_parser():
        """Create simulation arguments, including optional Kamino LDL backend and ordering selection."""
        parser = newton.examples.create_parser()
        add_ldl_arguments(parser)
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_mujoco_contacts_arg(parser)
        parser.add_argument("--solver", choices=["mujoco", "kamino"], default="mujoco")
        parser.add_argument(
            "--bilateral-solver-type",
            choices=("LDL", "LLTB", "LLTBRCM"),
            default=None,
            help="DVI bilateral solver (default: LLTBRCM).",
        )
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
