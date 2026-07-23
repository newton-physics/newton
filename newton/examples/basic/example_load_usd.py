# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Load USD
#
# Loads an arbitrary USD file (.usd, .usda, .usdc, or .usdz) from the local
# filesystem into a Newton model and simulates it, so you can inspect a scene
# without writing a bespoke script. Exposes the ModelBuilder.add_usd options
# people most commonly need to set, plus a choice of solver.
#
# Command: python -m newton.examples load_usd /path/to/scene.usdz
#
###########################################################################

import os

import warp as wp

import newton
import newton.examples
from newton import JointTargetMode

_SOLVERS = {
    "xpbd": newton.solvers.SolverXPBD,
    "mujoco": newton.solvers.SolverMuJoCo,
    "featherstone": newton.solvers.SolverFeatherstone,
    "semi_implicit": newton.solvers.SolverSemiImplicit,
}

_UP_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


class Example:
    def __init__(self, viewer, args):
        self.args = args
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.current_path = None
        self.status = ""

        # A path is required for CLI usage (enforced in __main__), but the example
        # browser instantiates examples with default args (no path). Fall back to a
        # bundled asset so the browser entry is viewable instead of failing to load.
        path = args.path or args.path_flag
        if not path:
            path = newton.examples.get_asset("cartpole.usda")
            print(f"load_usd: no path given; loading bundled demo asset '{path}'.")

        self._load(path)

    def _load(self, path):
        """Build (or rebuild) the model, solver, and state from the USD at ``path``."""
        args = self.args
        up_axis = newton.Axis[args.up_axis]

        articulation = newton.ModelBuilder(up_axis=up_axis)
        if args.solver == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(articulation)

        articulation.add_usd(
            path,
            floating=args.floating,
            collapse_fixed_joints=args.collapse_fixed_joints,
            enable_self_collisions=args.self_collisions,
            hide_collision_shapes=args.hide_collision_shapes,
            load_visual_shapes=args.visual_shapes,
            joint_drive_gains_scaling=args.joint_drive_gains_scaling,
            verbose=args.verbose,
        )

        # Lift the free-floating base along the up axis so the scene starts above
        # the ground plane. A free joint stores [translation (3), rotation (4)].
        if args.height and len(articulation.joint_q) >= 7:
            articulation.joint_q[_UP_AXIS_INDEX[args.up_axis]] += args.height

        # Hold the imported configuration with a PD drive so articulated robots
        # stand instead of collapsing under gravity.
        if args.hold_pose:
            for i in range(articulation.joint_dof_count):
                articulation.joint_target_ke[i] = args.kp
                articulation.joint_target_kd[i] = args.kd
                articulation.joint_target_mode[i] = int(JointTargetMode.POSITION)

        builder = newton.ModelBuilder(up_axis=up_axis)
        for _ in range(args.world_count):
            builder.add_world(articulation)
        if args.ground:
            builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = _SOLVERS[args.solver](self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Populate maximal-coordinate state from the imported joint configuration.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.sim_time = 0.0
        self.current_path = path
        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
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
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def reset_sim(self):
        """Reset the simulation to the imported configuration without reloading the file."""
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.sim_time = 0.0
        self.status = "Reset simulation"

    def _reload(self, path):
        """Rebuild the scene from ``path``, keeping the current scene on failure."""
        try:
            self._load(path)
            self.status = f"Loaded {os.path.basename(path)}"
        except Exception as exc:
            self.status = f"Failed to load {os.path.basename(path)}: {exc}"
            print(f"load_usd: failed to load '{path}': {exc}")

    def gui(self, imgui):
        # File dialogs are asynchronous: the picker returns immediately and the
        # chosen path arrives on a later frame via consume_file_dialog_result().
        ui = getattr(self.viewer, "ui", None)
        if ui is not None:
            picked = ui.consume_file_dialog_result()
            if picked:
                self._reload(picked)

        imgui.text("Current file:")
        imgui.text(os.path.basename(self.current_path) if self.current_path else "(none)")
        if imgui.button("Load USD...") and ui is not None:
            ui.open_load_file_dialog(title="Select a USD file")
        imgui.same_line()
        if imgui.button("Reset simulation"):
            self.reset_sim()

        if self.status:
            imgui.separator()
            imgui.text(self.status)

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body positions remain bounded",
            lambda q, qd: (abs(q[0]) < 1.0e6) and (abs(q[1]) < 1.0e6) and (abs(q[2]) < 1.0e6),
        )

    @staticmethod
    def create_parser():
        import argparse  # noqa: PLC0415

        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)

        parser.add_argument(
            "path",
            nargs="?",
            default=None,
            metavar="PATH",
            help="Path to the USD file to load (.usd, .usda, .usdc, or .usdz).",
        )
        parser.add_argument(
            "--path",
            dest="path_flag",
            default=None,
            metavar="PATH",
            help="Alternative to the positional PATH argument.",
        )
        parser.add_argument(
            "--solver",
            type=str,
            default="xpbd",
            choices=list(_SOLVERS),
            help="Solver used to simulate the loaded scene. Use 'mujoco' for the "
            "highest-fidelity articulated-robot dynamics.",
        )
        parser.add_argument(
            "--up-axis",
            type=str,
            default="Z",
            choices=list(_UP_AXIS_INDEX),
            help="Up axis for the model.",
        )
        parser.add_argument(
            "--floating",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Force the root body's base joint to FREE (--floating) or FIXED "
            "(--no-floating). Defaults to the USD format default.",
        )
        parser.add_argument(
            "--collapse-fixed-joints",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Merge bodies connected by fixed joints.",
        )
        parser.add_argument(
            "--self-collisions",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable self-collisions within each articulation. Off by default "
            "since it is expensive and can destabilize articulated robots.",
        )
        parser.add_argument(
            "--hide-collision-shapes",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Hide collision shapes in the viewer.",
        )
        parser.add_argument(
            "--visual-shapes",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Load visual (render) shapes from the USD.",
        )
        parser.add_argument(
            "--joint-drive-gains-scaling",
            type=float,
            default=1.0,
            help="Scale factor applied to joint drive gains parsed from the USD.",
        )
        parser.add_argument(
            "--ground",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Add a ground plane to the scene.",
        )
        parser.add_argument(
            "--height",
            type=float,
            default=0.0,
            help="Offset [m] added to the free-floating base along the up axis.",
        )
        parser.add_argument(
            "--hold-pose",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Hold the imported joint configuration with a PD drive.",
        )
        parser.add_argument(
            "--kp",
            type=float,
            default=100.0,
            help="Proportional gain for the pose-holding PD drive.",
        )
        parser.add_argument(
            "--kd",
            type=float,
            default=10.0,
            help="Derivative gain for the pose-holding PD drive.",
        )
        parser.add_argument(
            "--verbose",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Print details about the parsed USD scene.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    # Require a path for CLI usage (validated before the viewer is created). The
    # example browser bypasses this and falls back to a bundled demo asset.
    prelim, _ = parser.parse_known_args()
    if not (prelim.path or prelim.path_flag):
        parser.error(
            "a path to a USD file is required, e.g.:\n  python -m newton.examples load_usd /path/to/scene.usdz"
        )

    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
