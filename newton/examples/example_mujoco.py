# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example using MuJoCo solver
#
# This script allows us to choose between several predefined robots and
# provides a large range of customizable options.
# The simulation runs with MuJoCo solver.
#
# Future improvements:
# - Add options to run with a pre-trained policy
# - Add the Anymal world
# - Fix the use-mujoco-cpu option (currently crashes)
###########################################################################


import time

import numpy as np
import warp as wp

wp.config.enable_backward = False

import newton
import newton.examples
import newton.utils
from newton.utils import EventTracer

ROBOT_CONFIGS = {
    "humanoid": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 80,
        "nconmax": 25,
        "ls_parallel": True,
        "cone": "pyramidal",
    },
    "g1": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 210,
        "nconmax": 35,
        "ls_parallel": True,
        "cone": "pyramidal",
    },
    "h1": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 65,
        "nconmax": 15,
        "ls_parallel": True,
        "cone": "pyramidal",
    },
    "cartpole": {  # TODO: use the Lab version of cartpole and revert param value
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 24,  # 5
        "nconmax": 6,  # 5
        "ls_parallel": False,
        "cone": "pyramidal",
    },
    "ant": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 38,
        "nconmax": 15,
        "ls_parallel": True,
        "cone": "pyramidal",
    },
    "quadruped": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 75,
        "nconmax": 50,
        "ls_parallel": True,
        "cone": "pyramidal",
    },
    "allegro": {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 60,
        "nconmax": 35,
        "ls_parallel": True,
        "cone": "elliptic",
    },
    "kitchen": {
        "setup_builder": lambda x: _setup_kitchen(x),
        "njmax": 3800,
        "nconmax": 900,
    },
}


def _setup_humanoid(articulation_builder):
    articulation_builder.add_mjcf(
        newton.examples.get_asset("nv_humanoid.xml"),
        ignore_names=["floor", "ground"],
        up_axis="Z",
    )

    # Setting root pose
    root_dofs = 7
    articulation_builder.joint_q[:3] = [0.0, 0.0, 1.5]

    return root_dofs


def _setup_g1(articulation_builder):
    articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    articulation_builder.default_shape_cfg.ke = 5.0e4
    articulation_builder.default_shape_cfg.kd = 5.0e2
    articulation_builder.default_shape_cfg.kf = 1.0e3
    articulation_builder.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("unitree_g1")

    articulation_builder.add_usd(
        str(asset_path / "usd" / "g1_isaac.usd"),
        xform=wp.transform(wp.vec3(0, 0, 0.8)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )

    for i in range(6, articulation_builder.joint_dof_count):
        articulation_builder.joint_target_ke[i] = 1000.0
        articulation_builder.joint_target_kd[i] = 5.0

    # approximate meshes for faster collision detection
    articulation_builder.approximate_meshes("bounding_box")

    root_dofs = 7

    return root_dofs


def _setup_h1(articulation_builder):
    articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    articulation_builder.default_shape_cfg.ke = 5.0e4
    articulation_builder.default_shape_cfg.kd = 5.0e2
    articulation_builder.default_shape_cfg.kf = 1.0e3
    articulation_builder.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("unitree_h1")
    asset_file = str(asset_path / "usd" / "h1_minimal.usda")
    articulation_builder.add_usd(
        asset_file,
        ignore_paths=["/GroundPlane"],
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        load_non_physics_prims=True,
        hide_collision_shapes=True,
    )
    # approximate meshes for faster collision detection
    articulation_builder.approximate_meshes("bounding_box")

    for i in range(len(articulation_builder.joint_dof_mode)):
        articulation_builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
        articulation_builder.joint_target_ke[i] = 150
        articulation_builder.joint_target_kd[i] = 5

    root_dofs = 7

    return root_dofs


def _setup_cartpole(articulation_builder):
    articulation_builder.default_shape_cfg.density = 100.0
    articulation_builder.default_joint_cfg.armature = 0.1
    articulation_builder.default_body_armature = 0.1

    articulation_builder.add_usd(
        newton.examples.get_asset("cartpole.usda"),
        enable_self_collisions=False,
        collapse_fixed_joints=True,
    )
    # set initial joint positions
    articulation_builder.joint_q[-3:] = [0.0, 0.3, 0.0]

    # Setting root pose
    root_dofs = 1
    articulation_builder.joint_q[:3] = [0.0, 0.3, 0.0]

    return root_dofs


def _setup_ant(articulation_builder):
    articulation_builder.add_usd(
        newton.examples.get_asset("ant.usda"),
        collapse_fixed_joints=True,
    )

    # Setting root pose
    root_dofs = 7
    articulation_builder.joint_q[:3] = [0.0, 0.0, 1.5]

    return root_dofs


def _setup_quadruped(articulation_builder):
    articulation_builder.default_body_armature = 0.01
    articulation_builder.default_joint_cfg.armature = 0.01
    articulation_builder.default_shape_cfg.ke = 1.0e4
    articulation_builder.default_shape_cfg.kd = 1.0e2
    articulation_builder.default_shape_cfg.kf = 1.0e2
    articulation_builder.default_shape_cfg.mu = 1.0
    articulation_builder.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform([0.0, 0.0, 0.7], wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
    )
    root_dofs = 7

    return root_dofs


def _setup_allegro(articulation_builder):
    asset_path = newton.utils.download_asset("wonik_allegro")
    asset_file = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
    articulation_builder.add_usd(
        asset_file,
        xform=wp.transform(wp.vec3(0, 0, 0.5)),
        enable_self_collisions=True,
        ignore_paths=[".*Dummy", ".*CollisionPlane", ".*goal", ".*DexCube/visuals"],
        load_non_physics_prims=True,
    )

    # set joint targets and joint drive gains
    for i in range(len(articulation_builder.joint_dof_mode)):
        articulation_builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
        articulation_builder.joint_target_ke[i] = 150
        articulation_builder.joint_target_kd[i] = 5
        articulation_builder.joint_target[i] = 0.0
    root_dofs = 1

    return root_dofs


def _setup_kitchen(articulation_builder):
    asset_path = newton.utils.download_asset("kitchen")
    asset_file = str(asset_path / "mjcf" / "kitchen.xml")
    articulation_builder.add_mjcf(
        asset_file,
        collapse_fixed_joints=True,
    )

    # Change pose of the robot to minimize overlap
    articulation_builder.joint_q[:2] = [1.5, -1.5]


class Example:
    def __init__(
        self,
        robot="humanoid",
        environment="None",
        stage_path=None,
        num_worlds=1,
        use_cuda_graph=True,
        use_mujoco_cpu=False,
        randomize=False,
        headless=False,
        actuation="None",
        solver=None,
        integrator=None,
        solver_iteration=None,
        ls_iteration=None,
        njmax=None,
        nconmax=None,
        builder=None,
        ls_parallel=None,
        cone=None,
    ):
        fps = 600
        self.sim_time = 0.0
        self.benchmark_time = 0.0
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 10
        self.contacts = None
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_worlds = num_worlds
        self.use_cuda_graph = use_cuda_graph
        self.use_mujoco_cpu = use_mujoco_cpu
        self.actuation = actuation

        # set numpy random seed
        self.seed = 123
        self.rng = np.random.default_rng(self.seed)

        if not stage_path:
            stage_path = "example_" + robot + ".usd"

        if builder is None:
            builder = Example.create_model_builder(robot, num_worlds, environment, randomize, self.seed)

        # finalize model
        self.model = builder.finalize()

        self.solver = Example.create_solver(
            self.model,
            robot,
            use_mujoco_cpu=use_mujoco_cpu,
            environment=environment,
            solver=solver,
            integrator=integrator,
            solver_iteration=solver_iteration,
            ls_iteration=ls_iteration,
            njmax=njmax,
            nconmax=nconmax,
            ls_parallel=ls_parallel,
            cone=cone,
        )

        if stage_path and not headless:
            self.renderer = newton.viewer.ViewerGL()
            self.renderer.set_model(self.model)
            self.renderer.set_world_offsets((4.0, 4.0, 0.0))
        else:
            self.renderer = None

        self.control = self.model.control()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.graph = None
        if self.use_cuda_graph:
            # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
            cuda_graph_comp = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
            if not cuda_graph_comp:
                print("Cannot use graph capture. Graph capture is disabled.")
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.actuation == "random":
            joint_target = wp.array(self.rng.uniform(-1.0, 1.0, size=self.model.joint_dof_count), dtype=float)
            wp.copy(self.control.joint_target, joint_target)

        wp.synchronize_device()
        start_time = time.time()
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        wp.synchronize_device()
        end_time = time.time()

        self.benchmark_time += end_time - start_time
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.sim_time)
        self.renderer.log_state(self.state_0)
        self.renderer.end_frame()

    @staticmethod
    def create_model_builder(robot, num_worlds, environment="None", randomize=False, seed=123) -> newton.ModelBuilder:
        rng = np.random.default_rng(seed)

        articulation_builder = newton.ModelBuilder()
        if robot == "humanoid":
            root_dofs = _setup_humanoid(articulation_builder)
        elif robot == "g1":
            root_dofs = _setup_g1(articulation_builder)
        elif robot == "h1":
            root_dofs = _setup_h1(articulation_builder)
        elif robot == "cartpole":
            root_dofs = _setup_cartpole(articulation_builder)
        elif robot == "ant":
            root_dofs = _setup_ant(articulation_builder)
        elif robot == "quadruped":
            root_dofs = _setup_quadruped(articulation_builder)
        elif robot == "allegro":
            root_dofs = _setup_allegro(articulation_builder)
        else:
            raise ValueError(f"Name of the provided robot not recognized: {robot}")

        custom_setup_fn = ROBOT_CONFIGS.get(environment, {}).get("setup_builder", None)
        if custom_setup_fn is not None:
            custom_setup_fn(articulation_builder)

        builder = newton.ModelBuilder()
        builder.replicate(articulation_builder, num_worlds)
        if randomize:
            njoint = len(articulation_builder.joint_q)
            for i in range(num_worlds):
                istart = i * njoint
                builder.joint_q[istart + root_dofs : istart + njoint] = rng.uniform(
                    -1.0, 1.0, size=(njoint - root_dofs)
                ).tolist()
        builder.add_ground_plane()
        return builder

    @staticmethod
    def create_solver(
        model,
        robot,
        *,
        use_mujoco_cpu=False,
        environment="None",
        solver=None,
        integrator=None,
        solver_iteration=None,
        ls_iteration=None,
        njmax=None,
        nconmax=None,
        ls_parallel=None,
        cone=None,
    ):
        solver_iteration = solver_iteration if solver_iteration is not None else 100
        ls_iteration = ls_iteration if ls_iteration is not None else 50
        solver = solver if solver is not None else ROBOT_CONFIGS[robot]["solver"]
        integrator = integrator if integrator is not None else ROBOT_CONFIGS[robot]["integrator"]
        njmax = njmax if njmax is not None else ROBOT_CONFIGS[robot]["njmax"]
        nconmax = nconmax if nconmax is not None else ROBOT_CONFIGS[robot]["nconmax"]
        ls_parallel = ls_parallel if ls_parallel is not None else ROBOT_CONFIGS[robot]["ls_parallel"]
        cone = cone if cone is not None else ROBOT_CONFIGS[robot]["cone"]

        njmax += ROBOT_CONFIGS.get(environment, {}).get("njmax", 0)
        nconmax += ROBOT_CONFIGS.get(environment, {}).get("nconmax", 0)

        return newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=use_mujoco_cpu,
            solver=solver,
            integrator=integrator,
            iterations=solver_iteration,
            ls_iterations=ls_iteration,
            njmax=njmax,
            ncon_per_world=nconmax,
            ls_parallel=ls_parallel,
            cone=cone,
        )


def print_trace(trace, indent, steps):
    if indent == 0:
        print("================= Profiling =================")
    for k, v in trace.items():
        times, sub_trace = v
        print("  " * indent + f"{k}: {times / steps:.4f}")
        print_trace(sub_trace, indent + 1, steps)
    if indent == 0:
        step_time = trace["step"][0]
        mujoco_warp_step_time = trace["step"][1]["mujoco_warp_step"][0]
        overhead = 100.0 * (step_time - mujoco_warp_step_time) / step_time
        print("---------------------------------------------")
        print(f"Newton overhead:\t{overhead:.2f} %")
        print("=============================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--robot", type=str, default="humanoid", help="Name of the robot to simulate.")
    parser.add_argument("--env", type=str, default="None", help="Name of the environment where the robot is located.")
    parser.add_argument(
        "--event-trace", default=False, action=argparse.BooleanOptionalAction, help="Print profiling information."
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=12000, help="Total number of frames.")
    parser.add_argument("--num-worlds", type=int, default=1, help="Total number of simulated worlds.")
    parser.add_argument("--use-cuda-graph", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--use-mujoco-cpu",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use Mujoco-C CPU (Not yet supported).",
    )
    parser.add_argument(
        "--headless", default=False, action=argparse.BooleanOptionalAction, help="Run the simulation in headless mode."
    )

    parser.add_argument(
        "--random-init", default=False, action=argparse.BooleanOptionalAction, help="Randomize initial pose."
    )
    parser.add_argument(
        "--actuation",
        type=str,
        default="None",
        choices=["None", "random"],
        help="Type of action to apply at each step.",
    )

    parser.add_argument(
        "--solver", type=str, default=None, choices=["cg", "newton"], help="Mujoco model constraint solver used."
    )
    parser.add_argument(
        "--integrator", type=str, default=None, choices=["euler", "rk4", "implicit"], help="Mujoco integrator used."
    )
    parser.add_argument("--solver-iteration", type=int, default=None, help="Number of solver iterations.")
    parser.add_argument("--ls-iteration", type=int, default=None, help="Number of linesearch iterations.")
    parser.add_argument("--njmax", type=int, default=None, help="Maximum number of constraints per world.")
    parser.add_argument("--nconmax", type=int, default=None, help="Maximum number of collision per world.")
    parser.add_argument(
        "--ls-parallel", default=None, action=argparse.BooleanOptionalAction, help="Use parallel line search."
    )
    parser.add_argument("--cone", type=str, default=None, choices=["pyramidal", "elliptic"], help="Friction cone type.")

    args = parser.parse_known_args()[0]

    if args.use_mujoco_cpu:
        args.use_mujoco_cpu = False
        print("The option ``use-mujoco-cpu`` is not yet supported. Disabling it.")

    trace = {}
    with EventTracer(enabled=args.event_trace) as tracer:
        with wp.ScopedDevice(args.device):
            example = Example(
                robot=args.robot,
                environment=args.env,
                stage_path=args.stage_path,
                num_worlds=args.num_worlds,
                use_cuda_graph=args.use_cuda_graph,
                use_mujoco_cpu=args.use_mujoco_cpu,
                randomize=args.random_init,
                headless=args.headless,
                actuation=args.actuation,
                solver=args.solver,
                integrator=args.integrator,
                solver_iteration=args.solver_iteration,
                ls_iteration=args.ls_iteration,
                njmax=args.njmax,
                nconmax=args.nconmax,
                ls_parallel=args.ls_parallel,
                cone=args.cone,
            )

            # Print simulation configuration summary
            LABEL_WIDTH = 25
            TOTAL_WIDTH = 45
            title = " Simulation Configuration "
            print(f"\n{title.center(TOTAL_WIDTH, '=')}")
            print(f"{'Simulation Steps':<{LABEL_WIDTH}}: {args.num_frames * example.sim_substeps}")
            print(f"{'World Count':<{LABEL_WIDTH}}: {args.num_worlds}")
            print(f"{'Robot Type':<{LABEL_WIDTH}}: {args.robot}")
            print(f"{'Timestep (dt)':<{LABEL_WIDTH}}: {example.sim_dt:.6f}s")
            print(f"{'Randomize Initial Pose':<{LABEL_WIDTH}}: {args.random_init!s}")
            print("-" * TOTAL_WIDTH)

            # Map MuJoCo solver enum back to string
            solver_value = example.solver.mj_model.opt.solver
            solver_map = {0: "PGS", 1: "CG", 2: "Newton"}  # mjSOL_PGS = 0, mjSOL_CG = 1, mjSOL_NEWTON = 2
            actual_solver = solver_map.get(solver_value, f"unknown({solver_value})")
            # Map MuJoCo integrator enum back to string
            integrator_map = {
                0: "Euler",
                1: "RK4",
                2: "Implicit",
                3: "Implicitfast",
            }  # mjINT_EULER = 0, mjINT_RK4 = 1, mjINT_IMPLICIT = 2, mjINT_IMPLICITFAST = 3
            actual_integrator = integrator_map.get(example.solver.mj_model.opt.integrator, "unknown")
            # Map MuJoCo cone enum back to string
            cone_value = example.solver.mj_model.opt.cone
            cone_map = {0: "pyramidal", 1: "elliptic"}  # mjCONE_PYRAMIDAL = 0, mjCONE_ELLIPTIC = 1
            actual_cone = cone_map.get(cone_value, f"unknown({cone_value})")
            # Get actual max constraints and contacts from MuJoCo Warp data
            actual_njmax = example.solver.mjw_data.njmax
            actual_nconmax = example.solver.mjw_data.nconmax
            print(f"{'Solver':<{LABEL_WIDTH}}: {actual_solver}")
            print(f"{'Integrator':<{LABEL_WIDTH}}: {actual_integrator}")
            # print(f"{'Parallel Line Search':<{LABEL_WIDTH}}: {example.solver.mj_model.opt.ls_parallel}")
            print(f"{'Cone':<{LABEL_WIDTH}}: {actual_cone}")
            print(f"{'Solver Iterations':<{LABEL_WIDTH}}: {example.solver.mj_model.opt.iterations}")
            print(f"{'Line Search Iterations':<{LABEL_WIDTH}}: {example.solver.mj_model.opt.ls_iterations}")
            print(f"{'Max Constraints / world':<{LABEL_WIDTH}}: {actual_njmax}")
            print(f"{'Max Contacts / world':<{LABEL_WIDTH}}: {actual_nconmax}")
            print(f"{'Joint DOFs':<{LABEL_WIDTH}}: {example.model.joint_dof_count}")
            print(f"{'Body Count':<{LABEL_WIDTH}}: {example.model.body_count}")
            print("-" * TOTAL_WIDTH)

            print(f"{'Execution Device':<{LABEL_WIDTH}}: {wp.get_device()}")
            print(f"{'Use CUDA Graph':<{LABEL_WIDTH}}: {example.use_cuda_graph!s}")
            print("=" * TOTAL_WIDTH + "\n")

            for _ in range(args.num_frames):
                example.step()
                example.render()
                if args.event_trace:
                    trace = EventTracer.add_trace(trace, tracer.trace())

    if args.event_trace:
        print_trace(trace, 0, args.num_frames * example.sim_substeps)
