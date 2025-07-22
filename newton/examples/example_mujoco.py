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
# - Add the Ant environment
# - Add the Anymal environment
# - Fix the use_mujoco option (currently crash)
###########################################################################


import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

wp.config.enable_backward = False


ROBOT_CONFIGS = {
    "humanoid": {
        "solver": "newton",
        "integrator": "euler",
        "njmax": 100,
        "nconmax": 50,
    },
    "g1": {
        "solver": "newton",
        "integrator": "euler",
        "njmax": 400,
        "nconmax": 150,
    },
    "h1": {
        "solver": "newton",
        "integrator": "euler",
        "njmax": 400,
        "nconmax": 150,
    },
    "cartpole": {
        "solver": "newton",
        "integrator": "euler",
        "njmax": 50,
        "nconmax": 50,
    },
    "quadruped": {
        "solver": "newton",
        "integrator": "euler",
        "njmax": 75,
        "nconmax": 50,
    },
}


def _setup_humanoid(articulation_builder):
    newton.utils.parse_mjcf(
        newton.examples.get_asset("nv_humanoid.xml"),
        articulation_builder,
        ignore_names=["floor", "ground"],
        up_axis="Z",
    )

    # Setting root pose
    root_dofs = 7
    articulation_builder.joint_q[:3] = [0.0, 0.0, 1.5]

    return root_dofs


def _setup_g1(articulation_builder):
    asset_path = newton.utils.download_asset("g1_description")

    newton.utils.parse_mjcf(
        str(asset_path / "g1_29dof_with_hand_rev_1_0.xml"),
        articulation_builder,
        collapse_fixed_joints=True,
        up_axis="Z",
        enable_self_collisions=False,
    )
    simplified_meshes = {}
    try:
        import tqdm  # noqa: PLC0415

        meshes = tqdm.tqdm(articulation_builder.shape_geo_src, desc="Simplifying meshes")
    except ImportError:
        meshes = articulation_builder.shape_geo_src
    for i, m in enumerate(meshes):
        if m is None:
            continue
        hash_m = hash(m)
        if hash_m in simplified_meshes:
            articulation_builder.shape_geo_src[i] = simplified_meshes[hash_m]
        else:
            simplified = newton.geometry.utils.remesh_mesh(
                m, visualize=False, method="convex_hull", recompute_inertia=False
            )
            articulation_builder.shape_geo_src[i] = simplified
            simplified_meshes[hash_m] = simplified
    root_dofs = 7

    return root_dofs


def _setup_h1(articulation_builder):
    articulation_builder.default_shape_cfg.density = 100.0
    articulation_builder.default_joint_cfg.armature = 0.1
    articulation_builder.default_body_armature = 0.1

    asset_path = newton.utils.download_asset("h1_description")
    newton.utils.parse_mjcf(
        str(asset_path / "mjcf" / "h1_with_hand.xml"),
        articulation_builder,
        collapse_fixed_joints=True,
        up_axis="Z",
        enable_self_collisions=False,
    )
    root_dofs = 7

    return root_dofs


def _setup_cartpole(articulation_builder):
    articulation_builder.default_shape_cfg.density = 100.0
    articulation_builder.default_joint_cfg.armature = 0.1
    articulation_builder.default_body_armature = 0.1

    newton.utils.parse_urdf(
        newton.examples.get_asset("cartpole.urdf"),
        articulation_builder,
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
    )

    # Setting root pose
    root_dofs = 3
    articulation_builder.joint_q[:3] = [0.0, 0.3, 0.0]

    return root_dofs


def _setup_quadruped(articulation_builder):
    articulation_builder.default_body_armature = 0.01
    articulation_builder.default_joint_cfg.armature = 0.01
    articulation_builder.default_shape_cfg.ke = 1.0e4
    articulation_builder.default_shape_cfg.kd = 1.0e2
    articulation_builder.default_shape_cfg.kf = 1.0e2
    articulation_builder.default_shape_cfg.mu = 1.0
    newton.utils.parse_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        articulation_builder,
        xform=wp.transform([0.0, 0.0, 0.7], wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
    )
    root_dofs = 7

    return root_dofs


class Example:
    def __init__(
        self,
        robot="humanoid",
        stage_path=None,
        num_envs=1,
        use_cuda_graph=True,
        use_mujoco=False,
        randomize=False,
        headless=False,
        actuation="None",
        solver=None,
        integrator=None,
        solver_iteration=None,
        ls_iteration=None,
        njmax=None,
        nconmax=None,
    ):
        fps = 600
        self.sim_time = 0.0
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 10
        self.contacts = None
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_envs = num_envs
        self.use_cuda_graph = use_cuda_graph
        self.use_mujoco = use_mujoco
        self.actuation = actuation
        solver_iteration = solver_iteration if solver_iteration is not None else 100
        ls_iteration = ls_iteration if ls_iteration is not None else 50

        # set numpy random seed
        self.seed = 123
        self.rng = np.random.default_rng(self.seed)

        if not stage_path:
            stage_path = "example_" + robot + ".usd"

        articulation_builder = newton.ModelBuilder()
        if robot == "humanoid":
            root_dofs = _setup_humanoid(articulation_builder)
        elif robot == "g1":
            root_dofs = _setup_g1(articulation_builder)
        elif robot == "h1":
            root_dofs = _setup_h1(articulation_builder)
        elif robot == "cartpole":
            root_dofs = _setup_cartpole(articulation_builder)
        elif robot == "quadruped":
            root_dofs = _setup_quadruped(articulation_builder)
        else:
            raise ValueError(f"Name of the provided robot not recognized: {robot}")

        builder = newton.ModelBuilder()
        offsets = newton.examples.compute_env_offsets(self.num_envs)
        for i in range(self.num_envs):
            if randomize:
                articulation_builder.joint_q[root_dofs:] = self.rng.uniform(
                    -1.0, 1.0, size=(len(articulation_builder.joint_q) - root_dofs,)
                ).tolist()
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        solver = solver if solver is not None else ROBOT_CONFIGS[robot]["solver"]
        integrator = integrator if integrator is not None else ROBOT_CONFIGS[robot]["integrator"]
        njmax = njmax if njmax is not None else ROBOT_CONFIGS[robot]["njmax"]
        nconmax = nconmax if nconmax is not None else ROBOT_CONFIGS[robot]["nconmax"]
        self.solver = newton.solvers.MuJoCoSolver(
            self.model,
            use_mujoco=use_mujoco,
            solver=solver,
            integrator=integrator,
            iterations=solver_iteration,
            ls_iterations=ls_iteration,
            nefc_per_env=njmax,
            ncon_per_env=nconmax,
        )

        if stage_path and not headless:
            self.renderer = newton.utils.SimRendererOpenGL(self.model, stage_path)
        else:
            self.renderer = None

        self.control = self.model.control()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        newton.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

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

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--robot", type=str, default="humanoid", help="Name of the robot to simulate.")
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=12000, help="Total number of frames.")
    parser.add_argument("--num-envs", type=int, default=1, help="Total number of simulated environments.")
    parser.add_argument("--use-cuda-graph", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--use-mujoco", default=False, action=argparse.BooleanOptionalAction, help="Use Mujoco C (Not yet supported)."
    )
    parser.add_argument(
        "--headless", default=False, action=argparse.BooleanOptionalAction, help="Run the simulation in headless mode."
    )
    parser.add_argument(
        "--show-mujoco-viewer",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Toggle MuJoCo viewer next to Newton renderer when MuJoCoSolver is active.",
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
    parser.add_argument("--njmax", type=int, default=None, help="Maximum number of constraints per environment.")
    parser.add_argument("--nconmax", type=int, default=None, help="Maximum number of collision per environment.")

    args = parser.parse_known_args()[0]

    if args.use_mujoco:
        args.use_mujoco = False
        print("The option ``use_mujoco`` is not yet supported. Disabling it.")

    with wp.ScopedDevice(args.device):
        example = Example(
            robot=args.robot,
            stage_path=args.stage_path,
            num_envs=args.num_envs,
            use_cuda_graph=args.use_cuda_graph,
            use_mujoco=args.use_mujoco,
            randomize=args.random_init,
            headless=args.headless,
            actuation=args.actuation,
            solver=args.solver,
            integrator=args.integrator,
            solver_iteration=args.solver_iteration,
            ls_iteration=args.ls_iteration,
            njmax=args.njmax,
            nconmax=args.nconmax,
        )

        show_mujoco_viewer = args.show_mujoco_viewer and example.use_mujoco
        if show_mujoco_viewer:
            import mujoco
            import mujoco.viewer
            import mujoco_warp

            mjm, mjd = example.solver.mj_model, example.solver.mj_data
            m, d = example.solver.mjw_model, example.solver.mjw_data
            viewer = mujoco.viewer.launch_passive(mjm, mjd)

        for _ in range(args.num_frames):
            example.step()
            example.render()

            if show_mujoco_viewer:
                if not example.solver.use_mujoco:
                    mujoco_warp.get_data_into(mjd, mjm, d)
                viewer.sync()

        if example.renderer:
            example.renderer.save()
