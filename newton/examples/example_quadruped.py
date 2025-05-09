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
# Example Sim Quadruped
#
# Shows how to set up a simulation of a rigid-body quadruped articulation
# from a URDF using the newton.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import math
from typing import Dict, Callable

import numpy as np
import warp as wp

import newton
import newton.collision
import newton.core.articulation
import newton.examples
import newton.utils


SolverFactory = Callable[[newton.Model], newton.solvers.SolverBase]
SOLVER_MAP: Dict[str, SolverFactory] = {
    "featherstone": newton.solvers.FeatherstoneSolver,
    "mujoco": newton.solvers.MuJoCoSolver,
    "mujoco-native": lambda model: newton.solvers.MuJoCoSolver(model, use_mujoco=True),
    "xpbd": newton.solvers.XPBDSolver,
}


class Example:
    """Quadruped simulation helper that can be run interactively or head-less.

    Parameters
    ----------
    stage_path : str | None, optional
        USD stage path to write frames to. If ``None`` no renderer is created.
    num_envs : int, optional
        Number of environment copies to simulate in parallel.
    solver_cls : SolverFactory, optional
        Factory that returns an initialized solver. Defaults to
        :class:`XPBDSolver`.
    enable_timers : bool, optional
        Whether to enable per-step timing prints.
    """

    def __init__(
        self,
        stage_path: str | None = "example_quadruped.usd",
        num_envs: int = 8,
        solver_cls: SolverFactory = newton.solvers.FeatherstoneSolver,
        enable_timers: bool = True,
    ):
        articulation_builder = newton.ModelBuilder()
        newton.utils.parse_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            articulation_builder,
            xform=wp.transform([0.0, 0.7, 0.0], wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=True,
            density=1000,
            armature=0.01,
            stiffness=200,
            damping=1,
            contact_ke=1.0e4,
            contact_kd=1.0e2,
            contact_kf=1.0e2,
            contact_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
        )

        builder = newton.ModelBuilder()

        self.sim_time = 0.0
        fps = 100
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        offsets = newton.examples.compute_env_offsets(self.num_envs)
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

            builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]

            builder.joint_axis_mode = [newton.JOINT_MODE_TARGET_POSITION] * len(builder.joint_axis_mode)
            builder.joint_act[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        self.enable_timers = enable_timers

        # create solver from provided class
        self.solver = solver_cls(self.model)

        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(self.model, stage_path)
        else:
            self.renderer = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.collision.collide(self.model, self.state_0)
            self.solver.step(self.model, self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=self.enable_timers):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current frame if a renderer is attached."""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=self.enable_timers):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return key tensors as NumPy arrays for regression testing."""

        def _to_numpy(arr):
            return None if arr is None else arr.numpy()

        return {
            "joint_q": _to_numpy(self.state_0.joint_q),
            "joint_qd": _to_numpy(self.state_0.joint_qd),
            "body_q": _to_numpy(self.state_0.body_q),
            "body_qd": _to_numpy(self.state_0.body_qd),
        }


# -------------------------------------------------------------------------
# Public helper function for tests & scripts
# -------------------------------------------------------------------------

def run_quadruped(
    solver_name: str = "featherstone",
    num_frames: int = 300,
    num_envs: int = 8,
    device: str | wp.context.Device | None = None,
    render: bool = False,
    stage_path: str | None = None,
    enable_timers: bool = True,
) -> dict:
    """Run the quadruped example head-less or with rendering and return final state.

    Parameters
    ----------
    solver_name : str, optional
        Key identifying which solver to use (see ``SOLVER_MAP``).
    num_frames : int, optional
        Number of simulation frames to advance.
    num_envs : int, optional
        Number of parallel environments.
    device : str | wp.context.Device | None, optional
        Warp device to run the simulation on.
    render : bool, optional
        Whether to render the simulation to a USD stage.
    stage_path : str | None, optional
        Path to the output USD stage when ``render=True``.
    enable_timers : bool, optional
        If ``False`` disables per-step timing prints (``wp.ScopedTimer``).
    """

    # Determine solver factory from map
    solver_factory = SOLVER_MAP.get(solver_name.lower())
    if solver_factory is None:
        raise ValueError(f"Unknown solver '{solver_name}'. Valid keys: {list(SOLVER_MAP)}")

    if render and stage_path is None:
        stage_path = "example_quadruped.usd"

    # Skip rendering entirely when not requested.
    if not render:
        stage_path = None

    with wp.ScopedDevice(device):
        example = Example(stage_path=stage_path, num_envs=num_envs, solver_cls=solver_factory, enable_timers=enable_timers)

        for _ in range(num_frames):
            example.step()
            if render:
                example.render()

        # Save USD if we rendered frames.
        if render and example.renderer:
            example.renderer.save()

        return example.get_state()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_quadruped.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=8, help="Total number of simulated environments.")
    parser.add_argument(
        "--solver",
        type=str,
        default="xpbd",
        choices=list(SOLVER_MAP.keys()),
        help="Which integrator to use for the simulation.",
    )

    args = parser.parse_known_args()[0]

    run_quadruped(
        solver_name=args.solver,
        num_frames=args.num_frames,
        num_envs=args.num_envs,
        device=args.device,
        render=True,
        stage_path=args.stage_path,
        enable_timers=True,
    )
