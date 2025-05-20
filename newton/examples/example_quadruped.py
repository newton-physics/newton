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


import sys

import numpy as np
import warp as wp

import newton
import newton.collision
import newton.core.articulation
import newton.examples
import newton.utils

SOLVER_MAP = {
    "featherstone": newton.solvers.FeatherstoneSolver,
    "mujoco": newton.solvers.MuJoCoSolver,
    "mujoco-native": lambda model, **kwargs: newton.solvers.MuJoCoSolver(model, use_mujoco=True, **kwargs),
    "xpbd": newton.solvers.XPBDSolver,
}


class Example:
    def __init__(
        self,
        stage_path="example_quadruped.usd",
        num_envs=8,
        solver_cls=newton.solvers.XPBDSolver,
        enable_timers=True,
        policy="none",
        solver_kwargs=None,
    ):
        self.num_envs = num_envs
        self.policy = policy.lower()
        if self.policy not in ("none", "sin-joint-target", "sin-joint-f"):
            raise ValueError(
                f"Unsupported policy '{policy}'. Valid options are 'none', 'sin-joint-target', or 'sin-joint-f'."
            )

        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_body_armature = 0.01
        articulation_builder.default_joint_cfg.armature = 0.01
        articulation_builder.default_joint_cfg.mode = newton.JOINT_MODE_TARGET_POSITION
        articulation_builder.default_joint_cfg.target_ke = 200.0
        articulation_builder.default_joint_cfg.target_kd = 1.0
        articulation_builder.default_shape_cfg.ke = 1.0e4
        articulation_builder.default_shape_cfg.kd = 1.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e2
        articulation_builder.default_shape_cfg.mu = 1.0
        newton.utils.parse_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            articulation_builder,
            xform=wp.transform([0.0, 0.7, 0.0], wp.quat_identity()),
            floating=True,
        )
        articulation_builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
        articulation_builder.joint_target[-12:] = articulation_builder.joint_q[-12:]

        builder = newton.ModelBuilder()

        self.sim_time = 0.0
        fps = 100
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        offsets = newton.examples.compute_env_offsets(self.num_envs)
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        self.enable_timers = enable_timers

        solver_kwargs = solver_kwargs or {}
        self.solver = solver_cls(self.model, **solver_kwargs)

        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(self.model, stage_path)
        else:
            self.renderer = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        if self.policy != "none":
            self._time_device = wp.array([0.0], dtype=wp.float32, device=self.model.device)

        if self.policy == "sin-joint-target":
            self._init_sin_joint_target_policy()
        elif self.policy == "sin-joint-f":
            self._init_sin_force_policy()

        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            wp.set_module_options({"block_dim": 256})
            wp.load_module(device=self.model.device)
            wp.set_module_options({"block_dim": 256}, sys.modules[type(self.solver).__module__])
            wp.load_module(type(self.solver).__module__, device=self.model.device)
            wp.set_module_options({"block_dim": 256}, newton.solvers.solver)
            wp.load_module(newton.solvers.solver, device=self.model.device)
            if isinstance(self.solver, newton.solvers.MuJoCoSolver):
                # warm-up step to trigger compilation of factory-defined kernels in mujoco_warp
                self.simulate()
                self.state_0 = self.model.state()
                self.state_1 = self.model.state()
                self.control = self.model.control()
                if self.policy != "none":
                    self._time_device.zero_()

            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            if self.policy == "sin-joint-target":
                wp.launch(
                    _generate_sin_targets,
                    dim=self.control.joint_target.shape[0],
                    inputs=[
                        self._base_targets_w,
                        self._amp_targets_w,
                        self._phase_targets_w,
                        self._time_device,
                        self._freq_target,
                    ],
                    outputs=[self.control.joint_target],
                    device=self.model.device,
                )
            elif self.policy == "sin-joint-f":
                wp.launch(
                    _generate_sin_forces,
                    dim=self.control.joint_f.shape[0],
                    inputs=[
                        self._base_forces_w,
                        self._amp_forces_w,
                        self._phase_forces_w,
                        self._time_device,
                        self._freq_force,
                    ],
                    outputs=[self.control.joint_f],
                    device=self.model.device,
                )

            if self.policy != "none":
                wp.launch(_update_time, dim=1, inputs=[self._time_device, self.sim_dt], device=self.model.device)

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
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=self.enable_timers):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

    def get_state(self):
        def to_numpy(arr):
            return None if arr is None else arr.numpy()

        return {
            "joint_q": to_numpy(self.state_0.joint_q),
            "joint_qd": to_numpy(self.state_0.joint_qd),
            "body_q": to_numpy(self.state_0.body_q),
            "body_qd": to_numpy(self.state_0.body_qd),
        }

    def _init_sin_joint_target_policy(self):
        # initial target matching initial joint_q
        base_single_env = np.array(
            [
                0.2,
                0.4,
                -0.6,
                -0.2,
                -0.4,
                0.6,
                -0.2,
                0.4,
                -0.6,
                0.2,
                -0.4,
                0.6,
            ],
            dtype=np.float32,
        )

        # amplitudes: ±20-40 % of the nominal range.
        # for each leg we use hip 0.2 rad, knee 0.4 rad, ankle 0.2 rad.
        amp_leg = np.array([0.2, 0.4, 0.2], dtype=np.float32)
        amp_single_env = np.tile(amp_leg, 4)

        # phase offsets between legs
        leg_phase_offsets = [0.0, np.pi, np.pi / 2.0, 3.0 * np.pi / 2.0]
        phase_single_env = np.concatenate([np.full(3, p, dtype=np.float32) for p in leg_phase_offsets])

        # tile across environments
        base_np = np.tile(base_single_env, self.num_envs).astype(np.float32)
        amp_np = np.tile(amp_single_env, self.num_envs).astype(np.float32)
        phase_np = np.tile(phase_single_env, self.num_envs).astype(np.float32)

        self._base_targets_w = wp.array(base_np, dtype=wp.float32, device=self.model.device)
        self._amp_targets_w = wp.array(amp_np, dtype=wp.float32, device=self.model.device)
        self._phase_targets_w = wp.array(phase_np, dtype=wp.float32, device=self.model.device)

        self._freq_target = 1.0

        self.control.joint_target = wp.array(base_np, dtype=wp.float32, device=self.model.device)

    def _init_sin_force_policy(self):
        # base forces (zeros for all 12 joints)
        base_single_env_f = np.zeros(12, dtype=np.float32)

        # amplitudes for forces
        amp_leg_f = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        amp_single_env_f = np.tile(amp_leg_f, 4)  # 4 legs

        # phase offsets (all zero for simplicity)
        phase_single_env_f = np.zeros(12, dtype=np.float32)

        # tile across environments
        base_f_np = np.tile(base_single_env_f, self.num_envs).astype(np.float32)
        amp_f_np = np.tile(amp_single_env_f, self.num_envs).astype(np.float32)
        phase_f_np = np.tile(phase_single_env_f, self.num_envs).astype(np.float32)

        self._base_forces_w = wp.array(base_f_np, dtype=wp.float32, device=self.model.device)
        self._amp_forces_w = wp.array(amp_f_np, dtype=wp.float32, device=self.model.device)
        self._phase_forces_w = wp.array(phase_f_np, dtype=wp.float32, device=self.model.device)

        self._freq_force = 1.0

        self.control.joint_f = wp.array(base_f_np, dtype=wp.float32, device=self.model.device)


@wp.kernel
def _generate_sin_targets(
    base: wp.array(dtype=float),
    amp: wp.array(dtype=float),
    phase: wp.array(dtype=float),
    time_val: wp.array(dtype=float),
    freq: float,
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    t = time_val[0]
    out[tid] = base[tid] + amp[tid] * wp.sin(2.0 * wp.pi * freq * t + phase[tid])


@wp.kernel
def _update_time(t: wp.array(dtype=float), dt: float):
    t[0] = t[0] + dt


@wp.kernel
def _generate_sin_forces(
    base: wp.array(dtype=float),
    amp: wp.array(dtype=float),
    phase: wp.array(dtype=float),
    time_val: wp.array(dtype=float),
    freq: float,
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    t = time_val[0]
    out[tid] = base[tid] + amp[tid] * wp.sin(2.0 * wp.pi * freq * t + phase[tid])


def run_quadruped(
    solver_name="xpbd",
    num_frames=300,
    num_envs=8,
    device=None,
    stage_path=None,
    enable_timers=True,
    policy="none",
    solver_kwargs=None,
):
    solver_factory = SOLVER_MAP.get(solver_name.lower())
    if solver_factory is None:
        raise ValueError(f"Unknown solver '{solver_name}'. Valid keys: {list(SOLVER_MAP)}")

    with wp.ScopedDevice(device):
        example = Example(
            stage_path=stage_path,
            num_envs=num_envs,
            solver_cls=solver_factory,
            enable_timers=enable_timers,
            policy=policy,
            solver_kwargs=solver_kwargs,
        )

        for _ in range(num_frames):
            example.step()
            if example.renderer:
                example.render()

        if example.renderer:
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
    parser.add_argument(
        "--policy",
        type=str,
        default="none",
        choices=("none", "sin-joint-target", "sin-joint-f"),
        help="Which control policy to apply during the simulation.",
    )

    args = parser.parse_known_args()[0]

    run_quadruped(
        solver_name=args.solver,
        num_frames=args.num_frames,
        num_envs=args.num_envs,
        device=args.device,
        stage_path=args.stage_path,
        enable_timers=True,
        policy=args.policy,
    )
