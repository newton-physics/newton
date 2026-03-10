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
# Quick smoke test: DR Legs with many parallel environments
#
# Usage:
#   python test_multi_env_dr_legs.py                          # default 2048 envs
#   python test_multi_env_dr_legs.py --num-worlds 4096
#   python test_multi_env_dr_legs.py --num-worlds 1024 2048 4096  # sweep
###########################################################################

import argparse
import time

import warp as wp

import newton
from newton._src.solvers.kamino._src.models.builders.utils import (
    build_usd,
    make_homogeneous_builder,
    set_uniform_body_pose_offset,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.sim import Simulator

wp.set_module_options({"enable_backward": False})


def make_settings(sim_dt: float = 0.004) -> Simulator.Config:
    settings = Simulator.Config()
    settings.dt = sim_dt
    settings.solver.integrator = "moreau"
    settings.solver.constraints.alpha = 0.1
    settings.solver.padmm.primal_tolerance = 1e-6
    settings.solver.padmm.dual_tolerance = 1e-6
    settings.solver.padmm.compl_tolerance = 1e-6
    settings.solver.padmm.max_iterations = 200
    settings.solver.padmm.rho_0 = 0.1
    settings.solver.padmm.use_acceleration = True
    settings.solver.padmm.warmstart_mode = "containers"
    settings.solver.collect_solver_info = False
    settings.solver.compute_solution_metrics = False
    return settings


def run_test(num_worlds: int, num_steps: int, device):
    asset_path = newton.utils.download_asset("disneyresearch")
    usd_path = str(asset_path / "dr_legs/usd/dr_legs_with_boxes.usda")

    msg.notif(f"--- Testing {num_worlds} environments ---")

    # Build model
    msg.info("Building model...")
    builder = make_homogeneous_builder(
        num_worlds=num_worlds,
        build_fn=build_usd,
        source=usd_path,
        load_static_geometry=True,
        load_drive_dynamics=True,
        ground=True,
    )
    builder.max_contacts_per_pair = 8  # Cap contact budget to avoid Warp tile API shared memory bug
    offset = wp.transformf(0.0, 0.0, 0.265, 0.0, 0.0, 0.0, 1.0)
    set_uniform_body_pose_offset(builder=builder, offset=offset)
    for w in range(builder.num_worlds):
        builder.gravity[w].enabled = True

    # Create simulator
    msg.info("Creating simulator...")
    settings = make_settings(0.004)
    sim = Simulator(builder=builder, config=settings, device=device)
    sim.set_control_callback(lambda _: None)

    msg.info(f"Model size: {sim.model.size}")
    msg.info(f"Contacts capacity: {sim.contacts.model_max_contacts_host}")

    # Warm-up: step without CUDA graphs to compile kernels
    msg.info("Stepping (warmup)...")
    try:
        sim.step()
        wp.synchronize()
        msg.notif("Warmup step 1 OK")
    except Exception as e:
        msg.error(f"Warmup step 1 FAILED: {e}")
        return False

    # Reset
    msg.info("Resetting...")
    sim.reset()
    wp.synchronize()
    msg.notif("Reset OK")

    # Step multiple times
    msg.info(f"Stepping {num_steps} times...")
    t0 = time.perf_counter()
    for _i in range(num_steps):
        sim.step()
    wp.synchronize()
    t1 = time.perf_counter()

    msg.notif(f"OK: {num_worlds} envs x {num_steps} steps in {t1 - t0:.2f}s ({num_steps / (t1 - t0):.0f} steps/s)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-env smoke test for DR Legs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-worlds", type=int, nargs="+", default=[2048])
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()

    msg.set_log_level(msg.LogLevel.INFO)

    if args.device:
        device = wp.get_device(args.device)
        wp.set_device(device)
    else:
        device = wp.get_preferred_device()

    for nw in args.num_worlds:
        ok = run_test(nw, args.num_steps, device)
        if not ok:
            msg.error(f"FAILED at {nw} envs")
            break

    msg.notif("Done!")
