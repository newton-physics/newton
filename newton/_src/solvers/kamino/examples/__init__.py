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

import os
import sys
import time

import warp as wp

from ..simulation.simulator import Simulator
from ..utils import logger as msg

###
# Example Paths
###


def get_examples_output_path() -> str:
    path = os.path.dirname(os.path.realpath(__file__)) + "/output"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_examples_data_root_path() -> str:
    path = os.path.dirname(os.path.realpath(__file__)) + "/data"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_examples_data_hdf5_path() -> str:
    path = os.path.dirname(os.path.realpath(__file__)) + "/data/hdf5"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_examples_data_npy_path() -> str:
    path = os.path.dirname(os.path.realpath(__file__)) + "/data/npy"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


###
# Utilities
###


def run_headless(example, progress: bool = True):
    """Run the simulation in headless mode for a fixed number of steps."""
    msg.notif(f"Running for {example.max_steps} steps...")
    start_time = time.time()
    for i in range(example.max_steps):
        example.step_once()
        wp.synchronize()
        if progress:
            print_progress_bar(i + 1, example.max_steps, start_time, prefix="Progress", suffix="")
    msg.notif("Finished headless run")


def print_progress_bar(iteration, total, start_time, length=40, prefix="", suffix=""):
    """
    Display a progress bar with ETA and estimated FPS.

    Args:
        iteration (int) : Current iteration
        total (int) : Total iterations
        start_time (float) : Start time from time.time()
        length (int) : Character length of the bar
        prefix (str) : Prefix string
        suffix (str) : Suffix string
    """
    elapsed = time.time() - start_time
    progress = iteration / total
    filled_length = int(length * progress)
    bar = "█" * filled_length + "-" * (length - filled_length)

    # Estimated Time of Arrival
    if iteration > 0:
        eta = elapsed / iteration * (total - iteration)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        fps = iteration / elapsed
        fps_str = f"{fps:.2f} fps"
    else:
        eta_str = "Calculating..."
        fps_str = "-- fps"

    line_reset = " " * 120
    sys.stdout.write(f"\r{line_reset}")
    sys.stdout.write(f"\r{prefix} |{bar}| {iteration}/{total} ETA: {eta_str} ({fps_str}) {suffix}")
    sys.stdout.flush()

    if iteration == total:
        sys.stdout.write("\n")


def print_frame(sim: Simulator, i: int, pfunc=msg.debug):
    # Extract active constraint dimensions
    nbd = sim.model.info.num_body_dofs.numpy()[0]
    maxcts = sim.model.info.max_total_cts.numpy()[0]
    ncts = sim.model_data.info.num_total_cts.numpy()[0]
    nl = sim.limits.world_max_limits.numpy()[0]
    nc = sim.contacts.world_active_contacts.numpy()[0]

    # Print the simulation state
    pfunc(f"[s={i}]: nbd: {nbd}, ncts: {ncts}, nl: {nl}, nc: {nc}")
    pfunc(f"[s={i}]: sim.J:\n", sim.jacobians.data.J_cts_data.numpy().reshape(maxcts, nbd)[:ncts, :])
    pfunc(f"[s={i}]: problem.u_f:\n", sim._dual_problem._data.u_f.numpy())
    pfunc(f"[s={i}]: problem.v_b:\n", sim._dual_problem._data.v_b.numpy()[:ncts])
    pfunc(f"[s={i}]: problem.v_i:\n", sim._dual_problem._data.v_i.numpy()[:ncts])
    pfunc(f"[s={i}]: problem.D:\n", sim._dual_problem.data.D.numpy().reshape(maxcts, maxcts)[:ncts, :ncts])
    pfunc(f"[s={i}]: problem.v_f:\n", sim._dual_problem.data.v_f.numpy()[:ncts])
    pfunc(f"[s={i}]: problem.mu:\n", sim._dual_problem.data.mu.numpy()[:nc])
    pfunc(f"[s={i}]: sim._dual_solver.lambdas.numpy()\n", sim._dual_solver.data.solution.lambdas.numpy()[:ncts])
    pfunc(f"[s={i}]: sim._dual_solver.v_plus.numpy()\n", sim._dual_solver.data.solution.v_plus.numpy()[:ncts])
    pfunc(f"[s={i}]: sim.model_data.bodies.q_i:\n", sim.model_data.bodies.q_i.numpy())
    pfunc(f"[s={i}]: sim.model_data.bodies.u_i:\n", sim.model_data.bodies.u_i.numpy())
    pfunc(f"[s={i}]: sim.model_data.bodies.w_i:\n", sim.model_data.bodies.w_i.numpy())
    pfunc(f"[s={i}]: sim.model_data.bodies.w_a_i:\n", sim.model_data.bodies.w_a_i.numpy())
    pfunc(f"[s={i}]: sim.model_data.bodies.w_j_i:\n", sim.model_data.bodies.w_j_i.numpy())
    pfunc(f"[s={i}]: sim.model_data.bodies.w_l_i:\n", sim.model_data.bodies.w_l_i.numpy())
    pfunc(f"[s={i}]: sim.model_data.bodies.w_c_i:\n", sim.model_data.bodies.w_c_i.numpy())
    pfunc(f"[s={i}]: sim.model_data.bodies.w_e_i:\n", sim.model_data.bodies.w_e_i.numpy())
