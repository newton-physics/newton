# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time

import warp as wp

from .._src.utils import logger as msg

###
# Example Paths
###


def get_examples_output_path() -> str:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
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
    if sys.stdout.encoding == "cp1252":  # Fix for Windows terminal
        bar = "x" * filled_length + "-" * (length - filled_length)
    else:
        bar = "█" * filled_length + "-" * (length - filled_length)

    # Estimated Time of Arrival
    if iteration > 0 and elapsed > 0:
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
