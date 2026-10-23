# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from functools import lru_cache

import warp as wp

from .._src.utils import logger as msg

###
# Example Paths
###


@lru_cache(maxsize=1)
def get_examples_output_path() -> str:
    """Return the preferred folder for example outputs, resolved once per session.
    The returned path is not created on disk.

    Prefers ``<package>/newton/_src/solvers/kamino/examples/output/`` (git-ignored
    in cloned checkouts), and falls back to ``<cwd>/kamino_examples_output/`` when
    the package folder is not writable (e.g. a pip-installed package in a read-only
    site-packages). The resolved path is cached so later cwd changes do not split
    outputs across folders, and a notification is emitted on the first call.
    """
    pkg_dir = os.path.dirname(os.path.realpath(__file__))
    # Probe write access with a per-PID file. ``os.access`` is race-free but
    # unreliable on Windows (never consults the ACL for a directory; returns
    # True for e.g. ``C:\Program Files``) and ``tempfile.TemporaryFile`` would
    # busy-loop on an ACL-restricted dir because ``_mkstemp_inner`` retries up
    # to ``TMP_MAX`` times when ``os.access`` says the dir is writable. Direct
    # ``os.open`` fails fast with ``PermissionError``.
    probe_path = os.path.join(pkg_dir, f".kamino_write_probe.{os.getpid()}")
    try:
        fd = os.open(probe_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        os.unlink(probe_path)
        path = os.path.join(pkg_dir, "output")
    except OSError:
        path = os.path.join(os.getcwd(), "kamino_examples_output")
    msg.info("Kamino examples output folder: %s", path)
    return path


###
# Utilities
###


def run_headless(example, progress: bool = True):
    """Run the simulation in headless mode for a fixed number of steps."""
    msg.info("Running for %s steps...", example.max_steps)
    start_time = time.time()
    for i in range(example.max_steps):
        example.step_once()
        wp.synchronize()
        if progress:
            print_progress_bar(i + 1, example.max_steps, start_time, prefix="Progress", suffix="")
    msg.info("Finished headless run")


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

    if sys.platform != "win32":
        line_reset = " " * 120
        sys.stdout.write(f"\r{line_reset}")
    sys.stdout.write(f"\r{prefix} |{bar}| {iteration}/{total} ETA: {eta_str} ({fps_str}) {suffix}")
    sys.stdout.flush()

    if iteration == total:
        sys.stdout.write("\n")
