"""Run the chysx_twist example for ~300 frames and report
self-collision contact counts every 25 frames.  Smoke-checks that the
detector actually fires once the cloth starts wringing up.

Usage (from repo root):
    uv run python diagnose_twist.py
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import newton
import newton.examples
from newton.examples.chysx.example_chysx_twist import Example


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--report-every", type=int, default=25)
    args = parser.parse_args()
    args.viewer = "null"
    args.headless = True
    args.test = True
    args.num_frames = args.num_frames
    args.output_path = None
    args.device = None
    args.disable_cuda_graph = False
    args.disable_self_contact = False
    args.no_viewer = True

    viewer, _ = newton.examples.init(argparse.ArgumentParser().parse_known_args(["--viewer", "null"])[0]) \
        if False else (newton.viewer.ViewerNull(), None)

    sim = Example(viewer, args)

    n_frames = args.num_frames
    print(f"# frame  sim_t [s]  contacts  max|q|  max|v|")
    t0 = time.perf_counter()
    for i in range(n_frames):
        sim.step()
        if i % args.report_every == 0 or i == n_frames - 1:
            count = sim.solver._sim.self_collision_count()
            q = sim.state_0.particle_q.numpy().reshape(-1, 3)
            v = sim.state_0.particle_qd.numpy().reshape(-1, 3)
            mq = float(np.abs(q).max())
            mv = float(np.linalg.norm(v, axis=1).max())
            print(f"  {i:4d}    {sim.sim_time:6.3f}    {count:5d}    {mq:5.3f}    {mv:5.3f}")
    dt_total = time.perf_counter() - t0
    print(f"# total wall time {dt_total:.2f} s, "
          f"{n_frames / dt_total:.1f} steps/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
