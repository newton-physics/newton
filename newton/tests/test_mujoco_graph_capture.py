# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for SolverMuJoCo CUDA graph recapture."""

import subprocess
import sys
import unittest

from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_strict_warning_args

_RECAPTURE_SCRIPT = """
import sys

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

device, first_graph_mode = sys.argv[1:]
wp.set_device(device)
builder = newton.ModelBuilder()
builder.add_ground_plane()
body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
model = builder.finalize()
solver = SolverMuJoCo(model, use_mujoco_contacts=False)
pipeline = newton.CollisionPipeline(model)
contacts = pipeline.contacts()
state_0, state_1 = model.state(), model.state()
control = model.control()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

def simulate():
    global state_0, state_1
    # Keep the same input/output buffers between captures and replays.
    for _ in range(2):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0

with wp.ScopedCapture() as capture:
    simulate()
first_graph = capture.graph
if first_graph_mode == "replay":
    wp.capture_launch(first_graph)

with wp.ScopedCapture() as capture:
    simulate()
graph = capture.graph
if first_graph_mode != "retain":
    del first_graph

for _ in range(240):
    wp.capture_launch(graph)
np.testing.assert_allclose(state_0.body_q.numpy()[0, :3], (0.0, 0.0, 0.05), atol=5e-4, rtol=0.0)
np.testing.assert_allclose(state_0.body_qd.numpy(), 0.0, atol=1e-3, rtol=0.0)

if first_graph_mode == "retain":
    # Both graphs must remain usable when their lifetimes overlap.
    wp.capture_launch(first_graph)
    wp.capture_launch(graph)
    np.testing.assert_allclose(state_0.body_q.numpy()[0, :3], (0.0, 0.0, 0.05), atol=5e-4, rtol=0.0)
"""


def test_recapture(test, device, first_graph_mode):
    """Verify recapture with discarded, retained, and previously replayed graphs."""
    try:
        SolverMuJoCo.import_mujoco()
    except ImportError as exc:
        test.skipTest(str(exc))

    # A regression can segfault or poison the CUDA context; isolate each case.
    result = subprocess.run(
        [
            sys.executable,
            *get_strict_warning_args(),
            "-X",
            "faulthandler",
            "-c",
            _RECAPTURE_SCRIPT,
            device.alias,
            first_graph_mode,
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    test.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
    sys.stderr.write(result.stderr)


class TestMuJoCoGraphCapture(unittest.TestCase):
    pass


for first_graph_mode in ("discard", "retain", "replay"):
    add_function_test(
        TestMuJoCoGraphCapture,
        f"test_recapture_{first_graph_mode}",
        test_recapture,
        devices=get_cuda_test_devices(),
        first_graph_mode=first_graph_mode,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
