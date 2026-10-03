# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

import newton


def _build_fk_model(topology, joint_count, world_count, device):
    """Replicate chains or a root with up to four two-joint branches."""
    template = newton.ModelBuilder()
    bodies, joints = [], []
    for index in range(joint_count):
        body = template.add_link(
            mass=1.0,
            inertia=wp.diag(wp.vec3(1.0, 1.0, 1.0)),
            com=wp.vec3(0.05, 0.0, 0.02),
        )
        parent = index - 1 if topology == "serial" or index % 2 == 0 else 0
        joints.append(
            template.add_joint_revolute(
                parent=-1 if parent < 0 else bodies[parent],
                child=body,
                axis=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[index % 3],
                parent_xform=wp.transform(wp.vec3(0.2, 0.02, 0.03), wp.quat_identity()),
            )
        )
        bodies.append(body)
        template.joint_q[-1] = 0.1 * (index + 1)
        template.joint_qd[-1] = 0.2 * (index + 1)
    template.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(template, world_count)
    return builder.finalize(device=device)


class FastForwardKinematics:
    """Time 100 fixed-state public FK calls for tiny and larger articulations."""

    params = (["serial", "branched"], [2, 9], [1, 16, 17, 256, 4096, 65536])
    param_names = ["topology", "joint_count", "world_count"]
    repeat = 8
    number = 1
    rounds = 2

    def setup(self, topology, joint_count, world_count):
        device = wp.get_device()
        if not device.is_cuda or not wp.is_mempool_enabled(device):
            raise SkipNotImplemented

        self.model = _build_fk_model(topology, joint_count, world_count, device)
        self.state = self.model.state()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(100):
                newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.graph = capture.graph
        for _ in range(3):
            wp.capture_launch(self.graph)
        wp.synchronize_device(device)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_fk(self, topology, joint_count, world_count):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.model.device)

    def teardown(self, topology, joint_count, world_count):
        if not all(np.isfinite(array.numpy()).all() for array in (self.state.body_q, self.state.body_qd)):
            raise RuntimeError("Forward kinematics produced non-finite body state")
