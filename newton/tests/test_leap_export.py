# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that LEAPP preserves a Newton impedance controller's output.

Run from the Newton repository root with::

    uv run --no-sync -m newton.tests -k test_leap_export

The test follows LEAPP's official Warp workflow: execute the annotated node
twice (discovery and APIC capture), export it as PT2, load the exported
pipeline with ``InferenceManager``, and compare its torque output with the
native Newton controller.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.controllers import ControllerJointImpedance

try:
    import leapp
    import torch
    from leapp import InferenceManager, annotate
except ImportError:
    leapp = None
    torch = None
    InferenceManager = None
    annotate = None


GRAPH_NAME = "newton_controller"
NODE_NAME = "impedance_controller"


def build_arm(device: wp.DeviceLike, link_count: int = 2) -> newton.Model:
    """Build the same small revolute arm used by the other APIC export test."""
    builder = newton.ModelBuilder()
    inertia = wp.mat33(np.diag([0.02, 0.02, 0.02]).astype(np.float32))
    parent = -1

    for index in range(link_count):
        link = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.25, 0.0, 0.0),
            inertia=inertia,
            lock_inertia=True,
        )
        builder.add_joint_revolute(
            parent=parent,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(
                p=wp.vec3(0.5 if index else 0.0, 0.0, 0.0)
            ),
            label=f"joint_{index}",
        )
        parent = link

    builder.add_articulation(list(range(link_count)), label="arm")
    return builder.finalize(device=device)


def make_controller(device: wp.DeviceLike) -> ControllerJointImpedance:
    """Create the model-based controller exercised by the PR."""
    model = build_arm(device)
    return ControllerJointImpedance(
        model,
        stiffness=wp.array([50.0, 30.0], dtype=wp.float32, device=device),
        damping=wp.array([5.0, 3.0], dtype=wp.float32, device=device),
    )


def run_native_controller(
    controller: ControllerJointImpedance,
    source_values: dict[str, np.ndarray],
) -> np.ndarray:
    """Run Newton normally to produce the reference torque."""
    inputs = controller.input()
    outputs = controller.output()
    inputs.joint_q.assign(source_values["joint_q"])
    inputs.joint_qd.assign(source_values["joint_qd"])
    inputs.joint_q_des.assign(source_values["joint_q_des"])
    inputs.joint_qd_des.assign(source_values["joint_qd_des"])

    controller.step(inputs=inputs, outputs=outputs, dt=0.01)
    wp.synchronize_device(controller.device)
    return outputs.joint_f.numpy().copy()


def export_controller(
    device: wp.DeviceLike,
    source_values: dict[str, np.ndarray],
    output_root: Path,
) -> Path:
    """Capture two identical controller passes and export the node as PT2."""
    # Construct outside leapp.start(): model construction calls scalar Warp
    # built-ins that are unrelated to the exported tensor computation.
    controllers = [make_controller(device) for _ in range(2)]

    leapp.start(name=GRAPH_NAME, save_path=str(output_root))
    try:
        # LEAPP requires two executions: discovery first, then APIC capture.
        for controller in controllers:
            # Use fresh controller and boundary objects on each pass. LEAPP
            # promotes Warp arrays to traced subclasses in place; rebuilding
            # also prevents persistent first-pass controller buffers from
            # retaining first-pass FX proxies during APIC capture.
            source_arrays = {
                name: wp.array(value, dtype=wp.float32, device=device)
                for name, value in source_values.items()
            }
            inputs = controller.input()
            outputs = controller.output()
            (
                inputs.joint_q,
                inputs.joint_qd,
                inputs.joint_q_des,
                inputs.joint_qd_des,
            ) = annotate.input_tensors(NODE_NAME, source_arrays)

            # Keep the entire controller step in one replayable Warp segment.
            with annotate.warp_op(NODE_NAME):
                controller.step(inputs=inputs, outputs=outputs, dt=0.01)

            annotate.output_tensors(
                NODE_NAME,
                {"joint_f": outputs.joint_f},
                export_with="pt2",
            )
    finally:
        leapp.stop()

    # strict=True makes any source-versus-export mismatch fail immediately.
    leapp.compile_graph(
        visualize=False,
        validate=True,
        atol=1.0e-4,
        strict=True,
    )
    return output_root / GRAPH_NAME / f"{GRAPH_NAME}.yaml"


def run_exported_controller(
    graph_path: Path,
    source_values: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the saved LEAPP pipeline through its deployment-facing runtime."""
    manager = InferenceManager(str(graph_path))
    runtime_device = manager.nodes[NODE_NAME].device
    runtime_inputs = {
        f"{NODE_NAME}/{name}": torch.as_tensor(value, device=runtime_device)
        for name, value in source_values.items()
    }
    runtime_outputs = manager.run_policy(runtime_inputs)
    return runtime_outputs[f"{NODE_NAME}/joint_f"].detach().cpu().numpy()


@unittest.skipUnless(leapp is not None and wp.is_cuda_available(), "requires LEAPP, PyTorch, and CUDA")
class TestLeappExport(unittest.TestCase):
    def test_controller_round_trip(self):
        """Assert that native and LEAPP-exported controller torques agree."""
        device = wp.get_device("cuda")
        source_values = {
            "joint_q": np.array([0.3, -0.4], dtype=np.float32),
            "joint_qd": np.array([0.1, 0.2], dtype=np.float32),
            "joint_q_des": np.array([0.0, 0.5], dtype=np.float32),
            "joint_qd_des": np.zeros(2, dtype=np.float32),
        }

        controller = make_controller(device)
        native_joint_f = run_native_controller(controller, source_values)

        with tempfile.TemporaryDirectory(prefix="newton_leapp_export_") as tmp:
            graph_path = export_controller(device, source_values, Path(tmp))
            exported_joint_f = run_exported_controller(graph_path, source_values)

        np.testing.assert_allclose(
            exported_joint_f,
            native_joint_f,
            atol=1.0e-4,
        )

        print(f"Native Newton joint_f: {native_joint_f}")
        print(f"LEAPP runtime joint_f: {exported_joint_f}")
        print("PASS: LEAPP preserved the Newton controller output.")


if __name__ == "__main__":
    unittest.main()
