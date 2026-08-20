# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for newton.controllers.export_controller_graph."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.controllers import (
    ControllerJointImpedance,
    ControllerJointImpedanceModelFree,
    export_controller_graph,
    select_joints,
)


def _build_arm(device, link_count=2):
    """Build a revolute chain of ``link_count`` links, one articulation."""
    builder = newton.ModelBuilder()
    inertia = wp.mat33(np.diag([0.02, 0.02, 0.02]).astype(np.float32))
    parent = -1
    for index in range(link_count):
        link = builder.add_link(mass=1.0, com=wp.vec3(0.25, 0.0, 0.0), inertia=inertia, lock_inertia=True)
        builder.add_joint_revolute(
            parent=parent,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(0.5 if index else 0.0, 0.0, 0.0)),
            label=f"joint_{index}",
        )
        parent = link
    builder.add_articulation(list(range(link_count)), label="arm")
    return builder.finalize(device=device)


def _run_loaded(graph, name, count, device):
    """Launch a loaded graph and read one output parameter back."""
    wp.capture_launch(graph)
    wp.synchronize_device(device)
    result = wp.zeros(count, dtype=wp.float32, device=device)
    graph.get_param(name, result)
    return result.numpy()


@unittest.skipUnless(wp.is_cuda_available(), "graph capture requires CUDA")
class TestExportGraph(unittest.TestCase):
    def _model_based(self, device):
        model = _build_arm(device)
        selection = select_joints(model)
        return ControllerJointImpedance(
            model,
            joint_selection=selection,
            stiffness=wp.array([50.0, 30.0], dtype=wp.float32, device=device),
            damping=wp.array([5.0, 3.0], dtype=wp.float32, device=device),
            device=device,
        )

    def test_exports_a_named_parameter_per_port(self):
        """Verify every array port becomes a prefixed graph parameter, plus dt."""
        device = wp.get_device()
        controller = self._model_based(device)
        inputs, outputs = controller.input(), controller.output()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller"
            export_controller_graph(controller=controller, inputs=inputs, outputs=outputs, path=path)
            graph = wp.capture_load(str(path), device=device)
        self.assertEqual(
            set(graph.params),
            {"dt", "input.joint_q", "input.joint_qd", "input.joint_q_des", "input.joint_qd_des", "output.joint_f"},
        )

    def test_loaded_graph_reproduces_the_python_torques(self):
        """Verify the exported graph computes what the controller computed."""
        device = wp.get_device()
        controller = self._model_based(device)
        inputs, outputs = controller.input(), controller.output()
        inputs.joint_q.assign(np.array([0.3, -0.4], dtype=np.float32))
        inputs.joint_qd.assign(np.array([0.1, 0.2], dtype=np.float32))
        inputs.joint_q_des.assign(np.array([0.0, 0.5], dtype=np.float32))
        inputs.joint_qd_des.zero_()

        controller.step(inputs=inputs, outputs=outputs, dt=0.01)
        expected = outputs.joint_f.numpy().copy()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller"
            export_controller_graph(controller=controller, inputs=inputs, outputs=outputs, path=path)
            graph = wp.capture_load(str(path), device=device)
            actual = _run_loaded(graph, "output.joint_f", 2, device)
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_ports_bound_to_views_export_and_scatter(self):
        """Verify a port bound to an indexed view exports and still scatters correctly.

        The view has no device pointer of its own, so the parameter registered is
        the simulation-sized array underneath it; the gather and scatter indices
        travel inside the graph.
        """
        device = wp.get_device()
        controller = ControllerJointImpedanceModelFree(
            controlled_dofs_per_robot=wp.array([2], dtype=wp.int32, device=device),
            stiffness=wp.array([10.0, 10.0], dtype=wp.float32, device=device),
            damping=wp.array([0.0, 0.0], dtype=wp.float32, device=device),
            use_gravity_compensation=False,
            use_coriolis_compensation=False,
            use_inertia_decoupling=False,
            device=device,
        )
        inputs, outputs = controller.input(), controller.output()
        sim_q_des = wp.array(np.array([0.0, 0.0, 1.0, 2.0, 0.0], dtype=np.float32), dtype=wp.float32, device=device)
        sim_f = wp.zeros(5, dtype=wp.float32, device=device)
        qd_idx = wp.array(np.array([2, 3], dtype=np.int32), dtype=wp.int32, device=device)
        inputs.joint_q_des = sim_q_des[qd_idx]
        outputs.joint_f = sim_f[qd_idx]

        controller.step(inputs=inputs, outputs=outputs, dt=0.0)
        expected = sim_f.numpy().copy()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller"
            export_controller_graph(controller=controller, inputs=inputs, outputs=outputs, path=path)
            graph = wp.capture_load(str(path), device=device)
            actual = _run_loaded(graph, "output.joint_f", 5, device)

        # The torques land in the simulation slots the view addresses.
        np.testing.assert_allclose(actual, expected, atol=1e-4)
        self.assertTrue(np.allclose(actual[[0, 1, 4]], 0.0))

    def test_non_float32_port_raises(self):
        """Verify a port the C++ runtime cannot address raises rather than exporting."""
        device = wp.get_device()
        controller = self._model_based(device)
        inputs, outputs = controller.input(), controller.output()
        inputs.joint_q = wp.zeros(2, dtype=wp.float64, device=device)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                export_controller_graph(
                    controller=controller, inputs=inputs, outputs=outputs, path=Path(tmp) / "controller"
                )


if __name__ == "__main__":
    unittest.main()
