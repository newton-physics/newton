# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Write the graph and reference values the C++ runtime tests run against.

    uv run python newton/_src/controllers/cpp/tests/generate_artifacts.py <out_dir>

CTest runs this as a fixture before the C++ tests. The scene is built in code
rather than imported from an asset so the artifacts are reproducible offline.

Both the model and the inputs below are mirrored in ``controller_test.cpp``;
change them together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.controllers import (
    ControllerJointImpedance,
    ControllerJointImpedanceModelFree,
    export_controller_graph,
)

# One robot: a four-link revolute arm, of which the first three joints are
# controlled. The uncontrolled wrist keeps the model-sized ports (joint_q,
# joint_qd) a different length from the compact ones, which is what the C++
# buffer sizing has to get right.
#
# The joints rotate about y and the links extend along x, so the arm swings in a
# vertical plane and gravity loads every joint. Rotating about z instead would
# put the arm in the xy-plane, where gravity exerts no moment and the gravity
# term would silently contribute nothing to the reference torques.
LINK_COUNT = 4
CONTROLLED = ["shoulder", "elbow", "wrist"]
LINK_LENGTH = 0.4  # [m]
LINK_MASS = 1.5  # [kg]

Q = np.array([0.35, -0.62, 0.18, 0.9], dtype=np.float32)  # [rad]
QD = np.array([0.12, 0.4, -0.25, 0.05], dtype=np.float32)  # [rad/s]
Q_DES = np.array([0.0, -0.3, 0.5], dtype=np.float32)  # [rad]
QD_DES = np.array([0.0, 0.1, 0.0], dtype=np.float32)  # [rad/s]
STIFFNESS = np.array([120.0, 80.0, 40.0], dtype=np.float32)  # [1/s^2]
DAMPING = np.array([12.0, 8.0, 4.0], dtype=np.float32)  # [1/s]
DT = 0.01  # [s], unused by the impedance law but written into the graph

# The second artifact binds two ports to views of simulation-sized arrays, with
# the controlled DOFs scattered so that the indices in the graph demonstrably
# matter. Same physics as the first artifact, so both produce the same torques.
VIEW_SIM_DOFS = 6
VIEW_INDICES = np.array([1, 3, 5], dtype=np.int32)

# A third artifact drives ControllerJointImpedanceModelFree, whose mass matrix is
# a port rather than something it computes. It is bound to a view selecting one
# robot's block out of a fleet, so the host exchanges every block and the graph
# reads only the addressed one.
FLEET_ROBOTS = 3
FLEET_CONTROLLED_ROBOT = 1
FLEET_DOFS = 2
FLEET_BLOCK = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32)
FLEET_DECOY = 99.0  # fills the blocks the controller must not read
FLEET_Q_DES = np.array([1.0, 1.0], dtype=np.float32)  # [rad]
FLEET_STIFFNESS = np.array([1.0, 1.0], dtype=np.float32)  # [1/s^2]


def build_model(device) -> newton.Model:
    """Return a four-link revolute arm, one link per named joint plus a free end."""
    builder = newton.ModelBuilder()
    labels = [*CONTROLLED, "tip"]
    parent = -1
    for index in range(LINK_COUNT):
        link = builder.add_link(
            mass=LINK_MASS,
            com=wp.vec3(LINK_LENGTH * 0.5, 0.0, 0.0),
            inertia=wp.mat33(np.diag([0.02, 0.02, 0.02]).astype(np.float32)),
            lock_inertia=True,
        )
        builder.add_joint_revolute(
            parent=parent,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(LINK_LENGTH if index else 0.0, 0.0, 0.0)),
            label=labels[index],
        )
        parent = link
    builder.add_articulation(list(range(LINK_COUNT)), label="arm")
    return builder.finalize(device=device)


def build_controller(model, device) -> ControllerJointImpedance:
    return ControllerJointImpedance(
        model,
        joints=CONTROLLED,
        stiffness=wp.array(STIFFNESS, dtype=wp.float32, device=device),
        damping=wp.array(DAMPING, dtype=wp.float32, device=device),
    )


def export_with_views(model, device, out_dir: Path) -> np.ndarray:
    """Export the same controller with joint_q_des and joint_f bound to views.

    Each of those ports becomes a parameter of VIEW_SIM_DOFS floats rather than
    one per controlled DOF, and the graph reads and writes only VIEW_INDICES.
    """
    controller = build_controller(model, device)
    inputs, outputs = controller.input(), controller.output()
    inputs.joint_q.assign(Q)
    inputs.joint_qd.assign(QD)
    inputs.joint_qd_des.assign(QD_DES)

    scattered_q_des = np.zeros(VIEW_SIM_DOFS, dtype=np.float32)
    scattered_q_des[VIEW_INDICES] = Q_DES
    sim_q_des = wp.array(scattered_q_des, dtype=wp.float32, device=device)
    sim_f = wp.zeros(VIEW_SIM_DOFS, dtype=wp.float32, device=device)
    indices = wp.array(VIEW_INDICES, dtype=wp.int32, device=device)

    inputs.joint_q_des = sim_q_des[indices]
    outputs.joint_f = sim_f[indices]

    controller.step(inputs=inputs, outputs=outputs, dt=DT)
    scattered_torques = sim_f.numpy().copy()

    export_controller_graph(
        controller=controller, inputs=inputs, outputs=outputs, path=out_dir / "joint_impedance_views"
    )
    return scattered_torques


def export_model_free_mass_matrix_view(device, out_dir: Path) -> np.ndarray:
    """Export a model-free controller whose mass matrix is bound to a view.

    The parameter the host exchanges is the whole fleet of blocks, not the one
    block this controller uses, so the C++ buffer has to be sized from the graph
    rather than from the controlled-DOF count.
    """
    controller = ControllerJointImpedanceModelFree(
        controlled_dofs_per_robot=wp.array([FLEET_DOFS], dtype=wp.int32, device=device),
        stiffness=wp.array(FLEET_STIFFNESS, dtype=wp.float32, device=device),
        damping=wp.zeros(FLEET_DOFS, dtype=wp.float32, device=device),
        use_gravity_compensation=False,
        use_coriolis_compensation=False,
        use_inertia_decoupling=True,
        device=device,
    )
    inputs, outputs = controller.input(), controller.output()
    inputs.joint_q_des.assign(FLEET_Q_DES)

    blocks_np = np.full((FLEET_ROBOTS, FLEET_DOFS, FLEET_DOFS), FLEET_DECOY, dtype=np.float32)
    blocks_np[FLEET_CONTROLLED_ROBOT] = FLEET_BLOCK
    blocks = wp.array(blocks_np, dtype=wp.float32, device=device)
    indices = wp.array(np.array([FLEET_CONTROLLED_ROBOT], dtype=np.int32), dtype=wp.int32, device=device)
    inputs.mass_matrix = blocks[indices]

    controller.step(inputs=inputs, outputs=outputs, dt=DT)
    torques = outputs.joint_f.numpy().copy()

    export_controller_graph(controller=controller, inputs=inputs, outputs=outputs, path=out_dir / "mass_matrix_view")
    return torques


def export_cpu_variant(out_dir: Path) -> np.ndarray:
    """Export the same controller captured on the CPU instead of CUDA.

    Exercises the C++ runtime's device auto-detection: it reads the device a
    .wrp graph was captured for out of the file's own header, rather than
    being told, so a CPU-captured graph must load and replay correctly
    without a CUDA device ever entering the picture.
    """
    model = build_model("cpu")
    controller = build_controller(model, "cpu")
    inputs, outputs = controller.input(), controller.output()
    inputs.joint_q.assign(Q)
    inputs.joint_qd.assign(QD)
    inputs.joint_q_des.assign(Q_DES)
    inputs.joint_qd_des.assign(QD_DES)

    controller.step(inputs=inputs, outputs=outputs, dt=DT)
    expected = outputs.joint_f.numpy().copy()

    export_controller_graph(controller=controller, inputs=inputs, outputs=outputs, path=out_dir / "joint_impedance_cpu")
    return expected


def main(out_dir: Path) -> int:
    if not wp.is_cuda_available():
        print("generate_artifacts: CUDA is required to capture a graph", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    device = wp.get_device()

    model = build_model(device)
    controller = build_controller(model, device)

    inputs, outputs = controller.input(), controller.output()
    inputs.joint_q.assign(Q)
    inputs.joint_qd.assign(QD)
    inputs.joint_q_des.assign(Q_DES)
    inputs.joint_qd_des.assign(QD_DES)

    # Reference values, computed in Python before the capture.
    controller.step(inputs=inputs, outputs=outputs, dt=DT)
    expected = outputs.joint_f.numpy().copy()

    export_controller_graph(controller=controller, inputs=inputs, outputs=outputs, path=out_dir / "joint_impedance")
    (out_dir / "joint_impedance_expected.txt").write_text("\n".join(f"{value:.9g}" for value in expected) + "\n")

    scattered = export_with_views(model, device, out_dir)
    # Binding a port to a view changes where the values live, not the control
    # law, so the same torques must appear at the addressed slots.
    np.testing.assert_allclose(scattered[VIEW_INDICES], expected, atol=1e-5)

    fleet_torques = export_model_free_mass_matrix_view(device, out_dir)
    # tau = M @ (Kp * dq), with M the addressed block and dq = FLEET_Q_DES.
    np.testing.assert_allclose(fleet_torques, FLEET_BLOCK @ (FLEET_STIFFNESS * FLEET_Q_DES), atol=1e-5)
    (out_dir / "mass_matrix_view_expected.txt").write_text("\n".join(f"{value:.9g}" for value in fleet_torques) + "\n")

    cpu_expected = export_cpu_variant(out_dir)
    (out_dir / "joint_impedance_cpu_expected.txt").write_text(
        "\n".join(f"{value:.9g}" for value in cpu_expected) + "\n"
    )

    print(
        "generate_artifacts: wrote joint_impedance.wrp, joint_impedance_views.wrp, "
        "mass_matrix_view.wrp, joint_impedance_cpu.wrp"
    )
    print(f"                    reference torques {expected}")
    print(f"                    scattered into slots {VIEW_INDICES.tolist()} of {VIEW_SIM_DOFS}: {scattered}")
    print(f"                    model-free torques from a view-bound mass matrix: {fleet_torques}")
    print(f"                    CPU-captured torques: {cpu_expected}")
    return 0


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "artifacts"
    raise SystemExit(main(destination))
