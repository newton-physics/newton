# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the one-segment cable gravity sanity plot for the report."""

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton
from newton import Axis, TendonLinkType

FPS = 60
DT = 1.0 / FPS
GRAVITY = -9.81
REST_LENGTH = 0.1
REPORT_DIR = os.path.expanduser("~/reports/cable-sim-research")
STATIC_CHECK_SECONDS = 2.0
TRANSIENT_SECONDS = 20.0


def build_simple_cable(mass=10.0, compliance=1.0e-3, start_at_equilibrium=True):
    load = mass * abs(GRAVITY)
    initial_z = -REST_LENGTH - (load * compliance if start_at_equilibrium else 0.0)
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=GRAVITY)

    anchor = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)),
        mass=0.0,
        is_kinematic=True,
    )
    builder.add_shape_sphere(anchor, radius=0.01)

    inertia = 0.4 * mass * 0.02 * 0.02
    body = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, initial_z)),
        mass=mass,
        inertia=wp.mat33(
            inertia,
            0.0,
            0.0,
            0.0,
            inertia,
            0.0,
            0.0,
            0.0,
            inertia,
        ),
        lock_inertia=True,
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02, cfg=cfg)

    dof = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        parent=-1,
        child=body,
        linear_axes=[dof(axis=Axis.Z)],
        angular_axes=[],
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, initial_z)),
        child_xform=wp.transform(),
    )
    builder.add_articulation([joint])

    builder.add_tendon()
    builder.add_tendon_link(
        body=anchor,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )
    builder.add_tendon_link(
        body=body,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        compliance=compliance,
        damping=0.0,
        rest_length=REST_LENGTH,
    )

    return builder.finalize(), body


def run_case(
    mass=10.0,
    compliance=1.0e-3,
    iterations=16,
    seconds=STATIC_CHECK_SECONDS,
    record=False,
    start_at_equilibrium=True,
):
    model, body_idx = build_simple_cable(mass=mass, compliance=compliance, start_at_equilibrium=start_at_equilibrium)
    solver = newton.solvers.SolverXPBD(model, iterations=iterations, joint_linear_relaxation=1.0)
    state_0, state_1 = model.state(), model.state()
    control, contacts = model.control(), model.contacts()

    frames = int(round(seconds * FPS))
    history = []
    for frame in range(frames):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0

        if record:
            z = float(state_0.body_q.numpy()[body_idx][2])
            history.append((frame * DT, z))

    body_q = state_0.body_q.numpy()
    att_l = solver.tendon_seg_attachment_l.numpy()
    att_r = solver.tendon_seg_attachment_r.numpy()
    rest = solver.tendon_seg_rest_length.numpy()
    seg_len = float(np.linalg.norm(att_r[0] - att_l[0]))
    stretch = max(0.0, seg_len - float(rest[0]))
    tension = stretch / compliance
    load = mass * abs(GRAVITY)

    return {
        "mass": mass,
        "compliance": compliance,
        "iterations": iterations,
        "seconds": seconds,
        "load": load,
        "expected_stretch_mm": load * compliance * 1000.0,
        "stretch_mm": stretch * 1000.0,
        "tension": tension,
        "ratio": tension / load,
        "z": float(body_q[body_idx][2]),
        "expected_z": -REST_LENGTH - load * compliance,
        "history": history,
    }


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORT_DIR, exist_ok=True)

    compliance_results = [
        run_case(mass=10.0, compliance=comp, iterations=16) for comp in [1.0e-2, 5.0e-3, 1.0e-3, 5.0e-4, 1.0e-4, 1.0e-5]
    ]
    iter_results = [run_case(mass=10.0, compliance=1.0e-3, iterations=iters) for iters in [4, 8, 16, 32, 64, 128, 256]]
    mass_results = [run_case(mass=mass, compliance=1.0e-3, iterations=16) for mass in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]]
    history_results = [
        run_case(
            mass=10.0,
            compliance=1.0e-3,
            iterations=iters,
            seconds=TRANSIENT_SECONDS,
            record=True,
            start_at_equilibrium=False,
        )
        for iters in [4, 16, 64, 256]
    ]

    print("Settled simple cable gravity results")
    for result in compliance_results:
        print(
            f"C={result['compliance']:.1e}: T/mg={result['ratio']:.6f}, "
            f"stretch={result['stretch_mm']:.3f} mm, expected={result['expected_stretch_mm']:.3f} mm"
        )
    for result in mass_results:
        print(
            f"m={result['mass']:.1f} kg: T/mg={result['ratio']:.6f}, "
            f"stretch={result['stretch_mm']:.3f} mm, expected={result['expected_stretch_mm']:.3f} mm"
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogx([r["compliance"] for r in compliance_results], [r["ratio"] for r in compliance_results], "ko-")
    ax.axhline(y=1.0, color="r", linestyle="--", label="T/mg = 1")
    ax.set_xlabel("Compliance [m/N]")
    ax.set_ylabel("T / mg")
    ax.set_title("Static Tension Fraction vs Compliance")
    ax.set_ylim(0.95, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogx([r["iterations"] for r in iter_results], [r["ratio"] for r in iter_results], "ko-")
    ax.axhline(y=1.0, color="r", linestyle="--", label="T/mg = 1")
    ax.set_xlabel("XPBD Iterations")
    ax.set_ylabel("T / mg")
    ax.set_title("Static Tension Fraction vs Iterations")
    ax.set_ylim(0.95, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx([r["mass"] for r in mass_results], [r["ratio"] for r in mass_results], "ko-")
    ax.axhline(y=1.0, color="r", linestyle="--", label="T/mg = 1")
    ax.set_xlabel("Mass [kg]")
    ax.set_ylabel("T / mg")
    ax.set_title("Static Tension Fraction vs Mass")
    ax.set_ylim(0.95, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for result in history_results:
        history = np.asarray(result["history"], dtype=np.float64)
        ax.plot(history[:, 0], history[:, 1], linewidth=1.1, label=f"{result['iterations']} iters")
    ax.axhline(y=history_results[0]["expected_z"], color="r", linestyle="--", label="Expected equilibrium")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("z [m]")
    ax.set_title("Body Position vs Time from Original Initial Condition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(REPORT_DIR, "simple_cable_gravity_current.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
