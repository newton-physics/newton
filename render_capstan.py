"""Render capstan friction examples and generate slip-vs-mu analysis."""

import importlib
import os

os.environ["DISPLAY"] = ":99"

import imageio
import numpy as np
import warp as wp

import newton
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.viewer import ViewerGL

NUM_FRAMES = 100
FPS = 60
REPORT_DIR = os.path.expanduser("~/reports/cable-sim-research")
os.makedirs(REPORT_DIR, exist_ok=True)


class FakeArgs:
    headless = True
    record = False
    num_frames = NUM_FRAMES
    episode_frames = None


def render_example(name, module_path, viewer):
    print(f"\n{'=' * 60}")
    print(f"Rendering: {name}")
    print(f"{'=' * 60}")

    mod = importlib.import_module(module_path)
    example = mod.Example(viewer, FakeArgs())

    mp4_path = os.path.join(REPORT_DIR, f"{name}.mp4")
    writer = imageio.get_writer(
        mp4_path,
        fps=FPS,
        codec="libx264",
        output_params=["-crf", "20", "-pix_fmt", "yuv420p"],
    )

    frame_buf = None
    for frame in range(NUM_FRAMES):
        example.step()
        example.render()

        frame_buf = viewer.get_frame(target_image=frame_buf)
        frame_np = frame_buf.numpy()
        writer.append_data(frame_np)

        if frame % 60 == 0:
            print(f"  frame {frame}/{NUM_FRAMES}")

    writer.close()
    print(f"  Saved: {mp4_path}")
    return mp4_path


def run_slip_sweep():
    """Sweep mu values and measure heavy-mass displacement after 5 seconds."""
    mu_values = np.concatenate(
        [
            np.linspace(0.0, 0.1, 5),
            np.linspace(0.15, 0.5, 8),
            np.linspace(0.6, 1.0, 3),
            np.array([2.0, 5.0, 10.0]),
        ]
    )
    mu_values = np.sort(np.unique(mu_values))

    fps = 60
    frame_dt = 1.0 / fps
    sim_substeps = 16
    sim_dt = frame_dt / sim_substeps
    sim_seconds = 1.5
    num_frames = int(sim_seconds * fps)

    mass_light = 1.0
    mass_heavy = 3.0
    pulley_radius = 0.15

    results_kinematic = []
    results_dynamic = []

    for mu in mu_values:
        for mode in ["kinematic", "dynamic"]:
            builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

            pulley_pos = wp.vec3(0.0, 0.0, 3.5)
            if mode == "kinematic":
                pulley = builder.add_body(
                    xform=wp.transform(p=pulley_pos, q=wp.quat_identity()),
                    mass=0.0,
                    is_kinematic=True,
                )
            else:
                pulley = builder.add_body(
                    xform=wp.transform(p=pulley_pos, q=wp.quat_identity()),
                    mass=5.0,
                )

            q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
            builder.add_shape_cylinder(
                pulley,
                xform=wp.transform(q=q_cyl),
                radius=pulley_radius,
                half_height=0.04,
            )

            if mode == "dynamic":
                Dof = newton.ModelBuilder.JointDofConfig
                j_pulley = builder.add_joint_d6(
                    parent=-1,
                    child=pulley,
                    linear_axes=[],
                    angular_axes=[Dof(axis=Axis.Y)],
                    parent_xform=wp.transform(p=pulley_pos),
                    child_xform=wp.transform(),
                )
                builder.add_articulation([j_pulley])

            Dof = newton.ModelBuilder.JointDofConfig
            planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
            planar_ang = [Dof(axis=Axis.Y)]

            left_pos = wp.vec3(-0.4, 0.0, 2.0)
            left = builder.add_link(
                xform=wp.transform(p=left_pos, q=wp.quat_identity()),
                mass=mass_light,
            )
            builder.add_shape_box(left, hx=0.06, hy=0.06, hz=0.06)
            j1 = builder.add_joint_d6(
                parent=-1,
                child=left,
                linear_axes=planar_lin,
                angular_axes=planar_ang,
                parent_xform=wp.transform(p=left_pos),
                child_xform=wp.transform(),
            )
            builder.add_articulation([j1])

            right_pos = wp.vec3(0.4, 0.0, 2.0)
            right = builder.add_link(
                xform=wp.transform(p=right_pos, q=wp.quat_identity()),
                mass=mass_heavy,
            )
            builder.add_shape_box(right, hx=0.09, hy=0.09, hz=0.09)
            j2 = builder.add_joint_d6(
                parent=-1,
                child=right,
                linear_axes=planar_lin,
                angular_axes=planar_ang,
                parent_xform=wp.transform(p=right_pos),
                child_xform=wp.transform(),
            )
            builder.add_articulation([j2])

            axis = (0.0, 1.0, 0.0)
            builder.add_tendon()
            builder.add_tendon_link(
                body=left,
                link_type=int(TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.06),
                axis=axis,
            )
            builder.add_tendon_link(
                body=pulley,
                link_type=int(TendonLinkType.ROLLING),
                radius=pulley_radius,
                orientation=1,
                mu=mu,
                offset=(0.0, 0.0, 0.0),
                axis=axis,
                compliance=1.0e-5,
                damping=0.1,
                rest_length=-1.0,
            )
            builder.add_tendon_link(
                body=right,
                link_type=int(TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.06),
                axis=axis,
                compliance=1.0e-5,
                damping=0.1,
                rest_length=-1.0,
            )

            builder.add_ground_plane()
            model = builder.finalize()

            solver = newton.solvers.SolverXPBD(
                model,
                iterations=8,
                joint_linear_relaxation=0.8,
            )

            state_0 = model.state()
            state_1 = model.state()
            control = model.control()
            contacts = model.contacts()

            initial_z_heavy = state_0.body_q.numpy()[right][2]
            max_descent = 0.0

            for _ in range(num_frames):
                for __ in range(sim_substeps):
                    state_0.clear_forces()
                    model.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, sim_dt)
                    state_0, state_1 = state_1, state_0
                cur_z = state_0.body_q.numpy()[right][2]
                descent = initial_z_heavy - cur_z
                if descent > max_descent:
                    max_descent = descent

            displacement = max_descent

            if mode == "kinematic":
                results_kinematic.append((mu, displacement))
            else:
                results_dynamic.append((mu, displacement))

            print(f"  mu={mu:.3f} ({mode}): displacement={displacement:.4f}")

    return np.array(results_kinematic), np.array(results_dynamic)


def generate_graph(results_kin, results_dyn):
    """Generate slip-vs-mu graph as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    mu_kin, disp_kin = results_kin[:, 0], results_kin[:, 1]
    mu_dyn, disp_dyn = results_dyn[:, 0], results_dyn[:, 1]

    # normalize to frictionless displacement
    if disp_kin[0] > 1e-6:
        norm_kin = disp_kin / disp_kin[0]
    else:
        norm_kin = disp_kin
    if disp_dyn[0] > 1e-6:
        norm_dyn = disp_dyn / disp_dyn[0]
    else:
        norm_dyn = disp_dyn

    ax.plot(mu_kin, norm_kin, "o-", color="#2196F3", linewidth=2, markersize=5, label="Kinematic pulley (fixed)")
    ax.plot(mu_dyn, norm_dyn, "s-", color="#FF5722", linewidth=2, markersize=5, label="Dynamic pulley (m=5 kg)")

    mu_crit_kin = np.log(3.0) / np.pi
    ax.axvline(
        mu_crit_kin,
        color="#2196F3",
        linestyle="--",
        alpha=0.5,
        label=f"$\\mu_{{crit}}$ kinematic = ln(3)/$\\pi$ = {mu_crit_kin:.3f}",
    )

    ax.set_xlabel("Friction coefficient $\\mu$", fontsize=13)
    ax.set_ylabel("Normalized peak displacement\n(1.0 = frictionless)", fontsize=13)
    ax.set_title(
        "Capstan Friction: Heavy-Mass Peak Displacement vs. $\\mu$\n(3:1 Atwood machine, half-wrap, 1.5s simulation)",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.set_xlim(-0.02, 1.1)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    png_path = os.path.join(REPORT_DIR, "capstan_slip_vs_mu.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved graph: {png_path}")
    return png_path


def main():
    viewer = ViewerGL(width=960, height=720, headless=True)

    # render dynamic example
    p1 = render_example(
        "tendon_capstan_dynamic",
        "newton.examples.cable.example_tendon_capstan_friction",
        viewer,
    )

    # render kinematic example
    p2 = render_example(
        "tendon_capstan_kinematic",
        "newton.examples.cable.example_tendon_capstan_kinematic",
        viewer,
    )

    # # run mu sweep and generate graph
    # print(f"\n{'='*60}")
    # print("Running slip-vs-mu sweep...")
    # print(f"{'='*60}")
    # results_kin, results_dyn = run_slip_sweep()
    # p3 = generate_graph(results_kin, results_dyn)
    p3 = None

    print(f"\n{'=' * 60}")
    print("Done. Outputs:")
    base = "https://reports.mmacklin.com/cable-sim-research"
    for p in [p1, p2, p3] if p3 else [p1, p2]:
        print(f"  {base}/{os.path.basename(p)}")


if __name__ == "__main__":
    main()
