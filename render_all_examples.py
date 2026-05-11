"""Render all tendon examples headless to MP4."""

import importlib
import os
import traceback

os.environ["DISPLAY"] = ":99"

import imageio
import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerGL

EXAMPLES = [
    ("tendon_pulley", "newton.examples.cable.example_tendon_pulley", 240),
    ("tendon_pinhole_friction", "newton.examples.cable.example_tendon_pinhole_friction", 100),
    ("tendon_pinhole_routing", "newton.examples.cable.example_tendon_pinhole_routing", 240),
    ("tendon_rolling_pulley", "newton.examples.cable.example_tendon_rolling_pulley", 180),
    ("tendon_xy_table", "newton.examples.cable.example_tendon_xy_table", 600),
    ("tendon_capstan_dynamic", "newton.examples.cable.example_tendon_capstan_friction", 100),
    ("tendon_capstan_kinematic", "newton.examples.cable.example_tendon_capstan_kinematic", 180),
    ("tendon_compound_pulley", "newton.examples.cable.example_tendon_compound_pulley", 220),
    ("tendon_gear_pulley", "newton.examples.cable.example_tendon_gear_pulley", 180),
    ("tendon_3d_routing", "newton.examples.cable.example_tendon_3d_routing", 140),
    ("tendon_cable_machine", "newton.examples.cable.example_tendon_cable_machine", 100),
]

NUM_FRAMES = 240
FPS = 60
OUTPUT_WIDTH = int(os.environ.get("NEWTON_RENDER_WIDTH", "960"))
OUTPUT_HEIGHT = int(os.environ.get("NEWTON_RENDER_HEIGHT", "720"))
SUPERSAMPLE = max(1, int(os.environ.get("NEWTON_RENDER_SUPERSAMPLE", "1")))
SOLVER = os.environ.get("NEWTON_RENDER_SOLVER", "xpbd").strip().lower()
OUTPUT_PREFIX = os.environ.get("NEWTON_RENDER_PREFIX", "" if SOLVER == "xpbd" else f"{SOLVER}_")
EXAMPLE_FILTER = {name.strip() for name in os.environ.get("NEWTON_RENDER_EXAMPLES", "").split(",") if name.strip()}
REPORT_DIR = os.path.expanduser("~/reports/cable-sim-research")
os.makedirs(REPORT_DIR, exist_ok=True)

TENDON_VBD_SOLVER_KWARGS = {
    "iterations": 20,
    "rigid_avbd_beta": 1.0e6,
    "rigid_joint_linear_k_start": 1.0e7,
    "rigid_joint_angular_k_start": 1.0e5,
    "rigid_joint_linear_kd": 5.0e-2,
    "rigid_joint_angular_kd": 2.0e-2,
    "rigid_tendon_relaxation": 0.7,
}


class FakeArgs:
    def __init__(self, num_frames):
        self.headless = True
        self.record = False
        self.num_frames = num_frames
        self.episode_frames = None


def downsample_frame(frame_np):
    if SUPERSAMPLE == 1:
        return frame_np

    height, width, channels = frame_np.shape
    out_height = height // SUPERSAMPLE
    out_width = width // SUPERSAMPLE
    cropped = frame_np[: out_height * SUPERSAMPLE, : out_width * SUPERSAMPLE]
    reduced = cropped.reshape(out_height, SUPERSAMPLE, out_width, SUPERSAMPLE, channels).mean(axis=(1, 3))
    return np.rint(reduced).astype(frame_np.dtype)


def _set_serial_body_coloring(model):
    model.body_color_groups = [
        wp.array([body], dtype=wp.int32, device=model.device) for body in range(model.body_count)
    ]


def apply_solver_override(example):
    if SOLVER == "xpbd":
        return
    if SOLVER != "vbd":
        raise ValueError(f"Unsupported NEWTON_RENDER_SOLVER={SOLVER!r}")

    _set_serial_body_coloring(example.model)
    solver_kwargs = dict(TENDON_VBD_SOLVER_KWARGS)
    if all(hasattr(example, attr) for attr in ("p2_dof_start", "p6_dof_start", "_apply_cable_pretension")):
        example.model.joint_target_kd[example.p2_dof_start : example.p2_dof_start + 1].fill_(200.0)
        example.model.joint_target_kd[example.p6_dof_start : example.p6_dof_start + 1].fill_(200.0)
        solver_kwargs["iterations"] = 30
        solver_kwargs["rigid_tendon_relaxation"] = 0.6

    example.solver = newton.solvers.SolverVBD(example.model, **solver_kwargs)
    if all(hasattr(example, attr) for attr in ("p2_dof_start", "p6_dof_start", "_apply_cable_pretension")):
        example._apply_cable_pretension(0.99995)


def render_example(name, module_path, num_frames, viewer):
    print(f"\n{'=' * 60}")
    print(f"Rendering: {name}")
    print(f"{'=' * 60}")

    mod = importlib.import_module(module_path)
    example = mod.Example(viewer, FakeArgs(num_frames))
    apply_solver_override(example)

    mp4_path = os.path.join(REPORT_DIR, f"{OUTPUT_PREFIX}{name}.mp4")
    writer = imageio.get_writer(
        mp4_path,
        fps=FPS,
        codec="libx264",
        output_params=["-crf", "20", "-pix_fmt", "yuv420p"],
    )

    frame_buf = None
    last_frame_np = None
    for frame in range(num_frames):
        example.step()

        if hasattr(example, "state_0") and not np.isfinite(example.state_0.body_q.numpy()).all():
            print(f"  frame {frame}: non-finite body state; freezing remaining diagnostic frames")
            if last_frame_np is None:
                break
            for freeze_frame in range(frame, num_frames):
                writer.append_data(last_frame_np)
                if freeze_frame % 60 == 0:
                    print(f"  frame {freeze_frame}/{num_frames}")
            break

        example.render()

        frame_buf = viewer.get_frame(target_image=frame_buf)
        frame_np = frame_buf.numpy()
        frame_np = downsample_frame(frame_np)
        writer.append_data(frame_np)
        last_frame_np = np.array(frame_np, copy=True)

        if frame % 60 == 0:
            print(f"  frame {frame}/{num_frames}")

    writer.close()
    print(f"  Saved: {mp4_path}")
    return mp4_path


def main():
    viewer = ViewerGL(width=OUTPUT_WIDTH * SUPERSAMPLE, height=OUTPUT_HEIGHT * SUPERSAMPLE, headless=True)
    print(f"Rendering with solver: {SOLVER}")
    if SUPERSAMPLE > 1:
        print(
            f"Supersampling {OUTPUT_WIDTH * SUPERSAMPLE}x{OUTPUT_HEIGHT * SUPERSAMPLE} "
            f"-> {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}"
        )

    paths = {}
    for name, module_path, num_frames in EXAMPLES:
        if EXAMPLE_FILTER and name not in EXAMPLE_FILTER:
            continue
        try:
            mp4_path = render_example(name, module_path, num_frames, viewer)
            paths[name] = mp4_path
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("Done. Videos:")
    for name, path in paths.items():
        url = f"https://reports.mmacklin.com/cable-sim-research/{os.path.basename(path)}"
        print(f"  {name}: {url}")


if __name__ == "__main__":
    main()
