# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example demonstrating the unified Cosserat rod simulation with multiple backends.

This example shows rods simulated with different backends side-by-side:
- Reference (C/C++ DLL) - Orange rod
- NumPy (CPU) - Cyan rod
- Warp CPU - Green rod (optional, for debugging)
- Warp GPU - Purple rod

All rods should behave identically, demonstrating backend correctness.

Usage:
    uv run -m newton.examples cosserat_dll.refactor.example_unified
"""

from __future__ import annotations

import argparse

import numpy as np

import newton


class Example:
    """Unified Cosserat rod example with multiple backends."""

    def __init__(self, viewer: newton.Viewer | None, args: argparse.Namespace | None = None):
        """Initialize the example."""
        from newton.examples.cosserat_dll.refactor import (
            CosseratRodModel,
            CosseratSolver,
            BackendType,
            create_straight_rod,
        )

        # Simulation parameters
        self.fps = 60
        self.dt = 1.0 / self.fps
        self.substeps = 4
        self.sub_dt = self.dt / self.substeps

        # Rod parameters
        n_particles = 32
        segment_length = 0.03
        start_height = 1.5

        # Parse arguments
        self.enable_reference = not (args and args.no_reference)
        self.enable_numpy = not (args and args.no_numpy)
        self.enable_warp_cpu = args and args.warp_cpu
        self.enable_warp_gpu = not (args and args.no_warp_gpu)
        self.use_cuda_graph = args and args.cuda_graph

        # Create rod models and solvers
        self.models: list[CosseratRodModel] = []
        self.solvers: list[CosseratSolver] = []
        self.colors: list[tuple[float, float, float]] = []
        self.y_offsets: list[float] = []

        y_offset = 0.0
        y_spacing = 0.3

        # Reference backend (Orange)
        if self.enable_reference:
            try:
                model_ref = create_straight_rod(
                    n_particles=n_particles,
                    start_pos=(0, y_offset, start_height),
                    direction=(0, 0, -1),
                    segment_length=segment_length,
                )
                solver_ref = CosseratSolver(
                    model_ref,
                    backend=BackendType.REFERENCE,
                    dll_path="unity_ref",
                )
                self.models.append(model_ref)
                self.solvers.append(solver_ref)
                self.colors.append((1.0, 0.5, 0.0))  # Orange
                self.y_offsets.append(y_offset)
                y_offset += y_spacing
                print(f"[OK] Reference backend initialized")
            except Exception as e:
                print(f"[SKIP] Reference backend: {e}")
                self.enable_reference = False

        # NumPy backend (Cyan)
        if self.enable_numpy:
            model_np = create_straight_rod(
                n_particles=n_particles,
                start_pos=(0, y_offset, start_height),
                direction=(0, 0, -1),
                segment_length=segment_length,
            )
            solver_np = CosseratSolver(model_np, backend=BackendType.NUMPY)
            self.models.append(model_np)
            self.solvers.append(solver_np)
            self.colors.append((0.0, 0.8, 0.8))  # Cyan
            self.y_offsets.append(y_offset)
            y_offset += y_spacing
            print(f"[OK] NumPy backend initialized")

        # Warp CPU backend (Green) - optional
        if self.enable_warp_cpu:
            try:
                model_warp_cpu = create_straight_rod(
                    n_particles=n_particles,
                    start_pos=(0, y_offset, start_height),
                    direction=(0, 0, -1),
                    segment_length=segment_length,
                )
                solver_warp_cpu = CosseratSolver(
                    model_warp_cpu, backend=BackendType.WARP_CPU
                )
                self.models.append(model_warp_cpu)
                self.solvers.append(solver_warp_cpu)
                self.colors.append((0.0, 0.8, 0.0))  # Green
                self.y_offsets.append(y_offset)
                y_offset += y_spacing
                print(f"[OK] Warp CPU backend initialized")
            except Exception as e:
                print(f"[SKIP] Warp CPU backend: {e}")

        # Warp GPU backend (Purple)
        if self.enable_warp_gpu:
            try:
                model_warp_gpu = create_straight_rod(
                    n_particles=n_particles,
                    start_pos=(0, y_offset, start_height),
                    direction=(0, 0, -1),
                    segment_length=segment_length,
                )
                solver_warp_gpu = CosseratSolver(
                    model_warp_gpu,
                    backend=BackendType.WARP_GPU,
                    device="cuda:0",
                    use_cuda_graph=self.use_cuda_graph,
                )
                self.models.append(model_warp_gpu)
                self.solvers.append(solver_warp_gpu)
                self.colors.append((0.6, 0.2, 0.8))  # Purple
                self.y_offsets.append(y_offset)
                y_offset += y_spacing
                print(f"[OK] Warp GPU backend initialized (CUDA graph: {self.use_cuda_graph})")
            except Exception as e:
                print(f"[SKIP] Warp GPU backend: {e}")
                self.enable_warp_gpu = False

        if not self.models:
            raise RuntimeError("No backends available!")

        # Simulation state
        self.gravity_on = True
        self.frame = 0

        # Build Newton model for visualization
        self._build_visualization_model()

        # Setup viewer
        self.viewer = viewer
        if self.viewer:
            self.viewer.init(self.newton_model, self.newton_state)

    def _build_visualization_model(self):
        """Build Newton model for rendering."""
        import warp as wp

        builder = newton.ModelBuilder()

        # Create particles for each rod
        self.particle_offsets = []
        total_particles = 0

        for idx, model in enumerate(self.models):
            self.particle_offsets.append(total_particles)
            n = model.n_particles
            positions = model.get_positions_3d()

            for i in range(n):
                pos = positions[i]
                builder.add_body(
                    xform=wp.transform(
                        p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                        q=wp.quat_identity(),
                    )
                )
                builder.add_shape_sphere(
                    body=total_particles + i,
                    radius=0.01,
                )

            total_particles += n

        self.newton_model = builder.finalize()
        self.newton_state = self.newton_model.state()

        # Create color array
        self.colors_array = np.zeros(total_particles, dtype=np.float32)
        for idx, model in enumerate(self.models):
            offset = self.particle_offsets[idx]
            n = model.n_particles
            # Encode color as single float for visualization
            r, g, b = self.colors[idx]
            color_val = r * 0.3 + g * 0.59 + b * 0.11  # Luminance
            self.colors_array[offset : offset + n] = color_val

    def step(self):
        """Advance simulation by one frame."""
        # Check for keyboard input
        if self.viewer:
            if self.viewer.key_pressed("g"):
                self.gravity_on = not self.gravity_on
                gravity = (0, 0, -9.81) if self.gravity_on else (0, 0, 0)
                for solver in self.solvers:
                    solver.set_gravity(*gravity)
                print(f"Gravity: {'ON' if self.gravity_on else 'OFF'}")

            if self.viewer.key_pressed("r"):
                for solver in self.solvers:
                    solver.reset()
                print("Reset")

        # Step each solver
        for solver in self.solvers:
            for _ in range(self.substeps):
                solver.step(self.sub_dt)

        # Update visualization state
        self._update_visualization()

        self.frame += 1

        # Print stats every 60 frames
        if self.frame % 60 == 0:
            self._print_stats()

    def _update_visualization(self):
        """Update Newton state from rod models."""
        import warp as wp

        # Get numpy view of body_q
        body_q_np = self.newton_state.body_q.numpy()

        for idx, model in enumerate(self.models):
            offset = self.particle_offsets[idx]
            positions = model.get_positions_3d()
            for i in range(model.n_particles):
                body_q_np[offset + i, :3] = positions[i]

        # Write back to GPU
        self.newton_state.body_q.assign(wp.array(body_q_np, dtype=wp.transformf))

    def _print_stats(self):
        """Print timing and comparison statistics."""
        print(f"\n--- Frame {self.frame} ---")
        for idx, solver in enumerate(self.solvers):
            stats = solver.get_stats()
            tip = solver.model.get_tip_position()
            print(
                f"  {stats['backend']:20s}: tip=({tip[0]:+.4f}, {tip[1]:+.4f}, {tip[2]:+.4f})"
                f"  step_time={stats['last_step_time_ms']:.2f}ms"
            )

        # Compare tip positions (accounting for y_offset)
        if len(self.solvers) > 1:
            ref_tip = self.solvers[0].model.get_tip_position()
            ref_y_offset = self.y_offsets[0]
            ref_tip_relative = ref_tip.copy()
            ref_tip_relative[1] -= ref_y_offset

            max_diff = 0.0
            for idx, solver in enumerate(self.solvers[1:], 1):
                tip = solver.model.get_tip_position()
                y_offset = self.y_offsets[idx]
                tip_relative = tip.copy()
                tip_relative[1] -= y_offset
                diff = np.linalg.norm(tip_relative - ref_tip_relative)
                max_diff = max(max_diff, diff)
            print(f"  Max tip difference from first backend: {max_diff*1000:.3f} mm")

    def render(self):
        """Render current state."""
        if self.viewer:
            self.viewer.render(self.newton_state)

    def gui(self, ui):
        """Build GUI elements."""
        for idx, solver in enumerate(self.solvers):
            r, g, b = self.colors[idx]
            color_str = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            ui.text(f"{solver.get_backend_name()}")

    def test_final(self):
        """Validate final simulation state."""
        # All rods should have similar tip positions (accounting for y_offset)
        if len(self.solvers) < 2:
            return

        ref_tip = self.solvers[0].model.get_tip_position()
        ref_y_offset = self.y_offsets[0]

        for idx, solver in enumerate(self.solvers[1:], 1):
            tip = solver.model.get_tip_position()
            y_offset = self.y_offsets[idx]

            # Compare relative positions (subtract y_offset)
            ref_tip_relative = ref_tip.copy()
            ref_tip_relative[1] -= ref_y_offset

            tip_relative = tip.copy()
            tip_relative[1] -= y_offset

            diff = np.linalg.norm(tip_relative - ref_tip_relative)

            # Allow up to 5mm difference (accounts for floating point variations)
            assert diff < 0.005, (
                f"Backend {solver.get_backend_name()} tip position differs from reference by {diff*1000:.2f}mm"
            )
            print(f"[PASS] {solver.get_backend_name()} matches reference within {diff*1000:.3f}mm")


def register_args(parser: argparse.ArgumentParser):
    """Register command-line arguments for this example."""
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Disable Reference (C++ DLL) backend",
    )
    parser.add_argument(
        "--no-numpy",
        action="store_true",
        help="Disable NumPy backend",
    )
    parser.add_argument(
        "--warp-cpu",
        action="store_true",
        help="Enable Warp CPU backend (disabled by default)",
    )
    parser.add_argument(
        "--no-warp-gpu",
        action="store_true",
        help="Disable Warp GPU backend",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture for Warp GPU backend",
    )


if __name__ == "__main__":
    # Allow running directly: python -m newton.examples.cosserat_dll.refactor.example_unified
    import argparse

    parser = argparse.ArgumentParser(description="Unified Cosserat rod example")
    register_args(parser)
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to simulate")
    args = parser.parse_args()

    # Create viewer if not headless
    viewer = None
    if not args.headless:
        try:
            import newton

            viewer = newton.Viewer()
        except Exception as e:
            print(f"Could not create viewer: {e}")
            print("Running headless...")

    # Create and run example
    example = Example(viewer, args)

    print(f"\nRunning {args.frames} frames...")
    for frame in range(args.frames):
        example.step()
        if viewer:
            example.render()

    print(f"\nDone! Final state:")
    example._print_stats()

    # Run test
    try:
        example.test_final()
        print("\n[PASS] All backends validated successfully!")
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
