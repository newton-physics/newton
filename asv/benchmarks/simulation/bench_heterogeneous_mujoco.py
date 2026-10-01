# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure native and Newton contacts for heterogeneous convex decompositions.

Each world contains one free cuboid above a shared plane. A small mesh bank is
reused across worlds, as with repeated assets in robot learning. The uniform
case cycles through one to five hulls; the skewed case uses eight hulls in one
world out of sixteen and one hull elsewhere. Homogeneous controls use three
hulls everywhere and compare the default native path with the experimental
opt-in. These synthetic scenes do not measure a complete robot task.
"""

import statistics
import time
from typing import ClassVar

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented
from mujoco_warp import OverflowType

import newton
from newton.solvers import SolverMuJoCo


def _build_model(world_count: int, distribution: str, device: wp.Device) -> tuple[newton.Model, np.ndarray]:
    """Build independent worlds while sharing mesh assets between repeated variants."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    variants = {}
    heights = []
    for world in range(world_count):
        if distribution == "uniform":
            count, height = 1 + world % 5, 0.12 + (world % 5) * 0.01
        elif distribution == "skew":
            count, height = (8 if world % 16 == 0 else 1), 0.15
        else:
            count, height = 3, 0.15
        key = (count, height)
        if key not in variants:
            template = newton.ModelBuilder()
            inertia = np.diag([(0.2**2 + height**2) / 3, (0.24**2 + height**2) / 3, (0.24**2 + 0.2**2) / 3])
            body = template.add_link(
                label="object/body",
                xform=wp.transform((0.0, 0.0, 0.8), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(inertia),
            )
            joint = template.add_joint_free(body, label="object/free")
            template.add_articulation([joint], label="object")
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e5, kd=1.0e3, mu=0.5)
            half_width = 0.24 / count
            mesh = newton.Mesh.create_box(half_width, 0.2, height, compute_inertia=False)
            for part in range(count):
                template.add_shape_convex_hull(
                    body,
                    mesh=mesh,
                    cfg=cfg,
                    label=f"object/hull_{part}",
                    xform=wp.transform((-0.24 + (2 * part + 1) * half_width, 0.0, 0.0), wp.quat_identity()),
                )
            variants[key] = template
        builder.add_world(variants[key])
        heights.append(height)
    return builder.finalize(device=device), np.asarray(heights)


class HeterogeneousMuJoCo:
    """Compare contact generation with identical physics and buffer capacities.

    Compilation, model construction, graph capture, and settling are outside
    the timed region. Each sample advances 200 physics steps using eight-step
    CUDA graphs. The median includes graph launch overhead and one device
    synchronization per sample. Solver iteration warnings retain their normal
    behavior; collision-buffer overflow or incorrect resting heights fail the
    benchmark instead of producing a misleading timing.
    """

    params: ClassVar = (
        [16, 256, 1024],
        [
            "uniform_native",
            "uniform_newton",
            "skew_native",
            "skew_newton",
            "homogeneous_default",
            "homogeneous_enabled",
        ],
    )
    param_names: ClassVar[list[str]] = ["world_count", "case"]
    timeout = 600
    graph_steps = 8
    launch_count = 25
    samples = 5

    def setup(self, world_count: int, case: str) -> None:
        """Compile and settle the requested contact workload before timing."""
        if wp.get_cuda_device_count() == 0:
            raise SkipNotImplemented
        self.device = wp.get_device("cuda:0")
        if not wp.is_mempool_enabled(self.device):
            raise SkipNotImplemented
        distribution, mode = case.split("_")
        native = mode != "newton"
        with wp.ScopedDevice(self.device):
            self.model, self.heights = _build_model(world_count, distribution, self.device)
            self.solver = SolverMuJoCo(
                self.model,
                allow_heterogeneous_shapes=mode != "default",
                use_mujoco_contacts=native,
                iterations=50,
                ls_iterations=20,
                nconmax=128,
                njmax=512,
            )
            self.pipeline = (
                None
                if native
                else newton.CollisionPipeline(self.model, rigid_contact_max=max(256, self.model.shape_count * 8))
            )
            self.contacts = None if native else self.pipeline.contacts()
            self.state = self.model.state()
            self.next_state = self.model.state()
            self.control = self.model.control()
            newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
            for _ in range(2):
                self._step()
            with wp.ScopedCapture(device=self.device) as capture:
                for _ in range(self.graph_steps):
                    self._step()
            self.graph = capture.graph
            for _ in range(40):
                wp.capture_launch(self.graph)
        self._validate()

    def _step(self) -> None:
        self.state.clear_forces()
        if self.pipeline is not None:
            self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, 1.0 / 240.0)
        self.state, self.next_state = self.next_state, self.state

    def _validate(self) -> None:
        poses = self.state.body_q.numpy()
        np.testing.assert_allclose(poses[:, 2], self.heights, atol=0.015)
        if not np.isfinite(poses).all():
            raise RuntimeError("The benchmark produced nonfinite body poses.")
        collision_bits = int(
            OverflowType.BROADPHASE | OverflowType.NARROWPHASE | OverflowType.CCD | OverflowType.EPA_HORIZON
        )
        if np.any(self.solver.mjw_data.overflow.numpy() & collision_bits):
            raise RuntimeError("The benchmark overflowed a collision buffer.")

    def track_median_step_ms(self, world_count: int, case: str) -> float:
        """Report median milliseconds per batched physics step after warmup."""
        timings = []
        with wp.ScopedDevice(self.device):
            for _ in range(self.samples):
                start = time.perf_counter()
                for _ in range(self.launch_count):
                    wp.capture_launch(self.graph)
                wp.synchronize_device(self.device)
                timings.append(1000.0 * (time.perf_counter() - start) / (self.launch_count * self.graph_steps))
        self._validate()
        return statistics.median(timings)

    track_median_step_ms.unit = "ms/step"

    def track_filter_scratch_mib(self, world_count: int, case: str) -> float:
        """Report owned compaction scratch, excluding lookup tables and scan workspace."""
        contact_filter = self.solver._heterogeneous_contact_filter
        if contact_filter is None:
            return 0.0
        arrays = [scratch for _, scratch in contact_filter._buffers]
        arrays.extend(
            [
                contact_filter._keep,
                contact_filter._offsets,
                contact_filter._original_count,
                contact_filter._rejected,
                contact_filter._fields,
            ]
        )
        return sum(array.capacity for array in arrays) / 1024**2

    track_filter_scratch_mib.unit = "MiB"


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--world-count", type=int, nargs="+", choices=HeterogeneousMuJoCo.params[0])
    parser.add_argument("--case", nargs="+", choices=HeterogeneousMuJoCo.params[1])
    args = parser.parse_args()
    HeterogeneousMuJoCo.params = (
        args.world_count or HeterogeneousMuJoCo.params[0],
        args.case or HeterogeneousMuJoCo.params[1],
    )
    run_benchmark(HeterogeneousMuJoCo)
