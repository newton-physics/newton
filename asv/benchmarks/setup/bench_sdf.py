# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

wp.config.enable_backward = False
wp.config.quiet = True

from asv_runner.benchmarks.mark import skip_benchmark_if

import newton

# Subdivision 4 yields 20 * 4**4 = 5120 triangles, which is representative of
# typical collision meshes used with SDF-based contact (YCB, nut-bolt, gears).
_SPHERE_SUBDIVISIONS = 4

# Number of SDFs built per timing sample, keyed by ``max_resolution``.
# Measuring a batch rather than a single build amortizes GPU boost-clock and
# thermal transients that otherwise make this benchmark bimodal across AWS CI
# runs (see #2534).  Counts decrease with resolution so each sample takes
# roughly the same wall time (~0.5 s); each SDF is released immediately after
# construction so peak GPU memory stays bounded to one SDF at a time.
_BUILDS_PER_SAMPLE = {
    32: 20,
    64: 20,
    128: 10,
    256: 5,
}

# Number of untimed warm-up builds in ``setup`` to push the GPU into a stable
# boost-clock state before any timed iterations run.
_WARMUP_BUILDS = 3


def _create_icosphere(radius: float, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Build an icosphere by subdividing an icosahedron.

    Returns:
        ``(vertices, indices)`` as ``float32`` and flat ``int32`` numpy arrays.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * radius

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    for _ in range(subdivisions):
        edge_midpoints: dict[tuple[int, int], int] = {}
        new_faces = np.empty((faces.shape[0] * 4, 3), dtype=np.int32)
        new_verts = [verts]
        next_idx = verts.shape[0]
        for fi, tri in enumerate(faces):
            mids = [0, 0, 0]
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                key = (a, b) if a < b else (b, a)
                mid = edge_midpoints.get(key)
                if mid is None:
                    m = (verts[a] + verts[b]) * 0.5
                    m = m / np.linalg.norm(m) * radius
                    mid = next_idx
                    edge_midpoints[key] = mid
                    new_verts.append(m[None, :].astype(np.float32))
                    next_idx += 1
                mids[i] = mid
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            m0, m1, m2 = mids
            new_faces[4 * fi + 0] = (a, m0, m2)
            new_faces[4 * fi + 1] = (b, m1, m0)
            new_faces[4 * fi + 2] = (c, m2, m1)
            new_faces[4 * fi + 3] = (m0, m1, m2)
        verts = np.concatenate(new_verts, axis=0)
        faces = new_faces

    return verts, faces.reshape(-1)


class FastBuildSdf:
    """Time ``Mesh.build_sdf`` across a range of grid resolutions.

    Uses an icosphere with ~5120 triangles (subdivision 4), representative of
    typical collision meshes used with Newton's SDF contact path.
    """

    params = ([32, 64, 128, 256],)
    param_names = ["max_resolution"]

    rounds = 2
    repeat = 5
    number = 1
    min_run_count = 1
    timeout = 600

    def setup(self, max_resolution):
        wp.init()
        if wp.get_cuda_device_count() == 0:
            # Matches the ``skip_benchmark_if`` guard on ``time_build_sdf``.
            return

        self._vertices, self._indices = _create_icosphere(radius=0.5, subdivisions=_SPHERE_SUBDIVISIONS)

        # Extended warmup: multiple builds push the GPU into a sustained
        # boost-clock state, not just compile kernels.
        for _ in range(_WARMUP_BUILDS):
            warm_mesh = newton.Mesh(self._vertices, self._indices, compute_inertia=False)
            warm_sdf = warm_mesh.build_sdf(max_resolution=max_resolution)
            del warm_sdf
            del warm_mesh
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_build_sdf(self, max_resolution):
        for _ in range(_BUILDS_PER_SAMPLE[max_resolution]):
            mesh = newton.Mesh(self._vertices, self._indices, compute_inertia=False)
            sdf = mesh.build_sdf(max_resolution=max_resolution)
            # Release immediately so peak GPU memory stays bounded.
            del sdf
            del mesh
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastBuildSdf": FastBuildSdf,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
