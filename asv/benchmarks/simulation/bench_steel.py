# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Steel contact-friction benchmarks: captured steps, patch construction, pair updates and torsion preparation.

None of these are in the PR gate; run them with e.g. ``uvx --with virtualenv asv run --bench SteelFriction``.
"""

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

from newton._src.solvers.steel.friction import FRICTION_PAIR_CUDA

# Each benchmark body is a fixed 0.2 m cube of 1 kg built from tiles x tiles touching boxes, so the
# contact count grows with tiles while the footprint, mass and load stay the same.
_TILED_BODY_HALF_EXTENT = 0.1
_TILED_BODY_LOAD = (1.0, 0.0, 0.0, 0.0, 0.0, 0.03)


def _require_cuda_graphs():
    device = wp.get_device("cuda:0") if wp.get_cuda_device_count() else None
    if device is None or not wp.is_mempool_enabled(device):
        raise SkipNotImplemented


class _TiledBodyScene:
    """Tiled boxes pushed sideways and twisted on a ground plane, simulated with SolverSteel."""

    def __init__(self, world_count, tiles, friction_anchor_beta=None):
        import newton  # noqa: PLC0415

        h = _TILED_BODY_HALF_EXTENT / tiles
        template = newton.ModelBuilder()
        template.add_ground_plane()
        body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity()))
        cfg = newton.ModelBuilder.ShapeConfig(density=125.0, mu=0.5)
        for x in range(tiles):
            for y in range(tiles):
                center = wp.vec3(
                    -_TILED_BODY_HALF_EXTENT + h + 2 * h * x, -_TILED_BODY_HALF_EXTENT + h + 2 * h * y, 0.0
                )
                template.add_shape_box(
                    body, hx=h, hy=h, hz=0.1, xform=wp.transform(center, wp.quat_identity()), cfg=cfg
                )
        builder = newton.ModelBuilder()
        builder.replicate(template, world_count)
        self.model = builder.finalize(device="cuda:0")
        self.model.rigid_contact_max = world_count * tiles * tiles * 4
        options = {} if friction_anchor_beta is None else {"friction_anchor_beta": friction_anchor_beta}
        self.solver = newton.solvers.SolverSteel(
            self.model,
            iterations=32,
            baumgarte=0.05,
            articulation_constraint_max=32,
            free_body_constraint_max=max(256, 12 * tiles * tiles),
            **options,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model, rigid_contact_max=self.model.rigid_contact_max, broad_phase="nxn", contact_matching="latest"
        )
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.load = wp.full(self.model.body_count, wp.spatial_vector(*_TILED_BODY_LOAD), device=self.model.device)
        self.dt = 0.005

    def step(self, source, dest):
        wp.copy(source.body_f, self.load)
        self.pipeline.collide(source, self.contacts)
        self.solver.step(source, dest, self.control, self.contacts, self.dt)

    def step_pair(self):
        self.step(self.state_0, self.state_1)
        self.step(self.state_1, self.state_0)

    def check_finite(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise RuntimeError("SolverSteel produced non-finite body poses")


class SteelFrictionPatches:
    """100 captured collide-and-solve steps with persistent friction patches or point friction.

    ``tiles=1`` gives four contacts per body; ``tiles=4`` gives 64 contacts in one patch per body.
    """

    params = [[1, 4], ["patch", "point"]]
    param_names = ["tiles", "friction"]
    number = 1
    rounds = 2
    repeat = 5
    warmup_time = 0
    timeout = 600
    world_count = 64
    steps = 100

    def setup(self, tiles, friction):
        _require_cuda_graphs()
        # ASV calls setup before each sample; keep the warmed scene and its graph.
        if getattr(self, "_key", None) == (tiles, friction):
            return
        self.scene = _TiledBodyScene(self.world_count, tiles, None if friction == "patch" else 0.0)
        for _ in range(10):
            self.scene.step_pair()
        with wp.ScopedCapture(device=self.scene.model.device) as capture:
            self.scene.step_pair()
        self.graph = capture.graph
        wp.synchronize_device(self.scene.model.device)
        self._key = (tiles, friction)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self, tiles, friction):
        for _ in range(self.steps // 2):
            wp.capture_launch(self.graph)
        wp.synchronize_device(self.scene.model.device)

    def teardown(self, tiles, friction):
        self.scene.check_finite()


class SteelFrictionPatchBuild:
    """20 captured friction-patch constructions on fixed contacts, without collision or solving.

    ``tiles=13`` puts 169 box shapes on one body, as a stress case for patch grouping.
    """

    params = [[4, 13], [1, 64]]
    param_names = ["tiles", "world_count"]
    number = 1
    rounds = 2
    repeat = 5
    warmup_time = 0
    timeout = 600
    launches = 20

    def setup(self, tiles, world_count):
        _require_cuda_graphs()
        if getattr(self, "_key", None) == (tiles, world_count):
            return
        scene = _TiledBodyScene(world_count, tiles)
        for _ in range(10):
            scene.step_pair()
        solver = scene.solver
        patches = solver._friction_patches

        def build():
            patches.build(
                scene.model,
                scene.state_0,
                scene.contacts,
                shape_material_mu=solver._shape_mu,
                contact_gap_limit=solver._contact_gap_limit,
                friction_gap_limit=solver._friction_gap,
                dt=scene.dt,
            )

        build()
        with wp.ScopedCapture(device=scene.model.device) as capture:
            for _ in range(self.launches):
                build()
        self.graph = capture.graph
        self.scene = scene
        wp.synchronize_device(scene.model.device)
        self._key = (tiles, world_count)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_build(self, tiles, world_count):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.scene.model.device)


@wp.func_native(
    "#if defined(__CUDA_ARCH__)\n"
    + FRICTION_PAIR_CUDA
    + """
    const float2 pair = friction_pair_candidate(a, c, d, r0, r1, old0, old1, 1.0f, 1.0f);
    const float length = sqrtf(pair.x * pair.x + pair.y * pair.y);
    const float scale = length > 1.0f ? 1.0f / length : 1.0f;
    out.data[2 * i] = pair.x * scale;
    out.data[2 * i + 1] = pair.y * scale;
#endif
"""
)
def _friction_pair_cuda(
    a: float, c: float, d: float, r0: float, r1: float, old0: float, old1: float, i: int, out: wp.array[float]
): ...


@wp.kernel
def _friction_pair_update(
    blocks: wp.array[wp.vec3], residuals: wp.array[wp.vec2], old: wp.array[wp.vec2], out: wp.array[float]
):
    i = wp.tid()
    a = blocks[i]
    r = residuals[i]
    o = old[i]
    _friction_pair_cuda(a[0], a[1], a[2], r[0], r[1], o[0], o[1], i, out)


class SteelFrictionPair:
    """20 captured launches of the CUDA coupled two-row tangent update on 65536 independent 2x2 blocks.

    It runs ``FRICTION_PAIR_CUDA``, the snippet the CUDA solves splice in. The blocks are positive
    definite with condition numbers from 1 to 8 and a unit friction radius; ``sliding`` targets lie
    outside the radius and ``sticking`` targets inside it.
    """

    params = [["sticking", "sliding", "mixed"]]
    param_names = ["case"]
    number = 1
    rounds = 2
    repeat = 5
    warmup_time = 0
    timeout = 300
    pairs = 65536
    launches = 20

    def setup(self, case):
        _require_cuda_graphs()
        rng = np.random.default_rng(102)
        n = self.pairs
        angle = rng.uniform(-np.pi, np.pi, n)
        cosine, sine = np.cos(angle), np.sin(angle)
        eigenvalue = rng.uniform(1.0, 8.0, n)
        blocks = np.column_stack(
            (cosine**2 + eigenvalue * sine**2, (1 - eigenvalue) * cosine * sine, sine**2 + eigenvalue * cosine**2)
        ).astype(np.float32)
        old = rng.uniform(-0.15, 0.15, (n, 2)).astype(np.float32)
        direction = rng.uniform(-np.pi, np.pi, n)
        sliding = np.full(n, case == "sliding") if case != "mixed" else rng.random(n) < 0.5
        target = np.column_stack((np.cos(direction), np.sin(direction))) * np.where(sliding, 2.0, 0.25)[:, None]
        self.sliding, self.target = sliding, target
        delta = old - target
        residuals = np.column_stack(
            (
                blocks[:, 0] * delta[:, 0] + blocks[:, 1] * delta[:, 1],
                blocks[:, 1] * delta[:, 0] + blocks[:, 2] * delta[:, 1],
            )
        ).astype(np.float32)
        with wp.ScopedDevice("cuda:0"):
            inputs = [wp.array(blocks, dtype=wp.vec3), wp.array(residuals, dtype=wp.vec2), wp.array(old, dtype=wp.vec2)]
            self.out = wp.empty(2 * n, dtype=float)
            wp.launch(_friction_pair_update, dim=n, inputs=inputs, outputs=[self.out])
            with wp.ScopedCapture() as capture:
                for _ in range(self.launches):
                    wp.launch(_friction_pair_update, dim=n, inputs=inputs, outputs=[self.out])
            self.graph = capture.graph
            wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_pair_update(self, case):
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")

    def teardown(self, case):
        out = self.out.numpy().reshape(-1, 2)
        # Sliding ends on the unit disk's rim; sticking reaches its target inside the disk.
        rim_error = np.abs(np.linalg.norm(out[self.sliding], axis=1) - 1.0)
        target_error = np.abs(out[~self.sliding] - self.target[~self.sliding])
        if not (np.isfinite(out).all() and np.all(rim_error < 1.0e-4) and np.all(target_error < 1.0e-4)):
            raise RuntimeError("Friction pair update missed the friction disk solution")


class SteelContactTorsionPreparation:
    """Captured device preparation of contact torsion rows for one contact-dense world, including validation.

    Uses the synthetic contact tables of the Steel torsion tests; the solver's row buffers are restored
    before every sample, outside the timed region.
    """

    params = [[64, 1024, 4096], [True, False]]
    param_names = ["contacts", "patches"]
    number = 1
    rounds = 2
    repeat = 5
    warmup_time = 0
    timeout = 600

    def setup(self, contacts, patches):
        _require_cuda_graphs()
        if getattr(self, "_key", None) != (contacts, patches):
            from newton._src.solvers.steel.contact_torsion_device import DeviceTorsionPreparation  # noqa: PLC0415
            from newton.tests.steel.test_torsion import synthetic_fixture  # noqa: PLC0415

            solver, state, augmented, contact_tables = synthetic_fixture(
                worlds=1, witnesses=contacts, patches=patches, device="cuda:0"
            )
            names = (
                "_constraint_count",
                "_slot_counter",
                "_row_type",
                "_row_parent",
                "_row_mu",
                "_phi",
                "_target_velocity",
                "_row_rebound_velocity",
                "_contact_torsion_group",
            )
            targets = [getattr(solver, name) for name in names] + list(solver._J_by_size.values())
            self._saved = [(target, wp.clone(target)) for target in targets]
            self._preparer = DeviceTorsionPreparation(solver, deferred_errors=True)
            self._restore()
            self._preparer.prepare(state, augmented, contact_tables)
            self._restore()
            with wp.ScopedCapture(device="cuda:0") as capture:
                self._preparer.prepare(state, augmented, contact_tables)
            self.graph = capture.graph
            self._key = (contacts, patches)
        self._restore()
        wp.synchronize_device("cuda:0")

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_prepare(self, contacts, patches):
        wp.capture_launch(self.graph)
        self._preparer.validate()

    def _restore(self):
        for target, saved in self._saved:
            wp.copy(target, saved)


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "SteelFrictionPatches": SteelFrictionPatches,
        "SteelFrictionPatchBuild": SteelFrictionPatchBuild,
        "SteelFrictionPair": SteelFrictionPair,
        "SteelContactTorsionPreparation": SteelContactTorsionPreparation,
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
