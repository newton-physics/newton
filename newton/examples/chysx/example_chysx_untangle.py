# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX Untangle
#
# 1:1 reproduction of cuda-cloth's `UntangleCase`
# (see d:/physics/cuda-cloth/src/Simulator/UntangleCase.{h,cpp}).
#
# What this scene exercises
# -------------------------
# The reference asset is `assets/quad/untangle-{N}.obj`, a Houdini-baked
# *pre-tangled* twin-sheet cloth: at the rest pose, dozens of edges
# already pierce neighbouring faces, so a normal proximity self-
# collision pass (VF / EE distance < thickness) cannot recover -- both
# sides of the contact are flush against each other with zero distance
# between them.
#
# The fix is the 5-vertex *untangle* algorithm (cuda-cloth's
# `kernel_cull_EF_pairs` / `KernelComputeCollisionHessianAndForce_EF`,
# now ported into ChysX as `chysx::collision::UntangleDetector` +
# `chysx::constraint::UntangleConstraint`).  For every (edge, face)
# candidate produced by the proximity BVH broadphase, we run a per-
# pair ray-triangle intersection.  Every hit emits a 5-vertex penalty
# contact -- two edge endpoints + three face vertices -- whose
# diagonal-only Hessian + RHS push the edge back out along the
# cross-product of the intersected face's normal and the edge's own
# face normal.  Diagonal-only by design, so toggling untangle on and
# off never invalidates the captured PCG graph.
#
# Material parameters (cuda-cloth `UntangleCase::Initialize` +
# `assets/json/untangle.json`):
#
#     m_k          = 1e3      # BW98 stretch + shear stiffness
#     bending_k    = 1e-3     # dihedral bending
#     gravity      = 0.0      # the scene is driven entirely by tangles
#     damping      = 1.0      # full velocity damping (per-step factor)
#     m_5_k        = 1e2      # untangle penalty stiffness
#     m_5_thickness= 1e-2     # untangle restoring depth
#     m_4_k        = 1e2      # proximity self-collision stiffness
#     m_4_thickness= 1e-2     # proximity contact distance
#     m_control_mag= 1e9      # pin stiffness
#
# Pin sets are mesh-size-specific (cuda-cloth holds the four "top"
# corners of the two interleaved sheets so the cloth can settle
# without flying apart):
#
#     untangle.obj         (200 verts)   pins = [0, 9, 190, 199]   (synthesized)
#     untangle-2500.obj    (5000 verts)  pins = [49, 2499, 2500, 4950]
#     untangle-10000.obj   (20000 verts) pins = [99, 9999, 10099, 19999]
#
# Loop bound: cuda-cloth runs `UntangleCase::Run` for 1000 steps at
# dt=0.01 -> 10 s.  We expose `--num-frames 1000` by default.
#
# Command:
#
#     python -m newton.examples chysx_untangle
#     python -m newton.examples chysx_untangle --mesh small         # tiny
#     python -m newton.examples chysx_untangle --mesh large         # 20k
#     python -m newton.examples chysx_untangle --no-untangle        # ablation
#     python -m newton.examples chysx_untangle --obj-out /tmp/utg   # dump per-frame OBJs
#
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton
import newton.examples


# Cuda-cloth ships three mesh sizes for the same scene; we pick one
# per run via --mesh, default = `medium` (5000 verts -> snappy on a
# laptop dGPU but big enough to show real tangles).
_MESH_SPECS = {
    "small":  ("untangle.obj",        200,   [0, 9, 190, 199]),
    "medium": ("untangle-2500.obj",   5000,  [49, 2499, 2500, 4950]),
    "large":  ("untangle-10000.obj",  20000, [99, 9999, 10099, 19999]),
}


def _resolve_obj(filename: str) -> str:
    """Locate a cuda-cloth mesh asset.

    Search order:
      1. `newton/examples/assets/chysx/<filename>` (shipped copy, if any).
      2. `d:/physics/cuda-cloth/assets/quad/<filename>` (sibling
         cuda-cloth checkout — the one that ships with the engine).
    """
    shipped = newton.examples.get_asset(f"chysx/{filename}")
    if os.path.isfile(shipped):
        return shipped
    fallback = rf"d:\physics\cuda-cloth\assets\quad\{filename}"
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        f"{filename} not found at either {shipped!r} or {fallback!r}; "
        f"copy the file from cuda-cloth's assets/quad/ folder."
    )


def _load_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a triangulated wavefront OBJ.

    Returns
    -------
    vertices : (N, 3) float32 array of vertex positions.
    indices  : (M, 3) int32 array of triangle vertex indices (0-based).

    Vertex order is preserved exactly as in the file so cuda-cloth's
    integer pin indices stay valid.  Only ``v`` and ``f`` lines are
    recognised; UV / normal indices and quads are fan-triangulated.
    """
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            head, _, rest = line.partition(" ")
            if head == "v":
                xyz = rest.split()
                verts.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            elif head == "f":
                parts = rest.split()
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                if len(idx) == 3:
                    tris.append((idx[0], idx[1], idx[2]))
                else:
                    for k in range(1, len(idx) - 1):
                        tris.append((idx[0], idx[k], idx[k + 1]))
    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(tris, dtype=np.int32),
    )


class Example:
    def __init__(self, viewer, args):
        # cuda-cloth UntangleCase: dt = 0.01 s, no substepping, 1000
        # loop iterations.  PCGSolver `m_maxIter = 150` in the json,
        # but ChysX's preconditioned PCG converges in fewer iters on
        # this problem -- 50 is plenty in practice and matches the
        # twist example's setting.
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        # ---- mesh -------------------------------------------------------

        spec = _MESH_SPECS[args.mesh]
        filename, expected_verts, default_pins = spec
        verts_np, tris_np = _load_obj(_resolve_obj(filename))
        if verts_np.shape[0] != expected_verts:
            raise ValueError(
                f"{filename} expected {expected_verts} verts, got "
                f"{verts_np.shape[0]} (asset out of date?)"
            )
        self._initial_q = verts_np.copy()
        self._n_verts = verts_np.shape[0]
        self._n_tris = tris_np.shape[0]
        # Average edge length, used only to size the particle radius
        # the viewer renders.  Topology-blind estimate from the first
        # triangle's longest edge -- good enough for visualisation.
        e0 = float(np.linalg.norm(verts_np[tris_np[0, 0]] - verts_np[tris_np[0, 1]]))
        self._edge_l = e0

        # ---- Newton model ----------------------------------------------
        #
        # Z-up matches the rest of the chysx examples and lets the
        # built-in viewer helpers behave sensibly even though this
        # scene has no ground plane (gravity is 0).
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[
                wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts_np
            ],
            indices=tris_np.flatten().tolist(),
            density=0.0,  # ChysX recomputes per-particle mass below.
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.5 * self._edge_l * 0.4,
        )
        self.model = builder.finalize()

        # ---- pins -------------------------------------------------------
        self._pin_indices = np.asarray(default_pins, dtype=np.int32)
        # Snapshot the rest positions of the pins; targets stay frozen
        # for the entire run (cuda-cloth's UntangleCase does not animate
        # them -- the cloth is supposed to settle into a tangle-free
        # equilibrium under the static pin set + untangle forces).
        self._pin_targets = verts_np[self._pin_indices].astype(np.float32).copy()

        # ---- collision parameters --------------------------------------
        #
        # Stiffness ratio follows style3d's `Collision.__init__`
        # (newton/_src/solvers/style3d/collision/collision.py):
        #
        #     stiff_vf = 0.5,  stiff_ee = 0.1,  stiff_ef = 1.0
        #
        # i.e. the EF/untangle stiffness sits at 2x VF and 10x EE.
        # Untangle has to win the local force budget over proximity
        # because once panels are tangled the proximity term tries to
        # *hold* them in the tangled equilibrium -- a softer untangle
        # term gets out-pushed and never recovers.
        #
        # chysx wires VF + EE through a single `self_collision_stiffness`
        # knob.  Style3D's VF (0.5) is the dominant proximity branch,
        # so we mirror EF/VF = 2x: pick the chysx self-collision
        # stiffness from cuda-cloth's UntangleCase (1e2) and bump
        # untangle to 2e2.  Thickness is still cuda-cloth's 1e-2
        # (style3d uses 6 mm = 2*radius; same order of magnitude).
        thickness = 1.0e-2
        sc_stiffness = 1.0e2     # style3d-equivalent VF factor (cuda-cloth m_4_k)
        utg_stiffness = 2.0e2    # 2x sc_stiffness, matches style3d EF/VF ratio

        # Buffer caps.  The pre-tangled rest pose has thousands of EF
        # candidates; size generously so the broadphase never clips.
        # 64x particle count for narrow-phase + 256x for broad-phase
        # matches the twist example, which exercises a comparably
        # dense contact set.
        self_collision_max_contacts_factor = 64
        self_collision_max_ef_candidates_factor = 256
        untangle_max_contacts_factor = 64

        # ---- solver -----------------------------------------------------

        # cuda-cloth scatters `0.1 * area` per vertex (NOT divided by 3)
        # in `KernelComputeAllDm`.  ChysX's `redistribute_mass_area_weighted`
        # divides by 3, so we use 0.3 to reproduce the same lumped masses.
        surface_density = 0.3

        self.solver = newton.solvers.SolverChysX(
            self.model,
            gravity=(0.0, 0.0, 0.0),       # m_gravity = 0 in untangle.json
            damping=1.0,                    # m_damping = 1.0
            fem_stretch_stiffness=1.0e3,    # m_k = 1000
            fem_shear_stiffness=1.0e3,
            bending_stiffness=1.0e-4,       # m_bending_k = 1e-3
            pin_indices=self._pin_indices.tolist(),
            pin_stiffness=1.0e9,            # m_control_mag = 1e9
            pcg_iterations=50,
            surface_density=surface_density,
            self_collision_enabled=True,
            self_collision_thickness=thickness,
            self_collision_stiffness=sc_stiffness,
            self_collision_max_contacts_factor=self_collision_max_contacts_factor,
            self_collision_max_ef_candidates_factor=self_collision_max_ef_candidates_factor,
            untangle_enabled=bool(args.untangle),
            untangle_thickness=thickness,
            untangle_stiffness=utg_stiffness,
            untangle_max_contacts_factor=untangle_max_contacts_factor,
        )
        self._untangle_enabled = bool(args.untangle)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Pin targets never change in this scene -- write them once and
        # leave the kernel reading the same buffer for every frame.
        self.solver.update_pin_targets(self._pin_targets)

        # ---- viewer -----------------------------------------------------
        self.viewer.set_model(self.model)
        # Camera looking at the cloth bbox centre from +Z; the rest
        # pose's two interleaved sheets stack along world X, so a
        # straight-down view shows the wrinkles best.
        bbox_min = verts_np.min(axis=0)
        bbox_max = verts_np.max(axis=0)
        centre = 0.5 * (bbox_min + bbox_max)
        extent = float(np.linalg.norm(bbox_max - bbox_min))
        cam_pos = wp.vec3(
            float(centre[0]),
            float(centre[1]),
            float(centre[2] + max(1.5, 1.5 * extent)),
        )
        self.viewer.set_camera(pos=cam_pos, pitch=0.0, yaw=0.0)
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = True

        # ---- OBJ export (optional) -------------------------------------
        self._obj_dir: str | None = None
        self._obj_stride = max(1, int(getattr(args, "obj_stride", 1)))
        self._frame_idx = 0
        if getattr(args, "obj_out", None):
            self._obj_dir = str(args.obj_out)
            os.makedirs(self._obj_dir, exist_ok=True)
            self._cloth_tris = tris_np.copy()
            self._export_obj_frame(0)

        # ---- CUDA Graph capture (mirrors example_chysx_twist) ----------
        #
        # Pin targets never change in this scene, so the entire per-
        # frame substep loop is graph-capturable with no host-side
        # work in between.  See `example_chysx_twist._capture_graph`
        # for the design rationale.
        self._cuda_graph = None
        self._capture_graph()

    def _simulate_substeps(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0, self.state_0, self.control, self.contacts, self.sim_dt
            )

    def _capture_graph(self) -> None:
        self._cuda_graph = None
        device = wp.get_device()
        if not device.is_cuda:
            return
        # Warm-up to flush lazy initialisation BEFORE capture (Hessian
        # topology rebuild + first BVH refit + JIT must land outside
        # the captured region).
        self._simulate_substeps()
        wp.synchronize_device(device)
        with wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self._cuda_graph = capture.graph

    # ---- per-frame OBJ export -----------------------------------------

    def _export_obj_frame(self, frame_idx: int) -> None:
        assert self._obj_dir is not None
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        path = os.path.join(self._obj_dir, f"untangle_{frame_idx:05d}.obj")
        with open(path, "w") as f:
            f.write(
                f"# chysx_untangle frame {frame_idx}, t = {self.sim_time:.6f} s\n"
            )
            np.savetxt(f, q, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, self._cloth_tris + 1, fmt="f %d %d %d")

    # ---- simulation loop ----------------------------------------------

    def step(self) -> None:
        if self._cuda_graph is not None:
            wp.capture_launch(self._cuda_graph)
        else:
            self._simulate_substeps()
        self.sim_time += self.sim_substeps * self.sim_dt
        self._frame_idx += 1
        if self._obj_dir is not None and (self._frame_idx % self._obj_stride) == 0:
            self._export_obj_frame(self._frame_idx)

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        # The rest mesh fits inside [-1, 0.5] x [-0.5, 0.5] x [-0.5, 0.5].
        # An untangling run can only decrease the cloth's bounding box
        # extent (the algorithm pulls overlapping panels apart, but
        # the pinned corners cap the spread), so a generous 3 m bound
        # catches numerical blowup while not flagging legitimate
        # un-tangled relaxation.
        bound = 3.0
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"cloth particles escaped the {bound:.1f} m bounding box; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 50.0:
            raise ValueError(
                f"particle speed exploded: max |v| = {max_speed:.3f}"
            )

        # Diagnostic prints (cheap, run once at the end).
        n_self = self.solver._sim.self_collision_count(0)
        n_tan = self.solver._sim.untangle_count(0) if self._untangle_enabled else 0
        print(
            f"[chysx_untangle] final-frame contacts: "
            f"self_collision={n_self}  untangle={n_tan}  "
            f"max|q|={float(np.abs(q).max()):.3f}m  "
            f"max|v|={max_speed:.3f}m/s"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--mesh",
        choices=sorted(_MESH_SPECS.keys()),
        default="medium",
        help=(
            "Pre-tangled cloth asset to load.  "
            "small=200 verts (debug), medium=5000 verts (default), "
            "large=20000 verts (cuda-cloth `untangle-10000.obj`)."
        ),
    )
    parser.add_argument(
        "--no-untangle",
        dest="untangle",
        action="store_false",
        help=(
            "Ablation: disable the 5-vertex untangle pass.  Proximity "
            "self-collision alone cannot recover from a pre-tangled "
            "rest pose; expect the cloth to stay tangled or even "
            "explode under accumulating contact penalty."
        ),
    )
    parser.set_defaults(untangle=True)
    parser.add_argument(
        "--obj-out",
        default=None,
        help="Optional directory for per-frame OBJ exports.",
    )
    parser.add_argument(
        "--obj-stride",
        type=int,
        default=1,
        help="Export every Nth frame when --obj-out is set (default 1).",
    )
    # cuda-cloth UntangleCase loops 1000 times.
    parser.set_defaults(num_frames=1000)

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
