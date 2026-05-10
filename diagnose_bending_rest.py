"""Bending-rest-angle diagnostic.

Loads the unisex T-shirt USD asset, installs the mesh into chysx with
ALL contact disabled (no self-collision, no ground, no table), pins a
single vertex at the top of the garment, and lets the rest of it
free-fall.

Expected behaviour
------------------

If bending rest angles are preserved correctly, the garment should
hang from the pinned vertex and keep its sculpted 3-D shape (modulo
the small downward elongation gravity adds against the FEM stretch
spring).  The front and back panels should stay roughly 27 cm apart.

If bending rest angles default to 0 (i.e. the constraint thinks every
interior edge wants to be flat), gravity + bending will collapse the
garment into a flat sheet hanging from the pin.

Usage:  uv run python diagnose_bending_rest.py
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.solvers


def load_centered_tshirt_m() -> tuple[np.ndarray, np.ndarray, float]:
    """Same as example_chysx_tshirt_drop._load_centered_tshirt_m()."""
    stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
    prim = stage.GetPrimAtPath("/root/shirt")
    m = newton.usd.get_mesh(prim)

    v_cm = np.asarray(m.vertices, dtype=np.float32)
    idx = np.asarray(m.indices, dtype=np.int32).reshape(-1, 3)

    v_m = v_cm * 0.01  # cm -> m
    centre = 0.5 * (v_m.min(axis=0) + v_m.max(axis=0))
    v_m = v_m - centre

    e = np.concatenate(
        [
            np.linalg.norm(v_m[idx[:, a]] - v_m[idx[:, b]], axis=1)
            for (a, b) in [(0, 1), (1, 2), (2, 0)]
        ]
    )
    edge_med = float(np.median(e))
    return v_m, idx, edge_med


def main():
    # ---- build the model ------------------------------------------------
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

    verts, tris, edge_med = load_centered_tshirt_m()
    print(f"T-shirt mesh: {verts.shape[0]} verts, {tris.shape[0]} tris, edge_med={edge_med:.4f} m")
    print(f"  bbox: x[{verts[:,0].min():.3f},{verts[:,0].max():.3f}] "
          f"y[{verts[:,1].min():.3f},{verts[:,1].max():.3f}] "
          f"z[{verts[:,2].min():.3f},{verts[:,2].max():.3f}]")
    z_extent = float(verts[:, 2].max() - verts[:, 2].min())
    print(f"  initial front-back panel separation (z extent): {z_extent:.3f} m")

    # Hang the garment in the air (no ground, no table).  The pin
    # vertex is the one with the largest y (top of the shoulder area
    # in the T-shirt asset's body-vertical direction).
    drop_height = 1.0
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, drop_height),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts],
        indices=tris.flatten().tolist(),
        density=0.0,
        tri_ke=0.0, tri_ka=0.0, tri_kd=0.0,
        edge_ke=0.0, edge_kd=0.0,
        particle_radius=0.4 * edge_med,
    )

    model = builder.finalize()

    # Pick the highest-z particle as the pin (keeps the garment
    # hanging upright during the free fall).
    q0 = model.particle_q.numpy().reshape(-1, 3)
    pin_idx = int(np.argmax(q0[:, 2]))
    print(f"\nPin vertex: particle {pin_idx} at {q0[pin_idx]}")

    # ---- solver: ALL contact disabled -----------------------------------
    solver = newton.solvers.SolverChysX(
        model,
        damping=0.05,
        fem_stretch_stiffness=5.0e2,
        fem_shear_stiffness=5.0e2,
        bending_stiffness=5.0e-4,
        pcg_iterations=50,
        surface_density=0.3,
        pin_indices=[pin_idx],
        pin_stiffness=1.0e9,
        # Everything off so we isolate bending behaviour
        self_collision_enabled=False,
        static_contact_enabled=False,
    )

    # ---- inspect rest angles immediately --------------------------------
    n_dihedrals = solver._sim.num_bending_dihedrals()
    print(f"\nBending dihedrals installed: {n_dihedrals}")
    if n_dihedrals == 0:
        print("WARN: no bending dihedrals installed -- nothing to test.")
        return

    # ---- run --------------------------------------------------------------
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    fps = 60
    sim_substeps = 10
    sim_dt = (1.0 / fps) / sim_substeps

    print("\nframe |   t   |   z range          |  z extent (panel sep)  | vmax")
    print("-" * 75)

    z_extent_t0 = float(state_0.particle_q.numpy().reshape(-1, 3)[:, 2].ptp() if hasattr(np.ndarray, 'ptp') else q0[:,2].max() - q0[:,2].min())
    z_min0 = q0[:, 2].min()
    z_max0 = q0[:, 2].max()
    z_extent_t0 = z_max0 - z_min0
    print(f"  -1  | 0.00  | [{z_min0:.3f}, {z_max0:.3f}] |  {z_extent_t0:.3f} m  (initial)  |   --")

    t0 = time.time()
    for f in range(180):
        for _ in range(sim_substeps):
            state_0.clear_forces()
            solver.step(state_0, state_0, control, contacts, sim_dt)

        if f in (0, 5, 15, 30, 60, 120, 179):
            q = state_0.particle_q.numpy().reshape(-1, 3)
            v = state_0.particle_qd.numpy().reshape(-1, 3)
            zmin, zmax = float(q[:, 2].min()), float(q[:, 2].max())
            z_extent = zmax - zmin
            x_extent = float(q[:, 0].max() - q[:, 0].min())
            y_extent = float(q[:, 1].max() - q[:, 1].min())
            vmax = float(np.linalg.norm(v, axis=1).max())
            print(f"{f:>5} | {f/fps:>5.2f} | [{zmin:.3f}, {zmax:.3f}] |  {z_extent:.3f} m  (xy: {x_extent:.2f} x {y_extent:.2f})  | {vmax:.3f}")

    print(f"\nwall: {time.time()-t0:.2f} s")
    print("\nDiagnosis:")
    q_final = state_0.particle_q.numpy().reshape(-1, 3)
    z_extent_final = float(q_final[:, 2].max() - q_final[:, 2].min())
    print(f"  initial panel sep   : {z_extent_t0:.3f} m  (should be ~0.27)")
    print(f"  final panel sep (z) : {z_extent_final:.3f} m")
    if z_extent_final > 0.5 * z_extent_t0:
        print("  -> bending rest angles APPEAR PRESERVED (panels still separated)")
    else:
        print("  -> bending rest angles APPEAR LOST (cloth collapsed flat)")


if __name__ == "__main__":
    main()
