# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Correctness-matrix tests for the force-based conveyor model.

Exercises :class:`newton.examples.basic.conveyor_force_model.ConveyorForceModel`
across solver backends with quantitative (not just validity) checks: zero speed
(no drift), constant/reversed speed (direction and convergence), rotated belt,
curved (pivot) velocity field, contact loss, and long-running stability.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.basic.conveyor_force_model import ConveyorForceModel
from newton.tests.unittest_utils import add_function_test, get_test_devices

BELT_HALF_X = 1.5
BELT_HALF_Y = 6.0
BELT_HALF_Z = 0.1
BELT_TOP_Z = 0.5
BOX_HALF = 0.2
CONTACT_FRICTION = 2.0e-5


def _make_solver(solver_name, model):
    if solver_name == "vbd":
        return newton.solvers.SolverVBD(model, iterations=5, rigid_body_contact_buffer_size=512)
    if solver_name == "mujoco":
        # MuJoCo configuration for Newton-generated contacts.
        return newton.solvers.SolverMuJoCo(
            model, cone="elliptic", use_mujoco_contacts=False, njmax=200, nconmax=100
        )
    return newton.solvers.SolverXPBD(model)


def run_conveyor(
    device,
    solver_name,
    *,
    velocity=None,
    pivot_point=None,
    angular_velocity=None,
    box_xy=(0.0, 0.0),
    box_mass=1.0,
    belt_half_x=BELT_HALF_X,
    belt_half_y=BELT_HALF_Y,
    belt_rotation=None,
    frames=120,
    substeps=8,
    fps=60,
    collect_diagnostics=False,
):
    """Build a single-box-on-flat-belt scene, run it, and return the box position history.

    Returns an array of shape ``(frames + 1, 3)`` of the box world position per frame
    (index 0 is the initial pose).
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    belt_cfg = newton.ModelBuilder.ShapeConfig(mu=CONTACT_FRICTION, ke=1.0e5, kd=0.0)
    if belt_rotation is None:
        belt_rotation = wp.quat_identity()
    belt_shape = builder.add_shape_box(
        body=-1,
        hx=belt_half_x,
        hy=belt_half_y,
        hz=BELT_HALF_Z,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, BELT_TOP_Z - BELT_HALF_Z), q=belt_rotation),
        cfg=belt_cfg,
    )

    box_body = builder.add_link(
        xform=wp.transform(p=wp.vec3(box_xy[0], box_xy[1], BELT_TOP_Z + BOX_HALF + 0.01), q=wp.quat_identity()),
        mass=box_mass,
        label="box",
    )
    builder.add_shape_box(
        box_body,
        hx=BOX_HALF,
        hy=BOX_HALF,
        hz=BOX_HALF,
        cfg=newton.ModelBuilder.ShapeConfig(mu=CONTACT_FRICTION, restitution=0.0),
    )
    builder.add_articulation([builder.add_joint_free(box_body)], label="box")
    builder.color()

    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    solver = _make_solver(solver_name, model)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    conveyor = ConveyorForceModel(model, solver_type=solver_name)
    if pivot_point is not None:
        conveyor.add_pivot_belt(belt_shape, pivot_point=pivot_point, angular_velocity=angular_velocity)
    else:
        conveyor.add_constant_belt(belt_shape, velocity=velocity if velocity is not None else wp.vec3(0.0, 0.0, 0.0))
    conveyor.finalize(contacts)

    sim_dt = 1.0 / fps / substeps
    positions = np.zeros((frames + 1, 3), dtype=np.float32)
    positions[0] = state_0.body_q.numpy()[box_body][:3]
    if collect_diagnostics:
        sample_count = frames * substeps
        belt_contact_counts = np.zeros(sample_count, dtype=np.int32)
        conveyor_force_norms = np.zeros(sample_count, dtype=np.float32)
        sample = 0

    for f in range(frames):
        for _ in range(substeps):
            state_0.clear_forces()
            conveyor.apply(state_0)
            conveyor.snapshot_prev(state_0)
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            conveyor.update(solver, contacts, state_1, sim_dt)
            if collect_diagnostics:
                belt_contact_counts[sample] = conveyor.body_contact_count.numpy()[box_body]
                conveyor_force_norms[sample] = np.linalg.norm(conveyor.conveyor_body_f.numpy()[box_body][:3])
                sample += 1
            state_0, state_1 = state_1, state_0
        positions[f + 1] = state_0.body_q.numpy()[box_body][:3]

    if collect_diagnostics:
        count = int(contacts.rigid_contact_count.numpy()[0])
        diagnostics = {
            "belt_contact_counts": belt_contact_counts,
            "conveyor_force_norms": conveyor_force_norms,
            "contact_forces": conveyor.contact_force_vec.numpy()[:count],
            "contact_normals": contacts.rigid_contact_normal.numpy()[:count],
            "contact_points0": contacts.rigid_contact_point0.numpy()[:count],
            "contact_points1": contacts.rigid_contact_point1.numpy()[:count],
            "contact_shape0": contacts.rigid_contact_shape0.numpy()[:count],
            "contact_shape1": contacts.rigid_contact_shape1.numpy()[:count],
            "belt_shape": belt_shape,
            "conveyor_force": conveyor.conveyor_body_f.numpy()[box_body][:3],
        }
        return positions, 1.0 / fps, diagnostics

    return positions, 1.0 / fps


def _final_velocity(positions, frame_dt, n=30):
    """Average velocity over the last ``n`` frames."""
    seg = positions[-n - 1 :]
    return (seg[-1] - seg[0]) / (n * frame_dt)


def _yaw_deg(quat):
    """Yaw angle [deg] about +Z from an (x, y, z, w) quaternion."""
    x, y, z, w = quat
    return np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def run_multi_belt(device, solver_name, belts, box_xy, *, box_half=(0.45, 0.2, 0.2), frames=150, substeps=8, fps=60):
    """Run a box over multiple static belts.

    ``belts`` is a list of ``(center_xyz, half_xyz, velocity)`` tuples. Returns
    ``(positions, quats, frame_dt)`` where positions is ``(frames + 1, 3)`` and quats is
    ``(frames + 1, 4)`` (x, y, z, w) of the box per frame.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    belt_cfg = newton.ModelBuilder.ShapeConfig(mu=CONTACT_FRICTION, ke=1.0e5, kd=0.0)

    belt_shapes = []
    for center, half, _vel in belts:
        s = builder.add_shape_box(
            body=-1,
            hx=half[0],
            hy=half[1],
            hz=half[2],
            xform=wp.transform(p=wp.vec3(*center), q=wp.quat_identity()),
            cfg=belt_cfg,
        )
        belt_shapes.append(s)

    box_body = builder.add_link(
        xform=wp.transform(p=wp.vec3(box_xy[0], box_xy[1], BELT_TOP_Z + box_half[2] + 0.01), q=wp.quat_identity()),
        mass=1.0,
        label="box",
    )
    builder.add_shape_box(
        box_body,
        hx=box_half[0],
        hy=box_half[1],
        hz=box_half[2],
        cfg=newton.ModelBuilder.ShapeConfig(mu=CONTACT_FRICTION, restitution=0.0),
    )
    builder.add_articulation([builder.add_joint_free(box_body)], label="box")
    builder.color()

    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    solver = _make_solver(solver_name, model)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    conveyor = ConveyorForceModel(model, solver_type=solver_name)
    for shape, (_center, _half, vel) in zip(belt_shapes, belts, strict=True):
        conveyor.add_constant_belt(shape, velocity=wp.vec3(*vel))
    conveyor.finalize(contacts)

    sim_dt = 1.0 / fps / substeps
    positions = np.zeros((frames + 1, 3), dtype=np.float32)
    quats = np.zeros((frames + 1, 4), dtype=np.float32)
    q0 = state_0.body_q.numpy()[box_body]
    positions[0], quats[0] = q0[:3], q0[3:7]

    for f in range(frames):
        for _ in range(substeps):
            state_0.clear_forces()
            conveyor.apply(state_0)
            conveyor.snapshot_prev(state_0)
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            conveyor.update(solver, contacts, state_1, sim_dt)
            state_0, state_1 = state_1, state_0
        q = state_0.body_q.numpy()[box_body]
        positions[f + 1], quats[f + 1] = q[:3], q[3:7]

    return positions, quats, 1.0 / fps


def test_zero_speed_no_drift(test, device, solver_name):
    positions, _ = run_conveyor(device, solver_name, velocity=wp.vec3(0.0, 0.0, 0.0), frames=90)
    disp = positions[-1] - positions[0]
    test.assertLess(abs(float(disp[0])), 0.05, "unexpected X drift at zero belt speed")
    test.assertLess(abs(float(disp[1])), 0.05, "unexpected Y drift at zero belt speed")
    test.assertTrue(np.all(np.isfinite(positions)))


def test_constant_speed_forward(test, device, solver_name):
    speed = 2.0
    positions, frame_dt = run_conveyor(device, solver_name, velocity=wp.vec3(0.0, speed, 0.0), frames=150)
    vy = float(_final_velocity(positions, frame_dt)[1])
    # Converges toward the target speed (with some slip), in the +Y direction.
    test.assertGreater(vy, 0.7 * speed, f"box did not converge toward belt speed: vy={vy:.3f}")
    test.assertLess(vy, 1.1 * speed, f"box overshot belt speed: vy={vy:.3f}")
    # Negligible lateral drift.
    test.assertLess(abs(float(positions[-1][0])), 0.1)


def test_reversed_speed(test, device, solver_name):
    speed = 2.0
    positions, frame_dt = run_conveyor(device, solver_name, velocity=wp.vec3(0.0, -speed, 0.0), frames=150)
    vy = float(_final_velocity(positions, frame_dt)[1])
    test.assertLess(vy, -0.7 * speed, f"box not transported in -Y: vy={vy:.3f}")


def test_rotated_belt_direction(test, device, solver_name):
    # Rotate the long belt axis from +Y to +X and drive it in world-space +X.
    speed = 2.0
    positions, frame_dt = run_conveyor(
        device,
        solver_name,
        velocity=wp.vec3(speed, 0.0, 0.0),
        belt_rotation=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi),
        frames=90,
    )
    vel = _final_velocity(positions, frame_dt)
    test.assertGreater(float(vel[0]), 0.7 * speed, "box not transported along +X")
    test.assertLess(abs(float(vel[1])), 0.3, "unexpected Y motion for an X-driven belt")


def test_curved_velocity_field(test, device, solver_name):
    # Pivot field about the +Z axis at the origin: a box at +Y should be driven toward -X
    # (tangent = omega x r, omega=+Z, r=+Y -> -X).
    positions, frame_dt = run_conveyor(
        device,
        solver_name,
        pivot_point=wp.vec3(0.0, 0.0, BELT_TOP_Z),
        angular_velocity=wp.vec3(0.0, 0.0, 1.0),
        box_xy=(0.0, 2.0),
        frames=60,
    )
    vel = _final_velocity(positions, frame_dt, n=20)
    test.assertLess(float(vel[0]), -0.2, f"pivot field did not drive box toward -X: vx={vel[0]:.3f}")
    test.assertTrue(np.all(np.isfinite(positions)))


def test_contact_loss_no_force(test, device, solver_name):
    # Drive a box off a short belt and verify that the previously computed wrench is cleared.
    positions, _, diagnostics = run_conveyor(
        device,
        solver_name,
        velocity=wp.vec3(0.0, 3.0, 0.0),
        belt_half_y=1.0,
        frames=120,
        collect_diagnostics=True,
    )
    test.assertGreater(int(np.max(diagnostics["belt_contact_counts"])), 0, "box never contacted the belt")
    test.assertTrue(np.all(diagnostics["belt_contact_counts"][-16:] == 0), "belt contact did not end")
    test.assertTrue(np.all(diagnostics["conveyor_force_norms"][-16:] < 1.0e-6), "stale conveyor force remained")
    test.assertTrue(np.all(np.isfinite(positions)))


def test_long_running_stability(test, device, solver_name):
    positions, _ = run_conveyor(device, solver_name, velocity=wp.vec3(0.0, 0.5, 0.0), frames=400)
    test.assertTrue(np.all(np.isfinite(positions)), "state became non-finite over a long run")
    # Box stays on the belt band (height near belt top + box half extent).
    test.assertLess(abs(float(positions[-1][2] - (BELT_TOP_Z + BOX_HALF))), 0.15)


def test_multi_belt_handoff(test, device, solver_name):
    # Two belts meet along Y, both driving +Y. The box spans both surfaces during handoff.
    belts = [
        ((0.0, -2.0, BELT_TOP_Z - BELT_HALF_Z), (1.0, 2.0, BELT_HALF_Z), (0.0, 2.0, 0.0)),
        ((0.0, 2.0, BELT_TOP_Z - BELT_HALF_Z), (1.0, 2.0, BELT_HALF_Z), (0.0, 2.0, 0.0)),
    ]
    positions, _quats, _dt = run_multi_belt(device, solver_name, belts, box_xy=(0.0, -3.0), frames=150)
    test.assertTrue(np.all(np.isfinite(positions)))
    # Crossed the seam (started at y=-3) and kept going into positive Y.
    test.assertGreater(float(positions[-1][1]), 0.0, "box did not hand off across the belt seam")


def test_multi_belt_not_doubled(test, device, solver_name):
    # A box spanning two coplanar belts running at the SAME speed must converge to that speed,
    # not a multiple of it (behavior must not be biased by the contact count).
    speed = 2.0
    belts = [
        ((-0.5, 0.0, BELT_TOP_Z - BELT_HALF_Z), (0.5, 4.0, BELT_HALF_Z), (0.0, speed, 0.0)),
        ((0.5, 0.0, BELT_TOP_Z - BELT_HALF_Z), (0.5, 4.0, BELT_HALF_Z), (0.0, speed, 0.0)),
    ]
    positions, _quats, frame_dt = run_multi_belt(device, solver_name, belts, box_xy=(0.0, -2.0), frames=150)
    vy = float(_final_velocity(positions, frame_dt)[1])
    test.assertGreater(vy, 0.7 * speed, f"box did not converge toward belt speed: vy={vy:.3f}")
    test.assertLess(vy, 1.2 * speed, f"contact count doubled the effective speed: vy={vy:.3f}")


def test_differential_belts_rotate(test, device, solver_name):
    # A box straddling two side-by-side belts running at different speeds must turn (yaw),
    # matching the expected differential-drive behavior.
    belts = [
        ((-0.5, 0.0, BELT_TOP_Z - BELT_HALF_Z), (0.5, 4.0, BELT_HALF_Z), (0.0, 1.0, 0.0)),
        ((0.5, 0.0, BELT_TOP_Z - BELT_HALF_Z), (0.5, 4.0, BELT_HALF_Z), (0.0, 3.0, 0.0)),
    ]
    positions, quats, _dt = run_multi_belt(device, solver_name, belts, box_xy=(0.0, 0.0), frames=120)
    test.assertTrue(np.all(np.isfinite(positions)))
    yaw = abs(float(_yaw_deg(quats[-1])))
    test.assertGreater(yaw, 10.0, f"differential belts did not induce turning: yaw={yaw:.1f} deg")


def test_load_invariance(test, device, solver_name):
    speed = 2.0
    velocities = []
    for mass in (1.0, 5.0):
        positions, frame_dt = run_conveyor(
            device, solver_name, velocity=wp.vec3(0.0, speed, 0.0), box_mass=mass, frames=120
        )
        velocities.append(float(_final_velocity(positions, frame_dt)[1]))
    test.assertGreater(min(velocities), 0.7 * speed)
    test.assertLess(abs(velocities[0] - velocities[1]), 0.2, f"transport depends on load: {velocities}")


def test_timestep_stability(test, device, solver_name):
    trajectories = []
    for substeps in (4, 8):
        positions, frame_dt = run_conveyor(
            device, solver_name, velocity=wp.vec3(0.0, 2.0, 0.0), frames=120, substeps=substeps
        )
        trajectories.append((float(positions[-1][1]), float(_final_velocity(positions, frame_dt)[1])))
    test.assertLess(abs(trajectories[0][0] - trajectories[1][0]), 0.15, f"timestep changed travel: {trajectories}")
    test.assertLess(abs(trajectories[0][1] - trajectories[1][1]), 0.1, f"timestep changed speed: {trajectories}")


def test_contact_data_consistency(test, device, solver_name):
    _, _, diagnostics = run_conveyor(
        device,
        solver_name,
        velocity=wp.vec3(0.0, 1.0, 0.0),
        frames=90,
        collect_diagnostics=True,
    )
    test.assertTrue(np.all(diagnostics["belt_contact_counts"][-16:] > 0), "contact data was unavailable")

    forces = diagnostics["contact_forces"]
    normals = diagnostics["contact_normals"]
    test.assertGreater(len(forces), 0)
    test.assertTrue(np.all(np.isfinite(forces)))
    test.assertTrue(np.all(np.isfinite(normals)))
    test.assertTrue(np.all(np.isfinite(diagnostics["contact_points0"])))
    test.assertTrue(np.all(np.isfinite(diagnostics["contact_points1"])))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-4)

    belt_mask = (diagnostics["contact_shape0"] == diagnostics["belt_shape"]) | (
        diagnostics["contact_shape1"] == diagnostics["belt_shape"]
    )
    normal_force = np.abs(np.einsum("ij,ij->i", forces[belt_mask], normals[belt_mask])).sum()
    test.assertGreater(float(normal_force), 0.0)
    # The externally applied tangential force must remain within the summed Coulomb limit.
    conveyor_force = float(np.linalg.norm(diagnostics["conveyor_force"]))
    test.assertLessEqual(conveyor_force, 0.5 * float(normal_force) + 1.0e-4)


def test_backend_parity(test, device, solver_name):
    # The same force model must transport a box consistently across all three solvers.
    # (solver_name is unused; this case drives XPBD, VBD, and MuJoCo itself.)
    ys = {}
    for name in ("xpbd", "vbd", "mujoco"):
        positions, _ = run_conveyor(device, name, velocity=wp.vec3(0.0, 2.0, 0.0), frames=120)
        ys[name] = float(positions[-1][1])
    test.assertTrue(all(np.isfinite(v) for v in ys.values()), f"non-finite result: {ys}")
    spread = max(ys.values()) - min(ys.values())
    test.assertLess(spread, 0.6, f"backends disagree on transport distance: {ys}")


class TestConveyorForce(unittest.TestCase):
    pass


_devices = get_test_devices()
_cases = [
    ("zero_speed_no_drift", test_zero_speed_no_drift),
    ("constant_speed_forward", test_constant_speed_forward),
    ("reversed_speed", test_reversed_speed),
    ("rotated_belt_direction", test_rotated_belt_direction),
    ("curved_velocity_field", test_curved_velocity_field),
    ("contact_loss_no_force", test_contact_loss_no_force),
    ("long_running_stability", test_long_running_stability),
    ("multi_belt_handoff", test_multi_belt_handoff),
    ("multi_belt_not_doubled", test_multi_belt_not_doubled),
    ("differential_belts_rotate", test_differential_belts_rotate),
    ("load_invariance", test_load_invariance),
    ("timestep_stability", test_timestep_stability),
    ("contact_data_consistency", test_contact_data_consistency),
]

for _device in _devices:
    # MuJoCo and VBD rigid-contact support target CUDA.
    if _device.is_cpu:
        continue
    for _solver_name in ("xpbd", "vbd", "mujoco"):
        for _case_name, _case_fn in _cases:
            add_function_test(
                TestConveyorForce,
                f"test_{_case_name}_{_solver_name}",
                _case_fn,
                devices=[_device],
                solver_name=_solver_name,
            )
    add_function_test(
        TestConveyorForce,
        "test_backend_parity",
        test_backend_parity,
        devices=[_device],
        solver_name="mujoco",
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
