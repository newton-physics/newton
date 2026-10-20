# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Steel as a rigid entry of the experimental coupled solvers."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo, SolverSteel, SolverVBD
from newton.solvers.experimental.coupled import (
    CouplingInterface,
    SolverCoupled,
    SolverCoupledADMM,
    SolverCoupledProxy,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 480.0


def _steel(view, **kwargs):
    return SolverSteel(view, solve_mode="matrix_free", **kwargs)


def _build_box_cloth(
    device,
    *,
    box_masses=(5.0,),
    box_z=1.3,
    box_joint_qd=None,
    cloth_z=1.0,
    pinned=True,
    gravity=-9.81,
    ground=False,
    box_xy=(0.0, 0.0),
    cloth=True,
):
    """Build one free box above one cloth sheet per world; returns the model and per-world box ids."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    builder.default_shape_cfg.ke = 2.0e4
    if ground:
        builder.add_ground_plane()
    boxes = []
    for mass in box_masses:
        builder.begin_world()
        box = builder.add_body(xform=wp.transform((box_xy[0], box_xy[1], box_z), wp.quat_identity()))
        cfg = newton.ModelBuilder.ShapeConfig(density=mass / 0.2**3, mu=0.8)
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        if box_joint_qd is not None:
            start = builder.joint_qd_start[-1]
            builder.joint_qd[start : start + 6] = list(box_joint_qd)
        boxes.append(box)
        if not cloth:
            builder.end_world()
            continue
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, cloth_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            fix_left=pinned,
            fix_right=pinned,
            dim_x=16,
            dim_y=16,
            cell_x=1.0 / 16.0,
            cell_y=1.0 / 16.0,
            mass=0.1,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=1.0e1,
            edge_ke=0.01,
            particle_radius=0.01,
        )
        builder.end_world()
    if cloth:
        builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e4
    model.soft_contact_mu = 0.5
    return model, boxes


def _coupled(model, boxes, rigid_factory=_steel, *, iterations=1, mode="lagged", in_place=False):
    rigid_bodies = list(range(model.body_count))
    return SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupledProxy.Entry(
                name="rigid",
                solver=rigid_factory,
                bodies=rigid_bodies,
                joints=list(range(model.joint_count)),
                in_place=in_place,
            ),
            SolverCoupledProxy.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, iterations=10, rigid_compliant_alm=True),
                particles=list(range(model.particle_count)),
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="rigid",
                    destination="vbd",
                    bodies=rigid_bodies,
                    mode=mode,
                    collision_pipeline=newton.CollisionPipeline,
                    collide_interval=1,
                )
            ],
            iterations=iterations,
        ),
    )


class _Rollout:
    """Owns a solver, a double-buffered state and the outer contact buffer."""

    def __init__(self, model, solver):
        self.model = model
        self.solver = solver
        self.state_0 = model.state()
        self.state_1 = model.state()
        self.control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        self.pipeline = newton.CollisionPipeline(model)
        self.contacts = self.pipeline.contacts()
        if isinstance(solver, SolverCoupledProxy):
            solver.prepare_contacts(self.contacts)

    def step(self):
        self.pipeline.collide(self.state_0, self.contacts)
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def run(self, steps):
        body_q = []
        for _ in range(steps):
            self.step()
            body_q.append(self.state_0.body_q.numpy().copy())
        return np.asarray(body_q)


def test_no_contact_matches_standalone(test, device):
    """Match standalone Steel for a box far from the cloth: no feedback, no double gravity."""
    for iterations in (1, 3):
        model, _ = _build_box_cloth(device, box_z=3.0, ground=True)
        standalone_model, _ = _build_box_cloth(device, box_z=3.0, ground=True, cloth=False)
        coupled = _Rollout(model, _coupled(model, None, iterations=iterations))
        reference = _Rollout(standalone_model, _steel(standalone_model))
        coupled_q = coupled.run(60)
        reference_q = reference.run(60)
        np.testing.assert_allclose(coupled_q, reference_q, atol=1.0e-6)
        fall = reference_q[0, 0, 2] - reference_q[-1, 0, 2]
        expected = 0.5 * 9.81 * ((60 * DT) ** 2 - DT**2)
        test.assertAlmostEqual(fall, expected, delta=0.02 * expected)
        feedback = coupled.solver._proxy_mappings[0].coupling_forces.numpy()
        test.assertEqual(float(np.abs(feedback).max()), 0.0)


def test_in_place_restart_refreshes_kinematics(test, device):
    """Re-derive kinematics when an in-place source restarts instead of reusing the previous solve's."""
    kwargs = {"box_z": 3.0, "box_joint_qd": (0.3, 0.0, 0.0, 0.0, 2.0, 1.0)}
    reference_model, _ = _build_box_cloth(device, cloth=False, **kwargs)
    reference_q = _Rollout(reference_model, _steel(reference_model)).run(60)
    model, _ = _build_box_cloth(device, **kwargs)
    coupled_q = _Rollout(model, _coupled(model, None, iterations=3, mode="staggered", in_place=True)).run(60)
    np.testing.assert_allclose(coupled_q, reference_q, atol=1.0e-5)


def test_iteration_restart_preserves_patch_history(test, device):
    """Keep persistent patch friction identical to standalone across repeated proxy solves of one step."""
    # A spinning box lands on the Steel-owned ground far from the cloth; landing creates new patches.
    kwargs = {"box_z": 0.13, "box_xy": (3.0, 0.0), "box_joint_qd": (1.0, 0.3, -0.5, 0.0, 0.0, 3.0), "ground": True}
    reference_model, _ = _build_box_cloth(device, cloth=False, **kwargs)
    reference_q = _Rollout(reference_model, _steel(reference_model)).run(120)
    test.assertGreater(np.linalg.norm(reference_q[-1, 0, :2] - reference_q[0, 0, :2]), 0.01)
    for iterations in (1, 3):
        model, _ = _build_box_cloth(device, **kwargs)
        coupled_q = _Rollout(model, _coupled(model, None, iterations=iterations)).run(120)
        np.testing.assert_allclose(coupled_q, reference_q, atol=1.0e-5, err_msg=f"iterations={iterations}")


def test_contact_feedback_is_consumed_exactly(test, device):
    """Apply exactly the previous VBD harvest, force and torque, as each step's box impulse in gravity-free contact."""
    # Off-center impact so the harvested wrench carries torque.
    model, boxes = _build_box_cloth(
        device, box_z=1.13, box_xy=(0.15, 0.05), box_joint_qd=(0.0, 0.0, -1.0, 0.0, 0.0, 0.0), pinned=False, gravity=0.0
    )
    rollout = _Rollout(model, _coupled(model, boxes))
    mapping = rollout.solver._proxy_mappings[0]
    mass = float(model.body_mass.numpy()[0])
    inertia = model.body_inertia.numpy()[0]
    max_force = 0.0
    previous_wrench = np.zeros(6)
    for _ in range(150):
        qd_before = rollout.state_0.body_qd.numpy()[0].copy()
        q_before = rollout.state_0.body_q.numpy()[0].copy()
        rollout.step()
        qd_after = rollout.state_0.body_qd.numpy()[0]
        # Lagged mode: this step's source solve consumed the previous harvest.
        np.testing.assert_allclose(qd_after[:3] - qd_before[:3], previous_wrench[:3] * DT / mass, atol=1.0e-4)
        rotation = np.array(wp.quat_to_matrix(wp.quat(*q_before[3:7]))).reshape(3, 3)
        world_inertia = rotation @ inertia @ rotation.T
        np.testing.assert_allclose(
            qd_after[3:] - qd_before[3:], np.linalg.solve(world_inertia, previous_wrench[3:]) * DT, atol=2.0e-3
        )
        previous_wrench = mapping.coupling_forces.numpy()[0].astype(np.float64)
        max_force = max(max_force, float(np.linalg.norm(previous_wrench[:3])))
    test.assertGreater(max_force, 1.0, "the box never pressed into the cloth")
    # Reciprocal reaction: the box slowed down and the free cloth was pushed along -z.
    test.assertGreater(float(rollout.state_0.body_qd.numpy()[0][2]), -0.95)
    test.assertLess(float(rollout.state_0.particle_qd.numpy()[:, 2].mean()), -0.05)


def test_box_rests_on_cloth_like_mujoco(test, device):
    """Settle a box on a pinned cloth within 2 mm of the MuJoCo source."""
    rest = {}
    for name, factory in (
        ("steel", _steel),
        ("mjc", lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=200)),
    ):
        model, boxes = _build_box_cloth(device)
        rollout = _Rollout(model, _coupled(model, boxes, factory))
        rollout.run(480)
        rest[name] = (
            float(rollout.state_0.body_q.numpy()[0][2]),
            float(rollout.state_0.particle_q.numpy()[:, 2].min()),
        )
    box_z, cloth_z = rest["steel"]
    test.assertLess(cloth_z, 0.97, "the cloth did not deform")
    test.assertGreater(box_z, cloth_z + 0.09, "the box sank through the cloth")
    test.assertAlmostEqual(box_z, rest["mjc"][0], delta=2.0e-3)
    test.assertAlmostEqual(cloth_z, rest["mjc"][1], delta=2.0e-3)


def test_worlds_are_isolated(test, device):
    """Match single-world rollouts for two worlds with unequal masses."""
    model, boxes = _build_box_cloth(device, box_masses=(2.0, 10.0))
    both = _Rollout(model, _coupled(model, boxes)).run(240)
    test.assertGreater(
        abs(both[-1, boxes[0], 2] - both[-1, boxes[1], 2]), 2.0e-2, "unequal masses should sag differently"
    )
    for world, mass in enumerate((2.0, 10.0)):
        single_model, _ = _build_box_cloth(device, box_masses=(mass,))
        single = _Rollout(single_model, _coupled(single_model, None)).run(240)
        # Contact ordering differs between layouts; leakage would show at the 2 cm inter-world scale.
        np.testing.assert_allclose(both[:, boxes[world], :3], single[:, 0, :3], atol=2.0e-3)


def test_graph_capture_matches_eager(test, device):
    """Reproduce eager stepping with repeated graph replay of the coupled step, including after a reset."""
    results = []
    for use_graph in (False, True):
        model, boxes = _build_box_cloth(device, box_masses=(5.0, 8.0))
        rollout = _Rollout(model, _coupled(model, boxes, iterations=2))
        graph = None
        if use_graph:
            # Both buffers must swap back to the captured layout, so capture two steps.
            with wp.ScopedCapture(device) as capture:
                rollout.step()
                rollout.step()
            graph = capture.graph
        trajectory = []
        for _ in range(60):
            if graph is None:
                rollout.step()
                rollout.step()
            else:
                wp.capture_launch(graph)
            trajectory.append(rollout.state_0.body_q.numpy().copy())
        rollout.solver.reset(rollout.state_0)
        for _ in range(10):
            if graph is None:
                rollout.step()
                rollout.step()
            else:
                wp.capture_launch(graph)
            trajectory.append(rollout.state_0.body_q.numpy().copy())
        results.append(np.asarray(trajectory))
    np.testing.assert_allclose(results[1][:60], results[0][:60], atol=1.0e-5)
    # Host-side reset bookkeeping makes post-reset replay differ at the 1e-4 level for any source solver.
    np.testing.assert_allclose(results[1][60:], results[0][60:], atol=1.0e-3)


def _build_rolling_sphere(device):
    """Build an articulated sphere rolling and spinning on the ground, next to a cloth sheet."""
    radius = 0.02
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    material = newton.ModelBuilder.ShapeConfig(mu=0.8, mu_torsional=0.005)
    builder.add_ground_plane(cfg=material)
    pose = wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity())
    tiny = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-8)
    carriers = [builder.add_link(xform=pose, mass=1.0e-4, inertia=tiny) for _ in range(3)]
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.4 * radius * radius)
    sphere = builder.add_link(xform=pose, mass=1.0, inertia=inertia)
    identity = wp.transform_identity()
    builder.add_articulation(
        [
            builder.add_joint_prismatic(-1, carriers[0], parent_xform=pose, child_xform=identity, axis=(1.0, 0.0, 0.0)),
            builder.add_joint_prismatic(
                carriers[0], carriers[1], parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)
            ),
            builder.add_joint_revolute(
                carriers[1], carriers[2], parent_xform=identity, child_xform=identity, axis=(0.0, 1.0, 0.0)
            ),
            builder.add_joint_revolute(
                carriers[2], sphere, parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)
            ),
        ]
    )
    builder.add_shape_sphere(
        sphere, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8, mu_torsional=0.005)
    )
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5, -0.5, 2.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.5,
        cell_y=0.5,
        mass=0.1,
        fix_left=True,
        fix_right=True,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=10.0,
    )
    builder.color()
    model = builder.finalize(device=device)
    model.joint_qd.assign(np.array([1.0, 0.0, 1.0 / radius, 1.0], dtype=np.float32))
    return model


def _undamped(solver):
    """Turn off the default angular damping of the free roots."""
    solver._rigid_body_angular_damping.zero_()
    return solver


def test_articulated_graph_capture_matches_eager(test, device):
    """Match eager stepping with graph replay of an articulated source across proxy iterations and friction models."""
    options = {
        "solve_mode": "matrix_free",
        "articulated_contact_response": "immediate",
        "iterations": 16,
        "articulation_constraint_max": 32,
        "free_body_constraint_max": 16,
        "double_buffer": False,
    }
    cases = [
        ("point", {"friction_anchor_beta": 0.0}, (1, 3)),
        ("patch", {}, (1, 3)),
        ("bisection", {"friction_anchor_beta": 0.0, "friction_mode": "bisection"}, (3,)),
        ("coulomb_newton", {"friction_anchor_beta": 0.0, "friction_mode": "coulomb_newton"}, (3,)),
    ]
    fields = ("joint_q", "joint_qd", "particle_q", "particle_qd")
    for friction, friction_options, iteration_counts in cases:
        for iterations in iteration_counts:
            with test.subTest(friction=friction, iterations=iterations):
                trajectories = []
                for use_graph in (False, True):
                    model = _build_rolling_sphere(device)
                    rollout = _Rollout(
                        model,
                        _coupled(
                            model,
                            None,
                            lambda view, kw=friction_options: _undamped(SolverSteel(view, **options, **kw)),
                            iterations=iterations,
                        ),
                    )
                    for _ in range(12):
                        rollout.step()
                    graph = None
                    if use_graph:
                        # Both buffers must swap back to the captured layout, so capture two steps.
                        with wp.ScopedCapture(device) as capture:
                            rollout.step()
                            rollout.step()
                        graph = capture.graph
                    trajectory = {name: [] for name in fields}
                    for _ in range(10):
                        if graph is None:
                            rollout.step()
                            rollout.step()
                        else:
                            wp.capture_launch(graph)
                        for name in fields:
                            trajectory[name].append(getattr(rollout.state_0, name).numpy().copy())
                    trajectories.append({name: np.asarray(values) for name, values in trajectory.items()})
                eager, replay = trajectories
                # The slide along x and the roll about y both advance while the sphere rolls.
                motion = eager["joint_q"][-1] - eager["joint_q"][0]
                test.assertTrue(np.all(motion[[0, 2]] > 0.0), motion)
                for name in fields:
                    test.assertTrue(np.isfinite(eager[name]).all(), name)
                    test.assertTrue(np.isfinite(replay[name]).all(), name)
                    np.testing.assert_allclose(replay[name], eager[name], atol=1.0e-6, err_msg=name)


def test_unsupported_options_raise(test, device):
    """Reject options that are not validated in coupled use."""
    for kwargs in (
        {"contact_warmstart": True},
        {"enable_sleeping": True},
    ):
        model, boxes = _build_box_cloth(device)
        with test.assertRaises(NotImplementedError):
            _Rollout(model, _coupled(model, boxes, lambda v, kwargs=kwargs: _steel(v, **kwargs))).step()


def test_articulated_effective_mass(test, device):
    """Report the translational mobility of a hinged link's COM, not its free mass."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    mass, half_length, armature = 2.0, 0.25, 0.01
    link = builder.add_link()
    builder.add_shape_box(
        link,
        hx=half_length,
        hy=0.05,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(density=mass / (8 * half_length * 0.05**2)),
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform((-half_length, 0.0, 0.0), wp.quat_identity()),
        armature=armature,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = _steel(model)
    kind = wp.array([int(CouplingInterface.EndpointKind.BODY)] * 2, dtype=int, device=device)
    index = wp.array([link, link], dtype=int, device=device)
    local_pos = wp.array([wp.vec3(0.0), wp.vec3(half_length, 0.0, 0.0)], dtype=wp.vec3, device=device)
    out_mass = wp.zeros(2, dtype=float, device=device)
    out_inertia = wp.zeros(2, dtype=wp.mat33, device=device)
    solver.coupling_eval_effective_mass_block(kind, index, local_pos, out_mass, out_inertia)

    inertia_yy = float(model.body_inertia.numpy()[link][1, 1])
    joint_inertia = inertia_yy + mass * half_length**2 + armature
    # Only one direction is mobile, so the axis-mean inverse weight is r^2 / (3 H).
    for row, lever in enumerate((half_length, 2.0 * half_length)):
        test.assertAlmostEqual(float(out_mass.numpy()[row]), 3.0 * joint_inertia / lever**2, delta=1.0e-3)
    scale = (np.trace(np.linalg.inv(model.body_inertia.numpy()[link])) / 3.0) / (1.0 / (3.0 * joint_inertia))
    np.testing.assert_allclose(out_inertia.numpy()[0], model.body_inertia.numpy()[link] * scale, rtol=1.0e-4)


def test_offset_contact_turns_hinge(test, device):
    """Turn the Steel hinge with a soft block landing off its axis, more so with a longer lever."""
    rotation = {}
    for lever in (0.08, 0.32):
        # Without gravity the hinge rests at its target until the block, thrown downward, lands on it.
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.2, hy=0.1, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=1.0 / 0.0032))
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            child_xform=wp.transform((-0.2, 0.0, 0.0), wp.quat_identity()),
            target_ke=20.0,
            target_kd=1.0,
        )
        builder.add_articulation([joint])
        builder.add_soft_grid(
            pos=wp.vec3(lever - 0.05, -0.05, 1.04),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, -1.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=1.0e3,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1.0e-2,
            particle_radius=0.01,
        )
        builder.color()
        model = builder.finalize(device=device)
        model.soft_contact_ke = 1.0e4
        rollout = _Rollout(model, _coupled(model, [link]))
        peak = 0.0
        for _ in range(240):
            rollout.step()
            peak = max(peak, float(rollout.state_0.joint_q.numpy()[0]))
        rotation[lever] = peak
    # Pushing down on +x rotates the link positively about +y.
    test.assertGreater(rotation[0.08], 0.0)
    test.assertGreater(rotation[0.32], 1.5 * rotation[0.08])


def _build_box_rod(device, *, box_z=0.15, box_xy=(0.0, 0.0), box_joint_qd=None, rod=True):
    """Build a free Steel box above two VBD cables lying on the ground; returns model, box and rod bodies."""
    builder = newton.ModelBuilder()
    SolverVBD.register_custom_attributes(builder)
    builder.add_ground_plane()
    box = builder.add_body(xform=wp.transform((box_xy[0], box_xy[1], box_z), wp.quat_identity()))
    # The box overhangs both cables so it cannot drop between them.
    builder.add_shape_box(box, hx=0.05, hy=0.08, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=2.0 / 0.0016))
    if box_joint_qd is not None:
        builder.joint_qd[0:6] = list(box_joint_qd)
    rod_bodies = []
    # Two parallel cables support the box stably.
    for y in (-0.04, 0.04) if rod else ():
        cable = newton.Rod.create_straight(
            start=wp.vec3(-0.2, y, 0.012),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=0.4,
            segment_count=8,
            twist_total=0.0,
            radius=0.012,
        )
        bodies, _ = builder.add_rod(
            rod=cable,
            body_frame_origin="start",
            cfg=newton.ModelBuilder.ShapeConfig(density=1400.0, ke=5.0e4, kd=1.0e1, mu=0.9, margin=0.001, gap=0.002),
            stretch_stiffness=2.0e5,
            stretch_damping=2.0e-2,
            bend_stiffness=0.08,
            bend_damping=1.6e-3,
        )
        rod_bodies += list(bodies)
    builder.color()
    return builder.finalize(device=device), box, rod_bodies


def _admm(model, box, rod_bodies, rigid_factory=_steel, *, iterations=3, gamma=0.001):
    model.rigid_contact_max = max(model.rigid_contact_max or 0, 4096)
    box_joints = np.flatnonzero(model.joint_child.numpy() == box).tolist()
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(name="rigid", solver=rigid_factory, bodies=[box], joints=box_joints),
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(
                    model=v, iterations=8, rigid_compliant_alm=True, rigid_contact_history=False
                ),
                bodies=rod_bodies,
                joints=[j for j in range(model.joint_count) if j not in box_joints],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=iterations,
            rho=200.0,
            gamma=gamma,
            baumgarte=0.5,
            contact_pairs=[SolverCoupledADMM.ContactPair(source="rigid", destination="vbd")],
        ),
    )


def test_admm_no_contact_matches_standalone(test, device):
    """Match standalone Steel under ADMM restarts for a box that never touches the cable."""
    kwargs = {"box_z": 0.08, "box_xy": (3.0, 0.0), "box_joint_qd": (1.0, 0.3, -0.5, 0.0, 0.0, 3.0)}
    reference_model, _, _ = _build_box_rod(device, rod=False, **kwargs)
    reference_q = _Rollout(reference_model, _steel(reference_model)).run(120)
    for iterations in (1, 3):
        model, box, rod_bodies = _build_box_rod(device, **kwargs)
        rollout = _Rollout(model, _admm(model, box, rod_bodies, iterations=iterations, gamma=0.0))
        coupled_q = rollout.run(120)
        np.testing.assert_allclose(
            coupled_q[:, box], reference_q[:, 0], atol=1.0e-5, err_msg=f"iterations={iterations}"
        )


def _history_snapshot(solver):
    patches = solver._friction_patches
    body_world = solver.model.body_world.numpy()
    body_a, body_b = patches.previous.body_a.numpy(), patches.previous.body_b.numpy()
    return {
        "q": patches.previous_q.numpy().copy(),
        # Every carried patch here pairs a box with the ground or with a box of the same world.
        "world": np.where(body_a >= 0, body_world[body_a], body_world[body_b]),
        "valid": patches.previous.valid.numpy().copy(),
        "displacement": patches.previous.displacement.numpy().copy(),
    }


def test_reset_preserves_unselected_patch_history(test, device):
    """Keep patch history through no-op resets, and the other world's through a single-world reset."""
    kwargs = {
        "box_masses": (5.0, 5.0),
        "box_z": 0.13,
        "box_xy": (3.0, 0.0),
        "box_joint_qd": (1.0, 0.3, -0.5, 0.0, 0.0, 3.0),
        "ground": True,
    }
    for mask_values in ((False, False, False), (False, False, True), (False, True, False)):
        model, boxes = _build_box_cloth(device, **kwargs)
        rollout = _Rollout(model, _coupled(model, boxes, iterations=2))
        rollout.run(60)
        solver = rollout.solver.solver("rigid")
        before = _history_snapshot(solver)
        test.assertGreater(int(np.count_nonzero(before["valid"])), 0, "no carried patches to protect")
        rollout.solver.reset(rollout.state_0, wp.array(mask_values, dtype=wp.bool, device=device))
        after = _history_snapshot(solver)
        kept = before["world"] != 1 if mask_values[1] else np.ones_like(before["world"], dtype=bool)
        test.assertTrue(np.any(kept & (before["valid"] != 0)))
        np.testing.assert_array_equal(after["valid"][kept], before["valid"][kept], err_msg=str(mask_values))
        np.testing.assert_array_equal(after["displacement"][kept], before["displacement"][kept])
        np.testing.assert_array_equal(after["q"][boxes[0]], before["q"][boxes[0]])
        if mask_values[1]:
            test.assertEqual(int(np.count_nonzero(after["valid"][before["world"] == 1])), 0)
        else:
            np.testing.assert_array_equal(after["q"], before["q"])


def test_effective_mass_ignores_loop_closures(test, device):
    """Exclude disabled loop-closing joints from the tree-only effective mass, also before a later articulation."""
    results = {}
    for links in (1, 2):
        for closure in (False, True):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            for _articulation in range(2):
                bodies, joints = [], []
                for index in range(links):
                    body = builder.add_link()
                    builder.add_shape_box(
                        body, hx=0.25, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=400.0)
                    )
                    joints.append(
                        builder.add_joint_revolute(
                            -1 if index == 0 else bodies[-1],
                            body,
                            axis=wp.vec3(0.0, 1.0, 0.0),
                            parent_xform=wp.transform((0.0 if index == 0 else 0.25, 0.0, 0.0), wp.quat_identity()),
                            child_xform=wp.transform((-0.25, 0.0, 0.0), wp.quat_identity()),
                            armature=0.01,
                        )
                    )
                    bodies.append(body)
                builder.add_articulation(joints)
                if closure:
                    with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "Parallel joints"):
                        builder.add_joint_ball(-1, bodies[0], enabled=False)
            model = builder.finalize(device=device)
            solver = _steel(model)
            last = [links - 1, 2 * links - 1]
            kind = wp.array([int(CouplingInterface.EndpointKind.BODY)] * 2, dtype=int, device=device)
            mass = wp.zeros(2, dtype=float, device=device)
            solver.coupling_eval_effective_mass(
                kind, wp.array(last, dtype=int, device=device), wp.zeros(2, dtype=wp.vec3, device=device), mass
            )
            results[(links, closure)] = mass.numpy()
        np.testing.assert_allclose(results[(links, True)], results[(links, False)], rtol=1.0e-5)
        np.testing.assert_allclose(results[(links, False)][0], results[(links, False)][1], rtol=1.0e-5)
    test.assertAlmostEqual(float(results[(1, False)][0]), 8.56, delta=1.0e-3)


def test_unsupported_roles_raise(test, device):
    """Reject owning particles and acting as a proxy destination."""
    model, _ = _build_box_cloth(device)
    with test.assertRaises(NotImplementedError):
        solver = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="rigid",
                    solver=_steel,
                    bodies=list(range(model.body_count)),
                    joints=list(range(model.joint_count)),
                    particles=list(range(model.particle_count)),
                ),
            ],
        )
        _Rollout(model, solver).step()

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    rigid_box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(rigid_box, hx=0.1, hy=0.1, hz=0.1)
    soft_box = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(soft_box, hx=0.1, hy=0.1, hz=0.1)
    builder.color()
    model = builder.finalize(device=device)
    with test.assertRaises(NotImplementedError):
        solver = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, iterations=4, rigid_compliant_alm=True),
                    bodies=[soft_box],
                    joints=[1],
                ),
                SolverCoupledProxy.Entry(name="rigid", solver=_steel, bodies=[rigid_box], joints=[0]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(source="vbd", destination="rigid", bodies=[soft_box])]
            ),
        )
        _Rollout(model, solver).step()


def _build_split_arm(device, *, base: bool, world_count: int = 2):
    """Build a FREE base, a ball shoulder and a turning revolute elbow per world, or the arm alone on a FREE root."""
    template = newton.ModelBuilder()
    inertia = wp.mat33(np.eye(3) * 0.01)
    half = wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity())
    half_back = wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity())
    joints = []
    if base:
        base_body = template.add_link(mass=1.0, inertia=inertia)
        joints.append(template.add_joint_free(base_body))
    upper, lower = (
        template.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia)
        for x in (0.3, 0.6)
    )
    if base:
        joints.append(template.add_joint_ball(base_body, upper, parent_xform=half, child_xform=half_back))
    else:
        joints.append(template.add_joint_free(upper))
    elbow = template.add_joint_revolute(upper, lower, axis=(0.0, 1.0, 0.0), parent_xform=half, child_xform=half_back)
    joints.append(elbow)
    template.add_articulation(joints)
    template.joint_qd[template.joint_qd_start[elbow]] = 2.0
    builder = newton.ModelBuilder()
    builder.replicate(template, world_count=world_count)
    return builder.finalize(device=device), template.body_count, template.joint_count


def _run_split_arm(model, solver):
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    for _ in range(60):
        solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return state_0.body_q.numpy()


def test_split_articulation_gets_free_root(test, device):
    """Step a split arm subtree on its view's FREE root like the same arm built on a FREE root."""

    def steel(view):
        return SolverSteel(view, solve_mode="split")

    world_count = 2
    model, bodies_per_world, joints_per_world = _build_split_arm(device, base=True, world_count=world_count)

    def expand(ids, stride):
        return [world * stride + i for world in range(world_count) for i in ids]

    # Neither entry owns the shoulder, so the arm view roots the upper link at a FREE joint.
    solver = SolverCoupled(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="base", solver=steel, bodies=expand([0], bodies_per_world), joints=expand([0], joints_per_world)
            ),
            SolverCoupled.Entry(
                name="arm", solver=steel, bodies=expand([1, 2], bodies_per_world), joints=expand([2], joints_per_world)
            ),
        ],
    )
    view = solver.view("arm")
    free, revolute = int(newton.JointType.FREE), int(newton.JointType.REVOLUTE)
    np.testing.assert_array_equal(view.joint_type.numpy(), [free, revolute] * world_count)
    test.assertTrue(bool(np.all(view.joint_enabled.numpy())))
    np.testing.assert_array_equal(view.articulation_start.numpy(), [0, 2, 4])

    body_q = _run_split_arm(model, solver).reshape(world_count, bodies_per_world, 7)
    reference_model, _, _ = _build_split_arm(device, base=False, world_count=world_count)
    reference_q = _run_split_arm(reference_model, steel(reference_model)).reshape(world_count, 2, 7)
    np.testing.assert_allclose(body_q[:, 1:], reference_q, atol=1.0e-5)
    # The elbow turned, so the arm did not fall as a rigid body.
    test.assertGreater(abs(float(reference_q[0, 1, 2] - reference_q[0, 0, 2])), 0.05)


class TestCoupledSolverSteel(unittest.TestCase):
    pass


for _name, _func in (
    ("test_no_contact_matches_standalone", test_no_contact_matches_standalone),
    ("test_iteration_restart_preserves_patch_history", test_iteration_restart_preserves_patch_history),
    ("test_in_place_restart_refreshes_kinematics", test_in_place_restart_refreshes_kinematics),
    ("test_contact_feedback_is_consumed_exactly", test_contact_feedback_is_consumed_exactly),
    ("test_box_rests_on_cloth_like_mujoco", test_box_rests_on_cloth_like_mujoco),
    ("test_worlds_are_isolated", test_worlds_are_isolated),
    ("test_graph_capture_matches_eager", test_graph_capture_matches_eager),
    ("test_articulated_graph_capture_matches_eager", test_articulated_graph_capture_matches_eager),
    ("test_unsupported_options_raise", test_unsupported_options_raise),
    ("test_articulated_effective_mass", test_articulated_effective_mass),
    ("test_offset_contact_turns_hinge", test_offset_contact_turns_hinge),
    ("test_reset_preserves_unselected_patch_history", test_reset_preserves_unselected_patch_history),
    ("test_effective_mass_ignores_loop_closures", test_effective_mass_ignores_loop_closures),
    ("test_unsupported_roles_raise", test_unsupported_roles_raise),
    ("test_admm_no_contact_matches_standalone", test_admm_no_contact_matches_standalone),
):
    add_function_test(TestCoupledSolverSteel, _name, _func, devices=get_cuda_test_devices())


add_function_test(
    TestCoupledSolverSteel,
    "test_split_articulation_gets_free_root",
    test_split_articulation_gets_free_root,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
