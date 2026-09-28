# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SAT box-box manifolds and their feature-based contact identity."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.geometry.collision_primitive import collide_box_box, collide_box_box_features
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


@wp.kernel
def _eval_box_box_features(
    box1_pos: wp.vec3,
    box1_rot: wp.mat33,
    box1_size: wp.vec3,
    box2_pos: wp.vec3,
    box2_rot: wp.mat33,
    box2_size: wp.vec3,
    margin: float,
    # outputs
    out_dist: wp.array[float],
    out_pos: wp.array[wp.vec3],
    out_feat: wp.array[int],
):
    dist, pos, _normals, feat = collide_box_box_features(
        box1_pos, box1_rot, box1_size, box2_pos, box2_rot, box2_size, margin
    )
    for i in range(8):
        out_dist[i] = dist[i]
        out_pos[i] = wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2])
        out_feat[i] = feat[i]


@wp.kernel
def _eval_box_box_wrapper(
    box1_pos: wp.vec3,
    box1_rot: wp.mat33,
    box1_size: wp.vec3,
    box2_pos: wp.vec3,
    box2_rot: wp.mat33,
    box2_size: wp.vec3,
    margin: float,
    # outputs
    out_dist: wp.array[float],
    out_pos: wp.array[wp.vec3],
):
    dist, pos, _normals = collide_box_box(box1_pos, box1_rot, box1_size, box2_pos, box2_rot, box2_size, margin)
    for i in range(8):
        out_dist[i] = dist[i]
        out_pos[i] = wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2])


def _features_at(device, big_pos, big_size, small_pos, small_size, margin=0.003):
    ident = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    out_dist = wp.zeros(8, dtype=wp.float32)
    out_pos = wp.zeros(8, dtype=wp.vec3)
    out_feat = wp.zeros(8, dtype=wp.int32)
    wp.launch(
        _eval_box_box_features,
        dim=1,
        inputs=[
            wp.vec3(*big_pos),
            ident,
            wp.vec3(*big_size),
            wp.vec3(*small_pos),
            ident,
            wp.vec3(*small_size),
            margin,
        ],
        outputs=[out_dist, out_pos, out_feat],
        device=device,
    )
    dist = out_dist.numpy()
    feats = out_feat.numpy()
    valid = [(int(feats[i]), float(dist[i])) for i in range(8) if dist[i] < 1.0e9]
    return valid, out_pos.numpy()


def test_box_box_feature_wrapper_parity(test: unittest.TestCase, device):
    """The public 3-tuple wrapper returns exactly the feature variant's
    geometry (byte-identical distances and positions)."""
    with wp.ScopedDevice(device):
        ident = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        args = [
            wp.vec3(0.0, 0.0, 0.0),
            ident,
            wp.vec3(0.25, 0.25, 0.05),
            wp.vec3(0.02, 0.01, 0.0995),
            ident,
            wp.vec3(0.05, 0.05, 0.05),
            0.003,
        ]
        d_a = wp.zeros(8, dtype=wp.float32)
        p_a = wp.zeros(8, dtype=wp.vec3)
        f_a = wp.zeros(8, dtype=wp.int32)
        wp.launch(_eval_box_box_features, dim=1, inputs=args, outputs=[d_a, p_a, f_a], device=device)
        d_b = wp.zeros(8, dtype=wp.float32)
        p_b = wp.zeros(8, dtype=wp.vec3)
        wp.launch(_eval_box_box_wrapper, dim=1, inputs=args, outputs=[d_b, p_b], device=device)
        np.testing.assert_array_equal(d_a.numpy(), d_b.numpy())
        np.testing.assert_array_equal(p_a.numpy(), p_b.numpy())


def test_box_box_feature_identity_under_sliding(test: unittest.TestCase, device):
    """Sliding a small box across the center of a large face must not churn
    contact identities: the persisting contacts keep the same feature ids at
    every position, including the center crossing (any center-based quadrant
    binning fails this)."""
    with wp.ScopedDevice(device):
        big_size = (0.25, 0.25, 0.05)
        small_size = (0.05, 0.05, 0.05)
        id_sets = []
        for x_mm in range(-20, 21, 4):
            valid, _ = _features_at(
                device,
                (0.0, 0.0, 0.0),
                big_size,
                (x_mm * 1.0e-3, 0.0, 0.0995),
                small_size,
            )
            id_sets.append({f for f, _d in valid})
        test.assertGreaterEqual(len(id_sets[0]), 4, "expected a full face manifold")
        for k, ids in enumerate(id_sets):
            test.assertEqual(
                ids, id_sets[0], f"feature ids churned at x={-20 + 4 * k} mm: {sorted(ids)} vs {sorted(id_sets[0])}"
            )


def test_box_box_corner_manifold_preserved(test: unittest.TestCase, device):
    """A small box resting near the corner of a large face keeps its full
    4-contact manifold: all four incident-face corners have distinct feature
    ids, so the reduction cannot merge them (quadrant binning around the
    other box's center collapsed them into one slot and dropped support)."""
    with wp.ScopedDevice(device):
        valid, _ = _features_at(
            device,
            (0.0, 0.0, 0.0),
            (0.25, 0.25, 0.05),
            # near the +x/+y corner of the big face, fully supported
            (0.19, 0.19, 0.0995),
            (0.05, 0.05, 0.05),
        )
        ids = {f for f, _d in valid}
        test.assertGreaterEqual(len(ids), 4, f"corner manifold collapsed: features {sorted(ids)}")
        test.assertEqual(len(ids), len(valid), "duplicate feature ids in one manifold")
        # physical check: it must rest flat through the SAT pipeline
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
        base = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_box(base, hx=0.25, hy=0.25, hz=0.05, cfg=cfg)
        top = builder.add_body(xform=wp.transform(wp.vec3(0.19, 0.19, 0.1505), wp.quat_identity()))
        builder.add_shape_box(top, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=128,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
            box_box_sat=True,
        )
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=8, mf_warmstart=True)
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        for _k in range(120):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, 1.0 / 60.0)
            s0, s1 = s1, s0
        q = s0.body_q.numpy()[top]
        tilt = float(np.degrees(2.0 * np.arcsin(min(1.0, float(np.linalg.norm(q[3:6]))))))
        test.assertLess(tilt, 1.0, f"corner-supported box tipped {tilt:.2f} deg (manifold lost)")
        test.assertLess(abs(float(q[2]) - 0.1505), 0.01, "corner-supported box lost rest height")


def test_box_box_sat_speculative_approach(test: unittest.TestCase, device):
    """With a speculative pipeline, SAT box-box must produce candidates for
    separated approaching boxes (the clip margin includes the speculative
    extension): a fast-falling box is admitted before contact and lands
    without a deep-penetration spike."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
        base = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_box(base, hx=0.2, hy=0.2, hz=0.05, cfg=cfg)
        top = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()))
        builder.add_shape_box(top, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=128,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
            box_box_sat=True,
            speculative_contact_gap_max=0.1,
        )
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=8, mf_warmstart=True)
        s0, s1 = model.state(), model.state()
        control = model.control()
        joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
        joint_qd[8] = -5.0  # top box falls fast
        s0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
        seen_speculative = False
        min_z = 1.0
        for _k in range(60):
            pipeline.collide(s0, contacts, dt=1.0 / 60.0)
            if not seen_speculative:
                count = int(contacts.rigid_contact_count.numpy()[0])
                z = float(s0.body_q.numpy()[top][2])
                if count > 0 and z > 0.152:  # box-box contacts while separated > 1mm
                    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
                    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
                    if any(0 not in (int(shape0[i]), int(shape1[i])) for i in range(count)):
                        seen_speculative = True
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, 1.0 / 60.0)
            s0, s1 = s1, s0
            min_z = min(min_z, float(s0.body_q.numpy()[top][2]))
        test.assertTrue(seen_speculative, "no speculative box-box contact was generated before touchdown")
        test.assertGreater(min_z, 0.1505 - 0.01, f"deep penetration spike on impact (min z {min_z:.4f})")
        z = float(s0.body_q.numpy()[top][2])
        test.assertLess(abs(z - 0.1505), 0.01, f"box did not settle on the slab (z={z:.4f})")


def test_box_box_sat_honors_shape_margin(test: unittest.TestCase, device):
    """SAT candidate generation includes per-shape collision margins.

    The raw boxes are separated by 15 mm, but their two 10 mm margins
    overlap. The SAT path must therefore emit the same four-corner contact
    manifold instead of rejecting the pair before the shared gap gate.
    """
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, margin=0.01)
        lower = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(lower, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        upper = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.115), wp.quat_identity()))
        builder.add_shape_box(upper, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=64,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
            box_box_sat=True,
        )
        contacts = pipeline.contacts()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertEqual(count, 4, f"shape-margin overlap should produce four SAT contacts, got {count}")


def test_box_box_aligned_manifold_distinct_corners(test: unittest.TestCase, device):
    """Exactly aligned equal boxes emit coincident duplicate candidates from
    different clip lines; the reduction must admit four DISTINCT spread
    corners, not a duplicated corner plus a missing one (a degenerate
    support that collapses under load). The dedup tolerance scales with the
    box extent, so millimeter-size parts keep their distinct corners too."""
    for hx, gap, label in ((0.05, 0.003, "5 cm"), (0.001, 0.0001, "1 mm")):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.rigid_gap = gap
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
            a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, hx), wp.quat_identity()))
            builder.add_shape_box(a, hx=hx, hy=hx, hz=hx, cfg=cfg)
            b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.98 * hx), wp.quat_identity()))
            builder.add_shape_box(b, hx=hx, hy=hx, hz=hx, cfg=cfg)
            # Millimeter cubes fall below the inertia floor; this test only checks contact geometry.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Inertia validation corrected", category=UserWarning)
                model = builder.finalize()
            pipeline = newton.CollisionPipeline(
                model,
                reduce_contacts=True,
                rigid_contact_max=64,
                broad_phase="nxn",
                deterministic=True,
                contact_matching="latest",
                box_box_sat=True,
            )
            contacts = pipeline.contacts()
            state = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertEqual(count, 4, f"{label}: expected a 4-contact manifold, got {count}")
            # contact points on each shape, in world space via body transforms
            p0 = contacts.rigid_contact_point0.numpy()[:count]
            quads = set()
            for i in range(count):
                quads.add((p0[i][0] > 0.0, p0[i][1] > 0.0))
                for j in range(i):
                    d = np.linalg.norm(np.asarray(p0[i]) - np.asarray(p0[j]))
                    test.assertGreater(
                        d, 0.2 * hx, f"{label}: contacts {i},{j} nearly coincident ({d * 1000:.3f} mm apart)"
                    )
            test.assertEqual(len(quads), 4, f"{label}: manifold does not span four corners: {sorted(quads)}")


def test_box_box_sat_feature_keys_survive_convex_key_width(test: unittest.TestCase, device):
    """Keep all four SAT feature ids distinct in convex-only scenes, whose default
    contact sub-key is only a few bits wide, so each contact matches itself next frame."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
        a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_box(a, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3)
        b = builder.add_body(xform=wp.transform(wp.vec3(0.01, 0.0, 0.129), rot))
        builder.add_shape_box(b, hx=0.03, hy=0.03, hz=0.03, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=64,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
            box_box_sat=True,
        )
        contacts = pipeline.contacts()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        pipeline.collide(state, contacts)
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertEqual(count, 4)
        match = contacts.rigid_contact_match_index.numpy()[:count]
        test.assertEqual(sorted(match.tolist()), list(range(count)))
        # The sort key must carry the full generation-time feature id (a Z-face axis code is >= 2).
        bits = pipeline._contact_sort_sub_key_bits
        sub_keys = pipeline._contact_sorter.sorted_keys_view.numpy()[:count] & ((1 << bits) - 1)
        test.assertEqual(len(set(sub_keys.tolist())), count)
        test.assertTrue(np.all(sub_keys >= 64), f"feature ids truncated to {bits} bits: {sub_keys.tolist()}")


class TestBoxBoxSAT(unittest.TestCase):
    pass


add_function_test(
    TestBoxBoxSAT,
    "test_box_box_feature_wrapper_parity",
    test_box_box_feature_wrapper_parity,
    devices=get_test_devices(),
)
add_function_test(
    TestBoxBoxSAT,
    "test_box_box_feature_identity_under_sliding",
    test_box_box_feature_identity_under_sliding,
    devices=get_test_devices(),
)
add_function_test(
    TestBoxBoxSAT,
    "test_box_box_corner_manifold_preserved",
    test_box_box_corner_manifold_preserved,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestBoxBoxSAT,
    "test_box_box_sat_speculative_approach",
    test_box_box_sat_speculative_approach,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestBoxBoxSAT,
    "test_box_box_sat_honors_shape_margin",
    test_box_box_sat_honors_shape_margin,
    devices=get_test_devices(),
)
add_function_test(
    TestBoxBoxSAT,
    "test_box_box_sat_feature_keys_survive_convex_key_width",
    test_box_box_sat_feature_keys_survive_convex_key_width,
    devices=get_test_devices(),
)
add_function_test(
    TestBoxBoxSAT,
    "test_box_box_aligned_manifold_distinct_corners",
    test_box_box_aligned_manifold_distinct_corners,
    devices=get_selected_cuda_test_devices(),
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
