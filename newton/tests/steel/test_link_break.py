# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Weld (FIXED) loop closures on SolverSteel, and link-break loops built from them.

A link-break loop keeps every joint of the loop as a tree joint: one body of the loop is split into a primary and a
shadow that share its mass, the loop's last joint attaches to the shadow, and a FIXED loop joint welds the shadow to
its primary. The fixtures below build such models by hand, the way a user would.
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_cuda_test_devices

AXIS_Y = wp.vec3(0.0, 1.0, 0.0)


def _rod_inertia(mass, length, direction, radius=0.008):
    u = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    perp = mass * (3.0 * radius**2 + length**2) / 12.0
    axial = mass * radius**2 / 2.0
    inertia = perp * (np.eye(3) - np.outer(u, u)) + axial * np.outer(u, u)
    return wp.mat33(*inertia.flatten().tolist())


def _add_link_with_shadow(builder, xform, mass, inertia, label):
    """Split one body into a primary and a shadow at the same pose, each with half its mass and inertia."""
    half = wp.mat33(*(np.array(inertia, dtype=float).reshape(3, 3) / 2.0).flatten().tolist())
    primary = builder.add_link(xform=xform, mass=mass / 2.0, inertia=half, label=label)
    shadow = builder.add_link(xform=xform, mass=mass / 2.0, inertia=half, label=f"{label}_shadow")
    return primary, shadow


def _four_bar_links(theta0_deg=60.0):
    """Ends [m] and masses [kg] of the crank, coupler and rocker of a planar crank-rocker with the crank at ``theta0``.

    Ground 0.30, crank 0.10, coupler 0.25, rocker 0.20 m, in the x-z plane: the crank runs from A (origin) to B, the
    coupler from B to C, the rocker from D (0.3, 0, 0) to C.
    """
    theta0 = np.radians(theta0_deg)
    ground, crank, coupler, rocker = 0.30, 0.10, 0.25, 0.20
    a = np.array([0.0, 0.0, 0.0])
    d = np.array([ground, 0.0, 0.0])
    b = a + crank * np.array([np.cos(theta0), 0.0, np.sin(theta0)])
    db = d - b
    L = np.linalg.norm(db)
    along = (coupler**2 - rocker**2 + L**2) / (2 * L)
    h = np.sqrt(coupler**2 - along**2)
    perp = np.array([-db[2], 0.0, db[0]]) / L
    c = b + along * db / L + h * perp
    return {"crank": (a, b, 0.2), "coupler": (b, c, 0.4), "rocker": (d, c, 0.3)}


def _four_bar(closure, gravity=-9.81, theta0_deg=60.0):
    """Planar crank-rocker (x-z plane, hinges about y): ground 0.30, crank 0.10, coupler 0.25, rocker 0.20 m.

    Hinges A (world-crank), B (crank-coupler) and D (world-rocker) are tree joints. The loop is closed at hinge C
    (coupler-rocker) by ``closure``:

    * ``"link"``: link-break. C is a tree joint from the rocker to a shadow of the coupler, and a FIXED loop joint
      welds the shadow to the coupler.
    * ``"joint"``: joint-break. C is a REVOLUTE loop joint outside the articulation.

    Returns the model builder, the user bodies by name, their COMs [m], the hinges as ``(joint, parent, child,
    anchor_world)`` and the C joint.
    """
    ends = _four_bar_links(theta0_deg)
    (a, b, _), (_, c, _), (d, _, _) = ends["crank"], ends["coupler"], ends["rocker"]
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, gravity))
    body, com = {}, {}
    for name, (p, q, m) in ends.items():
        com[name] = 0.5 * (p + q)
        xform = wp.transform(wp.vec3(*com[name]), wp.quat_identity())
        inertia = _rod_inertia(m, np.linalg.norm(q - p), q - p)
        if name == "coupler" and closure == "link":
            body[name], body["coupler_shadow"] = _add_link_with_shadow(builder, xform, m, inertia, name)
            com["coupler_shadow"] = com[name]
        else:
            body[name] = builder.add_link(xform=xform, mass=m, inertia=inertia, label=name)

    def hinge(parent, child, point, label):
        xp = wp.transform(wp.vec3(*(point - (com[parent] if parent else 0.0))), wp.quat_identity())
        xc = wp.transform(wp.vec3(*(point - com[child])), wp.quat_identity())
        return builder.add_joint_revolute(
            parent=body[parent] if parent else -1,
            child=body[child],
            axis=AXIS_Y,
            parent_xform=xp,
            child_xform=xc,
            label=label,
        )

    tree = [
        hinge(None, "crank", a, "A"),
        hinge("crank", "coupler", b, "B"),
        hinge(None, "rocker", d, "D"),
    ]
    if closure == "link":
        j_c = hinge("rocker", "coupler_shadow", c, "C")
        builder.add_articulation([*tree, j_c])
        builder.add_joint_fixed(
            parent=body["coupler"],
            child=body["coupler_shadow"],
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            label="coupler_weld",
        )
    else:
        builder.add_articulation(tree)
        j_c = hinge("rocker", "coupler", c, "C")
    hinges = [(tree[0], None, "crank", a), (tree[1], "crank", "coupler", b), (j_c, "rocker", "coupler", c)]
    hinges.append((tree[2], None, "rocker", d))
    return builder, body, com, hinges, j_c


def _anchor_gaps(body_q, body, com, hinges):
    """Separation [m] of every hinge's two anchor copies, measured on the user bodies (primaries) only."""

    def point(name, p):
        q = body_q[body[name]]
        return np.array(wp.transform_point(wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:])), wp.vec3(*(p - com[name]))))

    gaps = []
    for _, parent, child, p in hinges:
        pa = p if parent is None else point(parent, p)
        gaps.append(float(np.linalg.norm(pa - point(child, p))))
    return gaps


def _crank_angle(body_q, body, com, hinges):
    a = hinges[0][3]
    b = hinges[1][3]
    q = body_q[body["crank"]]
    xf = wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:]))
    pa = np.array(wp.transform_point(xf, wp.vec3(*(a - com["crank"]))))
    pb = np.array(wp.transform_point(xf, wp.vec3(*(b - com["crank"]))))
    return float(np.arctan2(pb[2] - pa[2], pb[0] - pa[0]))


def _simulate(model, steps, dt, iterations=16, history=False):
    """Step ``model`` on SolverSteel and return the final state.

    With ``history``, also return ``body_q`` [steps, body_count, 7] and ``body_qd`` [steps, body_count, 6] after every
    step. They are recorded into device buffers and read back once.
    """
    solver = SolverSteel(model, iterations=iterations)
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s1)
    control = model.control()
    n = model.body_count
    if history:
        body_q = wp.empty(steps * n, dtype=wp.transform, device=model.device)
        body_qd = wp.empty(steps * n, dtype=wp.spatial_vector, device=model.device)
    for i in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        if history:
            wp.copy(body_q, s0.body_q, dest_offset=i * n, count=n)
            wp.copy(body_qd, s0.body_qd, dest_offset=i * n, count=n)
    if not history:
        return s0
    return s0, body_q.numpy().reshape(steps, n, 7), body_qd.numpy().reshape(steps, n, 6)


def _four_bar_series(body_q, body, com, hinges):
    """Crank angle [rad] and largest hinge-anchor gap [m] at every step of a ``body_q`` history."""
    return np.array(
        [(_crank_angle(bq, body, com, hinges), max(_anchor_gaps(bq, body, com, hinges))) for bq in body_q]
    ).T


def _four_bar_reference(t_end, dt=2.0e-4, theta0_deg=60.0):
    """Crank angle [rad] of the ``_four_bar`` mechanism from an independent 1-DOF Lagrangian, on ``dt`` steps.

    The loop is solved in closed form for the crank angle ``th``; ``M(th) th'' + M'(th) th'^2 / 2 + V'(th) = 0`` is
    integrated with RK4, the generalized inertia ``M`` and potential ``V`` built from finite differences of the closed
    form. Agrees with a symbolic-derivative DOP853 solution (rtol 1e-12) to 2e-4 deg over the first second.
    """
    ground, lengths = 0.30, {"crank": 0.10, "coupler": 0.25, "rocker": 0.20}
    masses = {"crank": 0.2, "coupler": 0.4, "rocker": 0.3}

    def segments(th):
        b = lengths["crank"] * np.array([np.cos(th), np.sin(th)])
        d = np.array([ground, 0.0])
        db = d - b
        L = np.hypot(*db)
        along = (lengths["coupler"] ** 2 - lengths["rocker"] ** 2 + L**2) / (2 * L)
        h = np.sqrt(lengths["coupler"] ** 2 - along**2)
        c = b + along * db / L + h * np.array([-db[1], db[0]]) / L
        return {"crank": (np.zeros(2), b), "coupler": (b, c), "rocker": (d, c)}

    def inertia_and_potential(th, eps=1.0e-6):
        s0, sp, sm = segments(th), segments(th + eps), segments(th - eps)
        M = V = 0.0
        for name, m in masses.items():
            rod = m * (3 * 0.008**2 + lengths[name] ** 2) / 12.0
            dc = 0.5 * (sum(sp[name]) - sum(sm[name])) / (2 * eps)
            dphi = (np.arctan2(*(sp[name][1] - sp[name][0])[::-1]) - np.arctan2(*(sm[name][1] - sm[name][0])[::-1])) / (
                2 * eps
            )
            M += m * dc @ dc + rod * dphi**2
            V += m * 9.81 * 0.5 * sum(s0[name])[1]
        return M, V

    def accel(q, w, h=1.0e-4):
        M = inertia_and_potential(q)[0]
        (Mp, Vp), (Mm, Vm) = inertia_and_potential(q + h), inertia_and_potential(q - h)
        return -(0.25 * (Mp - Mm) / h * w * w + 0.5 * (Vp - Vm) / h) / M

    q, w = np.radians(theta0_deg), 0.0
    out = [q]
    for _ in range(int(round(t_end / dt))):
        k1 = (w, accel(q, w))
        k2 = (w + 0.5 * dt * k1[1], accel(q + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1]))
        k3 = (w + 0.5 * dt * k2[1], accel(q + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1]))
        k4 = (w + dt * k3[1], accel(q + dt * k3[0], w + dt * k3[1]))
        q += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        w += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        out.append(q)
    return dt * np.arange(len(out)), np.array(out)


def test_weld_loop_holds_a_hinge(test, device):
    """Lock two hinged links' relative rotation with a FIXED loop joint; without it the hinge swings."""

    def build(weld):
        b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
        l1 = b.add_link(
            xform=wp.transform(wp.vec3(0.1, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=_rod_inertia(1.0, 0.2, (1, 0, 0)),
        )
        l2 = b.add_link(
            xform=wp.transform(wp.vec3(0.3, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=_rod_inertia(1.0, 0.2, (1, 0, 0)),
        )
        j0 = b.add_joint_revolute(
            parent=-1,
            child=l1,
            axis=AXIS_Y,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
        )
        j1 = b.add_joint_revolute(
            parent=l1,
            child=l2,
            axis=AXIS_Y,
            parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
        )
        b.add_articulation([j0, j1])
        if weld:
            with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "another joint already connects these bodies"):
                b.add_joint_fixed(
                    parent=l1,
                    child=l2,
                    parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
                    child_xform=wp.transform_identity(),
                )
        return b.finalize(device=device)

    def bend(model):
        q = _simulate(model, 50, 1.0 / 500.0).body_q.numpy()
        r1 = np.array(wp.quat_to_matrix(wp.quat(*q[0, 3:]))).reshape(3, 3)
        r2 = np.array(wp.quat_to_matrix(wp.quat(*q[1, 3:]))).reshape(3, 3)
        rel = r1.T @ r2
        return abs(float(np.degrees(np.arctan2(rel[0, 2], rel[0, 0]))))

    test.assertLess(bend(build(True)), 0.05)
    test.assertGreater(bend(build(False)), 5.0)


def test_link_break_four_bar_matches_joint_break_early(test, device):
    """Follow the joint-break four-bar with link-break while the motion is slow (first 0.1 s), keeping loops shut."""
    series = {}
    for closure in ("link", "joint"):
        builder, body, com, hinges, _ = _four_bar(closure)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = builder.finalize(device=device)
        _, body_q, _ = _simulate(model, 100, 1.0e-3, history=True)
        series[closure] = _four_bar_series(body_q, body, com, hinges)
    (la, lg), (ja, jg) = series["link"], series["joint"]
    test.assertLess(np.degrees(np.abs(la - ja)).max(), 0.1)
    test.assertLess(lg.max(), 5.0e-5)
    test.assertLess(jg.max(), 5.0e-5)
    test.assertLess(la[-1], la[0] - 0.05, "the crank should have fallen")


def test_link_break_four_bar_tracks_ground_truth(test, device):
    """Track the analytic four-bar to first order in dt with link-break, with a loop gap that shrinks like dt^2.

    The crank falls from 60 deg to -40 deg in 0.25 s. Halving the step must roughly halve the peak crank-angle error
    against ``_four_bar_reference`` and quarter the peak anchor gap.
    """
    t_end = 0.25
    ref_t, ref_q = _four_bar_reference(t_end, dt=5.0e-4)  # within 5e-5 deg of the default 2e-4 s reference
    errors, gaps = [], []
    for dt in (1.0e-3, 5.0e-4):
        builder, body, com, hinges, _ = _four_bar("link")
        model = builder.finalize(device=device)
        steps = int(round(t_end / dt))
        _, body_q, _ = _simulate(model, steps, dt, history=True)
        angle, gap = _four_bar_series(body_q, body, com, hinges)
        t = dt * np.arange(1, steps + 1)
        errors.append(float(np.degrees(np.abs(np.unwrap(angle) - np.interp(t, ref_t, ref_q))).max()))
        gaps.append(float(gap.max()))
    test.assertLess(errors[0], 0.6, f"peak crank-angle error {errors[0]:.3f} deg at 1 ms")
    test.assertLess(errors[1] / errors[0], 0.65, f"errors {errors} deg are not first order in dt")
    test.assertLess(gaps[0], 1.5e-4, f"peak loop gap {gaps[0]:.2e} m at 1 ms")
    test.assertLess(gaps[1] / gaps[0], 0.35, f"loop gaps {gaps} m do not shrink like dt^2")


def _four_bar_link_break_stage():
    """The link-break ``_four_bar`` authored in USD: a shadow coupler welded by an excluded ``PhysicsFixedJoint``."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/physicsScene").CreateGravityMagnitudeAttr(9.81)
    # Both branches of the loop live under one articulation root.
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, "/fourbar").GetPrim())
    ends = _four_bar_links()
    com = {}
    for name, split in (("crank", 1.0), ("coupler", 0.5), ("coupler_shadow", 0.5), ("rocker", 1.0)):
        p, q, m = ends[name.removesuffix("_shadow")]
        com[name] = 0.5 * (p + q)
        xform = UsdGeom.Xform.Define(stage, f"/fourbar/{name}")
        xform.AddTranslateOp().Set(Gf.Vec3d(*com[name]))
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        length, radius = np.linalg.norm(q - p), 0.008
        perp = m * (3.0 * radius**2 + length**2) / 12.0
        mass.CreateMassAttr(m * split)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(m * radius**2 / 2.0 * split, perp * split, perp * split))
        angle = -np.arctan2(q[2] - p[2], q[0] - p[0])  # rotation about y that takes x onto the rod
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(np.cos(angle / 2)), 0.0, float(np.sin(angle / 2)), 0.0))

    def hinge(label, parent, child, point):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/fourbar/{label}")
        joint.CreateAxisAttr("Y")
        if parent is not None:
            joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/fourbar/{parent}")])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(point - (com[parent] if parent else 0.0))))
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/fourbar/{child}")])
        joint.CreateLocalPos1Attr(Gf.Vec3f(*(point - com[child])))

    hinge("A", None, "crank", ends["crank"][0])
    hinge("B", "crank", "coupler", ends["crank"][1])
    hinge("D", None, "rocker", ends["rocker"][0])
    hinge("C", "rocker", "coupler_shadow", ends["rocker"][1])
    weld = UsdPhysics.FixedJoint.Define(stage, "/fourbar/coupler_weld")
    weld.CreateBody0Rel().SetTargets([Sdf.Path("/fourbar/coupler")])
    weld.CreateBody1Rel().SetTargets([Sdf.Path("/fourbar/coupler_shadow")])
    weld.CreateExcludeFromArticulationAttr(True)
    return stage


def test_link_break_four_bar_from_usd(test, device):
    """Import a link-break four-bar from USD as one articulation plus the weld that moves like the built one."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    builder.add_usd(_four_bar_link_break_stage())
    usd_model = builder.finalize(device=device)
    test.assertEqual(usd_model.articulation_count, 1)
    weld = usd_model.joint_label.index("/fourbar/coupler_weld")
    test.assertEqual(int(usd_model.joint_type.numpy()[weld]), int(newton.JointType.FIXED))
    test.assertEqual(int(usd_model.joint_articulation.numpy()[weld]), -1)

    built, body, com, hinges, _ = _four_bar("link")
    usd_body = {name: usd_model.body_label.index(f"/fourbar/{name}") for name in ("crank", "coupler", "rocker")}
    angles = []
    for model, bodies in ((usd_model, usd_body), (built.finalize(device=device), body)):
        _, body_q, _ = _simulate(model, 100, 1.0e-3, history=True)
        angles.append(np.array([_crank_angle(bq, bodies, com, hinges) for bq in body_q]))
    test.assertLess(angles[1][-1], angles[1][0] - 0.1, "the crank should have fallen")
    np.testing.assert_allclose(angles[0], angles[1], atol=1.0e-4)


def _floating_loop(closure):
    """A free-floating planar four-link loop (no world hinge, zero gravity) whose hinge 0 is PD-driven open.

    Hinge k joins links k and k + 1 (mod 4). With ``closure="joint"`` a free joint carries link 0, hinges 0-2 are
    tree joints and hinge 3 is a REVOLUTE loop joint. With ``"link"`` the free joint carries the heaviest link (3),
    two short branches 3-2-1 and 3-0-1' grow from it (hinges 2 and 1 reversed), link 1 is split into a primary and a
    shadow 1', and a FIXED loop joint welds 1' to 1. Returns the model and the index of hinge 0's coordinate.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    corners = [np.array(p, dtype=float) for p in ((0.0, 0, 0.0), (0.3, 0, 0.0), (0.3, 0, 0.2), (0.0, 0, 0.25))]
    bodies, com = [], []
    shadow = None
    for k in range(4):
        p, q = corners[k], corners[(k + 1) % 4]
        m = 0.2 + 0.1 * k
        com.append(0.5 * (p + q))
        xform = wp.transform(wp.vec3(*com[k]), wp.quat_identity())
        inertia = _rod_inertia(m, np.linalg.norm(q - p), q - p)
        if k == 1 and closure == "link":
            primary, shadow = _add_link_with_shadow(builder, xform, m, inertia, "link1")
            bodies.append(primary)
        else:
            bodies.append(builder.add_link(xform=xform, mass=m, inertia=inertia))

    def hinge(k, reverse=False, child=None):
        """Hinge k at the corner shared by links k and k + 1, optionally from link k + 1 to link k."""
        point = corners[(k + 1) % 4]
        parent, other = (k + 1) % 4, k
        if not reverse:
            parent, other = other, parent
        return builder.add_joint_revolute(
            parent=bodies[parent],
            child=bodies[other] if child is None else child,
            axis=AXIS_Y,
            parent_xform=wp.transform(wp.vec3(*(point - com[parent])), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(*(point - com[other])), wp.quat_identity()),
        )

    if closure == "joint":
        free = builder.add_joint_free(child=bodies[0])
        tree = [hinge(0), hinge(1), hinge(2)]
        driven = tree[0]
        builder.add_articulation([free, *tree])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hinge(3)
    else:
        free = builder.add_joint_free(child=bodies[3])
        tree = [hinge(2, reverse=True), hinge(1, reverse=True), hinge(3), hinge(0, child=shadow)]
        driven = tree[3]
        builder.add_articulation([free, *tree])
        builder.add_joint_fixed(
            parent=bodies[1], child=shadow, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
        )
    d0 = builder.joint_qd_start[driven]
    builder.joint_target_ke[d0] = 5.0
    builder.joint_target_kd[d0] = 0.1
    builder.joint_target_mode[d0] = int(newton.JointTargetMode.POSITION)
    builder.joint_target_q[builder.joint_q_start[driven]] = 0.6  # internal drive: opens hinge 0
    return builder.finalize(), builder.joint_q_start[driven]


def _momentum_drift(model, q_index, steps=250, dt=1.0e-3):
    """Peak |P| [kg m/s] and |L| [kg m^2/s] of an isolated system that starts at rest, and the driven hinge angle [rad]."""
    mass = model.body_mass.numpy()
    inertia = model.body_inertia.numpy()

    def momentum(bq, bqd):  # body_qd is the COM twist
        c = (mass[:, None] * bq[:, :3]).sum(0) / mass.sum()
        P = (mass[:, None] * bqd[:, :3]).sum(0)
        L = np.zeros(3)
        for i in range(len(mass)):
            R = np.array(wp.quat_to_matrix(wp.quat(*bq[i, 3:]))).reshape(3, 3)
            L += mass[i] * np.cross(bq[i, :3] - c, bqd[i, :3]) + R @ inertia[i] @ R.T @ bqd[i, 3:]
        return np.linalg.norm(P), np.linalg.norm(L)

    state, body_q, body_qd = _simulate(model, steps, dt, history=True)
    series = np.array([momentum(bq, bqd) for bq, bqd in zip(body_q, body_qd, strict=True)])
    return series[:, 0].max(), series[:, 1].max(), abs(float(state.joint_q.numpy()[q_index]))


def test_link_break_floating_loop_conserves_momentum(test, device):
    """Keep the momentum of an internally driven free-floating loop that starts at rest near zero.

    Neither closure style is exact (Steel leaks some linear momentum through the loop rows), so this guards the
    measured ordering at a 1 ms step: link-break, cut opposite the heaviest link with two short branches, leaks several
    times less linear momentum than the joint-break REVOLUTE closure.
    """
    with wp.ScopedDevice(device):
        link_p, link_l, link_q = _momentum_drift(*_floating_loop("link"))
        joint_p, _, joint_q = _momentum_drift(*_floating_loop("joint"))
    test.assertGreater(link_q, 0.5, "the drive should have opened the hinge")
    test.assertAlmostEqual(link_q, joint_q, delta=0.01)
    test.assertLess(link_p, 0.5 * joint_p)
    test.assertLess(link_p, 0.01)
    test.assertLess(link_l, 2.0e-3)


def test_link_break_drives_the_loop_closing_joint(test, device):
    """Drive the loop-closing hinge C, which keeps its coordinate with link-break (joint-break cannot)."""
    builder, body, com, hinges, j_c = _four_bar("link", gravity=0.0)
    dof = builder.joint_qd_start[j_c]
    builder.joint_target_ke[dof] = 20.0
    builder.joint_target_kd[dof] = 1.0
    builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
    target = -0.3
    builder.joint_target_q[builder.joint_q_start[j_c]] = target
    model = builder.finalize(device=device)
    state = _simulate(model, 300, 1.0 / 1000.0)
    test.assertAlmostEqual(float(state.joint_q.numpy()[builder.joint_q_start[j_c]]), target, delta=0.01)
    test.assertLess(max(_anchor_gaps(state.body_q.numpy(), body, com, hinges)), 1.0e-4)


def test_link_break_preelimination_falls_back(test, device):
    """Pre-eliminate one weld (6 rows), and fall back when a second weld exceeds the 8-row capacity."""
    builder, body, com, hinges, _ = _four_bar("link")
    model = builder.finalize(device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver = SolverSteel(model, iterations=2, enable_bilateral_preelimination=True)
    test.assertTrue(solver._preelim_active)
    s0, s1, control = model.state(), model.state(), model.control()
    for _ in range(50):
        solver.step(s0, s1, control, None, 1.0e-3)
        s0, s1 = s1, s0
    test.assertLess(max(_anchor_gaps(s0.body_q.numpy(), body, com, hinges)), 1.0e-5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the second weld parallels the first
        builder.add_joint_fixed(parent=body["coupler"], child=body["coupler_shadow"], label="second_weld")
    model = builder.finalize(device=device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverSteel(model, iterations=16, enable_bilateral_preelimination=True)
    test.assertFalse(solver._preelim_active)
    test.assertTrue(any("pre-elimination capacity" in str(w.message) for w in caught))


def test_weld_to_world_returns_the_short_way(test, device):
    """Return a free body welded to the world 170 deg off about an oblique axis along the short way.

    The planar four-bars leave the x/z rotation rows and the y point row of a weld inert; this loads all six rows. The
    weld anchor is off the COM and the inertia is anisotropic, so the point and rotation rows are coupled.
    """
    axis = wp.normalize(wp.vec3(1.0, 2.0, 3.0))
    pos, anchor = wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.1, 0.05, -0.02)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(pos, wp.quat_from_axis_angle(axis, float(np.radians(170.0)))),
        mass=1.0,
        inertia=_rod_inertia(1.0, 0.3, (1.0, 0.0, 0.0)),
    )
    builder.add_articulation([builder.add_joint_free(child=body)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the weld parallels the free joint on purpose
        builder.add_joint_fixed(
            parent=-1,
            child=body,
            parent_xform=wp.transform(pos + anchor, wp.quat_identity()),
            child_xform=wp.transform(anchor, wp.quat_identity()),
        )
    model = builder.finalize(device=device)
    # The off-COM anchor couples the point and rotation rows.
    _, body_q, _ = _simulate(model, 100, 1.0e-3, iterations=64, history=True)
    xyz, w = body_q[:, 0, 3:6], body_q[:, 0, 6]
    angle = np.degrees(2.0 * np.arctan2(np.linalg.norm(xyz, axis=1), np.abs(w)))
    # The short way shrinks the angle from 170 deg about the same axis; the long way would first pass 180 deg.
    test.assertTrue(np.all(np.diff(angle) < 1.0e-3), f"angle {angle[:5]} ... does not shrink monotonically")
    test.assertGreater(float(np.dot(xyz[0] * np.sign(w[0]), np.array(axis))), 0.0)
    test.assertLess(angle[-1], 0.01)
    test.assertLess(float(np.linalg.norm(body_q[-1, 0, :3] - np.array(pos))), 1.0e-5)


class TestSteelLinkBreak(unittest.TestCase):
    def test_collapse_fixed_joints_warns_on_a_weld_closure(self):
        """Warn when collapsing would open a link-break loop; collapse a joint-break loop and a MuJoCo weld quietly."""
        builder, *_ = _four_bar("link")
        with self.assertWarnsRegex(
            newton.exceptions.NewtonWarning, r"does not preserve FIXED loop-closing joints \('coupler_weld'\)"
        ):
            builder.collapse_fixed_joints()
        builder, *_ = _four_bar("joint")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            builder.collapse_fixed_joints()
        # A converted MuJoCo weld keeps its equality row, so collapsing it does not warn.
        mjcf = """
        <mujoco>
          <worldbody>
            <body name="a" pos="0 0 1">
              <joint type="hinge" axis="0 1 0"/>
              <geom type="sphere" size="0.05"/>
              <body name="b" pos="0.3 0 0">
                <joint type="hinge" axis="0 1 0"/>
                <geom type="sphere" size="0.05"/>
              </body>
            </body>
          </worldbody>
          <equality><weld body1="a" body2="b"/></equality>
        </mujoco>
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            newton.ModelBuilder().add_mjcf(mjcf, collapse_fixed_joints=True)
        self.assertFalse([w for w in caught if "loop-closing" in str(w.message)])
        if USD_AVAILABLE:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            with self.assertWarnsRegex(newton.exceptions.NewtonWarning, r"does not preserve FIXED loop-closing joints"):
                builder.add_usd(_four_bar_link_break_stage(), collapse_fixed_joints=True)


for _name in (
    "test_weld_loop_holds_a_hinge",
    "test_link_break_four_bar_matches_joint_break_early",
    "test_link_break_four_bar_tracks_ground_truth",
    "test_link_break_floating_loop_conserves_momentum",
    "test_link_break_drives_the_loop_closing_joint",
    "test_link_break_preelimination_falls_back",
    "test_weld_to_world_returns_the_short_way",
):
    add_function_test(TestSteelLinkBreak, _name, globals()[_name], devices=get_cuda_test_devices())
if USD_AVAILABLE:
    add_function_test(
        TestSteelLinkBreak,
        "test_link_break_four_bar_from_usd",
        test_link_break_four_bar_from_usd,
        devices=get_cuda_test_devices(),
    )


if __name__ == "__main__":
    unittest.main()
