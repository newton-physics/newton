# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full-surface (edge/face) rigid-soft coupling onto proxy bodies.

Covers ``_harvest_vbd_body_particle_contact_forces_on_proxy_bodies_kernel``, which reads the
unified ``soft_contact_indices`` / ``soft_contact_barycentric`` records so a soft edge/face
contact reacts on a proxy-coupled rigid body. The kernel is launched on one hand-seeded record,
so the reaction is deterministic and does not depend on SDF collision detection.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.vbd_coupling_kernels import (
    _harvest_vbd_body_particle_contact_forces_on_proxy_bodies_kernel,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()

_PENETRATION = 0.05  # depth of the soft feature below the rigid surface [m]
_NORMAL = (0.0, 0.0, 1.0)  # rigid contact normal, out of the surface
_KE = 1.0e4  # contact stiffness seeded directly, so the expected force is ke * penetration
# v2 is lifted off the z = 0 plane on purpose: with the triangle perpendicular to the normal the
# penetration depth is the same for every barycentric weighting, so a harvest that ignored ``bary``
# and used corner 0 alone would still pass. The offset makes the contact point depend on the weights.
_VERTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])


def _run_proxy_harvest(device, corners, bary):
    """Seed one soft contact record and return the harvested proxy-body wrench.

    ``corners`` is the -1-padded ``vec3i`` of soft particle ids and ``bary`` the matching weights,
    so ``(p, -1, -1)`` / ``(1, 0, 0)`` is a particle record and ``(p0, p1, p2)`` a face record.
    Body and particles are static (prev == cur) and damping/friction are zero, so the reaction is
    a pure normal force at the barycentric contact point.
    """
    builder = newton.ModelBuilder()
    builder.gravity = (0.0, 0.0, 0.0)
    inertia = wp.mat33(1.0e-2, 0.0, 0.0, 0.0, 1.0e-2, 0.0, 0.0, 0.0, 1.0e-2)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0), wp.quat_identity()), mass=1.0, inertia=inertia)
    builder.add_shape_box(body=body, hx=1.0, hy=1.0, hz=1.0)
    for v in _VERTS:
        builder.add_particle(wp.vec3(*v), wp.vec3(0.0), 0.1, radius=0.0)
    builder.add_triangle(0, 1, 2)
    model = builder.finalize(device=device)

    # Place the rigid point _PENETRATION above x = sum_i bary_i * v_i along +z, so that
    # penetration = -dot(n, x - bx) = _PENETRATION > 0.
    active = [c for c in corners if c >= 0]
    contact_x = (np.array(bary)[: len(active), None] * _VERTS[: len(active)]).sum(axis=0)
    body_pos = contact_x + np.array(_NORMAL) * _PENETRATION

    smax = 8
    contacts = newton.Contacts(0, smax, device=device)

    def _set(arr, value):
        a = arr.numpy()
        a[0] = value
        arr.assign(a)

    contacts.soft_contact_count.assign([1])
    _set(contacts.soft_contact_indices, list(corners))
    _set(contacts.soft_contact_barycentric, list(bary))
    _set(contacts.soft_contact_shape, 0)
    _set(contacts.soft_contact_body_pos, list(body_pos))
    _set(contacts.soft_contact_normal, list(_NORMAL))

    out_body_f = wp.zeros(1, dtype=wp.spatial_vector, device=device)
    wp.launch(
        _harvest_vbd_body_particle_contact_forces_on_proxy_bodies_kernel,
        dim=smax,
        inputs=[
            0.01,  # dt
            wp.array([0], dtype=int, device=device),  # body 0 -> proxy global 0
            model.particle_q,
            model.particle_q,  # prev == cur
            model.particle_radius,
            model.body_q,
            model.body_q,  # prev == cur
            wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device),
            model.body_com,
            1.0,  # friction_epsilon
            wp.full(smax, _KE, dtype=float, device=device),
            wp.zeros(smax, dtype=float, device=device),  # material_kd
            wp.zeros(smax, dtype=float, device=device),  # material_mu
            contacts.soft_contact_count,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
            model.shape_margin,
            model.shape_body,
        ],
        outputs=[out_body_f],
        device=device,
    )
    # Body sits at identity, so the world contact point is body_pos.
    return out_body_f.numpy()[0], body_pos, model.body_com.numpy()[body]


def _assert_reaction(test, wrench, cp_world, com_world):
    """The reaction is ke * penetration along -normal, with the matching contact-point torque."""
    force, torque = wrench[:3], wrench[3:]
    np.testing.assert_allclose(force, [0.0, 0.0, -_KE * _PENETRATION], rtol=2e-4, atol=1e-4)
    np.testing.assert_allclose(torque, np.cross(cp_world - com_world, force), rtol=2e-4, atol=1e-5)


def test_face_contact_reacts_on_proxy_body(test, device):
    """A soft FACE record, dropped by the pre-change per-particle harvest, reacts on the proxy body."""
    _assert_reaction(test, *_run_proxy_harvest(device, [0, 1, 2], [0.6, 0.3, 0.1]))


def test_particle_contact_reacts_on_proxy_body(test, device):
    """A particle record ``(p, -1, -1)`` still yields the identical analytic reaction."""
    _assert_reaction(test, *_run_proxy_harvest(device, [0, -1, -1], [1.0, 0.0, 0.0]))


class TestVBDProxyFullSurfaceContact(unittest.TestCase):
    pass


add_function_test(
    TestVBDProxyFullSurfaceContact,
    "test_face_contact_reacts_on_proxy_body",
    test_face_contact_reacts_on_proxy_body,
    devices=devices,
)
add_function_test(
    TestVBDProxyFullSurfaceContact,
    "test_particle_contact_reacts_on_proxy_body",
    test_particle_contact_reacts_on_proxy_body,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main()
