# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestFeatherstoneTileGemm(unittest.TestCase):
    pass


def _build_18_dof_chain(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]])
    parent = -1
    joints = []

    for index in range(18):
        body = builder.add_link(
            xform=wp.transform((0.1 * index, 0.0, 0.0), wp.quat_identity()),
            com=wp.vec3(0.05, 0.0, 0.0),
            mass=1.0,
            inertia=inertia,
        )
        joint = builder.add_joint_revolute(
            parent=parent,
            child=body,
            axis=(0.0, 1.0, 0.0),
            parent_xform=(
                wp.transform_identity()
                if parent < 0
                else wp.transform((0.1, 0.0, 0.0), wp.quat_identity())
            ),
        )
        joints.append(joint)
        parent = body

    builder.add_articulation(joints)
    return builder.finalize(device=device)


def test_tile_gemm_loads_generated_module(test: TestFeatherstoneTileGemm, device):
    """Verify tile-GEMM construction loads the generated kernel module on the model device."""
    if not device.is_cuda:
        return

    model = _build_18_dof_chain(device)
    with wp.ScopedDevice("cpu"):
        for fuse_cholesky in (False, True):
            with test.subTest(fuse_cholesky=fuse_cholesky):
                solver = newton.solvers.SolverFeatherstone(
                    model,
                    use_tile_gemm=True,
                    fuse_cholesky=fuse_cholesky,
                )
                test.assertTrue(solver.use_tile_gemm)


devices = get_test_devices()
add_function_test(
    TestFeatherstoneTileGemm,
    "test_tile_gemm_loads_generated_module",
    test_tile_gemm_loads_generated_module,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
