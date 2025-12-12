# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KAMINO: UNIT TESTS: COMPARISON UTILITIES
"""

import unittest

import numpy as np

from newton._src.solvers.kamino.core.builder import ModelBuilder

###
# Array-like comparisons
###


def lists_equal(list1, list2) -> bool:
    return np.array_equal(list1, list2)


def arrays_equal(arr1, arr2, tolerance=1e-6) -> bool:
    return np.allclose(arr1, arr2, atol=tolerance)


def matrices_equal(m1, m2, tolerance=1e-6) -> bool:
    return np.allclose(m1, m2, atol=tolerance)


def vectors_equal(v1, v2, tolerance=1e-6) -> bool:
    return np.allclose(v1, v2, atol=tolerance)


###
# Container comparisons
###


def assert_builders_equal(
    fixture: unittest.TestCase,
    builder1: ModelBuilder,
    builder2: ModelBuilder,
    skip_colliders: bool = False,
):
    """
    Compares two ModelBuilder instances for equality.
    """
    fixture.assertEqual(builder1.num_bodies, builder2.num_bodies)
    fixture.assertEqual(builder1.num_joints, builder2.num_joints)
    fixture.assertEqual(builder1.num_physical_geoms, builder2.num_physical_geoms)
    fixture.assertEqual(builder1.num_collision_geoms, builder2.num_collision_geoms)
    fixture.assertEqual(builder1.num_materials, builder2.num_materials)
    for i in range(builder1.num_bodies):
        fixture.assertEqual(builder1.bodies[i].wid, builder2.bodies[i].wid)
        fixture.assertEqual(builder1.bodies[i].bid, builder2.bodies[i].bid)
        fixture.assertAlmostEqual(builder1.bodies[i].m_i, builder2.bodies[i].m_i)
        fixture.assertTrue(matrices_equal(builder1.bodies[i].i_I_i, builder2.bodies[i].i_I_i))
        fixture.assertTrue(vectors_equal(builder1.bodies[i].q_i_0, builder2.bodies[i].q_i_0))
        fixture.assertTrue(vectors_equal(builder1.bodies[i].u_i_0, builder2.bodies[i].u_i_0))
    for j in range(builder1.num_joints):
        fixture.assertEqual(builder1.joints[j].wid, builder2.joints[j].wid)
        fixture.assertEqual(builder1.joints[j].jid, builder2.joints[j].jid)
        fixture.assertEqual(builder1.joints[j].act_type, builder2.joints[j].act_type)
        fixture.assertEqual(builder1.joints[j].dof_type, builder2.joints[j].dof_type)
        fixture.assertEqual(builder1.joints[j].cts_offset, builder2.joints[j].cts_offset)
        fixture.assertEqual(builder1.joints[j].dofs_offset, builder2.joints[j].dofs_offset)
        fixture.assertEqual(builder1.joints[j].bid_B, builder2.joints[j].bid_B)
        fixture.assertEqual(builder1.joints[j].bid_F, builder2.joints[j].bid_F)
        fixture.assertTrue(
            vectors_equal(builder1.joints[j].B_r_Bj, builder2.joints[j].B_r_Bj),
            f"Joint {j} B_r_Bj mismatch:\nleft:\n{builder1.joints[j].B_r_Bj}\nright:\n{builder2.joints[j].B_r_Bj}",
        )
        fixture.assertTrue(
            vectors_equal(builder1.joints[j].F_r_Fj, builder2.joints[j].F_r_Fj),
            f"Joint {j} F_r_Fj mismatch:\nleft:\n{builder1.joints[j].F_r_Fj}\nright:\n{builder2.joints[j].F_r_Fj}",
        )
        fixture.assertTrue(
            matrices_equal(builder1.joints[j].X_j, builder2.joints[j].X_j),
            f"Joint {j} X_j mismatch:\nleft:\n{builder1.joints[j].X_j}\nright:\n{builder2.joints[j].X_j}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].q_j_min, builder2.joints[j].q_j_min),
            f"Joint {j} q_j_min mismatch:\nleft:\n{builder1.joints[j].q_j_min}\nright:\n{builder2.joints[j].q_j_min}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].q_j_max, builder2.joints[j].q_j_max),
            f"Joint {j} q_j_max mismatch:\nleft:\n{builder1.joints[j].q_j_max}\nright:\n{builder2.joints[j].q_j_max}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].dq_j_max, builder2.joints[j].dq_j_max),
            f"Joint {j} dq_j_max mismatch:\nleft:\n{builder1.joints[j].dq_j_max}\nright:\n{builder2.joints[j].dq_j_max}",
        )
        fixture.assertTrue(
            arrays_equal(builder1.joints[j].tau_j_max, builder2.joints[j].tau_j_max),
            f"Joint {j} tau_j_max mismatch:\nleft:\n{builder1.joints[j].tau_j_max}\nright:\n{builder2.joints[j].tau_j_max}",
        )
    for k in range(builder1.num_collision_geoms):
        fixture.assertEqual(builder1.collision_geoms[k].wid, builder2.collision_geoms[k].wid)
        fixture.assertEqual(builder1.collision_geoms[k].gid, builder2.collision_geoms[k].gid)
        fixture.assertEqual(builder1.collision_geoms[k].lid, builder2.collision_geoms[k].lid)
        fixture.assertEqual(builder1.collision_geoms[k].bid, builder2.collision_geoms[k].bid)
        fixture.assertEqual(builder1.collision_geoms[k].shape.type, builder2.collision_geoms[k].shape.type)
        fixture.assertEqual(builder1.collision_geoms[k].shape.num_params, builder2.collision_geoms[k].shape.num_params)
        fixture.assertTrue(
            lists_equal(builder1.collision_geoms[k].shape.paramsvec, builder2.collision_geoms[k].shape.paramsvec)
        )
        fixture.assertEqual(builder1.collision_geoms[k].mid, builder2.collision_geoms[k].mid)
        fixture.assertEqual(builder1.collision_geoms[k].max_contacts, builder2.collision_geoms[k].max_contacts)
        if not skip_colliders:
            fixture.assertEqual(builder1.collision_geoms[k].group, builder2.collision_geoms[k].group)
            fixture.assertEqual(builder1.collision_geoms[k].collides, builder2.collision_geoms[k].collides)
    for m in range(builder1.num_materials):
        fixture.assertEqual(builder1.materials[m].wid, builder2.materials[m].wid)
        fixture.assertEqual(builder1.materials[m].mid, builder2.materials[m].mid)
        fixture.assertEqual(builder1.materials[m].density, builder2.materials[m].density)
        fixture.assertEqual(builder1.materials[m].restitution, builder2.materials[m].restitution)
        fixture.assertEqual(builder1.materials[m].static_friction, builder2.materials[m].static_friction)
        fixture.assertEqual(builder1.materials[m].dynamic_friction, builder2.materials[m].dynamic_friction)
