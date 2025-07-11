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

import unittest

import warp as wp

from newton.geometry import (
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CYLINDER,
    GEO_SPHERE,
)
from newton.geometry.gjk import build_ccd_generic

MAX_ITERATIONS = 10

identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


class Model:
    def __init__(self, n_geoms: int):
        self.geom_type = wp.array(shape=(n_geoms,), dtype=int)
        self.geom_dataid = wp.array(shape=(n_geoms,), dtype=int)
        self.geom_size = wp.array2d(shape=(1, n_geoms), dtype=wp.vec3)


class Data:
    def __init__(self, n_geoms: int):
        self.geom_xpos = wp.array2d(shape=(1, n_geoms), dtype=wp.vec3)
        self.geom_xmat = wp.array2d(shape=(1, n_geoms), dtype=wp.mat33)


FLOAT_MIN = -1e30
FLOAT_MAX = 1e30
MJ_MINVAL = 1e-15


@wp.struct
class Geom:
    pos: wp.vec3
    rot: wp.mat33
    normal: wp.vec3
    size: wp.vec3
    index: int


@wp.func
def _support(geom: Geom, geomtype: int, dir: wp.vec3):
    index = -1
    local_dir = wp.transpose(geom.rot) @ dir
    if geomtype == GEO_SPHERE:
        support_pt = geom.pos + geom.size[0] * dir
    elif geomtype == GEO_BOX:
        res = wp.cw_mul(wp.sign(local_dir), geom.size)
        support_pt = geom.rot @ res + geom.pos
    elif geomtype == GEO_CAPSULE:
        res = local_dir * geom.size[0]
        # add cylinder contribution
        res[2] += wp.sign(local_dir[2]) * geom.size[1]
        support_pt = geom.rot @ res + geom.pos
    #   elif geomtype == GEO_ELLIPSOID:
    #     res = wp.cw_mul(local_dir, geom.size)
    #     res = wp.normalize(res)
    #     # transform to ellipsoid
    #     res = wp.cw_mul(res, geom.size)
    #     support_pt = geom.rot @ res + geom.pos
    elif geomtype == GEO_CYLINDER:
        res = wp.vec3(0.0, 0.0, 0.0)
        # set result in XY plane: support on circle
        d = wp.sqrt(wp.dot(local_dir, local_dir))
        if d > MJ_MINVAL:
            scl = geom.size[0] / d
            res[0] = local_dir[0] * scl
            res[1] = local_dir[1] * scl
        # set result in Z direction
        res[2] = wp.sign(local_dir[2]) * geom.size[1]
        support_pt = geom.rot @ res + geom.pos

    return index, support_pt


def _geom_dist(m: Model, d: Data, gid1: int, gid2: int, iterations: int):
    @wp.kernel
    def _gjk_kernel(
        # Model:
        geom_type: wp.array(dtype=int),
        geom_size: wp.array(dtype=wp.vec3),
        # Data in:
        geom_xpos_in: wp.array(dtype=wp.vec3),
        geom_xmat_in: wp.array(dtype=wp.mat33),
        # In:
        gid1: int,
        gid2: int,
        iterations: int,
        vert: wp.array(dtype=wp.vec3),
        vert1: wp.array(dtype=wp.vec3),
        vert2: wp.array(dtype=wp.vec3),
        face: wp.array(dtype=wp.vec3i),
        face_pr: wp.array(dtype=wp.vec3),
        face_norm2: wp.array(dtype=float),
        face_index: wp.array(dtype=int),
        map: wp.array(dtype=int),
        horizon: wp.array(dtype=int),
        # Out:
        dist_out: wp.array(dtype=float),
        pos_out: wp.array(dtype=wp.vec3),
    ):
        geom1 = Geom()
        geom1.index = -1
        geomtype1 = geom_type[gid1]
        geom1.pos = geom_xpos_in[gid1]
        geom1.rot = geom_xmat_in[gid1]
        geom1.size = geom_size[gid1]

        geom2 = Geom()
        geom2.index = -1
        geomtype2 = geom_type[gid2]
        geom2.pos = geom_xpos_in[gid2]
        geom2.rot = geom_xmat_in[gid2]
        geom2.size = geom_size[gid2]

        x_1 = geom_xpos_in[gid1]
        x_2 = geom_xpos_in[gid2]

        (
            dist,
            x1,
            x2,
        ) = wp.static(build_ccd_generic(_support))(
            1e-6,
            iterations,
            iterations,
            geom1,
            geom2,
            geomtype1,
            geomtype2,
            x_1,
            x_2,
            vert,
            vert1,
            vert2,
            face,
            face_pr,
            face_norm2,
            face_index,
            map,
            horizon,
        )

        dist_out[0] = dist
        pos_out[0] = x1
        pos_out[1] = x2

    vert = wp.array(shape=(iterations,), dtype=wp.vec3)
    vert1 = wp.array(shape=(iterations,), dtype=wp.vec3)
    vert2 = wp.array(shape=(iterations,), dtype=wp.vec3)
    face = wp.array(shape=(2 * iterations,), dtype=wp.vec3i)
    face_pr = wp.array(shape=(2 * iterations,), dtype=wp.vec3)
    face_norm2 = wp.array(shape=(2 * iterations,), dtype=float)
    face_index = wp.array(shape=(2 * iterations,), dtype=int)
    map = wp.array(shape=(2 * iterations,), dtype=int)
    horizon = wp.array(shape=(2 * iterations,), dtype=int)
    dist_out = wp.array(shape=(1,), dtype=float)
    pos_out = wp.array(shape=(2,), dtype=wp.vec3)
    wp.launch(
        _gjk_kernel,
        dim=(1,),
        inputs=[
            m.geom_type,
            m.geom_size,
            d.geom_xpos,
            d.geom_xmat,
            gid1,
            gid2,
            iterations,
            vert,
            vert1,
            vert2,
            face,
            face_pr,
            face_norm2,
            face_index,
            map,
            horizon,
        ],
        outputs=[
            dist_out,
            pos_out,
        ],
    )
    return dist_out.numpy()[0], pos_out.numpy()[0], pos_out.numpy()[1]


class TestGJK(unittest.TestCase):
    """Tests for GJK/EPA."""

    def test_spheres_distance(self):
        """Test distance between two spheres."""

        # Create model and state
        m = Model(2)
        d = Data(2)

        # Add two spheres
        m.geom_type = wp.array([GEO_SPHERE, GEO_SPHERE], dtype=int)
        m.geom_size = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3)
        m.geom_dataid = wp.array([0, 0], dtype=int)

        # Set positions
        d.geom_xpos = wp.array([wp.vec3(-1.5, 0.0, 0.0), wp.vec3(1.5, 0.0, 0.0)], dtype=wp.vec3)
        d.geom_xmat = wp.array([identity, identity], dtype=wp.mat33)

        dist, _, _ = _geom_dist(m, d, 0, 1, MAX_ITERATIONS)
        self.assertEqual(1.0, dist)

    def test_spheres_touching(self):
        """Test two touching spheres have zero distance"""

        # Create model and state
        m = Model(2)
        d = Data(2)

        # Add two spheres
        m.geom_type = wp.array([GEO_SPHERE, GEO_SPHERE], dtype=int)
        m.geom_size = wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3)
        m.geom_dataid = wp.array([0, 0], dtype=int)

        # Set positions
        d.geom_xpos = wp.array([wp.vec3(-1.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3)
        d.geom_xmat = wp.array([identity, identity], dtype=wp.mat33)

        dist, _, _ = _geom_dist(m, d, 0, 1, MAX_ITERATIONS)
        self.assertEqual(0.0, dist)

    def test_sphere_sphere_contact(self):
        """Test penetration depth between two spheres."""

        # Create model and state
        m = Model(2)
        d = Data(2)

        # Add two spheres
        m.geom_type = wp.array([GEO_SPHERE, GEO_SPHERE], dtype=int)
        m.geom_size = wp.array([wp.vec3(3.0, 0.0, 0.0), wp.vec3(3.0, 0.0, 0.0)], dtype=wp.vec3)
        m.geom_dataid = wp.array([0, 0], dtype=int)

        # Set positions
        d.geom_xpos = wp.array([wp.vec3(-1.0, 0.0, 0.0), wp.vec3(3.0, 0.0, 0.0)], dtype=wp.vec3)
        d.geom_xmat = wp.array([identity, identity], dtype=wp.mat33)

        # TODO(kbayes): use margin trick instead of EPA for penetration recovery
        dist, _, _ = _geom_dist(m, d, 0, 1, 500)
        self.assertAlmostEqual(-2, dist)

    def test_box_box_contact(self):
        """Test penetration between two boxes."""

        # Create model and state
        m = Model(2)
        d = Data(2)

        # Add two boxes
        m.geom_type = wp.array([GEO_BOX, GEO_BOX], dtype=int)
        m.geom_size = wp.array([wp.vec3(2.5, 2.5, 2.5), wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3)
        m.geom_dataid = wp.array([0, 0], dtype=int)

        # Set positions
        d.geom_xpos = wp.array([wp.vec3(-1.0, 0.0, 0.0), wp.vec3(1.5, 0.0, 0.0)], dtype=wp.vec3)
        d.geom_xmat = wp.array([identity, identity], dtype=wp.mat33)

        dist, x1, x2 = _geom_dist(m, d, 0, 1, MAX_ITERATIONS)
        self.assertAlmostEqual(-1, dist)
        normal = wp.normalize(x1 - x2)
        self.assertAlmostEqual(normal[0], 1)
        self.assertAlmostEqual(normal[1], 0)
        self.assertAlmostEqual(normal[2], 0)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
