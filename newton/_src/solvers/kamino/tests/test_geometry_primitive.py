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

"""Unit tests for the collider functions of narrow-phase collision detection"""

import unittest

import numpy as np
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.types import float32, int32, vec2i, vec6f
from newton._src.solvers.kamino.geometry.contacts import DEFAULT_GEOM_PAIR_CONTACT_MARGIN, Contacts
from newton._src.solvers.kamino.geometry.primitive import (
    BoundingVolumeType,
    CollisionPipelinePrimitive,
)
from newton._src.solvers.kamino.geometry.primitive.broadphase import (
    PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES,
    BoundingVolumesData,
    CollisionCandidatesData,
    CollisionCandidatesModel,
    nxn_broadphase_aabb,
    nxn_broadphase_bs,
    update_geoms_aabb,
    update_geoms_bs,
)
from newton._src.solvers.kamino.geometry.primitive.narrowphase import (
    PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS,
    primitive_narrowphase,
)
from newton._src.solvers.kamino.models import builders as test_builders
from newton._src.solvers.kamino.utils import logger as msg

###
# Constants
###


nominal_expected_contacts_per_shape_pair = {
    ("sphere", "sphere"): 1,
    ("sphere", "cylinder"): 1,
    ("sphere", "capsule"): 1,
    ("sphere", "box"): 1,
    # TODO: ("sphere", "plane"): 1,
    ("cylinder", "sphere"): 1,
    # TODO: ("cylinder", "plane"): 4,
    ("capsule", "sphere"): 1,
    ("capsule", "capsule"): 1,
    ("capsule", "box"): 1,
    # TODO: ("capsule", "plane"): 1,
    ("box", "box"): 4,
    # TODO: ("box", "plane"): 4,
    # TODO: ("ellipsoid", "plane"): 1,
}
"""
Defines the expected number of contacts for each supported
shape combination under the following idealized conditions:
- all shapes are perfectly stacked along the vertical (z) axis
- all shapes are centered at the origin in the (x,y) plane
- the geoms are perfectly touching (i.e. penetration is exactly zero)
- all contact margins are set to zero
- all shapes are positioned and oriented in configurations
that would would generate a "nominal" number of contacts per shape combination

Notes:
- We refer to these "nominal" expected contacts as those that are neither the worst-case
(i.e. maximum possible contacts) nor the best-case (i.e. minimum possible contacts).
- An example of a "nominal" configuration is a box-on-box arrangement where two boxes are
perfectly aligned and touching face-to-face, generating 4 contact points. The worst-case
would be if the boxes were slightly offset, generating 8 contact points (i.e. full face-face
contact with 4 points on each face). The best-case would be if the boxes were touching at a
single edge or corner, generating only 1 contact point.
"""


###
# Test Scaffolding
###


class PrimitiveBroadPhaseTestBS:
    def __init__(self, builder: ModelBuilder, device: Devicelike = None):
        # Retrieve the number of world
        num_worlds = builder.num_worlds
        num_geoms = len(builder.collision_geoms)
        # Construct collision pairs
        world_num_geom_pairs, model_geom_pair, model_pairid, model_wid = builder.make_collision_candidate_pairs()
        model_num_geom_pairs = len(model_geom_pair)
        # Allocate the collision model data
        with wp.ScopedDevice(device):
            # Allocate the bounding volumes data
            self.bvdata = BoundingVolumesData(radius=wp.zeros(shape=(num_geoms,), dtype=float32))
            # Allocate the time-invariant collision candidates model
            self._cmodel = CollisionCandidatesModel(
                num_model_geom_pairs=model_num_geom_pairs,
                num_world_geom_pairs=world_num_geom_pairs,
                model_num_pairs=wp.array([model_num_geom_pairs], dtype=int32),
                world_num_pairs=wp.array(world_num_geom_pairs, dtype=int32),
                wid=wp.array(model_wid, dtype=int32),
                pairid=wp.array(model_pairid, dtype=int32),
                geom_pair=wp.array(model_geom_pair, dtype=vec2i),
            )
            # Allocate the time-varying collision candidates data
            self._cdata = CollisionCandidatesData(
                num_model_geom_pairs=model_num_geom_pairs,
                model_num_collisions=wp.zeros(shape=(1,), dtype=int32),
                world_num_collisions=wp.zeros(shape=(num_worlds,), dtype=int32),
                wid=wp.zeros(shape=(model_num_geom_pairs,), dtype=int32),
                geom_pair=wp.zeros(shape=(model_num_geom_pairs,), dtype=vec2i),
            )

    def collide(self, model: Model, data: ModelData, default_margin: float = 0.0):
        self._cdata.clear()
        update_geoms_bs(data.bodies.q_i, model.cgeoms, data.cgeoms, self.bvdata, default_margin)
        nxn_broadphase_bs(model.cgeoms, data.cgeoms, self.bvdata, self._cmodel, self._cdata)


class PrimitiveBroadPhaseTestAABB:
    def __init__(self, builder: ModelBuilder, device: Devicelike = None):
        # Retrieve the number of world
        num_worlds = builder.num_worlds
        num_geoms = len(builder.collision_geoms)
        # Construct collision pairs
        world_num_geom_pairs, model_geom_pair, model_pairid, model_wid = builder.make_collision_candidate_pairs()
        model_num_geom_pairs = len(model_geom_pair)
        # Allocate the collision model data
        with wp.ScopedDevice(device):
            # Allocate the bounding volumes data
            self.bvdata = BoundingVolumesData(aabb=wp.zeros(shape=(num_geoms,), dtype=vec6f))
            # Allocate the time-invariant collision candidates model
            self._cmodel = CollisionCandidatesModel(
                num_model_geom_pairs=model_num_geom_pairs,
                num_world_geom_pairs=world_num_geom_pairs,
                model_num_pairs=wp.array([model_num_geom_pairs], dtype=int32),
                world_num_pairs=wp.array(world_num_geom_pairs, dtype=int32),
                wid=wp.array(model_wid, dtype=int32),
                pairid=wp.array(model_pairid, dtype=int32),
                geom_pair=wp.array(model_geom_pair, dtype=vec2i),
            )
            # Allocate the time-varying collision candidates data
            self._cdata = CollisionCandidatesData(
                num_model_geom_pairs=model_num_geom_pairs,
                model_num_collisions=wp.zeros(shape=(1,), dtype=int32),
                world_num_collisions=wp.zeros(shape=(num_worlds,), dtype=int32),
                wid=wp.zeros(shape=(model_num_geom_pairs,), dtype=int32),
                geom_pair=wp.zeros(shape=(model_num_geom_pairs,), dtype=vec2i),
            )

    def collide(self, model: Model, data: ModelData, default_margin: float = 0.0):
        self._cdata.clear()
        update_geoms_aabb(data.bodies.q_i, model.cgeoms, data.cgeoms, self.bvdata, default_margin)
        nxn_broadphase_aabb(model.cgeoms, self.bvdata, self._cmodel, self._cdata)


PrimitiveBroadPhaseType = PrimitiveBroadPhaseTestBS | PrimitiveBroadPhaseTestAABB
"""Type alias for all primitive broad-phase implementations."""

###
# Testing Operations
###


def check_broadphase_allocations(
    testcase: unittest.TestCase,
    builder: ModelBuilder,
    broadphase: PrimitiveBroadPhaseType,
):
    # Calculate the maximum number of geometry pairs
    _, model_geom_pairs, *_ = builder.make_collision_candidate_pairs()
    num_geom_pairs = len(model_geom_pairs)
    # Construct a broad-phase
    testcase.assertEqual(broadphase._cmodel.num_model_geom_pairs, num_geom_pairs)
    testcase.assertEqual(sum(broadphase._cmodel.num_world_geom_pairs), num_geom_pairs)
    testcase.assertEqual(broadphase._cmodel.model_num_pairs.size, 1)
    testcase.assertEqual(broadphase._cmodel.world_num_pairs.size, builder.num_worlds)
    testcase.assertEqual(broadphase._cmodel.wid.size, num_geom_pairs)
    testcase.assertEqual(broadphase._cmodel.pairid.size, num_geom_pairs)
    testcase.assertEqual(broadphase._cmodel.geom_pair.size, num_geom_pairs)
    np.testing.assert_array_equal(broadphase._cmodel.geom_pair.numpy(), model_geom_pairs)
    testcase.assertEqual(broadphase._cdata.model_num_collisions.size, 1)
    testcase.assertEqual(broadphase._cdata.world_num_collisions.size, builder.num_worlds)
    testcase.assertEqual(broadphase._cdata.wid.size, num_geom_pairs)
    testcase.assertEqual(broadphase._cdata.geom_pair.size, num_geom_pairs)


def test_broadphase(
    testcase: unittest.TestCase,
    broadphase_type: PrimitiveBroadPhaseType,
    builder: ModelBuilder,
    expected_model_collisions: int,
    expected_world_collisions: list[int],
    expected_worlds: list[int] | None = None,
    margin: float = 0.0,
    case_name: str = "",
    device: Devicelike = None,
):
    """
    Tests a primitive broad-phase backend on a system specified via a ModelBuilder.
    """
    # Create a test model and data
    model = builder.finalize(device)
    data = model.data()

    # Create a broad-phase backend
    broadphase = broadphase_type(builder=builder, device=device)
    check_broadphase_allocations(testcase, builder, broadphase)

    # Perform broad-phase collision detection and check results
    broadphase.collide(model, data, default_margin=margin)

    # Check overall collision counts
    num_model_collisions = broadphase._cdata.model_num_collisions.numpy()[0]
    np.testing.assert_array_equal(
        actual=num_model_collisions,
        desired=expected_model_collisions,
        err_msg=f"\n{broadphase_type.__name__}: Failed `model_num_collisions` check for {case_name}\n",
    )
    np.testing.assert_array_equal(
        actual=broadphase._cdata.world_num_collisions.numpy(),
        desired=expected_world_collisions,
        err_msg=f"\n{broadphase_type.__name__}: Failed `world_num_collisions` check for {case_name}\n",
    )

    # Skip per-collision pair checks if there are no active collisions
    if num_model_collisions == 0:
        return

    # Run per-collision checks
    if expected_worlds is not None:
        np.testing.assert_array_equal(
            actual=broadphase._cdata.wid.numpy()[:num_model_collisions],
            desired=expected_worlds,
            err_msg=f"\n{broadphase_type.__name__}: Failed `wid` check for {case_name}\n",
        )


def test_broadphase_on_single_pair(
    testcase: unittest.TestCase,
    broadphase_type: PrimitiveBroadPhaseType,
    shape_pair: tuple[str, str],
    expected_collisions: int,
    distance: float = 0.0,
    margin: float = 0.0,
    device: Devicelike = None,
):
    """
    Tests a primitive broad-phase backend on a single shape pair.
    """
    # Create a test model builder, model, and data
    builder = test_builders.make_single_shape_pair_builder(shapes=shape_pair, distance=distance)

    # Run the broad-phase test
    test_broadphase(
        testcase,
        broadphase_type,
        builder,
        expected_collisions,
        [expected_collisions],
        margin=margin,
        case_name=f"shape_pair='{shape_pair}', distance={distance}, margin={margin}",
        device=device,
    )


def check_contacts(
    contacts: Contacts,
    expected: dict,
    header: str,
    case: str,
    rtol: float = 1e-6,
    atol: float = 0.0,
):
    """
    Checks the contents of a Contacts container against expected values.
    """
    # Run contact counts checks
    if "model_num_contacts" in expected:
        np.testing.assert_equal(
            actual=int(contacts.model_num_contacts.numpy()[0]),
            desired=int(expected["model_num_contacts"]),
            err_msg=f"\n{header}: Failed `model_num_contacts` check for `{case}`\n",
        )
    if "world_num_contacts" in expected:
        np.testing.assert_equal(
            actual=contacts.world_num_contacts.numpy(),
            desired=expected["world_num_contacts"],
            err_msg=f"\n{header}: Failed `world_num_contacts` check for `{case}`\n",
        )

    # Skip per-contact checks if there are no active contacts
    num_active = contacts.model_num_contacts.numpy()[0]
    if num_active == 0:
        return

    # Run per-contact assignment checks
    if "wid" in expected:
        np.testing.assert_equal(
            actual=contacts.wid.numpy()[:num_active],
            desired=np.zeros((num_active,), dtype=np.int32),
            err_msg=f"\n{header}: Failed `wid` check for `{case}`\n",
        )
    if "cid" in expected:
        np.testing.assert_equal(
            actual=contacts.cid.numpy()[:num_active],
            desired=np.arange(num_active, dtype=np.int32),
            err_msg=f"\n{header}: Failed `cid` check for `{case}`\n",
        )

    # Run per-contact detailed checks
    if "gid_AB" in expected:
        np.testing.assert_equal(
            actual=contacts.gid_AB.numpy()[:num_active],
            desired=expected["gid_AB"],
            err_msg=f"\n{header}: Failed `gid_AB` check for `{case}`\n",
        )
    if "bid_AB" in expected:
        np.testing.assert_equal(
            actual=contacts.bid_AB.numpy()[:num_active],
            desired=expected["bid_AB"],
            err_msg=f"\n{header}: Failed `bid_AB` check for `{case}`\n",
        )
    if "position_A" in expected:
        np.testing.assert_allclose(
            actual=contacts.position_A.numpy()[:num_active],
            desired=expected["position_A"],
            rtol=rtol,
            atol=atol,
            err_msg=f"\n{header}: Failed `position_A` check for `{case}`\n",
        )
    if "position_B" in expected:
        np.testing.assert_allclose(
            actual=contacts.position_B.numpy()[:num_active],
            desired=expected["position_B"],
            rtol=rtol,
            atol=atol,
            err_msg=f"\n{header}: Failed `position_B` check for `{case}`\n",
        )
    if "gapfunc" in expected:
        np.testing.assert_allclose(
            actual=contacts.gapfunc.numpy()[:num_active],
            desired=expected["gapfunc"],
            rtol=rtol,
            atol=atol,
            err_msg=f"{header}: Failed `gapfunc` check for `{case}`",
        )
    if "frame" in expected:
        np.testing.assert_allclose(
            actual=contacts.frame.numpy()[:num_active],
            desired=expected["frame"],
            rtol=rtol,
            atol=atol,
            err_msg=f"\n{header}: Failed `frame` check for `{case}`\n",
        )


def test_narrowphase(
    testcase: unittest.TestCase,
    builder: ModelBuilder,
    expected: dict,
    max_contacts_per_pair: int = 8,
    margin: float = 0.0,
    rtol: float = 1e-6,
    atol: float = 0.0,
    case: str = "",
    device: Devicelike = None,
):
    """
    Runs the primitive narrow-phase collider using all broad-phase backends
    on a system specified via a ModelBuilder and checks the results.
    """
    # Run the narrow-phase test over each broad-phase backend
    broadphase_types = [PrimitiveBroadPhaseTestAABB, PrimitiveBroadPhaseTestBS]
    for bp_type in broadphase_types:
        bp_name = bp_type.__name__
        msg.info("Running narrow-phase test on '%s' using '%s'", case, bp_name)

        # Create a test model and data
        model = builder.finalize(device)
        data = model.data()

        # Create a broad-phase backend
        broadphase = bp_type(builder=builder, device=device)
        broadphase.collide(model, data, default_margin=margin)

        # Calculate expected model geom pairs
        num_world_geom_pairs, *_ = builder.make_collision_candidate_pairs()

        # Create a contacts container
        capacity = [ngp * max_contacts_per_pair for ngp in num_world_geom_pairs]
        contacts = Contacts(capacity=capacity, device=device)
        contacts.clear()

        # Execute narrowphase for primitive shapes
        primitive_narrowphase(model, data, broadphase._cdata, contacts, margin)

        # Optional verbose output
        msg.debug("[%s][%s]: bodies.q_i:\n%s", case, bp_name, data.bodies.q_i)
        msg.debug("[%s][%s]: contacts.model_num_contacts: %s", case, bp_name, contacts.model_num_contacts)
        msg.debug("[%s][%s]: contacts.world_num_contacts: %s", case, bp_name, contacts.world_num_contacts)
        msg.debug("[%s][%s]: contacts.wid: %s", case, bp_name, contacts.wid)
        msg.debug("[%s][%s]: contacts.cid: %s", case, bp_name, contacts.cid)
        msg.debug("[%s][%s]: contacts.gid_AB:\n%s", case, bp_name, contacts.gid_AB)
        msg.debug("[%s][%s]: contacts.bid_AB:\n%s", case, bp_name, contacts.bid_AB)
        msg.debug("[%s][%s]: contacts.position_A:\n%s", case, bp_name, contacts.position_A)
        msg.debug("[%s][%s]: contacts.position_B:\n%s", case, bp_name, contacts.position_B)
        msg.debug("[%s][%s]: contacts.gapfunc:\n%s", case, bp_name, contacts.gapfunc)
        msg.debug("[%s][%s]: contacts.frame:\n%s", case, bp_name, contacts.frame)
        msg.debug("[%s][%s]: contacts.material:\n%s", case, bp_name, contacts.material)

        # Check results
        check_contacts(
            contacts,
            expected,
            rtol=rtol,
            atol=atol,
            case=f"{case} using {bp_name}",
            header="primitive narrow-phase",
        )


def test_narrowphase_on_shape_pair(
    testcase: unittest.TestCase,
    shape_pair: tuple[str, str],
    expected_contacts: int,
    distance: float = 0.0,
    margin: float = 0.0,
    builder_kwargs: dict | None = None,
):
    """
    Tests the primitive narrow-phase collider on a single shape pair.

    Note:
        This test only checks the number of generated contacts.
    """
    # Set default builder kwargs if none provided
    if builder_kwargs is None:
        builder_kwargs = {}

    # Create a builder for the specified shape pair
    builder = test_builders.make_single_shape_pair_builder(shapes=shape_pair, distance=distance, **builder_kwargs)

    # Define expected contacts dictionary
    expected = {
        "model_num_contacts": expected_contacts,
        "world_num_contacts": [expected_contacts],
    }

    # Run the narrow-phase test
    test_narrowphase(
        testcase=testcase,
        builder=builder,
        expected=expected,
        margin=margin,
        case=f"shape_pair='{shape_pair}'",
        device=testcase.default_device,
    )


###
# Tests
###


class TestPrimitiveBroadPhase(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

        # Construct a list of all supported primitive shape pairs
        self.supported_shape_pairs: list[tuple[str, str]] = []
        for shape_A in PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES:
            shape_A_name = shape_A.name.lower()
            for shape_B in PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES:
                shape_B_name = shape_B.name.lower()
                self.supported_shape_pairs.append((shape_A_name, shape_B_name))
        msg.debug("supported_shape_pairs:\n%s\n", self.supported_shape_pairs)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_bspheres_on_each_primitive_shape_pair_exact(self):
        # Each shape pair in its own world with
        # - zero distance: (i.e., exactly touching)
        # - zero margin: no preemption of collisions
        for shape_pair in self.supported_shape_pairs:
            msg.info("[BS]: testing broadphase with exact boundaries on shape pair: %s", shape_pair)
            test_broadphase_on_single_pair(
                self,
                broadphase_type=PrimitiveBroadPhaseTestBS,
                shape_pair=shape_pair,
                expected_collisions=1,
                distance=0.0,
                margin=0.0,
                device=self.default_device,
            )

    def test_02_bspheres_on_each_primitive_shape_pair_apart(self):
        # Each shape pair in its own world with
        # - positive distance: (i.e., apart)
        # - zero margin: no preemption of collisions
        for shape_pair in self.supported_shape_pairs:
            msg.info("[BS]: testing broadphase with shapes apart on shape pair: %s", shape_pair)
            test_broadphase_on_single_pair(
                self,
                broadphase_type=PrimitiveBroadPhaseTestBS,
                shape_pair=shape_pair,
                expected_collisions=0,
                distance=1.5,
                margin=0.0,
                device=self.default_device,
            )

    def test_03_bspheres_on_each_primitive_shape_pair_apart_with_margin(self):
        # Each shape pair in its own world with
        # - positive distance: (i.e., apart)
        # - positive margin: preemption of collisions
        for shape_pair in self.supported_shape_pairs:
            msg.info("[BS]: testing broadphase with shapes apart but margin on shape pair: %s", shape_pair)
            test_broadphase_on_single_pair(
                self,
                broadphase_type=PrimitiveBroadPhaseTestBS,
                shape_pair=shape_pair,
                expected_collisions=1,
                distance=1.0,
                margin=1.0,
                device=self.default_device,
            )

    def test_04_bspheres_on_each_primitive_shape_pair_with_overlap(self):
        # Each shape pair in its own world with
        # - negative distance: (i.e., overlapping)
        # - zero margin: no preemption of collisions
        for shape_pair in self.supported_shape_pairs:
            msg.info("[BS]: testing broadphase with overlapping shapes on shape pair: %s", shape_pair)
            test_broadphase_on_single_pair(
                self,
                broadphase_type=PrimitiveBroadPhaseTestBS,
                shape_pair=shape_pair,
                expected_collisions=1,
                distance=-0.01,
                margin=0.0,
                device=self.default_device,
            )

    def test_05_bspheres_on_all_primitive_shape_pairs(self):
        # All shape pairs, but each in its own world with
        # - zero distance: (i.e., exactly touching)
        # - zero margin: no preemption of collisions
        msg.info("[BS]: testing broadphase with overlapping shapes on all shape pairs")
        builder = test_builders.make_shape_pairs_builder(
            shape_pairs=self.supported_shape_pairs,
            distance=0.0,
        )
        test_broadphase(
            self,
            builder=builder,
            broadphase_type=PrimitiveBroadPhaseTestBS,
            expected_model_collisions=len(self.supported_shape_pairs),
            expected_world_collisions=[1] * len(self.supported_shape_pairs),
            margin=0.0,
            case_name="all shape pairs",
            device=self.default_device,
        )

    def test_06_aabbs_on_each_primitive_shape_pair_exact(self):
        # Each shape pair in its own world with
        # - zero distance: (i.e., exactly touching)
        # - zero margin: no preemption of collisions
        for shape_pair in self.supported_shape_pairs:
            msg.info("[AABB]: testing broadphase with exact boundaries on shape pair: %s", shape_pair)
            test_broadphase_on_single_pair(
                self,
                broadphase_type=PrimitiveBroadPhaseTestAABB,
                shape_pair=shape_pair,
                expected_collisions=1,
                distance=0.0,
                margin=0.0,
                device=self.default_device,
            )

    def test_07_aabbs_on_each_primitive_shape_pair_apart(self):
        # Each shape pair in its own world with
        # - positive distance: (i.e., apart)
        # - zero margin: no preemption of collisions
        for shape_pair in self.supported_shape_pairs:
            msg.info("[AABB]: testing broadphase with shapes apart on shape pair: %s", shape_pair)
            test_broadphase_on_single_pair(
                self,
                broadphase_type=PrimitiveBroadPhaseTestAABB,
                shape_pair=shape_pair,
                expected_collisions=0,
                distance=1e-6,
                margin=0.0,
                device=self.default_device,
            )

    def test_08_aabbs_on_each_primitive_shape_pair_apart_with_margin(self):
        # Each shape pair in its own world with
        # - positive distance: (i.e., apart)
        # - positive margin: preemption of collisions
        for shape_pair in self.supported_shape_pairs:
            msg.info("[AABB]: testing broadphase with shapes apart but margin on shape pair: %s", shape_pair)
            test_broadphase_on_single_pair(
                self,
                broadphase_type=PrimitiveBroadPhaseTestAABB,
                shape_pair=shape_pair,
                expected_collisions=1,
                distance=1e-6,
                margin=1e-6,
                device=self.default_device,
            )

    def test_09_aabbs_on_each_primitive_shape_pair_with_overlap(self):
        # Each shape pair in its own world with
        # - negative distance: (i.e., overlapping)
        # - zero margin: no preemption of collisions
        for shape_pair in self.supported_shape_pairs:
            msg.info("[AABB]: testing broadphase with overlapping shapes on shape pair: %s", shape_pair)
            test_broadphase_on_single_pair(
                self,
                broadphase_type=PrimitiveBroadPhaseTestAABB,
                shape_pair=shape_pair,
                expected_collisions=1,
                distance=-0.01,
                margin=0.0,
                device=self.default_device,
            )

    def test_10_aabbs_on_all_primitive_shape_pairs(self):
        # All shape pairs, but each in its own world with
        # - zero distance: (i.e., exactly touching)
        # - zero margin: no preemption of collisions
        msg.info("[AABB]: testing broadphase with overlapping shapes on all shape pairs")
        builder = test_builders.make_shape_pairs_builder(
            shape_pairs=self.supported_shape_pairs,
            distance=0.0,
        )
        test_broadphase(
            self,
            builder=builder,
            broadphase_type=PrimitiveBroadPhaseTestAABB,
            expected_model_collisions=len(self.supported_shape_pairs),
            expected_world_collisions=[1] * len(self.supported_shape_pairs),
            expected_worlds=list(range(len(self.supported_shape_pairs))),
            margin=0.0,
            case_name="all shape pairs",
            device=self.default_device,
        )

    def test_11_bspheres_on_boxes_nunchaku(self):
        msg.info("[BS]: testing broadphase on `boxes_nunchaku`")
        builder = test_builders.build_boxes_nunchaku()
        test_broadphase(
            self,
            builder=builder,
            broadphase_type=PrimitiveBroadPhaseTestBS,
            expected_model_collisions=3,
            expected_world_collisions=[3],
            expected_worlds=[0, 0, 0],
            margin=0.0,
            case_name="boxes_nunchaku",
            device=self.default_device,
        )

    def test_12_aabbs_on_boxes_nunchaku(self):
        msg.info("[AABB]: testing broadphase on `boxes_nunchaku`")
        builder = test_builders.build_boxes_nunchaku()
        test_broadphase(
            self,
            builder=builder,
            broadphase_type=PrimitiveBroadPhaseTestAABB,
            expected_model_collisions=3,
            expected_world_collisions=[3],
            expected_worlds=[0, 0, 0],
            margin=0.0,
            case_name="boxes_nunchaku",
            device=self.default_device,
        )


class TestPrimitiveNarrowPhase(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

        # Construct a list of all supported primitive shape pairs
        self.supported_shape_pairs: list[tuple[str, str]] = []
        for shape_A in PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES:
            shape_A_name = shape_A.name.lower()
            for shape_B in PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES:
                shape_B_name = shape_B.name.lower()
                if (shape_A, shape_B) in PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS:
                    self.supported_shape_pairs.append((shape_A_name, shape_B_name))
        msg.debug("supported_shape_pairs:\n%s\n", self.supported_shape_pairs)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_on_each_primitive_shape_pair_exact(self):
        """
        Tests the narrow-phase collision detection for each supported primitive
        shape pair when placed exactly at their contact boundaries.
        """
        msg.info("Testing narrow-phase tests with exact boundaries")
        # Each shape pair in its own world with
        # - zero distance: (i.e., exactly touching)
        # - zero margin: no preemption of collisions
        for shape_pair in nominal_expected_contacts_per_shape_pair.keys():
            # Define any special kwargs for specific shape pairs
            kwargs = {}
            if shape_pair == ("box", "box"):
                # NOTE: To asses "nominal" contacts for box-box,
                # we need to specify larger box dimensions for
                # the bottom box to avoid contacts on edges
                kwargs["bottom_dims"] = (2.0, 2.0, 1.0)

            # Retrieve the nominal expected contacts for the shape pair
            expected_contacts = nominal_expected_contacts_per_shape_pair.get(shape_pair, 0)

            # Run the narrow-phase test on the shape pair
            test_narrowphase_on_shape_pair(
                self,
                shape_pair=shape_pair,
                expected_contacts=expected_contacts,
                margin=0.0,  # No contact margin
                distance=0.0,  # Exactly touching
                builder_kwargs=kwargs,
            )

    def test_02_on_each_primitive_shape_pair_apart(self):
        """
        Tests the narrow-phase collision detection for each
        supported primitive shape pair when placed apart.
        """
        msg.info("Testing narrow-phase tests with shapes apart")
        # Each shape pair in its own world with
        # - zero distance: (i.e., exactly touching)
        # - zero margin: no preemption of collisions
        for shape_pair in nominal_expected_contacts_per_shape_pair.keys():
            test_narrowphase_on_shape_pair(
                self,
                shape_pair=shape_pair,
                expected_contacts=0,
                margin=0.0,  # No contact margin
                distance=1e-6,  # Shapes apart
            )

    def test_03_on_each_primitive_shape_pair_apart_with_margin(self):
        """
        Tests the narrow-phase collision detection for each supported
        primitive shape pair when placed apart but with contact margin.
        """
        msg.info("Testing narrow-phase tests with shapes apart")
        # Each shape pair in its own world with
        # - zero distance: (i.e., exactly touching)
        # - zero margin: no preemption of collisions
        for shape_pair in nominal_expected_contacts_per_shape_pair.keys():
            # Define any special kwargs for specific shape pairs
            kwargs = {}
            if shape_pair == ("box", "box"):
                # NOTE: To asses "nominal" contacts for box-box,
                # we need to specify larger box dimensions for
                # the bottom box to avoid contacts on edges
                kwargs["bottom_dims"] = (2.0, 2.0, 1.0)

            # Retrieve the nominal expected contacts for the shape pair
            expected_contacts = nominal_expected_contacts_per_shape_pair.get(shape_pair, 0)

            # Run the narrow-phase test on the shape pair
            test_narrowphase_on_shape_pair(
                self,
                shape_pair=shape_pair,
                expected_contacts=expected_contacts,
                margin=1e-6,  # Contact margin
                distance=1e-6,  # Shapes apart
                builder_kwargs=kwargs,
            )

    ###
    # Tests for special cases of shape combinations/configurations
    ###

    def test_04_on_sphere_on_sphere_full(self):
        """
        Tests all narrow-phase output data for the case of two spheres
        stacked along the vertical (z) axis, centered at the origin
        in the (x,y) plane, and slightly penetrating each other.
        """
        # NOTE: We set to negative value to move the geoms into each other,
        # i.e. move the bottom geom upwards and the top geom downwards.
        distance = -0.01

        # Define expected contact data
        expected = {
            "model_num_contacts": 1,
            "world_num_contacts": [1],
            "gid_AB": np.array([[0, 1]], dtype=np.int32),
            "bid_AB": np.array([[0, 1]], dtype=np.int32),
            "position_A": np.array([[0.0, 0.0, 0.5 * abs(distance)]], dtype=np.float32),
            "position_B": np.array([[0.0, 0.0, -0.5 * abs(distance)]], dtype=np.float32),
            "gapfunc": np.array([[0.0, 0.0, 1.0, distance]], dtype=np.float32),
            "frame": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        }

        # Create a builder for the specified shape pair
        builder = test_builders.make_single_shape_pair_builder(
            shapes=("sphere", "sphere"),
            distance=distance,
        )

        # Run the narrow-phase test on the shape pair
        test_narrowphase(
            self,
            builder=builder,
            expected=expected,
            max_contacts_per_pair=2,
            case="sphere_on_sphere_detailed",
            device=self.default_device,
        )

    def test_05_box_on_box_with_four_points(self):
        """
        Tests all narrow-phase output data for the case of two boxes
        stacked along the vertical (z) axis, centered at the origin
        in the (x,y) plane, and slightly penetrating each other.

        This test makes the bottom box larger in the (x,y) dimensions
        to ensure that only four contact points are generated at the
        face of the top box.
        """
        # NOTE: We set to negative value to move the geoms into each other,
        # i.e. move the bottom geom upwards and the top geom downwards.
        distance = -0.01

        # Define expected contact data
        expected = {
            "model_num_contacts": 4,
            "world_num_contacts": [4],
            "gid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(4, 1)),
            "bid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(4, 1)),
            "position_A": np.array(
                [
                    [-0.5, -0.5, 0.5 * abs(distance)],
                    [0.5, -0.5, 0.5 * abs(distance)],
                    [-0.5, 0.5, 0.5 * abs(distance)],
                    [0.5, 0.5, 0.5 * abs(distance)],
                ],
                dtype=np.float32,
            ),
            "position_B": np.array(
                [
                    [-0.5, -0.5, -0.5 * abs(distance)],
                    [0.5, -0.5, -0.5 * abs(distance)],
                    [-0.5, 0.5, -0.5 * abs(distance)],
                    [0.5, 0.5, -0.5 * abs(distance)],
                ],
                dtype=np.float32,
            ),
            "gapfunc": np.tile(np.array([0.0, 0.0, 1.0, distance], dtype=np.float32), reps=(4, 1)),
            "frame": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), reps=(4, 1)),
        }

        # Create a builder for the specified shape pair
        builder = test_builders.make_single_shape_pair_builder(
            shapes=("box", "box"),
            distance=distance,
            bottom_dims=(2.0, 2.0, 1.0),  # Larger bottom box
        )

        # Run the narrow-phase test on the shape pair
        test_narrowphase(
            self,
            builder=builder,
            expected=expected,
            case="box_on_box_four_points",
            device=self.default_device,
        )

    def test_06_box_on_box_eight_points(self):
        """
        Tests the narrow-phase collision detection for a special case of
        two boxes stacked along the vertical (z) axis, centered at the origin
        in the (x,y) plane, and slightly penetrating each other.
        """
        # NOTE: We set to negative value to move the geoms into each other,
        # i.e. move the bottom geom upwards and the top geom downwards.
        distance = -0.01

        # Define expected contact data
        expected = {
            "model_num_contacts": 8,
            "world_num_contacts": [8],
            "gid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(8, 1)),
            "bid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(8, 1)),
            "position_A": np.array(
                [
                    [-0.207107, -0.5, 0.5 * abs(distance)],
                    [0.207107, -0.5, 0.5 * abs(distance)],
                    [-0.5, -0.207107, 0.5 * abs(distance)],
                    [-0.5, 0.207107, 0.5 * abs(distance)],
                    [0.5, 0.207107, 0.5 * abs(distance)],
                    [0.5, -0.207107, 0.5 * abs(distance)],
                    [0.207107, 0.5, 0.5 * abs(distance)],
                    [-0.207107, 0.5, 0.5 * abs(distance)],
                ],
                dtype=np.float32,
            ),
            "position_B": np.array(
                [
                    [-0.207107, -0.5, -0.5 * abs(distance)],
                    [0.207107, -0.5, -0.5 * abs(distance)],
                    [-0.5, -0.207107, -0.5 * abs(distance)],
                    [-0.5, 0.207107, -0.5 * abs(distance)],
                    [0.5, 0.207107, -0.5 * abs(distance)],
                    [0.5, -0.207107, -0.5 * abs(distance)],
                    [0.207107, 0.5, -0.5 * abs(distance)],
                    [-0.207107, 0.5, -0.5 * abs(distance)],
                ],
                dtype=np.float32,
            ),
            "gapfunc": np.tile(np.array([0.0, 0.0, 1.0, distance], dtype=np.float32), reps=(8, 1)),
            "frame": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), reps=(8, 1)),
        }

        # Create a builder for the specified shape pair
        builder = test_builders.make_single_shape_pair_builder(
            shapes=("box", "box"),
            distance=distance,
            top_rpy=[0.0, 0.0, np.pi / 4],
        )

        # Run the narrow-phase test on the shape pair
        test_narrowphase(
            self,
            builder=builder,
            expected=expected,
            case="box_on_box_eight_points",
            device=self.default_device,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_07_on_box_on_box_one_point(self):
        """
        Tests the narrow-phase collision detection for a special case of
        two boxes stacked along the vertical (z) axis, centered at the origin
        in the (x,y) plane, and the top box rotated so two diagonally opposing corners
        lie exactly on the Z-axis. thus the bottom corner of the top box touches the
        top face of the bottom box at a single point, slightly penetrating each other.
        """
        # NOTE: We set to negative value to move the geoms into each other,
        # i.e. move the bottom geom upwards and the top geom downwards.
        penetration = -0.01

        # Define expected contact data
        expected = {
            "num_contacts": 1,
            "gid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(1, 1)),
            "bid_AB": np.tile(np.array([0, 1], dtype=np.int32), reps=(1, 1)),
            "position_A": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "position_B": np.array([[0.0, 0.0, -abs(penetration)]], dtype=np.float32),
            "gapfunc": np.tile(np.array([0.0, 0.0, 1.0, penetration], dtype=np.float32), reps=(1, 1)),
            "frame": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), reps=(1, 1)),
        }

        # Create a builder for the specified shape pair
        builder = test_builders.make_single_shape_pair_builder(
            shapes=("box", "box"),
            top_xyz=[0.0, 0.0, 0.5 * np.sqrt(3) + 0.5],
            top_rpy=[np.pi / 4, -np.arctan(1.0 / np.sqrt(2)), 0.0],
        )

        # Run the narrow-phase test on the shape pair
        test_narrowphase(
            self,
            builder=builder,
            expected=expected,
            case="box_on_box_one_point",
            device=self.default_device,
            rtol=1e-5,
            atol=1e-6,
        )


class TestPipelinePrimitive(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default(self):
        """Tests the default constructor of CollisionPipelinePrimitive."""
        pipeline = CollisionPipelinePrimitive()
        self.assertIsNone(pipeline._device)
        self.assertEqual(pipeline._bvtype, BoundingVolumeType.AABB)
        self.assertEqual(pipeline._default_margin, DEFAULT_GEOM_PAIR_CONTACT_MARGIN)
        self.assertRaises(RuntimeError, pipeline.collide, Model(), ModelData(), Contacts())

    def test_02_make_and_collide(self):
        """
        Tests the construction and execution
        of the CollisionPipelinePrimitive on
        all supported primitive shape pairs.
        """
        # Create a list of collidable shape pairs and their reversed versions
        collidable_shape_pairs = list(nominal_expected_contacts_per_shape_pair.keys())
        msg.debug("collidable_shape_pairs:\n%s\n", collidable_shape_pairs)

        # Define any special kwargs for specific shape pairs
        per_shape_pair_args = {}
        per_shape_pair_args[("box", "box")] = {
            # NOTE: To asses "nominal" contacts for box-box,
            # we need to specify larger box dimensions for
            # the bottom box to avoid contacts on edges
            "bottom_dims": (2.0, 2.0, 1.0)
        }

        # Create a builder for all supported shape pairs
        builder = test_builders.make_shape_pairs_builder(
            shape_pairs=collidable_shape_pairs, per_shape_pair_args=per_shape_pair_args
        )
        model = builder.finalize(device=self.default_device)
        data = model.data()

        # Create a contacts container
        num_world_geom_pairs, *_ = builder.make_collision_candidate_pairs()
        capacity = [ngp * 8 for ngp in num_world_geom_pairs]
        contacts = Contacts(capacity=capacity, device=self.default_device)
        contacts.clear()

        # Create the collision pipeline
        pipeline = CollisionPipelinePrimitive(builder=builder, device=self.default_device)

        # Run collision detection
        pipeline.collide(model, data, contacts)

        # Create a list of expected number of contacts per shape pair
        expected_contacts_per_pair: list[int] = list(nominal_expected_contacts_per_shape_pair.values())
        msg.debug("expected_contacts_per_pair:\n%s\n", expected_contacts_per_pair)

        # Define expected contacts dictionary
        expected = {
            "model_num_contacts": sum(expected_contacts_per_pair),
            "world_num_contacts": np.array(expected_contacts_per_pair, dtype=np.int32),
        }

        # Check results
        check_contacts(
            contacts,
            expected,
            case="all shape pairs",
            header="pipeline primitive narrow-phase",
        )


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=20000, threshold=20000, precision=10, suppress=True)

    # Global warp configurations
    wp.config.verbose = False
    wp.config.verify_fp = False
    wp.config.verify_cuda = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
