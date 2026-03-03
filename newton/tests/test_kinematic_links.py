# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

from newton import BodyFlags, ModelBuilder


class TestKinematicLinks(unittest.TestCase):
    """Tests for kinematic body flag handling."""

    def test_body_flags_persist_through_finalize(self):
        """body_flags array on the finalized Model has correct length and values."""
        builder = ModelBuilder()
        builder.add_body(mass=1.0)
        builder.add_body(mass=0.0, is_kinematic=True)
        builder.add_body(mass=2.0)

        model = builder.finalize()
        flags = model.body_flags.numpy()

        self.assertEqual(len(flags), 3)
        self.assertEqual(flags[0], BodyFlags.DYNAMIC)
        self.assertTrue(flags[1] & BodyFlags.KINEMATIC)
        self.assertEqual(flags[2], BodyFlags.DYNAMIC)

    def test_kinematic_root_link_in_articulation(self):
        """A kinematic root link with dynamic children should be valid."""
        builder = ModelBuilder()
        root = builder.add_link(mass=0.0, is_kinematic=True, label="root")
        child = builder.add_link(mass=1.0, label="child")

        j0 = builder.add_joint_fixed(parent=-1, child=root)
        j1 = builder.add_joint_revolute(
            parent=root,
            child=child,
            axis=(0.0, 0.0, 1.0),
        )
        builder.add_articulation([j0, j1])

        model = builder.finalize()
        flags = model.body_flags.numpy()
        self.assertTrue(flags[root] & BodyFlags.KINEMATIC)
        self.assertEqual(flags[child], BodyFlags.DYNAMIC)

    def test_kinematic_non_root_link_raises(self):
        """A kinematic link attached to a non-world parent must raise ValueError."""
        builder = ModelBuilder()
        root = builder.add_link(mass=1.0, label="root")
        child = builder.add_link(mass=0.0, is_kinematic=True, label="child")

        j0 = builder.add_joint_free(parent=-1, child=root)
        j1 = builder.add_joint_revolute(
            parent=root,
            child=child,
            axis=(0.0, 0.0, 1.0),
        )

        with self.assertRaises(ValueError, msg="Only root bodies"):
            builder.add_articulation([j0, j1])

    def test_kinematic_middle_link_raises(self):
        """A kinematic link in the middle of a chain must raise ValueError."""
        builder = ModelBuilder()
        b0 = builder.add_link(mass=1.0, label="b0")
        b1 = builder.add_link(mass=1.0, is_kinematic=True, label="b1")
        b2 = builder.add_link(mass=1.0, label="b2")

        j0 = builder.add_joint_free(parent=-1, child=b0)
        j1 = builder.add_joint_revolute(parent=b0, child=b1, axis=(0.0, 0.0, 1.0))
        j2 = builder.add_joint_revolute(parent=b1, child=b2, axis=(0.0, 0.0, 1.0))

        with self.assertRaises(ValueError, msg="Only root bodies"):
            builder.add_articulation([j0, j1, j2])


if __name__ == "__main__":
    unittest.main()
