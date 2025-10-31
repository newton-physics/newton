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
KAMINO: UNIT TESTS: CORE: MATERIALS
"""

import unittest

import numpy as np
import warp as wp

# Module to be tested
from newton._src.solvers.kamino.core.materials import (
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    MaterialDescriptor,
    MaterialManager,
    MaterialPairProperties,
)

###
# Tests
###


class TestMaterials(unittest.TestCase):
    def setUp(self):
        self.verbose = False  # Set to True for verbose output

    def test_default_material_pair_properties(self):
        # Create a default-constructed surface material
        material_pair = MaterialPairProperties()

        # Check default values
        self.assertEqual(material_pair.restitution, DEFAULT_RESTITUTION)
        self.assertEqual(material_pair.static_friction, DEFAULT_FRICTION)
        self.assertEqual(material_pair.dynamic_friction, DEFAULT_FRICTION)

    def test_material_manager_default_material(self):
        # Create a default-constructed material manager
        manager = MaterialManager()
        self.assertEqual(manager.num_materials, 1)

        # Create a default-constructed material descriptor
        dm = manager.default

        # Check initial default material values
        self.assertIsInstance(dm, MaterialDescriptor)
        self.assertEqual(dm.name, "default")
        self.assertEqual(type(dm.uid), str)

        # Check initial material-pair properties
        mp = manager.pairs
        self.assertEqual(len(mp), 1)
        self.assertEqual(len(mp[0]), 1)
        self.assertEqual(mp[0][0].restitution, DEFAULT_RESTITUTION)
        self.assertEqual(mp[0][0].static_friction, DEFAULT_FRICTION)
        self.assertEqual(mp[0][0].dynamic_friction, DEFAULT_FRICTION)

        # Check restitution matrix of the default material
        drm = manager.restitution_matrix()
        self.assertEqual(drm.shape, (1, 1))
        self.assertEqual(drm[0, 0], DEFAULT_RESTITUTION)

        # Check the static friction matrix of the default material
        dfm = manager.static_friction_matrix()
        self.assertEqual(dfm.shape, (1, 1))
        self.assertEqual(dfm[0, 0], DEFAULT_FRICTION)

        # Check the dynamic friction matrix of the default material
        dym = manager.dynamic_friction_matrix()
        self.assertEqual(dym.shape, (1, 1))
        self.assertEqual(dym[0, 0], DEFAULT_FRICTION)

        # Modify the default material properties
        manager.configure_pair(
            first="default",
            second="default",
            material_pair=MaterialPairProperties(restitution=0.5, static_friction=0.5, dynamic_friction=0.5),
        )

        # Check modified material-pair properties
        mp = manager.pairs
        self.assertEqual(len(mp), 1)
        self.assertEqual(len(mp[0]), 1)
        self.assertEqual(mp[0][0].restitution, 0.5)
        self.assertEqual(mp[0][0].static_friction, 0.5)
        self.assertEqual(mp[0][0].dynamic_friction, 0.5)

        # Check restitution matrix of the default material
        drm = manager.restitution_matrix()
        self.assertEqual(drm.shape, (1, 1))
        self.assertEqual(drm[0, 0], 0.5)

        # Check friction matrix of the default material
        dfm = manager.static_friction_matrix()
        self.assertEqual(dfm.shape, (1, 1))
        self.assertEqual(dfm[0, 0], 0.5)

        # Check dynamic friction matrix of the default material
        dym = manager.dynamic_friction_matrix()
        self.assertEqual(dym.shape, (1, 1))
        self.assertEqual(dym[0, 0], 0.5)

    def test_material_manager_register_material(self):
        # Create a default-constructed material manager
        manager = MaterialManager()

        # Define a new material
        steel = MaterialDescriptor("steel")

        # Add a new material
        mid = manager.register(steel)
        self.assertEqual(mid, 1)
        self.assertEqual(manager.num_materials, 2)
        self.assertEqual(manager.index("steel"), mid)
        self.assertIsInstance(manager["steel"], MaterialDescriptor)
        self.assertIsInstance(manager[mid], MaterialDescriptor)
        self.assertEqual(manager[mid].name, "steel")
        self.assertEqual(manager[mid].uid, steel.uid)

        # Check the material-pair properties
        mp = manager.pairs
        self.assertEqual(len(mp), 2)
        self.assertEqual(len(mp[1]), 2)
        self.assertEqual(mp[1][0], None)
        self.assertEqual(mp[0][1], None)
        self.assertEqual(mp[1][1], None)

        # Check if friction and restitution matrices rais errors since material pair properties are not set
        self.assertRaises(ValueError, manager.restitution_matrix)
        self.assertRaises(ValueError, manager.static_friction_matrix)
        self.assertRaises(ValueError, manager.dynamic_friction_matrix)

        # Define material pair properties for the new material
        steel_on_steel = MaterialPairProperties(restitution=0.2, static_friction=0.1, dynamic_friction=0.1)
        default_on_steel = MaterialPairProperties(restitution=1.0, static_friction=1.0, dynamic_friction=1.0)

        # Register properties for the new material
        manager.register_pair(steel, steel, steel_on_steel)
        manager.register_pair(manager.default, steel, default_on_steel)

        # Check the material-pair properties
        mp = manager.pairs
        self.assertEqual(len(mp), 2)
        self.assertEqual(len(mp[1]), 2)
        self.assertEqual(mp[1][0].restitution, 1.0)
        self.assertEqual(mp[1][0].static_friction, 1.0)
        self.assertEqual(mp[1][0].dynamic_friction, 1.0)
        self.assertEqual(mp[1][1].restitution, 0.2)
        self.assertEqual(mp[1][1].static_friction, 0.1)
        self.assertEqual(mp[1][1].dynamic_friction, 0.1)

        # Check the friction matrix
        fm = manager.static_friction_matrix()
        self.assertEqual(fm.shape, (2, 2))
        self.assertEqual(fm[0, 0], DEFAULT_FRICTION)
        self.assertEqual(fm[0, 1], 1.0)
        self.assertEqual(fm[1, 0], 1.0)
        self.assertEqual(fm[1, 1], 0.1)

        # Check the dynamic friction matrix
        dym = manager.dynamic_friction_matrix()
        self.assertEqual(dym.shape, (2, 2))
        self.assertEqual(dym[0, 0], DEFAULT_FRICTION)
        self.assertEqual(dym[0, 1], 1.0)
        self.assertEqual(dym[1, 0], 1.0)
        self.assertEqual(dym[1, 1], 0.1)

        # Check the restitution matrix
        rm = manager.restitution_matrix()
        self.assertEqual(rm.shape, (2, 2))
        self.assertEqual(rm[0, 0], DEFAULT_RESTITUTION)
        self.assertEqual(rm[0, 1], 1.0)
        self.assertEqual(rm[1, 0], 1.0)
        self.assertEqual(rm[1, 1], 0.2)

        # Configure the material pair
        manager.configure_pair(
            first="default",
            second="steel",
            material_pair=MaterialPairProperties(restitution=0.5, static_friction=0.5, dynamic_friction=0.5),
        )

        # Check the material-pair properties
        mp = manager.pairs
        self.assertEqual(mp[1][0].restitution, 0.5)
        self.assertEqual(mp[1][0].static_friction, 0.5)
        self.assertEqual(mp[1][0].dynamic_friction, 0.5)
        self.assertEqual(mp[1][1].restitution, 0.2)
        self.assertEqual(mp[1][1].static_friction, 0.1)
        self.assertEqual(mp[1][1].dynamic_friction, 0.1)

        # Check the updated restitution matrix
        rm = manager.restitution_matrix()
        self.assertEqual(rm.shape, (2, 2))
        self.assertEqual(rm[0, 0], DEFAULT_RESTITUTION)
        self.assertEqual(rm[0, 1], 0.5)
        self.assertEqual(rm[1, 0], 0.5)
        self.assertEqual(rm[1, 1], 0.2)

        # Check the updated friction matrix
        fm = manager.static_friction_matrix()
        self.assertEqual(fm.shape, (2, 2))
        self.assertEqual(fm[0, 0], DEFAULT_FRICTION)
        self.assertEqual(fm[0, 1], 0.5)
        self.assertEqual(fm[1, 0], 0.5)
        self.assertEqual(fm[1, 1], 0.1)

        # Check the updated dynamic friction matrix
        dym = manager.dynamic_friction_matrix()
        self.assertEqual(dym.shape, (2, 2))
        self.assertEqual(dym[0, 0], DEFAULT_FRICTION)
        self.assertEqual(dym[0, 1], 0.5)
        self.assertEqual(dym[1, 0], 0.5)
        self.assertEqual(dym[1, 1], 0.1)

    def test_material_manager_register_pair(self):
        # Create a default-constructed material manager
        manager = MaterialManager()

        # Define two new materials
        steel = MaterialDescriptor("steel")
        rubber = MaterialDescriptor("rubber")

        # Register the new materials
        manager.register(steel)
        manager.register(rubber)
        self.assertEqual(manager.num_materials, 3)
        self.assertEqual(manager.index("steel"), 1)
        self.assertEqual(manager.index("rubber"), 2)

        # Define material pair properties
        steel_on_steel = MaterialPairProperties(restitution=0.2, static_friction=0.1, dynamic_friction=0.1)
        rubber_on_rubber = MaterialPairProperties(restitution=0.4, static_friction=0.3, dynamic_friction=0.3)
        rubber_on_steel = MaterialPairProperties(restitution=0.6, static_friction=0.5, dynamic_friction=0.5)
        default_on_steel = MaterialPairProperties(restitution=0.8, static_friction=0.7, dynamic_friction=0.7)
        default_on_rubber = MaterialPairProperties(restitution=1.0, static_friction=0.9, dynamic_friction=0.9)

        # Register the material pair
        manager.register_pair(steel, steel, steel_on_steel)
        manager.register_pair(rubber, rubber, rubber_on_rubber)
        manager.register_pair(rubber, steel, rubber_on_steel)
        manager.register_pair(manager.default, steel, default_on_steel)
        manager.register_pair(manager.default, rubber, default_on_rubber)

        # Check the material-pair properties
        mp = manager.pairs
        self.assertEqual(len(mp), 3)
        self.assertEqual(len(mp[1]), 3)
        self.assertEqual(len(mp[2]), 3)
        self.assertEqual(mp[1][0].restitution, 0.8)
        self.assertEqual(mp[1][0].static_friction, 0.7)
        self.assertEqual(mp[1][0].dynamic_friction, 0.7)
        self.assertEqual(mp[1][1].restitution, 0.2)
        self.assertEqual(mp[1][1].static_friction, 0.1)
        self.assertEqual(mp[1][1].dynamic_friction, 0.1)
        self.assertEqual(mp[1][2].restitution, 0.6)
        self.assertEqual(mp[1][2].static_friction, 0.5)
        self.assertEqual(mp[1][2].dynamic_friction, 0.5)
        self.assertEqual(mp[2][0].restitution, 1.0)
        self.assertEqual(mp[2][0].static_friction, 0.9)
        self.assertEqual(mp[2][0].dynamic_friction, 0.9)
        self.assertEqual(mp[2][1].restitution, 0.6)
        self.assertEqual(mp[2][1].static_friction, 0.5)
        self.assertEqual(mp[2][1].dynamic_friction, 0.5)
        self.assertEqual(mp[2][2].restitution, 0.4)
        self.assertEqual(mp[2][2].static_friction, 0.3)
        self.assertEqual(mp[2][2].dynamic_friction, 0.3)

        # Check the restitution matrix
        rm = manager.restitution_matrix()
        self.assertEqual(rm.shape, (3, 3))
        self.assertTrue(np.allclose(rm, rm.T), "Restitution matrix is not symmetric")
        self.assertEqual(rm[0, 0], DEFAULT_RESTITUTION)
        self.assertEqual(rm[0, 1], 0.8)
        self.assertEqual(rm[0, 2], 1.0)
        self.assertEqual(rm[1, 0], 0.8)
        self.assertEqual(rm[1, 1], 0.2)
        self.assertEqual(rm[1, 2], 0.6)
        self.assertEqual(rm[2, 0], 1.0)
        self.assertEqual(rm[2, 1], 0.6)
        self.assertEqual(rm[2, 2], 0.4)

        # Check the static friction matrix
        fm = manager.static_friction_matrix()
        self.assertEqual(fm.shape, (3, 3))
        self.assertTrue(np.allclose(fm, fm.T), "Static friction matrix is not symmetric")
        self.assertEqual(fm[0, 0], DEFAULT_FRICTION)
        self.assertEqual(fm[0, 1], 0.7)
        self.assertEqual(fm[0, 2], 0.9)
        self.assertEqual(fm[1, 0], 0.7)
        self.assertEqual(fm[1, 1], 0.1)
        self.assertEqual(fm[1, 2], 0.5)
        self.assertEqual(fm[2, 0], 0.9)
        self.assertEqual(fm[2, 1], 0.5)
        self.assertEqual(fm[2, 2], 0.3)

        # Check the dynamic friction matrix
        dym = manager.dynamic_friction_matrix()
        self.assertEqual(dym.shape, (3, 3))
        self.assertTrue(np.allclose(dym, dym.T), "Dynamic friction matrix is not symmetric")
        self.assertEqual(dym[0, 0], DEFAULT_FRICTION)
        self.assertEqual(dym[0, 1], 0.7)
        self.assertEqual(dym[0, 2], 0.9)
        self.assertEqual(dym[1, 0], 0.7)
        self.assertEqual(dym[1, 1], 0.1)
        self.assertEqual(dym[1, 2], 0.5)
        self.assertEqual(dym[2, 0], 0.9)
        self.assertEqual(dym[2, 1], 0.5)
        self.assertEqual(dym[2, 2], 0.3)

        # Optional verbose output
        if self.verbose:
            print(f"\nRestitution Matrix:\n{rm}")
            print(f"\nStatic Friction Matrix:\n{fm}")
            print(f"\nDynamic Friction Matrix:\n{dym}")


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=6, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.verbose = True
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
