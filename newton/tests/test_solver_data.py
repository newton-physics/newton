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

import newton
from newton._src.solvers.solver_data import CustomDataField, SolverData
from newton.solvers import SolverBase


class TestSolverData(unittest.TestCase):
    def setUp(self):
        self.model = newton.Model()
        self.model.body_count = 4
        self.sd = SolverData(
            model=self.model,
            generic_fields={
                "body_acceleration": self.model.body_count,
                "contact_force_scalar": 5,
            },
            custom_fields=[],
            verbose=False,
        )

    def test_require_generic_field_allocates_and_reads(self):
        self.sd._require_fields({"body_acceleration": True})
        state = newton.State()
        self.sd.allocate_data(state)
        arr = state.data.body_acceleration
        self.assertIsInstance(arr, wp.array)
        self.assertEqual(len(arr), self.model.body_count)
        self.assertEqual(arr.dtype, wp.spatial_vector)

    def test_dynamic_contact_frequency_and_allocation(self):
        self.sd._require_fields({"contact_force_scalar": True})
        self.assertEqual(self.sd.frequency_sizes["contact"], 5)
        state = newton.State()
        self.sd.allocate_data(state)
        self.assertEqual(len(state.data.contact_force_scalar), 5)

    def test_require_nonexistent_field_raises(self):
        with self.assertRaises(TypeError):
            self.sd._require_fields({"body_missing": True})

    def test_require_custom_field_allocates_and_reads(self):
        # construct SolverData with custom field defined up front
        sd2 = SolverData(
            model=self.model,
            generic_fields={
                "body_acceleration": self.model.body_count,
                "contact_force_scalar": 5,
            },
            custom_fields=[
                CustomDataField(
                    name="body_custom",
                    frequency="body",
                    field_type=wp.array(dtype=wp.float32),
                    size=self.model.body_count,
                    namespace="",
                )
            ],
            verbose=False,
        )
        sd2._require_fields({"body_custom": True})
        state2 = newton.State()
        sd2.allocate_data(state2)
        arr = state2.data.body_custom
        self.assertIsInstance(arr, wp.array)
        self.assertEqual(len(arr), self.model.body_count)
        self.assertEqual(arr.dtype, wp.float32)

    def test_find_attribute_frequency(self):
        self.assertEqual(self.sd.find_attribute_frequency("body_acceleration"), "body")
        with self.assertRaises(AttributeError):
            self.sd.find_attribute_frequency("foo")


class TestSolverDataIntegration(unittest.TestCase):
    class CustomSolver(SolverBase):
        def __init__(self, model, generic_fields_fn=None, custom_fields_fn=None):
            super().__init__(model)
            self._generic_fields_fn = generic_fields_fn or (
                lambda m: {
                    "body_acceleration": m.body_count,
                }
            )
            self._custom_fields_fn = custom_fields_fn or (lambda m: [])

        def get_generic_data_fields(self) -> dict[str, int]:
            return dict(self._generic_fields_fn(self.model))

        def get_custom_data_fields(self) -> list:
            return list(self._custom_fields_fn(self.model))

        def step(self, state_in, state_out, control, contacts, dt):
            pass

        def update_contacts(self, contacts):
            pass

    def setUp(self):
        self.model = newton.Model()
        self.model.body_count = 3
        self.solver = self.CustomSolver(self.model)

    def test_lazy_data_instantiation(self):
        self.assertIsNone(self.solver.data)
        self.solver.require_data("body_acceleration")
        self.assertIsInstance(self.solver.data, SolverData)

    def test_require_generic_and_custom_fields(self):
        # create a solver that declares custom fields via API callback
        solver = self.CustomSolver(
            self.model,
            custom_fields_fn=lambda m: [
                CustomDataField(
                    name="body_custom",
                    frequency="body",
                    field_type=wp.array(dtype=wp.float32),
                    size=m.body_count,
                    namespace="",
                )
            ],
        )
        solver.require_data("body_acceleration", "body_custom")
        state = newton.State()
        solver.allocate_data(state)
        self.assertIsInstance(state.data.body_acceleration, wp.array)
        self.assertEqual(len(state.data.body_acceleration), self.model.body_count)
        self.assertEqual(state.data.body_acceleration.dtype, wp.spatial_vector)
        self.assertIsInstance(state.data.body_custom, wp.array)
        self.assertEqual(len(state.data.body_custom), self.model.body_count)
        self.assertEqual(state.data.body_custom.dtype, wp.float32)

    def test_activation_toggle(self):
        self.solver.require_data("body_acceleration")
        self.solver.data.set_field_active("body_acceleration", active=False)
        self.assertFalse(self.solver.data.required_fields["body_acceleration"])

    def test_missing_field_raises(self):
        with self.assertRaises(TypeError):
            self.solver.require_data("does_not_exist")

    def test_get_data_fields_callback_works(self):
        def generic(model):
            return {"body_acceleration": model.body_count}

        def custom(model):
            return [
                CustomDataField(
                    name="body_custom",
                    frequency="body",
                    field_type=wp.array(dtype=wp.float32),
                    size=model.body_count,
                    namespace="",
                )
            ]

        solver = self.CustomSolver(self.model, generic_fields_fn=generic, custom_fields_fn=custom)
        solver.require_data("body_acceleration", "body_custom")
        state = newton.State()
        solver.allocate_data(state)
        self.assertTrue(hasattr(state.data, "body_custom"))
        self.assertEqual(len(state.data.body_custom), self.model.body_count)


if __name__ == "__main__":
    unittest.main()
