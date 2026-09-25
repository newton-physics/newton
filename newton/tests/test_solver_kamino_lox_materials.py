# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check experimental LOX configuration and angular contact materials."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino, convert_contacts_newton_to_kamino
from newton._src.solvers.kamino._src.geometry.primitive.pipeline import CollisionPipelinePrimitive
from newton._src.solvers.kamino._src.geometry.unified import CollisionPipelineUnifiedKamino
from newton._src.solvers.kamino.config import LOXSolverConfig


class TestLOXContactMaterials(unittest.TestCase):
    def test_config_defaults_and_validation(self):
        """Keep contact extensions opt-in and reject inconsistent parameters."""
        config = LOXSolverConfig()
        self.assertEqual(config.contact_compliance, 0.0)
        self.assertEqual(config.contact_compliance_fraction, 1.0)
        self.assertFalse(config.contact_restitution)
        self.assertFalse(config.contact_spatial_friction)
        for field, values in (
            ("contact_compliance", (-1.0, float("inf"), float("nan"))),
            ("contact_compliance_fraction", (0.0, -1.0, 1.1, float("nan"))),
            ("contact_restitution", (1, "true")),
            ("contact_spatial_friction", (1, "true")),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    LOXSolverConfig(**{field: value})
        with self.assertRaisesRegex(ValueError, "cannot both"):
            LOXSolverConfig(contact_restitution=True, contact_recoverable_response=True)
        LOXSolverConfig(contact_compliance=1.0e-5, contact_compliance_fraction=0.5, contact_restitution=True)

    def test_collision_paths_and_material_updates(self):
        """Transport angular friction through both collision paths and shape edits."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.49), wp.quat_identity()))
            builder.add_shape_sphere(
                body,
                radius=0.5,
                cfg=builder.ShapeConfig(mu=0.5, mu_torsional=0.02, mu_rolling=0.04),
            )
            builder.add_shape_sphere(
                -1,
                xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()),
                radius=0.5,
                cfg=builder.ShapeConfig(mu=0.5, mu_torsional=0.06, mu_rolling=0.08),
            )
            model = builder.finalize(device="cpu")
            kamino = ModelKamino.from_newton(model)
            contacts = ContactsKamino(model=kamino, device="cpu", remappable=True)
            detector = CollisionPipelineUnifiedKamino(kamino)
            primitive = CollisionPipelinePrimitive(kamino)
            state = model.state()
            data = kamino.data()
            external_pipeline = newton.CollisionPipeline(model, rigid_contact_max=contacts.model_max_contacts_host)
            external = external_pipeline.contacts()

            for torsion, rolling in (([0.02, 0.06], [0.04, 0.08]), ([0.1, 0.3], [0.2, 0.6])):
                model.shape_material_mu_torsional.assign(np.asarray(torsion, dtype=np.float32))
                model.shape_material_mu_rolling.assign(np.asarray(rolling, dtype=np.float32))
                expected = np.asarray([np.mean(torsion), np.mean(rolling)], dtype=np.float32)
                for route in ("unified", "primitive", "external"):
                    with self.subTest(route=route, expected=expected):
                        if route == "unified":
                            detector.collide(data, contacts)
                        elif route == "primitive":
                            primitive.collide(data, contacts)
                        else:
                            external_pipeline.collide(state, external)
                            convert_contacts_newton_to_kamino(model, state, external, contacts)
                        count = int(contacts.model_active_contacts.numpy()[0])
                        self.assertGreater(count, 0)
                        np.testing.assert_allclose(
                            contacts.angular_friction.numpy()[:count], np.tile(expected, (count, 1)), rtol=1.0e-6
                        )

    def test_native_geometry_defaults(self):
        """Give native geometries without angular material arrays zero friction."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.49), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.5)
            builder.add_ground_plane()
            model = ModelKamino.from_newton(builder.finalize(device="cpu"))
            model.geoms.torsional_friction = None
            model.geoms.rolling_friction = None
            contacts = ContactsKamino(model=model, device="cpu")
            detector = CollisionPipelineUnifiedKamino(model)
            detector.collide(model.data(), contacts)
            count = int(contacts.model_active_contacts.numpy()[0])
            self.assertGreater(count, 0)
            np.testing.assert_array_equal(contacts.angular_friction.numpy()[:count], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
