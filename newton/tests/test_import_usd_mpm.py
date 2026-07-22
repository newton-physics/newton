# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestImportUsdMPM(unittest.TestCase):
    @staticmethod
    def _stage():
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").GetPrim()
        scene.ApplyAPI("NewtonMPMSceneAPI")
        return stage, scene

    @staticmethod
    def _points(stage, path, positions, *, velocities=None, widths=None):
        from pxr import UsdGeom

        points = UsdGeom.Points.Define(stage, path)
        points.GetPrim().ApplyAPI("NewtonMPMParticleAPI")
        points.CreatePointsAttr(positions)
        if velocities is not None:
            points.CreateVelocitiesAttr(velocities)
        if widths is not None:
            points.CreateWidthsAttr(widths)
        return points

    @staticmethod
    def _material(stage, path, **attributes):
        from pxr import UsdShade

        material = UsdShade.Material.Define(stage, path)
        material.GetPrim().ApplyAPI("NewtonMPMMaterialAPI")
        for name, value in attributes.items():
            material.GetPrim().GetAttribute(name).Set(value)
        return material

    @staticmethod
    def _bind(points, material):
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(points.GetPrim()).Bind(material, materialPurpose="physics")

    @staticmethod
    def _bind_subset(points, name, indices, material, *, element_type=None):
        from pxr import UsdGeom, UsdShade

        if element_type is None:
            element_type = UsdGeom.Tokens.point
        subset = UsdShade.MaterialBindingAPI.Apply(points.GetPrim()).CreateMaterialBindSubset(
            name,
            indices,
            elementType=element_type,
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material, materialPurpose="physics")
        return subset

    def test_config_schema_fallbacks_match_python_defaults(self):
        """Keep registered scene-schema fallbacks aligned with Config defaults."""
        _stage, scene = self._stage()

        defaults = dataclasses.asdict(SolverImplicitMPM.Config())
        fallback_fields = {
            "newton:maxSolverIterations": "max_iterations",
            "newton:mpm:tolerance": "tolerance",
            "newton:mpm:rheologySolvers": "solver",
            "newton:mpm:warmstartMode": "warmstart_mode",
            "newton:mpm:colliderVelocityMode": "collider_velocity_mode",
            "newton:mpm:voxelSize": "voxel_size",
            "newton:mpm:gridType": "grid_type",
            "newton:mpm:gridPadding": "grid_padding",
            "newton:mpm:maxActiveCellCount": "max_active_cell_count",
            "newton:mpm:maxLeafNodeCount": "max_leaf_node_count",
            "newton:mpm:maxLowerNodeCount": "max_lower_node_count",
            "newton:mpm:maxUpperNodeCount": "max_upper_node_count",
            "newton:mpm:transferScheme": "transfer_scheme",
            "newton:mpm:integrationScheme": "integration_scheme",
            "newton:mpm:criticalFraction": "critical_fraction",
            "newton:mpm:airDrag": "air_drag",
            "newton:mpm:colliderNormalFromSdfGradient": "collider_normal_from_sdf_gradient",
            "newton:mpm:colliderBasis": "collider_basis",
            "newton:mpm:strainBasis": "strain_basis",
            "newton:mpm:velocityBasis": "velocity_basis",
        }
        self.assertEqual(set(fallback_fields.values()), set(defaults))

        for usd_name, field_name in fallback_fields.items():
            with self.subTest(attribute=usd_name):
                attr = scene.GetAttribute(usd_name)
                self.assertTrue(attr)
                self.assertFalse(attr.HasAuthoredValue())
                fallback = attr.Get()
                if usd_name == "newton:maxSolverIterations":
                    self.assertEqual(fallback, -1)
                    fallback = defaults[field_name]
                elif usd_name == "newton:mpm:rheologySolvers":
                    solvers = tuple(str(value) for value in fallback)
                    fallback = solvers[0] if len(solvers) == 1 else solvers

                expected = defaults[field_name]
                if isinstance(expected, float):
                    self.assertTrue(np.isclose(float(fallback), expected, rtol=1.0e-6, atol=1.0e-8))
                else:
                    self.assertEqual(fallback, expected)

        self.assertEqual(dataclasses.asdict(SolverImplicitMPM.Config.create_from_usd(scene)), defaults)

    def test_material_schema_fallbacks_match_model_defaults(self):
        """Keep registered material fallbacks aligned with Newton's particle defaults."""
        stage, _scene = self._stage()
        material = self._material(stage, "/World/Material")
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        fallback_fields = {
            "newton:mpm:youngModulus": "mpm:young_modulus",
            "newton:mpm:poissonRatio": "mpm:poisson_ratio",
            "newton:mpm:damping": "mpm:damping",
            "newton:mpm:friction": "mpm:friction",
            "newton:mpm:yieldPressure": "mpm:yield_pressure",
            "newton:mpm:tensileYieldRatio": "mpm:tensile_yield_ratio",
            "newton:mpm:yieldStress": "mpm:yield_stress",
            "newton:mpm:viscosity": "mpm:viscosity",
            "newton:mpm:hardening": "mpm:hardening",
            "newton:mpm:hardeningRate": "mpm:hardening_rate",
            "newton:mpm:softeningRate": "mpm:softening_rate",
            "newton:mpm:dilatancy": "mpm:dilatancy",
        }

        for usd_name, model_name in fallback_fields.items():
            with self.subTest(attribute=usd_name):
                attr = material.GetPrim().GetAttribute(usd_name)
                self.assertTrue(attr)
                self.assertFalse(attr.HasAuthoredValue())
                self.assertTrue(
                    np.isclose(
                        float(attr.Get()),
                        float(builder.custom_attributes[model_name].default),
                        rtol=1.0e-6,
                        atol=1.0e-8,
                    )
                )

    def test_config_uses_unit_scale_one_when_stage_units_are_unauthored(self):
        """Treat authored quantities as SI when stage units are unauthored."""
        from pxr import UsdGeom, UsdPhysics

        stage, scene = self._stage()
        self.assertFalse(UsdGeom.StageHasAuthoredMetersPerUnit(stage))
        self.assertFalse(UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage))
        scene.GetAttribute("newton:mpm:voxelSize").Set(0.25)
        scene.GetAttribute("newton:mpm:airDrag").Set(0.75)

        config = SolverImplicitMPM.Config.create_from_usd(scene)

        self.assertAlmostEqual(config.voxel_size, 0.25)
        self.assertAlmostEqual(config.air_drag, 0.75)

    def test_config_reads_authored_values_and_units(self):
        """Convert authored scene values to a validated SI configuration."""
        from pxr import UsdGeom, UsdPhysics, Vt

        stage, scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
        values = {
            "newton:maxSolverIterations": 64,
            "newton:mpm:tolerance": 2.0e-5,
            "newton:mpm:rheologySolvers": Vt.TokenArray(["cg", "gs"]),
            "newton:mpm:warmstartMode": "particles",
            "newton:mpm:colliderVelocityMode": "backward",
            "newton:mpm:voxelSize": 5.0,
            "newton:mpm:gridType": "fixed",
            "newton:mpm:gridPadding": 2,
            "newton:mpm:maxActiveCellCount": 1024,
            "newton:mpm:transferScheme": "pic",
            "newton:mpm:integrationScheme": "gimp",
            "newton:mpm:criticalFraction": 0.25,
            "newton:mpm:airDrag": 3.0,
            "newton:mpm:colliderNormalFromSdfGradient": True,
            "newton:mpm:colliderBasis": "pic64",
            "newton:mpm:strainBasis": "Q1d",
            "newton:mpm:velocityBasis": "B2",
        }
        for name, value in values.items():
            scene.GetAttribute(name).Set(value)

        config = SolverImplicitMPM.Config.create_from_usd(scene)

        self.assertEqual(config.max_iterations, 64)
        self.assertEqual(config.solver, ("cg", "gs"))
        self.assertAlmostEqual(config.voxel_size, 0.05)
        self.assertAlmostEqual(config.air_drag, 0.003)
        self.assertEqual(config.collider_basis, "pic64")
        self.assertEqual(config.velocity_basis, "B2")

    def test_imports_opted_in_points_materials_and_ranges(self):
        """Import only opted-in Points with transforms, units, and bound materials."""
        from pxr import Gf, UsdGeom, UsdPhysics

        stage, _scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.1)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)

        render_points = UsdGeom.Points.Define(stage, "/World/RenderOnly")
        render_points.CreatePointsAttr([Gf.Vec3f(99.0, 0.0, 0.0)])

        sand = self._points(
            stage,
            "/World/Sand",
            [Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(2.0, 0.0, 0.0)],
            velocities=[Gf.Vec3f(3.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)],
            widths=[0.5, 1.0],
        )
        sand.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
        sand.AddScaleOp().Set(Gf.Vec3f(2.0))
        sand_material = self._material(
            stage,
            "/World/SandMaterial",
            **{
                "physics:density": 2000.0,
                "newton:mpm:youngModulus": 500.0,
                "newton:mpm:friction": 0.6,
            },
        )
        self._bind(sand, sand_material)

        snow = self._points(stage, "/World/Snow", [Gf.Vec3f(0.0, 1.0, 0.0)])
        snow_material = self._material(
            stage,
            "/World/SnowMaterial",
            **{
                "physics:density": 1000.0,
                "newton:mpm:friction": 0.2,
            },
        )
        self._bind(snow, snow_material)

        builder = newton.ModelBuilder()
        builder.add_particle(pos=wp.vec3(-1.0), vel=wp.vec3(), mass=1.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="USD stages with non-unit .* units are not supported.*")
            result = builder.add_usd(
                stage,
                xform=wp.transform(wp.vec3(4.0, 0.0, 0.0), wp.quat_identity()),
                load_visual_shapes=False,
            )

        self.assertEqual(result["path_mpm_particle_map"], {"/World/Sand": (1, 3), "/World/Snow": (3, 4)})
        self.assertIsInstance(result["mpm_config"], SolverImplicitMPM.Config)
        self.assertEqual(builder.particle_count, 4)
        np.testing.assert_allclose(np.asarray(builder.particle_q[1]), [5.2, 0.0, 0.0], rtol=1.0e-6)
        np.testing.assert_allclose(np.asarray(builder.particle_qd[1]), [0.6, 0.0, 0.0], rtol=1.0e-6)
        np.testing.assert_allclose(builder.particle_radius[1:3], [0.05, 0.1], rtol=1.0e-6)
        np.testing.assert_allclose(builder.particle_mass[1:3], [2.0, 16.0], rtol=1.0e-6)
        self.assertAlmostEqual(builder.particle_mass[3], 8.0)
        self.assertAlmostEqual(builder.custom_attributes["mpm:young_modulus"].values[1], 5.0)
        self.assertAlmostEqual(builder.custom_attributes["mpm:friction"].values[3], 0.2)

        model = builder.finalize(device="cpu")
        self.assertEqual(model.particle_count, 4)
        np.testing.assert_allclose(model.mpm.friction.numpy(), [0.5, 0.6, 0.6, 0.2], rtol=1.0e-6)

    def test_validates_particle_array_lengths_before_mutation(self):
        """Reject malformed particle arrays before adding any particles."""
        from pxr import Gf

        stage, _scene = self._stage()
        self._points(
            stage,
            "/World/Sand",
            [Gf.Vec3f(), Gf.Vec3f(1.0)],
            velocities=[Gf.Vec3f()],
            widths=[0.1, 0.1],
        )
        builder = newton.ModelBuilder()

        with self.assertRaisesRegex(ValueError, "velocities must have shape"):
            builder.add_usd(stage, load_visual_shapes=False)
        self.assertEqual(builder.particle_count, 0)

        invalid_stage, _scene = self._stage()
        self._points(
            invalid_stage,
            "/World/Sand",
            [Gf.Vec3f(), Gf.Vec3f(1.0)],
            widths=[0.1, 0.2, 0.3],
        )
        invalid_builder = newton.ModelBuilder()
        with self.assertRaisesRegex(ValueError, "widths must contain one or 2"):
            invalid_builder.add_usd(invalid_stage, load_visual_shapes=False)
        self.assertEqual(invalid_builder.particle_count, 0)

    def test_derives_particle_masses_from_density_widths_and_stage_units(self):
        """Derive SI particle masses from density, widths, and authored stage units."""
        from pxr import Gf, UsdGeom, UsdPhysics

        stage, _scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.5)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
        points = self._points(
            stage,
            "/World/Sand",
            [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
            widths=[2.0, 4.0],
        )
        material = self._material(
            stage,
            "/World/SandMaterial",
            **{"physics:density": 1000.0},
        )
        self._bind(points, material)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="USD stages with non-unit .* units are not supported.*")
            builder = newton.ModelBuilder()
            builder.add_usd(stage, load_visual_shapes=False)

        np.testing.assert_allclose(builder.particle_mass, [8.0, 64.0], rtol=1.0e-6)

    def test_requires_mpm_api_on_parent_material_binding(self):
        """Reject a whole-prim physics material without NewtonMPMMaterialAPI."""
        from pxr import Gf, UsdShade

        stage, _scene = self._stage()
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
        material = UsdShade.Material.Define(stage, "/World/GenericMaterial")
        UsdShade.MaterialBindingAPI.Apply(points.GetPrim()).Bind(material, materialPurpose="physics")

        builder = newton.ModelBuilder()
        with self.assertRaisesRegex(ValueError, r"/World/GenericMaterial:.*NewtonMPMMaterialAPI"):
            builder.add_usd(stage, load_visual_shapes=False)
        self.assertEqual(builder.particle_count, 0)

    def test_requires_mpm_api_on_subset_material_binding(self):
        """Reject a point-subset physics material without NewtonMPMMaterialAPI."""
        from pxr import Gf, UsdShade

        stage, _scene = self._stage()
        points = self._points(
            stage,
            "/World/Sand",
            [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
            widths=[0.1],
        )
        self._bind(points, self._material(stage, "/World/DefaultMaterial"))
        material = UsdShade.Material.Define(stage, "/World/GenericSubsetMaterial")
        self._bind_subset(points, "GenericSubset", [1], material)

        builder = newton.ModelBuilder()
        with self.assertRaisesRegex(ValueError, r"/World/GenericSubsetMaterial:.*NewtonMPMMaterialAPI"):
            builder.add_usd(stage, load_visual_shapes=False)
        self.assertEqual(builder.particle_count, 0)

    def test_unbound_points_use_builder_material_defaults(self):
        """Use Newton material and builder density defaults when no material is bound."""
        from pxr import Gf

        stage, _scene = self._stage()
        self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.2])
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.density = 125.0

        result = builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(result["path_mpm_particle_map"], {"/World/Sand": (0, 1)})
        self.assertAlmostEqual(builder.particle_mass[0], 1.0)
        model = builder.finalize(device="cpu")
        self.assertAlmostEqual(float(model.mpm.friction.numpy()[0]), 0.5)

    def test_imports_point_material_subsets_in_one_range(self):
        """Overlay two point-subset materials without splitting a Points prim range."""
        from pxr import Gf, UsdGeom, UsdPhysics

        stage, _scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        points = self._points(
            stage,
            "/World/Mixed",
            [Gf.Vec3f(float(i), 0.0, 0.0) for i in range(4)],
            widths=[1.0],
        )
        default_material = self._material(
            stage,
            "/World/DefaultMaterial",
            **{
                "physics:density": 1000.0,
                "newton:mpm:youngModulus": 100.0,
                "newton:mpm:friction": 0.1,
            },
        )
        material_a = self._material(
            stage,
            "/World/MaterialA",
            **{
                "physics:density": 2000.0,
                "newton:mpm:youngModulus": 200.0,
                "newton:mpm:friction": 0.2,
            },
        )
        material_b = self._material(
            stage,
            "/World/MaterialB",
            **{
                "physics:density": 3000.0,
                "newton:mpm:youngModulus": 300.0,
                "newton:mpm:friction": 0.3,
            },
        )
        self._bind(points, default_material)
        self._bind_subset(points, "MaterialA", [0, 2], material_a)
        self._bind_subset(points, "MaterialB", [1], material_b)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(result["path_mpm_particle_map"], {"/World/Mixed": (0, 4)})
        np.testing.assert_allclose(builder.particle_mass, [2000.0, 3000.0, 2000.0, 1000.0])
        model = builder.finalize(device="cpu")
        np.testing.assert_allclose(model.mpm.young_modulus.numpy(), [200.0, 300.0, 200.0, 100.0])
        np.testing.assert_allclose(model.mpm.friction.numpy(), [0.2, 0.3, 0.2, 0.1])

    def test_rejects_invalid_point_material_subsets(self):
        """Reject overlapping, out-of-range, and non-point physics material subsets."""
        from pxr import Gf, UsdGeom

        for case in ("overlap", "range", "element_type"):
            with self.subTest(case=case):
                stage, _scene = self._stage()
                points = self._points(
                    stage,
                    "/World/Mixed",
                    [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
                    widths=[0.1],
                )
                material_a = self._material(stage, "/World/MaterialA")
                material_b = self._material(stage, "/World/MaterialB")
                if case == "overlap":
                    self._bind_subset(points, "MaterialA", [0], material_a)
                    self._bind_subset(points, "MaterialB", [0], material_b)
                    message = "invalid point material subsets"
                elif case == "range":
                    self._bind_subset(points, "MaterialA", [2], material_a)
                    message = "invalid point material subsets"
                else:
                    self._bind_subset(
                        points,
                        "MaterialA",
                        [0],
                        material_a,
                        element_type=UsdGeom.Tokens.face,
                    )
                    message = "elementType='point'"

                builder = newton.ModelBuilder()
                with self.assertRaisesRegex(ValueError, message):
                    builder.add_usd(stage, load_visual_shapes=False)
                self.assertEqual(builder.particle_count, 0)

    def test_ignores_render_only_point_material_subsets(self):
        """Keep all-purpose render subsets out of MPM material assignment."""
        from pxr import Gf, UsdGeom, UsdShade

        stage, _scene = self._stage()
        points = self._points(
            stage,
            "/World/Sand",
            [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
            widths=[0.1],
        )
        physics_material = self._material(
            stage,
            "/World/PhysicsMaterial",
            **{"newton:mpm:friction": 0.25},
        )
        self._bind(points, physics_material)
        render_material = UsdShade.Material.Define(stage, "/World/RenderMaterial")
        subset = UsdShade.MaterialBindingAPI.Apply(points.GetPrim()).CreateMaterialBindSubset(
            "RenderSubset",
            [0],
            elementType=UsdGeom.Tokens.point,
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(render_material)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize(device="cpu")

        np.testing.assert_allclose(model.mpm.friction.numpy(), [0.25, 0.25])

    def test_rejects_nonfinite_particle_values(self):
        """Reject non-finite authored particle data before builder mutation."""
        from pxr import Gf

        for name, positions, kwargs in (
            ("points", [Gf.Vec3f(float("nan"), 0.0, 0.0)], {}),
            ("widths", [Gf.Vec3f()], {"widths": [float("inf")]}),
        ):
            with self.subTest(name=name):
                stage, _scene = self._stage()
                self._points(stage, "/World/Sand", positions, **kwargs)
                builder = newton.ModelBuilder()
                with self.assertRaisesRegex(ValueError, "non-finite|finite positive|finite non-negative"):
                    builder.add_usd(stage, load_visual_shapes=False)
                self.assertEqual(builder.particle_count, 0)

    def test_keeps_mpm_registration_idempotent(self):
        """Accept builders whose MPM attributes were registered before USD import."""
        from pxr import Gf

        stage, _scene = self._stage()
        self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        SolverImplicitMPM.register_custom_attributes(builder)

        result = builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(result["path_mpm_particle_map"], {"/World/Sand": (0, 1)})
        self.assertEqual(builder.particle_count, 1)

    def test_validates_authored_material_ranges(self):
        """Reject authored MPM material values outside documented ranges."""
        from pxr import Gf

        for name, value, message in (
            ("physics:density", -1.0, "physics:density"),
            ("newton:mpm:youngModulus", 0.0, "youngModulus"),
            ("newton:mpm:tensileYieldRatio", 1.1, "tensile_yield_ratio"),
            ("newton:mpm:dilatancy", 1.1, "dilatancy"),
        ):
            with self.subTest(name=name):
                stage, _scene = self._stage()
                points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
                material = self._material(stage, "/World/Material", **{name: value})
                self._bind(points, material)
                with self.assertRaisesRegex(ValueError, message):
                    newton.ModelBuilder().add_usd(stage, load_visual_shapes=False)

    def test_rejects_nonuniform_particle_transform(self):
        """Reject a transform that cannot preserve spherical particle widths."""
        from pxr import Gf

        stage, _scene = self._stage()
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
        points.AddScaleOp().Set(Gf.Vec3f(1.0, 2.0, 1.0))

        with self.assertRaisesRegex(ValueError, "uniform, shear-free"):
            newton.ModelBuilder().add_usd(stage, load_visual_shapes=False)

    def test_rejects_invalid_scene_token(self):
        """Reject authored config tokens outside the solver contract."""
        _stage, scene = self._stage()
        scene.GetAttribute("newton:mpm:gridType").Set("octree")

        with self.assertRaisesRegex(ValueError, "gridType"):
            SolverImplicitMPM.Config.create_from_usd(scene)

    def test_warns_for_mixed_nonunit_rigid_stage(self):
        """Warn when SI-converted MPM particles share a non-unit legacy rigid stage."""
        from pxr import Gf, UsdGeom, UsdPhysics

        stage, _scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[1.0])
        collider = UsdGeom.Cube.Define(stage, "/World/Collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            newton.ModelBuilder().add_usd(stage, load_visual_shapes=False)

        self.assertTrue(any("Mixed rigid/collider and MPM stages" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
