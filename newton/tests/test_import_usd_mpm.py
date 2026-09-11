# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import pathlib
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import USD_AVAILABLE

try:
    import newton_usd_schemas
except ImportError:
    newton_usd_schemas = None


def _has_mpm_schemas() -> bool:
    """Return whether the installed schema package contains the MPM APIs."""
    if not USD_AVAILABLE or newton_usd_schemas is None:
        return False
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Points.Define(stage, "/Particles").GetPrim()
    try:
        return bool(prim.ApplyAPI("NewtonPointsDeformableSimAPI"))
    except Exception:
        return False


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMpmConfigUsdEntryGuards(unittest.TestCase):
    def test_rejects_non_scene_prim(self):
        """Reject a non-PhysicsScene prim as a configuration source."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()

        with self.assertRaisesRegex(TypeError, "valid UsdPhysics.Scene prim"):
            SolverImplicitMPM.Config.create_from_usd(prim)

    def test_requires_mpm_scene_api(self):
        """Reject a PhysicsScene without NewtonMPMSceneAPI."""
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").GetPrim()

        with self.assertRaisesRegex(ValueError, "NewtonMPMSceneAPI is not applied"):
            SolverImplicitMPM.Config.create_from_usd(scene)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestImportUsdMPMFixtures(unittest.TestCase):
    def test_imports_checked_in_two_prim_fixture(self):
        """Parse the checked-in transformed, unit-scaled, two-material USDA fixture."""
        from pxr import Usd

        fixture = pathlib.Path(__file__).parent / "assets" / "mpm_two_prims.usda"
        stage = Usd.Stage.Open(fixture.as_posix())
        builder = newton.ModelBuilder()

        result = builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(
            result["path_particle_map"],
            {"/World/TranslatedAndScaled/Sand": (0, 2), "/World/Snow": (2, 3)},
        )
        self.assertEqual(builder.particle_count, 3)
        np.testing.assert_allclose(
            np.asarray(builder.particle_q),
            [[1.2, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 0.1, 0.0]],
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            np.asarray(builder.particle_qd),
            [[0.6, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]],
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(builder.particle_radius, [0.05, 0.1, 0.1], rtol=1.0e-6)
        np.testing.assert_allclose(builder.particle_mass, [2.0, 16.0, 8.0], rtol=1.0e-6)
        np.testing.assert_allclose(np.asarray(builder.gravity), [0.0, 0.0, -9.81], rtol=1.0e-6)
        self.assertNotIn("mpm_config", result)
        self.assertEqual(result["particle_scene_path"], "/World/PhysicsScene")
        mpm_config = SolverImplicitMPM.Config.create_from_usd(stage.GetPrimAtPath(result["particle_scene_path"]))
        self.assertAlmostEqual(mpm_config.voxel_size, 0.05)

        model = builder.finalize(device="cpu")
        np.testing.assert_allclose(model.mpm.young_modulus.numpy(), [5.0, 5.0, 1.0e15], rtol=1.0e-6)
        np.testing.assert_allclose(model.mpm.friction.numpy(), [0.6, 0.6, 0.2], rtol=1.0e-6)

    def test_imports_checked_in_material_subset_fixture(self):
        """Parse checked-in point-subset bindings and parent-material fallback."""
        fixture = pathlib.Path(__file__).parent / "assets" / "mpm_material_subsets.usda"
        builder = newton.ModelBuilder()

        result = builder.add_usd(fixture.as_posix(), load_visual_shapes=False)

        self.assertEqual(result["path_particle_map"], {"/World/Mixed": (0, 4)})
        np.testing.assert_allclose(builder.particle_mass, [2000.0, 3000.0, 2000.0, 1000.0])
        model = builder.finalize(device="cpu")
        np.testing.assert_allclose(model.mpm.young_modulus.numpy(), [200.0, 300.0, 200.0, 100.0])
        np.testing.assert_allclose(model.mpm.friction.numpy(), [0.2, 0.3, 0.2, 0.1])
        np.testing.assert_allclose(model.state().mpm.particle_Jp.numpy(), [0.9, 0.8, 0.9, 1.0])


@unittest.skipUnless(_has_mpm_schemas(), "Requires Newton USD schemas with MPM support")
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
    def _points(
        stage,
        path,
        positions,
        *,
        velocities=None,
        widths=None,
        widths_interpolation=None,
        masses=None,
        simulation_owner=None,
        body_prim=None,
    ):
        from pxr import Sdf, UsdGeom

        points = UsdGeom.Points.Define(stage, path)
        points.GetPrim().ApplyAPI("NewtonPointsDeformableSimAPI")
        body_prim = body_prim or points.GetPrim()
        body_prim.AddAppliedSchema("PhysicsDeformableBodyAPI")
        points.CreatePointsAttr(positions)
        if velocities is not None:
            points.CreateVelocitiesAttr(velocities)
        if widths is not None:
            points.CreateWidthsAttr(widths)
            if widths_interpolation is not None:
                points.SetWidthsInterpolation(widths_interpolation)
            elif len(widths) == 1:
                points.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        if masses is not None:
            points.GetPrim().GetAttribute("physics:masses").Set(masses)
        if simulation_owner is not None:
            relationship = body_prim.GetRelationship("physics:simulationOwner")
            if not relationship:
                relationship = body_prim.CreateRelationship("physics:simulationOwner")
            relationship.SetTargets([Sdf.Path(simulation_owner)])
        return points

    @staticmethod
    def _material(stage, path, **attributes):
        from pxr import Sdf, UsdShade

        material = UsdShade.Material.Define(stage, path)
        material.GetPrim().ApplyAPI("NewtonMPMMaterialAPI")
        for name, value in attributes.items():
            attr = material.GetPrim().GetAttribute(name)
            if not attr:
                attr = material.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Float)
            attr.Set(value)
        return material

    @staticmethod
    def _author_float(prim, name, value):
        """Author a float attribute that may belong to a prospective schema."""
        from pxr import Sdf

        attr = prim.GetAttribute(name)
        if not attr:
            attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
        attr.Set(value)

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
            "newton:mpm:voxelSize": "voxel_size",
            "newton:mpm:gridType": "grid_type",
            "newton:mpm:gridPadding": "grid_padding",
            "newton:mpm:maxActiveCellCount": "max_active_cell_count",
            "newton:mpm:transferScheme": "transfer_scheme",
            "newton:mpm:integrationScheme": "integration_scheme",
            "newton:mpm:criticalFraction": "critical_fraction",
            "newton:mpm:airDrag": "air_drag",
        }
        simulator_default_fields = {
            "newton:mpm:voxelSize",
            "newton:mpm:airDrag",
        }

        for usd_name, field_name in fallback_fields.items():
            with self.subTest(attribute=usd_name):
                attr = scene.GetAttribute(usd_name)
                self.assertTrue(attr)
                self.assertFalse(attr.HasAuthoredValue())
                fallback = attr.Get()
                if usd_name == "newton:maxSolverIterations":
                    self.assertEqual(fallback, -1)
                    fallback = defaults[field_name]
                elif usd_name in simulator_default_fields:
                    self.assertEqual(float(fallback), float("-inf"))
                    continue
                elif usd_name == "newton:mpm:rheologySolvers":
                    solvers = tuple(str(value) for value in fallback)
                    fallback = solvers[0] if len(solvers) == 1 else solvers

                expected = defaults[field_name]
                if isinstance(expected, float):
                    self.assertTrue(np.isclose(float(fallback), expected, rtol=1.0e-6, atol=1.0e-8))
                else:
                    self.assertEqual(fallback, expected)

        basis_fallbacks = {
            "newton:mpm:colliderBasisType": "serendipity",
            "newton:mpm:colliderBasisOrder": 2,
            "newton:mpm:colliderDiscontinuousBasis": False,
            "newton:mpm:strainBasisType": "linear",
            "newton:mpm:strainBasisOrder": 0,
            "newton:mpm:strainDiscontinuousBasis": False,
            "newton:mpm:velocityBasisType": "trilinear",
            "newton:mpm:velocityBasisOrder": 1,
        }
        for usd_name, expected in basis_fallbacks.items():
            with self.subTest(attribute=usd_name):
                attr = scene.GetAttribute(usd_name)
                self.assertTrue(attr)
                self.assertFalse(attr.HasAuthoredValue())
                self.assertEqual(attr.Get(), expected)

        self.assertEqual(dataclasses.asdict(SolverImplicitMPM.Config.create_from_usd(scene)), defaults)

    def test_config_reports_usd_context_for_invalid_capacity(self):
        """Include the prim path and USD name in capacity validation errors."""
        _stage, scene = self._stage()
        scene.GetAttribute("newton:mpm:maxActiveCellCount").Set(0)

        with self.assertRaisesRegex(ValueError, "/World/PhysicsScene.*newton:mpm:maxActiveCellCount"):
            SolverImplicitMPM.Config.create_from_usd(scene)

    def test_material_schema_fallbacks_match_model_defaults(self):
        """Keep registered material fallbacks aligned with Newton's particle defaults."""
        stage, _scene = self._stage()
        material = self._material(stage, "/World/Material")
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        fallback_fields = {
            "newton:mpm:youngsModulus": "mpm:young_modulus",
            "newton:mpm:poissonsRatio": "mpm:poisson_ratio",
            "newton:mpm:elasticDamping": "mpm:damping",
            "newton:mpm:internalFriction": "mpm:friction",
            "newton:mpm:yieldPressure": "mpm:yield_pressure",
            "newton:mpm:tensileYieldRatio": "mpm:tensile_yield_ratio",
            "newton:mpm:yieldStress": "mpm:yield_stress",
            "newton:mpm:viscosity": "mpm:viscosity",
            "newton:mpm:hardening": "mpm:hardening",
            "newton:mpm:hardeningRate": "mpm:hardening_rate",
            "newton:mpm:softeningRate": "mpm:softening_rate",
            "newton:mpm:dilatancy": "mpm:dilatancy",
            "newton:mpm:initialPlasticVolumeStrain": "mpm:particle_Jp",
        }
        simulator_default_fields = {
            "newton:mpm:youngsModulus",
            "newton:mpm:yieldPressure",
        }

        for usd_name, model_name in fallback_fields.items():
            with self.subTest(attribute=usd_name):
                attr = material.GetPrim().GetAttribute(usd_name)
                self.assertTrue(attr)
                self.assertFalse(attr.HasAuthoredValue())
                if usd_name in simulator_default_fields:
                    self.assertEqual(float(attr.Get()), float("-inf"))
                    continue
                self.assertTrue(
                    np.isclose(
                        float(attr.Get()),
                        float(builder.custom_attributes[model_name].default),
                        rtol=1.0e-6,
                        atol=1.0e-8,
                    )
                )

    def test_config_accepts_authored_simulator_default_sentinels(self):
        """Retain Python config defaults for authored dimensional sentinels."""
        _stage, scene = self._stage()
        scene.GetAttribute("newton:mpm:voxelSize").Set(float("-inf"))
        scene.GetAttribute("newton:mpm:airDrag").Set(float("-inf"))

        config = SolverImplicitMPM.Config.create_from_usd(scene)

        self.assertEqual(config.voxel_size, SolverImplicitMPM.Config().voxel_size)
        self.assertEqual(config.air_drag, SolverImplicitMPM.Config().air_drag)

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
            "newton:mpm:rheologySolvers": Vt.TokenArray(["conjugate-gradient", "gauss-seidel"]),
            "newton:mpm:voxelSize": 5.0,
            "newton:mpm:gridType": "fixed",
            "newton:mpm:gridPadding": 2,
            "newton:mpm:maxActiveCellCount": 1024,
            "newton:mpm:transferScheme": "pic",
            "newton:mpm:integrationScheme": "gimp",
            "newton:mpm:criticalFraction": 0.25,
            "newton:mpm:airDrag": 3.0,
            "newton:mpm:colliderBasisType": "particle",
            "newton:mpm:colliderBasisOrder": 64,
            "newton:mpm:colliderDiscontinuousBasis": True,
            "newton:mpm:strainBasisType": "trilinear",
            "newton:mpm:strainBasisOrder": 1,
            "newton:mpm:strainDiscontinuousBasis": True,
            "newton:mpm:velocityBasisType": "bspline",
            "newton:mpm:velocityBasisOrder": 2,
        }
        for name, value in values.items():
            scene.GetAttribute(name).Set(value)

        config = SolverImplicitMPM.Config.create_from_usd(scene)

        self.assertEqual(config.max_iterations, 64)
        self.assertEqual(config.solver, ("cg", "gauss-seidel"))
        self.assertAlmostEqual(config.voxel_size, 0.05)
        self.assertAlmostEqual(config.air_drag, 0.003)
        self.assertEqual(config.collider_basis, "pic")
        self.assertEqual(config.strain_basis, "Q1d")
        self.assertEqual(config.velocity_basis, "B2")

    def test_config_maps_schema_rheology_solver_tokens(self):
        """Map descriptive USD rheology tokens to Newton solver names."""
        from pxr import Vt

        _stage, scene = self._stage()
        scene.GetAttribute("newton:mpm:rheologySolvers").Set(
            Vt.TokenArray(
                [
                    "conjugate-gradient",
                    "conjugate-residual",
                    "generalized-minimal-residual",
                ]
            )
        )

        config = SolverImplicitMPM.Config.create_from_usd(scene)

        self.assertEqual(config.solver, ("cg", "cr", "gmres"))

    def test_config_validates_split_basis_fields(self):
        """Compose multi-digit orders and reject unsupported basis combinations."""
        _stage, scene = self._stage()
        scene.GetAttribute("newton:mpm:strainBasisType").Set("trilinear")
        scene.GetAttribute("newton:mpm:strainBasisOrder").Set(12)
        scene.GetAttribute("newton:mpm:strainDiscontinuousBasis").Set(False)

        config = SolverImplicitMPM.Config.create_from_usd(scene)

        self.assertEqual(config.strain_basis, "Q12")

        scene.GetAttribute("newton:mpm:strainBasisType").Set("bspline")
        with self.assertRaisesRegex(ValueError, "must be one of"):
            SolverImplicitMPM.Config.create_from_usd(scene)

        scene.GetAttribute("newton:mpm:strainBasisType").Set("linear")
        scene.GetAttribute("newton:mpm:strainBasisOrder").Set(2)
        with self.assertRaisesRegex(ValueError, "must be discontinuous"):
            SolverImplicitMPM.Config.create_from_usd(scene)

        scene.GetAttribute("newton:mpm:strainBasisType").Clear()
        scene.GetAttribute("newton:mpm:strainBasisOrder").Clear()
        scene.GetAttribute("newton:mpm:velocityBasisType").Set("bspline")
        scene.GetAttribute("newton:mpm:velocityBasisOrder").Set(3)
        self.assertEqual(SolverImplicitMPM.Config.create_from_usd(scene).velocity_basis, "B3")

        scene.GetAttribute("newton:mpm:velocityBasisOrder").Set(4)
        with self.assertRaisesRegex(ValueError, "from 1 through 3"):
            SolverImplicitMPM.Config.create_from_usd(scene)

    def test_config_maps_authored_zero_order_fallback_to_p0(self):
        """Map the authored schema fallback tuple to Newton's intrinsic P0 basis."""
        _stage, scene = self._stage()
        scene.GetAttribute("newton:mpm:strainBasisType").Set("linear")
        scene.GetAttribute("newton:mpm:strainBasisOrder").Set(0)
        scene.GetAttribute("newton:mpm:strainDiscontinuousBasis").Set(False)

        config = SolverImplicitMPM.Config.create_from_usd(scene)

        self.assertEqual(config.strain_basis, "P0")

    def test_config_rejects_unit_conversion_overflow(self):
        """Reject scene values whose authored units overflow SI conversion."""
        from pxr import UsdGeom, UsdPhysics

        for case in ("voxel", "air_drag"):
            with self.subTest(case=case):
                stage, scene = self._stage()
                if case == "voxel":
                    UsdGeom.SetStageMetersPerUnit(stage, 1.0e300)
                    scene.GetAttribute("newton:mpm:voxelSize").Set(1.0e30)
                else:
                    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0e300)
                    scene.GetAttribute("newton:mpm:airDrag").Set(1.0e30)

                with self.assertRaisesRegex(ValueError, "representable|finite positive"):
                    SolverImplicitMPM.Config.create_from_usd(scene)

    def test_selects_mpm_particles_by_simulation_owner(self):
        """Import only deformable Points owned by an MPM scene."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/World/DefaultScene")
        mpm_scene = UsdPhysics.Scene.Define(stage, "/World/MPMScene").GetPrim()
        mpm_scene.ApplyAPI("NewtonMPMSceneAPI")
        self._points(
            stage,
            "/World/MPM",
            [Gf.Vec3f()],
            widths=[0.1],
            simulation_owner="/World/MPMScene",
        )
        self._points(
            stage,
            "/World/OtherSolver",
            [Gf.Vec3f()],
            widths=[0.1],
            simulation_owner="/World/DefaultScene",
        )
        self._points(stage, "/World/DefaultOwned", [Gf.Vec3f()], widths=[0.1])

        result = newton.ModelBuilder().add_usd(stage, load_visual_shapes=False)

        self.assertEqual(result["path_particle_map"], {"/World/MPM": (0, 1)})

    def test_resolves_simulation_owner_from_parent_deformable_body(self):
        """Resolve simulation ownership from a Points prim's parent body."""
        from pxr import Gf, UsdGeom

        stage, _scene = self._stage()
        body = UsdGeom.Xform.Define(stage, "/World/SandBody")
        self._points(
            stage,
            "/World/SandBody/Particles",
            [Gf.Vec3f()],
            widths=[0.1],
            simulation_owner="/World/PhysicsScene",
            body_prim=body.GetPrim(),
        )

        result = newton.ModelBuilder().add_usd(stage, load_visual_shapes=False)

        self.assertEqual(result["path_particle_map"], {"/World/SandBody/Particles": (0, 1)})

    def test_rejects_invalid_simulation_owner_before_particle_mutation(self):
        """Reject an explicit owner that does not resolve to one PhysicsScene."""
        from pxr import Gf, Sdf, UsdGeom

        for owner_path in ("/World/Missing", "/World/NotAScene"):
            with self.subTest(owner_path=owner_path):
                stage, _scene = self._stage()
                UsdGeom.Xform.Define(stage, "/World/NotAScene")
                self._points(
                    stage,
                    "/World/Sand",
                    [Gf.Vec3f()],
                    widths=[0.1],
                    simulation_owner=owner_path,
                )
                builder = newton.ModelBuilder()
                builder.add_particle(pos=wp.vec3(-1.0), vel=wp.vec3(), mass=1.0)

                with self.assertRaisesRegex(ValueError, "must resolve to a PhysicsScene"):
                    builder.add_usd(stage, load_visual_shapes=False)

                self.assertEqual(builder.particle_count, 1)

        stage, _scene = self._stage()
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
        points.GetPrim().GetRelationship("physics:simulationOwner").SetTargets(
            [Sdf.Path("/World/PhysicsScene"), Sdf.Path("/World/OtherScene")]
        )
        builder = newton.ModelBuilder()
        with self.assertRaisesRegex(ValueError, "must target exactly one PhysicsScene"):
            builder.add_usd(stage, load_visual_shapes=False)
        self.assertEqual(builder.particle_count, 0)

    def test_rejects_multiple_mpm_owners_before_particle_mutation(self):
        """Reject one import call spanning more than one MPM scene."""
        from pxr import Gf, UsdPhysics

        stage, _scene = self._stage()
        other_scene = UsdPhysics.Scene.Define(stage, "/World/OtherScene").GetPrim()
        other_scene.ApplyAPI("NewtonMPMSceneAPI")
        self._points(
            stage,
            "/World/Sand",
            [Gf.Vec3f()],
            widths=[0.1],
            simulation_owner="/World/PhysicsScene",
        )
        self._points(
            stage,
            "/World/Snow",
            [Gf.Vec3f()],
            widths=[0.1],
            simulation_owner="/World/OtherScene",
        )
        builder = newton.ModelBuilder()
        builder.add_particle(pos=wp.vec3(-1.0), vel=wp.vec3(), mass=1.0)

        with self.assertRaisesRegex(ValueError, "owned by different"):
            builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(builder.particle_count, 1)

    def test_resolves_owned_mpm_scene_outside_selected_root(self):
        """Resolve an explicit MPM owner even when root_path excludes its prim."""
        from pxr import Gf, UsdGeom

        stage, scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        scene.GetAttribute("newton:mpm:tolerance").Set(2.5e-5)
        self._points(
            stage,
            "/World/Sand",
            [Gf.Vec3f()],
            widths=[0.1],
            simulation_owner="/World/PhysicsScene",
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, root_path="/World/Sand", load_visual_shapes=False)

        self.assertEqual(result["path_particle_map"], {"/World/Sand": (0, 1)})
        self.assertEqual(result["particle_scene_path"], "/World/PhysicsScene")
        mpm_config = SolverImplicitMPM.Config.create_from_usd(stage.GetPrimAtPath(result["particle_scene_path"]))
        self.assertAlmostEqual(mpm_config.tolerance, 2.5e-5)
        np.testing.assert_allclose(np.asarray(builder.gravity), [0.0, 0.0, -9.81], rtol=1.0e-6)

    def test_converts_default_and_authored_mpm_gravity_to_si(self):
        """Convert default and authored stage gravity to SI exactly once."""
        from pxr import Gf, UsdGeom

        for authored_magnitude, expected_magnitude in ((None, 9.81), (250.0, 2.5), (-1.0, 9.81)):
            with self.subTest(authored_magnitude=authored_magnitude):
                stage, scene = self._stage()
                UsdGeom.SetStageMetersPerUnit(stage, 0.01)
                if authored_magnitude is not None:
                    scene.GetAttribute("physics:gravityMagnitude").Set(authored_magnitude)
                self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])

                builder = newton.ModelBuilder()
                builder.add_usd(stage, load_visual_shapes=False)

                np.testing.assert_allclose(
                    np.asarray(builder.gravity),
                    [0.0, 0.0, -expected_magnitude],
                    rtol=1.0e-6,
                )

    def test_rejects_invalid_gravity_before_particle_mutation(self):
        """Validate MPM scene gravity before appending imported particles."""
        from pxr import Gf

        stage, scene = self._stage()
        scene.GetAttribute("physics:gravityMagnitude").Set(float("nan"))
        self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
        builder = newton.ModelBuilder()
        builder.add_particle(pos=wp.vec3(-1.0), vel=wp.vec3(), mass=1.0)

        with self.assertRaisesRegex(ValueError, "gravityMagnitude"):
            builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(builder.particle_count, 1)

    def test_omits_particle_scene_without_imported_points(self):
        """Return no particle owner scene when no Points are imported."""
        from pxr import UsdPhysics

        stage, _scene = self._stage()
        UsdPhysics.Scene.Define(stage, "/World/OtherScene")

        result = newton.ModelBuilder().add_usd(stage, load_visual_shapes=False)

        self.assertEqual(result["path_particle_map"], {})
        self.assertIsNone(result["particle_scene_path"])
        self.assertNotIn("mpm_config", result)

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
                "newton:mpm:youngsModulus": 500.0,
                "newton:mpm:internalFriction": 0.6,
            },
        )
        self._bind(sand, sand_material)

        snow = self._points(stage, "/World/Snow", [Gf.Vec3f(0.0, 1.0, 0.0)])
        snow_material = self._material(
            stage,
            "/World/SnowMaterial",
            **{
                "physics:density": 1000.0,
                "newton:mpm:internalFriction": 0.2,
            },
        )
        self._bind(snow, snow_material)

        builder = newton.ModelBuilder()
        builder.add_particle(pos=wp.vec3(-1.0), vel=wp.vec3(), mass=1.0)
        result = builder.add_usd(
            stage,
            xform=wp.transform(wp.vec3(4.0, 0.0, 0.0), wp.quat_identity()),
            load_visual_shapes=False,
        )

        self.assertEqual(result["path_particle_map"], {"/World/Sand": (1, 3), "/World/Snow": (3, 4)})
        self.assertEqual(result["particle_scene_path"], "/World/PhysicsScene")
        self.assertNotIn("mpm_config", result)
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

    def test_maps_all_authored_material_fields(self):
        """Map every authored MPM material field and its SI conversion."""
        from pxr import Gf, UsdGeom, UsdPhysics

        stage, _scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.1)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.2])
        material = self._material(
            stage,
            "/World/Material",
            **{
                "physics:density": 1000.0,
                "newton:mpm:youngsModulus": 500.0,
                "newton:mpm:poissonsRatio": 0.25,
                "newton:mpm:elasticDamping": 10.0,
                "newton:mpm:internalFriction": 0.4,
                "newton:mpm:yieldPressure": 200.0,
                "newton:mpm:tensileYieldRatio": 0.3,
                "newton:mpm:yieldStress": 300.0,
                "newton:mpm:viscosity": 400.0,
                "newton:mpm:hardening": 0.5,
                "newton:mpm:hardeningRate": 0.6,
                "newton:mpm:softeningRate": 0.7,
                "newton:mpm:dilatancy": 0.8,
                "newton:mpm:initialPlasticVolumeStrain": 0.9,
            },
        )
        self._bind(points, material)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize(device="cpu")

        expected = {
            "young_modulus": 5.0,
            "poisson_ratio": 0.25,
            "damping": 0.02,
            "friction": 0.4,
            "yield_pressure": 2.0,
            "tensile_yield_ratio": 0.3,
            "yield_stress": 3.0,
            "viscosity": 4.0,
            "hardening": 0.5,
            "hardening_rate": 0.6,
            "softening_rate": 0.7,
            "dilatancy": 0.8,
            "particle_Jp": 0.9,
        }
        for field_name, expected_value in expected.items():
            with self.subTest(field=field_name):
                self.assertAlmostEqual(
                    builder.custom_attributes[f"mpm:{field_name}"].values[0],
                    expected_value,
                )
                self.assertAlmostEqual(
                    float(getattr(model.mpm, field_name).numpy()[0]),
                    expected_value,
                )

        self.assertAlmostEqual(float(model.state().mpm.particle_Jp.numpy()[0]), 0.9)

    def test_validates_particle_array_lengths_before_mutation(self):
        """Reject malformed particle arrays before adding any particles."""
        from pxr import Gf, UsdGeom

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
        with self.assertRaisesRegex(ValueError, "widths with interpolation='vertex' must contain 2"):
            invalid_builder.add_usd(invalid_stage, load_visual_shapes=False)
        self.assertEqual(invalid_builder.particle_count, 0)

        vertex_stage, _scene = self._stage()
        self._points(
            vertex_stage,
            "/World/Sand",
            [Gf.Vec3f(), Gf.Vec3f(1.0)],
            widths=[0.1],
            widths_interpolation=UsdGeom.Tokens.vertex,
        )
        vertex_builder = newton.ModelBuilder()
        with self.assertRaisesRegex(ValueError, "widths with interpolation='vertex' must contain 2"):
            vertex_builder.add_usd(vertex_stage, load_visual_shapes=False)
        self.assertEqual(vertex_builder.particle_count, 0)

    def test_reads_indexed_scalar_and_inherited_width_primvars(self):
        """Honor primvar precedence, flattening, constants, and inheritance."""
        from pxr import Gf, Sdf, UsdGeom

        stage, _scene = self._stage()
        indexed = self._points(
            stage,
            "/World/Indexed",
            [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
            widths=[9.0, 9.0],
        )
        indexed_widths = UsdGeom.PrimvarsAPI(indexed.GetPrim()).CreatePrimvar(
            "widths",
            Sdf.ValueTypeNames.FloatArray,
            UsdGeom.Tokens.vertex,
        )
        indexed_widths.Set([0.2, 0.4])
        indexed_widths.SetIndices([1, 0])

        scalar = self._points(
            stage,
            "/World/Scalar",
            [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
        )
        UsdGeom.PrimvarsAPI(scalar.GetPrim()).CreatePrimvar(
            "widths",
            Sdf.ValueTypeNames.Float,
            UsdGeom.Tokens.constant,
        ).Set(0.8)

        group = UsdGeom.Xform.Define(stage, "/World/Inherited")
        UsdGeom.PrimvarsAPI(group.GetPrim()).CreatePrimvar(
            "widths",
            Sdf.ValueTypeNames.Float,
            UsdGeom.Tokens.constant,
        ).Set(0.6)
        self._points(
            stage,
            "/World/Inherited/Points",
            [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
        )

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, load_visual_shapes=False)

        ranges = result["path_particle_map"]
        np.testing.assert_allclose(builder.particle_radius[slice(*ranges["/World/Indexed"])], [0.2, 0.1])
        np.testing.assert_allclose(builder.particle_radius[slice(*ranges["/World/Scalar"])], [0.4, 0.4])
        np.testing.assert_allclose(
            builder.particle_radius[slice(*ranges["/World/Inherited/Points"])],
            [0.3, 0.3],
        )

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

        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)

        np.testing.assert_allclose(builder.particle_mass, [8.0, 64.0], rtol=1.0e-6)

    def test_prefers_authored_particle_masses_and_converts_mass_units(self):
        """Prefer per-point masses and convert them independently of transform scale."""
        from pxr import Gf, UsdGeom, UsdPhysics

        stage, _scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.5)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
        points = self._points(
            stage,
            "/World/Sand",
            [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
            widths=[2.0, 4.0],
            masses=[2000.0, 3000.0],
        )
        points.AddScaleOp().Set(Gf.Vec3f(4.0))
        self._author_float(points.GetPrim(), "physics:mass", 100000.0)
        self._author_float(points.GetPrim(), "physics:density", 100000.0)
        self._bind(points, self._material(stage, "/World/SandMaterial", **{"physics:density": 1.0e6}))

        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)

        np.testing.assert_allclose(builder.particle_mass, [2.0, 3.0], rtol=1.0e-6)

    def test_resolves_deformable_body_mass_and_density(self):
        """Apply body density, then distribute an optional body-total mass."""
        from pxr import Gf, UsdGeom, UsdPhysics

        for total_mass, expected in ((None, [8.0, 64.0]), (9000.0, [1.0, 8.0])):
            with self.subTest(total_mass=total_mass):
                stage, _scene = self._stage()
                UsdGeom.SetStageMetersPerUnit(stage, 0.5)
                UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
                points = self._points(
                    stage,
                    "/World/Sand",
                    [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
                    widths=[2.0, 4.0],
                )
                self._author_float(points.GetPrim(), "physics:density", 1000.0)
                if total_mass is not None:
                    self._author_float(points.GetPrim(), "physics:mass", total_mass)
                self._bind(points, self._material(stage, "/World/Material", **{"physics:density": 250.0}))

                builder = newton.ModelBuilder()
                builder.add_usd(stage, load_visual_shapes=False)

                np.testing.assert_allclose(builder.particle_mass, expected, rtol=1.0e-6)

    def test_uses_default_density_for_unbound_subset_with_total_mass(self):
        """Use builder density for unbound points before total-mass normalization."""
        from pxr import Gf

        stage, _scene = self._stage()
        points = self._points(
            stage,
            "/World/Sand",
            [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
            widths=[1.0],
        )
        self._author_float(points.GetPrim(), "physics:mass", 3.0)
        dense_material = self._material(stage, "/World/Dense", **{"physics:density": 1000.0})
        self._bind_subset(points, "Dense", [0], dense_material)
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.density = 500.0

        builder.add_usd(stage, load_visual_shapes=False)

        np.testing.assert_allclose(builder.particle_mass, [2.0, 1.0], rtol=1.0e-6)

    def test_validates_authored_particle_masses_before_mutation(self):
        """Reject malformed per-point masses before appending particles."""
        from pxr import Gf

        for masses in ([1.0], [1.0, 0.0], [1.0, -1.0], [1.0, float("nan")]):
            with self.subTest(masses=masses):
                stage, _scene = self._stage()
                self._points(
                    stage,
                    "/World/Sand",
                    [Gf.Vec3f(), Gf.Vec3f(1.0, 0.0, 0.0)],
                    widths=[0.1],
                    masses=masses,
                )
                builder = newton.ModelBuilder()
                builder.add_particle(pos=wp.vec3(-1.0), vel=wp.vec3(), mass=1.0)

                with self.assertRaisesRegex(ValueError, "physics:masses"):
                    builder.add_usd(stage, load_visual_shapes=False)

                self.assertEqual(builder.particle_count, 1)

    def test_accepts_standard_deformable_material_elasticity(self):
        """Read canonical elasticity from recognized standard material APIs."""
        from pxr import Gf, UsdShade

        stage, _scene = self._stage()
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
        material = UsdShade.Material.Define(stage, "/World/StandardMaterial")
        material.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableMaterialAPI")
        self._author_float(material.GetPrim(), "physics:youngsModulus", 250.0)
        self._author_float(material.GetPrim(), "physics:poissonsRatio", 0.2)
        self._bind(points, material)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize(device="cpu")

        self.assertAlmostEqual(builder.custom_attributes["mpm:young_modulus"].values[0], 250.0)
        self.assertAlmostEqual(builder.custom_attributes["mpm:poisson_ratio"].values[0], 0.2)
        self.assertAlmostEqual(
            float(model.mpm.friction.numpy()[0]),
            builder.custom_attributes["mpm:friction"].default,
        )

    def test_accepts_standard_physics_material_with_mpm_defaults(self):
        """Accept PhysicsMaterialAPI density while retaining MPM-specific defaults."""
        from pxr import Gf, UsdPhysics, UsdShade

        stage, _scene = self._stage()
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.2])
        material = UsdShade.Material.Define(stage, "/World/PhysicsMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateDensityAttr(125.0)
        self._author_float(material.GetPrim(), "newton:mpm:internalFriction", 0.9)
        self._author_float(material.GetPrim(), "physics:youngsModulus", 20.0)
        self._bind(points, material)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize(device="cpu")

        self.assertAlmostEqual(builder.particle_mass[0], 1.0)
        self.assertTrue(
            np.isclose(
                float(model.mpm.young_modulus.numpy()[0]),
                builder.custom_attributes["mpm:young_modulus"].default,
                rtol=1.0e-6,
            )
        )
        self.assertAlmostEqual(
            float(model.mpm.friction.numpy()[0]),
            builder.custom_attributes["mpm:friction"].default,
        )

    def test_prefers_newton_elasticity_over_standard_volume_material(self):
        """Prefer MPM elasticity when both material APIs author values."""
        from pxr import Gf

        stage, _scene = self._stage()
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
        material = self._material(
            stage,
            "/World/Material",
            **{
                "newton:mpm:youngsModulus": 600.0,
                "newton:mpm:poissonsRatio": 0.35,
            },
        )
        material.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableMaterialAPI")
        self._author_float(material.GetPrim(), "physics:youngsModulus", 300.0)
        self._author_float(material.GetPrim(), "physics:poissonsRatio", 0.25)
        self._bind(points, material)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize(device="cpu")

        self.assertAlmostEqual(float(model.mpm.young_modulus.numpy()[0]), 600.0)
        self.assertAlmostEqual(float(model.mpm.poisson_ratio.numpy()[0]), 0.35)

    def test_material_sentinels_retain_registered_defaults(self):
        """Treat dimensional simulator-default sentinels as registered defaults."""
        from pxr import Gf

        stage, _scene = self._stage()
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
        material = self._material(
            stage,
            "/World/Material",
            **{"newton:mpm:yieldPressure": float("-inf")},
        )
        self._author_float(material.GetPrim(), "newton:mpm:youngsModulus", float("-inf"))
        self._bind(points, material)

        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize(device="cpu")

        self.assertTrue(
            np.isclose(
                float(model.mpm.young_modulus.numpy()[0]),
                builder.custom_attributes["mpm:young_modulus"].default,
                rtol=1.0e-6,
            )
        )
        self.assertTrue(
            np.isclose(
                float(model.mpm.yield_pressure.numpy()[0]),
                builder.custom_attributes["mpm:yield_pressure"].default,
                rtol=1.0e-6,
            )
        )

    def test_rejects_plain_visual_parent_material_binding(self):
        """Reject a whole-prim visual material without a recognized physics API."""
        from pxr import Gf, UsdShade

        stage, _scene = self._stage()
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
        material = UsdShade.Material.Define(stage, "/World/GenericMaterial")
        UsdShade.MaterialBindingAPI.Apply(points.GetPrim()).Bind(material, materialPurpose="physics")

        builder = newton.ModelBuilder()
        with self.assertRaisesRegex(ValueError, r"/World/GenericMaterial:.*NewtonMPMMaterialAPI"):
            builder.add_usd(stage, load_visual_shapes=False)
        self.assertEqual(builder.particle_count, 0)

    def test_rejects_plain_visual_subset_material_binding(self):
        """Reject a point-subset visual material without a recognized physics API."""
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

        self.assertEqual(result["path_particle_map"], {"/World/Sand": (0, 1)})
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
                "newton:mpm:youngsModulus": 100.0,
                "newton:mpm:internalFriction": 0.1,
            },
        )
        material_a = self._material(
            stage,
            "/World/MaterialA",
            **{
                "physics:density": 2000.0,
                "newton:mpm:youngsModulus": 200.0,
                "newton:mpm:internalFriction": 0.2,
            },
        )
        material_b = self._material(
            stage,
            "/World/MaterialB",
            **{
                "physics:density": 3000.0,
                "newton:mpm:youngsModulus": 300.0,
                "newton:mpm:internalFriction": 0.3,
            },
        )
        self._bind(points, default_material)
        self._bind_subset(points, "MaterialA", [0, 2], material_a)
        self._bind_subset(points, "MaterialB", [1], material_b)

        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(result["path_particle_map"], {"/World/Mixed": (0, 4)})
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
            **{"newton:mpm:internalFriction": 0.25},
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

    def test_ignores_render_only_parent_material_binding(self):
        """Do not use an all-purpose parent binding as an MPM physics material."""
        from pxr import Gf, UsdShade

        stage, _scene = self._stage()
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.2])
        render_material = self._material(
            stage,
            "/World/RenderMaterial",
            **{
                "physics:density": 1000.0,
                "newton:mpm:internalFriction": 0.25,
            },
        )
        UsdShade.MaterialBindingAPI.Apply(points.GetPrim()).Bind(render_material)
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.density = 125.0

        builder.add_usd(stage, load_visual_shapes=False)

        self.assertAlmostEqual(builder.particle_mass[0], 1.0)
        model = builder.finalize(device="cpu")
        self.assertAlmostEqual(float(model.mpm.friction.numpy()[0]), 0.5)

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

    def test_rejects_unit_conversion_overflow_before_mutation(self):
        """Reject finite source values that overflow converted SI material data."""
        from pxr import Gf, UsdGeom, UsdPhysics

        for case in ("density", "pressure", "width", "mass"):
            with self.subTest(case=case):
                stage, _scene = self._stage()
                points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[1.0])
                if case in ("density", "pressure"):
                    UsdGeom.SetStageMetersPerUnit(stage, 1.0e-50)
                    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0e300)
                    attributes = (
                        {"physics:density": 1.0e30} if case == "density" else {"newton:mpm:youngsModulus": 1.0e30}
                    )
                    self._bind(points, self._material(stage, "/World/Material", **attributes))
                else:
                    scale = 1.0e300 if case == "width" else 1.0e110
                    if case == "width":
                        points.GetWidthsAttr().Set([1.0e30])
                    points.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(scale))

                builder = newton.ModelBuilder()
                builder.add_particle(pos=wp.vec3(-1.0), vel=wp.vec3(), mass=1.0)
                with self.assertRaisesRegex(ValueError, "finite|representable"):
                    builder.add_usd(stage, load_visual_shapes=False)
                self.assertEqual(builder.particle_count, 1)

    def test_keeps_mpm_registration_idempotent(self):
        """Accept builders whose MPM attributes were registered before USD import."""
        from pxr import Gf

        stage, _scene = self._stage()
        self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[0.1])
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        SolverImplicitMPM.register_custom_attributes(builder)

        result = builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(result["path_particle_map"], {"/World/Sand": (0, 1)})
        self.assertEqual(builder.particle_count, 1)

    def test_validates_authored_material_ranges(self):
        """Reject authored MPM material values outside documented ranges."""
        from pxr import Gf

        for name, value, message in (
            ("physics:density", -1.0, "physics:density"),
            ("newton:mpm:youngsModulus", -1.0, "youngsModulus"),
            ("newton:mpm:poissonsRatio", -1.0, "poissonsRatio"),
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
        """Warn when SI-converted particles share a non-unit legacy rigid stage."""
        from pxr import Gf, UsdGeom, UsdPhysics

        stage, _scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[1.0])
        collider = UsdGeom.Cube.Define(stage, "/World/Collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            newton.ModelBuilder().add_usd(stage, load_visual_shapes=False)

        messages = [str(item.message) for item in caught]
        self.assertTrue(any("Mixed rigid/collider and particle USD content" in message for message in messages))
        self.assertFalse(any("non-unit linear units are not supported" in message for message in messages))
        self.assertFalse(any("non-unit mass units are not supported" in message for message in messages))

    def test_warns_for_mixed_units_before_particle_validation_error(self):
        """Warn about mixed unit conversion before particle validation fails."""
        from pxr import Gf, UsdGeom, UsdPhysics

        stage, _scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        points = self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[1.0])
        points.AddScaleOp().Set(Gf.Vec3f(1.0, 2.0, 1.0))
        collider = UsdGeom.Cube.Define(stage, "/World/Collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(ValueError, "uniform, shear-free"):
                newton.ModelBuilder().add_usd(stage, load_visual_shapes=False)

        messages = [str(item.message) for item in caught]
        self.assertTrue(any("Mixed rigid/collider and particle USD content" in message for message in messages))

    def test_warns_for_nonunit_mpm_with_deformable(self):
        """Warn when non-unit particles share a stage with another import path."""
        from pxr import Gf, Sdf, UsdGeom

        stage, _scene = self._stage()
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        self._points(stage, "/World/Sand", [Gf.Vec3f()], widths=[1.0])
        tetmesh = stage.DefinePrim("/World/SoftBody", "TetMesh")
        tetmesh.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        )
        tetmesh.CreateAttribute("tetVertexIndices", Sdf.ValueTypeNames.Int4Array).Set([(0, 1, 2, 3)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = newton.ModelBuilder().add_usd(
                stage,
                load_visual_shapes=False,
                return_deformable_results=True,
            )

        messages = [str(item.message) for item in caught]
        self.assertIn("/World/SoftBody", result["path_soft_map"])
        self.assertTrue(any("Mixed particles and other imported USD content" in message for message in messages))
        self.assertFalse(any("Mixed rigid/collider and particle USD content" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
