# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise Kamino's USD scene settings through the public import path."""

import math
import unittest

import newton
from newton._src.solvers.kamino.config import (
    ConstrainedDynamicsConfig,
    ConstraintStabilizationConfig,
    SCCAPGDConfig,
    SCCSolverConfig,
)
from newton.solvers import SolverKamino
from newton.tests.unittest_utils import requires


@requires("pxr.Usd", "newton_usd_schemas")
class TestKaminoUsdConfig(unittest.TestCase):
    def _stage(self, properties="", schemas='(prepend apiSchemas = ["NewtonKaminoSceneAPI"])'):
        from pxr import Sdf, Usd

        layer = Sdf.Layer.CreateAnonymous(".usda")
        self.assertTrue(layer.ImportFromString(f'#usda 1.0\ndef PhysicsScene "Scene" {schemas}\n{{\n{properties}\n}}'))
        return Usd.Stage.Open(layer)

    def _model(self, stage):
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.add_usd(stage)
        return builder.finalize(device="cpu")

    def test_schema_only_selects_scc(self):
        """Read the composed solver and bilateral defaults from the applied API."""
        from pxr import Usd

        stage = self._stage(schemas='(prepend apiSchemas = ["NewtonKaminoSccSceneAPI"])')
        if not Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("NewtonKaminoSccSceneAPI"):
            with self.assertRaisesRegex(ImportError, "newton-usd-schemas.*NewtonKaminoSccSceneAPI"):
                self._model(stage)
            return
        model = self._model(stage)
        config = SolverKamino.Config.from_model(model)
        self.assertEqual(config.dynamics_solver, "scc")
        self.assertEqual(config.scc.bilateral_solver_type, "LLTB")
        self.assertFalse(config.dynamics.preconditioning)
        self.assertEqual(SolverKamino.Config.from_model(model, dynamics_solver="padmm").dynamics_solver, "padmm")
        python_config = SCCSolverConfig(bilateral_solver_type="LDL")
        self.assertEqual(SolverKamino.Config.from_model(model, scc=python_config).scc.bilateral_solver_type, "LDL")

        solver_attr = stage.GetPrimAtPath("/Scene").GetAttribute("newton:kamino:dynamicsSolver")
        solver_attr.Set("padmm", time=1.0)
        self.assertEqual(SolverKamino.Config.from_model(self._model(stage)).dynamics_solver, "scc")
        solver_attr.Set("padmm")
        self.assertEqual(SolverKamino.Config.from_model(self._model(stage)).dynamics_solver, "padmm")

    def test_scene_api_order(self):
        """Honor the composed schema fallback and let an authored selection override it."""
        from pxr import Sdf, Usd

        stage = self._stage(schemas='(prepend apiSchemas = ["NewtonKaminoSceneAPI", "NewtonKaminoSccSceneAPI"])')
        if Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("NewtonKaminoSccSceneAPI"):
            self.assertEqual(SolverKamino.Config.from_model(self._model(stage)).dynamics_solver, "padmm")
        scene = stage.GetPrimAtPath("/Scene")
        scene.CreateAttribute("newton:kamino:dynamicsSolver", Sdf.ValueTypeNames.Token, custom=False).Set("scc")
        self.assertEqual(SolverKamino.Config.from_model(self._model(stage)).dynamics_solver, "scc")

    def test_scc_uses_compatible_dynamics_fallback(self):
        """Disable preconditioning before validating an imported SCC configuration."""
        stage = self._stage('uniform token newton:kamino:dynamicsSolver = "scc"')
        dynamics = ConstrainedDynamicsConfig(cull_speculative_contacts=False)
        config = SolverKamino.Config.from_model(self._model(stage), dynamics=dynamics)
        self.assertEqual(config.dynamics_solver, "scc")
        self.assertFalse(config.dynamics.preconditioning)
        self.assertFalse(config.dynamics.cull_speculative_contacts)
        self.assertTrue(dynamics.preconditioning)
        with self.assertRaisesRegex(ValueError, "preconditioning=False"):
            SolverKamino.Config(dynamics_solver="scc", dynamics=dynamics)

    def test_authored_scc_configuration(self):
        """Apply authored SCC and APGD settings over Python fallbacks without mutating them."""
        stage = self._stage("""
    uniform token newton:kamino:dynamicsSolver = "scc"
    uniform bool newton:kamino:constraints:usePreconditioning = true
    uniform int newton:maxSolverIterations = 13
    uniform token newton:kamino:scc:warmstarting = "none"
    uniform token newton:kamino:scc:bilateral:numericalSolver = "lltb"
    uniform token newton:kamino:scc:unilateral:numericalSolver = "apgd"
    uniform float newton:kamino:scc:unilateral:pgs:omega = 0
    uniform float newton:kamino:scc:tolerance = 0.0002
    uniform int newton:kamino:scc:unilateral:apgd:maxIterations = 17
    uniform int newton:kamino:scc:unilateral:apgd:maxNonlinearCorrections = 3
    uniform float newton:kamino:scc:unilateral:apgd:tolerance = 0.0004
""")
        fallback = SCCSolverConfig(apgd=SCCAPGDConfig(max_iterations=41, max_backtracks=12))
        config = SolverKamino.Config.from_model(self._model(stage), dynamics_solver="padmm", scc=fallback)
        self.assertEqual(config.dynamics_solver, "scc")
        self.assertTrue(config.sparse_jacobian)
        self.assertFalse(config.dynamics.preconditioning)
        self.assertEqual(config.scc.max_alternating_iterations, 13)
        self.assertEqual(config.scc.warmstart_mode, "none")
        self.assertEqual(config.scc.bilateral_solver_type, "LLTB")
        self.assertEqual(config.scc.unilateral_solver, "apgd")
        self.assertEqual(config.scc.omega, fallback.omega)
        self.assertAlmostEqual(config.scc.tolerance, 2.0e-4)
        self.assertEqual(config.scc.apgd.max_iterations, 17)
        self.assertEqual(config.scc.apgd.max_nonlinear_corrections, 3)
        self.assertAlmostEqual(config.scc.apgd.tolerance, 4.0e-4)
        self.assertEqual(config.scc.apgd.max_backtracks, 12)
        self.assertEqual(fallback.apgd.max_iterations, 41)

    def test_recovery_values_and_pgs_tuning(self):
        """Preserve explicit zero, false, and recovery sentinels across config merging."""
        stage = self._stage("""
    uniform token newton:kamino:dynamicsSolver = "scc"
    uniform float newton:kamino:constraints:jointRecoveryTime = -inf
    uniform float newton:kamino:constraints:limitRecoveryTime = 0
    uniform float newton:kamino:constraints:contactRecoveryTime = 0.02
    uniform float newton:kamino:constraints:jointRecoverySpeed = 0
    uniform float newton:kamino:constraints:limitRecoverySpeed = inf
    uniform bool newton:kamino:constraints:limitCompliance = false
    uniform bool newton:kamino:constraints:contactCompliance = true
    uniform token newton:kamino:scc:bilateral:numericalSolver = "lltbrcm"
    uniform float newton:kamino:scc:unilateral:pgs:omega = 1.25
    uniform int newton:kamino:scc:unilateral:pgs:sweepsPerIteration = 5
    uniform int newton:kamino:scc:unilateral:apgd:maxIterations = 0
""")
        config = SolverKamino.Config.from_model(
            self._model(stage),
            constraints=ConstraintStabilizationConfig(
                joint_recovery_time=0.5, limit_compliance=True, contact_recovery_speed=0.75
            ),
        )
        self.assertIsNone(config.constraints.joint_recovery_time)
        self.assertEqual(config.constraints.limit_recovery_time, 0)
        self.assertAlmostEqual(config.constraints.contact_recovery_time, 0.02)
        self.assertEqual(config.constraints.joint_recovery_speed, 0)
        self.assertEqual(config.constraints.limit_recovery_speed, math.inf)
        self.assertEqual(config.constraints.contact_recovery_speed, 0.75)
        self.assertFalse(config.constraints.limit_compliance)
        self.assertTrue(config.constraints.contact_compliance)
        self.assertEqual(config.scc.bilateral_solver_type, "LLTBRCM")
        self.assertEqual(config.scc.omega, 1.25)
        self.assertEqual(config.scc.inequality_sweeps_per_iteration, 5)

    def test_invalid_active_value(self):
        """Reject an invalid active SCC setting while ignoring SCC tuning under PADMM."""
        stage = self._stage("uniform float newton:kamino:scc:unilateral:pgs:omega = 0")
        model = self._model(stage)
        self.assertEqual(SolverKamino.Config.from_model(model).dynamics_solver, "padmm")
        with self.assertRaisesRegex(ValueError, "omega"):
            SolverKamino.Config.from_model(model, dynamics_solver="scc")
