import importlib.metadata
import os
import tempfile
import unittest
from unittest import mock

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.test_mujoco_solver_usd import _stage_source
from pxr import Sdf, Usd, UsdPhysics


class ReviewProbe(unittest.TestCase):
    def write(self, source):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, 'scene.usda')
        with open(path, 'w', encoding='utf-8') as file:
            file.write(source)
        return path

    def test_subtree_scene_selection(self):
        stage = Usd.Stage.CreateInMemory()
        for name, value in (('A', 31), ('B', 7)):
            scene = UsdPhysics.Scene.Define(stage, f'/{name}/Scene').GetPrim()
            scene.ApplyAPI('NewtonMuJoCoSceneAPI')
            scene.CreateAttribute('mjc:option:iterations', Sdf.ValueTypeNames.Int).Set(value)
            scene.GetAttribute('newton:mujoco:nconmax').Set(value)
        path = self.write(stage.GetRootLayer().ExportToString())
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        imported = builder.add_usd(path, root_path='/B')
        model = builder.finalize()
        with mock.patch.object(SolverMuJoCo, '__init__', return_value=None) as init:
            SolverMuJoCo.create_from_usd(path, model)
        print('SUBTREE', imported['physics_scene_path'], 'MODEL ITERATIONS', model.mujoco.iterations.numpy().tolist(), 'KWARGS', init.call_args.kwargs)
        self.assertEqual(init.call_args.kwargs['nconmax'], 7)

    def test_ignored_scene_selection(self):
        stage = Usd.Stage.CreateInMemory()
        UsdPhysics.Scene.Define(stage, '/A/Scene')
        scene = UsdPhysics.Scene.Define(stage, '/B/Scene').GetPrim()
        scene.ApplyAPI('NewtonMuJoCoSceneAPI')
        path = self.write(stage.GetRootLayer().ExportToString())
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        imported = builder.add_usd(path, ignore_paths=['/A'])
        print('IGNORED SCENE IMPORT', imported['physics_scene_path'])
        with mock.patch.object(SolverMuJoCo, '__init__', return_value=None):
            SolverMuJoCo.create_from_usd(path, builder.finalize())

    def test_canonical_iterations(self):
        path = self.write(_stage_source('uniform bool newton:mujoco:useMujocoCpu = true\n uniform int newton:maxSolverIterations = 7'))
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        imported = builder.add_usd(path)
        solver = SolverMuJoCo.create_from_usd(path, builder.finalize())
        print('CANONICAL ITERATIONS IMPORT', imported['max_solver_iterations'], 'SOLVER', solver.mj_model.opt.iterations)
        self.assertEqual(solver.mj_model.opt.iterations, 7)

    def test_invalid_bool_type(self):
        path = self.write(_stage_source('uniform string newton:mujoco:useMujocoCpu = "false"'))
        with mock.patch.object(SolverMuJoCo, '__init__', return_value=None) as init:
            SolverMuJoCo.create_from_usd(path, newton.ModelBuilder().finalize())
        print('BOOLEAN STRING', init.call_args.kwargs)
        self.assertFalse(init.call_args.kwargs['use_mujoco_cpu'])

    def test_invalid_float_type(self):
        path = self.write(_stage_source('uniform string newton:mujoco:sleep_tolerance = "bad"'))
        with self.assertRaises(ValueError):
            SolverMuJoCo.create_from_usd(path, newton.ModelBuilder().finalize())

    def test_all_attribute_forwarding(self):
        attrs = '''
uniform int newton:mujoco:njmax = 8
uniform int newton:mujoco:njmax_nnz = 9
uniform int newton:mujoco:nconmax = 7
uniform int newton:mujoco:nvmax = 6
uniform float newton:mujoco:sleep_tolerance = 0.125
uniform int newton:mujoco:updateDataInterval = 0
uniform token newton:mujoco:deterministic = "gpuToGpu"
uniform bool newton:mujoco:useMujocoCpu = false
uniform bool newton:mujoco:useMujocoContacts = false
uniform bool newton:mujoco:enableSleeping = true
uniform bool newton:mujoco:includeSites = false
uniform bool newton:mujoco:skipVisualOnlyGeoms = false
uniform bool mjc:flag:multiccd = false
uniform bool mjc:flag:contact = true
uniform bool mjc:flag:sensor = true
'''
        path = self.write(_stage_source(attrs))
        with mock.patch.object(SolverMuJoCo, '__init__', return_value=None) as init:
            SolverMuJoCo.create_from_usd(path, newton.ModelBuilder().finalize())
        kwargs = init.call_args.kwargs
        self.assertEqual(kwargs['njmax_nnz'], 9)
        self.assertEqual(kwargs['update_data_interval'], 0)
        for name in ('use_mujoco_cpu', 'use_mujoco_contacts', 'include_sites', 'skip_visual_only_geoms', 'enable_multiccd', 'disable_contacts', 'disable_sensors'):
            self.assertIs(kwargs[name], False)
        self.assertEqual(len(kwargs), 15)

    def test_dependency_provenance(self):
        import newton_usd_schemas
        print('SCHEMA MODULE', newton_usd_schemas.__file__)
        print('SCHEMA DIRECT URL', importlib.metadata.distribution('newton-usd-schemas').read_text('direct_url.json'))
