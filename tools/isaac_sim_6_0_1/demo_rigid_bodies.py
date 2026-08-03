"""Run a lit rigid-body demo with the forked Newton source."""

import asyncio
import os
from pathlib import Path

import omni.timeline
import omni.usd
import warp as wp
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics  # noqa: TID253

import newton


def validate_source(module_file: str, expected_repo: str) -> Path:
    """Validate that a module is contained by the expected repository."""
    module_path = Path(module_file).resolve()
    repo_path = Path(expected_repo).resolve()
    if not module_path.is_relative_to(repo_path):
        raise RuntimeError(f"Newton source {module_path} is outside expected repository {repo_path}")
    return repo_path


def add_cube(stage, path, position, size, color, dynamic=True):
    """Add a collidable cube to the stage."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(size)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(*position))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)
    return cube


def add_sphere(stage, path, position, radius, color):
    """Add a dynamic collidable sphere to the stage."""
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(*position))
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
    UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(1.0)
    return sphere


async def build_and_run():
    """Build the demo stage, verify the runtime, and start simulation."""
    expected_repo = os.environ["NEWTON_DEV_REPO"]
    validate_source(newton.__file__, expected_repo)
    engine = SimulationManager.get_active_physics_engine()
    if engine != "newton":
        raise RuntimeError(f"Expected active physics engine 'newton', got {engine!r}")
    device = wp.get_device()
    if not device.is_cuda:
        raise RuntimeError(f"Expected a CUDA Warp device, got {device}")

    await omni.usd.get_context().new_stage_async()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)

    dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome_light.CreateIntensityAttr(700.0)
    dome_light.CreateColorAttr(Gf.Vec3f(0.75, 0.82, 1.0))
    key_light = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    key_light.CreateIntensityAttr(3500.0)
    key_light.CreateAngleAttr(1.0)
    key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.78))
    UsdGeom.XformCommonAPI(key_light).SetRotate(Gf.Vec3f(35.0, -25.0, -35.0))

    add_cube(stage, "/World/Ground", (0.0, 0.0, -0.25), 0.5, (0.15, 0.18, 0.22), dynamic=False)
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/Ground")).SetScale(Gf.Vec3f(16.0, 16.0, 1.0))

    colors = (
        (0.95, 0.25, 0.18),
        (0.20, 0.55, 0.95),
        (0.25, 0.85, 0.40),
        (0.95, 0.70, 0.15),
    )
    for index, color in enumerate(colors):
        add_cube(stage, f"/World/Cube_{index}", (-1.5 + index, 0.0, 1.0 + index * 1.25), 0.65, color)
    add_sphere(stage, "/World/Sphere", (0.0, 1.0, 6.0), 0.42, (0.75, 0.25, 0.95))

    set_camera_view(
        eye=[8.0, 10.0, 7.0],
        target=[0.0, 0.0, 2.0],
        camera_prim_path="/OmniverseKit_Persp",
    )
    print(f"NEWTON_DEV runtime_source={Path(newton.__file__).resolve()}")
    print(f"NEWTON_DEV active_engine={engine} device={device}")
    omni.timeline.get_timeline_interface().play()


demo_task = asyncio.ensure_future(build_and_run())  # noqa: RUF006
