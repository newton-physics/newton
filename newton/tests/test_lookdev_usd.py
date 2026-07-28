# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.viewer import viewer_usd as vusd
from newton._src.viewer.lookdev import _PARAMS, LookdevMode

try:
    from pxr import Usd, UsdGeom, UsdLux
except ImportError:
    Usd = UsdGeom = UsdLux = None


def _read_hdr(path):
    """Minimal Radiance RGBE (new-format RLE) decoder -> (H, W, 3) float radiance."""
    data = open(path, "rb").read()
    i = 0
    while True:  # skip the header lines up to the blank separator
        j = data.index(b"\n", i)
        line = data[i:j]
        i = j + 1
        if line.strip() == b"":
            break
    j = data.index(b"\n", i)
    dims = data[i:j].split()  # e.g. b"-Y 512 +X 1024"
    height, width = int(dims[1]), int(dims[3])
    buf = data[j + 1 :]
    pos = 0
    rows = []
    for _ in range(height):
        assert buf[pos] == 2 and buf[pos + 1] == 2, "not new-format RLE"
        pos += 4
        chans = np.zeros((4, width), np.uint8)
        for c in range(4):
            x = 0
            while x < width:
                n = buf[pos]
                pos += 1
                if n > 128:  # run of (n - 128) copies of one byte
                    chans[c, x : x + n - 128] = buf[pos]
                    pos += 1
                    x += n - 128
                else:  # n literal bytes
                    chans[c, x : x + n] = np.frombuffer(buf[pos : pos + n], np.uint8)
                    pos += n
                    x += n
        rows.append(chans.T)
    rgbe = np.array(rows)  # (H, W, 4)
    exp = rgbe[..., 3].astype(np.int32)
    scale = np.ldexp(1.0, exp - 136)  # mantissa/256 * 2**(exp-128)
    out = rgbe[..., :3].astype(np.float64) * scale[..., None]
    out[exp == 0] = 0.0
    return out


class TestSkyTextureAssets(unittest.TestCase):
    """The committed sky-gradient equirects ship alongside the code."""

    def test_committed_textures_exist(self):
        for mode in LookdevMode:
            path = vusd._lookdev_sky_texture_path(mode)
            self.assertTrue(os.path.exists(path), f"missing committed sky texture {path}")

    def test_committed_textures_hold_scene_radiance(self):
        # Each HDR equirect holds the scene radiance the GL sky shader feeds to
        # its tone-map: pow(sky_srgb, 2.2) * exposure (sRGB->linear, then
        # exposure), from the sky_upper zenith (row 0) to the sky_lower
        # nadir/ground (last row). The renderer tone-maps this radiance
        # itself, so nothing is baked in. This is the crux: skipping the
        # sRGB->linear step (sky * exposure) is ~16x too bright and reads grey.
        for mode in LookdevMode:
            img = _read_hdr(vusd._lookdev_sky_texture_path(mode))
            params = _PARAMS[mode]
            exposure = float(params["exposure"])
            # Both modes share the same Kit-calibration and gradient shape: a 2x
            # lift (Kit renders the dome darker than GL's ACES) and a 0.3x zenith
            # darkening.
            #
            # The endpoints below are the Kit-calibrated sky, pinned here rather
            # than read from ``_PARAMS``: Kit's renderer and GL's ACES pipeline
            # are different, so the two carry *different* numbers on purpose --
            # ``_PARAMS`` holds whatever makes GL *look* like these already-shipped
            # domes. Editing the GL colours must therefore not disturb this test;
            # only regenerating the textures may, and then these move with them.
            lift = 2.0
            darken = 0.3
            usd_sky = {
                LookdevMode.DARK: ((0.099, 0.101, 0.116), (0.167, 0.167, 0.177)),
                LookdevMode.LIGHT: ((0.92, 0.92, 0.94), (0.40, 0.40, 0.42)),
            }
            sky_upper, sky_lower = (np.array(c, float) for c in usd_sky[mode])
            zenith = np.power(sky_upper, 2.2) * exposure * lift * darken
            nadir = np.power(sky_lower, 2.2) * exposure * lift
            np.testing.assert_allclose(img[0, 0], zenith, rtol=0.02, atol=1e-4)  # row 0 (top scanline) = zenith
            np.testing.assert_allclose(img[-1, 0], nadir, rtol=0.02, atol=1e-4)  # last row = nadir/ground
            self.assertGreater(img.max(), 5e-3)  # true radiance, not a pre-inverted ~2e-5 sliver
            # Non-linear ramp (cf. the studio prototype): a visible monotonic
            # gradient with the horizon between the two ends. Direction differs by
            # mode -- DARK darkens toward the zenith, LIGHT toward the ground.
            ends = sorted((float(img[0].max()), float(img[-1].max())))
            horizon = float(img[img.shape[0] // 2].max())
            self.assertGreater(ends[1], 1.3 * ends[0])  # ends clearly different, not uniform
            self.assertTrue(ends[0] - 1e-6 <= horizon <= ends[1] + 1e-6)  # horizon between the ends


class TestRelativeAssetPath(unittest.TestCase):
    """USD asset paths must be explicitly layer-relative (leading ``./``)."""

    def test_subdirectory_gets_dot_prefix(self):
        # A bare ``newton/...`` would be a USD search path, not stage-relative.
        self.assertEqual(vusd._relative_asset_path("/scene/newton/x/sky.png", "/scene"), "./newton/x/sky.png")

    def test_same_directory_gets_dot_prefix(self):
        self.assertEqual(vusd._relative_asset_path("/scene/sky.png", "/scene"), "./sky.png")

    def test_parent_directory_kept_explicit(self):
        self.assertEqual(vusd._relative_asset_path("/a/sky.png", "/a/b/c"), "../../sky.png")


@unittest.skipIf(Usd is None, "usd-core not available")
class TestAuthorLookdev(unittest.TestCase):
    """``_author_lookdev`` writes a portable studio rig from the lookdev params."""

    _DEFAULT_AABB = object()

    @classmethod
    def _author(cls, mode=LookdevMode.LIGHT, up_axis=2, aabb=_DEFAULT_AABB, floor_height=0.0):
        if aabb is cls._DEFAULT_AABB:
            aabb = (np.array([-1.0, -1.0, 0.5]), np.array([2.0, 3.0, 4.0]))
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/root")
        vusd._author_lookdev(stage, "/root", mode, up_axis, aabb, floor_height=floor_height)
        return stage

    def _by_type(self, stage, type_name):
        return [p for p in stage.Traverse() if p.GetTypeName() == type_name]

    def test_authors_dome_three_lights_and_floor(self):
        stage = self._author()
        self.assertEqual(len(self._by_type(stage, "DomeLight")), 1)
        self.assertEqual(len(self._by_type(stage, "DistantLight")), 3)
        self.assertEqual(len(self._by_type(stage, "Mesh")), 1)
        self.assertEqual(len(self._by_type(stage, "Material")), 1)

    def test_lights_leave_shadows_at_default(self):
        # No light authors shadow settings -- no ShadowAPI shadow:enable and no
        # soft-shadow angle -- so shadows stay at the renderer default for all
        # three lights (Kit applies no inconsistent per-light overrides).
        stage = self._author()
        for name in ("Key", "Fill", "Rim"):
            light = stage.GetPrimAtPath(f"/root/_Lookdev/{name}")
            self.assertIsNone(UsdLux.ShadowAPI(light).GetShadowEnableAttr().Get())
            self.assertFalse(UsdLux.DistantLight(light).GetAngleAttr().IsAuthored())

    def test_lights_have_no_wrapper_xform(self):
        # Each light/dome is a light prim directly (its transform op on itself) —
        # no extra Xform wrapper above it.
        stage = self._author()
        self.assertEqual(stage.GetPrimAtPath("/root/_Lookdev/Sky").GetTypeName(), "DomeLight")
        for name in ("Key", "Fill", "Rim"):
            self.assertEqual(stage.GetPrimAtPath(f"/root/_Lookdev/{name}").GetTypeName(), "DistantLight")

    def test_light_intensity_and_color_from_params(self):
        stage = self._author(LookdevMode.LIGHT)
        key = UsdLux.DistantLight(stage.GetPrimAtPath("/root/_Lookdev/Key"))
        _direction, color, intensity = _PARAMS[LookdevMode.LIGHT]["lights"][0]
        exposure = _PARAMS[LookdevMode.LIGHT]["exposure"]
        self.assertAlmostEqual(
            key.GetIntensityAttr().Get(), intensity * exposure * vusd._LOOKDEV_DISTANT_INTENSITY_SCALE, places=3
        )
        np.testing.assert_allclose(np.array(key.GetColorAttr().Get()), color, atol=1e-5)

    def test_key_light_aims_along_lookdev_direction(self):
        # The DistantLight's local +Z (emission is -Z; source is +Z) must point
        # toward the source, i.e. along the mode's key direction.
        from pxr import Gf

        stage = self._author(LookdevMode.LIGHT, up_axis=2)
        key_xf = UsdGeom.Xformable(stage.GetPrimAtPath("/root/_Lookdev/Key"))
        quat = key_xf.GetOrderedXformOps()[0].Get()
        aimed = Gf.Rotation(quat).TransformDir(Gf.Vec3d(0, 0, 1))
        d = np.array(_PARAMS[LookdevMode.LIGHT]["lights"][0][0], dtype=float)
        d /= np.linalg.norm(d)
        np.testing.assert_allclose(np.array(aimed), d, atol=1e-5)

    def test_dome_references_committed_hdr(self):
        stage = self._author(LookdevMode.DARK)
        dome = UsdLux.DomeLight(stage.GetPrimAtPath("/root/_Lookdev/Sky"))
        tex = dome.GetTextureFileAttr()
        self.assertEqual(
            os.path.basename(tex.Get().path),
            os.path.basename(vusd._lookdev_sky_texture_path(LookdevMode.DARK)),
        )
        # HDR holds true scene radiance, so the dome runs at Kit's DomeLight
        # intensity with a raw (linear) colorspace and automatic (equirect)
        # projection — a real environment dome, not a tone-map baked in.
        self.assertEqual(vusd._LOOKDEV_DOME_INTENSITY, 1000.0)
        self.assertEqual(dome.GetIntensityAttr().Get(), vusd._LOOKDEV_DOME_INTENSITY)
        self.assertEqual(dome.GetTextureFormatAttr().Get(), UsdLux.Tokens.automatic)
        self.assertEqual(tex.GetColorSpace(), "raw")

    def test_dome_is_camera_visible(self):
        # The dome is flagged visible to the primary ray so RTX renders it as
        # the viewport background (its default background source).
        stage = self._author()
        dome = stage.GetPrimAtPath("/root/_Lookdev/Sky")
        self.assertTrue(dome.GetAttribute("visibleInPrimaryRay").Get())

    def test_dome_pole_aligns_texture_with_up_axis(self):
        # The dome texture's gradient pole must land on the scene up-axis so the
        # gradient runs vertically. Kit's DomeLight defaults its pole to +Z, so
        # the dome's rotateXYZ op must carry +Z onto the active up-axis (a Z-up
        # scene needs none). rotateXYZ, not orient: Kit ignores orient on a dome.
        from pxr import Gf

        for up in (0, 1, 2):
            stage = self._author(up_axis=up)
            m = UsdGeom.Xformable(stage.GetPrimAtPath("/root/_Lookdev/Sky")).GetLocalTransformation()
            pole = m.TransformDir(Gf.Vec3d(0, 0, 1))  # Kit's dome pole defaults to +Z
            expected = [0.0, 0.0, 0.0]
            expected[up] = 1.0
            np.testing.assert_allclose(np.array(pole), expected, atol=1e-6)

    def test_floor_sits_at_floor_height(self):
        # The shadow-catcher floor sits at the given ground altitude, flat along
        # the up-axis — not below the scene's lowest geometry.
        for up in (0, 1, 2):
            lo = np.array([-2.0, -2.0, -2.0])
            hi = np.array([3.0, 3.0, 3.0])
            lo[up] = 0.5
            stage = self._author(up_axis=up, aabb=(lo, hi), floor_height=1.25)
            pts = np.array(UsdGeom.Mesh(stage.GetPrimAtPath("/root/_Lookdev/Floor")).GetPointsAttr().Get())
            self.assertTrue(np.allclose(pts[:, up], 1.25))

    def test_floor_size_scales_with_scene(self):
        # Scale-invariance: the floor extent is a multiple of the scene diagonal
        # (never a world constant); its altitude stays the fixed ground level.
        halves = {}
        for scale in (0.01, 100.0):
            lo = np.array([-2.0, -2.0, 0.5]) * scale
            hi = np.array([3.0, 3.0, 4.0]) * scale
            stage = self._author(up_axis=2, aabb=(lo, hi))  # floor_height default 0
            pts = np.array(UsdGeom.Mesh(stage.GetPrimAtPath("/root/_Lookdev/Floor")).GetPointsAttr().Get())
            halves[scale] = float(pts[:, 0].max() - pts[:, 0].min()) / 2.0
            self.assertTrue(np.allclose(pts[:, 2], 0.0))  # floor at ground level regardless of scale
        self.assertAlmostEqual(halves[100.0] / halves[0.01], 10000.0, delta=1.0)

    def test_floor_skipped_without_bounds(self):
        stage = self._author(aabb=None)
        self.assertFalse(stage.GetPrimAtPath("/root/_Lookdev/Floor").IsValid())
        # Dome + lights are still authored (they are scale-invariant).
        self.assertEqual(len(self._by_type(stage, "DistantLight")), 3)

    def test_floor_is_rtx_matte_object(self):
        # The floor is flagged as an Omniverse RTX matte object so it renders as
        # a transparent shadow catcher (sky shows through) rather than an opaque
        # plate.
        stage = self._author()
        matte = stage.GetPrimAtPath("/root/_Lookdev/Floor").GetAttribute("primvars:isMatteObject")
        self.assertTrue(matte.IsValid() and matte.Get())


@unittest.skipIf(Usd is None, "usd-core not available")
class TestViewerUSDIntegration(unittest.TestCase):
    """``ViewerUSD(lookdev=...)`` authors the rig on ``set_model``."""

    @staticmethod
    def _scene():
        b = newton.ModelBuilder()
        b.add_ground_plane()
        b.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        b.add_shape_sphere(body=0, radius=0.5)
        return b.finalize()

    def _viewer(self, tmp_name, lookdev):
        path = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", tmp_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return vusd.ViewerUSD(path, up_axis="Z", lookdev=lookdev)

    def test_lookdev_mode_authors_rig(self):
        v = self._viewer("ld_on.usda", LookdevMode.LIGHT)
        v.set_model(self._scene())
        self.assertTrue(v.stage.GetPrimAtPath("/root/_Lookdev/Sky").IsValid())
        self.assertTrue(v.stage.GetPrimAtPath("/root/_Lookdev/Floor").IsValid())

    def test_floor_sits_on_ground_plane_and_texture_is_layer_relative(self):
        # The scene has ``add_ground_plane`` at z=0, so the shadow-catcher floor
        # sits at z=0 (objects rest on it). The dome texture is an explicitly
        # layer-relative path (leading ``./`` or ``../``) resolving to the
        # committed asset.
        v = self._viewer("ld_ground.usda", LookdevMode.DARK)
        v.set_model(self._scene())
        pts = np.array(UsdGeom.Mesh(v.stage.GetPrimAtPath("/root/_Lookdev/Floor")).GetPointsAttr().Get())
        self.assertTrue(np.allclose(pts[:, 2], 0.0))
        tex = UsdLux.DomeLight(v.stage.GetPrimAtPath("/root/_Lookdev/Sky")).GetTextureFileAttr().Get().path
        self.assertTrue(tex.startswith(("./", "../")), f"expected explicit layer-relative path, got {tex}")
        resolved = os.path.normpath(os.path.join(os.path.dirname(v.output_path), tex))
        self.assertEqual(resolved, vusd._lookdev_sky_texture_path(LookdevMode.DARK))

    def test_ground_plane_altitude(self):
        alt = vusd._ground_plane_altitude(self._scene(), up_axis=2)
        self.assertEqual(alt, 0.0)
        # No ground plane -> None (floor falls back to 0).
        b = newton.ModelBuilder()
        b.add_body(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        b.add_shape_sphere(body=0, radius=0.5)
        self.assertIsNone(vusd._ground_plane_altitude(b.finalize(), up_axis=2))

    def test_off_authors_no_rig(self):
        # The dome is the default background source, so lookdev needs no viewport
        # render-setting overrides; with lookdev off, nothing is authored at all.
        v = self._viewer("ld_off.usda", None)
        v.set_model(self._scene())
        self.assertFalse(v.stage.GetPrimAtPath("/root/_Lookdev").IsValid())

    def test_lookdev_hides_ground_plane(self):
        # With lookdev active the shadow-catcher floor replaces the collision
        # plane, so ``GeoType.PLANE`` is hidden (mirrors ViewerGL); off, it shows.
        flags = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.VISIBLE)
        plane, sphere = int(newton.GeoType.PLANE), int(newton.GeoType.SPHERE)
        on = self._viewer("ld_plane_on.usda", LookdevMode.LIGHT)
        self.assertFalse(on._should_show_shape(flags, False, geo_type=plane))
        # A non-plane shape is unaffected by the lookdev filter.
        self.assertTrue(on._should_show_shape(int(newton.ShapeFlags.VISIBLE), False, geo_type=sphere))
        off = self._viewer("ld_plane_off.usda", None)
        self.assertTrue(off._should_show_shape(flags, False, geo_type=plane))


if __name__ == "__main__":
    unittest.main(verbosity=2)
