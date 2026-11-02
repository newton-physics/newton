# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check ranges and deferred defaults against the native compiler."""

import importlib.util
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_MJCF_DEFAULT, SOLREF_MODE_RAW
from newton.exceptions import NewtonWarning
from newton.solvers import SolverMuJoCo


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is required")
class TestImportMjcfNativeSemantics(unittest.TestCase):
    def test_native_custom_attribute_mapping(self):
        """Extract inherited values before scaling and retain their filtered, reordered, copied rows."""
        frequency = newton.Model.AttributeFrequency
        builder = newton.ModelBuilder()

        def source_size(element, context):
            self.assertEqual(context["scale"], 2)
            self.assertIs(context["attr"].dtype, wp.vec3)
            self.assertTrue(context["spec"].compiler.degree)
            return element.size

        for name, freq, dtype, default, transform in (
            ("body_tag", frequency.BODY, wp.float32, -1, lambda e, _: e.userdata[0] if len(e.userdata) else None),
            ("shape_tag", frequency.SHAPE, wp.float32, -1, lambda e, _: e.userdata[0] if len(e.userdata) else None),
            ("source_size", frequency.SHAPE, wp.vec3, wp.vec3(), source_size),
            ("joint_tag", frequency.JOINT, wp.float32, -1, lambda e, _: e.userdata[0]),
            ("dof_tag", frequency.JOINT_DOF, wp.float32, -1, lambda e, _: e.userdata[0]),
        ):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=freq,
                    dtype=dtype,
                    default=default,
                    mjcf_native_value_transformer=transform,
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "parts.xml").write_text("""<mujoco>
                <body name="omit" user="99"><geom size=".1"/></body>
                <body name="arm" user="10"><joint name="hinge" user="1"/>
                    <joint name="slide" type="slide" user="2"/><geom size=".1"/>
                    <site name="site" size=".02" user="8"/></body>
                <body name="ball"><joint type="ball"/><geom size=".2"/></body>
                <body name="free"><joint type="free" user="9"/><geom size=".3"/></body>
            </mujoco>""")
            (path / "main.xml").write_text("""<mujoco><default><joint user="3"/>
                <geom user="7"/></default><worldbody><include file="parts.xml"/></worldbody></mujoco>""")
            builder.add_mjcf_native(path / "main.xml", scale=2, ignore_names=("omit",))
        builder.add_mjcf_native(
            '<mujoco><worldbody><body user="11"><geom size=".4"/></body></worldbody></mujoco>',
            parent_body=2,
            scale=2,
            parse_mujoco_options=False,
        )
        copied = newton.ModelBuilder()
        copied.replicate(builder, 2)
        model = copied.finalize()
        np.testing.assert_array_equal(model.body_tag.numpy(), [10, -1, -1, 11] * 2)
        np.testing.assert_array_equal(model.shape_tag.numpy(), [7, 7, 7, 8, -1] * 2)
        np.testing.assert_array_equal(model.joint_tag.numpy(), [1, 3, 9, -1] * 2)
        np.testing.assert_array_equal(model.dof_tag.numpy(), [2, 1, 3, 3, 3, 9, 9, 9, 9, 9, 9] * 2)
        np.testing.assert_allclose(model.source_size.numpy()[:, 0], [0.1, 0.2, 0.3, 0.02, 0.4] * 2)

    def test_native_custom_attribute_rows(self):
        """Map custom-frequency and option values through repeated imports and composition."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        for kind in ("tendon", "actuator", "pair", "equality_constraint"):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=f"source_{kind}",
                    frequency=f"mujoco:{kind}",
                    dtype=str,
                    mjcf_native_value_transformer=lambda e, _: e.name,
                )
            )
        for name, frequency, transform in (
            ("source_model", newton.Model.AttributeFrequency.ONCE, lambda spec, _: spec.modelname),
            ("source_angle", newton.Model.AttributeFrequency.WORLD, lambda spec, _: str(spec.compiler.degree)),
        ):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=frequency,
                    dtype=str,
                    mjcf_native_value_transformer=transform,
                )
            )
        xml = """<mujoco model="custom"><worldbody>
            <body><joint name="j"/><geom name="a" size=".1"/></body>
            <body><joint name="k"/><geom name="b" size=".1"/></body>
            <body><joint/><geom name="c" size=".1"/></body></worldbody>
            <tendon><fixed name="t"><joint joint="j" coef="1"/></fixed></tendon>
            <actuator><motor name="motor" joint="j"/></actuator>
            <contact><pair name="bc" geom1="b" geom2="c"/>
                <pair name="ab" geom1="a" geom2="b"/></contact>
            <equality><joint name="equality" joint1="j" joint2="k"/></equality></mujoco>"""
        builder.add_mjcf_native(xml)
        builder.add_mjcf_native(
            xml.replace('model="custom"', 'model="later"').replace(
                "<worldbody>", '<compiler angle="radian"/><worldbody>'
            ),
            parse_mujoco_options=False,
        )
        copied = newton.ModelBuilder()
        copied.replicate(builder, 2)
        model = copied.finalize()
        for kind, expected in (
            ("tendon", "t"),
            ("actuator", "motor"),
            ("equality_constraint", "equality"),
        ):
            self.assertEqual(getattr(model, f"source_{kind}"), [expected] * 4)
        self.assertEqual(model.source_pair, ["ab", "bc"] * 4)
        self.assertEqual(model.source_model, ["custom"])
        self.assertEqual(model.source_angle, ["True", "True"])

    def test_native_custom_attribute_compiler_fusion(self):
        """Keep source values with surviving elements when MuJoCo fuses static bodies."""
        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="source_body",
                frequency=newton.Model.AttributeFrequency.BODY,
                dtype=str,
                mjcf_native_value_transformer=lambda body, _: body.name,
            )
        )
        builder.add_mjcf_native("""<mujoco><compiler fusestatic="true"/><worldbody>
            <body name="fixed"><geom size=".1"/></body>
            <body name="moving"><joint/><geom size=".1"/></body></worldbody></mujoco>""")
        self.assertEqual(builder.finalize().source_body, ["moving"])

    def test_native_custom_attribute_errors(self):
        """Propagate hook errors and reject unmapped frequencies before adding entities."""

        def fail(_element, _context):
            raise RuntimeError("extraction failed")

        xml = '<mujoco><worldbody><body><geom size=".1"/></body></worldbody></mujoco>'
        for frequency, hook, error, message in (
            (newton.Model.AttributeFrequency.BODY, fail, RuntimeError, "extraction failed"),
            (newton.Model.AttributeFrequency.PARTICLE, lambda e, c: 1, ValueError, "frequency"),
        ):
            with self.subTest(frequency=frequency):
                builder = newton.ModelBuilder()
                builder.add_custom_attribute(
                    newton.ModelBuilder.CustomAttribute(
                        name="value",
                        frequency=frequency,
                        dtype=wp.float32,
                        mjcf_native_value_transformer=hook,
                    )
                )
                with self.assertRaisesRegex(error, message):
                    builder.add_mjcf_native(xml)
                self.assertEqual((builder.body_count, builder.shape_count, builder.joint_count), (0, 0, 0))

    def test_joint_limit_default_provenance(self):
        cases = (
            ("", "", None),
            ("", 'solreflimit=".02 1"', [0.02, 1]),
            ("", 'solreflimit=".03 1"', [0.03, 1]),
            ("", 'solreflimit="0 0"', [0, 0]),
            ('<default><joint solreflimit="-100 -2"/></default>', "", [-100, -2]),
            ('<default><default class="child"/></default>', 'class="child"', None),
            (
                '<default><joint solreflimit=".04 .8"/><default class="child"/></default>',
                'class="child"',
                [0.04, 0.8],
            ),
            (
                '<default><joint solreflimit=".04 .8"/><default class="child"><joint solreflimit="0 0"/></default></default>',
                'class="child"',
                [0, 0],
            ),
        )
        for kind in ("hinge", "slide", "ball"):
            for custom in (False, True):
                for defaults, attributes, expected in cases:
                    with self.subTest(kind=kind, custom=custom, defaults=defaults, attributes=attributes):
                        builder = newton.ModelBuilder()
                        if custom:
                            builder.default_joint_cfg.limit_ke = 4321
                            builder.default_joint_cfg.limit_kd = 76
                        builder.add_mjcf_native(f'''<mujoco>{defaults}<worldbody><body>
                            <joint type="{kind}" limited="true" range="0 1" {attributes}/>
                            <geom size=".1"/></body></worldbody></mujoco>''')
                        model = builder.finalize()
                        np.testing.assert_allclose(model.joint_limit_ke.numpy(), builder.default_joint_cfg.limit_ke)
                        np.testing.assert_allclose(model.joint_limit_kd.numpy(), builder.default_joint_cfg.limit_kd)
                        mode = (
                            SOLREF_MODE_RAW
                            if expected is not None
                            else (SOLREF_MODE_FORCE_SPACE if custom else SOLREF_MODE_MJCF_DEFAULT)
                        )
                        np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), mode)
                        np.testing.assert_allclose(
                            model.mujoco.solreflimit.numpy(),
                            np.tile([0, 0] if expected is None else expected, (model.joint_dof_count, 1)),
                        )
                        np.testing.assert_allclose(
                            model.mujoco.solreflimit_gain_baseline.numpy(),
                            np.tile(
                                [builder.default_joint_cfg.limit_ke, builder.default_joint_cfg.limit_kd],
                                (model.joint_dof_count, 1),
                            ),
                        )

    def test_joint_limit_provenance_on_compound_joints(self):
        builder = newton.ModelBuilder()
        builder.default_joint_cfg.limit_ke = 4321
        builder.add_mjcf_native("""<mujoco><worldbody><body>
            <joint name="slide" type="slide" solreflimit=".02 1"/>
            <joint name="hinge" axis="1 0 0"/>
            <joint name="other" axis="0 1 0" solreflimit=".03 1"/>
            <geom size=".1"/></body></worldbody></mujoco>""")
        model = builder.finalize()
        np.testing.assert_array_equal(
            model.mujoco.solreflimit_mode.numpy(), [SOLREF_MODE_RAW, SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_RAW]
        )
        np.testing.assert_allclose(model.mujoco.solreflimit.numpy(), [[0.02, 1], [0, 0], [0.03, 1]])

    def test_joint_limit_provenance_with_includes(self):
        files = {
            "defaults.xml": """<mujocoinclude><default><default class="authored">
                <joint solreflimit=".02 1"/></default><default class="empty"/></default></mujocoinclude>""",
            "bodies.xml": """<mujocoinclude><body childclass="authored"><joint/><geom size=".1"/></body>
                <body childclass="empty"><joint/><geom size=".1"/></body></mujocoinclude>""",
        }
        xml = """<mujoco><include file="defaults.xml"/>
            <worldbody><include file="bodies.xml"/></worldbody></mujoco>"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.xml"
            source.write_text(xml)
            for name, content in files.items():
                (Path(directory) / name).write_text(content)
            for custom_resolver in (False, True):
                with self.subTest(custom_resolver=custom_resolver):
                    calls = []

                    def resolve(base, name, calls=calls):
                        calls.append(name)
                        return files[name]

                    builder = newton.ModelBuilder()
                    builder.default_joint_cfg.limit_ke = 4321
                    builder.add_mjcf_native(
                        xml if custom_resolver else source, path_resolver=resolve if custom_resolver else None
                    )
                    model = builder.finalize()
                    np.testing.assert_array_equal(
                        model.mujoco.solreflimit_mode.numpy(), [SOLREF_MODE_RAW, SOLREF_MODE_FORCE_SPACE]
                    )
                    if custom_resolver:
                        self.assertCountEqual(calls, files)

    def test_joint_limit_provenance_after_composition_and_gain_edits(self):
        xml = """<mujoco><worldbody><body name="root"><joint name="j" range="-30 30"/>
            <geom type="box" size=".1 .2 .3" mass="1"/>
            <body name="authored" pos="0 .3 0"><joint name="raw" range="-30 30" solreflimit=".02 1"/>
            <geom size=".1" mass="1"/></body></body></worldbody></mujoco>"""
        load = """<mujoco><worldbody><body name="load" pos=".4 .1 0">
            <geom size=".2" mass="4"/></body></worldbody></mujoco>"""
        for edit in ("configured", "before_solver", "after_solver"):
            with self.subTest(edit=edit):
                solvers = []
                for importer in ("add_mjcf", "add_mjcf_native"):
                    builder = newton.ModelBuilder()
                    if edit == "configured":
                        builder.default_joint_cfg.limit_ke = 4321
                        builder.default_joint_cfg.limit_kd = 76
                    getattr(builder, importer)(xml)
                    getattr(builder, importer)(load, parent_body=0)
                    model = builder.finalize()
                    if edit == "before_solver":
                        model.joint_limit_ke.fill_(4321)
                        model.joint_limit_kd.fill_(76)
                    solver = SolverMuJoCo(model, disable_contacts=True)
                    if edit == "after_solver":
                        np.testing.assert_allclose(solver.mjw_model.jnt_solref.numpy()[0], [[0.02, 1], [0.02, 1]])
                        model.joint_limit_ke.fill_(4321)
                        model.joint_limit_kd.fill_(76)
                        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
                    np.testing.assert_array_equal(
                        model.mujoco.solreflimit_mode.numpy(), [SOLREF_MODE_FORCE_SPACE, SOLREF_MODE_RAW]
                    )
                    np.testing.assert_allclose(solver.mjw_model.jnt_solref.numpy()[0, 1], [0.02, 1])
                    self.assertFalse(np.allclose(solver.mjw_model.jnt_solref.numpy()[0, 0], [0.02, 1]))
                    solvers.append(solver)
                np.testing.assert_allclose(solvers[1].mj_model.jnt_solref, solvers[0].mj_model.jnt_solref, rtol=1e-5)
                np.testing.assert_allclose(
                    solvers[1].mjw_model.jnt_solref.numpy(), solvers[0].mjw_model.jnt_solref.numpy(), rtol=1e-5
                )

    def test_standard_mjcf_defaults(self):
        """Preserve standard default classes in Newton core and MuJoCo attributes."""
        import mujoco

        xml = """<mujoco><compiler angle="radian"/>
            <default><geom condim="4" priority="5" solmix="2" gap=".01"/>
                <joint damping="3" stiffness="2" ref=".3" springref=".4" actuatorgravcomp="true"/>
                <general gainprm="100" biasprm="0 -100 -10" biastype="affine"
                    dyntype="filter" dynprm=".5" ctrlrange="-2 2" forcerange="-100 100"/>
                <default class="special"><geom condim="6" solmix="5"/>
                    <joint damping="30" stiffness="20"/>
                    <general gainprm="200" biasprm="0 -200 -20"/>
                    <default class="child"><geom priority="99" gap=".05"/></default>
                </default>
            </default><worldbody>
                <body name="a" gravcomp=".5"><joint name="a"/><geom size=".1"/>
                    <body name="b" childclass="special" gravcomp=".75"><joint name="b"/><geom size=".1"/>
                        <body name="c" childclass="child"><joint name="c" damping="0"/>
                            <geom size=".1" condim="1"/>
                        </body>
                    </body>
                </body>
            </worldbody><actuator><general joint="a"/><general joint="b" class="special"/>
                <general joint="c" class="child" gainprm="999"/>
            </actuator></mujoco>"""
        reference = mujoco.MjModel.from_xml_string(xml)
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(xml, ctrl_direct=True)
        model = builder.finalize()
        for field, expected in (
            (model.shape_gap, reference.geom_gap),
            (model.joint_damping, reference.dof_damping),
            (model.mujoco.gravcomp, reference.body_gravcomp[1:]),
            (model.mujoco.condim, reference.geom_condim),
            (model.mujoco.geom_priority, reference.geom_priority),
            (model.mujoco.geom_solmix, reference.geom_solmix),
            (model.mujoco.dof_ref, reference.qpos0),
            (model.mujoco.dof_springref, reference.qpos_spring),
            (model.mujoco.dof_passive_stiffness, reference.jnt_stiffness),
            (model.mujoco.jnt_actgravcomp, reference.jnt_actgravcomp),
            (model.mujoco.actuator_gainprm, reference.actuator_gainprm),
            (model.mujoco.actuator_biasprm, reference.actuator_biasprm),
            (model.mujoco.actuator_dynprm, reference.actuator_dynprm),
        ):
            np.testing.assert_allclose(field.numpy(), expected, atol=1e-6)
        solver = SolverMuJoCo(model, disable_contacts=True)
        for field in ("actuator_ctrlrange", "actuator_forcerange", "actuator_dyntype", "actuator_actadr"):
            np.testing.assert_allclose(getattr(solver.mj_model, field), getattr(reference, field))

    def test_native_default_precedence(self):
        file_defaults = '<joint damping=".4" armature=".5"/><geom gap=".02"/>'
        zero_defaults = '<joint damping="0" armature="0"/><geom gap="0"/>'
        cases = (
            ("", "", "", "", 7, 2, None),
            (f"<default>{file_defaults}</default>", "", "", "", 0.4, 0.5, 0.02),
            (f"<default>{zero_defaults}</default>", "", "", "", 0, 0, 0),
            ('<default><default class="child"/></default>', 'childclass="child"', "", "", 7, 2, None),
            (
                f'<default>{file_defaults}<default class="child">{zero_defaults}</default></default>',
                'childclass="child"',
                "",
                "",
                0,
                0,
                0,
            ),
            (
                f'<default>{zero_defaults}<default class="child">{file_defaults}</default></default>',
                "",
                'class="child"',
                'class="child"',
                0.4,
                0.5,
                0.02,
            ),
            (
                f'<default>{file_defaults}<default class="child"/></default>',
                'childclass="child"',
                'damping="0" armature="0"',
                'gap="0"',
                0,
                0,
                0,
            ),
        )
        for kind in ("hinge", "slide", "ball", "free"):
            for scale in (1, 3):
                for defaults, body, joint, geom, damping, armature, gap in cases:
                    with self.subTest(kind=kind, scale=scale, defaults=defaults, joint=joint):
                        xml = f'''<mujoco>{defaults}<worldbody><body {body}>
                            <joint type="{kind}" {joint}/><geom size=".1" {geom}/>
                            </body></worldbody></mujoco>'''
                        builder = newton.ModelBuilder()
                        builder.default_joint_cfg.damping = 7
                        builder.default_joint_cfg.armature = 2
                        builder.rigid_gap = 0.3
                        builder.default_shape_cfg.gap = 0.2
                        builder.default_shape_cfg.kf = 321
                        builder.add_mjcf_native(xml, scale=scale, armature_scale=2)
                        np.testing.assert_allclose(builder.joint_damping, damping)
                        np.testing.assert_allclose(builder.joint_armature, armature * 2)
                        np.testing.assert_allclose(builder.shape_gap, 0.2 if gap is None else gap * scale)
                        np.testing.assert_allclose(builder.shape_material_kf, 321)
                        self.assertEqual(builder.default_joint_cfg.damping, 7)
                        self.assertEqual(builder.default_shape_cfg.gap, 0.2)

    def test_native_builder_defaults_per_import(self):
        for gap in (None, 0.0, 0.2):
            with self.subTest(gap=gap):
                builder = newton.ModelBuilder()
                builder.default_shape_cfg.gap = gap
                builder.rigid_gap = 0.3
                for damping in (7, 9):
                    builder.default_joint_cfg.damping = damping
                    builder.default_joint_cfg.armature = 2
                    builder.add_mjcf_native(
                        """<mujoco><worldbody><body><freejoint/>
                        <geom size=".1"/></body><body><joint/><geom size=".1"/>
                        </body></worldbody></mujoco>""",
                        scale=3,
                    )
                np.testing.assert_allclose(builder.joint_damping, [*[0] * 6, 7, *[0] * 6, 9])
                np.testing.assert_allclose(builder.joint_armature, [*[0] * 6, 2, *[0] * 6, 2])
                np.testing.assert_allclose(builder.shape_gap, 0.3 if gap is None else gap)

    def test_native_builder_defaults_with_includes(self):
        import mujoco

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parts").mkdir()
            (root / "meshes").mkdir()
            source = root / "model.xml"
            xml = """<mujoco><compiler meshdir="meshes"/>
                <include file="parts/defaults.xml"/>
                <asset><mesh name="mesh" file="tetra.obj"/></asset>
                <worldbody><include file="parts/body.xml"/></worldbody></mujoco>"""
            source.write_text(xml)
            (root / "parts/defaults.xml").write_text("""<mujocoinclude><default>
                <joint damping=".4"/><default class="child"><geom gap="0"/></default>
                </default></mujocoinclude>""")
            (root / "parts/body.xml").write_text("""<mujocoinclude><body childclass="child">
                <joint/><geom type="mesh" mesh="mesh"/></body></mujocoinclude>""")
            (root / "meshes/tetra.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
            )
            reference = mujoco.MjModel.from_xml_path(str(source))
            builder = newton.ModelBuilder()
            builder.default_joint_cfg.damping = 7
            builder.default_joint_cfg.armature = 2
            builder.add_mjcf_native(source)
            np.testing.assert_allclose(builder.joint_damping, 0.4)
            np.testing.assert_allclose(builder.joint_armature, 2)
            np.testing.assert_allclose(builder.shape_gap, 0)
            np.testing.assert_allclose(builder.body_mass, reference.body_mass[1:], rtol=1e-6)
            self.assertEqual(source.read_text(), xml)

    def test_native_builder_defaults_with_path_resolver(self):
        builder = newton.ModelBuilder()
        builder.default_joint_cfg.damping = 7
        builder.default_joint_cfg.armature = 2
        builder.add_mjcf_native(
            '<mujoco><include file="part.xml"/></mujoco>',
            path_resolver=lambda base, name: (
                """<mujocoinclude>
                <default><joint damping="0"/></default><worldbody><body><joint/>
                <geom size=".1"/></body></worldbody></mujocoinclude>"""
            ),
        )
        np.testing.assert_allclose(builder.joint_damping, 0)
        np.testing.assert_allclose(builder.joint_armature, 2)

    def test_inherited_actuator_ranges(self):
        """Resolve inherited control and activation ranges after builder edits."""
        import mujoco

        for actuator in ("position", "intvelocity"):
            for direct in (False, True):
                with self.subTest(actuator=actuator, direct=direct):
                    xml = f"""<mujoco><compiler angle="radian"/><default><joint range="-.2 .4"/>
                        <{actuator} kp="10" inheritrange=".8"/></default><worldbody><body>
                        <joint name="j"/><geom size=".1"/></body></worldbody>
                        <actuator><{actuator} joint="j"/></actuator></mujoco>"""
                    builder = newton.ModelBuilder()
                    builder.add_mjcf_native(xml, ctrl_direct=direct)
                    builder.joint_limit_lower[0], builder.joint_limit_upper[0] = -0.5, 0.7
                    solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                    reference = mujoco.MjModel.from_xml_string(xml.replace('range="-.2 .4"', 'range="-.5 .7"'))
                    for field in (
                        "actuator_ctrlrange",
                        "actuator_actrange",
                        "actuator_ctrllimited",
                        "actuator_actlimited",
                    ):
                        np.testing.assert_allclose(
                            getattr(solver.mj_model, field), getattr(reference, field), rtol=1e-6, atol=1e-7
                        )
                    solver.notify_model_changed(newton.ModelFlags.ACTUATOR_PROPERTIES)
                    for field in ("actuator_ctrlrange", "actuator_actrange"):
                        np.testing.assert_allclose(
                            getattr(solver.mjw_model, field).numpy()[0], getattr(reference, field), rtol=1e-6, atol=1e-7
                        )

    def test_inherited_ranges_cpu_updates(self):
        for actuator, field in (("position", "ctrlrange"), ("intvelocity", "actrange")):
            with self.subTest(actuator=actuator):
                xml = f"""<mujoco><worldbody><body><joint name="j" range="-20 40"/><geom size=".1"/></body></worldbody>
                    <actuator><{actuator} joint="j" kp="10" inheritrange=".8"/></actuator></mujoco>"""
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(xml)
                model = builder.finalize()
                solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
                getattr(model.mujoco, f"actuator_{field}").assign([[-0.1, 0.2]])
                solver.notify_model_changed(newton.ModelFlags.ACTUATOR_PROPERTIES)
                np.testing.assert_allclose(getattr(solver.mj_model, f"actuator_{field}"), [[-0.1, 0.2]], atol=1e-7)

    def test_authored_auto_flags(self):
        """Keep tri-state limits and automatic activation dimensions in builder data."""
        xml = """<mujoco><worldbody><body><joint name="j" range="-20 40"/><geom size=".1"/></body></worldbody>
            <tendon><fixed name="t" range="-1 2"><joint joint="j" coef="1"/></fixed></tendon>
            <actuator><general joint="j" dyntype="filter" actrange="-.1 .2" ctrlrange="-1 2"/></actuator></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf_native(xml)
        for name in (
            "native_joint_limited",
            "native_joint_actfrclimited",
            "tendon_limited",
            "tendon_actuator_force_limited",
            "actuator_ctrllimited",
            "actuator_forcelimited",
            "actuator_actlimited",
        ):
            self.assertEqual(int(builder.custom_attributes[f"mujoco:{name}"].values[0]), 2, name)
        self.assertEqual(builder.custom_attributes["mujoco:actuator_actdim"].values[0], -1)

    def test_limit_range_matrix(self):
        """Distinguish disabled, inferred, and explicit limits, including zero ranges."""
        import mujoco

        for flag in ("auto", "false", "true"):
            for bounds in ("0 0", "-2 3", "1 -1"):
                for autolimits in ("true", "false"):
                    with self.subTest(flag=flag, bounds=bounds, autolimits=autolimits):
                        xml = f'''<mujoco><compiler autolimits="{autolimits}"/><worldbody><body><joint name="j"/>
                            <geom size=".1"/></body></worldbody><actuator><general joint="j" dyntype="filter"
                            ctrllimited="{flag}" ctrlrange="{bounds}" forcelimited="{flag}" forcerange="{bounds}"
                            actlimited="{flag}" actrange="{bounds}"/></actuator></mujoco>'''
                        builder = newton.ModelBuilder()
                        try:
                            reference = mujoco.MjModel.from_xml_string(xml)
                        except ValueError:
                            with self.assertRaises(ValueError):
                                builder.add_mjcf_native(xml)
                            self.assertEqual(builder.body_count, 0)
                            continue
                        builder.add_mjcf_native(xml, ctrl_direct=True)
                        solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                        for field in (
                            "actuator_ctrlrange",
                            "actuator_forcerange",
                            "actuator_actrange",
                            "actuator_ctrllimited",
                            "actuator_forcelimited",
                            "actuator_actlimited",
                        ):
                            np.testing.assert_allclose(getattr(solver.mj_model, field), getattr(reference, field))

    def test_joint_limit_defaults(self):
        """Retain disabled ranges and automatic flags through solver notifications."""
        import mujoco

        for kind, bounds in (("hinge", "-20 40"), ("slide", "-.2 .4"), ("ball", "0 30")):
            for flag in ("auto", "true", "false"):
                with self.subTest(kind=kind, flag=flag):
                    xml = f'<mujoco><worldbody><body><joint type="{kind}" limited="{flag}" range="{bounds}"/><geom size=".1"/></body></worldbody></mujoco>'
                    reference = mujoco.MjModel.from_xml_string(xml)
                    builder = newton.ModelBuilder()
                    builder.add_mjcf_native(xml)
                    solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                    for field in ("jnt_limited", "jnt_range", "jnt_actfrclimited", "jnt_actfrcrange", "jnt_solref"):
                        np.testing.assert_allclose(
                            getattr(solver.mj_model, field), getattr(reference, field), atol=1e-7
                        )
                    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                    np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[0], reference.jnt_range, atol=1e-7)
                    if kind == "hinge" and flag == "false":
                        model = solver.model
                        model.mujoco.dof_ref.assign([0.5])
                        solver.notify_model_changed(newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES)
                        np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[0], reference.jnt_range)
                        model.joint_limit_lower.assign([-0.25])
                        model.joint_limit_upper.assign([0.75])
                        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                        np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[0], [[0.25, 1.25]])
                        rebuilt = SolverMuJoCo(model, disable_contacts=True)
                        np.testing.assert_array_equal(rebuilt.mj_model.jnt_limited, [True])
                        np.testing.assert_allclose(rebuilt.mj_model.jnt_range, [[0.25, 1.25]])

    def test_near_servo_parameters(self):
        """Preserve general gains and gears that only approximately match a servo."""
        import mujoco

        for params in ('biasprm="0 -10.00008 -1"', 'biasprm="0 -10 -1" gear="1.000009"'):
            with self.subTest(params=params):
                xml = f'<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody><actuator><general joint="j" biastype="affine" gainprm="10" {params}/></actuator></mujoco>'
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(xml)
                solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                reference = mujoco.MjModel.from_xml_string(xml)
                for field in ("actuator_gainprm", "actuator_biasprm", "actuator_gear"):
                    np.testing.assert_allclose(
                        getattr(solver.mj_model, field), getattr(reference, field), rtol=1e-7, atol=1e-8
                    )

    def test_shortcut_defaults_and_zero_overrides(self):
        """Keep native shortcut defaults and explicit zero overrides through both control routes."""
        import mujoco

        for actuator in (
            '<motor joint="j"/>',
            '<position joint="j" kp="0"/>',
            '<position joint="j" kv="0"/>',
            '<position joint="j" kp="10" timeconst=".02"/>',
            '<velocity joint="j" kv="0"/>',
            '<intvelocity joint="j" kp="10" actrange="-.2 .4"/>',
            '<damper joint="j" kv="2" ctrlrange="0 1"/>',
            '<cylinder joint="j"/>',
            '<muscle joint="j" lengthrange=".1 .5"/>',
        ):
            for direct in (False, True):
                with self.subTest(actuator=actuator, direct=direct):
                    defaults = (
                        '<default><position kp="3" kv="2"/><velocity kv="4"/></default>'
                        if not actuator.startswith("<muscle")
                        else ""
                    )
                    xml = f"""<mujoco>{defaults}
                        <worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody><actuator>{actuator}</actuator></mujoco>"""
                    reference = mujoco.MjModel.from_xml_string(xml)
                    builder = newton.ModelBuilder()
                    builder.add_mjcf_native(xml, ctrl_direct=direct)
                    solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                    for field in (
                        "actuator_gainprm",
                        "actuator_biasprm",
                        "actuator_dynprm",
                        "actuator_dyntype",
                        "actuator_actnum",
                        "actuator_ctrlrange",
                        "actuator_actrange",
                    ):
                        np.testing.assert_allclose(
                            getattr(solver.mj_model, field), getattr(reference, field), atol=1e-7, rtol=1e-6
                        )

    def test_world_option_defaults(self):
        """Assign per-world options to the world being imported."""
        builder = newton.ModelBuilder()
        for impratio in (2, 3):
            builder.begin_world()
            builder.add_mjcf_native(
                f'<mujoco><option impratio="{impratio}"/><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>'
            )
            builder.end_world()
        np.testing.assert_array_equal(builder.finalize().mujoco.impratio.numpy(), [2, 3])

    def test_tendon_springlength_sentinels(self):
        """Recompute automatic tendon rest lengths while retaining explicit zero."""
        import mujoco

        for springlength in ("-1", "0", "6", ".2 .4"):
            with self.subTest(springlength=springlength):
                xml = f'''<mujoco><worldbody><site name="anchor"/><body><joint name="j" type="slide"/>
                    <geom size=".1"/><site name="tip" pos="0 0 1"/></body></worldbody>
                    <tendon><spatial springlength="{springlength}"><site site="anchor"/><site site="tip"/></spatial></tendon></mujoco>'''
                builder = newton.ModelBuilder()
                builder.add_mjcf_native(xml)
                tip = next(i for i, name in enumerate(builder.shape_label) if name.endswith("/tip"))
                builder.shape_transform[tip] = wp.transform((0, 0, 2), wp.quat_identity())
                solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                reference = mujoco.MjModel.from_xml_string(xml.replace('pos="0 0 1"', 'pos="0 0 2"'))
                np.testing.assert_allclose(solver.mj_model.tendon_lengthspring, reference.tendon_lengthspring)

    def test_auto_weld_pose_across_worlds(self):
        import mujoco

        builder = newton.ModelBuilder()
        expected = []
        for position in (1, 2):
            xml = f'''<mujoco><worldbody><body name="b" pos="{position} 0 0"><freejoint/><geom size=".1"/></body></worldbody>
                <equality><weld body1="b"/></equality></mujoco>'''
            template = newton.ModelBuilder()
            template.add_mjcf_native(xml, convert_mjc_equality_constraints=False)
            builder.add_world(template)
            expected.append(mujoco.MjModel.from_xml_string(xml).eq_data[0])
        solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[:, 0], expected)
        solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[:, 0], expected)

    def test_joint_and_tendon_force_ranges(self):
        import mujoco

        for flag in ("auto", "true", "false"):
            for bounds in ("0 0", "-2 3", "1 -1"):
                with self.subTest(flag=flag, bounds=bounds):
                    xml = f'''<mujoco><default><joint actuatorfrclimited="{flag}" actuatorfrcrange="{bounds}"/>
                        <tendon limited="{flag}" range="{bounds}" actuatorfrclimited="{flag}" actuatorfrcrange="{bounds}"/></default>
                        <worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody>
                        <tendon><fixed><joint joint="j" coef="1"/></fixed></tendon></mujoco>'''
                    builder = newton.ModelBuilder()
                    try:
                        reference = mujoco.MjModel.from_xml_string(xml)
                    except ValueError:
                        with self.assertRaises(ValueError):
                            builder.add_mjcf_native(xml)
                        self.assertEqual(builder.body_count, 0)
                        continue
                    builder.add_mjcf_native(xml)
                    solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                    solver.notify_model_changed(
                        newton.ModelFlags.JOINT_DOF_PROPERTIES | newton.ModelFlags.TENDON_PROPERTIES
                    )
                    for field in ("jnt_actfrcrange", "tendon_range", "tendon_actfrcrange"):
                        np.testing.assert_allclose(getattr(solver.mj_model, field), getattr(reference, field))
                        np.testing.assert_allclose(
                            getattr(solver.mjw_model, field).numpy()[0], getattr(reference, field)
                        )
                    for field in ("jnt_actfrclimited", "tendon_limited", "tendon_actfrclimited"):
                        np.testing.assert_array_equal(getattr(solver.mj_model, field), getattr(reference, field))

    def test_weld_auto_relpose(self):
        """Resolve unspecified weld poses after placing the imported model."""
        import mujoco

        for convert in (False, True):
            with self.subTest(convert=convert):
                xml = """<mujoco><worldbody><body name="b" pos="1 0 0"><freejoint/><geom size=".1"/></body></worldbody>
                    <equality><weld body1="b"/></equality></mujoco>"""
                builder = newton.ModelBuilder()
                warning = self.assertWarnsRegex(NewtonWarning, "FREE joint parallel") if convert else nullcontext()
                with warning:
                    builder.add_mjcf_native(
                        xml, xform=wp.transform((2, 0, 0), wp.quat_identity()), convert_mjc_equality_constraints=convert
                    )
                solver = SolverMuJoCo(builder.finalize(), disable_contacts=True, kinematic_switchable_roots=False)
                reference = mujoco.MjModel.from_xml_string(xml.replace('pos="1 0 0"', 'pos="3 0 0"'))
                np.testing.assert_allclose(solver.mj_model.eq_data, reference.eq_data)
                solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
                np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[0], reference.eq_data)


if __name__ == "__main__":
    unittest.main()
