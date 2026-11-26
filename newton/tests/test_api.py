# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import subprocess
import sys
import textwrap
import typing as _t
import unittest


def _get_type_hints(obj):
    """Return evaluated type hints, including extras if available."""
    return _t.get_type_hints(obj, globalns=getattr(obj, "__globals__", None), include_extras=True)


def _param_list(sig: inspect.Signature):
    """Return list of parameters excluding the first one."""
    return list(sig.parameters.values())[1:]


def _is_builder_arg_doc_line(line: str) -> bool:
    """Return True for legacy and type-less Google-style builder arg docs."""
    return line.startswith("builder (ModelBuilder):") or line.startswith("builder:")


def _check_builder_method_matches_importer_function_signature(func, method):
    func_name = func.__name__
    method_name = method.__name__
    sig_func = inspect.signature(func)
    sig_method = inspect.signature(method)

    # Compare parameter lists (excluding the first, which differs: builder vs self)
    func_params = _param_list(sig_func)
    method_params = _param_list(sig_method)

    assert len(func_params) == len(method_params), (
        f"Parameter count mismatch (excluding first): "
        f"{len(func_params)} ({func_name}) != {len(method_params)} ({method_name})"
    )

    # Type hints (evaluated), used to check user-annotated types match
    hints_func = _get_type_hints(func)
    hints_method = _get_type_hints(method)

    # Helper to fetch the *user-annotated* type for a param name; missing => inspect._empty
    def annotated_type(hints_dict, obj, name):
        if name in getattr(obj, "__annotations__", {}):
            # If user provided an annotation, compare the evaluated version
            return hints_dict.get(name, inspect._empty)
        return inspect._empty

    for i, (pf, pm) in enumerate(zip(func_params, method_params, strict=False), start=1):
        # Names must match 1:1 (beyond builder/self)
        assert pf.name == pm.name, f"Param #{i} name mismatch: {pf.name!r} ({func_name}) != {pm.name!r} ({method_name})"
        # Kinds must match (*, /, positional-only, var-positional, keyword-only)
        assert pf.kind == pm.kind, (
            f"Param {pf.name!r} kind mismatch: {pf.kind} ({func_name}) != {pm.kind} ({method_name})"
        )
        # Defaults must match
        assert pf.default == pm.default, (
            f"Param {pf.name!r} default mismatch: {pf.default!r} ({func_name}) != {pm.default!r} ({method_name})"
        )
        # User-annotated type hints must match (if present)
        at_func = annotated_type(hints_func, func, pf.name)
        at_method = annotated_type(hints_method, method, pm.name)
        assert at_func == at_method, (
            f"Param {pf.name!r} annotation mismatch: {at_func!r} ({func_name}) != {at_method!r} ({method_name})"
        )

    # Return type annotations must match (only if user annotated them)
    func_has_ret_annot = "return" in getattr(func, "__annotations__", {})
    method_has_ret_annot = "return" in getattr(method, "__annotations__", {})

    if func_has_ret_annot or method_has_ret_annot:
        ret_func = hints_func.get("return", inspect._empty)
        ret_method = hints_method.get("return", inspect._empty)
        assert ret_func == ret_method, (
            f"Return type annotation mismatch: {ret_func!r} ({func_name}) != {ret_method!r} ({method_name})"
        )

    # Docstrings must match (ignoring surrounding whitespace and indentation)
    lines_doc_func = [line.strip() for line in (func.__doc__ or "").splitlines()]
    # Remove line that contains the docstring for the ModelBuilder argument
    # because this argument does not exist in the method
    doc_func = "\n".join(line for line in lines_doc_func if not _is_builder_arg_doc_line(line)).strip()
    lines_doc_method = [line.strip() for line in (method.__doc__ or "").splitlines()]
    assert not any(_is_builder_arg_doc_line(line) for line in lines_doc_method), (
        f"Docstring for {method_name} must not document the builder argument"
    )
    doc_method = "\n".join(lines_doc_method).strip()
    assert doc_func == doc_method, f"Docstring mismatch between {func_name} and {method_name}"


class TestApi(unittest.TestCase):
    def test_null_viewer_does_not_load_unused_backends(self):
        """Keep an unusable optional backend from affecting the null viewer."""
        environment = os.environ.copy()
        for name in ("DISPLAY", "WAYLAND_DISPLAY", "PYGLET_HEADLESS"):
            environment.pop(name, None)
        script = textwrap.dedent("""
            import importlib.abc
            import sys

            class UnavailableBackend(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "rerun":
                        raise RuntimeError("The unused Rerun backend cannot initialize")

            sys.meta_path.insert(0, UnavailableBackend())
            from newton.viewer import ViewerNull
            viewer = ViewerNull(num_frames=1)
            assert viewer.is_running()
            viewer.begin_frame(0.0)
            viewer.end_frame()
            assert not viewer.is_running()
            viewer.close()
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_narrow_phase_keeps_pipeline_controls_private(self):
        """Keep topology specializations and counter ownership out of the public API."""
        from newton.geometry import NarrowPhase  # noqa: PLC0415

        parameters = inspect.signature(NarrowPhase).parameters
        for name in ("_tetrahedron_separation",):
            self.assertNotIn(name, parameters)
        self.assertNotIn("_counters_prepared", inspect.signature(NarrowPhase.launch_custom_write).parameters)

    def test_joint_target_layout_flag_removed(self):
        """Expose coordinate-shaped position targets without a layout switch."""
        import newton  # noqa: PLC0415

        self.assertNotIn("use_coord_layout_targets", newton.__all__)
        self.assertFalse(hasattr(newton, "use_coord_layout_targets"))
        builder = newton.ModelBuilder()
        body = builder.add_link()
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        self.assertFalse(hasattr(model, "use_coord_layout_targets"))
        self.assertEqual(model.joint_target_q.shape, (7,))
        self.assertEqual(model.control().joint_target_q.shape, (7,))
        self.assertIs(model.joint_target_q_start, model.joint_q_start)

    def test_collision_pipeline_flattens_speculative_contact_config(self):
        """Configure speculative contacts without a single-field wrapper."""
        import newton  # noqa: PLC0415

        parameters = inspect.signature(newton.CollisionPipeline).parameters

        self.assertIn("speculative_contact_gap_max", parameters)
        self.assertNotIn("max_speculative_extension", parameters)
        self.assertNotIn("speculative_config", parameters)
        self.assertFalse(hasattr(newton.CollisionPipeline, "SpeculativeContactConfig"))

    def test_builder_urdf_signature_parity(self):
        from newton import ModelBuilder  # noqa: PLC0415
        from newton._src.utils.import_urdf import parse_urdf  # noqa: PLC0415

        _check_builder_method_matches_importer_function_signature(parse_urdf, ModelBuilder.add_urdf)

    def test_builder_mjcf_signature_parity(self):
        from newton import ModelBuilder  # noqa: PLC0415
        from newton._src.utils.import_mjcf import parse_mjcf  # noqa: PLC0415

        _check_builder_method_matches_importer_function_signature(parse_mjcf, ModelBuilder.add_mjcf)

    def test_builder_usd_signature_parity(self):
        from newton import ModelBuilder  # noqa: PLC0415
        from newton._src.utils.import_usd import parse_usd  # noqa: PLC0415

        _check_builder_method_matches_importer_function_signature(parse_usd, ModelBuilder.add_usd)

    def test_tetmesh_create_from_usd_docstring_parity(self):
        from newton import TetMesh  # noqa: PLC0415
        from newton._src.usd.utils import get_tetmesh  # noqa: PLC0415

        doc_func = "\n".join(line.strip() for line in (get_tetmesh.__doc__ or "").splitlines()).strip()
        doc_method = "\n".join(line.strip() for line in (TetMesh.create_from_usd.__doc__ or "").splitlines()).strip()
        assert doc_func == doc_method, "Docstring mismatch between get_tetmesh and TetMesh.create_from_usd"

    def test_get_tetmesh_hides_importer_material_control(self):
        """Keep importer-only material controls out of the public signature."""
        import newton.usd  # noqa: PLC0415

        parameters = inspect.signature(newton.usd.get_tetmesh).parameters

        self.assertNotIn("_load_material", parameters)

    def test_keyword_only_arguments_reject_positional_use(self):
        """Reject positional use of mature keyword-only API options."""
        import warp as wp  # noqa: PLC0415

        import newton  # noqa: PLC0415

        builder = newton.ModelBuilder()
        model = builder.finalize()

        calls = (
            lambda: builder.add_shape_box(-1, wp.transform(), 0.1, 0.2, 0.3),
            lambda: builder.add_body(wp.transform()),
            lambda: newton.ModelBuilder.ShapeConfig(12.0, 34.0),
            lambda: newton.ModelBuilder.JointDofConfig(newton.Axis.Y),
            lambda: newton.solvers.SolverSemiImplicit(model, 0.123),
        )

        for call in calls:
            with self.subTest(call=call), self.assertRaises(TypeError):
                call()

    def test_deprecate_nonkeyword_arguments_rebinds_options(self):
        """Rebind positional options while a deprecation window is active."""
        from newton._src.utils.deprecation import deprecate_nonkeyword_arguments  # noqa: PLC0415

        @deprecate_nonkeyword_arguments
        def func(value, *, option=None):
            return value, option

        with self.assertWarnsRegex(DeprecationWarning, "Passing 'option' positionally"):
            result = func(1, 2)

        self.assertEqual(result, (1, 2))

    def test_deprecate_nonkeyword_arguments_rejects_duplicate_keyword(self):
        """Reject options supplied both positionally and by keyword."""
        from newton._src.utils.deprecation import deprecate_nonkeyword_arguments  # noqa: PLC0415

        @deprecate_nonkeyword_arguments
        def func(value, *, option=None):
            return value, option

        with self.assertRaisesRegex(TypeError, "multiple values for argument 'option'"):
            func(1, 2, option=3)

    def test_keyword_only_api_signatures(self):
        """Expose mature optional API arguments as keyword-only."""
        import newton  # noqa: PLC0415

        body_sig = inspect.signature(newton.ModelBuilder.add_body)
        link_sig = inspect.signature(newton.ModelBuilder.add_link)
        solver_sig = inspect.signature(newton.solvers.SolverSemiImplicit.__init__)

        self.assertEqual(body_sig.parameters["xform"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(link_sig.parameters["xform"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(solver_sig.parameters["angular_damping"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_shape_opacity_follows_color(self):
        """Keep shape display arguments adjacent and keyword-only."""
        import newton  # noqa: PLC0415

        method_names = (
            "add_shape",
            "add_shape_plane",
            "add_ground_plane",
            "add_shape_sphere",
            "add_shape_ellipsoid",
            "add_shape_box",
            "add_shape_capsule",
            "add_shape_cylinder",
            "add_shape_cone",
            "add_shape_mesh",
            "add_shape_convex_hull",
            "add_shape_heightfield",
            "add_shape_gaussian",
        )

        for method_name in method_names:
            with self.subTest(method=method_name):
                parameters = list(inspect.signature(getattr(newton.ModelBuilder, method_name)).parameters.values())
                color_index = next(i for i, parameter in enumerate(parameters) if parameter.name == "color")
                opacity = parameters[color_index + 1]
                self.assertEqual(opacity.name, "opacity")
                self.assertEqual(opacity.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_viewer_workspace_covers_attribute_frequencies(self):
        """Require every Model.AttributeFrequency to be mapped or explicitly handled in the viewer workspace."""
        import newton  # noqa: PLC0415
        from newton._src.viewer.gui.properties import PropertyInspector  # noqa: PLC0415

        Frequency = newton.Model.AttributeFrequency
        # Frequencies resolved through composite domains (scene/world, joint spans, deformable objects, contacts)
        # rather than a direct 1:1 PropertyInspector.kinds leaf key.
        handled_indirectly = {
            Frequency.ONCE,
            Frequency.WORLD,
            Frequency.JOINT_COORD,
            Frequency.JOINT_DOF,
            Frequency.JOINT_CONSTRAINT,
            Frequency.SURFACE,
            Frequency.VOLUME,
            Frequency.CONTACT,
            Frequency.CONTACT_RIGID,
            Frequency.CONTACT_SOFT,
        }
        mapped = set(PropertyInspector.kinds.values()) | handled_indirectly
        missing = [freq.name for freq in Frequency if freq not in mapped]
        self.assertEqual(
            missing,
            [],
            f"Unmapped Model.AttributeFrequency member(s) in viewer workspace: {missing}. "
            "Wire them into PropertyInspector / ModelTree (or document indirect handling here).",
        )

    def test_viewer_workspace_covers_geometry_and_joint_icons(self):
        """Ensure every GeoType and JointType resolves to a real icon in the viewer atlas."""
        import newton  # noqa: PLC0415
        from newton._src.viewer.gui.icons import ICON_NAMES  # noqa: PLC0415
        from newton._src.viewer.gui.workspace import ViewerWorkspace  # noqa: PLC0415

        valid_icons = set(ICON_NAMES)
        for geo in newton.GeoType:
            with self.subTest(geo=geo.name):
                self.assertIn(int(geo), ViewerWorkspace.SHAPE_ICONS, f"GeoType.{geo.name} missing from SHAPE_ICONS")
                icon = ViewerWorkspace.SHAPE_ICONS[int(geo)]
                self.assertIn(icon, valid_icons, f"GeoType.{geo.name} maps to unknown icon {icon!r}")
        for joint in newton.JointType:
            with self.subTest(joint=joint.name):
                icon = ViewerWorkspace.joint_icon(joint)
                self.assertIn(icon, valid_icons, f"JointType.{joint.name} maps to unknown icon {icon!r}")

    def test_viewer_workspace_covers_viewer_display_options(self):
        """Ensure every ViewerBase show_* display option is exposed in ViewportControls."""
        import newton.viewer  # noqa: PLC0415
        from newton._src.viewer.gui.viewport import ViewportControls  # noqa: PLC0415

        viewer = newton.viewer.ViewerNull()
        discovered = {
            name
            for owner in (viewer.layer, viewer)
            for name, value in vars(owner).items()
            if name.startswith("show_") and isinstance(value, bool)
        }
        covered = (
            {name for name, _ in ViewportControls.body_guide_options}
            | {name for name, _, _ in ViewportControls.scene_guide_options}
            | set(ViewportControls.contact_display_options)
            | set(ViewportControls.mode_display_options)
        )
        missing = sorted(discovered - covered)
        self.assertEqual(
            missing,
            [],
            f"ViewerBase display option(s) not exposed in ViewportControls: {missing}. "
            "Add them to ViewportControls so the workspace GUI stays in sync.",
        )

    def test_viewer_collision_inspector_contact_fields_classification(self):
        """Ensure non-per-contact arrays on Contacts cannot leak into per-contact inspector rows."""
        import warp as wp  # noqa: PLC0415

        import newton  # noqa: PLC0415
        from newton._src.viewer.gui.collision import CollisionInspector  # noqa: PLC0415

        contacts = newton.Contacts(
            rigid_contact_max=4,
            soft_contact_max=3,
            device="cpu",
            per_contact_shape_properties=True,
            contact_matching=True,
            contact_report=True,
            rigid_contact_surface_velocity=True,
        )
        for prefix, capacity in (("rigid_contact_", 4), ("soft_contact_", 3)):
            for name, value in vars(contacts).items():
                if not name.startswith(prefix) or not isinstance(value, wp.array):
                    continue
                suffix = name[len(prefix) :]
                if suffix in CollisionInspector.NON_PER_CONTACT_SUFFIXES or suffix.startswith("diff_"):
                    continue
                self.assertEqual(
                    len(value),
                    capacity,
                    f"Contacts.{name} has length {len(value)} != {capacity} but is not excluded in "
                    "CollisionInspector.NON_PER_CONTACT_SUFFIXES.",
                )

    def test_viewer_attribute_summary_covers_all_core_specs(self):
        """Ensure attribute_summary formats every core Model AttributeSpec including update_flags cleanly."""
        import newton  # noqa: PLC0415
        from newton._src.viewer.viewer_attributes import Assignment, attribute_summary  # noqa: PLC0415

        for name, spec in newton.Model._CORE_ATTRIBUTE_SPECS.items():
            summary = attribute_summary(name, spec, None, Assignment.MODEL)
            self.assertIn(f"Model.{name}", summary)
            if spec.update_flags is not None:
                self.assertIn("Update flags:", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
