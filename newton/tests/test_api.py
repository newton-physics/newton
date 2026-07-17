# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import typing as _t
import unittest
import warnings


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
    def test_geometry_match_constants_deprecated(self):
        import newton  # noqa: PLC0415
        from newton._src.geometry.contact_match import MATCH_BROKEN, MATCH_NOT_FOUND  # noqa: PLC0415

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            public_match_broken = newton.geometry.MATCH_BROKEN
            public_match_not_found = newton.geometry.MATCH_NOT_FOUND

        self.assertEqual(public_match_broken, MATCH_BROKEN)
        self.assertEqual(public_match_not_found, MATCH_NOT_FOUND)
        self.assertEqual(len(caught), 2)
        self.assertTrue(all(issubclass(warning.category, DeprecationWarning) for warning in caught))

        with self.assertRaises(AttributeError):
            _ = newton.geometry.UNKNOWN_MATCH_STATUS

        star_namespace = {}
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            exec("from newton.geometry import *", star_namespace)
        self.assertNotIn("MATCH_BROKEN", star_namespace)
        self.assertNotIn("MATCH_NOT_FOUND", star_namespace)

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

    def test_keyword_only_deprecation_shim_rebinds_builder_args(self):
        import warp as wp  # noqa: PLC0415

        from newton import ModelBuilder  # noqa: PLC0415

        builder = ModelBuilder()

        with self.assertWarnsRegex(DeprecationWarning, "Passing 'xform', 'hx', 'hy', 'hz' positionally"):
            shape = builder.add_shape_box(-1, wp.transform(), 0.1, 0.2, 0.3)

        self.assertEqual(shape, 0)
        self.assertEqual(builder.shape_count, 1)

        with self.assertWarnsRegex(DeprecationWarning, "Passing 'xform' positionally"):
            body = builder.add_body(wp.transform())

        self.assertEqual(body, 0)

    def test_keyword_only_deprecation_shim_rejects_duplicate_keyword(self):
        import warp as wp  # noqa: PLC0415

        from newton import ModelBuilder  # noqa: PLC0415

        builder = ModelBuilder()

        with self.assertRaisesRegex(TypeError, "multiple values for argument 'xform'"):
            builder.add_shape_box(-1, wp.transform(), xform=wp.transform())

    def test_keyword_only_deprecation_shim_rebinds_config_constructors(self):
        import newton  # noqa: PLC0415

        with self.assertWarnsRegex(DeprecationWarning, "Passing 'density', 'ke' positionally"):
            shape_cfg = newton.ModelBuilder.ShapeConfig(12.0, 34.0)
        self.assertEqual(shape_cfg.density, 12.0)
        self.assertEqual(shape_cfg.ke, 34.0)

        with self.assertWarnsRegex(DeprecationWarning, "Passing 'axis' positionally"):
            dof_cfg = newton.ModelBuilder.JointDofConfig(newton.Axis.Y)
        self.assertAlmostEqual(dof_cfg.axis[1], 1.0)

    def test_keyword_only_deprecation_shim_rebinds_solver_options(self):
        import newton  # noqa: PLC0415

        builder = newton.ModelBuilder()
        model = builder.finalize()

        with self.assertWarnsRegex(DeprecationWarning, "Passing 'angular_damping' positionally"):
            solver = newton.solvers.SolverSemiImplicit(model, 0.123)

        self.assertEqual(solver.angular_damping, 0.123)

    def test_keyword_only_deprecation_shim_preserves_signature(self):
        import newton  # noqa: PLC0415

        body_sig = inspect.signature(newton.ModelBuilder.add_body)
        link_sig = inspect.signature(newton.ModelBuilder.add_link)
        shape_sig = inspect.signature(newton.ModelBuilder.add_shape_box)
        solver_sig = inspect.signature(newton.solvers.SolverSemiImplicit.__init__)

        self.assertEqual(body_sig.parameters["xform"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(link_sig.parameters["xform"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(shape_sig.parameters["xform"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(solver_sig.parameters["angular_damping"].kind, inspect.Parameter.KEYWORD_ONLY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
