# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import typing as _t
import unittest


def _get_type_hints(obj):
    """Return evaluated type hints, including extras if available."""
    return _t.get_type_hints(obj, globalns=getattr(obj, "__globals__", None), include_extras=True)


def _param_list(sig: inspect.Signature):
    """Return list of parameters excluding the first one."""
    return list(sig.parameters.values())[1:]


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
    doc_func = "\n".join(line for line in lines_doc_func if "builder (ModelBuilder)" not in line).strip()
    doc_method = "\n".join(line.strip() for line in (method.__doc__ or "").splitlines()).strip()
    assert "builder (ModelBuilder)" not in doc_method, (
        f"Docstring for {method_name} must not contain 'builder (ModelBuilder)'"
    )
    assert doc_func == doc_method, f"Docstring mismatch between {func_name} and {method_name}"


class TestApi(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
