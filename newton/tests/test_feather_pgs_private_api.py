# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import unittest
from pathlib import Path

_TILED_KERNEL_HELPERS = (
    "_get_pack_mf_meta_kernel",
    "_get_pgs_solve_mf_gs_kernel",
    "_get_cholesky_kernel",
    "_get_triangular_solve_kernel",
    "_get_hinv_jt_kernel",
    "_get_hinv_jt_plain_kernel",
    "_get_hinv_jt_fused_kernel",
    "_get_delassus_kernel",
    "_get_pgs_solve_tiled_row_kernel",
    "_get_pgs_solve_tiled_contact_kernel",
    "_get_pgs_solve_streaming_kernel",
    "_get_pgs_solve_mf_kernel",
)


class TestFeatherPGSPrivateApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solver_path = Path(__file__).parents[1] / "_src" / "solvers" / "feather_pgs" / "solver_feather_pgs.py"
        cls.solver_module = ast.parse(solver_path.read_text())
        cls.top_level_functions = {
            node.name: node for node in cls.solver_module.body if isinstance(node, ast.FunctionDef)
        }
        cls.solver_class = next(
            node
            for node in cls.solver_module.body
            if isinstance(node, ast.ClassDef) and node.name == "SolverFeatherPGS"
        )
        cls.solver_methods = {node.name: node for node in cls.solver_class.body if isinstance(node, ast.FunctionDef)}

    def test_prescribed_response_is_not_a_public_execution_knob(self):
        init_method = self.solver_methods["__init__"]
        parameters = {
            argument.arg
            for argument in [*init_method.args.posonlyargs, *init_method.args.args, *init_method.args.kwonlyargs]
        }
        self.assertNotIn("exclude_fully_kinematic_free_articulations", parameters)

    def test_tiled_kernel_factory_is_not_exported(self):
        class_names = {node.name for node in self.solver_module.body if isinstance(node, ast.ClassDef)}
        self.assertNotIn("TiledKernelFactory", class_names)

    def test_tiled_kernel_helpers_are_private_cached_functions(self):
        for helper_name in _TILED_KERNEL_HELPERS:
            with self.subTest(helper_name=helper_name):
                self.assertIn(helper_name, self.top_level_functions)
                helper = self.top_level_functions[helper_name]
                decorator_names = {
                    decorator.id for decorator in helper.decorator_list if isinstance(decorator, ast.Name)
                }
                parameters = {arg.arg for arg in [*helper.args.posonlyargs, *helper.args.args, *helper.args.kwonlyargs]}
                self.assertIn("cache", decorator_names)
                self.assertIn("device_arch", parameters)

    def test_plain_hinv_jt_kernel_omits_diagonal_output(self):
        plain_factory = self.top_level_functions["_get_hinv_jt_plain_kernel"]
        diagonal_factory = self.top_level_functions["_get_hinv_jt_kernel"]
        plain_template = next(node for node in plain_factory.body if isinstance(node, ast.FunctionDef))
        diagonal_template = next(node for node in diagonal_factory.body if isinstance(node, ast.FunctionDef))

        diagonal_factory_parameters = {argument.arg for argument in diagonal_factory.args.args}
        plain_parameters = {argument.arg for argument in plain_template.args.args}
        diagonal_parameters = {argument.arg for argument in diagonal_template.args.args}
        plain_names = {node.id for node in ast.walk(plain_template) if isinstance(node, ast.Name)}
        diagonal_names = {node.id for node in ast.walk(diagonal_factory) if isinstance(node, ast.Name)}

        self.assertNotIn("diag_group", plain_parameters)
        self.assertNotIn("diag_tile", plain_names)
        self.assertIn("compute_diag", diagonal_factory_parameters)
        self.assertIn("COMPUTE_DIAG", diagonal_names)
        self.assertIn("diag_group", diagonal_parameters)
        self.assertIn("diag_tile", diagonal_names)

    def test_cholesky_and_triangular_kernels_are_not_dense_constraint_gated(self):
        init_method = self.solver_methods["_init_tiled_kernels"]
        size_group_loop = self._find_size_group_loop(init_method)
        first_dense_continue = self._first_dense_guard_continue_line(size_group_loop)

        for helper_name in ("_get_cholesky_kernel", "_get_triangular_solve_kernel"):
            with self.subTest(helper_name=helper_name):
                call_line = self._first_call_line(size_group_loop, helper_name)
                if first_dense_continue is not None:
                    self.assertLess(call_line, first_dense_continue)

        dense_guarded_none_assignments = self._dense_guarded_none_assignments(size_group_loop)
        self.assertNotIn("_cholesky_kernels_by_size", dense_guarded_none_assignments)
        self.assertNotIn("_triangular_solve_kernels_by_size", dense_guarded_none_assignments)

    @staticmethod
    def _find_size_group_loop(function_node):
        for node in ast.walk(function_node):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Attribute):
                if node.iter.attr == "size_groups":
                    return node
        raise AssertionError("_init_tiled_kernels does not iterate over self.size_groups")

    @staticmethod
    def _references_dense_max_constraints(node):
        return any(
            isinstance(child, ast.Attribute) and child.attr == "dense_max_constraints" for child in ast.walk(node)
        )

    @classmethod
    def _first_dense_guard_continue_line(cls, node):
        continue_lines = []
        for child in ast.walk(node):
            if isinstance(child, ast.If) and cls._references_dense_max_constraints(child.test):
                for body_node in child.body:
                    continue_lines.extend(
                        descendant.lineno for descendant in ast.walk(body_node) if isinstance(descendant, ast.Continue)
                    )
        return min(continue_lines) if continue_lines else None

    @staticmethod
    def _first_call_line(node, function_name):
        call_lines = [
            child.lineno
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == function_name
        ]
        if not call_lines:
            raise AssertionError(f"{function_name} is not called while initializing tiled kernels")
        return min(call_lines)

    @classmethod
    def _dense_guarded_none_assignments(cls, node):
        assigned_attrs = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.If) or not cls._references_dense_max_constraints(child.test):
                continue
            for body_node in child.body:
                for descendant in ast.walk(body_node):
                    if not isinstance(descendant, ast.Assign):
                        continue
                    if not isinstance(descendant.value, ast.Constant) or descendant.value.value is not None:
                        continue
                    for target in descendant.targets:
                        attr_name = cls._assigned_self_collection_attr(target)
                        if attr_name is not None:
                            assigned_attrs.add(attr_name)
        return assigned_attrs

    @staticmethod
    def _assigned_self_collection_attr(target):
        if isinstance(target, ast.Subscript):
            target = target.value
        if isinstance(target, ast.Attribute):
            return target.attr
        return None


if __name__ == "__main__":
    unittest.main()
