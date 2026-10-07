# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static checks of the FeatherPGS public surface and its private kernel factories."""

import ast
import inspect
import re
import typing
import unittest
from pathlib import Path

import newton
from newton.solvers import SolverFeatherPGS

_PACKAGE_DIR = Path(__file__).parents[1] / "_src" / "solvers" / "feather_pgs"

_KERNEL_FACTORIES = (
    "_get_pack_mf_meta_kernel",
    "_get_pgs_solve_mf_gs_kernel",
    "_get_pgs_solve_local_owned_kernel",
    "_get_cholesky_kernel",
    "_get_crba_cholesky_kernel",
    "_get_crba_cholesky_warp_kernel",
    "_get_cholesky_inverse_tiled_kernel",
    "_get_inverse_cholesky_register_kernel",
    "_get_paired_hinv_jt_kernel",
    "_get_pgs_solve_paired_factor_kernel",
    "_get_triangular_solve_kernel",
    "_get_hinv_jt_kernel",
    "_get_hinv_jt_plain_kernel",
    "_get_joint_limit_warp_kernel",
    "_get_hinv_jt_fused_kernel",
    "_get_delassus_kernel",
    "_get_pgs_solve_tiled_row_kernel",
    "_get_pgs_solve_tiled_contact_kernel",
    "_get_pgs_solve_streaming_kernel",
    "_get_pgs_solve_mf_kernel",
    "_get_pgs_solve_propagation_contact_kernel",
    "_get_pgs_solve_propagation_full_iteration_kernel",
    "_get_factor_propagation_tree_revolute_kernel",
    "_get_propagation_tree_body_response_revolute_kernel",
    "_get_propagate_tree_impulses_revolute_kernel",
    "_get_refresh_propagation_tree_body_qd_warp_kernel",
    "_get_mark_independent_sparse_contact_candidates_kernel",
    "_get_build_independent_sparse_contact_groups_kernel",
    "_get_pgs_solve_sparse_diagonal_kernel",
    "_get_propagation_tree_cached_response_kernel",
    "_get_propagation_cached_gemv_kernel",
    "_get_color_propagation_prebuild_kernel",
    "_get_colored_friction_setup_kernel",
    "_get_colored_payload_pack_kernel",
    "_get_pgs_solve_propagation_colored_warp_kernel",
    "_get_pgs_solve_propagation_colored_block_kernel",
)


class TestFeatherPGSPrivateApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solver_path = _PACKAGE_DIR / "solver_feather_pgs.py"
        cls.solver_module = ast.parse(solver_path.read_text(encoding="utf-8"))
        cls.top_level_functions = {
            node.name: node for node in cls.solver_module.body if isinstance(node, ast.FunctionDef)
        }
        cls.solver_class = next(
            node
            for node in cls.solver_module.body
            if isinstance(node, ast.ClassDef) and node.name == "SolverFeatherPGS"
        )
        cls.solver_methods = {node.name: node for node in cls.solver_class.body if isinstance(node, ast.FunctionDef)}

    def test_solver_is_exported_once_from_newton_solvers(self):
        """Expose the solver through newton.solvers only."""
        self.assertIn("SolverFeatherPGS", newton.solvers.__all__)
        self.assertIs(newton.solvers.SolverFeatherPGS, SolverFeatherPGS)
        self.assertNotIn("SolverFeatherPGS", newton.__all__)

    def test_solver_is_marked_experimental(self):
        """Mark the public solver class with the Sphinx experimental directive."""
        self.assertIn(".. experimental::", SolverFeatherPGS.__doc__)

    def test_constructor_options_are_keyword_only(self):
        """Keep every option after the model keyword-only."""
        parameters = list(inspect.signature(SolverFeatherPGS.__init__).parameters.values())
        self.assertEqual([p.name for p in parameters[:2]], ["self", "model"])
        for parameter in parameters[2:]:
            with self.subTest(parameter=parameter.name):
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_global_physics_switches_default_to_the_model(self):
        """Keep physical behavior local to the model unless a solver-wide override is requested.

        ``enable_joint_limits``, the experimental ``contact_compliance`` and the experimental
        ``friction_mode`` are opt-in. The solver-wide damping, friction and restitution overrides
        default to the values that leave the model's per-body and per-shape properties in charge.
        """
        parameters = inspect.signature(SolverFeatherPGS.__init__).parameters
        for opt_in in ("enable_joint_limits", "contact_compliance"):
            with self.subTest(opt_in=opt_in):
                self.assertIs(parameters[opt_in].default, False)
        self.assertEqual(parameters["friction_mode"].default, "current")
        self.assertIn(".. experimental::", SolverFeatherPGS.__doc__)
        for option, default in (
            ("angular_damping", None),
            ("enable_contact_friction", True),
            ("contact_friction_scale", 1.0),
            ("contact_friction_position_iterations", -1),
            ("enable_restitution", True),
        ):
            with self.subTest(option=option):
                self.assertEqual(parameters[option].default, default)

    def test_pgs_mode_selects_only_the_supported_solves(self):
        """Offer exactly the matrix-free and split solves, with the matrix-free solve as the default."""
        parameter = inspect.signature(SolverFeatherPGS.__init__).parameters["pgs_mode"]
        self.assertEqual(parameter.default, "matrix_free")
        self.assertEqual(typing.get_args(parameter.annotation), ("matrix_free", "split"))
        parameters = inspect.signature(SolverFeatherPGS.__init__).parameters
        self.assertNotIn("delassus_kernel", parameters)
        for option, default in (
            ("pgs_kernel", "auto"),
            ("pgs_chunk_size", None),
            ("tile_threads", 64),
            ("serial_kernel_block_dim", 256),
            ("pgs_debug", False),
            ("nvtx", False),
        ):
            with self.subTest(option=option):
                self.assertEqual(parameters[option].default, default)

    def test_solver_reads_no_environment_variables(self):
        """Configure the solver only through its constructor, never through the environment."""
        pattern = re.compile(r"\bos\.(environ|getenv)\b|FEATHER_PGS_|IL_NEWTON")
        for path in sorted(_PACKAGE_DIR.glob("*.py")):
            with self.subTest(file=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")))

    def test_prescribed_response_is_not_a_public_execution_knob(self):
        """Select the kinematic-body response internally, not through a constructor option."""
        init_method = self.solver_methods["__init__"]
        parameters = {
            argument.arg
            for argument in [*init_method.args.posonlyargs, *init_method.args.args, *init_method.args.kwonlyargs]
        }
        self.assertNotIn("exclude_fully_kinematic_free_articulations", parameters)

    def test_kernel_factories_are_private_cached_functions(self):
        """Keep the size-specialized kernel factories private, cached and keyed by device architecture."""
        for helper_name in _KERNEL_FACTORIES:
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
        """Emit the row diagonal only from the diagonal-producing H^-1 J^T variant."""
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

    def test_cholesky_and_triangular_kernels_are_built_for_every_group(self):
        """Build the factorization kernels independently of the dense row capacity."""
        init_method = self.solver_methods["_init_tiled_kernels"]
        size_group_loop = next(
            node
            for node in ast.walk(init_method)
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Attribute) and node.iter.attr == "size_groups"
        )
        calls = {
            child.func.id: child.lineno
            for child in ast.walk(size_group_loop)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        first_continue = min(
            (child.lineno for child in ast.walk(size_group_loop) if isinstance(child, ast.Continue)), default=None
        )
        for helper_name in ("_get_cholesky_kernel", "_get_triangular_solve_kernel"):
            with self.subTest(helper_name=helper_name):
                self.assertIn(helper_name, calls)
                if first_continue is not None:
                    self.assertLess(calls[helper_name], first_continue)


if __name__ == "__main__":
    unittest.main()
