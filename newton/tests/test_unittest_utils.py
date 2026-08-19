# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import io
import subprocess
import sys
import unittest

import newton.tests.unittest_utils as unittest_utils

NewtonTestCase = unittest_utils.NewtonTestCase


class TestNewtonTestCaseOutputContract(unittest.TestCase):
    def _run_test_case(self, cls: type[unittest.TestCase]) -> unittest.TestResult:
        result = unittest.TestResult()
        unittest.defaultTestLoader.loadTestsFromTestCase(cls).run(result)
        return result

    def test_unexpected_stdout_fails(self):
        class EmitsOutput(NewtonTestCase):
            def test_output(self):
                print("unexpected output")

        result = self._run_test_case(EmitsOutput)

        self.assertEqual(len(result.failures), 1)
        self.assertIn("Unexpected stdout", result.failures[0][1])
        self.assertIn("unexpected output", result.failures[0][1])

    def test_expected_output_is_required(self):
        class MissingExpectedOutput(NewtonTestCase):
            def test_output(self):
                self.expectOutputRegex(r"expected output")

        result = self._run_test_case(MissingExpectedOutput)

        self.assertEqual(len(result.failures), 1)
        self.assertIn("Missing expected output", result.failures[0][1])
        self.assertIn("expected output", result.failures[0][1])

    def test_expected_output_allows_matching_output(self):
        class EmitsExpectedOutput(NewtonTestCase):
            def test_output(self):
                self.expectOutputRegex(r"expected output")
                print("expected output")

        result = self._run_test_case(EmitsExpectedOutput)

        self.assertTrue(result.wasSuccessful())

    def test_optional_output_allows_absent_or_present_output(self):
        class AllowsAbsentOutput(NewtonTestCase):
            def test_output(self):
                self.allowOutputRegex(r"optional output")

        class AllowsPresentOutput(NewtonTestCase):
            def test_output(self):
                self.allowOutputRegex(r"optional output")
                print("optional output")

        absent_result = self._run_test_case(AllowsAbsentOutput)
        present_result = self._run_test_case(AllowsPresentOutput)

        self.assertTrue(absent_result.wasSuccessful())
        self.assertTrue(present_result.wasSuccessful())

    def test_output_regex_stream_is_respected(self):
        class ExpectsStderrOutput(NewtonTestCase):
            def test_output(self):
                self.expectOutputRegex(r"stderr output", stream="stderr")
                print("stderr output", file=sys.stderr)

        class AllowsStdoutOnly(NewtonTestCase):
            def test_output(self):
                self.allowOutputRegex(r"stream output", stream="stdout")
                print("stream output", file=sys.stderr)

        expected_result = self._run_test_case(ExpectsStderrOutput)
        wrong_stream_result = self._run_test_case(AllowsStdoutOnly)

        self.assertTrue(expected_result.wasSuccessful())
        self.assertEqual(len(wrong_stream_result.failures), 1)
        self.assertIn("Unexpected stderr", wrong_stream_result.failures[0][1])
        self.assertIn("stream output", wrong_stream_result.failures[0][1])

    def test_allowed_output_does_not_hide_other_output(self):
        class EmitsAllowedAndUnexpectedOutput(NewtonTestCase):
            def test_output(self):
                self.allowOutputRegex(r"allowed output\n?")
                print("unexpected output")
                print("allowed output")

        result = self._run_test_case(EmitsAllowedAndUnexpectedOutput)

        self.assertEqual(len(result.failures), 1)
        self.assertIn("Unexpected stdout", result.failures[0][1])
        self.assertIn("unexpected output", result.failures[0][1])
        self.assertNotIn("allowed output", result.failures[0][1])

    def test_setup_teardown_and_cleanup_output_are_captured(self):
        class EmitsSetupOutput(NewtonTestCase):
            def setUp(self):
                print("setup output")

            def test_output(self):
                pass

        class EmitsTeardownOutput(NewtonTestCase):
            def test_output(self):
                pass

            def tearDown(self):
                print("teardown output")

        class EmitsCleanupOutput(NewtonTestCase):
            def setUp(self):
                self.addCleanup(print, "cleanup output")

            def test_output(self):
                pass

        setup_result = self._run_test_case(EmitsSetupOutput)
        teardown_result = self._run_test_case(EmitsTeardownOutput)
        cleanup_result = self._run_test_case(EmitsCleanupOutput)

        self.assertEqual(len(setup_result.failures), 1)
        self.assertIn("setup output", setup_result.failures[0][1])
        self.assertEqual(len(teardown_result.failures), 1)
        self.assertIn("teardown output", teardown_result.failures[0][1])
        self.assertEqual(len(cleanup_result.failures), 1)
        self.assertIn("cleanup output", cleanup_result.failures[0][1])

    def test_subprocess_stdout_uses_same_output_contract(self):
        class RunsSubprocess(NewtonTestCase):
            def test_output(self):
                result = subprocess.CompletedProcess(
                    args=["fake-command"],
                    returncode=0,
                    stdout="subprocess output\n",
                    stderr="",
                )
                self.assertSubprocessSuccess(result, command=result.args)

        result = self._run_test_case(RunsSubprocess)

        self.assertEqual(len(result.failures), 1)
        self.assertIn("Unexpected stdout", result.failures[0][1])
        self.assertIn("subprocess output", result.failures[0][1])

    def test_subprocess_failure_reports_command_and_output(self):
        class RunsFailingSubprocess(NewtonTestCase):
            def test_output(self):
                result = subprocess.CompletedProcess(
                    args=["fake-command", "arg with spaces"],
                    returncode=7,
                    stdout="subprocess stdout\n",
                    stderr="subprocess stderr\n",
                )
                self.assertSubprocessSuccess(result, command=result.args)

        result = self._run_test_case(RunsFailingSubprocess)

        self.assertEqual(len(result.failures), 1)
        self.assertIn("Failed with return code 7", result.failures[0][1])
        self.assertIn("fake-command 'arg with spaces'", result.failures[0][1])
        self.assertIn("subprocess stdout", result.failures[0][1])
        self.assertIn("subprocess stderr", result.failures[0][1])

    def test_caught_subprocess_failure_does_not_report_output_again(self):
        unittest_utils.wp.init()

        class CatchesFailingSubprocess(NewtonTestCase):
            def test_output(self):
                result = subprocess.CompletedProcess(
                    args=["fake-command"],
                    returncode=7,
                    stdout="subprocess stdout\n",
                    stderr="subprocess stderr\n",
                )

                with self.assertRaisesRegex(AssertionError, "Failed with return code 7"):
                    self.assertSubprocessSuccess(result, command=result.args)

        result = self._run_test_case(CatchesFailingSubprocess)

        self.assertTrue(result.wasSuccessful(), result.failures)

    def test_synchronized_output_is_captured(self):
        class SynchronizeEmitsOutput(NewtonTestCase):
            def test_output(self):
                pass

        original_synchronize = unittest_utils.wp.synchronize

        def emit_output():
            print("synchronized output")

        unittest_utils.wp.synchronize = emit_output
        try:
            result = self._run_test_case(SynchronizeEmitsOutput)
        finally:
            unittest_utils.wp.synchronize = original_synchronize

        self.assertEqual(len(result.failures), 1)
        self.assertIn("Unexpected stdout", result.failures[0][1])
        self.assertIn("synchronized output", result.failures[0][1])

    def test_text_runner_progress_is_not_captured(self):
        class QuietTest(NewtonTestCase):
            def test_quiet(self):
                pass

        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(QuietTest)
        )
        self.assertTrue(result.wasSuccessful(), stream.getvalue())

        stderr_capture = unittest_utils.StdErrCapture()
        stderr_capture.begin()
        try:
            result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(QuietTest)
            )
        finally:
            stderr_capture.end()

        self.assertTrue(result.wasSuccessful())

    def test_text_runner_buffering_is_supported(self):
        class EmitsExpectedOutput(NewtonTestCase):
            def test_output(self):
                self.expectOutputRegex(r"expected output")
                print("expected output")

        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, buffer=True).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(EmitsExpectedOutput)
        )

        self.assertTrue(result.wasSuccessful(), stream.getvalue())

    def test_add_function_test_uses_newton_output_contract(self):
        class GeneratedTest(NewtonTestCase):
            pass

        def emits_output(test, device):
            print("generated test output")

        unittest_utils.add_function_test(GeneratedTest, "test_generated", emits_output, check_output=True)

        result = self._run_test_case(GeneratedTest)

        self.assertEqual(len(result.failures), 1)
        self.assertIn("Unexpected stdout", result.failures[0][1])
        self.assertIn("generated test output", result.failures[0][1])

    def test_add_function_test_without_devices_skips_before_output_capture(self):
        class GeneratedTest(NewtonTestCase):
            pass

        def test_func(test, device):
            pass

        unittest_utils.add_function_test(GeneratedTest, "test_generated", test_func, devices=[])

        stderr_capture = unittest_utils.StdErrCapture()
        stderr_capture.begin()
        try:
            result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(GeneratedTest)
            )
        finally:
            runner_output = stderr_capture.end()

        self.assertTrue(result.wasSuccessful(), runner_output)
        self.assertEqual(len(result.skipped), 1)

    def test_output_capture_begin_rolls_back_stdout_if_stderr_fails(self):
        class CaptureStub:
            def __init__(self, *, raises=False):
                self.raises = raises
                self.begin_called = False
                self.end_called = False

            def begin(self):
                self.begin_called = True
                if self.raises:
                    raise RuntimeError("stderr capture failed")

            def end(self):
                self.end_called = True
                return ""

        output_capture = unittest_utils._OutputCapture()
        stdout_capture = CaptureStub()
        stderr_capture = CaptureStub(raises=True)
        output_capture.stdout_capture = stdout_capture
        output_capture.stderr_capture = stderr_capture

        with self.assertRaisesRegex(RuntimeError, "stderr capture failed"):
            output_capture.begin()

        self.assertTrue(stdout_capture.begin_called)
        self.assertTrue(stderr_capture.begin_called)
        self.assertTrue(stdout_capture.end_called)
        self.assertFalse(output_capture.active)


if __name__ == "__main__":
    unittest.main()
