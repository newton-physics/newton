# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run an example's argument variants and return their captured results."""

import gc
import json
import runpy
import sys
import traceback
import warnings
from unittest.mock import patch

import warp as wp

from newton.tests.unittest_utils import StdErrCapture, StdOutCapture


def run_variant(module, argv, allow_deprecation_warnings=False):
    stdout, stderr = StdOutCapture(), StdErrCapture()
    stdout.begin()
    stderr.begin()
    returncode = 0
    try:
        try:
            with patch.object(sys, "argv", [module, *argv]), warnings.catch_warnings():
                if allow_deprecation_warnings:
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    warnings.filterwarnings("default", category=DeprecationWarning, module=r"newton(\.|$)")
                elif not sys.warnoptions:
                    warnings.filterwarnings("default", category=DeprecationWarning, module=r"newton(\.|$)")
                runpy.run_module(module, run_name="__main__")
        finally:
            gc.collect()
            wp.synchronize()
    except (Exception, SystemExit):
        traceback.print_exc()
        returncode = 1
    finally:
        stdout_text, stderr_text = stdout.end(), stderr.end()
    return {"returncode": returncode, "stdout": stdout_text, "stderr": stderr_text}


def main():
    with open(sys.argv[1], encoding="utf-8") as stream:
        manifest = json.load(stream)
    wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)
    results = [run_variant(manifest["module"], **case) for case in manifest["cases"]]
    with open(sys.argv[2], "w", encoding="utf-8") as stream:
        json.dump(results, stream)


if __name__ == "__main__":
    main()
