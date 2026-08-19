# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

add_executable(controller_test controller_test.cpp)
target_link_libraries(controller_test PRIVATE newton_controllers)

set(ARTIFACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/artifacts")

# The artifacts are captured on a GPU by Newton itself, so the generator runs
# from the same environment Newton is installed in.
if(NOT NEWTON_PYTHON AND EXISTS "${NEWTON_ROOT}/.venv/bin/python")
    set(NEWTON_PYTHON "${NEWTON_ROOT}/.venv/bin/python")
elseif(NOT NEWTON_PYTHON)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(NEWTON_PYTHON "${Python3_EXECUTABLE}")
endif()

add_test(
    NAME controller_artifacts
    COMMAND "${NEWTON_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_artifacts.py" "${ARTIFACT_DIR}"
)
set_tests_properties(controller_artifacts PROPERTIES FIXTURES_SETUP controller_artifacts)

add_test(NAME controller_runtime COMMAND controller_test "${ARTIFACT_DIR}")
set_tests_properties(controller_runtime PROPERTIES FIXTURES_REQUIRED controller_artifacts)
