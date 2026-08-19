// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Runtime tests for newton::controllers::Controller. Takes the directory holding
// the artifacts written by generate_artifacts.py as argv[1]; CTest supplies it.
//
// The model and the inputs are mirrored from generate_artifacts.py; change them
// together.

#include <newton/controllers/controller.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << "\n";
        ++failures;
    }
}

void check_close(float got, float expected, float tolerance, const std::string& what) {
    if (!(std::fabs(got - expected) <= tolerance)) {
        std::cerr << "FAILED: " << what << " — expected " << expected << ", got " << got << "\n";
        ++failures;
    }
}

std::vector<float> read_reference(const std::filesystem::path& path) {
    std::ifstream file(path);
    std::vector<float> values;
    float value = 0.0f;
    while (file >> value) values.push_back(value);
    return values;
}

// Mirrors generate_artifacts.py.
const std::vector<float> kJointQ{0.35f, -0.62f, 0.18f, 0.9f};
const std::vector<float> kJointQd{0.12f, 0.4f, -0.25f, 0.05f};
const std::vector<float> kJointQDes{0.0f, -0.3f, 0.5f};
const std::vector<float> kJointQdDes{0.0f, 0.1f, 0.0f};
constexpr float kDt = 0.01f;

// A four-link arm with three controlled joints: the model-sized ports carry one
// entry per model DOF, the compact ports one per controlled DOF.
constexpr std::size_t kModelDofs = 4;
constexpr std::size_t kControlledDofs = 3;

// The second artifact binds joint_q_des and joint_f to views of a
// simulation-sized array, so those two fields have one entry per simulation DOF
// and the graph touches only the addressed slots.
constexpr std::size_t kViewSimDofs = 6;
const std::vector<std::size_t> kViewIndices{1, 3, 5};

void test_buffers_are_sized_from_the_graph(const std::filesystem::path& wrp) {
    newton::controllers::Controller controller{wrp};
    const auto input = controller.input();
    const auto output = controller.output();

    check(input.fields.count("joint_q") == 1, "input has joint_q");
    check(input.fields.count("joint_q_des") == 1, "input has joint_q_des");
    check(output.fields.count("joint_f") == 1, "output has joint_f");

    check(input["joint_q"].size() == kModelDofs, "joint_q covers every model DOF");
    check(input["joint_qd"].size() == kModelDofs, "joint_qd covers every model DOF");
    check(input["joint_q_des"].size() == kControlledDofs, "joint_q_des covers the controlled DOFs");
    check(input["joint_qd_des"].size() == kControlledDofs, "joint_qd_des covers the controlled DOFs");
    check(output["joint_f"].size() == kControlledDofs, "joint_f covers the controlled DOFs");

    // dt is a graph parameter but belongs to neither buffer.
    check(input.fields.count("dt") == 0, "dt is not an input field");
}

void test_params_lists_every_parameter(const std::filesystem::path& wrp) {
    newton::controllers::Controller controller{wrp};
    bool has_dt = false;
    bool has_joint_f = false;
    for (const auto& param : controller.params()) {
        if (param.name == "dt") {
            has_dt = true;
            check(param.size_bytes == sizeof(float), "dt is one float");
        }
        if (param.name == "output.joint_f") {
            has_joint_f = true;
            check(param.size_bytes == kControlledDofs * sizeof(float), "output.joint_f is one float per controlled DOF");
        }
    }
    check(has_dt, "params() reports dt");
    check(has_joint_f, "params() reports output.joint_f");
}

void test_unknown_field_throws(const std::filesystem::path& wrp) {
    newton::controllers::Controller controller{wrp};
    auto input = controller.input();
    bool threw = false;
    try {
        (void)input["not_a_field"];
    } catch (const std::out_of_range&) {
        threw = true;
    }
    check(threw, "an unknown field name throws std::out_of_range");
}

void test_step_matches_python(const std::filesystem::path& wrp, const std::filesystem::path& reference_path) {
    const std::vector<float> expected = read_reference(reference_path);
    check(expected.size() == kControlledDofs, "reference file holds one torque per controlled DOF");
    if (expected.size() != kControlledDofs) return;

    newton::controllers::Controller controller{wrp};
    auto input = controller.input();
    auto output = controller.output();

    input["joint_q"] = kJointQ;
    input["joint_qd"] = kJointQd;
    input["joint_q_des"] = kJointQDes;
    input["joint_qd_des"] = kJointQdDes;

    controller.step(input, output, kDt);

    for (std::size_t dof = 0; dof < expected.size(); ++dof) {
        check_close(output["joint_f"][dof], expected[dof], 1e-3f,
                    "joint_f[" + std::to_string(dof) + "] matches the Python controller");
    }

    // Stepping again with the same inputs must give the same torques.
    const std::vector<float> first = output["joint_f"];
    controller.step(input, output, kDt);
    for (std::size_t dof = 0; dof < first.size(); ++dof) {
        check_close(output["joint_f"][dof], first[dof], 0.0f, "a repeated step is deterministic");
    }
}

void test_view_bound_ports_round_trip(const std::filesystem::path& wrp,
                                      const std::filesystem::path& reference_path) {
    const std::vector<float> expected = read_reference(reference_path);
    if (expected.size() != kControlledDofs) {
        check(false, "reference file holds one torque per controlled DOF");
        return;
    }

    newton::controllers::Controller controller{wrp};
    auto input = controller.input();
    auto output = controller.output();

    // A view-bound port is exchanged at the simulation array's length; a port
    // bound to a compact array keeps the controlled-DOF length, so one buffer
    // holds both.
    check(input["joint_q_des"].size() == kViewSimDofs, "joint_q_des is exchanged at the simulation length");
    check(output["joint_f"].size() == kViewSimDofs, "joint_f is exchanged at the simulation length");
    check(input["joint_qd_des"].size() == kControlledDofs, "joint_qd_des stays at the controlled length");

    input["joint_q"] = kJointQ;
    input["joint_qd"] = kJointQd;
    input["joint_qd_des"] = kJointQdDes;

    // Desired positions at the addressed slots; deliberate junk everywhere else,
    // which the graph must ignore.
    std::vector<float>& q_des = input["joint_q_des"];
    q_des.assign(kViewSimDofs, -999.0f);
    for (std::size_t dof = 0; dof < kViewIndices.size(); ++dof) {
        q_des[kViewIndices[dof]] = kJointQDes[dof];
    }

    controller.step(input, output, kDt);

    const std::vector<float>& joint_f = output["joint_f"];
    for (std::size_t dof = 0; dof < kViewIndices.size(); ++dof) {
        check_close(joint_f[kViewIndices[dof]], expected[dof], 1e-3f,
                    "joint_f at simulation slot " + std::to_string(kViewIndices[dof])
                        + " matches the compactly-bound controller");
    }

    // Slots the view does not address keep the values they held at capture.
    for (std::size_t slot = 0; slot < kViewSimDofs; ++slot) {
        const bool addressed =
            std::find(kViewIndices.begin(), kViewIndices.end(), slot) != kViewIndices.end();
        if (!addressed) {
            check_close(joint_f[slot], 0.0f, 0.0f,
                        "unaddressed simulation slot " + std::to_string(slot) + " is untouched");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path artifacts = argc > 1 ? argv[1] : "artifacts";
    const std::filesystem::path wrp = artifacts / "joint_impedance.wrp";
    const std::filesystem::path views_wrp = artifacts / "joint_impedance_views.wrp";
    const std::filesystem::path reference = artifacts / "joint_impedance_expected.txt";

    if (!std::filesystem::exists(wrp) || !std::filesystem::exists(views_wrp)
        || !std::filesystem::exists(reference)) {
        std::cerr << "Missing artifacts in " << artifacts << "; run generate_artifacts.py first.\n";
        return 1;
    }

    try {
        test_buffers_are_sized_from_the_graph(wrp);
        test_params_lists_every_parameter(wrp);
        test_unknown_field_throws(wrp);
        test_step_matches_python(wrp, reference);
        test_view_bound_ports_round_trip(views_wrp, reference);
    } catch (const std::exception& error) {
        std::cerr << "FAILED: unexpected exception: " << error.what() << "\n";
        ++failures;
    }

    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all checks passed\n";
    return 0;
}
