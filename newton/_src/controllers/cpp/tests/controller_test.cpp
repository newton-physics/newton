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

void print_vector(const std::string& label, const std::vector<float>& values) {
    std::cout << label << ": [";
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::cout << values[index];
        if (index + 1 < values.size()) std::cout << ", ";
    }
    std::cout << "]\n";
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

// The third artifact drives ControllerJointImpedanceModelFree with its mass
// matrix bound to a view of one robot's block out of a fleet of three, so the
// parameter is every block rather than the one in use.
constexpr std::size_t kFleetRobots = 3;
constexpr std::size_t kFleetControlledRobot = 1;
constexpr std::size_t kFleetDofs = 2;
constexpr float kFleetDecoy = 99.0f;
const std::vector<float> kFleetBlock{2.0f, 0.0f, 0.0f, 4.0f};
const std::vector<float> kFleetQDes{1.0f, 1.0f};

void test_buffers_are_sized_from_the_graph(const std::filesystem::path& wrp) {
    std::cout << "\n=== test_buffers_are_sized_from_the_graph ===\n";
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
    std::cout << "\n=== test_params_lists_every_parameter ===\n";
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
    std::cout << "\n=== test_unknown_field_throws ===\n";
    newton::controllers::Controller controller{wrp};
    auto input = controller.input();
    bool threw = false;
    try {
        (void)input["not_a_field"];
    } catch (const std::out_of_range&) {
        threw = true;
    }
    check(threw, "reading an unknown field name throws std::out_of_range");
}

void test_step_reports_a_bad_field_without_throwing(const std::filesystem::path& wrp) {
    std::cout << "\n=== test_step_reports_a_bad_field_without_throwing ===\n";
    newton::controllers::Controller controller{wrp};
    auto input = controller.input();
    auto output = controller.output();

    // A field of the wrong length: step() must report it rather than throw,
    // since a control loop may not be able to afford an exception.
    input["joint_q"].resize(kModelDofs + 1, 0.0f);

    bool stepped = true;
    try {
        stepped = controller.step(input, output, kDt);
    } catch (...) {
        check(false, "step must not throw on a malformed buffer");
        return;
    }
    check(!stepped, "step reports a wrongly sized field");
    check(controller.last_error().find("joint_q") != std::string::npos,
          "last_error names the offending field, got: " + controller.last_error());

    // A field the graph does not define is reported the same way.
    auto extra = controller.input();
    extra.fields["not_a_field"] = {0.0f};
    check(!controller.step(extra, output, kDt), "step reports an unknown field");
    check(controller.last_error().find("not_a_field") != std::string::npos,
          "last_error names the unknown field, got: " + controller.last_error());
}

void test_step_matches_python(const std::filesystem::path& wrp, const std::filesystem::path& reference_path) {
    std::cout << "\n=== test_step_matches_python ===\n";
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

    check(controller.step(input, output, kDt), "step succeeds: " + controller.last_error());

    print_vector("python controller joint_f", expected);
    print_vector("C++ controller joint_f   ", output["joint_f"]);

    for (std::size_t dof = 0; dof < expected.size(); ++dof) {
        check_close(output["joint_f"][dof], expected[dof], 1e-3f,
                    "joint_f[" + std::to_string(dof) + "] matches the Python controller");
    }

    // Stepping again with the same inputs must give the same torques.
    const std::vector<float> first = output["joint_f"];
    check(controller.step(input, output, kDt), "a repeated step succeeds");
    for (std::size_t dof = 0; dof < first.size(); ++dof) {
        check_close(output["joint_f"][dof], first[dof], 0.0f, "a repeated step is deterministic");
    }
}

void test_view_bound_ports_round_trip(const std::filesystem::path& wrp,
                                      const std::filesystem::path& reference_path) {
    std::cout << "\n=== test_view_bound_ports_round_trip ===\n";
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

    check(controller.step(input, output, kDt), "step with view-bound ports succeeds: " + controller.last_error());

    const std::vector<float>& joint_f = output["joint_f"];
    print_vector("python controller joint_f (compact torques)", expected);
    print_vector("C++ controller joint_f (scattered)          ", joint_f);
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

void test_view_bound_mass_matrix(const std::filesystem::path& wrp, const std::filesystem::path& reference_path) {
    std::cout << "\n=== test_view_bound_mass_matrix ===\n";
    const std::vector<float> expected = read_reference(reference_path);
    if (expected.size() != kFleetDofs) {
        check(false, "reference file holds one torque per controlled DOF");
        return;
    }

    newton::controllers::Controller controller{wrp};
    auto input = controller.input();
    auto output = controller.output();

    // The mass matrix is exchanged as the whole fleet, not the block in use.
    check(input["mass_matrix"].size() == kFleetRobots * kFleetDofs * kFleetDofs,
          "mass_matrix is exchanged as every block in the fleet");
    check(input["joint_q_des"].size() == kFleetDofs, "joint_q_des stays at the controlled length");

    // Fill every block with a decoy, then write the real one where the view
    // points; reading the wrong block would be unmissable.
    std::vector<float>& blocks = input["mass_matrix"];
    blocks.assign(kFleetRobots * kFleetDofs * kFleetDofs, kFleetDecoy);
    const std::size_t block_start = kFleetControlledRobot * kFleetDofs * kFleetDofs;
    for (std::size_t element = 0; element < kFleetBlock.size(); ++element) {
        blocks[block_start + element] = kFleetBlock[element];
    }
    input["joint_q_des"] = kFleetQDes;

    check(controller.step(input, output, kDt), "step with a view-bound mass matrix succeeds: " + controller.last_error());

    print_vector("python controller joint_f", expected);
    print_vector("C++ controller joint_f   ", output["joint_f"]);

    for (std::size_t dof = 0; dof < expected.size(); ++dof) {
        check_close(output["joint_f"][dof], expected[dof], 1e-3f,
                    "joint_f[" + std::to_string(dof) + "] matches the Python controller");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path artifacts = argc > 1 ? argv[1] : "artifacts";
    const std::filesystem::path wrp = artifacts / "joint_impedance.wrp";
    const std::filesystem::path views_wrp = artifacts / "joint_impedance_views.wrp";
    const std::filesystem::path reference = artifacts / "joint_impedance_expected.txt";
    const std::filesystem::path mass_matrix_wrp = artifacts / "mass_matrix_view.wrp";
    const std::filesystem::path mass_matrix_reference = artifacts / "mass_matrix_view_expected.txt";

    if (!std::filesystem::exists(wrp) || !std::filesystem::exists(views_wrp)
        || !std::filesystem::exists(mass_matrix_wrp) || !std::filesystem::exists(reference)
        || !std::filesystem::exists(mass_matrix_reference)) {
        std::cerr << "Missing artifacts in " << artifacts << "; run generate_artifacts.py first.\n";
        return 1;
    }

    try {
        test_buffers_are_sized_from_the_graph(wrp);
        test_params_lists_every_parameter(wrp);
        test_unknown_field_throws(wrp);
        test_step_reports_a_bad_field_without_throwing(wrp);
        test_step_matches_python(wrp, reference);
        test_view_bound_ports_round_trip(views_wrp, reference);
        test_view_bound_mass_matrix(mass_matrix_wrp, mass_matrix_reference);
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
