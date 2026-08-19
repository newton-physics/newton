// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace newton::controllers {

// One named parameter of a loaded graph, as reported by params().
struct ParamInfo {
    std::string name;
    std::size_t size_bytes{};
};

// Host-side named float buffers for one side of a controller's interface.
//
// Instances come from Controller::input() and Controller::output(), pre-sized
// from the graph's parameter table and zero-initialised. operator[] throws
// std::out_of_range for a name the graph does not define.
//
// Sizes are the graph's byte counts divided by sizeof(float), which is why
// newton.controllers.export_graph rejects any port not composed of float32.
struct ControllerBuffer {
    std::unordered_map<std::string, std::vector<float>> fields;

    std::vector<float>& operator[](const std::string& name) { return fields.at(name); }
    const std::vector<float>& operator[](const std::string& name) const { return fields.at(name); }
};

// A controller step, loaded from a .wrp graph written by
// newton.controllers.export_graph.
//
// Construction initialises the Warp runtime, binds CUDA device 0, and loads the
// graph. The <path>_modules directory written beside the .wrp file must be
// present.
class Controller {
public:
    explicit Controller(const std::filesystem::path& wrp_path);
    ~Controller();

    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;
    Controller(Controller&&) noexcept;
    Controller& operator=(Controller&&) noexcept;

    // Buffers sized to this graph's "input." / "output." parameters, keyed by
    // the field name with the prefix stripped. Build once and reuse each step.
    //
    // A field's length follows how the Python port was bound at export time. A
    // port bound to a compact array gives one entry per controlled DOF. A port
    // bound to a view of a simulation-sized array gives one entry per element
    // of that array, because the view carries no address of its own and the
    // indices it selects travel inside the graph. One buffer may hold fields of
    // both kinds; params() reports the length of each.
    ControllerBuffer input() const;
    ControllerBuffer output() const;

    // Write the inputs and dt into the graph, launch it, read the outputs back.
    //
    // Fill every element of each input field: for a view-bound field the graph
    // reads only the elements the view addressed, and ignores the rest. The
    // same holds in reverse for outputs — a view-bound output field comes back
    // with the addressed elements written and the others still at the values
    // they held when the graph was captured.
    void step(const ControllerBuffer& input, ControllerBuffer& output, float dt);

    // Every parameter the graph exposes, including "dt".
    std::vector<ParamInfo> params() const;

private:
    struct Impl;
    Impl* impl_{nullptr};
};

}  // namespace newton::controllers
