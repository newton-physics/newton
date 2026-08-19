// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#include <newton/controllers/controller.hpp>

#include <apic.h>
#include <warp.h>

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace newton::controllers {
namespace {

[[noreturn]] void throw_warp_error(const std::string& what) {
    const char* detail = wp_get_error_string();
    if (detail && detail[0] != '\0') {
        throw std::runtime_error(what + ": " + detail);
    }
    throw std::runtime_error(what);
}

void check_size(APICGraph* graph, const std::string& name, std::size_t got_bytes) {
    const std::size_t expected = wp_apic_get_param_size(graph, name.c_str());
    if (got_bytes != expected) {
        throw std::runtime_error(
            "Parameter '" + name + "' takes " + std::to_string(expected) + " bytes, got "
            + std::to_string(got_bytes));
    }
}

void set_param(APICGraph* graph, const std::string& name, std::span<const float> data) {
    check_size(graph, name, data.size_bytes());
    if (!wp_apic_set_param(graph, name.c_str(), data.data(), data.size_bytes())) {
        throw_warp_error("Failed to set parameter '" + name + "'");
    }
}

void get_param(APICGraph* graph, const std::string& name, std::span<float> data) {
    check_size(graph, name, data.size_bytes());
    if (!wp_apic_get_param(graph, name.c_str(), data.data(), data.size_bytes())) {
        throw_warp_error("Failed to get parameter '" + name + "'");
    }
}

ControllerBuffer buffer_for_prefix(APICGraph* graph, std::string_view prefix) {
    const int count = wp_apic_get_num_params(graph);
    ControllerBuffer buffer;
    for (int i = 0; i < count; ++i) {
        const char* name = wp_apic_get_param_name(graph, i);
        if (!name) continue;
        const std::string_view full{name};
        if (full.size() > prefix.size() && full.substr(0, prefix.size()) == prefix) {
            const std::string field{full.substr(prefix.size())};
            buffer.fields[field].assign(wp_apic_get_param_size(graph, name) / sizeof(float), 0.0f);
        }
    }
    return buffer;
}

}  // namespace

struct Controller::Impl {
    void* context{nullptr};
    APICGraph* graph{nullptr};

    ~Impl() {
        if (graph) {
            wp_apic_destroy_graph(graph);
            graph = nullptr;
        }
    }
};

Controller::Controller(const std::filesystem::path& wrp_path) : impl_(new Impl()) {
    if (wp_init(nullptr) != 0) {
        throw_warp_error("Warp runtime initialization failed");
    }
    if (!wp_is_cuda_enabled()) {
        throw std::runtime_error("Warp was built without CUDA support");
    }
    if (wp_cuda_device_get_count() <= 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    impl_->context = wp_cuda_device_get_primary_context(0);
    if (!impl_->context) {
        throw std::runtime_error("Failed to get the CUDA primary context for device 0");
    }
    wp_cuda_context_set_current(impl_->context);

    const std::string path = wrp_path.string();
    impl_->graph = wp_apic_load_graph(impl_->context, path.c_str(), APIC_DEVICE_CUDA);
    if (!impl_->graph) {
        throw_warp_error("Failed to load a controller graph from " + path);
    }
}

Controller::~Controller() {
    delete impl_;
    impl_ = nullptr;
}

Controller::Controller(Controller&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

Controller& Controller::operator=(Controller&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

ControllerBuffer Controller::input() const { return buffer_for_prefix(impl_->graph, "input."); }
ControllerBuffer Controller::output() const { return buffer_for_prefix(impl_->graph, "output."); }

void Controller::step(const ControllerBuffer& input, ControllerBuffer& output, float dt) {
    for (const auto& [field, values] : input.fields) {
        set_param(impl_->graph, "input." + field, values);
    }
    set_param(impl_->graph, "dt", std::span<const float>{&dt, 1});

    if (!wp_apic_launch(impl_->graph, nullptr)) {
        throw_warp_error("Controller step failed");
    }
    wp_cuda_context_synchronize(impl_->context);

    for (auto& [field, values] : output.fields) {
        get_param(impl_->graph, "output." + field, values);
    }
}

std::vector<ParamInfo> Controller::params() const {
    const int count = wp_apic_get_num_params(impl_->graph);
    std::vector<ParamInfo> result;
    result.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const char* name = wp_apic_get_param_name(impl_->graph, i);
        if (!name) continue;
        result.push_back(ParamInfo{
            .name = name,
            .size_bytes = wp_apic_get_param_size(impl_->graph, name),
        });
    }
    return result;
}

}  // namespace newton::controllers
