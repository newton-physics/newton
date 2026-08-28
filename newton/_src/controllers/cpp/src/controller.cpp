// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#include <newton/controllers/controller.hpp>

#include "dynamic_library.hpp"

#include <apic.h>
#include <warp.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

// warp-clang.so's object-loading API (wp_load_obj / wp_lookup) resolves a CPU
// graph's compiled kernels at replay time, mirroring what Warp's own Python
// capture_load() does via ctypes. Warp does not ship a header for these; the
// signatures are copied from warp/native/clang/clang.cpp.
extern "C" {
int wp_load_obj(const char* object_file, const char* module_name, bool use_legacy_linker);
uint64_t wp_lookup(const char* dll_name, const char* function_name);
}

// NEWTON_WARP_LIBRARY / NEWTON_WARP_CLANG_LIBRARY are absolute paths baked in
// by CMake (see FindWarp.cmake), pointing at the uv-installed warp-lang
// package's runtime and LLVM-JIT libraries.
//
// Both are loaded dynamically (see dynamic_library.hpp) rather than linked at
// build time. This isn't a Windows workaround: Warp's own C++ consumers do the
// same on every platform (see github.com/erwincoumans/warp_cpp), because the
// pip wheel simply does not ship an import library (.lib) at all -- only the
// .dll/.so/.dylib. A .lib is only needed for implicit (link-time) linking;
// GetProcAddress/dlsym resolve symbols straight out of the loaded module's own
// export table, so no .lib is needed for this approach on any platform.
#ifndef NEWTON_WARP_LIBRARY
#error "NEWTON_WARP_LIBRARY must be defined by CMake (see FindWarp.cmake)"
#endif
#ifndef NEWTON_WARP_CLANG_LIBRARY
#error "NEWTON_WARP_CLANG_LIBRARY must be defined by CMake (see FindWarp.cmake)"
#endif

namespace newton::controllers {
namespace {

using detail::dynamic_library_last_error;
using detail::dynamic_library_open;
using detail::dynamic_library_resolve;

// Every Warp entry point this runtime calls, resolved once and cached for the
// life of the process. Each member is a function pointer typed by
// decltype(&::name): the real function's signature, copied from its
// declaration in warp.h/apic.h, rather than retyped by hand. That means a
// mismatch (wrong argument type, wrong order) fails to *compile* the call
// site below, instead of compiling silently and only breaking at runtime --
// resolve()'s reinterpret_cast trusts whatever type it's given, so a
// hand-written function pointer type would give it nothing to check against.
// The leading `::` forces each name to resolve to the free function in
// warp.h/apic.h, not the struct member of the same name being declared on
// that same line.
struct WarpApi {
    decltype(&::wp_init) wp_init = nullptr;
    decltype(&::wp_get_error_string) wp_get_error_string = nullptr;
    decltype(&::wp_is_cuda_enabled) wp_is_cuda_enabled = nullptr;
    decltype(&::wp_cuda_device_get_count) wp_cuda_device_get_count = nullptr;
    decltype(&::wp_cuda_device_get_primary_context) wp_cuda_device_get_primary_context = nullptr;
    decltype(&::wp_cuda_context_set_current) wp_cuda_context_set_current = nullptr;
    decltype(&::wp_cuda_context_synchronize) wp_cuda_context_synchronize = nullptr;
    decltype(&::wp_apic_load_graph) wp_apic_load_graph = nullptr;
    decltype(&::wp_apic_destroy_graph) wp_apic_destroy_graph = nullptr;
    decltype(&::wp_apic_set_param) wp_apic_set_param = nullptr;
    decltype(&::wp_apic_get_param) wp_apic_get_param = nullptr;
    decltype(&::wp_apic_launch) wp_apic_launch = nullptr;
    decltype(&::wp_apic_cpu_replay_graph) wp_apic_cpu_replay_graph = nullptr;
    decltype(&::wp_apic_get_num_params) wp_apic_get_num_params = nullptr;
    decltype(&::wp_apic_get_param_name) wp_apic_get_param_name = nullptr;
    decltype(&::wp_apic_get_param_size) wp_apic_get_param_size = nullptr;
    decltype(&::wp_apic_get_num_kernels) wp_apic_get_num_kernels = nullptr;
    decltype(&::wp_apic_get_kernel_key) wp_apic_get_kernel_key = nullptr;
    decltype(&::wp_apic_get_kernel_module_hash) wp_apic_get_kernel_module_hash = nullptr;
    decltype(&::wp_apic_get_kernel_module_binary_filename) wp_apic_get_kernel_module_binary_filename = nullptr;
    decltype(&::wp_apic_get_kernel_forward_name) wp_apic_get_kernel_forward_name = nullptr;
    decltype(&::wp_apic_get_kernel_backward_name) wp_apic_get_kernel_backward_name = nullptr;
    decltype(&::wp_apic_register_loaded_cpu_kernel) wp_apic_register_loaded_cpu_kernel = nullptr;
};

struct WarpClangApi {
    decltype(&::wp_load_obj) wp_load_obj = nullptr;
    decltype(&::wp_lookup) wp_lookup = nullptr;
};

// WarpApi/WarpClangApi start as all-nullptr slots (see above); this is what
// fills one in with the address dynamic_library_open() actually loaded, by
// looking up its name in the library's export table. Throws immediately
// rather than leaving a null slot behind, since calling through a null
// function pointer later would crash with no indication of which symbol was
// missing.
template <typename Fn>
void resolve(void* handle, const char* library, const char* name, Fn& out) {
    out = reinterpret_cast<Fn>(dynamic_library_resolve(handle, name));
    if (!out) {
        throw std::runtime_error(
            "Failed to resolve symbol '" + std::string(name) + "' in " + library + ": " + dynamic_library_last_error()
        );
    }
}

const WarpApi& warp_api() {
    static const WarpApi api = [] {
        void* handle = dynamic_library_open(NEWTON_WARP_LIBRARY);
        if (!handle) {
            throw std::runtime_error(
                "Failed to load " NEWTON_WARP_LIBRARY ": " + dynamic_library_last_error()
            );
        }
        WarpApi result;
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_init", result.wp_init);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_get_error_string", result.wp_get_error_string);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_is_cuda_enabled", result.wp_is_cuda_enabled);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_cuda_device_get_count", result.wp_cuda_device_get_count);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_cuda_device_get_primary_context",
                result.wp_cuda_device_get_primary_context);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_cuda_context_set_current", result.wp_cuda_context_set_current);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_cuda_context_synchronize", result.wp_cuda_context_synchronize);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_load_graph", result.wp_apic_load_graph);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_destroy_graph", result.wp_apic_destroy_graph);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_set_param", result.wp_apic_set_param);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_get_param", result.wp_apic_get_param);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_launch", result.wp_apic_launch);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_cpu_replay_graph", result.wp_apic_cpu_replay_graph);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_get_num_params", result.wp_apic_get_num_params);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_get_param_name", result.wp_apic_get_param_name);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_get_param_size", result.wp_apic_get_param_size);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_get_num_kernels", result.wp_apic_get_num_kernels);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_get_kernel_key", result.wp_apic_get_kernel_key);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_get_kernel_module_hash", result.wp_apic_get_kernel_module_hash);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_get_kernel_module_binary_filename",
                result.wp_apic_get_kernel_module_binary_filename);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_get_kernel_forward_name",
                result.wp_apic_get_kernel_forward_name);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_get_kernel_backward_name",
                result.wp_apic_get_kernel_backward_name);
        resolve(handle, NEWTON_WARP_LIBRARY, "wp_apic_register_loaded_cpu_kernel",
                result.wp_apic_register_loaded_cpu_kernel);
        return result;
    }();
    return api;
}

// Loaded lazily -- only a CPU-captured graph needs Warp's LLVM-JIT backend to
// resolve its compiled kernels; a CUDA-captured graph never touches it.
const WarpClangApi& warp_clang_api() {
    static const WarpClangApi api = [] {
        void* handle = dynamic_library_open(NEWTON_WARP_CLANG_LIBRARY);
        if (!handle) {
            throw std::runtime_error(
                "Failed to load " NEWTON_WARP_CLANG_LIBRARY ": " + dynamic_library_last_error()
            );
        }
        WarpClangApi result;
        resolve(handle, NEWTON_WARP_CLANG_LIBRARY, "wp_load_obj", result.wp_load_obj);
        resolve(handle, NEWTON_WARP_CLANG_LIBRARY, "wp_lookup", result.wp_lookup);
        return result;
    }();
    return api;
}

[[noreturn]] void throw_warp_error(const std::string& what) {
    const char* detail = warp_api().wp_get_error_string();
    if (detail && detail[0] != '\0') {
        throw std::runtime_error(what + ": " + detail);
    }
    throw std::runtime_error(what);
}

std::string warp_detail(const std::string& what) {
    const char* detail = warp_api().wp_get_error_string();
    return (detail && detail[0] != '\0') ? what + ": " + detail : what;
}

// Reports rather than throws: step() is required not to throw, so every failure
// on the launch path becomes a message plus a false return.
bool check_size(APICGraph* graph, const std::string& name, std::size_t got_bytes, std::string& error) {
    const std::size_t expected = warp_api().wp_apic_get_param_size(graph, name.c_str());
    if (expected == 0) {
        error = "'" + name + "' is not a parameter of this graph";
        return false;
    }
    if (got_bytes != expected) {
        error = "Parameter '" + name + "' takes " + std::to_string(expected) + " bytes, got "
                + std::to_string(got_bytes);
        return false;
    }
    return true;
}

bool set_param(APICGraph* graph, const std::string& name, std::span<const float> data, std::string& error) {
    if (!check_size(graph, name, data.size_bytes(), error)) {
        return false;
    }
    if (!warp_api().wp_apic_set_param(graph, name.c_str(), data.data(), data.size_bytes())) {
        error = warp_detail("Failed to set parameter '" + name + "'");
        return false;
    }
    return true;
}

bool get_param(APICGraph* graph, const std::string& name, std::span<float> data, std::string& error) {
    if (!check_size(graph, name, data.size_bytes(), error)) {
        return false;
    }
    if (!warp_api().wp_apic_get_param(graph, name.c_str(), data.data(), data.size_bytes())) {
        error = warp_detail("Failed to get parameter '" + name + "'");
        return false;
    }
    return true;
}

// Reads the device a .wrp graph was captured for directly out of its 64-byte
// file header, without touching Warp at all. wp_apic_load_graph() does not
// itself read or validate this field (it trusts the device_type its caller
// passes in), so this is what makes device detection possible: peek the file
// before deciding how to load it.
APICDeviceType read_device_type(const std::filesystem::path& wrp_path) {
    std::ifstream file(wrp_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open " + wrp_path.string());
    }
    APICFileHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file || file.gcount() != static_cast<std::streamsize>(sizeof(header))) {
        throw std::runtime_error(wrp_path.string() + " is too short to be a .wrp file");
    }
    if (std::memcmp(header.magic, "WRP1", sizeof(header.magic)) != 0) {
        throw std::runtime_error(wrp_path.string() + " is not a .wrp file (bad magic)");
    }
    return static_cast<APICDeviceType>(header.device_type);
}

// Loads every compiled kernel a CPU-captured graph references, mirroring
// Warp's own Python capture_load(): wp_load_obj() JITs each <path>_modules/*.o
// into an in-memory module, then wp_lookup() resolves each kernel's forward /
// backward symbol so wp_apic_register_loaded_cpu_kernel() can wire it into the
// graph before replay.
void load_cpu_kernels(const std::filesystem::path& wrp_path, APICGraph* graph) {
    std::filesystem::path modules_dir = wrp_path;
    modules_dir.replace_extension();
    modules_dir += "_modules";

    const WarpClangApi& clang = warp_clang_api();
    const WarpApi& warp = warp_api();

    // Keyed by the module's binary filename (e.g. "wp_newton..._1ee0c42.cpu....o"),
    // matching what wp_apic_get_kernel_module_binary_filename() below reports for
    // each kernel -- mirrors Warp's own Python capture_load(), which keys the same
    // way for the same reason (see below).
    std::unordered_map<std::string, std::string> loaded_handles;
    if (std::filesystem::is_directory(modules_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(modules_dir)) {
            if (entry.path().extension() != ".o") continue;
            const std::string filename = entry.path().filename().string();
            const std::string handle = "wp_apic_" + entry.path().stem().string();
            if (clang.wp_load_obj(entry.path().string().c_str(), handle.c_str(), false) != 0) {
                throw std::runtime_error("Failed to load CPU module " + entry.path().string());
            }
            loaded_handles.emplace(filename, handle);
        }
    }

    const int num_kernels = warp.wp_apic_get_num_kernels(graph);
    if (loaded_handles.empty() && num_kernels > 0) {
        throw std::runtime_error(
            "No CPU modules found in " + modules_dir.string() + ", but the graph contains "
            + std::to_string(num_kernels) + " kernel(s). Ensure the .wrp file and its _modules directory are intact."
        );
    }

    for (int i = 0; i < num_kernels; ++i) {
        const char* kernel_key = warp.wp_apic_get_kernel_key(graph, i);
        const char* forward_name = warp.wp_apic_get_kernel_forward_name(graph, i);
        if (!kernel_key || !forward_name) continue;
        const char* module_hash = warp.wp_apic_get_kernel_module_hash(graph, i);
        const char* module_binary_filename = warp.wp_apic_get_kernel_module_binary_filename(graph, i);
        const char* backward_name = warp.wp_apic_get_kernel_backward_name(graph, i);

        // Resolve strictly from the module this kernel was recorded in. Two
        // modules can legitimately export a symbol of the same name (a shared
        // low-level kernel compiled into more than one Warp module), so
        // scanning every loaded handle and taking the first match risks
        // resolving the wrong module's copy. A .wrp captured by this same
        // Warp install always records this correctly, so no fallback: if the
        // symbol isn't found here, something is wrong with the artifact and
        // that should surface as an error, not a guess.
        uint64_t forward_fn = 0;
        uint64_t backward_fn = 0;
        if (module_binary_filename) {
            const auto found = loaded_handles.find(module_binary_filename);
            if (found != loaded_handles.end()) {
                forward_fn = clang.wp_lookup(found->second.c_str(), forward_name);
                if (forward_fn && backward_name) backward_fn = clang.wp_lookup(found->second.c_str(), backward_name);
            }
        }
        if (!forward_fn) {
            throw std::runtime_error(std::string("Failed to resolve CPU kernel ") + kernel_key + " (" + forward_name + ")");
        }
        warp.wp_apic_register_loaded_cpu_kernel(graph, kernel_key, module_hash, reinterpret_cast<void*>(forward_fn),
                                                 backward_fn ? reinterpret_cast<void*>(backward_fn) : nullptr);
    }
}

ControllerBuffer buffer_for_prefix(APICGraph* graph, std::string_view prefix) {
    const WarpApi& warp = warp_api();
    const int count = warp.wp_apic_get_num_params(graph);
    ControllerBuffer buffer;
    for (int i = 0; i < count; ++i) {
        const char* name = warp.wp_apic_get_param_name(graph, i);
        if (!name) continue;
        const std::string_view full{name};
        if (full.size() > prefix.size() && full.substr(0, prefix.size()) == prefix) {
            const std::string field{full.substr(prefix.size())};
            buffer.fields[field].assign(warp.wp_apic_get_param_size(graph, name) / sizeof(float), 0.0f);
        }
    }
    return buffer;
}

}  // namespace

struct Controller::Impl {
    APICDeviceType device_type{APIC_DEVICE_CUDA};
    void* context{nullptr};
    APICGraph* graph{nullptr};
    std::string last_error;

    ~Impl() {
        if (graph) {
            warp_api().wp_apic_destroy_graph(graph);
            graph = nullptr;
        }
    }
};

Controller::Controller(const std::filesystem::path& wrp_path) : impl_(new Impl()) {
    const WarpApi& warp = warp_api();

    if (warp.wp_init(nullptr) != 0) {
        throw_warp_error("Warp runtime initialization failed");
    }

    impl_->device_type = read_device_type(wrp_path);
    const std::string path = wrp_path.string();

    if (impl_->device_type == APIC_DEVICE_CUDA) {
        if (!warp.wp_is_cuda_enabled()) {
            throw std::runtime_error("Warp was built without CUDA support");
        }
        if (warp.wp_cuda_device_get_count() <= 0) {
            throw std::runtime_error("No CUDA devices available");
        }

        impl_->context = warp.wp_cuda_device_get_primary_context(0);
        if (!impl_->context) {
            throw std::runtime_error("Failed to get the CUDA primary context for device 0");
        }
        warp.wp_cuda_context_set_current(impl_->context);

        impl_->graph = warp.wp_apic_load_graph(impl_->context, path.c_str(), APIC_DEVICE_CUDA);
        if (!impl_->graph) {
            throw_warp_error("Failed to load a controller graph from " + path);
        }
    } else {
        impl_->graph = warp.wp_apic_load_graph(nullptr, path.c_str(), APIC_DEVICE_CPU);
        if (!impl_->graph) {
            throw_warp_error("Failed to load a controller graph from " + path);
        }
        load_cpu_kernels(wrp_path, impl_->graph);
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

bool Controller::step(const ControllerBuffer& input, ControllerBuffer& output, float dt) noexcept {
    impl_->last_error.clear();

    for (const auto& [field, values] : input.fields) {
        if (!set_param(impl_->graph, "input." + field, values, impl_->last_error)) {
            return false;
        }
    }
    if (!set_param(impl_->graph, "dt", std::span<const float>{&dt, 1}, impl_->last_error)) {
        return false;
    }

    const WarpApi& warp = warp_api();
    if (impl_->device_type == APIC_DEVICE_CUDA) {
        if (!warp.wp_apic_launch(impl_->graph, nullptr)) {
            impl_->last_error = warp_detail("Launch failed");
            return false;
        }
        warp.wp_cuda_context_synchronize(impl_->context);
    } else {
        if (!warp.wp_apic_cpu_replay_graph(impl_->graph)) {
            impl_->last_error = warp_detail("CPU replay failed");
            return false;
        }
    }

    for (auto& [field, values] : output.fields) {
        if (!get_param(impl_->graph, "output." + field, values, impl_->last_error)) {
            return false;
        }
    }
    return true;
}

const std::string& Controller::last_error() const noexcept {
    return impl_->last_error;
}

std::vector<ParamInfo> Controller::params() const {
    const WarpApi& warp = warp_api();
    const int count = warp.wp_apic_get_num_params(impl_->graph);
    std::vector<ParamInfo> result;
    result.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const char* name = warp.wp_apic_get_param_name(impl_->graph, i);
        if (!name) continue;
        result.push_back(ParamInfo{
            .name = name,
            .size_bytes = warp.wp_apic_get_param_size(impl_->graph, name),
        });
    }
    return result;
}

}  // namespace newton::controllers
