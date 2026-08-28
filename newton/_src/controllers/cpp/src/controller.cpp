// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#include <newton/controllers/controller.hpp>

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

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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
// Both are loaded dynamically (dlopen/dlsym, or LoadLibrary/GetProcAddress on
// Windows) rather than linked at build time. This isn't a Windows workaround:
// Warp's own C++ consumers do the same on every platform (see
// github.com/erwincoumans/warp_cpp), because the pip wheel simply does not
// ship an import library (.lib) at all -- only the .dll/.so/.dylib. A .lib is
// only needed for implicit (link-time) linking; GetProcAddress/dlsym resolve
// symbols straight out of the loaded module's own export table, so no .lib is
// needed for this approach on any platform.
#ifndef NEWTON_WARP_LIBRARY
#error "NEWTON_WARP_LIBRARY must be defined by CMake (see FindWarp.cmake)"
#endif
#ifndef NEWTON_WARP_CLANG_LIBRARY
#error "NEWTON_WARP_CLANG_LIBRARY must be defined by CMake (see FindWarp.cmake)"
#endif

namespace newton::controllers {
namespace {

void* load_library(const char* path) {
#if defined(_WIN32)
    return static_cast<void*>(LoadLibraryA(path));
#else
    return dlopen(path, RTLD_NOW);
#endif
}

void* resolve_symbol(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

// dlerror()/GetLastError() are the only error sources available before any
// Warp symbol (including wp_get_error_string) has been resolved.
std::string platform_error_string() {
#if defined(_WIN32)
    const DWORD code = GetLastError();
    char* message = nullptr;
    FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
        0, reinterpret_cast<char*>(&message), 0, nullptr
    );
    std::string result = message ? message : ("error code " + std::to_string(code));
    if (message) LocalFree(message);
    return result;
#else
    const char* detail = dlerror();
    return detail ? detail : "unknown error";
#endif
}

// Every Warp entry point this runtime calls, resolved once and cached for the
// life of the process. One line per symbol in the X-macro list below adds it
// to the struct, the loader, and its typedef together.
#define NEWTON_WARP_RUNTIME_SYMBOLS(X) \
    X(wp_init) \
    X(wp_get_error_string) \
    X(wp_is_cuda_enabled) \
    X(wp_cuda_device_get_count) \
    X(wp_cuda_device_get_primary_context) \
    X(wp_cuda_context_set_current) \
    X(wp_cuda_context_synchronize) \
    X(wp_apic_load_graph) \
    X(wp_apic_destroy_graph) \
    X(wp_apic_set_param) \
    X(wp_apic_get_param) \
    X(wp_apic_launch) \
    X(wp_apic_cpu_replay_graph) \
    X(wp_apic_get_num_params) \
    X(wp_apic_get_param_name) \
    X(wp_apic_get_param_size) \
    X(wp_apic_get_num_kernels) \
    X(wp_apic_get_kernel_key) \
    X(wp_apic_get_kernel_module_hash) \
    X(wp_apic_get_kernel_forward_name) \
    X(wp_apic_get_kernel_backward_name) \
    X(wp_apic_register_loaded_cpu_kernel)

#define NEWTON_WARP_CLANG_SYMBOLS(X) \
    X(wp_load_obj) \
    X(wp_lookup)

#define NEWTON_DECLARE_FN_PTR(name) decltype(&::name) name = nullptr;
struct WarpApi {
    NEWTON_WARP_RUNTIME_SYMBOLS(NEWTON_DECLARE_FN_PTR)
};
struct WarpClangApi {
    NEWTON_WARP_CLANG_SYMBOLS(NEWTON_DECLARE_FN_PTR)
};
#undef NEWTON_DECLARE_FN_PTR

const WarpApi& warp_api() {
    static const WarpApi api = [] {
        void* handle = load_library(NEWTON_WARP_LIBRARY);
        if (!handle) {
            throw std::runtime_error(
                "Failed to load " NEWTON_WARP_LIBRARY ": " + platform_error_string()
            );
        }
        WarpApi result;
#define NEWTON_RESOLVE(name) \
        result.name = reinterpret_cast<decltype(&::name)>(resolve_symbol(handle, #name)); \
        if (!result.name) { \
            throw std::runtime_error("Failed to resolve symbol '" #name "' in " NEWTON_WARP_LIBRARY); \
        }
        NEWTON_WARP_RUNTIME_SYMBOLS(NEWTON_RESOLVE)
#undef NEWTON_RESOLVE
        return result;
    }();
    return api;
}

// Loaded lazily -- only a CPU-captured graph needs Warp's LLVM-JIT backend to
// resolve its compiled kernels; a CUDA-captured graph never touches it.
const WarpClangApi& warp_clang_api() {
    static const WarpClangApi api = [] {
        void* handle = load_library(NEWTON_WARP_CLANG_LIBRARY);
        if (!handle) {
            throw std::runtime_error(
                "Failed to load " NEWTON_WARP_CLANG_LIBRARY ": " + platform_error_string()
            );
        }
        WarpClangApi result;
#define NEWTON_RESOLVE(name) \
        result.name = reinterpret_cast<decltype(&::name)>(resolve_symbol(handle, #name)); \
        if (!result.name) { \
            throw std::runtime_error("Failed to resolve symbol '" #name "' in " NEWTON_WARP_CLANG_LIBRARY); \
        }
        NEWTON_WARP_CLANG_SYMBOLS(NEWTON_RESOLVE)
#undef NEWTON_RESOLVE
        return result;
    }();
    return api;
}

#undef NEWTON_WARP_RUNTIME_SYMBOLS
#undef NEWTON_WARP_CLANG_SYMBOLS

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

    std::vector<std::string> loaded_handles;
    if (std::filesystem::is_directory(modules_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(modules_dir)) {
            if (entry.path().extension() != ".o") continue;
            const std::string handle = "wp_apic_" + entry.path().stem().string();
            if (clang.wp_load_obj(entry.path().string().c_str(), handle.c_str(), false) != 0) {
                throw std::runtime_error("Failed to load CPU module " + entry.path().string());
            }
            loaded_handles.push_back(handle);
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
        const char* backward_name = warp.wp_apic_get_kernel_backward_name(graph, i);

        uint64_t forward_fn = 0;
        uint64_t backward_fn = 0;
        for (const auto& handle : loaded_handles) {
            forward_fn = clang.wp_lookup(handle.c_str(), forward_name);
            if (forward_fn) {
                if (backward_name) backward_fn = clang.wp_lookup(handle.c_str(), backward_name);
                break;
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
