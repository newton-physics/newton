// SPDX-License-Identifier: Apache-2.0
//
// CUDA-runtime side of chysx::CudaArray<T>.
//
// The header keeps the templated class inline-only and free of
// `<cuda_runtime.h>`; this translation unit defines the small set of
// non-templated CUDA shims it dispatches through.  Doing it this way
// gives us:
//
//   * Templated, type-aware CudaArray<T> in user code, including
//     instantiations on user-defined POD types (vec3f, vec4i, ...).
//   * Exactly one place where cudaMalloc / cudaMemcpyAsync / friends
//     get called, so error-checking and pinned-memory choices stay
//     consistent across every element type.
//   * Pure C++ headers (vec.cuh, matrix.cuh, ...) can include
//     cuda_array.h without becoming CUDA TUs themselves.

#include "cuda_array.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace chysx {
namespace detail {

namespace {

// Wrap a CUDA call and turn any error into a std::runtime_error with
// the CUDA-provided message.
inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("chysx::CudaArray: ") + what +
                                 " failed: " + cudaGetErrorString(err));
    }
}

}  // namespace

void* cuda_alloc_host(std::size_t bytes) {
    void* p = nullptr;
    // cudaMallocHost gives us pinned (page-locked) memory, which is the
    // prerequisite for asynchronous H2D / D2H copies.
    check_cuda(cudaMallocHost(&p, bytes), "cudaMallocHost");
    return p;
}

void* cuda_alloc_device(std::size_t bytes) {
    void* p = nullptr;
    check_cuda(cudaMalloc(&p, bytes), "cudaMalloc");
    return p;
}

void cuda_free_host(void* p) noexcept {
    // Destructors / free_* paths must not throw — swallow CUDA errors
    // here; any earlier failure has already surfaced through a non-
    // destructor call.
    if (p != nullptr) {
        cudaFreeHost(p);
    }
}

void cuda_free_device(void* p) noexcept {
    if (p != nullptr) {
        cudaFree(p);
    }
}

void cuda_copy_h2d(void* dst, const void* src, std::size_t bytes,
                   std::uintptr_t cuda_stream) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync H2D");
    if (cuda_stream == 0) {
        // Default / null stream: finish synchronously so callers observe
        // normal blocking semantics.
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

void cuda_copy_d2h(void* dst, const void* src, std::size_t bytes,
                   std::uintptr_t cuda_stream) {
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync D2H");
    if (cuda_stream == 0) {
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
}

void throw_copy_unallocated(const char* what) {
    throw std::runtime_error(
        std::string("chysx::CudaArray::") + what +
        ": both host and device buffers must be allocated first");
}

void throw_copy_size_mismatch(const char* what,
                              std::size_t cpu_count,
                              std::size_t gpu_count) {
    throw std::runtime_error(std::string("chysx::CudaArray::") + what +
                             ": size mismatch (cpu_size=" +
                             std::to_string(cpu_count) +
                             ", gpu_size=" + std::to_string(gpu_count) + ")");
}

}  // namespace detail
}  // namespace chysx
