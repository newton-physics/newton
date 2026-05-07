// SPDX-License-Identifier: Apache-2.0
//
// RAII container that owns a paired CPU + GPU memory block, templated
// on the element type so callers can write
//
//     chysx::CudaArray<float>             scalars;
//     chysx::CudaArray<chysx::math::Vec3f> particles;
//     chysx::CudaArray<int>               indices;
//
// without juggling raw byte counts at the call site.
//
// Design goals
// ------------
// * Four core observable members (exposed via getters) match the toy
//   "cpu_ptr / gpu_ptr / cpu_size / gpu_size" specification:
//
//       T*           cpu_ptr_       host buffer  (page-locked / pinned)
//       T*           gpu_ptr_       device buffer
//       std::size_t  cpu_count_     element count on the host
//       std::size_t  gpu_count_     element count on the device
//
//   `cpu_size() / gpu_size()` return *element counts* (std::vector
//   convention).  Use `cpu_bytes() / gpu_bytes()` when you need the
//   byte count, e.g. for `cudaMemcpyAsync`.
//
// * Host memory is allocated with cudaMallocHost so copies between the
//   two sides can be asynchronous and run at full PCIe bandwidth.
//
// * Each side can be sized independently (e.g. allocate only the GPU
//   side for a working buffer), or grown / shrunk on its own.
//
// * No CUDA headers are pulled in here — every CUDA-runtime call lives
//   in `cuda_array.cu` behind the `chysx::detail::` shims.  The template
//   is therefore safe to include from plain C++ translation units.
//
// The class is move-only.  Copying a buffer would silently double the
// memory footprint, which is almost never what physics code wants —
// use the explicit copy_to_host / copy_to_device methods instead.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace chysx {

namespace detail {

// CUDA-runtime shims, implemented in cuda_array.cu.  They throw
// std::runtime_error on failure (allocations / copies) and are noexcept
// for the freeing variants so they can be called from destructors.
void* cuda_alloc_host(std::size_t bytes);
void* cuda_alloc_device(std::size_t bytes);
void cuda_free_host(void* p) noexcept;
void cuda_free_device(void* p) noexcept;
void cuda_copy_h2d(void* dst, const void* src, std::size_t bytes,
                   std::uintptr_t cuda_stream);
void cuda_copy_d2h(void* dst, const void* src, std::size_t bytes,
                   std::uintptr_t cuda_stream);

}  // namespace detail

template <typename T>
class CudaArray {
public:
    using value_type = T;
    using size_type = std::size_t;

    // Empty array (both sides null).  No CUDA calls are made.
    CudaArray() noexcept = default;

    // Allocate `count` elements on both host (pinned) and device.
    // Equivalent to calling `resize(count)` on a default instance.
    explicit CudaArray(size_type count) {
        resize(count);
    }

    ~CudaArray() {
        // Destructors must not throw — swallow any CUDA error here.
        if (cpu_ptr_ != nullptr) {
            detail::cuda_free_host(cpu_ptr_);
        }
        if (gpu_ptr_ != nullptr) {
            detail::cuda_free_device(gpu_ptr_);
        }
    }

    CudaArray(const CudaArray&) = delete;
    CudaArray& operator=(const CudaArray&) = delete;

    CudaArray(CudaArray&& other) noexcept
        : cpu_ptr_(other.cpu_ptr_),
          gpu_ptr_(other.gpu_ptr_),
          cpu_count_(other.cpu_count_),
          gpu_count_(other.gpu_count_) {
        other.cpu_ptr_ = nullptr;
        other.gpu_ptr_ = nullptr;
        other.cpu_count_ = 0;
        other.gpu_count_ = 0;
    }

    CudaArray& operator=(CudaArray&& other) noexcept {
        if (this != &other) {
            clear();
            cpu_ptr_ = other.cpu_ptr_;
            gpu_ptr_ = other.gpu_ptr_;
            cpu_count_ = other.cpu_count_;
            gpu_count_ = other.gpu_count_;
            other.cpu_ptr_ = nullptr;
            other.gpu_ptr_ = nullptr;
            other.cpu_count_ = 0;
            other.gpu_count_ = 0;
        }
        return *this;
    }

    // ---- allocation ----------------------------------------------------
    //
    // Each `allocate_*` call frees the existing buffer on that side
    // before (re)allocating.  When the requested count already matches,
    // the call is a no-op; `count == 0` becomes a free.

    void allocate_host(size_type count) {
        if (count == cpu_count_) {
            return;
        }
        free_host();
        if (count == 0) {
            return;
        }
        cpu_ptr_ = static_cast<T*>(detail::cuda_alloc_host(count * sizeof(T)));
        cpu_count_ = count;
    }

    void allocate_device(size_type count) {
        if (count == gpu_count_) {
            return;
        }
        free_device();
        if (count == 0) {
            return;
        }
        gpu_ptr_ = static_cast<T*>(detail::cuda_alloc_device(count * sizeof(T)));
        gpu_count_ = count;
    }

    // Allocate `count` elements on both sides.
    void resize(size_type count) {
        allocate_host(count);
        allocate_device(count);
    }

    void free_host() noexcept {
        if (cpu_ptr_ != nullptr) {
            detail::cuda_free_host(cpu_ptr_);
            cpu_ptr_ = nullptr;
        }
        cpu_count_ = 0;
    }

    void free_device() noexcept {
        if (gpu_ptr_ != nullptr) {
            detail::cuda_free_device(gpu_ptr_);
            gpu_ptr_ = nullptr;
        }
        gpu_count_ = 0;
    }

    // Free both sides.
    void clear() noexcept {
        free_host();
        free_device();
    }

    // ---- transfers -----------------------------------------------------
    //
    // Both directions require that `cpu_count_ == gpu_count_` and that
    // both sides are non-empty; otherwise they throw.  When
    // `cuda_stream != 0` the copy is issued asynchronously on that
    // stream (legal because host memory is pinned).

    void copy_to_device(std::uintptr_t cuda_stream = 0) {
        check_can_copy("copy_to_device");
        if (cpu_count_ == 0) return;
        detail::cuda_copy_h2d(gpu_ptr_, cpu_ptr_, cpu_count_ * sizeof(T),
                              cuda_stream);
    }

    void copy_to_host(std::uintptr_t cuda_stream = 0) {
        check_can_copy("copy_to_host");
        if (cpu_count_ == 0) return;
        detail::cuda_copy_d2h(cpu_ptr_, gpu_ptr_, gpu_count_ * sizeof(T),
                              cuda_stream);
    }

    // ---- the four "core" observers ------------------------------------

    // Raw addresses cast to int (for FFI / pybind11 / Warp interop).
    std::uintptr_t cpu_ptr() const noexcept {
        return reinterpret_cast<std::uintptr_t>(cpu_ptr_);
    }
    std::uintptr_t gpu_ptr() const noexcept {
        return reinterpret_cast<std::uintptr_t>(gpu_ptr_);
    }
    // Element counts (NOT bytes) — std::vector convention.
    size_type cpu_size() const noexcept { return cpu_count_; }
    size_type gpu_size() const noexcept { return gpu_count_; }

    // ---- typed convenience accessors ----------------------------------

    T* cpu_data() noexcept             { return cpu_ptr_; }
    const T* cpu_data() const noexcept { return cpu_ptr_; }
    T* gpu_data() noexcept             { return gpu_ptr_; }
    const T* gpu_data() const noexcept { return gpu_ptr_; }

    size_type cpu_bytes() const noexcept { return cpu_count_ * sizeof(T); }
    size_type gpu_bytes() const noexcept { return gpu_count_ * sizeof(T); }

    bool empty() const noexcept { return cpu_count_ == 0 && gpu_count_ == 0; }

    // Host-side element access.  Device-side access must go through a
    // CUDA kernel using gpu_data().
    T& operator[](size_type i) noexcept             { return cpu_ptr_[i]; }
    const T& operator[](size_type i) const noexcept { return cpu_ptr_[i]; }

private:
    void check_can_copy(const char* what) const;

    T* cpu_ptr_ = nullptr;
    T* gpu_ptr_ = nullptr;
    size_type cpu_count_ = 0;
    size_type gpu_count_ = 0;
};

namespace detail {

// Centralised mismatch / unallocated diagnostics, defined in
// cuda_array.cu so they don't have to be duplicated for every T.
[[noreturn]] void throw_copy_unallocated(const char* what);
[[noreturn]] void throw_copy_size_mismatch(const char* what,
                                           std::size_t cpu_count,
                                           std::size_t gpu_count);

}  // namespace detail

template <typename T>
void CudaArray<T>::check_can_copy(const char* what) const {
    if (cpu_ptr_ == nullptr || gpu_ptr_ == nullptr) {
        detail::throw_copy_unallocated(what);
    }
    if (cpu_count_ != gpu_count_) {
        detail::throw_copy_size_mismatch(what, cpu_count_, gpu_count_);
    }
}

}  // namespace chysx

// ---- DeviceSpan ↔ CudaArray bridge ---------------------------------------
//
// Defined here (rather than in device_span.h) so the templates have full
// visibility into both CudaArray<T> and DeviceSpan<T>.  Including
// cuda_array.h gives users both pieces.

#include "device_span.h"

namespace chysx {

template <typename T>
DeviceSpan<T> DeviceSpan<T>::from(CudaArray<T>& arr) noexcept {
    return DeviceSpan<T>(arr.gpu_data(), arr.gpu_size());
}

template <typename T>
DeviceSpan<T> DeviceSpan<T>::from(const CudaArray<T>& arr) noexcept {
    // const_cast is sound here: DeviceSpan stores a non-const T*, but a
    // span over a const array's GPU side is still read-only at the
    // language level — kernels that take it as input are expected to
    // treat it that way.  If you need a stricter guarantee, declare a
    // separate `DeviceSpan<const T>` overload.
    return DeviceSpan<T>(const_cast<T*>(arr.gpu_data()), arr.gpu_size());
}

}  // namespace chysx
