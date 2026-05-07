// SPDX-License-Identifier: Apache-2.0
//
// chysx::DeviceSpan<T>
//
// Non-owning, contiguous view over a GPU buffer of `count` elements of
// type T.  Pairs with chysx::CudaArray<T>:
//
//   * `CudaArray<T>` *owns* the memory.  cudaFree fires in its
//     destructor.  Use it for buffers ChysX itself allocated.
//
//   * `DeviceSpan<T>` *references* memory that lives somewhere else
//     (typically a Newton / Warp tensor handed in via `wp.array.ptr`).
//     Its destructor does nothing; the underlying memory keeps living
//     in whoever allocated it.
//
// Putting Newton's device pointers in a `CudaArray<T>` would let
// ChysX's RAII destructor cudaFree() Newton's buffers — a clean
// double-free.  `DeviceSpan<T>` exists specifically so that ownership
// shows up in the type system: anyone reading the code knows this
// pointer must NOT be freed here.
//
// The class is trivially copyable on purpose; multiple kernels and
// helper functions can pass spans by value just like int or float*.

#pragma once

#include <cstddef>
#include <cstdint>

namespace chysx {

template <typename T> class CudaArray;  // fwd decl, defined in cuda_array.h

template <typename T>
class DeviceSpan {
public:
    using value_type = T;
    using size_type = std::size_t;

    DeviceSpan() noexcept = default;

    DeviceSpan(T* ptr, size_type count) noexcept
        : ptr_(ptr), count_(count) {}

    // Build a span from a raw uintptr_t (the FFI-friendly type Python
    // and Warp speak).  Use this at the host-language boundary; once
    // inside C++ keep typed spans wherever possible.
    static DeviceSpan from_raw(std::uintptr_t ptr, size_type count) noexcept {
        return DeviceSpan(reinterpret_cast<T*>(ptr), count);
    }

    // Borrow the device side of a CudaArray — typed and length-checked.
    // Defined out-of-line below to avoid a circular include with
    // cuda_array.h.
    static DeviceSpan from(CudaArray<T>& arr) noexcept;
    static DeviceSpan from(const CudaArray<T>& arr) noexcept;

    // Typed access (use these inside C++/CUDA code).
    T* data() const noexcept { return ptr_; }
    size_type size() const noexcept { return count_; }
    size_type bytes() const noexcept { return count_ * sizeof(T); }
    bool empty() const noexcept { return count_ == 0; }

    // Raw int forms for FFI / pybind11 / printing.
    std::uintptr_t raw() const noexcept {
        return reinterpret_cast<std::uintptr_t>(ptr_);
    }

    // Drop the reference (does NOT free the underlying memory).
    void reset() noexcept {
        ptr_ = nullptr;
        count_ = 0;
    }

private:
    T* ptr_ = nullptr;
    size_type count_ = 0;
};

}  // namespace chysx
