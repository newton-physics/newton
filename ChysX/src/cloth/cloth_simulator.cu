// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::cloth::ClothSimulator.
//
// For now we ship a single semi-implicit Euler kernel that integrates
// gravity (and optional velocity damping) on every particle.  All real
// cloth physics — Lamé elasticity, dihedral bending, contact — will
// land here over time, behind the same `step()` entry point.

#include "cloth_simulator.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <stdexcept>
#include <string>

namespace chysx {
namespace cloth {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("chysx::cloth: ") + what +
                                 " failed: " + cudaGetErrorString(err));
    }
}

// Free-fall semi-implicit Euler step:
//   v <- (v + g * dt) * exp(-damping * dt)
//   x <- x + v * dt
//
// `damping` may be zero, in which case the exp() branch is skipped to
// keep the inner loop minimal.  Single-precision throughout to match
// Newton's particle buffers.
__global__ void freefall_step_kernel(float3* __restrict__ pos,
                                     float3* __restrict__ vel,
                                     int n,
                                     float3 g,
                                     float damping,
                                     float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float3 v = vel[i];
    v.x += g.x * dt;
    v.y += g.y * dt;
    v.z += g.z * dt;

    if (damping > 0.0f) {
        const float decay = expf(-damping * dt);
        v.x *= decay;
        v.y *= decay;
        v.z *= decay;
    }

    float3 p = pos[i];
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;

    pos[i] = p;
    vel[i] = v;
}

}  // namespace

void ClothSimulator::set_external_buffers(std::uintptr_t pos_ptr,
                                          std::uintptr_t vel_ptr,
                                          int particle_count,
                                          std::uintptr_t inv_mass_ptr) noexcept {
    // The Python / Warp side speaks raw uintptr_t.  Wrap into typed,
    // length-aware spans here so the rest of the C++ code can stay
    // strongly-typed.
    const auto n = static_cast<std::size_t>(particle_count);
    buffers_.pos = DeviceSpan<math::Vec3f>::from_raw(pos_ptr, n);
    buffers_.vel = DeviceSpan<math::Vec3f>::from_raw(vel_ptr, n);
    buffers_.inv_mass = DeviceSpan<float>::from_raw(inv_mass_ptr, n);
}

void ClothSimulator::step(float dt, std::uintptr_t cuda_stream) {
    const int n = buffers_.particle_count();
    if (n <= 0) {
        return;
    }
    if (buffers_.pos.data() == nullptr || buffers_.vel.data() == nullptr) {
        throw std::runtime_error(
            "chysx::cloth::ClothSimulator::step: pos / vel spans must be "
            "set via set_external_buffers() before stepping.");
    }

    // Vec3f and float3 share the same {x, y, z} layout (12 bytes,
    // 4-byte aligned), so reinterpreting between them is sound.
    auto* pos = reinterpret_cast<float3*>(buffers_.pos.data());
    auto* vel = reinterpret_cast<float3*>(buffers_.vel.data());
    const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    constexpr int block = 256;
    const int grid = (n + block - 1) / block;

    const float3 g = make_float3(material_.gx, material_.gy, material_.gz);

    freefall_step_kernel<<<grid, block, 0, stream>>>(
        pos, vel, n, g, material_.damping, dt);

    check_cuda(cudaGetLastError(), "kernel launch");
}

}  // namespace cloth
}  // namespace chysx
