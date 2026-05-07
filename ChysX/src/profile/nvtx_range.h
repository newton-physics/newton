// SPDX-License-Identifier: Apache-2.0
//
// Lightweight RAII NVTX range helper used to label timeline regions
// in Nsight Systems profiles.
//
// NVTX (NVIDIA Tools Extension) is a header-only library shipped with
// the CUDA Toolkit; pushing/popping a "range" creates a coloured bar
// in the Nsight Systems timeline that lines up with whatever GPU
// kernels and CUDA Runtime calls happen while the range is open.
// The cost when no profiler is attached is one read + one branch per
// push/pop (the underlying calls are no-ops), so it's safe to keep
// these in production builds.
//
// Usage:
//
//     {
//         CHYSX_NVTX_RANGE("cloth::step::pcg");
//         pcg_.solve(H_, ...);
//     }
//
// or with a colour (any 0xAARRGGBB ARGB int):
//
//     CHYSX_NVTX_RANGE_COLOUR("cloth::step::hessian", 0xff8e44ad);
//
// Defining `CHYSX_DISABLE_NVTX` strips every range to a no-op for
// builds that can't pull in the NVTX header.

#pragma once

#if !defined(CHYSX_DISABLE_NVTX)
  // The CUDA Toolkit (>= 12) ships an NVTX3 wrapper that re-exports
  // the v2 C API through `nvtx3/nvToolsExt.h`.  No link step needed —
  // every entry point is `static inline` once the header is included.
  #include <nvtx3/nvToolsExt.h>
#endif

namespace chysx {
namespace profile {

#if defined(CHYSX_DISABLE_NVTX)

class NvtxRange {
public:
    explicit NvtxRange(const char* /*name*/) noexcept {}
    NvtxRange(const char* /*name*/, unsigned int /*argb*/) noexcept {}
    ~NvtxRange() = default;

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

#else

// RAII wrapper around `nvtxRangePushEx` / `nvtxRangePop`.  Use the
// `CHYSX_NVTX_RANGE*` macros below rather than instantiating this
// directly, so the chosen colour / category can stay together with
// the source line.
class NvtxRange {
public:
    explicit NvtxRange(const char* name) noexcept {
        nvtxEventAttributes_t a{};
        a.version       = NVTX_VERSION;
        a.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        a.messageType   = NVTX_MESSAGE_TYPE_ASCII;
        a.message.ascii = name;
        nvtxRangePushEx(&a);
    }

    NvtxRange(const char* name, unsigned int argb) noexcept {
        nvtxEventAttributes_t a{};
        a.version       = NVTX_VERSION;
        a.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        a.colorType     = NVTX_COLOR_ARGB;
        a.color         = argb;
        a.messageType   = NVTX_MESSAGE_TYPE_ASCII;
        a.message.ascii = name;
        nvtxRangePushEx(&a);
    }

    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

#endif  // CHYSX_DISABLE_NVTX

}  // namespace profile
}  // namespace chysx

// Macro indirection so the variable name is unique per-source-line —
// otherwise two ranges in the same scope would collide.
#define CHYSX_NVTX_CONCAT_(a, b) a##b
#define CHYSX_NVTX_CONCAT(a, b)  CHYSX_NVTX_CONCAT_(a, b)

#define CHYSX_NVTX_RANGE(name) \
    ::chysx::profile::NvtxRange CHYSX_NVTX_CONCAT(_chysx_nvtx_, __LINE__){name}

#define CHYSX_NVTX_RANGE_COLOUR(name, argb) \
    ::chysx::profile::NvtxRange CHYSX_NVTX_CONCAT(_chysx_nvtx_, __LINE__){name, argb}
