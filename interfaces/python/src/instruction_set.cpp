// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <nanobind/nanobind.h>

#if defined(CPU_FEATURES_AVAILABLE)
#include "cpu_features_macros.h"
#endif

#if defined(CPU_FEATURES_ARCH_X86)
#include "cpuinfo_x86.h"
#endif

namespace nb = nanobind;
#if defined(CPU_FEATURES_AVAILABLE)
using namespace cpu_features;
#endif

#if defined(CPU_FEATURES_ARCH_X86)
const X86Info info = GetX86Info();
#endif

NB_MODULE(instruction_set, m) {
#if defined(CPU_FEATURES_ARCH_X86)
    m.attr("avx2") = (bool) info.features.avx2;
    m.attr("avx512f") = (bool) info.features.avx512f;
#else
    m.attr("avx2") = false;
    m.attr("avx512f") = false;
#endif
}
