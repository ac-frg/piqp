// This file is part of PIQP.
//
// Copyright (c) 2026 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_UTILS_SMOOTHING_HPP
#define PIQP_UTILS_SMOOTHING_HPP

#include <cmath>

namespace piqp
{

// Smoothing operator for the nonnegative orthant.
// Given input c and barrier parameter mu > 0, computes
//   s = (c + sqrt(c^2 + 4*mu)) / 2
// which satisfies s > 0 and s * (s - c) = mu.
// This places the pair (s, z) with z = s - c exactly
// on the central path with s * z = mu.
//
// Reference: Chen, Goulart, Jones,
// "A warmstarting technique for general conic optimization
//  in interior point methods", 2025.
template<typename T>
void nonneg_smoothing(T c, T mu, T& s, T& z)
{
    s = (c + std::sqrt(c * c + T(4) * mu)) / T(2);
    z = s - c;
}

} // namespace piqp

#endif // PIQP_UTILS_SMOOTHING_HPP
