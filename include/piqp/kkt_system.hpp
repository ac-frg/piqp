// This file is part of PIQP.
//
// Copyright (c) 2024 EPFL
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_KKT_SYSTEM_HPP
#define PIQP_KKT_SYSTEM_HPP

#include "piqp/typedefs.hpp"

namespace piqp
{

template<typename T>
class KKTSystem
{
public:
    virtual ~KKTSystem() = default;

    virtual void update_data(int options) = 0;

    virtual bool update_scalings_and_factor(bool iterative_refinement,
                                            const T& rho, const T& delta,
                                            const CVecRef<T>& s, const CVecRef<T>& s_lb, const CVecRef<T>& s_ub,
                                            const CVecRef<T>& z, const CVecRef<T>& z_lb, const CVecRef<T>& z_ub) = 0;

    virtual void multiply(const CVecRef<T>& delta_x, const CVecRef<T>& delta_y,
                          const CVecRef<T>& delta_z, const CVecRef<T>& delta_z_lb, const CVecRef<T>& delta_z_ub,
                          const CVecRef<T>& delta_s, const CVecRef<T>& delta_s_lb, const CVecRef<T>& delta_s_ub,
                          VecRef<T> rhs_x, VecRef<T> rhs_y,
                          VecRef<T> rhs_z, VecRef<T> rhs_z_lb, VecRef<T> rhs_z_ub,
                          VecRef<T> rhs_s, VecRef<T> rhs_s_lb, VecRef<T> rhs_s_ub) = 0;

    virtual void solve(const CVecRef<T>& rhs_x, const CVecRef<T>& rhs_y,
                       const CVecRef<T>& rhs_z, const CVecRef<T>& rhs_z_lb, const CVecRef<T>& rhs_z_ub,
                       const CVecRef<T>& rhs_s, const CVecRef<T>& rhs_s_lb, const CVecRef<T>& rhs_s_ub,
                       VecRef<T> delta_x, VecRef<T> delta_y,
                       VecRef<T> delta_z, VecRef<T> delta_z_lb, VecRef<T> delta_z_ub,
                       VecRef<T> delta_s, VecRef<T> delta_s_lb, VecRef<T> delta_s_ub) = 0;

    // z = alpha * P * x
    virtual void eval_P_x(const T& alpha, const CVecRef<T>& x, VecRef<T> z) = 0;
    // zn = alpha_n * A * xn, zt = alpha_t * A^T * xt
    virtual void eval_A_xn_and_AT_xt(const T& alpha_n, const T& alpha_t, const CVecRef<T>& xn, const CVecRef<T>& xt, VecRef<T> zn, VecRef<T> zt) = 0;
    // zn = alpha_n * G * xn, zt = alpha_t * G^T * xt
    virtual void eval_G_xn_and_GT_xt(const T& alpha_n, const T& alpha_t, const CVecRef<T>& xn, const CVecRef<T>& xt, VecRef<T> zn, VecRef<T> zt) = 0;

    virtual void print_info() {};
};

} // namespace piqp

#endif //PIQP_KKT_SYSTEM_HPP
