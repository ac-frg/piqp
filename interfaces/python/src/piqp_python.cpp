// This file is part of PIQP.
//
// Copyright (c) 2023 EPFL
// Copyright (c) 2022 INRIA
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

#include "piqp/piqp.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;

#ifndef PIQP_STD_OPTIONAL
namespace nanobind {
namespace detail {

template <typename T>
struct type_caster<tl::optional<T>> {
    using Caster = make_caster<T>;

    NB_TYPE_CASTER(tl::optional<T>, const_name("Optional[") + Caster::Name + const_name("]"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        if (src.is_none()) {
            value = tl::nullopt;
            return true;
        }
        Caster caster;
        if (!caster.from_python(src, flags, cleanup))
            return false;
        value.emplace(caster.operator cast_t<T>());
        return true;
    }

    static handle from_cpp(const tl::optional<T> &src, rv_policy policy, cleanup_list *cleanup) noexcept {
        if (!src.has_value())
            return none().release();
        return Caster::from_cpp(*src, policy, cleanup);
    }
};

template <>
struct type_caster<tl::nullopt_t> {
    NB_TYPE_CASTER(tl::nullopt_t, const_name("None"))

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
        return src.is_none();
    }

    static handle from_cpp(tl::nullopt_t, rv_policy, cleanup_list *) noexcept {
        return none().release();
    }
};

} // namespace detail
} // namespace nanobind
#endif

using T = double;
using I = int;

NB_MODULE(PYTHON_MODULE_NAME, m) {
    nb::enum_<piqp::Status>(m, "Status")
        .value("PIQP_SOLVED", piqp::Status::PIQP_SOLVED)
        .value("PIQP_MAX_ITER_REACHED", piqp::Status::PIQP_MAX_ITER_REACHED)
        .value("PIQP_PRIMAL_INFEASIBLE", piqp::Status::PIQP_PRIMAL_INFEASIBLE)
        .value("PIQP_DUAL_INFEASIBLE", piqp::Status::PIQP_DUAL_INFEASIBLE)
        .value("PIQP_NUMERICS", piqp::Status::PIQP_NUMERICS)
        .value("PIQP_UNSOLVED", piqp::Status::PIQP_UNSOLVED)
        .value("PIQP_INVALID_SETTINGS", piqp::Status::PIQP_INVALID_SETTINGS)
        .export_values();

    nb::class_<piqp::Info<T>>(m, "Info")
        .def(nb::init<>())
        .def_rw("status", &piqp::Info<T>::status)
        .def_rw("iter", &piqp::Info<T>::iter)
        .def_rw("rho", &piqp::Info<T>::rho)
        .def_rw("delta", &piqp::Info<T>::delta)
        .def_rw("mu", &piqp::Info<T>::mu)
        .def_rw("sigma", &piqp::Info<T>::sigma)
        .def_rw("primal_step", &piqp::Info<T>::primal_step)
        .def_rw("dual_step", &piqp::Info<T>::dual_step)
        .def_rw("primal_res", &piqp::Info<T>::primal_res)
        .def_rw("primal_res_rel", &piqp::Info<T>::primal_res_rel)
        .def_rw("dual_res", &piqp::Info<T>::dual_res)
        .def_rw("dual_res_rel", &piqp::Info<T>::dual_res_rel)
        .def_rw("primal_res_reg", &piqp::Info<T>::primal_res_reg)
        .def_rw("primal_res_reg_rel", &piqp::Info<T>::primal_res_reg_rel)
        .def_rw("dual_res_reg", &piqp::Info<T>::dual_res_reg)
        .def_rw("dual_res_reg_rel", &piqp::Info<T>::dual_res_reg_rel)
        .def_rw("primal_prox_inf", &piqp::Info<T>::primal_prox_inf)
        .def_rw("dual_prox_inf", &piqp::Info<T>::dual_prox_inf)
        .def_rw("prev_primal_res", &piqp::Info<T>::prev_primal_res)
        .def_rw("prev_dual_res", &piqp::Info<T>::prev_dual_res)
        .def_rw("primal_obj", &piqp::Info<T>::primal_obj)
        .def_rw("dual_obj", &piqp::Info<T>::dual_obj)
        .def_rw("duality_gap", &piqp::Info<T>::duality_gap)
        .def_rw("duality_gap_rel", &piqp::Info<T>::duality_gap_rel)
        .def_rw("factor_retires", &piqp::Info<T>::factor_retires)
        .def_rw("reg_limit", &piqp::Info<T>::reg_limit)
        .def_rw("no_primal_update", &piqp::Info<T>::no_primal_update)
        .def_rw("no_dual_update", &piqp::Info<T>::no_dual_update)
        .def_rw("setup_time", &piqp::Info<T>::setup_time)
        .def_rw("update_time", &piqp::Info<T>::update_time)
        .def_rw("solve_time", &piqp::Info<T>::solve_time)
        .def_rw("kkt_factor_time", &piqp::Info<T>::kkt_factor_time)
        .def_rw("kkt_solve_time", &piqp::Info<T>::kkt_solve_time)
        .def_rw("run_time", &piqp::Info<T>::run_time);

    nb::class_<piqp::Result<T>>(m, "Result")
        .def_rw("x", &piqp::Result<T>::x)
        .def_rw("y", &piqp::Result<T>::y)
        .def_rw("z_l", &piqp::Result<T>::z_l)
        .def_rw("z_u", &piqp::Result<T>::z_u)
        .def_rw("z_bl", &piqp::Result<T>::z_bl)
        .def_rw("z_bu", &piqp::Result<T>::z_bu)
        .def_rw("s_l", &piqp::Result<T>::s_l)
        .def_rw("s_u", &piqp::Result<T>::s_u)
        .def_rw("s_bl", &piqp::Result<T>::s_bl)
        .def_rw("s_bu", &piqp::Result<T>::s_bu)
        .def_rw("info", &piqp::Result<T>::info);

    nb::enum_<piqp::KKTSolver>(m, "KKTSolver")
            .value("dense_cholesky", piqp::KKTSolver::dense_cholesky)
            .value("sparse_ldlt", piqp::KKTSolver::sparse_ldlt)
            .value("sparse_ldlt_eq_cond", piqp::KKTSolver::sparse_ldlt_eq_cond)
            .value("sparse_ldlt_ineq_cond", piqp::KKTSolver::sparse_ldlt_ineq_cond)
            .value("sparse_ldlt_cond", piqp::KKTSolver::sparse_ldlt_cond)
            .value("sparse_multistage", piqp::KKTSolver::sparse_multistage)
            .export_values();

    nb::class_<piqp::Settings<T>>(m, "Settings")
        .def_rw("rho_init", &piqp::Settings<T>::rho_init)
        .def_rw("delta_init", &piqp::Settings<T>::delta_init)
        .def_rw("eps_abs", &piqp::Settings<T>::eps_abs)
        .def_rw("eps_rel", &piqp::Settings<T>::eps_rel)
        .def_rw("check_duality_gap", &piqp::Settings<T>::check_duality_gap)
        .def_rw("eps_duality_gap_abs", &piqp::Settings<T>::eps_duality_gap_abs)
        .def_rw("eps_duality_gap_rel", &piqp::Settings<T>::eps_duality_gap_rel)
        .def_rw("infeasibility_threshold", &piqp::Settings<T>::infeasibility_threshold)
        .def_rw("reg_lower_limit", &piqp::Settings<T>::reg_lower_limit)
        .def_rw("reg_finetune_lower_limit", &piqp::Settings<T>::reg_finetune_lower_limit)
        .def_rw("reg_finetune_primal_update_threshold", &piqp::Settings<T>::reg_finetune_primal_update_threshold)
        .def_rw("reg_finetune_dual_update_threshold", &piqp::Settings<T>::reg_finetune_dual_update_threshold)
        .def_rw("max_iter", &piqp::Settings<T>::max_iter)
        .def_rw("max_factor_retires", &piqp::Settings<T>::max_factor_retires)
        .def_rw("preconditioner_scale_cost", &piqp::Settings<T>::preconditioner_scale_cost)
        .def_rw("preconditioner_reuse_on_update", &piqp::Settings<T>::preconditioner_reuse_on_update)
        .def_rw("preconditioner_iter", &piqp::Settings<T>::preconditioner_iter)
        .def_rw("tau", &piqp::Settings<T>::tau)
        .def_rw("kkt_solver", &piqp::Settings<T>::kkt_solver)
        .def_rw("iterative_refinement_always_enabled", &piqp::Settings<T>::iterative_refinement_always_enabled)
        .def_rw("iterative_refinement_eps_abs", &piqp::Settings<T>::iterative_refinement_eps_abs)
        .def_rw("iterative_refinement_eps_rel", &piqp::Settings<T>::iterative_refinement_eps_rel)
        .def_rw("iterative_refinement_max_iter", &piqp::Settings<T>::iterative_refinement_max_iter)
        .def_rw("iterative_refinement_min_improvement_rate", &piqp::Settings<T>::iterative_refinement_min_improvement_rate)
        .def_rw("iterative_refinement_static_regularization_eps", &piqp::Settings<T>::iterative_refinement_static_regularization_eps)
        .def_rw("iterative_refinement_static_regularization_rel", &piqp::Settings<T>::iterative_refinement_static_regularization_rel)
        .def_rw("verbose", &piqp::Settings<T>::verbose)
        .def_rw("compute_timings", &piqp::Settings<T>::compute_timings);

    using SparseSolver = piqp::SparseSolver<T, I>;
    nb::class_<SparseSolver>(m, "SparseSolver")
        .def(nb::init<>())
        .def_prop_rw("settings", &SparseSolver::settings, &SparseSolver::settings)
        .def_prop_ro("result", &SparseSolver::result)
        .def("setup",
             [](SparseSolver &solver,
                const piqp::SparseMat<T, I>& P,
                const piqp::CVecRef<T>& c,
                const piqp::optional<piqp::SparseMat<T, I>>& A = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& b = piqp::nullopt,
                const piqp::optional<piqp::SparseMat<T, I>>& G = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& h_l = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& h_u = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& x_l = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& x_u = piqp::nullopt)
             {
                 solver.setup(P, c, A, b, G, h_l, h_u, x_l, x_u);
             },
             nb::arg("P"), nb::arg("c"),
             nb::arg("A") = piqp::nullopt, nb::arg("b") = piqp::nullopt,
             nb::arg("G") = piqp::nullopt, nb::arg("h_l") = piqp::nullopt, nb::arg("h_u") = piqp::nullopt,
             nb::arg("x_l") = piqp::nullopt, nb::arg("x_u") = piqp::nullopt)
        .def("update",
             [](SparseSolver &solver,
                const piqp::optional<piqp::SparseMat<T, I>>& P = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& c = piqp::nullopt,
                const piqp::optional<piqp::SparseMat<T, I>>& A = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& b = piqp::nullopt,
                const piqp::optional<piqp::SparseMat<T, I>>& G = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& h_l = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& h_u = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& x_l = piqp::nullopt,
                const piqp::optional<piqp::CVecRef<T>>& x_u = piqp::nullopt)
             {
                 solver.update(P, c, A, b, G, h_l, h_u, x_l, x_u);
             },
             nb::arg("P") = piqp::nullopt, nb::arg("c") = piqp::nullopt,
             nb::arg("A") = piqp::nullopt, nb::arg("b") = piqp::nullopt,
             nb::arg("G") = piqp::nullopt, nb::arg("h_l") = piqp::nullopt, nb::arg("h_u") = piqp::nullopt,
             nb::arg("x_l") = piqp::nullopt, nb::arg("x_u") = piqp::nullopt)
        .def("solve", &SparseSolver::solve);

    using DenseSolver = piqp::DenseSolver<T>;
    nb::class_<DenseSolver>(m, "DenseSolver")
        .def(nb::init<>())
        .def_prop_rw("settings", &DenseSolver::settings, &DenseSolver::settings)
        .def_prop_ro("result", &DenseSolver::result)
        .def("setup", &DenseSolver::setup,
             nb::arg("P"), nb::arg("c"),
             nb::arg("A") = piqp::nullopt, nb::arg("b") = piqp::nullopt,
             nb::arg("G") = piqp::nullopt, nb::arg("h_l") = piqp::nullopt, nb::arg("h_u") = piqp::nullopt,
             nb::arg("x_l") = piqp::nullopt, nb::arg("x_u") = piqp::nullopt)
        .def("update", &DenseSolver::update,
             nb::arg("P") = piqp::nullopt, nb::arg("c") = piqp::nullopt,
             nb::arg("A") = piqp::nullopt, nb::arg("b") = piqp::nullopt,
             nb::arg("G") = piqp::nullopt, nb::arg("h_l") = piqp::nullopt, nb::arg("h_u") = piqp::nullopt,
             nb::arg("x_l") = piqp::nullopt, nb::arg("x_u") = piqp::nullopt)
        .def("solve", &DenseSolver::solve);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
