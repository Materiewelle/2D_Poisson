#ifndef CURRENT_HPP
#define CURRENT_HPP

#include <armadillo>

#include "device.hpp"
#include "potential.hpp"

class current {
public:
    arma::vec lv;
    arma::vec rv;
    arma::vec lc;
    arma::vec rc;
    arma::vec lt;
    arma::vec rt;
    arma::vec total;

    inline current();
    inline current(const device & d, const potential & phi);
    inline current(const device & d, const wave_packet psi[6], const potential & phi0, const potential & phi);
};

//----------------------------------------------------------------------------------------------------------------------

current::current() {
}

current::current(const device & d, const potential & phi)
    : lv(d.N_x), rv(d.N_x), lc(d.N_x), rc(d.N_x), lt(d.N_x), rt(d.N_x) {
    using namespace arma;

    // transmission probability
    auto transmission = [&] (double E) -> double {
        cx_double Sigma_s, Sigma_d;
        cx_vec G = green_col<false>(d, phi, E, Sigma_s, Sigma_d);
        return 4 * Sigma_s.imag() * Sigma_d.imag() * (std::norm(G(1)) + std::norm(G(2)));
    };

    static constexpr auto scale = 2.0 * c::e * c::e / c::h;

    vec E_lv, E_rv, E_lc, E_rc, E_lt, E_rt;
    vec W_lv, W_rv, W_lc, W_rc, W_lt, W_rt;

    auto i_lv = linspace(phi.s() + charge_density_impl::E_min, phi.s() - 0.5 * d.E_gc, 50);
    auto i_rv = linspace(phi.d() + charge_density_impl::E_min, phi.d() - 0.5 * d.E_gc, 50);
    auto i_lc = linspace(phi.s() + 0.5 * d.E_gc, phi.s() + charge_density_impl::E_max, 50);
    auto i_rc = linspace(phi.d() + 0.5 * d.E_gc, phi.d() + charge_density_impl::E_max, 50);

    lv(0) = integral([&] (double E) {
        return scale * transmission(E) * (fermi(E - phi.s(), d.F_sc) - 1.0);
    }, 1, i_lv, charge_density_impl::rel_tol, c::epsilon(1e-10), E_lv, W_lv)(0);
    lv.fill(lv(0));

    rv(0) = integral([&] (double E) {
        return - scale * transmission(E) * (fermi(E - phi.d(), d.F_dc) - 1.0);
    }, 1, i_rv, charge_density_impl::rel_tol, c::epsilon(1e-10), E_rv, W_rv)(0);
    rv.fill(rv(0));

    lc(0) = integral([&] (double E) {
        return scale * transmission(E) * fermi(E - phi.s(), d.F_sc);
    }, 1, i_lc, charge_density_impl::rel_tol, c::epsilon(1e-10), E_lc, W_lc)(0);
    lc.fill(lc(0));

    rc(0) = integral([&] (double E) {
        return - scale * transmission(E) * fermi(E - phi.d(), d.F_dc);
    }, 1, i_rc, charge_density_impl::rel_tol, c::epsilon(1e-10), E_rc, W_rc)(0);
    rc.fill(rc(0));

    if (phi.s() > phi.d() + d.E_gc) {
        auto i_lt = linspace(phi.d() + 0.5 * d.E_gc, phi.s() - 0.5 * d.E_gc, 100);

        lt(0) = integral([&] (double E) {
            return scale * transmission(E);
        }, 1, i_lt, charge_density_impl::rel_tol, c::epsilon(1e-10), E_lt, W_lt)(0);
        lt.fill(lt(0));
        rt.fill(0.0);
    } else if (phi.d() > phi.s() + d.E_gc) {
        auto i_rt = linspace(phi.s() + 0.5 * d.E_gc, phi.d() - 0.5 * d.E_gc, 100);

        rt(0) = integral([&] (double E) {
            return - scale * transmission(E);
        }, 1, i_rt, charge_density_impl::rel_tol, c::epsilon(1e-10), E_rt, W_rt)(0);
        lt.fill(0.0);
        rt.fill(rt(0));
    } else {
        lt.fill(0.0);
        rt.fill(0.0);
    }

    total = lv + rv + lc + rc + lt + rt;
}

current::current(const device & d, const wave_packet psi[6], const potential & phi0, const potential & phi)
    : lv(d.N_x), rv(d.N_x), lc(d.N_x), rc(d.N_x), lt(d.N_x), rt(d.N_x) {
//    using namespace arma;

//    // get imag(conj(psi) * psi)
//    auto get_psi_I = [&] (const cx_mat & m) {
//        mat ret(m.n_rows / 2, m.n_cols);
//        auto ptr0 = m.memptr();
//        auto ptr1 = ret.memptr();
//        vec t_vec(d.t_vec.size() + 1);
//        t_vec({0, d.t_vec.size() - 1}) = d.t_vec;
//        for (unsigned i = 1; i < m.n_elem - 1; i += 2) {
//            (*ptr1++) = std::imag(ptr0[i] * std::conj(ptr0[i + 1])) * t_vec(i % t_vec.size());
//        }
//        for (unsigned i = 0; i < m.n_cols; ++i) {
//            ret(d.N_x-1, i) = ret(d.N_x-2, i);
//        }
//        return ret;
//    };
//    auto psi_I_lv = get_psi_I(psi[LV].data);
//    auto psi_I_rv = get_psi_I(psi[RV].data);
//    auto psi_I_lc = get_psi_I(psi[LC].data);
//    auto psi_I_rc = get_psi_I(psi[RC].data);

//    static constexpr auto scale = 4.0 * c::e * c::e / c::h_bar / M_PI;

//    lv = scale * psi_I_lv * psi[LV].W;
//    rv = scale * psi_I_rv * psi[RV].W;
//    lc = scale * psi_I_lc * psi[LC].W;
//    rc = scale * psi_I_rc * psi[RC].W;
//    lt.fill(0.0);
//    rt.fill(0.0);

//    if (psi[LT].E.size() > 0) {
//        auto psi_I_lt = get_psi_I(psi[LT].data);
//        unsigned i0;
//        unsigned i1 = psi[LT].E.size() - 1;
//        for (i0 = 0; i0 < i1; ++i0) {
//            if ((psi[LT].E(i0) + phi.s() - phi0.s()) > phi.d()) {
//                lt = scale * psi_I_lt.cols({i0, i1}) * psi[LT].W({i0, i1});
//                break;
//            }
//        }
//    }

//    if (psi[RT].E.size() > 0) {
//        auto psi_I_rt = get_psi_I(psi[RT].data);
//        unsigned i0;
//        unsigned i1 = psi[RT].E.size() - 1;
//        for (i0 = 0; i0 < i1; ++i0) {
//            if ((psi[RT].E(i0) + phi.d() - phi0.d()) > phi.s()) {
//                rt = scale * psi_I_rt.cols({i0, i1}) * psi[RT].W({i0, i1});
//                break;
//            }
//        }
//    }

//    total = lv + rv + lc + rc + lt + rt;
}

#endif
