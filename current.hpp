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
    arma::vec total;

    inline current();
    inline current(const device & d, const potential & phi);
    inline current(const device & d, const wave_packet psi[4], const potential & phi);
};

//----------------------------------------------------------------------------------------------------------------------

current::current() {
}

current::current(const device & d, const potential & phi)
    : lv(d.N_x), rv(d.N_x), lc(d.N_x), rc(d.N_x) {
    using namespace arma;

    // transmission probability
    auto transmission = [&] (double E) -> double {
        cx_double Sigma_s, Sigma_d;
        cx_vec G = green_col<false>(d, phi, E, Sigma_s, Sigma_d);
        return 4 * Sigma_s.imag() * Sigma_d.imag() * (std::norm(G(1)) + std::norm(G(2)));
    };

    static constexpr auto scale = 2.0 * c::e * c::e / c::h;

    auto i_v = linspace(phi.d() + charge_density_impl::E_min, phi.d() - 0.5 * d.E_gc, 20);
    auto i_c = linspace(phi.d() + 0.5 * d.E_gc, phi.d() + charge_density_impl::E_max, 20);

    auto I_s = [&] (double E) -> double {
        double ret = transmission(E) * scale;
        double f = fermi(E - phi.s(), d.F_sc);
        ret *= (E >= phi.d()) ? f : (f - 1);
        return ret;
    };

    auto I_d = [&] (double E) -> double {
        double ret = transmission(E) * scale;
        double f = fermi(E - phi.d(), d.F_dc);
        ret *= (E >= phi.d()) ? f : (f - 1);
        return ret;
    };

    vec E, W; // just dummy junk

    lv(0) = +integral(I_s, 1, i_v, charge_density_impl::rel_tol, c::epsilon(1e-10), E, W)(0);
    lv.fill(lv(0));

    rv(0) = -integral(I_d, 1, i_v, charge_density_impl::rel_tol, c::epsilon(1e-10), E, W)(0);
    rv.fill(rv(0));

    lc(0) = +integral(I_s, 1, i_c, charge_density_impl::rel_tol, c::epsilon(1e-10), E, W)(0);
    lc.fill(lc(0));

    rc(0) = -integral(I_d, 1, i_c, charge_density_impl::rel_tol, c::epsilon(1e-10), E, W)(0);
    rc.fill(rc(0));

    total = lv + rv + lc + rc;
}

current::current(const device & d, const wave_packet psi[4], const potential & phi)
    : lv(d.N_x), rv(d.N_x), lc(d.N_x), rc(d.N_x) {
    using namespace arma;

    auto get_I = [&d, &phi] (const wave_packet & psi, vec & I) {
        // initial value = 0
        I.fill(0.0);

        #pragma omp parallel
        {
            vec I_thread(I.size());
            I_thread.fill(0.0);

            // loop over all energies
            #pragma omp for schedule(static) nowait
            for (unsigned i = 0; i < psi.E0.size(); ++i) {
                // get fermi factor and weight
                double f = psi.F0(i);
                double W = psi.W(i);

                // a: current from cell j-1 to j; b: current from cell j to j+1
                double a;
                double b = d.t_vec(1) * (std::conj((*psi.data)(2, i)) * (*psi.data)(1, i)).imag();

                int j = 0;

                // first value: just take b (current from cell 0 to 1)
                I_thread(j) += b * W * ((psi.E(j, i) >= phi(j)) ? f : (f - 1));
                for (j = 1; j < d.N_x - 1; ++j) {
                    a = b;
                    b = d.t_vec(j * 2 + 1) * (std::conj((*psi.data)(2 * j + 2, i)) * (*psi.data)(2 * j + 1, i)).imag();

                    // average of a and b
                    I_thread(j) += 0.5 * (a + b) * W * ((psi.E(j, i) >= phi(j)) ? f : (f - 1));
                }
                // last value: take old b (current from cell N_x - 2 to N_x - 1)
                I_thread(j) += b * W * ((psi.E(j, i) >= phi(j)) ? f : (f - 1));
            }

            #pragma omp critical
            {
                I += I_thread;
            }
        }
    };

    // calculate currents for all areas
    get_I(psi[LV], lv);
    get_I(psi[RV], rv);
    get_I(psi[LC], lc);
    get_I(psi[RC], rc);

    // scaling
    static constexpr double scale = 4.0 * c::e * c::e / c::h_bar / M_PI;
    lv *= scale;
    rv *= scale;
    lc *= scale;
    rc *= scale;

    // calculate total current
    total = lv + rv + lc + rc;
}

#endif
