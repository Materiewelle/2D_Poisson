#ifndef GREEN_HPP
#define GREEN_HPP

#include <armadillo>
#include <complex>

#include "constant.hpp"
#include "device.hpp"
#include "inverse.hpp"
#include "potential.hpp"

static inline void self_energy(const potential & phi, double E, arma::cx_double & Sigma_s, arma::cx_double & Sigma_d) {
    using namespace arma;
    using namespace std;

    // kinetic energy in source and drain
    auto E_s = E - phi.s();
    auto E_d = E - phi.d();

    // shortcuts
    static constexpr double t12 = d::tc1 * d::tc1;
    static constexpr double t22 = d::tc2 * d::tc2;

    // self energy
    Sigma_s = E_s * E_s - t12 - t22;
    Sigma_s = 0.5 * (E_s * E_s - t12 + t22 + sqrt(Sigma_s * Sigma_s + - 4 * t12 * t22)) / E_s;
    Sigma_d = E_d * E_d - t12 - t22;
    Sigma_d = 0.5 * (E_d * E_d - t12 + t22 + sqrt(Sigma_d * Sigma_d + - 4 * t12 * t22)) / E_d;

    // imaginary part must be negative
    Sigma_s.imag(-std::abs(Sigma_s.imag()));
    Sigma_d.imag(-std::abs(Sigma_d.imag()));
}

template<bool source>
static inline arma::cx_vec green_col(const potential & phi, double E, arma::cx_double & Sigma_s, arma::cx_double & Sigma_d) {
    using namespace arma;

    static const arma::vec t_vec_neg     = - d::t_vec;

    self_energy(phi, E, Sigma_s, Sigma_d);

    // build diagonal part of hamiltonian
    auto D = conv_to<cx_vec>::from(E - phi.twice);
    D(0)            -= Sigma_s;
    D(D.size() - 1) -= Sigma_d;

    return inverse_col<source>(t_vec_neg, D);
}

static inline arma::mat get_lDOS(const potential & phi, int N_grid, arma::vec & E) {
    using namespace arma;
    using namespace std::complex_literals;

    static const arma::vec t_vec_neg     = - d::t_vec;

    mat ret(N_grid, d::N_x);

    double phi_min = min(phi.data);
    double phi_max = max(phi.data);

    E = linspace(phi_min - 0.5 * d::E_g - 0.2, phi_max + 0.5 * d::E_g + 0.2, N_grid);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N_grid; ++i) {
        cx_double Sigma_s;
        cx_double Sigma_d;
        self_energy(phi, E(i), Sigma_s, Sigma_d);

        auto D = conv_to<cx_vec>::from(E(i) - phi.twice);
        D(0)            -= Sigma_s;
        D(D.size() - 1) -= Sigma_d;
        D += 0.001i;

        vec mixed = -arma::imag(inverse_diag(t_vec_neg, D)) / M_PI;

        for (int j = 0; j < d::N_x; ++j) {
            ret(i, j) = mixed(2*j) + mixed(2*j+1);
        }
    }
    return ret;
}

static inline void plot_ldos(const potential & phi, const unsigned N_grid) {
    gnuplot gp;

    gp << "set title \"Logarithmic lDOS\"\n";
    gp << "set xlabel \"x / nm\"\n";
    gp << "set ylabel \"E / eV\"\n";
    gp << "set zlabel \"log(lDOS)\"\n";
    gp << "unset key\n";
    gp << "unset colorbox\n";
    //gp << "set terminal pdf rounded color enhanced font 'arial,12'\n";
    //gp << "set output 'lDOS.pdf'\n";

    arma::vec E;
    arma::mat lDOS = get_lDOS(phi, N_grid, E);
    gp.set_background(d::x, E, arma::log(lDOS));

    arma::vec vband = phi.data;
    vband(d::sc) += -0.5 * d::E_gc;
    vband(d::s)  += -0.5 * d::E_g;
    vband(d::g)  += -0.5 * d::E_g;
    vband(d::d)  += -0.5 * d::E_g;
    vband(d::dc) += -0.5 * d::E_gc;

    arma::vec cband = phi.data;
    cband(d::sc) += +0.5 * d::E_gc;
    cband(d::s)  += +0.5 * d::E_g;
    cband(d::g)  += +0.5 * d::E_g;
    cband(d::d)  += +0.5 * d::E_g;
    cband(d::dc) += +0.5 * d::E_gc;

    gp.add(d::x, vband);
    gp.add(d::x, cband);
    gp.add(d::x, phi.data);

    unsigned N_s = std::round(d::N_sc + 0.5 * d::N_s);
    arma::vec fermi_l(N_s);
    fermi_l.fill(d::F_sc + phi.s());
    arma::vec x_l = d::x(arma::span(0, N_s-1));
    gp.add(x_l, fermi_l);

    unsigned N_d = std::round(d::N_dc + 0.5 * d::N_d);
    arma::vec fermi_r(N_d);
    fermi_r.fill(d::F_dc + phi.d());
    arma::vec x_r = d::x(arma::span(d::N_x-N_d, d::N_x-1));
    gp.add(x_r, fermi_r);

    gp << "set style line 1 lt 1 lc rgb RWTH_Orange lw 2\n";
    gp << "set style line 2 lt 1 lc rgb RWTH_Orange lw 2\n";
    gp << "set style line 3 lt 1 lc rgb RWTH_Rot lw 2\n";
    gp << "set style line 4 lc rgb RWTH_Schwarz lw 1 lt 3\n";
    gp << "set style line 5 lc rgb RWTH_Schwarz lw 1 lt 3\n";

    gp.plot();
}

#endif

