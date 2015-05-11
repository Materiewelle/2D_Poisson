#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <armadillo>

#include "constant.hpp"
#include "fermi.hpp"
#include "integral.hpp"

namespace d {

    // material properties
    static constexpr double eps_g = 10;                                           // relative permittivity of channel
    static constexpr double eps_o = 25;                                           // relative permittivity of oxide
    static constexpr double E_g   = 0.62;                                         // bandgap
    static constexpr double m_eff = 0.04 * c::m_e;                                // effective mass
    static constexpr double E_gc  = 0.3;                                          // bandgap of contacts
    static constexpr double m_efc = 0.1 * c::m_e;                                 // effective mass of contacts
    static constexpr double F_s   = +(E_g/2 + 0.011);                              // Fermi level in source
    static constexpr double F_g   = 0;                                            // Fermi level in gate
<<<<<<< HEAD
    static constexpr double F_d   = +(E_g/2 + 0.011);                              // Fermi level in drain
    static constexpr double F_sc  = F_s;                                          // Fermi level in source contact
    static constexpr double F_dc  = F_d;                                          // Fermi level in drain contact
=======
    static constexpr double F_d   = +(E_g/2 + 0.03);                              // Fermi level in drain
    static constexpr double F_sc  = F_s - 0.05;                                   // Fermi level in source contact
    static constexpr double F_dc  = F_d - 0.05;                                   // Fermi level in drain contact
>>>>>>> e5e24eb97a40e1900cbe39ad326eb49de44532bc

    // geometry (everything in nm)
    static constexpr double l_sc  = 15;                                           // source contact length
    static constexpr double l_s   = 15;                                           // source length
    static constexpr double l_g   = 18;                                           // gate length
    static constexpr double l_d   = 15;                                           // drain length
    static constexpr double l_dc  = 15;                                           // drain contact length
    static constexpr double l     = l_sc + l_s + l_g + l_d + l_dc;                // device length
    static constexpr double d_g   = 1.3;                                          // channel thickness
    static constexpr double d_o   = 0.8;                                            // oxide thickness
    static constexpr double lam_g = sqrt(eps_g*d_g*d_g/8/eps_o*log(1+2*d_o/d_g)); // scr. length in channel
    static constexpr double lam_s = 1.0 < lam_g ? 1.0 : lam_g;                    // scr. length in source
    static constexpr double lam_d = 1.0 < lam_g ? 1.0 : lam_g;                    // scr. length in drain

    // lattice
    static constexpr double dx    = 0.1;                                          // lattice constant
    static constexpr int    N_sc  = round(l_sc / dx);                             // # of points in source contact
    static constexpr int    N_s   = round(l_s / dx);                              // # of points in source
    static constexpr int    N_g   = round(l_g / dx);                              // # of points in gate
    static constexpr int    N_d   = round(l_d / dx);                              // # of points in drain
    static constexpr int    N_dc  = round(l_dc / dx);                             // # of points in drain contact
    static constexpr int    N_x   = N_sc + N_s + N_g + N_d + N_dc;                // total # of points
    static const arma::vec  x     = arma::linspace(0.5 * dx, l - 0.5 * dx, N_x);  // lattice points

    // ranges
    static const arma::span sc    = arma::span(0, N_sc - 1);                      // source contact area
    static const arma::span s     = arma::span(sc.b + 1, sc.b + N_s);             // source area
    static const arma::span g     = arma::span( s.b + 1,  s.b + N_g);             // gate area
    static const arma::span d     = arma::span( g.b + 1,  g.b + N_d);             // drain area
    static const arma::span dc    = arma::span( d.b + 1,  d.b + N_dc);            // drain contact area
    static const arma::span sc2   = arma::span(sc.a * 2, sc.b * 2 + 1);           // source contact area twice
    static const arma::span s2    = arma::span( s.a * 2,  s.b * 2 + 1);           // source area twice
    static const arma::span g2    = arma::span( g.a * 2,  g.b * 2 + 1);           // gate area twice
    static const arma::span d2    = arma::span( d.a * 2,  d.b * 2 + 1);           // drain area twice
    static const arma::span dc2   = arma::span(dc.a * 2, dc.b * 2 + 1);           // drain contact area twice

    // hopping parameters central region
    static constexpr double t1    = 0.25 * E_g * (1 + sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_eff * E_g * c::e)));
    static constexpr double t2    = 0.25 * E_g * (1 - sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_eff * E_g * c::e)));

    // hopping parameters contact region
    static constexpr double tc1   = 0.25 * E_gc * (1 + sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_efc * E_gc * c::e)));
    static constexpr double tc2   = 0.25 * E_gc * (1 - sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_efc * E_gc * c::e)));
    static constexpr double tcc   = 2.0 / (1.0 / t2 + 1.0 / tc2);

    // constant parts of hamiltonian
    inline arma::vec create_t_vec() {
        arma::vec ret(N_x * 2 - 1);
        bool b = true;
        for (unsigned i = sc2.a; i < sc2.b; ++i) {
            ret(i) = b ? tc1 : tc2;
            b = !b;
        }
        ret(sc2.b) = tcc;
        b = true;
        for (unsigned i = s2.a; i < d2.b; ++i) {
            ret(i) = b ? t1 : t2;
            b = !b;
        }
        ret(d2.b) = tcc;
        b = true;
        for (unsigned i = dc2.a; i < dc2.b; ++i) {
            ret(i) = b ? tc1 : tc2;
            b = !b;
        }
        return ret;
    }

    static const auto t_vec = create_t_vec();

    // integration parameters
    static constexpr double E_min = -1.5;
    static constexpr double E_max = +1.5;
    static constexpr double rel_tol = 1e-4;

    // doping
    inline arma::vec create_n0() {
        using namespace arma;

        vec x0, x1, x2, x3, w0, w1, w2, w3;

        // valence band in contact region
        vec nvc = integral<2>([] (double E) {
            double dos = E / sqrt(4*tc1*tc1*tc2*tc2 - (E*E - tc1*tc1 - tc2*tc2) * (E*E - tc1*tc1 - tc2*tc2));
            vec ret = arma::vec(2);
            ret(0) = (1 - fermi(E, F_sc)) * dos;
            ret(1) = (1 - fermi(E, F_dc)) * dos;
            return ret;
        }, linspace(E_min, -0.5 * E_gc, 100), rel_tol, c::epsilon(), x0, w0);

        // conduction band in contact region
        vec ncc = integral<2>([] (double E) {
            double dos = E / sqrt(4*tc1*tc1*tc2*tc2 - (E*E - tc1*tc1 - tc2*tc2) * (E*E - tc1*tc1 - tc2*tc2));
            vec ret = arma::vec(2);
            ret(0) = fermi(E, F_sc) * dos;
            ret(1) = fermi(E, F_dc) * dos;
            return ret;
        }, linspace(0.5 * E_gc, E_max, 100), rel_tol, c::epsilon(), x1, w1);

        // valence band in central region
        vec nvsgd = integral<3>([] (double E) {
            double dos = E / sqrt(4*t1*t1*t2*t2 - (E*E - t1*t1 - t2*t2) * (E*E - t1*t1 - t2*t2));
            vec ret = arma::vec(3);
            ret(0) = (1 - fermi(E, F_s)) * dos;
            ret(1) = (1 - fermi(E, F_g)) * dos;
            ret(2) = (1 - fermi(E, F_d)) * dos;
            return ret;
        }, linspace(E_min, - 0.5 * E_g, 100), rel_tol, c::epsilon(), x2, w2);

        // conduction band in central region
        vec ncsgd = integral<3>([] (double E) {
            double dos = E / sqrt(4*t1*t1*t2*t2 - (E*E - t1*t1 - t2*t2) * (E*E - t1*t1 - t2*t2));
            vec ret = arma::vec(3);
            ret(0) = fermi(E, F_s) * dos;
            ret(1) = fermi(E, F_g) * dos;
            ret(2) = fermi(E, F_d) * dos;
            return ret;
        }, linspace(0.5 * E_g, E_max, 100), rel_tol, c::epsilon(), x3, w3);

        // total charge density in contact regions
        vec nc = nvc + ncc;
        // total charge density in central region
        vec nsgd = nvsgd + ncsgd;

        vec ret(N_x);
        ret(sc).fill(nc(0));
        ret(s).fill(nsgd(0));
        ret(g).fill(nsgd(1));
        ret(d).fill(nsgd(2));
        ret(dc).fill(nc(1));

        ret *= 16 * c::e / M_PI / M_PI / dx / d_g / d_g;

        return ret;
    }
    static const auto n0 = create_n0();

}

#endif

