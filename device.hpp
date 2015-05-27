#ifndef DEVICE_HPP
#define DEVICE_HPP
/*
#include <armadillo>

#include "constant.hpp"
#include "fermi.hpp"
#include "integral.hpp"
#include "gnuplot.hpp"

namespace d {

    // material properties
    static constexpr double eps_cnt = 10;                                         // relative permittivity in cnt
    static constexpr double eps_ox = 25;                                          // relative permittivity of oxide
    static constexpr double E_g   = 0.62;                                         // bandgap
    static constexpr double m_eff = 0.05 * c::m_e;                                 // effective mass
    static constexpr double E_gc  = 0.2;                                          // bandgap of contacts
    static constexpr double m_efc = 0.1 * c::m_e;                                 // effective mass of contacts
    static constexpr double F_s   = -(E_g/2 + 0.011);                             // Fermi level in source
    static constexpr double F_g   = 0;                                            // Fermi level in gate
    static constexpr double F_d   = +(E_g/2 + 0.011);                             // Fermi level in drain
    static constexpr double F_sc  = F_s;                                          // Fermi level in source contact
    static constexpr double F_dc  = F_d;                                          // Fermi level in drain contact

    // geometry (everything in nm)
    static constexpr double l_sc  = 12;                                           // source contact length
    static constexpr double l_s   = 5;                                            // source length
    static constexpr double l_sox = 5;                                            // source oxide length
    static constexpr double l_g   = 10;                                           // gate length
    static constexpr double l_dox = 5;                                            // drain oxide length
    static constexpr double l_d   = 5;                                            // drain length
    static constexpr double l_dc  = 12;                                           // drain contact length
    static constexpr double l     = l_sc + l_s + l_sox + l_g + l_dox + l_d + l_dc;// device length
    static constexpr double r_cnt = 1;                                            // CNT radius
    static constexpr double d_ox  = 2;                                            // oxide thickness
    static constexpr double r_ext = 2;                                            // extension thickness
    static constexpr double R     = r_cnt + d_ox + r_ext;                         // complete thickness

    // lattice in x direction
    static constexpr double dx    = 0.1;                                          // lattice constant
    static constexpr int    N_sc  = round(l_sc  / dx);                            // # of points in source contact
    static constexpr int    N_s   = round(l_s   / dx);                            // # of points in source
    static constexpr int    N_sox = round(l_sox / dx);                            // # of points in source oxide
    static constexpr int    N_g   = round(l_g   / dx);                            // # of points in gate
    static constexpr int    N_dox = round(l_dox / dx);                            // # of points in drain oxide
    static constexpr int    N_d   = round(l_d   / dx);                            // # of points in drain
    static constexpr int    N_dc  = round(l_dc  / dx);                            // # of points in drain contact
    static constexpr int    N_x   = N_sc + N_s + N_sox + N_g + N_dox + N_d + N_dc;// total # of points
    static const arma::vec  x     = arma::linspace(0.5 * dx, l - 0.5 * dx, N_x);  // lattice points

    // x ranges
    static const arma::span sc    = arma::span(        0,    N_sc - 1);           // source contact area
    static const arma::span s     = arma::span( sc.b + 1,  sc.b + N_s);           // source area
    static const arma::span sox   = arma::span(  s.b + 1,   s.b + N_sox);         // source oxide area
    static const arma::span g     = arma::span(sox.b + 1, sox.b + N_g);           // gate area
    static const arma::span dox   = arma::span(  g.b + 1,   g.b + N_dox);         // drain oxide area
    static const arma::span d     = arma::span(dox.b + 1, dox.b + N_d);           // drain area
    static const arma::span dc    = arma::span(  d.b + 1,   d.b + N_dc);          // drain contact area
    static const arma::span sc2   = arma::span( sc.a * 2,  sc.b * 2 + 1);         // source contact area twice
    static const arma::span s2    = arma::span(  s.a * 2,   s.b * 2 + 1);         // source area twice
    static const arma::span sox2  = arma::span(sox.a * 2, sox.b * 2 + 1);         // source oxide area twice
    static const arma::span g2    = arma::span(  g.a * 2,   g.b * 2 + 1);         // gate area twice
    static const arma::span dox2  = arma::span(dox.a * 2, dox.b * 2 + 1);         // drain oxide area twice
    static const arma::span d2    = arma::span(  d.a * 2,   d.b * 2 + 1);         // drain area twice
    static const arma::span dc2   = arma::span( dc.a * 2,  dc.b * 2 + 1);         // drain contact area twice

    // lattice in r-direction (for electrostatics)
    static constexpr double dr    = 0.1;                                          // lattice constant
    static constexpr int    M_cnt = round(r_cnt / dr);                            // # of points in nanotube
    static constexpr int    M_ox  = round(d_ox  / dr);                            // # of points in oxide
    static constexpr int    M_ext = round(r_ext / dr);                            // # of points in extension
    static constexpr int    M_r   = M_cnt + M_ox + M_ext;                         // total # of points
    static const arma::vec  r     = arma::linspace(0.5 * dr, R - 0.5 * dr, M_r);  // radial lattice points

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
    static constexpr double rel_tol = 1e-2;

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
        ret(sox).fill(0);
        ret(g).fill(nsgd(1));
        ret(dox).fill(0);
        ret(d).fill(nsgd(2));
        ret(dc).fill(nc(1));

        ret *= 2 * c::e / M_PI / M_PI / r_cnt / dr / dx; // spintel inside (?)

        return ret;
    }
    static const auto n0 = create_n0();

}
*/

#include <armadillo>
#include <sstream>
#include <string>

#include "constant.hpp"

class device {
public:
    std::string name;

    double eps_cnt  = 10;
    double eps_ox   = 25;                                           // relative permittivity of oxide
    double E_g      = 0.62;                                         // bandgap
    double m_eff    = 0.05 * c::m_e;                                // effective mass
    double E_gc     = 0.2;                                          // bandgap of contacts
    double m_efc    = 0.1 * c::m_e;                                 // effective mass of contacts
    double F_s      = +(E_g/2 + 0.011);                             // Fermi level in source
    double F_g      = 0;                                            // Fermi level in gate
    double F_d      = +(E_g/2 + 0.011);                             // Fermi level in drain
    double F_sc     = F_s;                                          // Fermi level in source contact
    double F_dc     = F_d;                                          // Fermi level in drain contact

    // geometry (everything in nm)
    double l_sc     = 12;                                           // source contact length
    double l_s      = 5;                                            // source length
    double l_sox    = 5;                                            // source oxide length
    double l_g      = 10;                                           // gate length
    double l_dox    = 5;                                            // drain oxide length
    double l_d      = 5;                                            // drain length
    double l_dc     = 12;                                           // drain contact length
    double l        = l_sc + l_s + l_sox + l_g + l_dox + l_d + l_dc;// device length
    double r_cnt    = 1;                                            // CNT radius
    double d_ox     = 2;                                            // oxide thickness
    double r_ext    = 2;                                            // extension thickness
    double R        = r_cnt + d_ox + r_ext;                         // complete thickness

    // lattice in x direction
    double dx       = 0.1;                                          // lattice constant
    int    N_sc     = round(l_sc  / dx);                            // # of points in source contact
    int    N_s      = round(l_s   / dx);                            // # of points in source
    int    N_sox    = round(l_sox / dx);                            // # of points in source oxide
    int    N_g      = round(l_g   / dx);                            // # of points in gate
    int    N_dox    = round(l_dox / dx);                            // # of points in drain oxide
    int    N_d      = round(l_d   / dx);                            // # of points in drain
    int    N_dc     = round(l_dc  / dx);                            // # of points in drain contact
    int    N_x      = N_sc + N_s + N_sox + N_g + N_dox + N_d + N_dc;// total # of points
    arma::vec  x    = arma::linspace(0.5 * dx, l - 0.5 * dx, N_x);  // lattice points

    // x ranges
    arma::span sc   = arma::span(        0,    N_sc - 1);           // source contact area
    arma::span s    = arma::span( sc.b + 1,  sc.b + N_s);           // source area
    arma::span sox  = arma::span(  s.b + 1,   s.b + N_sox);         // source oxide area
    arma::span g    = arma::span(sox.b + 1, sox.b + N_g);           // gate area
    arma::span dox  = arma::span(  g.b + 1,   g.b + N_dox);         // drain oxide area
    arma::span d    = arma::span(dox.b + 1, dox.b + N_d);           // drain area
    arma::span dc   = arma::span(  d.b + 1,   d.b + N_dc);          // drain contact area
    arma::span sc2  = arma::span( sc.a * 2,  sc.b * 2 + 1);         // source contact area twice
    arma::span s2   = arma::span(  s.a * 2,   s.b * 2 + 1);         // source area twice
    arma::span sox2 = arma::span(sox.a * 2, sox.b * 2 + 1);         // source oxide area twice
    arma::span g2   = arma::span(  g.a * 2,   g.b * 2 + 1);         // gate area twice
    arma::span dox2 = arma::span(dox.a * 2, dox.b * 2 + 1);         // drain oxide area twice
    arma::span d2   = arma::span(  d.a * 2,   d.b * 2 + 1);         // drain area twice
    arma::span dc2  = arma::span( dc.a * 2,  dc.b * 2 + 1);         // drain contact area twice

    // lattice in r-direction (for electrostatics)
    double dr       = 0.1;                                          // lattice constant
    int    M_cnt    = round(r_cnt / dr);                            // # of points in nanotube
    int    M_ox     = round(d_ox  / dr);                            // # of points in oxide
    int    M_ext    = round(r_ext / dr);                            // # of points in extension
    int    M_r      = M_cnt + M_ox + M_ext;                         // total # of points
    arma::vec  r    = arma::linspace(0.5 * dr, R - 0.5 * dr, M_r);  // radial lattice points

    // hopping parameters central region
    double t1       = 0.25 * E_g * (1 + sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_eff * E_g * c::e)));
    double t2       = 0.25 * E_g * (1 - sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_eff * E_g * c::e)));

    // hopping parameters contact region
    double tc1      = 0.25 * E_gc * (1 + sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_efc * E_gc * c::e)));
    double tc2      = 0.25 * E_gc * (1 - sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_efc * E_gc * c::e)));
    double tcc      = 2.0 / (1.0 / t2 + 1.0 / tc2);

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
    arma::vec t_vec = create_t_vec();

    inline device(std::string n);
    inline std::string to_string();
};

//----------------------------------------------------------------------------------------------------------------------

device::device(std::string n)
    : name(n) {
}

std::string device::to_string() {
    using namespace std;

    stringstream ss;

    ss << "name=" << name << endl;
    ss << endl;

    ss << "; material properties" << endl;
    ss << "eps_cnt=" << eps_cnt << endl;
    ss << "eps_ox="  << eps_ox  << endl;
    ss << "E_g="     << E_g     << endl;
    ss << "m_eff="   << m_eff   << endl;
    ss << "E_gc="    << E_gc    << endl;
    ss << "m_efc="   << m_efc   << endl;
    ss << "F_s="     << F_s     << endl;
    ss << "F_g="     << F_g     << endl;
    ss << "F_d="     << F_d     << endl;
    ss << "F_sc="    << F_sc    << endl;
    ss << "F_dc="    << F_dc    << endl;
    ss << endl;

    ss << "; geometry (everything in nm)" << endl;
    ss << "l_sc="  << l_sc  << endl;
    ss << "l_s="   << l_s   << endl;
    ss << "l_sox=" << l_sox << endl;
    ss << "l_g="   << l_g   << endl;
    ss << "l_dox=" << l_dox << endl;
    ss << "l_d="   << l_d   << endl;
    ss << "l_dc="  << l_dc  << endl;
    ss << "r_cnt=" << r_cnt << endl;
    ss << "d_ox="  << d_ox  << endl;
    ss << "r_ext=" << r_ext << endl;
    ss << endl;

    ss << "; lattice" << endl;
    ss << "dx=" << dx << endl;
    ss << "dr=" << dr << endl;
    ss << endl;

    return ss.str();
}

#endif

