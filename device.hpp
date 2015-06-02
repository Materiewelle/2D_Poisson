#ifndef DEVICE_HPP
#define DEVICE_HPP

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
    double l_s      = 2;                                            // source length
    double l_sox    = 4;                                            // source oxide length
    double l_g      = 9;                                           // gate length
    double l_dox    = 4;                                            // drain oxide length
    double l_d      = 2;                                            // drain length
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
//    inline void update();
    inline std::string to_string();
};

//----------------------------------------------------------------------------------------------------------------------

device::device(std::string n)
    : name(n) {
}

//void device::update() {

//}

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

