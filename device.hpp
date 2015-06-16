#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <armadillo>
#include <sstream>
#include <string>

//#include <QSettings>

#include "constant.hpp"

class device {
public:
    struct model {
        double E_g;
        double m_eff;
        double E_gc;
        double m_efc;
        double F_s;
        double F_g;
        double F_d;
    };

    struct geometry {
        double eps_cnt;
        double eps_ox;
        double l_sc;
        double l_sox;
        double l_sg;
        double l_g;
        double l_dg;
        double l_dox;
        double l_dc;
        double r_cnt;
        double d_ox;
        double r_ext;
        double dx;
        double dr;
    };

    std::string name;

    // model parameters
    double E_g;      // bandgap
    double m_eff;    // effective mass
    double E_gc;     // bandgap of contacts
    double m_efc;    // effective mass of contacts
    double F_s;      // Fermi level in source
    double F_g;      // Fermi level in gate
    double F_d;      // Fermi level in drain
    double F_sc;     // Fermi level in source contact
    double F_dc;     // Fermi level in drain contact

    // geometry
    double eps_cnt;  // relative permittivity of nanotube
    double eps_ox;   // relative permittivity of oxide
    double l_sc;     // source contact length
    double l_sox;    // source oxide length
    double l_sg;     // length between source and gate
    double l_g;      // gate length
    double l_dg;     // length between drain and gate
    double l_dox;    // drain oxide length
    double l_dc;     // drain contact length
    double r_cnt;    // nanotube radius
    double d_ox;     // oxide thickness
    double r_ext;    // extension thickness
    double dx;       // x lattice constant
    double dr;       // r lattice constant
    double l;        // device length
    double R;        // complete radius

    // lattice in x direction
    int N_sc;        // # of points in source contact
    int N_sox;       // # of points in source oxide
    int N_sg;        // # of points between source and gate
    int N_g;         // # of points in gate
    int N_dg;        // # of points between drain and gate
    int N_dox;       // # of points in drain oxide
    int N_dc;        // # of points in drain contact
    int N_x;         // total # of points
    arma::vec x;     // x lattice points

    // lattice in r direction
    int M_cnt;       // # of points in nanotube
    int M_ox;        // # of points in oxide
    int M_ext;       // # of points over oxide
    int M_r;         // total # of points
    arma::vec r;     // r lattice points

    // x ranges
    arma::span sc;   // source contact area
    arma::span sox;  // source oxide area
    arma::span sg;   // area between source and gate
    arma::span g;    // gate area;
    arma::span dg;   // area between drain and gate
    arma::span dox;  // drain oxide area
    arma::span dc;   // drain contact area
    arma::span sc2;  // source contact area twice
    arma::span sox2; // source oxide area twice;
    arma::span sg2;  // area between source and gate
    arma::span g2;   // gate area twice
    arma::span dg2;  // area between drain and gate twice
    arma::span dox2; // drain oxide area twice
    arma::span dc2;  // drain contact twice

    // hopping parameters
    double t1;       // hopping between orbitals in same unit cell
    double t2;       // hopping between orbitals in neighbouring unit cells
    double tc1;      // hopping between orbitals in same unit cell in contact area
    double tc2;      // hopping between orbitals in neighbouring unit cells in contact area
    double tcc;      // hopping between contact and central area
    arma::vec t_vec; // vector with t values

    inline device(const std::string & n, const model & m, const geometry & g);
    //inline device(const std::string & filename);
    inline void update(const std::string & n);
    inline std::string to_string();

};

static const device::geometry fet_geometry {
    10.0, // eps_cnt
    25.0, // eps_ox
    10.0, // l_sc
    10.0, // l_sox
     5.0, // l_sg
    10.0, // l_g
     5.0, // l_dg
    10.0, // l_dox
    10.0, // l_dc
     1.0, // r_cnt
     2.0, // d_ox
     2.0, // r_ext
     0.1, // dx
     0.1  // dr
};

static const device::geometry tfet_geometry {
    10.0, // eps_cnt
    25.0, // eps_ox
    10.0, // l_sc
     7.0, // l_sox
     3.0, // l_sg
    10.0, // l_g
    10.0, // l_dg
     0.0, // l_dox
    10.0, // l_dc
     1.0, // r_cnt
     2.0, // d_ox
     2.0, // r_ext
     0.1, // dx
     0.1  // dr
};

static const device::model nfet_model {
    0.62,            // E_g
    0.05 * c::m_e,   // m_eff
    0.62,            // E_gc
    0.05 * c::m_e,   // m_efc
    0.62 / 2 + 0.011,// F_s
    0.00,            // F_g
    0.62 / 2 + 0.011 // F_d
};

static const device::model pfet_model {
    nfet_model.E_g,   // E_g
    nfet_model.m_eff, // m_eff
    nfet_model.E_gc,  // E_gc
    nfet_model.m_efc, // m_efc
    - nfet_model.F_s, // F_s
    - nfet_model.F_g, // F_g
    - nfet_model.F_d  // F_d
};

static const device::model tfet_model {
    0.62,               // E_g
    0.05 * c::m_e,      // m_eff
    0.62,               // E_gc
    0.05 * c::m_e,      // m_efc
    - 0.62 / 2 + 0.011, // F_s
    0.00,               // F_g
    0.62 / 2 + 0.011    // F_d
};

//----------------------------------------------------------------------------------------------------------------------

device::device(const std::string & n, const model & m, const geometry & g) {
    eps_cnt = g.eps_cnt;
    eps_ox  = g.eps_ox;
    l_sc    = g.l_sc;
    l_sox   = g.l_sox;
    l_sg    = g.l_sg;
    l_g     = g.l_g;
    l_dg    = g.l_dg;
    l_dox   = g.l_dox;
    l_dc    = g.l_dc;
    r_cnt   = g.r_cnt;
    d_ox    = g.d_ox;
    r_ext   = g.r_ext;
    dx      = g.dx;
    dr      = g.dr;

    E_g   = m.E_g;
    m_eff = m.m_eff;
    E_gc  = m.E_gc;
    m_efc = m.m_efc;
    F_s   = m.F_s;
    F_g   = m.F_g;
    F_d   = m.F_d;

    update(n);
}

//device::device(const std::string & filename) {
//}

void device::update(const std::string & n) {
    name = n;
    F_sc = F_s;
    F_dc = F_d;

    l = l_sc + l_sox + l_sg + l_g + l_dg + l_dox + l_dc;
    R = r_cnt + d_ox + r_ext;

    N_sc  = round(l_sc  / dx);
    N_sox = round(l_sox / dx);
    N_sg  = round(l_sg  / dx);
    N_g   = round(l_g   / dx);
    N_dg  = round(l_dg  / dx);
    N_dox = round(l_dox / dx);
    N_dc  = round(l_dc  / dx);
    N_x   = N_sc + N_sox + N_sg + N_g + N_dg + N_dox + N_dc;
    x     = arma::linspace(0.5 * dx, l - 0.5 * dx, N_x);

    M_cnt = round(r_cnt / dr);
    M_ox  = round(d_ox  / dr);
    M_ext = round(r_ext / dr);
    M_r   = M_cnt + M_ox + M_ext;
    r     = arma::linspace(0.5 * dr, R - 0.5 * dr, M_r);

    sc   = arma::span(        0,   - 1 + N_sc );
    sox  = arma::span( sc.b + 1,  sc.b + N_sox);
    sg   = arma::span(sox.b + 1, sox.b + N_sg );
    g    = arma::span( sg.b + 1,  sg.b + N_g  );
    dg   = arma::span(  g.b + 1,   g.b + N_dg );
    dox  = arma::span( dg.b + 1,  dg.b + N_dox);
    dc   = arma::span(dox.b + 1, dox.b + N_dc );
    sc2  = arma::span( sc.a * 2,  sc.b * 2 + 1);
    sox2 = arma::span(sox.a * 2, sox.b * 2 + 1);
    sg2  = arma::span( sg.a * 2,  sg.b * 2 + 1);
    g2   = arma::span(  g.a * 2,   g.b * 2 + 1);
    dg2  = arma::span( dg.a * 2,  dg.b * 2 + 1);
    dox2 = arma::span(dox.a * 2, dox.b * 2 + 1);
    dc2  = arma::span( dc.a * 2,  dc.b * 2 + 1);

    t1  = 0.25 * E_g  * (1 + sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_eff * E_g  * c::e)));
    t2  = 0.25 * E_g  * (1 - sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_eff * E_g  * c::e)));
    tc1 = 0.25 * E_gc * (1 + sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_efc * E_gc * c::e)));
    tc2 = 0.25 * E_gc * (1 - sqrt(1 + 2 * c::h_bar2 / (dx*dx * 1E-18 * m_efc * E_gc * c::e)));
    tcc = 2.0 / (1.0 / t2 + 1.0 / tc2);

    t_vec = arma::vec(N_x * 2 - 1);
    bool b = true;
    for (unsigned i = sc2.a; i < sc2.b; ++i) {
        t_vec(i) = b ? tc1 : tc2;
        b = !b;
    }
    t_vec(sc2.b) = tcc;
    b = true;
    for (unsigned i = sox2.a; i < dox2.b; ++i) {
        t_vec(i) = b ? t1 : t2;
        b = !b;
    }
    t_vec(dox2.b) = tcc;
    b = true;
    for (unsigned i = dc2.a; i < dc2.b; ++i) {
        t_vec(i) = b ? tc1 : tc2;
        b = !b;
    }
}

std::string device::to_string() {
    using namespace std;

    stringstream ss;

    ss << "name    = " << name    << endl;
    ss << endl;

    ss << "; model" << endl;
    ss << "E_g     = " << E_g     << endl;
    ss << "m_eff   = " << m_eff   << endl;
    ss << "E_gc    = " << E_gc    << endl;
    ss << "m_efc   = " << m_efc   << endl;
    ss << "F_s     = " << F_s     << endl;
    ss << "F_g     = " << F_g     << endl;
    ss << "F_d     = " << F_d     << endl;

    ss << "; geometry" << endl;
    ss << "eps_cnt = " << eps_cnt << endl;
    ss << "eps_ox  = " << eps_ox  << endl;
    ss << "l_sc    = " << l_sc    << endl;
    ss << "l_sox   = " << l_sox   << endl;
    ss << "l_sg    = " << l_sg    << endl;
    ss << "l_g     = " << l_g     << endl;
    ss << "l_dg    = " << l_dg    << endl;
    ss << "l_dox   = " << l_dox   << endl;
    ss << "l_dc    = " << l_dc    << endl;
    ss << "r_cnt   = " << r_cnt   << endl;
    ss << "d_ox    = " << d_ox    << endl;
    ss << "r_ext   = " << r_ext   << endl;
    ss << "dx      = " << dx      << endl;
    ss << "dr      = " << dr      << endl;

    return ss.str();
}

#endif

