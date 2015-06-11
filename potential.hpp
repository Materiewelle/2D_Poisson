#ifndef POTENTIAL_HPP
#define POTENTIAL_HPP

#include <armadillo>
#include <array>
#include <unordered_map>

#include "anderson.hpp"
#include "voltage.hpp"
#include "gnuplot.hpp"

// forward declarations
#ifndef CHARGE_DENSITY_HPP
class charge_density;
#endif

class potential {
public:
    arma::vec data;
    arma::vec twice;

    inline potential();
    inline potential(const device & d, const voltage & V);
    inline potential(const device & d, const arma::vec & R0);
    inline potential(const device & d, const arma::vec & R0, const charge_density & n);
    inline double update(const device & d, const arma::vec & R0, const charge_density & n, anderson & mr_neo);

    inline void smooth(const device & d);

    inline double & operator()(int index);
    inline const double & operator()(int index) const;
    inline double s() const;
    inline double d() const;

private:

    template<bool minmax>
    inline void smooth(unsigned x0, unsigned x1);
    inline void update_twice();
};

// rest of includes
#include "charge_density.hpp"
#include "constant.hpp"
#include "device.hpp"

//----------------------------------------------------------------------------------------------------------------------

namespace potential_impl {

    static inline arma::vec poisson(const device & d, const arma::vec & R0);
    static inline arma::vec poisson(const device & d, const arma::vec & R0, const charge_density & n);
    template<bool duplicate = false, bool black = false>
    static inline arma::mat poisson2D(const device & d, const voltage & V, const charge_density & n);

    static inline std::array<arma::mat, 4> & get_eps(const device & d);
    static inline arma::sp_mat & get_S(const device & d);

    static inline arma::vec get_R0(const device & d, const voltage & V);
    static inline arma::vec get_R(const device & d, const arma::vec & R0, const charge_density & n);

    static std::unordered_map<std::string, std::array<arma::mat, 4>> eps_d;
    static std::unordered_map<std::string, arma::sp_mat> S_d;
}

//----------------------------------------------------------------------------------------------------------------------

static inline void plot_phi2D(const device & d, const voltage & V);
static inline void plot_phi2D(const device & d, const voltage & V, const charge_density & n);

potential::potential() {
}

potential::potential(const device & d, const voltage & V)
    : twice(d.N_x * 2) {
    using namespace potential_impl;

    arma::vec R0 = get_R0(d, V);

    arma::vec phi2D = poisson(d, R0);
    data = phi2D({arma::uword((d.M_cnt - 1) * d.N_x), arma::uword(d.M_cnt * d.N_x - 1)});

    update_twice();
}

potential::potential(const device & d, const arma::vec & R0)
    : twice(d.N_x * 2) {
    using namespace potential_impl;

    arma::vec phi2D = poisson(d, R0);
    data = phi2D({arma::uword((d.M_cnt - 1) * d.N_x), arma::uword(d.M_cnt * d.N_x - 1)});

    update_twice();
}

potential::potential(const device & d, const arma::vec & R0, const charge_density & n)
    : twice(d.N_x * 2) {
    using namespace potential_impl;

    arma::vec phi2D = poisson(d, R0, n);
    data = phi2D({arma::uword((d.M_cnt - 1) * d.N_x), arma::uword(d.M_cnt * d.N_x - 1)});

    update_twice();
}

double potential::update(const device & d, const arma::vec & R0, const charge_density & n, anderson & mr_neo) {
    using namespace arma;
    using namespace potential_impl;

    arma::vec phi2D = poisson(d, R0, n);
    vec f = phi2D({arma::uword((d.M_cnt - 1) * d.N_x), arma::uword(d.M_cnt * d.N_x - 1)}) - data;

    // anderson mixing
    mr_neo.update(data, f);

    update_twice();

    // return dphi
    return max(abs(f));
}

void potential::smooth(const device & d) {

    // smooth source region
    if (d.F_s > 0) {
        smooth<true>(0, d.N_sc + d.N_sox + d.N_sg + d.N_g * 0.05);
    } else {
        smooth<false>(0, d.N_sc + d.N_sox + d.N_sg + d.N_g * 0.05);
    }

    // smooth drain region
    if (d.F_d > 0) {
        smooth<true>(d.N_sc + d.N_sox + d.N_sg + d.N_g * 0.95, d.N_x);
    } else {
        smooth<false>(d.N_sc + d.N_sox + d.N_sg + d.N_g * 0.95, d.N_x);
    }

    update_twice();
}

double & potential::operator()(int index) {
    return data(index);
}
const double & potential::operator()(int index) const {
    return data(index);
}

double potential::s() const {
    return data(0);
}
double potential::d() const {
    return data(data.size() - 1);
}

template<bool minmax>
void potential::smooth(unsigned x0, unsigned x1) {
    using namespace arma;

    if (minmax) {
        for (unsigned i = x0; i < x1 - 1; ++i) {
            if (data(i+1) >= data(i)) {
                continue;
            }
            for (unsigned j = i + 1; j < x1; ++j) {
                if (data(j) >= data(i)) {
                    data({i+1, j-1}).fill(data(i));
                    break;
                }
            }
        }
        for (unsigned i = x1 - 1; i >= x0 + 1; --i) {
            if (data(i-1) >= data(i)) {
                continue;
            }
            for (unsigned j = i - 1; j >= 1; --j) {
                if (data(j) >= data(i)) {
                    data({j+1, i-1}).fill(data(i));
                    break;
                }
            }
        }
    } else {
        for (unsigned i = x0; i < x1 - 1; ++i) {
            if (data(i+1) <= data(i)) {
                continue;
            }
            for (unsigned j = i + 1; j < x1; ++j) {
                if (data(j) <= data(i)) {
                    data({i+1, j-1}).fill(data(i));
                    break;
                }
            }
        }
        for (unsigned i = x1 - 1; i >= x0 + 1; --i) {
            if (data(i-1) <= data(i)) {
                continue;
            }
            for (unsigned j = i - 1; j >= 1; --j) {
                if (data(j) <= data(i)) {
                    data({j+1, i - 1}).fill(data(i));
                    break;
                }
            }
        }
    }
}

void potential::update_twice() {
    // duplicate each entry
    for (unsigned i = 0; i < data.size(); ++i) {
        twice(2 * i    ) = data(i);
        twice(2 * i + 1) = data(i);
    }
}

arma::vec potential_impl::poisson(const device & d, const arma::vec & R0) {
    arma::sp_mat S = get_S(d);
    return arma::spsolve(S, R0);
}

arma::vec potential_impl::poisson(const device & d, const arma::vec & R0, const charge_density & n) {
    // build right side
    arma::vec R = get_R(d, R0, n);
    arma::sp_mat S = get_S(d);
    return arma::spsolve(S, R);
}

template<bool duplicate = true, bool black = false>
arma::mat potential_impl::poisson2D(const device & d, const voltage & V, const charge_density & n) {
    using namespace arma;

    sp_mat S = get_S(d);
    vec R0 = get_R0(d, V);
    vec R = get_R(d, R0, n);
    vec phivec = spsolve(S, R);
    mat phimat(d.N_x, d.M_r);

    int k = 0;
    for (int j = 0; j < d.M_r; ++j) {
        for (int i = 0; i < d.N_x; ++i) {
            if (j < d.M_cnt) {
                phimat(i, j) = phivec(k++);
            } else if (j < d.M_cnt + d.M_ox) {
                if (i < d.N_sc) {
                    phimat(i, j) = black ? -2 : -(V.s + d.F_s);
                } else if (i >= d.N_x - d.N_dc) {
                    phimat(i, j) = black ? -2 : -(V.d + d.F_d);
                } else {
                    phimat(i, j) = phivec(k++);
                }
            } else {
                if (i < d.N_sc + d.N_sox) {
                    phimat(i, j) = black ? -2 : -(V.s + d.F_s);
                } else if (i >= d.N_x - d.N_dc - d.N_dox) {
                    phimat(i, j) = black ? -2 : -(V.d + d.F_d);
                } else if ((i >= d.N_sc + d.N_sox + d.N_sg) && (i < d.N_sc + d.N_sox + d.N_sg + d.N_g)) {
                    phimat(i, j) = black ? -2 : -(V.g + d.F_g);
                } else {
                    phimat(i, j) = phivec(k++);
                }
            }
        }
    }

    if (duplicate) {
        phimat = join_horiz(fliplr(phimat), phimat);
    }

    return phimat;
}

std::array<arma::mat, 4> & potential_impl::get_eps(const device & d) {
    using namespace std;

    // check if eps_mat was already initialized for this device
    auto it = eps_d.find(d.name);
    if (it != end(eps_d)) {
        return it->second;
    }

    // new eps_mat
    std::array<arma::mat, 4> eps;

    for (int i = 0; i < 4; ++i) {
        eps[i] = arma::mat(d.N_x, d.M_r);
        eps[i].fill(c::eps_0);
    }

    enum {
        L = 0, //left
        R = 1, //right
        I = 2, //inside  (down)
        O = 3  //outside (up)
    };

    int i, j;

    // cnt
    for (j = 0; j < d.M_cnt - 1; ++j) {
        for (i = 0; i < d.N_x; ++i) {
            eps[L](i, j) = d.eps_cnt * c::eps_0;
            eps[R](i, j) = d.eps_cnt * c::eps_0;
            eps[I](i, j) = d.eps_cnt * c::eps_0;
            eps[O](i, j) = d.eps_cnt * c::eps_0;
        }
    }

    // cnt border
    for (i = 0; i < d.N_sc; ++i) {
        eps[L](i, j) = 0.5 * (1.0 + d.eps_cnt) * c::eps_0;
        eps[R](i, j) = 0.5 * (1.0 + d.eps_cnt) * c::eps_0;
        eps[I](i, j) = d.eps_cnt * c::eps_0;
        eps[O](i, j) = c::eps_0;
    }
    eps[L](i, j) = 0.5 * (1.0 + d.eps_cnt) * c::eps_0;
    eps[R](i, j) = 0.5 * (d.eps_ox + d.eps_cnt) * c::eps_0;
    eps[I](i, j) = d.eps_cnt * c::eps_0;
    eps[O](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
    for (++i; i < d.N_x - d.N_dc - 1; ++i) {
        eps[L](i, j) = 0.5 * (d.eps_ox + d.eps_cnt) * c::eps_0;
        eps[R](i, j) = 0.5 * (d.eps_ox + d.eps_cnt) * c::eps_0;
        eps[I](i, j) = d.eps_cnt * c::eps_0;
        eps[O](i, j) = d.eps_ox * c::eps_0;
    }
    eps[L](i, j) = 0.5 * (d.eps_ox + d.eps_cnt) * c::eps_0;
    eps[R](i, j) = 0.5 * (1.0 + d.eps_cnt) * c::eps_0;
    eps[I](i, j) = d.eps_cnt * c::eps_0;
    eps[O](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
    for (++i; i < d.N_x; ++i) {
        eps[L](i, j) = 0.5 * (1.0 + d.eps_cnt) * c::eps_0;
        eps[R](i, j) = 0.5 * (1.0 + d.eps_cnt) * c::eps_0;
        eps[I](i, j) = d.eps_cnt * c::eps_0;
        eps[O](i, j) = c::eps_0;
    }

    // oxide
    for (++j; j < d.M_cnt + d.M_ox - 1; ++j) {
        i = d.N_sc;
        eps[L](i, j) = c::eps_0;
        eps[R](i, j) = d.eps_ox * c::eps_0;
        eps[I](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
        eps[O](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
        for (++i; i < d.N_x - d.N_dc - 1; ++i) {
            eps[L](i, j) = d.eps_ox * c::eps_0;
            eps[R](i, j) = d.eps_ox * c::eps_0;
            eps[I](i, j) = d.eps_ox * c::eps_0;
            eps[O](i, j) = d.eps_ox * c::eps_0;
        }
        eps[L](i, j) = d.eps_ox * c::eps_0;
        eps[R](i, j) = c::eps_0;
        eps[I](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
        eps[O](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
    }

    // oxide border
    i = d.N_sc;
    eps[L](i, j) = c::eps_0;
    eps[R](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
    eps[I](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
    eps[O](i, j) = c::eps_0;
    for (++i; i < d.N_x - d.N_dc - 1; ++i) {
        eps[L](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
        eps[R](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
        eps[I](i, j) = d.eps_ox * c::eps_0;
        eps[O](i, j) = c::eps_0;
    }
    eps[L](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
    eps[R](i, j) = c::eps_0;
    eps[I](i, j) = 0.5 * (1.0 + d.eps_ox) * c::eps_0;
    eps[O](i, j) = c::eps_0;

    eps_d[d.name] = eps;
    return eps_d[d.name];
}

arma::sp_mat & potential_impl::get_S(const device & d) {
    // check if S was already calculated for this device
    auto it = S_d.find(d.name);
    if (it != std::end(S_d)) {
        return it->second;
    }

    using namespace arma;
    double dr2 = 1.0 / d.dr / d.dr;
    double dx2 = 1.0 / d.dx / d.dx;

    enum {
        L = 0,
        R = 1,
        I = 2,
        O = 3
    };

    auto eps = get_eps(d);

    umat indices(2, d.M_r * d.N_x * 5);
    vec values(d.M_r * d.N_x * 5);

    uword N_v = 0;
    auto set_value = [&] (uword ki, uword kj, double value) {
        indices(0, N_v) = ki;
        indices(1, N_v) = kj;
        values(N_v++) = value;
    };

    int k = 0;         // current main diagonal element
    int i0 = 0;        // left i limit
    int i1 = d.N_x;    // right i limit
    int delta = d.N_x; // distance to next vertical coupling off diagonal

    for (int j = 0; j < d.M_r; ++j) {
        double r = j * d.dr + 0.5 * d.dr;
        double rm = r - 0.5 * d.dr;
        double rp = r + 0.5 * d.dr;

        for (int i = i0; i < i1; ++i) {
            double diag    = - dx2 * (r  * eps[L](i, j) + r  * eps[R](i, j))
                             - dr2 * (rp * eps[O](i, j) + rm * eps[I](i, j));
            double left    = dx2 * r  * eps[L](i, j);
            double right   = (i > i0) ? dx2 * r  * eps[R](i - 1, j) : 0;
            double inside  = dr2 * rm * eps[I](i, j);
            double outside = (j >  0) ? dr2 * rm * eps[O](i, j - 1) : 0;

            // horizontal von Neumann boundary conditions
            if (i == 1) {
                right *= 2;
            }
            if (i == d.N_x - 1) {
                left *= 2;
            }

            // vertical von Neumann boundary conditions
            if (j == 1) {
                outside -= dr2 * 0.5 * d.dr * eps[O](i, 0);
                outside *= 2;
            }
            if (j == d.M_r - 1) {
                inside += dr2 * 0.5 * d.dr * eps[I](i, d.M_r - 1);
                inside *= 2;
            }

            // store values
            set_value(k, k, diag);
            if (i > i0) {
                set_value(k, k - 1, left);
                set_value(k - 1, k, right);
            }
            if (j > 0) {
                set_value(k, k - delta, inside);
                set_value(k - delta, k, outside);
            }

            // next diag element
            ++k;
        }

        // cut off source, drain contacts
        if (j == d.M_cnt - 1) {
            i0 = d.N_sc;
            i1 = d.N_x - d.N_dc;
            delta = d.N_x - d.N_sc;
        }
        if (j == d.M_cnt) {
            delta = d.N_x - d.N_sc - d.N_dc;
        }

        // cut off gate contact
        if (j == d.M_cnt + d.M_ox - 1) {
            i0 = d.N_sc + d.N_sox;
            i1 = i0 + d.N_sg;
            delta = d.N_x - d.N_sc - d.N_dc - d.N_sox;
        }
        if ((j == d.M_cnt + d.M_ox) && (i0 == d.N_sc + d.N_sox)) {
            delta = d.N_x - d.N_sc - d.N_dc - d.N_sox - d.N_g;
        }
        if ((j == d.M_cnt + d.M_ox) && (i0 == d.N_sc + d.N_sox + d.N_sg + d.N_g)) {
            delta = d.N_x - d.N_sc - d.N_dc - d.N_sox - d.N_g - d.N_dox;
        }
        if (j >= d.M_cnt + d.M_ox) {
            if (i0 == d.N_sc + d.N_sox) {
                i0 = i1 + d.N_g;
                i1 = i0 + d.N_dg;
                --j; // repeat i loop for second part (right side of gate)
            } else {
                i1 = i0 - d.N_g;
                i0 = i1 - d.N_sg;
            }
        }
    }

    indices.resize(2, N_v);
    values.resize(N_v);

    S_d[d.name] = sp_mat(indices, values);
    return S_d[d.name];
}

arma::vec potential_impl::get_R0(const device & d, const voltage & V) {
    using namespace arma;
    double dr2 = 1.0 / d.dr / d.dr;
    double dx2 = 1.0 / d.dx / d.dx;

    enum {
        L = 0,
        R = 1,
        I = 2,
        O = 3
    };

    auto eps = get_eps(d);

    double V_s = - (V.s + d.F_s);
    double V_g = - (V.g + d.F_g);
    double V_d = - (V.d + d.F_d);

    vec R0(d.N_x * d.M_r);
    R0.fill(0);
    int i, j, k;
    double r, rp;

    j = d.M_cnt - 1;
    r = j * d.dr + 0.5 * d.dr;
    rp = r + 0.5 * d.dr;

    k = j * d.N_x;
    for (i = 0; i < d.N_sc; ++i) {
        R0(k++) -= dr2 * rp * eps[O](i, j) * V_s;
    }
    k = j * d.N_x + d.N_x - d.N_dc;
    for (i = d.N_x - d.N_dc; i < d.N_x; ++i) {
        R0(k++) -= dr2 * rp * eps[O](i, j) * V_d;
    }
    for (j = d.M_cnt; j < d.M_cnt + d.M_ox - 1; ++j) {
        r = j * d.dr + 0.5 * d.dr;
        R0(k) -= dx2 * r * eps[L](d.N_sc, j) * V_s;
        k += d.N_sox + d.N_sg + d.N_g + d.N_dg + d.N_dox - 1;
        R0(k++) -= dx2 * r * eps[R](d.N_x - d.N_dc - 1, j) * V_d;
    }
    r = j * d.dr + 0.5 * d.dr;
    rp = r + 0.5 * d.dr;
    R0(k) -= dx2 * r * eps[L](d.N_sc, j) * V_s;
    for (i = d.N_sc; i < d.N_sc + d.N_sox; ++i) {
        R0(k++) -= dr2 * rp * eps[O](i, j) * V_s;
    }
    k += d.N_sg;
    for (i = d.N_sc + d.N_sox + d.N_sg; i < d.N_sc + d.N_sox + d.N_sg + d.N_g; ++i) {
        R0(k++) -= dr2 * rp * eps[O](i, j) * V_g;
    }
    k += d.N_dg;
    for (i = d.N_x - d.N_dc - d.N_dox; i < d.N_x - d.N_dc; ++i) {
        R0(k++) -= dr2 * rp * eps[O](i, j) * V_d;
    }
    R0(k - 1) -= dx2 * r * eps[R](d.N_x - d.N_dc - 1, j) * V_d;

    for (j = d.M_cnt + d.M_ox; j < d.M_r; ++j) {
        r = j * d.dr + 0.5 * d.dr;
        R0(k) -= dx2 * r * eps[L](d.N_sc + d.N_sox, j) * V_s;
        k += d.N_sg - 1;
        R0(k++) -= dx2 * r * eps[R](d.N_sc + d.N_sox + d.N_sg - 1, j) * V_g;
        R0(k) -= dx2 * r * eps[L](d.N_sc + d.N_sox + d.N_sg + d.N_g, j) * V_g;
        k += d.N_dg - 1;
        R0(k++) -= dx2 * r * eps[R](d.N_x - d.N_dc - d.N_dox - 1, j) * V_d;
    }

    R0.resize(k);
    return R0;
}

arma::vec potential_impl::get_R(const device & d, const arma::vec & R0, const charge_density & n) {
    arma::vec R = R0;
    R({arma::uword((d.M_cnt - 1) * d.N_x), arma::uword(d.M_cnt * d.N_x - 1)}) += n.total * d.r_cnt * 1e9; // 10^9 because of m->nm in epsilon_0
    return R;
}

void plot_phi2D(const device & d, const voltage & V) {
    charge_density n;
    n.total.resize(d.N_x);
    n.total.fill(0);
    plot_phi2D(d, V, n);
}

void plot_phi2D(const device & d, const voltage & V, const charge_density & n) {
    arma::mat phi2D = potential_impl::poisson2D<true, true>(d, V, n).t();

    gnuplot gp;

    gp << "set palette defined ( 0 '#D73027', 1 '#F46D43', 2 '#FDAE61', 3 '#FEE090', 4 '#E0F3F8', 5 '#ABD9E9', 6 '#74ADD1', 7 '#4575B4' )\n";
    gp << "set title \"2D Potential\"\n";
    gp << "set xlabel \"x / nm\"\n";
    gp << "set ylabel \"r / nm\"\n";
    gp << "set zlabel \"Phi / V\"\n";
    gp << "unset key\n";

//    // indicate cnt area
//    gp << "set obj rect from " << 0 << "," << d.r_cnt << " to " << d.l << "," << -d.r_cnt << "front fillstyle empty\n";
//    gp << "set label \"CNT\" at " << 0.5 * d.l << "," << 0 << " center front\n";

//    // indicate oxide area
//    double x0 = d.l_sc + d.l_s;
//    double x1 = d.l - d.l_dc - d.l_d;
//    double y0 = d.r_cnt + d.d_ox;
//    double y1 = d.r_cnt;
//    gp << "set obj rect from " << x0 << "," << y0 << " to " << x1 << "," << y1 << "front fillstyle empty\n";
//    gp << "set label \"gate oxide\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * (y1 - y0) - y1<< " center front\n";
//    gp << "set obj rect from " << x0 << "," << -y1 << " to " << x1 << "," << -y0 << "front fillstyle empty\n";
//    gp << "set label \"gate oxide\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";

//    // indicate gate contact area
//    x0 = d.l_sc + d.l_s + d.l_sox;
//    x1 = d.l - d.l_dc - d.l_d - d.l_dox;
//    y0 = d.R;
//    y1 = d.r_cnt + d.d_ox;
//    gp << "set obj rect from " << x0 << "," << y0 << " to " << x1 << "," << y1 << "front fillstyle empty\n";
//    gp << "set label \"gate contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * (y1 - y0) - y1<< " center front\n";
//    gp << "set obj rect from " << x0 << "," << -y1 << " to " << x1 << "," << -y0 << "front fillstyle empty\n";
//    gp << "set label \"gate contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";

//    // indicate left contact area
//    x0 = 0;
//    x1 = d.l_sc;
//    y0 = d.R;
//    y1 = d.r_cnt;
//    gp << "set obj rect from " << x0 << "," << y0 << " to " << x1 << "," << y1 << "front fillstyle empty\n";
//    gp << "set label \"source contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * (y1 - y0) - y1<< " center front\n";
//    gp << "set obj rect from " << x0 << "," << -y1 << " to " << x1 << "," << -y0 << "front fillstyle empty\n";
//    gp << "set label \"source contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";

//    // indicate left contact area
//    x0 = d.l - d.l_dc;
//    x1 = d.l;
//    y0 = d.R;
//    y1 = d.r_cnt;
//    gp << "set obj rect from " << x0 << "," << y0 << " to " << x1 << "," << y1 << "front fillstyle empty\n";
//    gp << "set label \"drain contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * (y1 - y0) - y1<< " center front\n";
//    gp << "set obj rect from " << x0 << "," << -y1 << " to " << x1 << "," << -y0 << "front fillstyle empty\n";
//    gp << "set label \"drain contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";

    gp.set_background(d.x, arma::join_vert(arma::flipud(-d.r), d.r), phi2D);
    gp.plot();
}

#endif

