#ifndef POTENTIAL_HPP
#define POTENTIAL_HPP

#include <armadillo>

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
    inline potential(const voltage & V);
    inline potential(const arma::vec & R0, const charge_density & n);
    inline double update(const arma::vec & R0, const charge_density & n, anderson & mr_neo);

    inline void smooth();

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

    static inline arma::vec poisson(const arma::vec & R0, const charge_density & n);
    template<bool duplicate = false, bool black = false>
    static inline arma::mat poisson2D(const voltage & V, const charge_density & n);
//    static inline std::vector<double> get_boxes(std::vector<int> & ibox, std::vector<int> & jbox);
    static inline void get_eps_mat(arma::mat eps[4]);
    template<int dir>
    static inline double eps(int i, int j);
    static inline arma::sp_mat get_S();

    static inline arma::vec get_R0(const voltage & V);
    static inline arma::vec get_R(const arma::vec & R0, const charge_density & n);

    static const arma::sp_mat S = get_S();
}

//----------------------------------------------------------------------------------------------------------------------

static inline void plot_phi2D(const voltage & V, const charge_density & n = charge_density());

potential::potential() {
}

potential::potential(const voltage & V)
    : twice(d::N_x * 2) {
    using namespace potential_impl;

    arma::vec R0 = get_R0(V);
    charge_density n;

    arma::vec phi2D = poisson(R0, n);
    data = phi2D({(d::M_cnt - 1) * d::N_x, d::M_cnt * d::N_x - 1});

    update_twice();
}

potential::potential(const arma::vec & R0, const charge_density & n)
    : twice(d::N_x * 2) {
    using namespace potential_impl;

    arma::vec phi2D = poisson(R0, n);
    data = phi2D({(d::M_cnt - 1) * d::N_x, d::M_cnt * d::N_x - 1});

    update_twice();
}

double potential::update(const arma::vec & R0, const charge_density & n, anderson & mr_neo) {
    using namespace arma;
    using namespace potential_impl;

    arma::vec phi2D = poisson(R0, n);
    vec f = phi2D({(d::M_cnt - 1) * d::N_x, d::M_cnt * d::N_x - 1}) - data;

    // anderson mixing
    mr_neo.update(data, f);

    update_twice();

    // return dphi
    return max(abs(f));
}

void potential::smooth() {

    // smooth source region
    smooth<(d::F_s > 0)>(0, d::N_sc + d::N_s + d::N_sox + d::N_g * 0.05);

    // smooth drain region
    smooth<(d::F_d > 0)>(d::N_sc + d::N_s + d::N_sox + d::N_g * 0.95, d::N_x);

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
    return data(d::N_x - 1);
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
    for (unsigned i = 0; i < d::N_x; ++i) {
        twice(2 * i    ) = data(i);
        twice(2 * i + 1) = data(i);
    }
}

arma::vec potential_impl::poisson(const arma::vec & R0, const charge_density & n) {
    // build right side
    arma::vec R = get_R(R0, n);
    return arma::spsolve(S, R);
}
template<bool duplicate = true, bool black = false>
arma::mat potential_impl::poisson2D(const voltage & V, const charge_density & n) {
    using namespace arma;

    vec R0 = get_R0(V);
    vec R = get_R(R0, n);
    vec phivec = spsolve(S, R);
    mat phimat(d::N_x, d::M_r);

    uword k = 0;
    for (uword j = 0; j < d::M_r; ++j) {
        for (uword i = 0; i < d::N_x; ++i) {
            if ((j >= d::M_cnt) && (i < d::N_sc)) {
                phimat(i, j) = black ? -2 : -(V.s + d::F_s);
            } else if ((j >= d::M_cnt) && (i >= d::N_x - d::N_dc)) {
                phimat(i, j) = black ? -2 : -(V.d + d::F_d);
            } else if ((j >= d::M_cnt + d::M_ox) && (i >= d::N_sc + d::N_s + d::N_sox) && (i < d::N_sc + d::N_s + d::N_sox + d::N_g)) {
                phimat(i, j) = black ? -2 : -(V.g + d::F_g);
            } else {
                phimat(i, j) = phivec(k++);
            }
        }
    }

    if (duplicate) {
        phimat = join_horiz(fliplr(phimat), phimat);
    }

    return phimat;
}

//std::vector<double> potential_impl::get_boxes(std::vector<int> & ibox, std::vector<int> & jbox) {
//    std::vector<double> eps_box;

//    // the following defines the device's 2D wrap-gate geometry
//    ibox.resize(7); // 7 different regions in lateral direction
//    jbox.resize(3); // 3 different regions in radial direction
//    eps_box.resize(7 * 3);

//    ibox[0] =           d::N_sc  * 2;
//    ibox[1] = ibox[0] + d::N_s   * 2 - 1;
//    ibox[2] = ibox[1] + d::N_sox * 2 + 2;
//    ibox[3] = ibox[2] + d::N_g   * 2 - 1;
//    ibox[4] = ibox[3] + d::N_dox * 2 + 2;
//    ibox[5] = ibox[4] + d::N_d   * 2 - 1;
//    ibox[6] = ibox[5] + d::N_dc  * 2;

//    jbox[0] =           d::M_cnt * 2;
//    jbox[1] = jbox[0] + d::M_ox  * 2;
//    jbox[2] = jbox[1] + d::M_ext * 2 + 1;

//    int j = 0; // nanotube region
//    for (int i = 0; i < 7; ++i) {
//        eps_box[j * 7 + i] = d::eps_cnt * c::eps_0;
//    }
//    j = 1; // gate-oxide region
//    for (int i = 0; i < 2; ++i) {
//        eps_box[j * 7 + i] = c::eps_0;
//    }
//    for (int i = 2; i < 5; ++i) {
//        eps_box[j * 7 + i] = d::eps_ox * c::eps_0;
//    }
//    for (int i = 5; i < 7; ++i) {
//        eps_box[j * 7 + i] = c::eps_0;
//    }
//    j = 2; // gate-contact/extended region
//    for (int i = 0; i < 7; ++i) {
//        eps_box[j * 7 + i] = c::eps_0;
//    }

//    return eps_box;
//}

void potential_impl::get_eps_mat(arma::mat eps[4]) {
    for (int i = 0; i < 4; ++i) {
        eps[i] = arma::mat(d::N_x, d::M_r);
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
    for (j = 0; j < d::M_cnt - 1; ++j) {
        for (i = 0; i < d::N_x; ++i) {
            eps[L](i, j) = d::eps_cnt * c::eps_0;
            eps[R](i, j) = d::eps_cnt * c::eps_0;
            eps[I](i, j) = d::eps_cnt * c::eps_0;
            eps[O](i, j) = d::eps_cnt * c::eps_0;
        }
    }

    // cnt border
    for (i = 0; i < d::N_sc + d::N_s - 1; ++i) {
        eps[L](i, j) = 0.5 * (1.0 + d::eps_cnt) * c::eps_0;
        eps[R](i, j) = 0.5 * (1.0 + d::eps_cnt) * c::eps_0;
        eps[I](i, j) = d::eps_cnt * c::eps_0;
        eps[O](i, j) = c::eps_0;
    }
    eps[L](i, j) = 0.5 * (1.0 + d::eps_cnt) * c::eps_0;
    eps[R](i, j) = 0.5 * (d::eps_ox + d::eps_cnt) * c::eps_0;
    eps[I](i, j) = d::eps_cnt * c::eps_0;
    eps[O](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
    for (++i; i < d::N_x - d::N_dc - d::N_d; ++i) {
        eps[L](i, j) = 0.5 * (d::eps_ox + d::eps_cnt) * c::eps_0;
        eps[R](i, j) = 0.5 * (d::eps_ox + d::eps_cnt) * c::eps_0;
        eps[I](i, j) = d::eps_cnt * c::eps_0;
        eps[O](i, j) = d::eps_ox * c::eps_0;
    }
    eps[L](i, j) = 0.5 * (d::eps_ox + d::eps_cnt) * c::eps_0;
    eps[R](i, j) = 0.5 * (1.0 + d::eps_cnt) * c::eps_0;
    eps[I](i, j) = d::eps_cnt * c::eps_0;
    eps[O](i, j) = 0.5  * (1.0 + d::eps_ox) * c::eps_0;
    for (++i; i < d::N_x; ++i) {
        eps[L](i, j) = 0.5 * (1.0 + d::eps_cnt) * c::eps_0;
        eps[R](i, j) = 0.5 * (1.0 + d::eps_cnt) * c::eps_0;
        eps[I](i, j) = d::eps_cnt * c::eps_0;
        eps[O](i, j) = c::eps_0;
    }

    // gate oxide
    for (j = d::M_cnt; j < d::M_cnt + d::M_ox - 1; ++j) {
        i = d::N_sc + d::N_s - 1;
        eps[L](i, j) = c::eps_0;
        eps[R](i, j) = d::eps_ox * c::eps_0;
        eps[I](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
        eps[O](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
        for (++i; i < d::N_x - d::N_dc - d::N_d; ++i) {
            eps[L](i, j) = d::eps_ox * c::eps_0;
            eps[R](i, j) = d::eps_ox * c::eps_0;
            eps[I](i, j) = d::eps_ox * c::eps_0;
            eps[O](i, j) = d::eps_ox * c::eps_0;
        }
        eps[L](i, j) = d::eps_ox * c::eps_0;
        eps[R](i, j) = c::eps_0;
        eps[I](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
        eps[O](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
    }

    // gate oxide border
    i = d::N_sc + d::N_s - 1;
    eps[L](i, j) = c::eps_0;
    eps[R](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
    eps[I](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
    eps[O](i, j) = c::eps_0;
    for (++i; i < d::N_x - d::N_dc - d::N_d; ++i) {
        eps[L](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
        eps[R](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
        eps[I](i, j) = d::eps_ox * c::eps_0;
        eps[O](i, j) = c::eps_0;
    }
    eps[L](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
    eps[R](i, j) = c::eps_0;
    eps[I](i, j) = 0.5 * (1.0 + d::eps_ox) * c::eps_0;
    eps[O](i, j) = c::eps_0;
}

template<int dir> // the direction of the surface normal vector (lrio)
double potential_impl::eps(int i, int j) {
    static arma::mat eps[4];
    static bool eps_initialized = false;

    if (!eps_initialized) {
        get_eps_mat(eps);
        eps_initialized = true;
    }

    return eps[dir](i, j);
//    // returns the correct dielectric constant for a given set of lattice-points
//    static std::vector<int> ibox;
//    static std::vector<int> jbox;
//    static std::vector<double> eps_box = get_boxes(ibox, jbox);
//    static int N_i = ibox.size();
//    static int N_j = jbox.size();

//    enum {
//        L = 0, //left
//        R = 1, //right
//        I = 2, //inside  (down)
//        O = 3  //outside (up)
//    };

//    // move to adjacent lattice points according to value of <dir>
//    int i2 = i * 2 + 1 + ((dir == R) ? 1 : 0) + ((dir == L) ? -1 : 0);
//    int j2 = j * 2 + 1 + ((dir == O) ? 1 : 0) + ((dir == I) ? -1 : 0);

//    // linear search (not many elements, so binary search not necessary)
//    int ki;
//    for (ki = 0; ki < N_i; ++ki) {
//        if (i2 < ibox[ki]) {
//            break;
//        }
//    }
//    int kj;
//    for (kj = 0; kj < N_j; ++kj) {
//        if (j2 < jbox[kj]) {
//            break;
//        }
//    }

//    return eps_box[kj * N_i + ki];
}

arma::sp_mat potential_impl::get_S() {
    using namespace arma;
    static constexpr double dr2 = 1.0 / d::dr / d::dr;
    static constexpr double dx2 = 1.0 / d::dx / d::dx;

    enum {
        L = 0,
        R = 1,
        I = 2,
        O = 3
    };

    umat indices(2, d::M_r * d::N_x * 5);
    vec values(d::M_r * d::N_x * 5);

    uword N_v = 0;
    auto set_value = [&] (uword ki, uword kj, double value) {
        indices(0, N_v) = ki;
        indices(1, N_v) = kj;
        values(N_v++) = value;
    };

    uword k = 0;          // current main diagonal element
    uword i0 = 0;         // left i limit
    uword i1 = d::N_x;    // right i limit
    uword delta = d::N_x; // distance to next vertical coupling off diagonal

    for (uword j = 0; j < d::M_r; ++j) {
        double r = j * d::dr + 0.5 * d::dr;
        double rm = r - 0.5 * d::dr;
        double rp = r + 0.5 * d::dr;

        for (uword i = i0; i < i1; ++i) {
            double diag    = - dx2 * (r  * eps<L>(i, j) + r  * eps<R>(i, j))
                             - dr2 * (rp * eps<O>(i, j) + rm * eps<I>(i, j));
            double left    = dx2 * r  * eps<L>(i    , j);
            double right   = (i > i0) ? dx2 * r  * eps<R>(i - 1, j) : 0;
            double inside  = dr2 * rm * eps<I>(i, j    );
            double outside = (j > 0) ? dr2 * rm * eps<O>(i, j - 1) : 0;

            // horizontal von Neumann boundary conditions
            if (i == 1) {
                right *= 2;
            }
            if (i == d::N_x - 1) {
                left *= 2;
            }

            // vertical von Neumann boundary conditions
            if (j == 1) {
                outside -= dr2 * 0.5 * d::dr * eps<O>(i, 0);
                outside *= 2;
            }
            if (j == d::M_r - 1) {
                inside += dr2 * 0.5 * d::dr * eps<I>(i, d::M_r - 1);
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
        if (j == d::M_cnt - 1) {
            i0 = d::N_sc;
            i1 = d::N_x - d::N_dc;
            delta = d::N_x - d::N_sc;
        }
        if (j == d::M_cnt) {
            delta = d::N_x - d::N_sc - d::N_dc;
        }

        // cut off gate contact
        if (j == d::M_cnt + d::M_ox - 1) {
            i1 = i0 + d::N_s + d::N_sox;
        }
        if ((j == d::M_cnt + d::M_ox) && (i0 == d::N_sc)) {
            delta = d::N_x - d::N_sc - d::N_dc - d::N_g;
        }
        if (j >= d::M_cnt + d::M_ox) {
            if (i0 == d::N_sc) {
                i0 = i1 + d::N_g;
                i1 = d::N_x - d::N_dc;
                --j; // repeat i loop for second part (right side of gate)
            } else {
                i1 = i0 - d::N_g;
                i0 = d::N_sc;
            }
        }
    }

    indices.resize(2, N_v);
    values.resize(N_v);
    return sp_mat(indices, values);
}

arma::vec potential_impl::get_R0(const voltage & V) {
    using namespace arma;
    static constexpr double dr2 = 1.0 / d::dr / d::dr;
    static constexpr double dx2 = 1.0 / d::dx / d::dx;

    enum {
        L = 0,
        R = 1,
        I = 2,
        O = 3
    };

    double V_s = - (V.s + d::F_s);
    double V_g = - (V.g + d::F_g);
    double V_d = - (V.d + d::F_d);

    vec R0(d::N_x * d::M_r);
    R0.fill(0);
    uword i, j, k;
    double r, rp;

    j = d::M_cnt - 1;
    r = j * d::dr + 0.5 * d::dr;
    rp = r + 0.5 * d::dr;

    k = j * d::N_x;
    for (i = 0; i < d::N_sc; ++i) {
        R0(k++) -= dr2 * rp * eps<O>(i, j) * V_s;
    }
    k = j * d::N_x + d::N_x - d::N_dc;
    for (i = d::N_x - d::N_dc; i < d::N_x; ++i) {
        R0(k++) -= dr2 * rp * eps<O>(i, j) * V_d;
    }
    for (j = d::M_cnt; j < d::M_cnt + d::M_ox - 1; ++j) {
        r = j * d::dr + 0.5 * d::dr;
        R0(k) -= dx2 * r * eps<L>(d::N_sc, j) * V_s;
        k += d::N_s + d::N_sox + d::N_g + d::N_dox + d::N_d - 1;
        R0(k++) -= dx2 * r * eps<R>(d::N_x - d::N_dc - 1, j) * V_d;
    }
    r = j * d::dr + 0.5 * d::dr;
    rp = r + 0.5 * d::dr;
    R0(k) -= dx2 * r * eps<L>(d::N_sc, j) * V_s;
    k += d::N_s + d::N_sox;
    for (i = d::N_sc + d::N_s + d::N_sox; i < d::N_sc + d::N_s + d::N_sox + d::N_g; ++i) {
        R0(k++) -= dr2 * rp * eps<O>(i, j) * V_g;
    }
    k += d::N_dox + d::N_d - 1;
    R0(k++) -= dx2 * r * eps<R>(d::N_x - d::N_dc - 1, j) * V_d;
    for (j = d::M_cnt + d::M_ox; j < d::M_r; ++j) {
        r = j * d::dr + 0.5 * d::dr;
        R0(k) -= dx2 * r * eps<L>(d::N_sc, j) * V_s;
        k += d::N_s + d::N_sox - 1;
        R0(k++) -= dx2 * r * eps<R>(d::N_sc + d::N_s + d::N_sox - 1, j) * V_g;
        R0(k) -= dx2 * r * eps<L>(d::N_sc + d::N_s + d::N_sox + d::N_g, j) * V_g;
        k += d::N_dox + d::N_d - 1;
        R0(k++) -= dx2 * r * eps<R>(d::N_x - d::N_dc - 1, j) * V_d;
    }

    R0.resize(k);
    return R0;
}

arma::vec potential_impl::get_R(const arma::vec & R0, const charge_density & n) {
    arma::vec R = R0;
    R({(d::M_cnt - 1) * d::N_x, d::M_cnt * d::N_x - 1}) += n.data * d::r_cnt * 1e9; // 10^9 because of m->nm in epsilon_0
    return R;
}

void plot_phi2D(const voltage & V, const charge_density & n) {
    arma::mat phi2D = potential_impl::poisson2D<true, false>(V, n).t();

    gnuplot gp;

    gp << "set palette defined ( 0 '#D73027', 1 '#F46D43', 2 '#FDAE61', 3 '#FEE090', 4 '#E0F3F8', 5 '#ABD9E9', 6 '#74ADD1', 7 '#4575B4' )\n";
    gp << "set title \"2D Potential\"\n";
    gp << "set xlabel \"x / nm\"\n";
    gp << "set ylabel \"r / nm\"\n";
    gp << "set zlabel \"Phi / V\"\n";
    gp << "unset key\n";

    using namespace d;

    // indicate cnt area
    gp << "set obj rect from " << 0 << "," << r_cnt << " to " << l << "," << -r_cnt << "front fillstyle empty\n";
    gp << "set label \"CNT\" at " << 0.5 * l << "," << 0 << " center front\n";

    // indicate oxide area
    double x0 = l_sc + l_s;
    double x1 = l - l_dc - l_d;
    double y0 = r_cnt + d_ox;
    double y1 = r_cnt;
    gp << "set obj rect from " << x0 << "," << y0 << " to " << x1 << "," << y1 << "front fillstyle empty\n";
    gp << "set label \"gate oxide\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * (y1 - y0) - y1<< " center front\n";
    gp << "set obj rect from " << x0 << "," << -y1 << " to " << x1 << "," << -y0 << "front fillstyle empty\n";
    gp << "set label \"gate oxide\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";

    // indicate gate contact area
    x0 = l_sc + l_s + l_sox;
    x1 = l - l_dc - l_d - l_dox;
    y0 = R;
    y1 = r_cnt + d_ox;
    gp << "set obj rect from " << x0 << "," << y0 << " to " << x1 << "," << y1 << "front fillstyle empty\n";
    gp << "set label \"gate contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * (y1 - y0) - y1<< " center front\n";
    gp << "set obj rect from " << x0 << "," << -y1 << " to " << x1 << "," << -y0 << "front fillstyle empty\n";
    gp << "set label \"gate contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";

    // indicate left contact area
    x0 = 0;
    x1 = l_sc;
    y0 = R;
    y1 = r_cnt;
    gp << "set obj rect from " << x0 << "," << y0 << " to " << x1 << "," << y1 << "front fillstyle empty\n";
    gp << "set label \"source contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * (y1 - y0) - y1<< " center front\n";
    gp << "set obj rect from " << x0 << "," << -y1 << " to " << x1 << "," << -y0 << "front fillstyle empty\n";
    gp << "set label \"source contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";

    // indicate left contact area
    x0 = l - l_dc;
    x1 = l;
    y0 = R;
    y1 = r_cnt;
    gp << "set obj rect from " << x0 << "," << y0 << " to " << x1 << "," << y1 << "front fillstyle empty\n";
    gp << "set label \"drain contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * (y1 - y0) - y1<< " center front\n";
    gp << "set obj rect from " << x0 << "," << -y1 << " to " << x1 << "," << -y0 << "front fillstyle empty\n";
    gp << "set label \"drain contact\" at " << 0.5 * (x1 - x0) + x0 << "," << 0.5 * -(y1 - y0) + y1 << " center front\n";

    gp.set_background(x, arma::join_vert(arma::flipud(-r), r), phi2D);
    gp.plot();
}

#endif

