#ifndef POTENTIAL_HPP
#define POTENTIAL_HPP

#include <armadillo>

#include "anderson.hpp"
#include "voltage.hpp"

// forward declarations
#ifndef CHARGE_DENSITY_HPP
class charge_density;
#endif

class potential {
public:
    arma::vec data;
    arma::vec twice;

    inline potential();
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

    static inline std::vector<double> get_boxes(std::vector<int> & ibox, std::vector<int> & jbox);
    template<int dir>
    static inline double eps(int i, int j);
    static inline arma::sp_mat get_S();

    static inline arma::vec get_R0(const voltage & V);
    static inline arma::vec get_R(const arma::vec & R0, const charge_density & n);

    static const arma::sp_mat S = get_S();
}

//----------------------------------------------------------------------------------------------------------------------

potential::potential() {
}

potential::potential(const arma::vec & R0, const charge_density & n)
    : twice(d::N_x * 2) {
    using namespace potential_impl;

    data = poisson(R0, n);
    update_twice();
}

double potential::update(const arma::vec & R0, const charge_density & n, anderson & mr_neo) {
    using namespace arma;
    using namespace potential_impl;

    vec f = poisson(R0, n) - data;

    // anderson mixing
    mr_neo.update(data, f);

    update_twice();

    // return dphi
    return max(abs(f));
}

void potential::smooth() {
    smooth<(d::F_s > 0)>(0, d::N_sc * 0.3);

    // smooth source region
    smooth<(d::F_s > 0)>(d::N_sc + 0.3 * d::N_s, d::N_sc + d::N_s + d::N_g * 0.05);

    // smooth drain region
    smooth<(d::F_d > 0)>(d::N_sc + d::N_s + d::N_g * 0.95, d::N_x - (d::N_dc + 0.3 * d::N_d));

    smooth<(d::F_d > 0)>(d::N_x - 0.3 * d::N_dc, d::N_x);

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
    arma::vec phi2D = arma::spsolve(potential_impl::S, R);
    return phi2D({(d::M_cnt - 1) * d::N_x, d::M_cnt * d::N_x - 1});
}

std::vector<double> potential_impl::get_boxes(std::vector<int> & ibox, std::vector<int> & jbox) {
    std::vector<double> eps_box;

    // the following defines the device's 2D wrap-gate geometry
    ibox.resize(7); // 7 different regions in lateral direction
    jbox.resize(3); // 3 different regions in radial direction
    eps_box.resize(7 * 3);

    ibox[0] =           d::N_sc  * 2;
    ibox[1] = ibox[0] + d::N_s   * 2;
    ibox[2] = ibox[1] + d::N_sox * 2 - 1;
    ibox[3] = ibox[2] + d::N_g   * 2 + 3;
    ibox[4] = ibox[3] + d::N_dox * 2 - 1;
    ibox[5] = ibox[4] + d::N_d   * 2;
    ibox[6] = ibox[5] + d::N_dc  * 2;

    jbox[0] =           d::M_cnt * 2;
    jbox[1] = jbox[0] + d::M_ox  * 2;
    jbox[2] = jbox[1] + d::M_ext * 2 + 1;

    int j = 0; // nanotube region
    for (int i = 0; i < 7; ++i) {
        eps_box[j * 7 + i] = d::eps_cnt * c::eps_0;
    }
    j = 1; // gate-oxide region
    for (int i = 0; i < 2; ++i) {
        eps_box[j * 7 + i] = c::eps_0;
    }
    for (int i = 2; i < 5; ++i) {
        eps_box[j * 7 + i] = d::eps_ox * c::eps_0;
    }
    for (int i = 5; i < 7; ++i) {
        eps_box[j * 7 + i] = c::eps_0;
    }
    j = 2; // gate-contact/extended region
    for (int i = 0; i < 7; ++i) {
        eps_box[j * 7 + i] = c::eps_0;
    }

    return eps_box;
}

template<int dir> // the direction of the surface normal vector (lrio)
double potential_impl::eps(int i, int j) {
    // returns the correct dielectric constant for a given set of lattice-points
    static std::vector<int> ibox;
    static std::vector<int> jbox;
    static std::vector<double> eps_box = get_boxes(ibox, jbox);
    static int N_i = ibox.size();
    static int N_j = jbox.size();

    enum {
        L = 0, //left
        R = 1, //right
        I = 2, //inside  (down)
        O = 3  //outside (up)
    };

    // move to adjacent lattice points according to value of <dir>
    int i2 = i * 2 + 1 + ((dir == R) ? 1 : 0) + ((dir == L) ? -1 : 0);
    int j2 = j * 2 + 1 + ((dir == O) ? 1 : 0) + ((dir == I) ? -1 : 0);

    // linear search (not many elements, so binary search not necessary)
    int ki;
    for (ki = 0; ki < N_i; ++ki) {
        if (i2 < ibox[ki]) {
            break;
        }
    }
    int kj;
    for (kj = 0; kj < N_j; ++kj) {
        if (j2 < jbox[kj]) {
            break;
        }
    }

    return eps_box[kj * N_i + ki];
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

    // construct S matrix
    std::cout << "Constructing S-matrix...";
    uword D = d::N_x * d::M_r;
    uword i, j, k;
    double r, rp, rm;
    sp_mat S(D, D);
    for (j = 0; j < d::M_r; ++j) {
        r = j * d::dr + 0.5 * d::dr;
        rp = r + 0.5 * d::dr;
        rm = r - 0.5 * d::dr;

        for (i = 0; i < d::N_x; ++i) {
            k = j * d::N_x + i;

            if (k >= d::N_x) {
                S(k, k - d::N_x) = dr2 * rm * eps<I>(i, j);
            }
            if (k >= 1) {
                S(k, k - 1) = dx2 * r * eps<L>(i, j);
            }
            S(k, k) = - dx2 * (r  * eps<L>(i, j) + r  * eps<R>(i, j))
                      - dr2 * (rp * eps<O>(i, j) + rm * eps<I>(i, j));
            if (k < D - 1) {
                S(k, k + 1) = dx2 * r * eps<R>(i, j);
            }
            if (k < D - d::N_x) {
                S(k, k + d::N_x) = dr2 * rp * eps<O>(i, j);
            }
        }
    }

    // remove coupling between end of j-th line and start of (j+1)-th line
    for (j = 1; j < d::M_r; ++j) {
        int k = j * d::N_x;
        S(k, k - 1) = 0;
        S(k - 1, k) = 0;
    }

    // horizontal von Neumann boundary conditions
    S(0, 1) *= 2;
    for (j = 1; j < d::M_cnt; ++j) {
        int k = j * d::N_x;
        S(k - 1, k - 2) *= 2;
        S(k, k + 1) *= 2;
    }
    S(d::M_cnt * d::N_x - 1, d::M_cnt * d::N_x - 2) *= 2;

    // vertical von Neumann boundary conditions
    for (i = 0; i < d::N_x; ++i) {
        S(i, d::N_x + i) -= dr2 * 0.5 * d::dr * eps<O>(i, 0);
        S(i, d::N_x + i) *= 2;
        S(D - d::N_x + i, D - d::N_x * 2 + i) += dr2 * 0.5 * d::dr * eps<I>(i, d::M_r - 1);
        S(D - d::N_x + i, D - d::N_x * 2 + i) *= 2;
    }

    // cut out the relevant parts
    uword N_ssox = d::N_s + d::N_sox;
    uword N_ddox = d::N_d + d::N_dox;
    uword N_ox   = N_ssox + d::N_g + N_ddox;
    D = d::N_x * d::M_cnt + N_ox * d::M_ox + (N_ssox + N_ddox) * d::M_ext;
    sp_mat S1 = sp_mat(D, D);

    // cnt part
    uword k0 = 0;
    uword k1 = d::N_x * d::M_cnt - 1;
    S1({k0, k1}, {k0, k1}) = S({k0, k1}, {k0, k1});

    // oxide part
    k0 = k1 + 1;
    k1 += N_ox;
    uword delta = d::N_dc; // offset for first left/up blocks
    for (j = d::M_cnt; j < d::M_cnt + d::M_ox; ++j) {
        uword l = (j-1) * d::N_x + d::N_sc;
        uword c =  j    * d::N_x + d::N_sc;
        auto left   = S({c, c + N_ox - 1}, {l, l + N_ox - 1});
        auto center = S({c, c + N_ox - 1}, {c, c + N_ox - 1});
        auto up     = S({l, l + N_ox - 1}, {c, c + N_ox - 1});
        S1({k0, k1},{k0 - N_ox - delta, k1 - N_ox - delta}) = left;
        S1({k0, k1},{k0, k1}) = center;
        S1({k0 - N_ox - delta, k1 - N_ox - delta},{k0, k1}) = up;
        k0 = k1 + 1;
        k1 += N_ox;
        delta = 0; // set offset to 0 for next blocks
    }

    // extension part
    k1 -= N_ox;
    k0 = k1 + 1;
    k1 += N_ssox;
    delta = d::N_g; // offset for first left/up blocks
    for (j = d::M_cnt + d::M_ox; j < d::M_r; ++j) {
        uword l1 = (j-1) * d::N_x + d::N_sc;
        uword c1 =  j    * d::N_x + d::N_sc;
        auto left1   = S({c1, c1 + N_ssox - 1},{l1, l1 + N_ssox - 1});
        auto center1 = S({c1, c1 + N_ssox - 1},{c1, c1 + N_ssox - 1});
        auto up1     = S({l1, l1 + N_ssox - 1},{c1, c1 + N_ssox - 1});
        S1({k0, k1},{k0 - N_ssox - N_ddox - delta, k1 - N_ssox - N_ddox - delta}) = left1;
        S1({k0, k1},{k0, k1}) = center1;
        S1({k0 - N_ssox - N_ddox - delta, k1 - N_ssox - N_ddox - delta},{k0, k1}) = up1;
        k0 = k1 + 1;
        k1 += N_ssox;
        delta = 0; // set offset to 0 for next blocks

        uword l2 =  j    * d::N_x - d::N_sc - N_ddox;
        uword c2 = (j+1) * d::N_x - d::N_sc - N_ddox;
        auto left2   = S({c2, c2 + N_ddox - 1},{l2, l2 + N_ddox - 1});
        auto center2 = S({c2, c2 + N_ddox - 1},{c2, c2 + N_ddox - 1});
        auto up2     = S({l2, l2 + N_ddox - 1},{c2, c2 + N_ddox - 1});
        S1({k0, k1},{k0 - N_ssox - N_ddox, k1 - N_ssox - N_ddox}) = left2;
        S1({k0, k1},{k0, k1}) = center2;
        S1({k0 - N_ssox - N_ddox, k1 - N_ssox - N_ddox},{k0, k1}) = up2;
        k0 = k1 + 1;
        k1 += N_ddox;
    }

    std::cout << " done!\n";
    return S1;
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

    const double V_s = -(V.s + d::F_s);
    const double V_g = -(V.g + d::F_g);
    const double V_d = -(V.d + d::F_d);

    // right side vector
    uword D = d::N_x * d::M_r;
    uword i, j, k;
    double r, rp;
    vec T(D);
    T.fill(0);

    // horizontal Dirichlet
    i = d::N_sc;
    for (j = d::M_cnt; j < d::M_r; ++j) {
        r = j * d::dr + 0.5 * d::dr;
        k = j * d::N_x + i;
        T(k) -= dx2 * r * eps<L>(i, j) * V_s;
    }
    i = d::N_x - d::N_dc - 1;
    for (j = d::M_cnt; j < d::M_r; ++j) {
        r = j * d::dr + 0.5 * d::dr;
        k = j * d::N_x + i;
        T(k) -= dx2 * r * eps<R>(i, j) * V_d;
    }
    i = d::N_sc + d::N_s + d::N_sox - 1;
    for (j = d::M_cnt + d::M_ox; j < d::M_r; ++j) {
        r = j * d::dr + 0.5 * d::dr;
        k = j * d::N_x + i;
        T(k) -= dx2 * r * eps<R>(i, j) * V_g;
    }
    i = d::N_sc + d::N_s + d::N_sox + d::N_g;
    for (j = d::M_cnt + d::M_ox; j < d::M_r; ++j) {
        r = j * d::dr + 0.5 * d::dr;
        k = j * d::N_x + i;
        T(k) -= dx2 * r * eps<L>(i, j) * V_g;
    }

    // vertical Dirichlet
    j = d::M_cnt - 1;
    r = j * d::dr + 0.5 * d::dr;
    rp = r + 0.5 * d::dr;
    for (i = 0; i < d::N_sc; ++i) {
        k = j * d::N_x + i;
        T(k) -= dr2 * rp * eps<O>(i, j) * V_s;
    }
    for (i = d::N_x - d::N_dc; i < d::N_x; ++i) {
        k = j * d::N_x + i;
        T(k) -= dr2 * rp * eps<O>(i, j) * V_d;
    }
    j = d::M_cnt + d::M_ox - 1;
    r = j * d::dr + 0.5 * d::dr;
    rp = r + 0.5 * d::dr;
    for (i = d::N_sc + d::N_s + d::N_sox; i < d::N_sc + d::N_s + d::N_sox + d::N_g; ++i) {
        k = j * d::N_x + i;
        T(k) -= dr2 * rp * eps<O>(i, j) * V_g;
    }

    // copy non metal parts T => R0
    uword N_ssox = d::N_s + d::N_sox;
    uword N_ddox = d::N_d + d::N_dox;
    uword N_ox   = N_ssox + d::N_g + N_ddox;
    D = d::N_x * d::M_cnt + N_ox * d::M_ox + (N_ssox + N_ddox) * d::M_ext;
    vec R0 = vec(D);

    // cnt part
    uword k0 = 0;
    uword k1 = d::N_x * d::M_cnt - 1;
    R0({k0, k1}) = T({k0, k1});

    // oxide part
    k0 = k1 + 1;
    k1 += N_ox;
    for (j = d::M_cnt; j < d::M_cnt + d::M_ox; ++j) {
        uword c =  j    * d::N_x + d::N_sc;
        R0({k0, k1}) = T({c, c + N_ox - 1});
        k0 = k1 + 1;
        k1 += N_ox;
    }

    // extension part
    k1 -= N_ox;
    k0 = k1 + 1;
    k1 += N_ssox;
    for (j = d::M_cnt + d::M_ox; j < d::M_r; ++j) {
        k0 = k1 + 1;
        k1 += N_ssox;
        uword c2 = (j+1) * d::N_x - d::N_sc - N_ddox;
        R0({k0, k1}) = T({c2, c2 + N_ddox - 1});
        k0 = k1 + 1;
        k1 += N_ddox;
    }

    return R0;
}

arma::vec potential_impl::get_R(const arma::vec & R0, const charge_density & n) {
    arma::vec R = R0;
    R({(d::M_cnt - 1) * d::N_x, d::M_cnt * d::N_x - 1}) += n.data * d::r_cnt; // TODO: scaling ????
    return R;
}

#endif

