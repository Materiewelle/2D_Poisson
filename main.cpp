//#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS

#include <iostream>
#include <algorithm>
#include <omp.h>
#include <xmmintrin.h>

#include <armadillo>

#include "device.hpp"
#include "gnuplot.hpp"

using namespace arma;
using namespace std;

static inline std::vector<double> get_boxes(std::vector<int> & ibox, std::vector<int> & jbox) {
    std::vector<double> eps_box;
    ibox.resize(7);
    jbox.resize(3);
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

    int j = 0;
    for (int i = 0; i < 7; ++i) {
        eps_box[j * 7 + i] = d::eps_cnt * c::eps_0;
    }
    j = 1;
    for (int i = 0; i < 2; ++i) {
        eps_box[j * 7 + i] = c::eps_0;
    }
    for (int i = 2; i < 5; ++i) {
        eps_box[j * 7 + i] = d::eps_ox * c::eps_0;
    }
    for (int i = 5; i < 7; ++i) {
        eps_box[j * 7 + i] = c::eps_0;
    }
    j = 2;
    for (int i = 0; i < 7; ++i) {
        eps_box[j * 7 + i] = c::eps_0;
    }

    return eps_box;
}

template<int dir>
static inline double eps(int i, int j) {
    static std::vector<int> ibox;
    static std::vector<int> jbox;
    static std::vector<double> eps_box = get_boxes(ibox, jbox);
    static int N_i = ibox.size();
    static int N_j = jbox.size();

    enum {
        L = 0,
        R = 1,
        I = 2,
        O = 3
    };

    int i2 = i * 2 + 1 + ((dir == R) ? 1 : 0) + ((dir == L) ? -1 : 0);
    int j2 = j * 2 + 1 + ((dir == O) ? 1 : 0) + ((dir == I) ? -1 : 0);

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

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    static constexpr double dr2 = 1.0 / d::dr / d::dr;
    static constexpr double dx2 = 1.0 / d::dx / d::dx;

    enum {
        L = 0,
        R = 1,
        I = 2,
        O = 3
    };

//    mat epss[4];
//    for (int dir = 0; dir < 4; ++dir) {
//        epss[dir] = mat(d::M_r, d::N_x);
//        for (int j = 0; j < d::M_r; ++j) {
//            for (int i = 0; i < d::N_x; ++i) {
//                epss[dir](j, i) = eps(dir, i, j);
//            }
//        }
//        image(epss[dir]);
//    }
//    return 0;


    double V_s = -(0.0 + d::F_s);
    double V_g = -(1.0 + d::F_g);
    double V_d = -(0.0 + d::F_d);

    // construct S matrix
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

    // horizontal von Neumann
    S(0, 1) *= 2;
    for (j = 1; j < d::M_cnt; ++j) {
        int k = j * d::N_x;
        S(k - 1, k - 2) *= 2;
        S(k, k + 1) *= 2;
    }
    S(d::M_cnt * d::N_x - 1, d::M_cnt * d::N_x - 2) *= 2;

    // vertical von Neumann
    for (i = 0; i < d::N_x; ++i) {
        S(i, d::N_x + i) -= dr2 * 0.5 * d::dr * eps<O>(i, 0);
        S(i, d::N_x + i) *= 2;
        S(D - d::N_x + i, D - d::N_x * 2 + i) += dr2 * 0.5 * d::dr * eps<I>(i, d::M_r - 1);
        S(D - d::N_x + i, D - d::N_x * 2 + i) *= 2;
    }

    // right side vector
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

    // copy non metal parts S => S1, T => T1
    uword N_ssox = d::N_s + d::N_sox;
    uword N_ddox = d::N_d + d::N_dox;
    uword N_ox   = N_ssox + d::N_g + N_ddox;
    D = d::N_x * d::M_cnt + N_ox * d::M_ox + (N_ssox + N_ddox) * d::M_ext;
    sp_mat S1 = sp_mat(D, D);
    vec T1 = vec(D);

    // cnt part
    uword k0 = 0;
    uword k1 = d::N_x * d::M_cnt - 1;
    S1({k0, k1}, {k0, k1}) = S({k0, k1}, {k0, k1});
    T1({k0, k1}) = T({k0, k1});

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
        T1({k0, k1}) = T({c, c + N_ox - 1});
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
        T1({k0, k1}) = T({c1, c1 + N_ssox - 1});
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
        T1({k0, k1}) = T({c2, c2 + N_ddox - 1});
        k0 = k1 + 1;
        k1 += N_ddox;
    }

//    gnuplot gp;
//    gp << "set terminal wxt size 800, 800" << endl;
//    gp << "unset colorbox" << endl;
//    gp.set_background(flipud(mat(S)));
//    gp.plot();

//    gnuplot gp2;
//    gp2 << "set terminal wxt size 800, 800" << endl;
//    gp2 << "unset colorbox" << endl;
//    gp2.set_background(flipud(mat(S1)));
//    gp2.plot();

    cout << "setup done" << endl;
    wall_clock timer;
    timer.tic();
    vec phivec1 = spsolve(S1, T1);
    cout << timer.toc() << endl;

    vec phivec = vec(d::N_x * d::M_r);
    phivec.fill(0);
    k0 = 0;
    k1 = d::N_x * d::M_cnt - 1;
    phivec({k0, k1}) = phivec1({k0, k1});

    k0 = k1 + 1;
    k1 += N_ox;
    for (j = d::M_cnt; j < d::M_cnt + d::M_ox; ++j) {
        uword c =  j * d::N_x + d::N_sc;
        phivec({c, c + N_ox - 1}) = phivec1({k0, k1});
        k0 = k1 + 1;
        k1 += N_ox;
    }

    k1 -= N_ox;
    k0 = k1 + 1;
    k1 += N_ssox;
    for (j = d::M_cnt + d::M_ox; j < d::M_r; ++j) {
        uword c1 =  j    * d::N_x + d::N_sc;
        phivec({c1, c1 + N_ssox - 1}) = phivec1({k0, k1});
        k0 = k1 + 1;
        k1 += N_ssox;

        uword c2 = (j+1) * d::N_x - d::N_sc - N_ddox;
        S1({k0, k1},{k0, k1}) = S({c2, c2 + N_ddox - 1},{c2, c2 + N_ddox - 1});
        phivec({c2, c2 + N_ddox - 1}) = phivec1({k0, k1});
        k0 = k1 + 1;
        k1 += N_ddox;
    }

    mat phi(phivec);
    phi.reshape(d::N_x, d::M_r);

    image(phi.t());

    vec phi_extract = phi.row(d::N_x / 2).t();
    phi_extract = phi_extract({0, uword(d::M_cnt + d::M_ox - 1)});
    plot(phi_extract);
    cout << phi_extract << endl;

    return 0;
}
