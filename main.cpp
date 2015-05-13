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

template<int dir>
static inline double eps(int i, int j) {
    return c::eps_0;
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
        A = 2,
        I = 3
    };

    int D = d::N_x * d::M_r;
    sp_mat S(D, D);

    for (int j = 0; j < d::M_r; ++j) {
        double r = j * d::dr + 0.5 * d::dr;
        double rp = r + 0.5 * d::dr;
        double rm = r - 0.5 * d::dr;

        for (int i = 0; i < d::N_x; ++i) {
            int k = j * d::N_x + i;

            if (k >= d::N_x) {
                S(k, k - d::N_x) = dr2 * rm * eps<I>(i, j);
            }
            if (k >= 1) {
                S(k, k - 1) = dx2 * r * eps<L>(i, j);
            }
            S(k, k) = - dx2 * (r  * eps<L>(i, j) + r  * eps<R>(i, j))
                      - dr2 * (rp * eps<A>(i, j) + rm * eps<I>(i, j));
            if (k < D - 1) {
                S(k, k + 1) = dx2 * r * eps<R>(i, j);
            }
            if (k < D - d::N_x) {
                S(k, k + d::N_x) = dr2 * rp * eps<A>(i, j);
            }
        }
    }

    for (int j = 1; j < d::M_r; ++j) {
        int k = j * d::N_x;
        S(k, k - 1) = 0;
        S(k - 1, k) = 0;
    }

    // horizontal von Neumann
    S(0, 1) *= 2;
    for (int j = 1; j < d::M_cnt; ++j) {
        int k = j * d::N_x;
        S(k - 1, k - 2) *= 2;
        S(k, k + 1) *= 2;
    }
    S(d::M_cnt * d::N_x - 1, d::M_cnt * d::N_x - 2) *= 2;

    // vertical von Neumann
    for (int i = 0; i < d::N_x; ++i) {
        S(i, d::N_x + i) -= dr2 * 0.5 * d::dr * eps<A>(i, 0);
        S(i, d::N_x + i) *= 2;
    }

    gnuplot gp;
    gp << "set terminal wxt size 640, 640" << endl;
    gp << "unset colorbox" << endl;
    gp.set_background(flipud(mat(S)));
    gp.plot();



//    double mull = 1;
//    int i = 0;
//    sp_mat left, left2, center, center2, right, right2;

//    // get the size of the complete sparse matrix:
//    int N_ox = d::N_x - d::N_sc - d::N_dc;
//    int N_sext = d::N_s + d::N_sox;
//    int N_dext = d::N_d + d::N_dox;
//    int D = 0;
//    D += d::N_x * d::M_cnt;
//    D +=   N_ox * d::M_ox;
//    D += (N_sext + N_dext) * d::M_ext;
//    sp_mat S(D, D);

//    // cnt start
//    double r = 0;
//    center = sp_mat(d::N_x, d::N_x);
//    center.diag(0).fill(mull);
//    center.diag( 1).fill(mull);
//    center.diag(-1).fill(mull);
//    center(0, 1) *= 2;
//    center(d::N_x - 1, d::N_x - 2) *= 2;
//    right = sp_mat(d::N_x, d::N_x);
//    right.diag().fill(2 * mull);
//    uword k0 = 0;
//    uword k1 = d::N_x - 1;
//    S({k0, k1}, {k0         , k1         }) = center;
//    S({k0, k1}, {k0 + d::N_x, k1 + d::N_x}) = right;

//    mull = 2;
//    // cnt mid
//    for (i = 1; i < d::M_cnt - 1; ++i) {
//        r = i * d::dr;
//        left = sp_mat(d::N_x, d::N_x);
//        left.diag().fill(mull);
//        center = sp_mat(d::N_x, d::N_x);
//        center.diag( 0).fill(mull);
//        center.diag( 1).fill(mull);
//        center.diag(-1).fill(mull);
//        center(0, 1) *= 2;
//        center(d::N_x - 1, d::N_x - 2) *= 2;
//        right = sp_mat(d::N_x, d::N_x);
//        right.diag().fill(mull);
//        k0 = k1 + 1;
//        k1 += d::N_x;
//        S({k0, k1}, {k0 - d::N_x, k1 - d::N_x}) = left;
//        S({k0, k1}, {k0         , k1         }) = center;
//        S({k0, k1}, {k0 + d::N_x, k1 + d::N_x}) = right;
//    }

//    mull = 3;
//    // cnt end
//    r = i * d::dr;
//    left = sp_mat(d::N_x, d::N_x);
//    left.diag().fill(mull);
//    center = sp_mat(d::N_x, d::N_x);
//    center.diag( 0).fill(mull);
//    center.diag( 1).fill(mull);
//    center.diag(-1).fill(mull);
//    center(0, 1) *= 2;
//    center(d::N_x - 1, d::N_x - 2) *= 2;
//    right = sp_mat(d::N_x, N_ox);
//    right.diag(-d::N_sc).fill(mull);
//    k0 = k1 + 1;
//    k1 += d::N_x;
//    S({k0, k1}, {k0 - d::N_x, k1 - d::N_x}) = left;
//    S({k0, k1}, {k0         , k1         }) = center;
//    S({k0, k1}, {k0 + d::N_x, k1 + N_ox  }) = right;
//    ++i;

//    mull = 4;
//    // ox start
//    r = i * d::dr;
//    left = sp_mat(N_ox, d::N_x);
//    left.diag(d::N_sc).fill(mull);
//    center = sp_mat(N_ox, N_ox);
//    center.diag( 0).fill(mull);
//    center.diag( 1).fill(mull);
//    center.diag(-1).fill(mull);
//    right = sp_mat(N_ox, N_ox);
//    right.diag().fill(mull);
//    k0 = k1 + 1;
//    k1 += N_ox;
//    S({k0, k1}, {k0 - d::N_x, k1 - N_ox}) = left;
//    S({k0, k1}, {k0         , k1       }) = center;
//    S({k0, k1}, {k0 + N_ox  , k1 + N_ox}) = right;
//    ++i;

//    mull = 5;
//    // ox mid
//    for (; i < d::M_cnt + d::M_ox - 1; ++i) {
//        r = i * d::dr;
//        left = sp_mat(N_ox, N_ox);
//        left.diag().fill(mull);
//        center = sp_mat(N_ox, N_ox);
//        center.diag( 0).fill(mull);
//        center.diag( 1).fill(mull);
//        center.diag(-1).fill(mull);
//        right = sp_mat(N_ox, N_ox);
//        right.diag().fill(mull);
//        k0 = k1 + 1;
//        k1 += N_ox;
//        S({k0, k1}, {k0 - N_ox, k1 - N_ox}) = left;
//        S({k0, k1}, {k0       , k1       }) = center;
//        S({k0, k1}, {k0 + N_ox, k1 + N_ox}) = right;
//    }

//    mull = 6;
//    // ox end
//    r = i * d::dr;
//    left = sp_mat(N_ox, N_ox);
//    left.diag().fill(mull);
//    center = sp_mat(N_ox, N_ox);
//    center.diag( 0).fill(mull);
//    center.diag( 1).fill(mull);
//    center.diag(-1).fill(mull);
//    right = sp_mat(N_sext, N_sext);
//    right.diag().fill(mull);
//    right2 = sp_mat(N_dext, N_dext);
//    right2.diag().fill(mull);
//    k0 = k1 + 1;
//    k1 += N_ox;
//    S({k0, k1}, {k0 - N_ox, k1 - N_ox}) = left;
//    S({k0, k1}, {k0       , k1       }) = center;
//    uword k2 = k1 - N_dext + 1;
//    uword k3 = k1;
//    k1 = k0 + N_sext - 1;
//    S({k0, k1}, {k0 + N_ox, k1 + N_ox}) = right;
//    S({k2, k3}, {k3 + N_sext + 1, k3 + N_sext + N_dext}) = right2;
//    ++i;

//    mull = 7;
//    // ext start
//    r = i * d::dr;
//    left = sp_mat(N_sext, N_sext);
//    left.diag().fill(mull);
//    left2 = sp_mat(N_dext, N_dext);
//    left2.diag().fill(mull*2);
//    center = sp_mat(N_sext, N_sext);
//    center.diag( 0).fill(mull);
//    center.diag( 1).fill(mull);
//    center.diag(-1).fill(mull);
//    center2 = sp_mat(N_dext, N_dext);
//    center2.diag( 0).fill(mull*2);
//    center2.diag( 1).fill(mull*2);
//    center2.diag(-1).fill(mull*2);
//    right = sp_mat(N_sext, N_sext);
//    right.diag().fill(mull);
//    right2 = sp_mat(N_dext, N_dext);
//    right2.diag().fill(mull*2);

//    k0 += N_ox;
//    k1 += N_ox;
//    k2 += N_sext + N_dext;
//    k3 += N_sext + N_dext;
//    S({k0, k1}, {k0 - N_ox, k1 - N_ox}) = left;
//    S({k0, k1}, {k0         , k1         }) = center;
//    S({k0, k1}, {k0 + N_sext + N_dext, k1 + N_sext + N_dext}) = right;
//    S({k2, k3}, {k2 - N_sext - N_dext, k3 - N_sext - N_dext}) = left2;
//    S({k2, k3}, {k2                  , k3}) = center2;
//    S({k2, k3}, {k2 + N_sext + N_dext, k3 + N_sext + N_dext}) = right2;
//    ++i;

//    mull = 10;
//    // ext mid
//    for (; i < d::M_cnt + d::M_ox + d::M_ext - 1; ++i) {
//        r = i * d::dr;
//        left = sp_mat(N_sext, N_sext);
//        left.diag().fill(mull);
//        left2 = sp_mat(N_dext, N_dext);
//        left2.diag().fill(mull*2);
//        center = sp_mat(N_sext, N_sext);
//        center.diag( 0).fill(mull);
//        center.diag( 1).fill(mull);
//        center.diag(-1).fill(mull);
//        center2 = sp_mat(N_dext, N_dext);
//        center2.diag( 0).fill(mull*2);
//        center2.diag( 1).fill(mull*2);
//        center2.diag(-1).fill(mull*2);
//        right = sp_mat(N_sext, N_sext);
//        right.diag().fill(mull);
//        right2 = sp_mat(N_dext, N_dext);
//        right2.diag().fill(mull*2);

//        k0 += N_sext + N_dext;
//        k1 += N_sext + N_dext;
//        k2 += N_sext + N_dext;
//        k3 += N_sext + N_dext;
//        S({k0, k1}, {k0 - N_sext - N_dext, k1 - N_sext - N_dext}) = left;
//        S({k0, k1}, {k0                  , k1                  }) = center;
//        S({k0, k1}, {k0 + N_sext + N_dext, k1 + N_sext + N_dext}) = right;
//        S({k2, k3}, {k2 - N_sext - N_dext, k3 - N_sext - N_dext}) = left2;
//        S({k2, k3}, {k2                  , k3                  }) = center2;
//        S({k2, k3}, {k2 + N_sext + N_dext, k3 + N_sext + N_dext}) = right2;
//    }

//    mull = 12;
//    // ext end
//    r = i * d::dr;
//    left = sp_mat(N_sext, N_sext);
//    left.diag().fill(mull*2);
//    left2 = sp_mat(N_dext, N_dext);
//    left2.diag().fill(mull*3);
//    center = sp_mat(N_sext, N_sext);
//    center.diag( 0).fill(mull);
//    center.diag( 1).fill(mull);
//    center.diag(-1).fill(mull);
//    center2 = sp_mat(N_dext, N_dext);
//    center2.diag( 0).fill(mull*2);
//    center2.diag( 1).fill(mull*2);
//    center2.diag(-1).fill(mull*2);
//    k0 += N_sext + N_dext;
//    k1 += N_sext + N_dext;
//    k2 += N_sext + N_dext;
//    k3 += N_sext + N_dext;
//    S({k0, k1}, {k0 - N_sext - N_dext, k1 - N_sext - N_dext}) = left;
//    S({k0, k1}, {k0                  , k1                  }) = center;
//    S({k2, k3}, {k2 - N_sext - N_dext, k3 - N_sext - N_dext}) = left2;
//    S({k2, k3}, {k2                  , k3                  }) = center2;

//    gnuplot gp;
//    gp << "set terminal wxt size 640, 640" << endl;
//    gp << "unset colorbox" << endl;
//    gp.set_background(flipud(mat(S)));
//    gp.plot();

//    double V_s = 0;
//    double V_g = 0.5;
//    double V_d = 1.0;
//    vec R(D);
//    i = d::M_cnt - 1;
//    r = i * d::dr;
//    for (int j = 0; j < d::N_sc; ++j) {
//        R(i * d::N_x + j) = - (r + 0.5 * d::dr) * eps<2>(i, j) * V_s * dr2;
//    }
//    for (int j = d::N_x - d::N_dc; j < d::N_x; ++j) {
//        R(i * d::N_x + j) = - (r + 0.5 * d::dr) * eps<2>(i, j) * V_d * dr2;
//    }
//    i = d::M_cnt + d::M_ox - 1;
//    r = i * d::dr;
//    for (int j = d::N_sc + d::N_sox; j < d::N_sc + d::N_sox + d::N_ox; ++j) {
//        R(i * d::N_)
//    }


//    vec phi = spsolve(S, R);

//    phi = phi({uword(i * d::N_x), uword((i+1) * d::N_x - 1)});

//    plot(phi);

    return 0;
}
