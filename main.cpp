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

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    static constexpr double dr2 = 1.0 / d::dr / d::dr;
    static constexpr double dx2 = 1.0 / d::dx / d::dx;

    double mull = 1;
    int i = 0;
    sp_mat left, left2, center, center2, right, right2;

    // get the size of the complete sparse matrix:
    int N_ox = d::N_x - d::N_sc - d::N_dc;
    int N_sext = d::N_s + d::N_sox;
    int N_dext = d::N_d + d::N_dox;
    int D = 0;
    D += d::N_x * d::M_cnt;
    D +=   N_ox * d::M_ox;
    D += (N_sext + N_dext) * d::M_ext;
    sp_mat S(D, D);

    // cnt start
    double r = 0;
    center = sp_mat(d::N_x, d::N_x);
    center.diag(0).fill(mull);
    center.diag( 1).fill(mull);
    center.diag(-1).fill(mull);
    center(0, 1) *= 2;
    center(d::N_x - 1, d::N_x - 2) *= 2;
    right = sp_mat(d::N_x, d::N_x);
    right.diag().fill(2 * mull);
    uword j0 = 0;
    uword j1 = d::N_x - 1;
    S({j0, j1}, {j0         , j1         }) = center;
    S({j0, j1}, {j0 + d::N_x, j1 + d::N_x}) = right;

    mull = 2;
    // cnt mid
    for (i = 1; i < d::M_cnt - 1; ++i) {
        r = i * d::dr;
        left = sp_mat(d::N_x, d::N_x);
        left.diag().fill(mull);
        center = sp_mat(d::N_x, d::N_x);
        center.diag( 0).fill(mull);
        center.diag( 1).fill(mull);
        center.diag(-1).fill(mull);
        center(0, 1) *= 2;
        center(d::N_x - 1, d::N_x - 2) *= 2;
        right = sp_mat(d::N_x, d::N_x);
        right.diag().fill(mull);
        j0 = j1 + 1;
        j1 += d::N_x;
        S({j0, j1}, {j0 - d::N_x, j1 - d::N_x}) = left;
        S({j0, j1}, {j0         , j1         }) = center;
        S({j0, j1}, {j0 + d::N_x, j1 + d::N_x}) = right;
    }

    mull = 3;
    // cnt end
    r = i * d::dr;
    left = sp_mat(d::N_x, d::N_x);
    left.diag().fill(mull);
    center = sp_mat(d::N_x, d::N_x);
    center.diag( 0).fill(mull);
    center.diag( 1).fill(mull);
    center.diag(-1).fill(mull);
    center(0, 1) *= 2;
    center(d::N_x - 1, d::N_x - 2) *= 2;
    right = sp_mat(d::N_x, N_ox);
    right.diag(-d::N_sc).fill(mull);
    j0 = j1 + 1;
    j1 += d::N_x;
    S({j0, j1}, {j0 - d::N_x, j1 - d::N_x}) = left;
    S({j0, j1}, {j0         , j1         }) = center;
    S({j0, j1}, {j0 + d::N_x, j1 + N_ox  }) = right;
    ++i;

    mull = 4;
    // ox start
    r = i * d::dr;
    left = sp_mat(N_ox, d::N_x);
    left.diag(d::N_sc).fill(mull);
    center = sp_mat(N_ox, N_ox);
    center.diag( 0).fill(mull);
    center.diag( 1).fill(mull);
    center.diag(-1).fill(mull);
    right = sp_mat(N_ox, N_ox);
    right.diag().fill(mull);
    j0 = j1 + 1;
    j1 += N_ox;
    S({j0, j1}, {j0 - d::N_x, j1 - N_ox}) = left;
    S({j0, j1}, {j0         , j1       }) = center;
    S({j0, j1}, {j0 + N_ox  , j1 + N_ox}) = right;
    ++i;

    mull = 5;
    // ox mid
    for (; i < d::M_cnt + d::M_ox - 1; ++i) {
        r = i * d::dr;
        left = sp_mat(N_ox, N_ox);
        left.diag().fill(mull);
        center = sp_mat(N_ox, N_ox);
        center.diag( 0).fill(mull);
        center.diag( 1).fill(mull);
        center.diag(-1).fill(mull);
        right = sp_mat(N_ox, N_ox);
        right.diag().fill(mull);
        j0 = j1 + 1;
        j1 += N_ox;
        S({j0, j1}, {j0 - N_ox, j1 - N_ox}) = left;
        S({j0, j1}, {j0       , j1       }) = center;
        S({j0, j1}, {j0 + N_ox, j1 + N_ox}) = right;
    }

    mull = 6;
    // ox end
    r = i * d::dr;
    left = sp_mat(N_ox, N_ox);
    left.diag().fill(mull);
    center = sp_mat(N_ox, N_ox);
    center.diag( 0).fill(mull);
    center.diag( 1).fill(mull);
    center.diag(-1).fill(mull);
    right = sp_mat(N_sext, N_sext);
    right.diag().fill(mull);
    right2 = sp_mat(N_dext, N_dext);
    right2.diag().fill(mull);
    j0 = j1 + 1;
    j1 += N_ox;
    S({j0, j1}, {j0 - N_ox, j1 - N_ox}) = left;
    S({j0, j1}, {j0       , j1       }) = center;
    uword j2 = j1 - N_dext + 1;
    uword j3 = j1;
    j1 = j0 + N_sext - 1;
    S({j0, j1}, {j0 + N_ox, j1 + N_ox}) = right;
    S({j2, j3}, {j3 + N_sext + 1, j3 + N_sext + N_dext}) = right2;
    ++i;

    mull = 7;
    // ext start
    r = i * d::dr;
    left = sp_mat(N_sext, N_sext);
    left.diag().fill(mull);
    left2 = sp_mat(N_dext, N_dext);
    left2.diag().fill(mull*2);
    center = sp_mat(N_sext, N_sext);
    center.diag( 0).fill(mull);
    center.diag( 1).fill(mull);
    center.diag(-1).fill(mull);
    center2 = sp_mat(N_dext, N_dext);
    center2.diag( 0).fill(mull*2);
    center2.diag( 1).fill(mull*2);
    center2.diag(-1).fill(mull*2);
    right = sp_mat(N_sext, N_sext);
    right.diag().fill(mull);
    right2 = sp_mat(N_dext, N_dext);
    right2.diag().fill(mull*2);

    j0 += N_ox;
    j1 += N_ox;
    j2 += N_sext + N_dext;
    j3 += N_sext + N_dext;
    S({j0, j1}, {j0 - N_ox, j1 - N_ox}) = left;
    S({j0, j1}, {j0         , j1         }) = center;
    S({j0, j1}, {j0 + N_sext + N_dext, j1 + N_sext + N_dext}) = right;
    S({j2, j3}, {j2 - N_sext - N_dext, j3 - N_sext - N_dext}) = left2;
    S({j2, j3}, {j2                  , j3}) = center2;
    S({j2, j3}, {j2 + N_sext + N_dext, j3 + N_sext + N_dext}) = right2;
    ++i;

    mull = 10;
    // ext mid
    for (; i < d::M_cnt + d::M_ox + d::M_ext - 1; ++i) {
        r = i * d::dr;
        left = sp_mat(N_sext, N_sext);
        left.diag().fill(mull);
        left2 = sp_mat(N_dext, N_dext);
        left2.diag().fill(mull*2);
        center = sp_mat(N_sext, N_sext);
        center.diag( 0).fill(mull);
        center.diag( 1).fill(mull);
        center.diag(-1).fill(mull);
        center2 = sp_mat(N_dext, N_dext);
        center2.diag( 0).fill(mull*2);
        center2.diag( 1).fill(mull*2);
        center2.diag(-1).fill(mull*2);
        right = sp_mat(N_sext, N_sext);
        right.diag().fill(mull);
        right2 = sp_mat(N_dext, N_dext);
        right2.diag().fill(mull*2);

        j0 += N_sext + N_dext;
        j1 += N_sext + N_dext;
        j2 += N_sext + N_dext;
        j3 += N_sext + N_dext;
        S({j0, j1}, {j0 - N_sext - N_dext, j1 - N_sext - N_dext}) = left;
        S({j0, j1}, {j0                  , j1                  }) = center;
        S({j0, j1}, {j0 + N_sext + N_dext, j1 + N_sext + N_dext}) = right;
        S({j2, j3}, {j2 - N_sext - N_dext, j3 - N_sext - N_dext}) = left2;
        S({j2, j3}, {j2                  , j3                  }) = center2;
        S({j2, j3}, {j2 + N_sext + N_dext, j3 + N_sext + N_dext}) = right2;
    }

    mull = 12;
    // ext end
    r = i * d::dr;
    left = sp_mat(N_sext, N_sext);
    left.diag().fill(mull*2);
    left2 = sp_mat(N_dext, N_dext);
    left2.diag().fill(mull*3);
    center = sp_mat(N_sext, N_sext);
    center.diag( 0).fill(mull);
    center.diag( 1).fill(mull);
    center.diag(-1).fill(mull);
    center2 = sp_mat(N_dext, N_dext);
    center2.diag( 0).fill(mull*2);
    center2.diag( 1).fill(mull*2);
    center2.diag(-1).fill(mull*2);
    j0 += N_sext + N_dext;
    j1 += N_sext + N_dext;
    j2 += N_sext + N_dext;
    j3 += N_sext + N_dext;
    S({j0, j1}, {j0 - N_sext - N_dext, j1 - N_sext - N_dext}) = left;
    S({j0, j1}, {j0                  , j1                  }) = center;
    S({j2, j3}, {j2 - N_sext - N_dext, j3 - N_sext - N_dext}) = left2;
    S({j2, j3}, {j2                  , j3                  }) = center2;

    gnuplot gp;
    gp << "set terminal wxt size 640, 640" << endl;
    gp << "unset colorbox" << endl;
    gp.set_background(flipud(mat(S)));
    gp.plot();

    double V_s = 0;
    double V_g = 0.5;
    double V_d = 1.0;
    vec R(D);
    i = d::M_cnt - 1;
    for (int j = 0; j < d::N_sc; ++j) {
        R(i * d::N_x + j) = V_s;
    }
    for (int j = d::N_x - d::N_dc; j < d::N_x; ++j) {
        R(i * d::N_x + j) = V_d;
    }

    vec phi = spsolve(S, R);

    phi = phi({uword(i * d::N_x), uword((i+1) * d::N_x - 1)});

    plot(phi);

    return 0;
}
