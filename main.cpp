//#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS

#include <iostream>
#include <algorithm>
#include <omp.h>
#include <xmmintrin.h>

#include <armadillo>

#include "device.hpp"

using namespace arma;
using namespace std;

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    static constexpr double dr2 = 1.0 / d::dr / d::dr;
    static constexpr double dx2 = 1.0 / d::dx / d::dx;

    // get the size of the complete sparse matrix:
    int D = 0;
    D += d::N_x * d::M_cnt;
    D += (d::N_x - d::N_sc - d::N_dc) * d::M_ox;
    D += ((d::N_s + d::N_sox) + (d::N_d + d::N_dox)) * d::M_ext;
    sp_mat S(D, D);

    // first row of blocks
    sp_mat S_00(d::N_x, d::N_x);
    S_00.diag(0).fill(-2.0 * (dr2 + dx2));
    S_00.diag( 1).fill(dx2);
    S_00.diag(-1).fill(dx2);
    S_00(0, 1) *= 2;
    S_00(d::N_x - 1, d::N_x - 2) *= 2;
    S({0, d::N_x-1}, {0, d::N_x-1}) = S_00;

    sp_mat S01(d::N_x, d::N_x);
    S01.diag().fill(2.0 * dr2);
    S({0, d::N_x-1},{d::N_x, 2*d::N_x-1}) = S01;

    // central rows of blocks
    for (int i = 1; i < d::M_cnt - 1; ++i) {
        double r = i * d::dr;

        sp_mat sides(d::N_x, d::N_x);
        sides.diag().fill(dr2 - 0.5 / r / d::dr);

        sp_mat center = sp_mat(d::N_x, d::N_x);
        center.diag( 0).fill(-2.0 * (dr2 + dx2));
        center.diag( 1).fill(dx2);
        center.diag(-1).fill(dx2);
        center(0, 1) *= 2;
        center(d::N_x - 1, d::N_x - 2) *= 2;

        S({i * d::N_x, (i+1) * d::N_x - 1}, {(i-1) * d::N_x, i     * d::N_x - 1}) = sides;
        S({i * d::N_x, (i+1) * d::N_x - 1}, {i     * d::N_x, (i+1) * d::N_x - 1}) = center;
        S({i * d::N_x, (i+1) * d::N_x - 1}, {(i+1) * d::N_x, (i+2) * d::N_x - 1}) = sides;
    }

    // transition from nanotube to gate-oxide
    int i = d::M_cnt - 1;
    sp_mat left(d::N_x, d::N_x);
    double r = i * d::dr;
    left.diag().fill(dr2 - 0.5 / r / d::dr);

    sp_mat center(d::N_x, d::N_x);
    center.diag(0).fill(-2.0 * (dr2 + dx2));
    center.diag( 1).fill(dx2);
    center.diag(-1).fill(dx2);
    center(0, 1) *= 2;
    center(d::N_x - 1, d::N_x - 2) *= 2;

    int N_ox = d::N_x - d::N_sc - d::N_dc;
    sp_mat right(N_ox, d::N_x);
    right.diag(-d::N_sc).fill(5); //dummy-value

    S({i * d::N_x, (i+1) * d::N_x - 1}, {(i-1) * d::N_x, i     * d::N_x - 1}) = left;
    S({i * d::N_x, (i+1) * d::N_x - 1}, {i     * d::N_x, (i+1) * d::N_x - 1}) = center;
    S({i * d::N_x, (i+1) * d::N_x - 1}, {(i+1) * d::N_x, (i+1) * d::N_x + d::N_sc - 1}) = right;

    cout << S << endl;


    return 0;
}
