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

    static constexpr double dr2 = 1.0 / d::dr * d::dr;
    static constexpr double dx2 = 1.0 / d::dx * d::dx;

    sp_mat S(d::N_x, d::N_x);
    S.diag( 0).fill(-2.0 * (dr2 + dx2));
    S.diag( 1).fill(dx2);
    S.diag(-1).fill(dx2);
    S(0, 1) *= 2;
    S(d::N_x - 1, d::N_x - 2) *= 2;

    sp_mat S2(d::N_x, d::N_x);
    S2.diag().fill(2.0 * dr2);

    S = join_horiz(S, S2);

    for (int i = 1; i < d::M_sc - 1; ++i) {
        double r = i * d::dr;

        sp_mat block0(d::N_x, d::N_x);
        block0.diag().fill(dr2 - 0.5 / r / d::dr);

        sp_mat block1 = sp_mat(d::N_x, d::N_x);
        block1.diag( 0).fill(-2.0 * (dr2 + dx2));
        block1.diag( 1).fill(dx2);
        block1.diag(-1).fill(dx2);
        block1(0, 1) *= 2;
        block1(d::N_x - 1, d::N_x - 2) *= 2;

        sp_mat block2 = sp_mat(d::N_x, d::N_x);
        block2.diag().fill(dr2 + 0.5 / r / d::dr);

        S2 = join_horiz(join_horiz(block0, block1), block2);
        S = join_horiz(S, sp_mat(d::N_x, d::N_x));
        S = join_vert(S, S2);
    }


    return 0;
}
