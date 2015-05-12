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

    // get the size of the complete sparse matrix:
    int D = 0;
    D += d::N_x * d::M_cnt;
    D += (d::N_x - d::N_sc - d::N_dc) * d::M_ox;
    D += ((d::N_s + d::N_sox) + (d::N_d + d::N_dox)) * d::M_ext;
    sp_mat S(D, D);


    // first two blocks
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

    cout << S << endl;

//    for (int i = 1; i < d::M_sc - 1; ++i) {
//        double r = i * d::dr;

//        sp_mat block0(d::N_x, d::N_x);
//        block0.diag().fill(dr2 - 0.5 / r / d::dr);

//        sp_mat block1 = sp_mat(d::N_x, d::N_x);
//        block1.diag( 0).fill(-2.0 * (dr2 + dx2));
//        block1.diag( 1).fill(dx2);
//        block1.diag(-1).fill(dx2);
//        block1(0, 1) *= 2;
//        block1(d::N_x - 1, d::N_x - 2) *= 2;

//        sp_mat block2 = sp_mat(d::N_x, d::N_x);
//        block2.diag().fill(dr2 + 0.5 / r / d::dr);

//        S2 = join_horiz(join_horiz(block0, block1), block2);
//        S = join_horiz(S, sp_mat(d::N_x, d::N_x));
//        S = join_vert(S, S2);
//    }

//    sp_mat A(10,10);
//    sp_mat B(5,5);
//    B.diag(0).fill(2.5);
//    A({0, 4}, {0, 4}) = B;

//    cout << A << endl;


    return 0;
}
