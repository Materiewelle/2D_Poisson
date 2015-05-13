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

class box {
public:
    int i0;
    int i1;
    int j0;
    int j1;

    double e_in;
    double e_out[4]; // lrtb
};

vector<box> init_boxes() {
    using namespace d;

    static constexpr int i1 = N_sc;
    static constexpr int i2 = i1 + N_s;
    static constexpr int i3 = i2 + N_sox;
    static constexpr int i4 = i3 + N_g;
    static constexpr int i5 = i4 + N_dox;
    static constexpr int i6 = i5 + N_d;
    static constexpr int i7 = i6 + N_dc;

    static constexpr int j1 = M_cnt;
    static constexpr int j2 = j1 + M_ox;
    static constexpr int j3 = j2 + M_ext;

    vector<box> boxes(15);

    // --------------------------- cnt-layer ---------------------------------------------
    // cnt-source-box
    boxes[0] = box({0, i1, 0, j1, eps_cnt, {eps_cnt, eps_cnt, eps_cnt, eps_cnt}});

    // cnt-air-source-box
    boxes[1] = box({i1, i2, 0, j1, eps_cnt, {eps_cnt, eps_cnt, c::eps_0, eps_cnt}});

    // cnt-oxide-box
    boxes[2] = box({i2, i5, 0, j1, eps_cnt, {eps_cnt, eps_cnt, eps_ox, eps_cnt}});

    // cnt-air-drain
    boxes[3] = box({i5, i6, 0, j1, eps_cnt, {eps_cnt, eps_cnt, c::eps_0, eps_cnt}});

    // cnt-drain-box
    boxes[4] = box({i6, i7, 0, j1, eps_cnt, {eps_cnt, eps_cnt, eps_cnt, eps_cnt}});

    // --------------------------- oxide-layer -------------------------------------------
    // left-air-box in oxide layer
    boxes[5] = box({i1, i2, j1, j2, c::eps_0, {c::eps_0, eps_ox, c::eps_0, eps_cnt}});

    // left-oxide-under-air-box
    boxes[6] = box({i2, i3, j1, j2, eps_ox, {c::eps_0, eps_ox, eps_ox, eps_cnt}});

    // oxide-under-gate-box
    boxes[7] = box({i3, i4, j1, j2, eps_ox, {eps_ox, eps_ox, eps_ox, eps_cnt}});

    // right-oxide-under-air-box
    boxes[8] = box({i4, i5, j1, j2, eps_ox, {eps_ox, c::eps_0, c::eps_0, eps_cnt}});

    // right-air-box in oxide layer
    boxes[9] = box({i5, i6, j1, j2, c::eps_0, {eps_ox, c::eps_0, c::eps_0, eps_cnt}});

    // --------------------------- gate-layer --------------------------------------------
    // left-air-box in gate layer
    boxes[10] = box({i1, i2, j2, j3, c::eps_0, {c::eps_0, c::eps_0, c::eps_0, c::eps_0}});

    // left-air-over-oxide-box
    boxes[11] = box({i2, i3, j2, j3, c::eps_0, {c::eps_0, eps_ox, c::eps_0, eps_ox}});

    // gate-box
    boxes[12] = box({i3, i4, j2, j3, 0, {c::eps_0, c::eps_0, 0, eps_ox}});

    // right-air-over-oxide-box
    boxes[13] = box({i4, i5, j2, j3, c::eps_0, {c::eps_0, c::eps_0, c::eps_0, eps_ox}});

    // right-air-box in gate layer
    boxes[14] = box({i5, i6, j2, j3, c::eps_0, {c::eps_0, c::eps_0, c::eps_0, c::eps_0}});

    // --------------------------- contacts ----------------------------------------------
    // left-contact-box
    boxes[15] = box({0, i1, j1, j3, 0, {0, c::eps_0, 0, eps_cnt}});

    //right-contact-box
    boxes[16] = box({i6, i7, j1, j3, 0, {c::eps_0, 0, 0, eps_cnt}});

    return boxes;
}

template<int dir>
static inline double eps(int i, int j) {
    static int last_box = 0;
    vector<box> boxes = init_boxes();
    // ToDo: do all boxes cyclically until last box (linked list of int?)
    for (int b = last_box; b < before_last; ++b) {
        if (inside_box) {
            eps = border_check(boxes[b]);
            last_box = b;
            return eps;
        }
    return -1; // something went wrong
    }
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
