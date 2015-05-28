//#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS
#define MOVIEMODE

#include <iostream>
#include <algorithm>
#include <omp.h>
#include <xmmintrin.h>

#include <armadillo>

#include "device.hpp"
#include "gnuplot.hpp"
#include "inverter.hpp"
#include "potential.hpp"
#include "steady_state.hpp"
#include "time_evolution.hpp"

using namespace arma;
using namespace std;

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

//    device n_fet("n_fet digga");
//    device p_fet("p_fet digga");
//    p_fet.F_s  = - p_fet.F_s;
//    p_fet.F_sc = - p_fet.F_sc;
//    p_fet.F_d  = - p_fet.F_d;
//    p_fet.F_dc = - p_fet.F_dc;

//    inverter i(n_fet, p_fet);

//    cout << "0.00 => " << i.solve({0, 0.1, 0.4}) << endl;
//    cout << "0.15 => " << i.solve({0, 0.15, 0.4}) << endl;
//    cout << "0.20 => " << i.solve({0, 0.2, 0.4}) << endl;
//    cout << "0.25 => " << i.solve({0, 0.25, 0.4}) << endl;
//    cout << "0.30 => " << i.solve({0, 0.3, 0.4}) << endl;

////    steady_state ss(p_fet, {0.8, 0, 1});
////    ss.solve();

    device der_geraet("der Gerät");

    voltage V0{0, .2, .5};
    vector<voltage> V(t::N_t);
    for (int i = 0; i < t::N_t; ++i) {
        V[i] = V0;
    }

    time_evolution te(der_geraet, V);
    te.solve();

    return 0;
}
