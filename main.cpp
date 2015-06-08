//#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS
#define MOVIEMODE

#include <iostream>
#include <algorithm>
#include <omp.h>
#include <xmmintrin.h>

#include <armadillo>

#include "brent.hpp"
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

    device tfet("tfet", tfet_model, standard_geometry);

    vec V_g;
    vec I;
    steady_state::transfer(tfet, {0.0, -0.2, 0.4}, 0.6, 250, V_g, I);

    mat res = join_horiz(V_g, I);
    res.save("tfet_transfer", raw_ascii);

//    time_evolution te(nfet);
//    std::fill(begin(te.V), begin(te.V) + 2, voltage{0.0, 0.2, 0.5});
//    vec ramp = linspace(0, 0.05, 20);
//    for (int i = 2; i < 22; ++i) {
//        te.V[i] = {ramp(i-2), 0.2, 0.5};
//    }
//    std::fill(begin(te.V) + 22, end(te.V), voltage{0.05, 0.2, 0.5});
//    te.solve();

    return 0;
}
