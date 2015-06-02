#define ARMA_NO_DEBUG    // no bound checks
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

    device n_fet("n_fet digga");
    device p_fet("p_fet digga");
    p_fet.F_s  = - p_fet.F_s;
    p_fet.F_sc = - p_fet.F_sc;
    p_fet.F_d  = - p_fet.F_d;
    p_fet.F_dc = - p_fet.F_dc;

//    steady_state s(n_fet, {0, 0.2, 0.5});
//    s.solve();
//    wave_packet psi;
//    psi.init< true>(n_fet, s.E[LC], s.W[LC], s.phi);

//    image(psi.E);

//    inverter i(n_fet, p_fet);
//    vec V_in;
//    vec V_out;
//    i.output({0, 0.17, 0.4}, 0.23, 20, V_in, V_out);

//    plot(make_pair(V_in, V_out));

    time_evolution te(n_fet);
    std::fill(begin(te.V), begin(te.V) + 2, voltage{0.0, 0.2, 0.5});
    vec ramp = linspace(0, 0.05, 20);
    for (int i = 2; i < 22; ++i) {
        te.V[i] = {ramp(i-2), 0.2, 0.5};
    }
    std::fill(begin(te.V) + 22, end(te.V), voltage{0.05, 0.2, 0.5});
    te.solve();

//    vec V_d;
//    vec I;
//    steady_state::output(p_fet, {0.0, -0.2, -0.6}, 0.2, 250, V_d, I);
//    plot(make_pair(V_d, I));

//    device der_geraet("der Gerät");

//    voltage V0{0, .2, .5};
//    vector<voltage> V(t::N_t);
//    for (int i = 0; i < t::N_t; ++i) {
//        V[i] = V0;
//    }

//    time_evolution te(der_geraet, V);
//    te.solve();

    return 0;
}
