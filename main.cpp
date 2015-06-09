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
#include <string>
#include <sstream>

using namespace arma;
using namespace std;

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    device tfet("tfet", tfet_model, standard_geometry);
    device nfet("nfet", nfet_model, standard_geometry);

    steady_state s(nfet, {0.0, 0.0, 0.8});
    s.solve();
    cout << s.E[LV].size() + s.E[LC].size() + s.E[RV].size() + s.E[RC].size() << endl;
    vec test = s.phi.data;

    plot(test);

    return 0;

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

//    time_evolution te(nfet, voltage { 0.0, 0.2, 0.5 });
//    vec ramp = linspace(0, 0.05, 20);
//    for (int i = 2; i < 22; ++i) {
//        te.V[i] = {ramp(i-2), 0.2, 0.5};
//    }

    vec V_g;
    vec I;
    for (double V_d = 0.2; V_d < 0.6; V_d += 0.05) {
        steady_state::transfer(tfet, {0.0, -0.2, 0.4}, 0.6, 3, V_g, I);
        mat res = join_horiz(V_g, I);
        std::stringstream ss;
        ss << "tfet_transfer_Vd=" << std::setprecision(2) << V_d;
        res.save(ss.str(), raw_ascii);
    }

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
