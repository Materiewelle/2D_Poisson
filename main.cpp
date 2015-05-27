//#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS

#include <iostream>
#include <algorithm>
#include <omp.h>
#include <xmmintrin.h>

#include <armadillo>

#include "device.hpp"
#include "gnuplot.hpp"
#include "steady_state.hpp"
#include "potential.hpp"
#include "time_evolution.hpp"

using namespace arma;
using namespace std;

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    device der_geraet("der Gerät");
//    cout << der_geraet.to_string() << endl;

//    vec V_g;
//    vec I;
//    steady_state::transfer(der_geraet, {0, -0.2, 0.6}, 0.8, 50, V_g, I);

//    plot(make_pair(V_g, log(I)));
//    return 0;

    voltage V{0, 0, 1};
    potential phi(der_geraet, V);

    plot_phi2D(der_geraet, V);
    plot_ldos(der_geraet, phi);

    steady_state s(der_geraet, V);
    s.solve();

    plot_phi2D(der_geraet, V, s.n);
    plot_ldos(der_geraet, s.phi);

    return 0;
}
