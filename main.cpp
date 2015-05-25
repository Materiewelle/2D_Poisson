#define ARMA_NO_DEBUG    // no bound checks
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

using namespace arma;
using namespace std;

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    voltage V{0, 0, 1};
//    plot_phi2D(V);
//    plot_ldos({V});

    steady_state s(V);
    s.solve();

//    plot_phi2D(V, s.n);
//    plot_ldos(s.phi);


    gnuplot gpn;
    gpn.add(make_pair(d::x, s.n.data));
    wave_packet psi[4];
    psi[LV].init< true>(s.E[LV], s.W[LV], s.phi);
    psi[RV].init<false>(s.E[RV], s.W[RV], s.phi);
    psi[LC].init< true>(s.E[LC], s.W[LC], s.phi);
    psi[RC].init<false>(s.E[RC], s.W[RC], s.phi);
    s.n.update(psi, s.phi);
    gpn.add(make_pair(d::x, s.n.data));
    gpn.plot();

    return 0;
}
