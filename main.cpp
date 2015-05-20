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

using namespace arma;
using namespace std;

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

//    steady_state s({0,0,0});
//    s.solve<false>();

    steady_state s({0,0,0});
    plot_phi2D(s.V, s.n);
    plot(make_pair(d::x, s.phi.data));





//    auto phi2D = potential_impl::poisson2D({0.0, 0.5, 1.0}, {});

//    image(phi2D.t());

    return 0;
}
