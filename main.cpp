#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS

#include <fstream>
#include <iostream>
#include <omp.h>
#include <xmmintrin.h>

#define CHARGE_DENSITY_HPP_BODY
#define WAVE_PACKET_HPP_BODY
#define SIGNAL_HPP_BODY

#include "anderson.hpp"
#include "constant.hpp"
#include "device.hpp"
#include "fermi.hpp"
#include "rwth.hpp"
#include "gnuplot.hpp"
#include "brent.hpp"
#include "integral.hpp"
#include "inverse.hpp"
#include "system.hpp"
#include "voltage.hpp"
#include "sd_quantity.hpp"
#include "charge_density.hpp"
#include "potential.hpp"
#include "green.hpp"
#include "wave_packet.hpp"
#include "current.hpp"
#include "steady_state.hpp"
#include "signal.hpp"
#include "time_evolution.hpp"
#include "inverter.hpp"
#include "movie.hpp"

#undef CHARGE_DENSITY_HPP_BODY
#undef WAVE_PACKET_HPP_BODY
#undef SIGNAL_HPP_BODY

#include "charge_density.hpp"
#include "wave_packet.hpp"
#include "signal.hpp"

using namespace arma;
using namespace std;

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

//    device nfet("nfet", nfet_model, fet_geometry);
    device tfet("tfet", tfet_model, tfet_geometry);

    signal sg = sine_signal(1e-12, {0.0, 0.35, 0.5}, {0.0, 0.2, 0.0}, {1e13}, {1e-14}, {0});

//    signal sg = linear_signal(1e-12, {6e-16, 4e-14}, {{0.0, 0.49, 0.0}, {0.0, 0.5, 0.8}});

//    // for checking if the signal came out fine
//    vec s(sg.V.size());
//    vec g(sg.V.size());
//    vec d(sg.V.size());
//    for (unsigned i = 0; i < sg.V.size(); ++i) {
//        s(i) = sg.V[i].s;
//        g(i) = sg.V[i].g;
//        d(i) = sg.V[i].d;
//    }
//    plot(s, g, d);

    steady_state ss(tfet, sg.V[0]);
    ss.solve();

//     // for identifying nice E-numbers
//    plot_ldos(ss.d, ss.phi);
//    plot(ss.E[LV]);
//    plot(ss.E[RC]);
//    plot(ss.E[RV]);

    time_evolution te(ss, sg);

    std::vector<std::pair<int, int>> E_ind(2);
    E_ind[0] = std::make_pair(LV, 4800);
    E_ind[1] = std::make_pair(RC, 2200);
//    E_ind[2] = std::make_pair(RV, 280);
    movie argo(te, E_ind);

    te.solve();
    te.save();

    return 0;
}
