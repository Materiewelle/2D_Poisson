#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS

#include <fstream>
#include <iostream>
#include <omp.h>
#include <xmmintrin.h>
#include <fstream>

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

    device ntfet("ntfet", ntfet_model, tfet_geometry);
    device ptfet("ptfet", ptfet_model, tfet_geometry);
    device nfet("nfet", nfet_model, fet_geometry);
    device pfet("pfet", pfet_model, fet_geometry);

    steady_state sn(ntfet, {0,.3,.3});
    sn.solve<false>();
    cout << sn.I.total(0) << endl;
    steady_state sp(ptfet, {0,-.3,-.3});
    sp.solve<false>();
    cout << sp.I.total(0) << endl;
    cout << "rel " << sn.I.total(0) / sp.I.total(0) << endl;

    plot_ldos(ntfet, sn.phi);
    plot_ldos(ptfet, sp.phi);

//    inverter i(nfet, pfet, 1e-13);
//    signal sg = linear_signal(5e-11, {10 * time_evolution::dt, 110 * time_evolution::dt}, {{0.0, 0.1, 0.45}, {0.0, 0.3, 0.45}});
//    i.solve(sg);
//    i.save();
//    return 0;

//    vec V_in, V_out;
//    inverter i(nfet, pfet);
//    i.output({0, .2, .4}, .4, 40, V_in, V_out);

//    ofstream fi("/home/fabian/tfet_inverter.txt");
//    for (unsigned i = 0; i < V_in.size(); ++i) {
//        fi << V_in(i) << "\t" << V_out(i) << endl;
//    }
//    fi.close();

}
