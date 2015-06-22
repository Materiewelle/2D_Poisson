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

    device nfet("n-tfet", ntfet_model, tfet_geometry);
    device pfet("p-tfet", ptfet_model, tfet_geometry);

//    inverter i(nfet, pfet, 1e-13);
//    signal sg = linear_signal(5e-11, {10 * time_evolution::dt, 110 * time_evolution::dt}, {{0.0, 0.1, 0.45}, {0.0, 0.3, 0.45}});
//    i.solve(sg);
//    i.save();
//    return 0;

//    steady_state sn0{nfet, {0, 0, 0}};
//    sn0.solve();
//    plot_ldos(nfet, sn0.phi);

//    steady_state sn1{nfet, {0, .4, .4}};
//    sn1.solve();
//    plot_ldos(nfet, sn1.phi);

//    steady_state sp0{pfet, {0, 0, -0.4}};
//    sp0.solve();
//    plot_ldos(pfet, sp0.phi);

//    steady_state sp1{pfet, {0, -.4, -0.4}};
//    sp1.solve();
//    plot_ldos(pfet, sp1.phi);

    vec nV_g, nI;
    steady_state::transfer<false>(nfet, {0, -.5, .4}, .5, 30, nV_g, nI);
    vec pV_g, pI;
    steady_state::transfer<false>(pfet, {0, .5, -.4}, -.5, 30, pV_g, pI);

    gnuplot gp;
    gp.add(make_pair(nV_g, nI));
    gp.add(make_pair(pV_g, -pI));
    gp << "set logscale y\n";
    gp << "set format y '%1.0g'\n";
    gp.plot();

    vec V_in, V_out;
    inverter i(nfet, pfet);
    i.output({0, .1, .4}, .3, 40, V_in, V_out);
    plot(make_pair(V_in, V_out));





//    device tfet("tfet", tfet_model, tfet_geometry);

//    signal sg = sine_signal(4e-12 + 1e-14, {0.0, 0.7, 0.5}, {0.0, 0.0, 0.25}, {5e11}, {1e-14}, {.5 * M_PI});

//    signal sg = linear_signal(1e-12, {6e-16, 4e-14}, {{0.0, 0.49, 0.0}, {0.0, 0.5, 0.8}});

//    //for checking if the signal came out fine
//    vec s(sg.V.size());
//    vec g(sg.V.size());
//    vec d(sg.V.size());
//    for (unsigned i = 0; i < sg.V.size(); ++i) {
//        s(i) = sg.V[i].s;
//        g(i) = sg.V[i].g;
//        d(i) = sg.V[i].d;
//    }
//    plot(s, g, d);

//    steady_state ss(tfet, sg.V[0]);
//    ss.solve();

//    // for identifying nice E-numbers
//    plot_ldos(ss.d, ss.phi);
//    plot(ss.E[LV]);
//    plot(ss.E[RC]);
//    plot(ss.E[RV]);

//    time_evolution te(ss, sg);

//    std::vector<std::pair<int, int>> E_ind(2);
//    E_ind[0] = std::make_pair(LV, 1500);
//    E_ind[1] = std::make_pair(LV, 1660);
//    E_ind[1] = std::make_pair(RC, 280);
//    movie argo(te, E_ind);

//    te.solve();
//    te.save();

//    return 0;
}
