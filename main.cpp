#define ARMA_NO_DEBUG    // no bound checks
#define GNUPLOT_NOPLOTS

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

    signal sg = sine_signal(3998 * time_evolution::dt, {0, 0, .5}, {0, .2, 0 }, 1e12, 10 * time_evolution::dt, .5 * M_PI);
    time_evolution te(ntfet, sg);
    te.solve();
    te.save();
//    vec s(sg.V.size());
//    vec g(sg.V.size());
//    vec d(sg.V.size());
//    for (unsigned i = 0; i < sg.V.size(); ++i) {
//        s[i] = sg.V[i].s;
//        g[i] = sg.V[i].g;
//        d[i] = sg.V[i].d;
//    }
//    plot(s,g,d);
//    return 0;
//    time_evolution te(ntfet, sg);

//    signal sg = const_signal(5e-12, {0.0, 0.2, 0.5});
//    time_evolution te(nfet, sg);
//    std::vector<std::pair<int, int>> E_i(16);
//    for (int i = 0; i < 4; ++i) {
//        for (int j = 0; j < 4; ++j) {
//            E_i[i * 4 + j] = std::make_pair(i, (int)(j * te.psi[i].E0.size() * 0.25));
//        }
//    }
//    movie argo(te, E_i);
//    vec qsums = abs(te.qsum.s) / max(abs(te.qsum.s));
//    vec qsumd = abs(te.qsum.d) / max(abs(te.qsum.d));
//    plot(qsums);
//    plot(qsumd);
//    te.solve();

//    steady_state sn(ntfet, {0,.3,.3});
//    sn.solve<false>();
//    cout << sn.I.total(0) << endl;
//    steady_state sp(ptfet, {0,-.3,-.3});
//    sp.solve<false>();
//    cout << sp.I.total(0) << endl;
//    cout << "rel " << sn.I.total(0) / sp.I.total(0) << endl;

//    plot_ldos(ntfet, sn.phi);
//    plot_ldos(ptfet, sp.phi);

//    inverter i(nfet, pfet, 1e-16);
//    signal sg = linear_signal(5e-11, {10 * time_evolution::dt, 110 * time_evolution::dt}, {{0.0, 0.2, 0.5}, {0.0, 0.3, 0.5}});
//    i.solve(sg);
//    i.save();
//    return 0;

//    steady_state ss_n(nfet, voltage{0.0, 0.0, 0.4});
//    steady_state ss_p(pfet, voltage{0.4, 0.01, 0.3838383838383838383838383838383838});

//    ss_n.solve();
//    ss_p.solve();

//    plot_ldos(nfet, ss_n.phi);
//    plot_ldos(pfet, ss_p.phi);
//    cout << ss_n.I.total(0) << endl;
//    cout << ss_p.I.total(0) << endl;
//    return 0;

//    vec V_in, V_out;
//    inverter i(nfet, pfet);
//    i.output({0.0, 0.2, 0.5}, 0.3, 200, V_in, V_out);
//    plot(make_pair(V_in, V_out));
}
