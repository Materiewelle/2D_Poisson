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

int main(int argc, char ** argv) {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    cout << "saving results in " << save_folder() << endl;

    omp_set_num_threads(stoi(argv[1]));

    device ntfet("ntfet", ntfet_model, tfet_geometry);

    double f = stod(argv[2]) * 1e9;
    double T = 1 / f;
    double dry = 10 * time_evolution::dt;

    double V0_g, V0_d, A_g, A_d, ph;
    switch (stoi(argv[3])) {
        case 1: // gate signal
        V0_g = 0.0;
        V0_d = 0.5;
        A_g  = 0.2;
        A_d  = 0.0;
        ph   = .5 * M_PI;
        break;

        case 2: // drain signal
        V0_g = 0.4;
        V0_d = 0.0;
        A_g  = 0.0;
        A_d  = 0.25;
        ph   = -.5 * M_PI;
        break;

    default:
        cout << "dummkopf" << endl;
        return 1;
    }

    signal sg = sine_signal(2 * T + dry, {0, V0_g, V0_d}, {0, A_g, A_d}, f, dry, ph);
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
    return 0;
}
