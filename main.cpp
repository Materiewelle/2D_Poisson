#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS

#include "include.hpp"

using namespace arma;
using namespace std;

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

//    device nfet("nfet", nfet_model, fet_geometry);
    device tfet("tfet", tfet_model, tfet_geometry);


    signal sg(1e-12, {6e-16, 4e-14}, {{0.0, -0.2, 0.5}, {0.0, 0.8, 0.5}});

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
//    plot_ldos(tfet, ss.phi);
//    plot(ss.E[LV]);
//    plot(ss.E[RC]);

    time_evolution te(ss, sg);

    std::vector<std::pair<int, int>> E_ind(2);
    E_ind[0] = std::make_pair(LV, 3600);
    E_ind[1] = std::make_pair(RC, 260);
    movie argo(te, E_ind);

    te.solve();
    te.save();

    return 0;
}
