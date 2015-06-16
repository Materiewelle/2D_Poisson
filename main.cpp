#define ARMA_NO_DEBUG    // no bound checks
#define GNUPLOT_NOPLOTS

#include "include.hpp"

using namespace arma;
using namespace std;

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    device nfet("nfet", nfet_model, fet_geometry);
//    device tfet("tfet", tfet_model, tfet_geometry);


    signal sg(1e-12, {6e-16, 2e-14}, {{0.0, 0.1, 0.5}, {0.0, 0.5, 0.5}});
//    vec s(sg.V.size());
//    vec g(sg.V.size());
//    vec d(sg.V.size());
//    for (unsigned i = 0; i < sg.V.size(); ++i) {
//        s(i) = sg.V[i].s;
//        g(i) = sg.V[i].g;
//        d(i) = sg.V[i].d;
//    }
//    plot(s, g, d);

    time_evolution te(nfet, sg);

    std::vector<std::pair<int, int>> E_ind(4);
    E_ind[0] = std::make_pair(LC, (int)(te.psi[LC].E0.size() * 0.50));
    E_ind[1] = std::make_pair(LC, (int)(te.psi[LC].E0.size() * 0.75));
    E_ind[2] = std::make_pair(RC, (int)(te.psi[RC].E0.size() * 0.50));
    E_ind[3] = std::make_pair(RC, (int)(te.psi[RC].E0.size() * 0.75));
    movie argo(te, E_ind);

    te.solve();
    te.save();

    return 0;
}
