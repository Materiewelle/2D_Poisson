//#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS

#include <iostream>
#include <algorithm>
#include <omp.h>
#include <xmmintrin.h>
#include <string>
#include <sstream>
#include <armadillo>

#include "brent.hpp"
#include "device.hpp"
#include "gnuplot.hpp"
#include "inverter.hpp"
#include "potential.hpp"
#include "steady_state.hpp"
#include "time_evolution.hpp"
#include "movie.hpp"


using namespace arma;
using namespace std;

int main() {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    device nfet("nfet", nfet_model, standard_geometry);

//    steady_state s(n_fet, {0, 0.2, 0.5});
//    s.solve();
//    wave_packet psi;
//    psi.init< true>(n_fet, s.E[LC], s.W[LC], s.phi);

//    image(psi.E);

//    inverter i(n_fet, p_fet);
//    vec V_in;
//    vec V_out;
//    i.output({0, 0.17, 0.4}, 0.23, 20, V_in, V_out);

//    plot(make_pair(V_in, V_out));

    time_evolution te(nfet, voltage { 0.0, 0.25, 0.4 });
    vec ramp = linspace(0, 0.15, 80);
    for (int i = 2; i < 82; ++i) {
        te.V[i] = {ramp(i-2), 0.25, 0.4};
    }

    std::vector<std::pair<int, int>> E_ind(4);

    E_ind[0] = std::make_pair(LV, te.psi[LV].E0.size() *  1 /  8);
    E_ind[1] = std::make_pair(RV, te.psi[RV].E0.size() *  1 /  8);
    E_ind[2] = std::make_pair(LC, te.psi[LC].E0.size() * 15 / 16);
    E_ind[3] = std::make_pair(RC, te.psi[RC].E0.size() * 15 / 16);

    movie argo(te, E_ind);
    argo.action();

//    vec V_g;
//    vec I;
//    vec l_g = {9, 18, 41, 320};
//    for (auto it = l_g.begin(); it != l_g.end(); ++it) {
//        nfet.l_g = *it;
//        std::stringstream ss;
//        ss << "nfet_transfer_lg=" << std::setprecision(2) << *it;
//        nfet.update(ss.str());
//        steady_state::transfer<false>(nfet, {0.0, -0.2, 0.4}, 0.6, 300, V_g, I);
//        mat res = join_horiz(V_g, I);
//        res.save(ss.str(), raw_ascii);
//    }


    return 0;
}
