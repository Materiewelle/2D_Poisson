#define ARMA_NO_DEBUG    // no bound checks
//#define GNUPLOT_NOPLOTS

#include <iostream>
#include <iomanip>
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
#include "movie.hpp"
#include "potential.hpp"
#include "steady_state.hpp"
#include "time_evolution.hpp"

using namespace arma;
using namespace std;

int main(int argc, char ** argv) {
    //flush denormal floats to zero for massive speedup
    //(i.e. set bits 15 and 6 in SSE control register MXCSR)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    // set number of threads used by OMP (<= n_core for OpenBlas)
//    omp_set_num_threads(stoi(argv[1]));

    device nfet("nfet", nfet_model, fet_geometry);
//    device tfet("tfet", tfet_model, tfet_geometry);


    time_evolution te(nfet, voltage { 0.0, 0.5, 0.5 });
    vec ramp = linspace(0.5, 0.8, 150);
    for (int i = 2; i < 152; ++i) {
        te.V[i] = { 0.0, 0.5, ramp(i - 2) };
    }
    std::fill(begin(te.V) + 152, end(te.V), voltage { 0.0, 0.5, 0.8 });

    std::vector<std::pair<int, int>> E_ind(4);
    E_ind[0] = std::make_pair(LC, (int)(te.psi[LC].E0.size() * 0.50));
    E_ind[1] = std::make_pair(LC, (int)(te.psi[LC].E0.size() * 0.75));
    E_ind[2] = std::make_pair(RC, (int)(te.psi[RC].E0.size() * 0.50));
    E_ind[3] = std::make_pair(RC, (int)(te.psi[RC].E0.size() * 0.75));

    movie argo(te, E_ind);
    te.solve();
    te.save();

//    vec V_g;
//    vec I;
//    for (int i = 2; i < argc; ++i) {
//        int l_g = stoi(argv[i]);
//        tfet.l_g = l_g;
//        std::stringstream ss;
//        ss << "tfet_lg=" << l_g << "nm";
//        tfet.update(ss.str());
//        steady_state::transfer<false>(tfet, {0.0, 0., 0.4}, 0.8, 150, V_g, I);
//        mat res = join_horiz(V_g, I);
//        res.save("data/" + ss.str(), raw_ascii);
//    }

    return 0;
}
