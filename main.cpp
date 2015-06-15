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
//    omp_set_num_threads(stoi(argv[0]));

//    device nfet("nfet", nfet_model, fet_geometry);
    device tfet("tfet", tfet_model, tfet_geometry);

//    inverter i(n_fet, p_fet);
//    vec V_in;
//    vec V_out;
//    i.output({0, 0.17, 0.4}, 0.23, 20, V_in, V_out);

//    plot(make_pair(V_in, V_out));

    time_evolution te(tfet, voltage { 0.0, 0.3, 0.5 });
    vec ramp = linspace(0.3, 0.5, 150);
    for (int i = 2; i < 152; ++i) {
        te.V[i] = {0.0, ramp(i - 2), 0.5};
    }
    std::fill(begin(te.V) + 152, end(te.V), voltage { 0.0, 0.5, 0.5 });

    std::vector<std::pair<int, int>> E_ind(64);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 16; ++j) {
            E_ind[i * 16 +  j] = std::make_pair(i, te.psi[i].E0.size() * j / 16);
        }
    }

    movie argo(te, E_ind);
    argo.action();

//    vec V_g;
//    vec I;
//    for (int i = 1; i < argc; ++i) {
//        int l_sox = stoi(argv[i]);
//        tfet.l_sox = l_sox;
//        tfet.l_sg = 20 - l_sox;
//        std::stringstream ss;
//        ss << "tfet_overlap=" << l_sox << "nm";
//        tfet.update(ss.str());
//        steady_state::transfer<true>(tfet, {0.0, 0., 0.4}, 0.8, 200, V_g, I);
//        mat res = join_horiz(V_g, I);
//        res.save("data/" + ss.str(), raw_ascii);
//    }

    return 0;
}
