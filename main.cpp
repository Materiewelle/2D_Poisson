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
    omp_set_num_threads(stoi(argv[1]));

//    device nfet("nfet", nfet_model, fet_geometry);
    device tfet("tfet", tfet_model, tfet_geometry);

//    voltage V { 0.0, 0.5, 0.4 };
//    steady_state s(tfet, V);
//    s.solve();
//    plot_phi2D(tfet, V, s.n);
//    plot_ldos(tfet, s.phi);
//    potential phi(nfet, V);
//    plot_ldos(nfet, phi);
//    plot_phi2D(tfet, V);
//    phi = potential(tfet, V);
//    plot_ldos(tfet, phi);
//    return 0;

//    steady_state s(n_fet, {0, 0.2, 0.5});
//    s.solve();
//    wave_packet psi;
//    psi.init< true>(n_fet, s.E[LC], s.W[LC], s.phi);

//    image(psi.E);

//    time_evolution te(nfet, voltage { 0.0, 0.25, 0.4 });
//    vec ramp = linspace(0, 0.15, 80);
//    for (int i = 2; i < 82; ++i) {
//        te.V[i] = {ramp(i-2), 0.25, 0.4};
//    }

//    std::vector<std::pair<int, int>> E_ind(4);

//    E_ind[0] = std::make_pair(LV, te.psi[LV].E0.size() *  1 /  8);
//    E_ind[1] = std::make_pair(RV, te.psi[RV].E0.size() *  1 /  8);
//    E_ind[2] = std::make_pair(LC, te.psi[LC].E0.size() * 15 / 16);
//    E_ind[3] = std::make_pair(RC, te.psi[RC].E0.size() * 15 / 16);

//    movie argo(te, E_ind);
//    argo.action();

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


    time_evolution te(tfet, voltage { 0.0, 0.3, 0.5 });
    vec ramp = linspace(0.3, 0.4, 100);
    for (int i = 2; i < 22; ++i) {
        te.V[i] = {0.0, ramp(i - 2), 0.5};
    }

    std::vector<std::pair<int, int>> E_ind(4);

    E_ind[0] = std::make_pair(LV, te.psi[LV].E0.size() *  1 /  4);
    E_ind[1] = std::make_pair(RV, te.psi[RV].E0.size() *  1 /  4);
    E_ind[2] = std::make_pair(LC, te.psi[LC].E0.size() *  1 /  4);
    E_ind[3] = std::make_pair(RC, te.psi[RC].E0.size() *  1 /  4);

    movie argo(te, E_ind);
    argo.action();

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
