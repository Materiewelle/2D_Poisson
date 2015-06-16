#ifndef INVERTER_HPP
#define INVERTER_HPP

#include <armadillo>

#include "device.hpp"

class inverter {
public:
    device n_fet;
    device p_fet;
    double capacitance;

    inline inverter(const device & n, const device & p, double c);

    inline bool solve(const voltage & V, double & V_o);

    inline void solve(const std::vector<voltage> & V, std::vector<double> & V_out);

    inline void output(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & V_out);
};

//----------------------------------------------------------------------------------------------------------------------

inverter::inverter(const device & n, const device & p, double c)
    : n_fet(n), p_fet(p), capacitance(c) {
}

bool inverter::solve(const voltage & V, double & V_o) {

    auto delta_I = [&] (double V_o) {

        steady_state s_n(n_fet, {V.s, V.g, V_o});
        steady_state s_p(p_fet, {V_o, V.g, V.d});

        std::cout << "n: " << V.s << ", " << V.g << ", " << V_o << ": ";
        std::flush(std::cout);
        s_n.solve();
        std::cout << "p: " << V_o << ", " << V.g << ", " << V.d << ": ";
        std::flush(std::cout);
        s_p.solve();

        return s_n.I.total(0) - s_p.I.total(0);
    };

    return brent(delta_I, -0.2, 0.6, 0.0005, V_o);
}

void inverter::solve(const std::vector<voltage> & V, std::vector<double> & V_out) {
    V_out.resize(V.size());

    // solve steady state
    steady_state s_n(n_fet, {0.0, 0.0, 0.0});
    steady_state s_p(p_fet, {0.0, 0.0, 0.0});
    auto delta_I = [&] (double V_o) {

        s_n = steady_state(n_fet, { V[0].s, V[0].g, V_o    });
        s_p = steady_state(p_fet, { V_o   , V[0].g, V[0].d });

        std::cout << "n: " << V[0].s << ", " << V[0].g << ", " << V_o << ": ";
        std::flush(std::cout);
        s_n.solve();
        std::cout << "p: " << V_o << ", " << V[0].g << ", " << V[0].d << ": ";
        std::flush(std::cout);
        s_p.solve();

        return s_n.I.total(0) - s_p.I.total(0);
    };
    brent(delta_I, -1.0, 1.0, 0.00001, V_out[0]);

//    time_evolution te_n(s_n);
//    time_evolution te_p(s_p);

//    unsigned & m = te_n.m;
//    while (m < V.size()) {
//        V_out[m] = V_out[m - 1] + (te_n.I[m - 1].d() - te_p.I[m - 1].s()) * t::dt / capacitance;
//        te_n.V[m] = { te_n.V[m - 1].s, te_n.V[m - 1].g,        V_out[m] };
//        te_p.V[m] = {        V_out[m], te_p.V[m - 1].g, te_p.V[m - 1].d };
//        te_n.step();
//        te_p.step();
//    }
}

void inverter::output(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & V_out) {
    V_g = arma::linspace(V0.g, V_g1, N);
    V_out = arma::vec(N);

    for (int i = 0; i < N; ++i) {
        if (solve({V0.s, V_g(i), V0.d}, V_out(i))) {
            std::cout << V_g(i) << ": " << V_out(i) << std::endl;
        } else {
            std::cout << V_g(i) << ": ERROR!" << std::endl;
        }
    }
}

#endif
