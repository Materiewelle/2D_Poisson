#ifndef INVERTER_HPP
#define INVERTER_HPP

#include <armadillo>

#include "device.hpp"
#include "time_evolution.hpp"

class inverter {
public:
    device n_fet;
    device p_fet;
    double capacitance;
    steady_state s_n;
    steady_state s_p;
    time_evolution te_n;
    time_evolution te_p;

    inline inverter(const device & n, const device & p, double c);

    inline bool solve(const voltage & V, double & V_o);
    inline void solve(const signal & sg);

    inline void output(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & V_out);
};

//----------------------------------------------------------------------------------------------------------------------

inverter::inverter(const device & n, const device & p, double c)
    : n_fet(n), p_fet(p), capacitance(c) {
}

bool inverter::solve(const voltage & V, double & V_o) {
    auto delta_I = [&] (double V_o) {
        s_n = steady_state(n_fet, {V.s, V.g, V_o});
        s_p = steady_state(p_fet, {V_o, V.g, V.d});

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

void inverter::solve(const signal & sg) {
    double V_out;
    // solve steady state
    if (!solve(sg[0], V_out)) {
        std::cout << "inverter: steady_state did not converge" << std::endl;
        return;
    }

    te_n = time_evolution(s_n, sg);
    te_p = time_evolution(s_p, sg);

    const unsigned & m = te_n.m;
    while (m < sg.N_t) {
        V_out += (te_n.I[m - 1].d() - te_p.I[m - 1].s()) * time_evolution::dt / capacitance;
        te_n.sg[m].d = V_out;
        te_p.sg[m].s = V_out;
        te_n.step();
        te_p.step();
    }
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
