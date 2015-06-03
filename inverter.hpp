#ifndef INVERTER_HPP
#define INVERTER_HPP

#include <armadillo>

#include "brent.hpp"
#include "device.hpp"
#include "voltage.hpp"
#include "steady_state.hpp"

class inverter {
public:
    device n_fet;
    device p_fet;

    inline inverter(const device & n, const device & p);

    inline bool solve(const voltage & V, double & V_o);

    inline void output(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & V_out);
};

//----------------------------------------------------------------------------------------------------------------------

inverter::inverter(const device & n, const device & p)
    : n_fet(n), p_fet(p) {
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

