#ifndef INVERTER_HPP
#define INVERTER_HPP

#include <armadillo>

#include "device.hpp"
#include "voltage.hpp"
#include "steady_state.hpp"

class inverter {
public:
    device n_fet;
    device p_fet;

    inline inverter(const device & n, const device & p);

    inline double solve(const voltage & V);

    inline void output(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & V_out);
};

//----------------------------------------------------------------------------------------------------------------------

inverter::inverter(const device & n, const device & p)
    : n_fet(n), p_fet(p) {
}

double inverter::solve(const voltage & V) {
    static constexpr double tol = 0.05;

    double V_0 = -0.2;
    double V_1 = 0.6;

    auto delta_I = [&] (double V_o) {

        steady_state s_n(n_fet, {V.s, V.g, V_o});
        steady_state s_p(p_fet, {V_o, V.g, V.d});

        std::cout << "n: " << V.s << ", " << V.g << ", " << V_o << ": ";
        std::flush(std::cout);
        s_n.solve();
        std::cout << "p: " << V_o << ", " << V.g << ", " << V.d << ": ";
        std::flush(std::cout);
        s_p.solve();

        return s_n.I.total(0) + s_p.I.total(0);
    };

//    arma::vec V_test = arma::linspace(V_0, V_1, 25);
//    arma::vec I_testn(V_test.size());
//    arma::vec I_testp(V_test.size());
//    for (int i = 0; i < V_test.size(); ++i) {
//        steady_state s_n(n_fet, {V.s, V.g, V_test(i)});
//        steady_state s_p(p_fet, {V_test(i), V.g, V.d});
//        std::cout << "n: " << V.s << ", " << V.g << ", " << V_test(i) << ": ";
//        std::flush(std::cout);
//        s_n.solve();
//        std::cout << "p: " << V_test(i) << ", " << V.g << ", " << V.d << ": ";
//        std::flush(std::cout);
//        s_p.solve();
//        I_testn(i) = s_n.I.total(0);
//        I_testp(i) = s_p.I.total(0);
//        if ((i == 0) || (i == V_test.size() - 1)) {
//            plot_ldos(n_fet, s_n.phi);
//            plot_ldos(p_fet, s_p.phi);
//        }
//    }
//    plot(std::make_pair(V_test, I_testn));
//    plot(std::make_pair(V_test, I_testp));
//    return 0;

    double dI_0 = delta_I(V_0);
    double dI_1 = delta_I(V_1);

    double sgn = dI_0 > 0;
    if ((sgn && (dI_1 > 0)) || (!sgn && (dI_1 < 0))) {
        std::cout << "ERROR!" << std::endl;
        return 0;
    }

    double dI_2 = 1000;
    while ((V_1 - V_0 > tol) && (dI_2 > 1e-10)) {
        double V_2 = V_0 - dI_0 * (V_1 - V_0) / (dI_1 - dI_0);
        double dI_2 = delta_I(V_2);

        if ((sgn && (dI_2 > 0)) || (!sgn && (dI_2 < 0))) {
            V_0 = V_2;
            dI_0 = dI_2;
        } else {
            V_1 = V_2;
            dI_1 = dI_2;
        }
    }

    return V_0 - dI_0 * (V_1 - V_0) / (dI_1 - dI_0);
}

void inverter::output(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & V_out) {
    V_g = arma::linspace(V0.g, V_g1, N);

    for (int i = 0; i < N; ++i) {
        V_out(i) = solve({V0.s, V_g(i), V0.d});
    }
}

#endif

