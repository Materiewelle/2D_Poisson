#ifndef SIGNAL_HPP_HEADER
#define SIGNAL_HPP_HEADER

#include <armadillo>
#include <algorithm>
#include <vector>

#include "voltage.hpp"

class signal {
public:
    double T;
    unsigned N_t;
    arma::vec t;
    std::vector<voltage> V;

    inline signal(double TT);
    inline signal(double TT, const voltage & VV);
    inline signal(const std::vector<voltage> & VV);
    inline signal(double TT, const std::vector<double> & t, const std::vector<voltage> & VV);
};

#endif

//----------------------------------------------------------------------------------------------------------------------

#ifndef SIGNAL_HPP_BODY
#define SIGNAL_HPP_BODY

#include "time_evolution.hpp"

signal::signal(double TT)
    : T(TT), N_t(std::round(T / time_evolution::dt)), t(arma::linspace(0, T - time_evolution::dt, N_t)), V(N_t) {
}

signal::signal(double TT, const voltage & VV)
    : signal(TT) {
    std::fill(V.begin(), V.end(), VV);
}

signal::signal(const std::vector<voltage> & VV)
    : T(VV.size() * time_evolution::dt), N_t(VV.size()), t(arma::linspace(0, T - time_evolution::dt, N_t)), V(VV) {
}

signal::signal(double TT, const std::vector<double> & t, const std::vector<voltage> & VV)
    : signal(TT) {
    std::vector<int> idx(t.size());

    for (unsigned i = 0; i < t.size(); ++i) {
        idx[i] = std::round(t[i] / time_evolution::dt);
    }

    int j0;
    int j1 = 0;
    voltage V0;
    voltage V1 = VV[0];
    for (unsigned i = 0; i <= idx.size(); ++i) {
        j0 = j1;
        j1 = (i < idx.size()) ? idx[i] : N_t;
        V0 = V1;
        V1 = (i < idx.size()) ? VV[i] : VV[i - 1];
        for (int j = j0; j < j1; ++j) {
            double r = (double)(j - j0) / ((double)(j1 - j0));
            V[j] = V0 * (1.0 - r) + V1 * r;
        }
    }
}

#endif
