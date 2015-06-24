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

    inline signal();
    inline signal(double TT);
    inline signal(double TT, const voltage & VV);
    inline signal(const std::vector<voltage> & VV);
    inline signal(double TT, const std::vector<double> & t, const std::vector<voltage> & VV);

    inline voltage & operator[](int index);
    inline const voltage & operator[](int index) const;
};

static inline signal const_signal(double T, const voltage & V0);
static inline signal linear_signal(double T, const std::vector<double> & t, const std::vector<voltage> & V);
static inline signal sine_signal(double T, const voltage & V0, const voltage & V_a, const tripled & f, const tripled & t0 = 0, const tripled & ph = 0);

#endif

//----------------------------------------------------------------------------------------------------------------------

#ifndef SIGNAL_HPP_BODY
#define SIGNAL_HPP_BODY

#include "time_evolution.hpp"

signal::signal() {
}

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

voltage & signal::operator[](int index) {
    return V[index];
}

const voltage & signal::operator[](int index) const {
    return V[index];
}

signal const_signal(double T, const voltage & V0) {
    signal s(T);

    for (unsigned i = 0; i < s.N_t; ++i) {
        s[i] = V0;
    }

    return s;
}

signal linear_signal(double T, const std::vector<double> & t, const std::vector<voltage> & V) {
    signal s(T);

    std::vector<int> idx(t.size());

    for (unsigned i = 0; i < t.size(); ++i) {
        idx[i] = std::round(t[i] / time_evolution::dt);
    }

    int j0;
    int j1 = 0;
    voltage V0;
    voltage V1 = V[0];
    for (unsigned i = 0; i <= idx.size(); ++i) {
        j0 = j1;
        j1 = (i < idx.size()) ? idx[i] : s.N_t;
        V0 = V1;
        V1 = (i < idx.size()) ? V[i] : V[i - 1];
        for (int j = j0; j < j1; ++j) {
            double r = (double)(j - j0) / ((double)(j1 - j0));
            s[j] = V0 * (1.0 - r) + V1 * r;
        }
    }

    return s;
}

signal sine_signal(double T, const voltage & V0, const voltage & V_a, const tripled & f, const tripled & t0, const tripled & ph) {
    signal s(T);

    tripled w = 2 * M_PI * f;

    for (unsigned i = 0; i < s.N_t; ++i) {
        double t = i * time_evolution::dt;

        auto cutoff = func([&] (double t0i) {
            return (t < t0i) ? 0.0 : 1.0;
        }, t0);

        auto sin = [] (double x) -> double {
            return std::sin(x);
        };

        s[i] = V0 + cutoff * V_a * (func(sin, w * (t - t0) + ph) - func(sin, ph));
    }

    return s;
}

#endif
