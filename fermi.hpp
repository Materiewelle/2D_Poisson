#ifndef FERMI_HPP
#define FERMI_HPP

#include <armadillo>

#include "constant.hpp"

inline double fermi(double E, double F) {
    return 1.0 / (1.0 + std::exp((E - F) * c::e / c::T / c::k_B));
}

inline arma::vec fermi(const arma::vec & E, double F) {
    using namespace arma;

    vec ret(E.size());
    for (unsigned i = 0; i < E.size(); ++i) {
        ret(i) = fermi(E(i), F);
    }

    return ret;
}

template<bool smooth>
inline double fermi(double E, double F, double E0, double slope = 500) {
    double f = fermi(E, F);
    if (smooth) {
        f -= 0.5 - 0.5 * std::tanh((E - E0) * slope);
    } else {
        f -= (E < E0) ? 1.0 : 0.0;
    }
    return f;
}

template<bool smooth>
inline arma::vec fermi(const arma::vec & E, double F, double E0, double slope = 500) {
    using namespace arma;

    vec ret(E.size());
    if (smooth) {
        for (unsigned i = 0; i < E.size(); ++i) {
            ret(i) = fermi<smooth>(E(i), F, E0, slope);
        }
    } else {
        auto E0_it = std::lower_bound(std::begin(E), std::end(E), E0);
        unsigned i = 0;
        for (auto E_it = std::begin(E); E_it != E0_it; ++E_it) {
            ret(i) = fermi(*E_it, F) - 1.0;
            ++i;
        }
        for (auto E_it = E0_it; E_it != std::end(E); ++E_it) {
            ret(i) = fermi(*E_it, F);
            ++i;
        }
    }

    return ret;
}

#endif
