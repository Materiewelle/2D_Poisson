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

#endif
