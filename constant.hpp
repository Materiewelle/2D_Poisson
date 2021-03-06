#ifndef CONSTANT_HPP
#define CONSTANT_HPP

#include <cmath>

namespace c {

    static constexpr double eps_0  = 8.854187817E-12; // vacuum permittivity
    static constexpr double e      = 1.602176565E-19; // elementary charge
    static constexpr double h      = 6.62606957E-34;  // Planck constant
    static constexpr double h_bar  = h / 2 / M_PI;    // reduced Planck's constant
    static constexpr double h_bar2 = h_bar * h_bar;   // ħ²
    static constexpr double k_B    = 1.3806488E-23;   // Boltzmann constant
    static constexpr double m_e    = 9.10938188E-31;  // electron mass
    static constexpr double T      = 300;             // Temperature in Kelvin

    static inline constexpr double epsilon(double x = 1.0) {
        return std::nextafter(x, x + 1.0) - x;
    }

}

#endif
