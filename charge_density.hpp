#ifndef CHARGE_DENSITY_HPP
#define CHARGE_DENSITY_HPP

#include <armadillo>
#include <stack>

// forward declarations
#ifndef POTENTIAL_HPP
class potential;
#endif
#ifndef WAVE_PACKET_HPP
class wave_packet;
#endif

class charge_density {
public:
    static constexpr int initial_waypoints = 30;
    arma::vec data;

    inline charge_density();

    inline void update(const potential & phi, arma::vec E[4], arma::vec W[4]);
    inline void update(const wave_packet psi[4]);
};

// rest of includes
#include "constant.hpp"
#include "device.hpp"
#include "fermi.hpp"
#include "green.hpp"
#include "integral.hpp"
#include "potential.hpp"
#include "wave_packet.hpp"
#include "gnuplot.hpp"

//----------------------------------------------------------------------------------------------------------------------

namespace charge_density_impl {

    static inline arma::vec get_bound_states(const potential & phi);
    static inline arma::vec get_bound_states(const potential & phi, double E0, double E1);
    template<bool zero_check = true>
    static inline int eval(const arma::vec & a, const arma::vec & a2, const arma::vec & b, double E);

    template<bool source>
    static inline arma::vec get_A(const potential & phi, double E);

}

//----------------------------------------------------------------------------------------------------------------------

charge_density::charge_density()
    : data(d::N_x) {
    data.fill(0.0);
}

void charge_density::update(const potential & phi, arma::vec E[4], arma::vec W[4]) {
    using namespace arma;
    using namespace charge_density_impl;

    auto E_bound = get_bound_states(phi);

    // get integration intervals
    auto get_intervals = [&] (double E_min, double E_max) {
        vec lin = linspace(E_min, E_max, initial_waypoints);

        if ((E_bound.size() > 0) && (E_bound(0) < E_max) && (E_bound(E_bound.size() - 1) > E_min)) {
            vec ret = vec(E_bound.size() + lin.size());

            // indices
            unsigned i0 = 0;
            unsigned i1 = 0;
            unsigned j = 0;

            // linear search, could be optimized to binary search
            while(E_bound(i1) < E_min) {
                ++i1;
            }

            // merge lin and E_bound
            while ((i0 < lin.size()) && (i1 < E_bound.size())) {
                if (lin(i0) < E_bound(i1)) {
                    ret(j++) = lin(i0++);
                } else {
                    ret(j++) = E_bound(i1++);
                }
            }

            // rest of lin, rest of E_bound out of range
            while(i0 < lin.size()) {
                ret(j++) = lin(i0++);
            }

            ret.resize(j);
            return ret;
        } else {
            return lin;
        }
    };
    vec i_sv = get_intervals(phi.s() + d::E_min, phi.s() - 0.5 * d::E_gc);
    vec i_sc = get_intervals(phi.s() + 0.5 * d::E_gc, phi.s() + d::E_max);
    vec i_dv = get_intervals(phi.d() + d::E_min, phi.d() - 0.5 * d::E_gc);
    vec i_dc = get_intervals(phi.d() + 0.5 * d::E_gc, phi.d() + d::E_max);

    // calculate charge density
    auto n_sv = integral<d::N_x>([&] (double E) -> vec {
        return get_A<true>(phi, E) * (fermi(E - phi.s(), d::F_sc) - 1);
    }, i_sv, d::rel_tol, c::epsilon(), E[LV], W[LV]);
    auto n_dv = integral<d::N_x>([&] (double E) -> vec {
        return get_A<false>(phi, E) * (fermi(E - phi.d(), d::F_dc) - 1);
    }, i_dv, d::rel_tol, c::epsilon(), E[RV], W[RV]);
    auto n_sc = integral<d::N_x>([&] (double E) -> vec {
        return get_A<true>(phi, E) * (fermi(E - phi.s(), d::F_sc));
    }, i_sc, d::rel_tol, c::epsilon(), E[LC], W[LC]);
    auto n_dc = integral<d::N_x>([&] (double E) -> vec {
        return get_A<false>(phi, E) * (fermi(E - phi.d(), d::F_dc));
    }, i_dc, d::rel_tol, c::epsilon(), E[RC], W[RC]);

    // multiply weights with fermi function
    for (unsigned i = 0; i < E[LV].size(); ++i) {
        W[LV](i) *= (1.0 - fermi(E[LV](i) - phi.s(), d::F_sc));
    }
    for (unsigned i = 0; i < E[RV].size(); ++i) {
        W[RV](i) *= (1.0 - fermi(E[RV](i) - phi.d(), d::F_dc));
    }
    for (unsigned i = 0; i < E[LC].size(); ++i) {
        W[LC](i) *= fermi(E[LC](i) - phi.s(), d::F_sc);
    }
    for (unsigned i = 0; i < E[RC].size(); ++i) {
        W[RC](i) *= fermi(E[RC](i) - phi.d(), d::F_dc);
    }

    // scaling factor
    static constexpr double scale = - c::e * 4 / M_PI / M_PI / d::dx / d::d_g / d::d_g;

    // scaling and doping
    data = (n_sv + n_sc + n_dv + n_dc) * scale + d::n0;
}

void charge_density::update(const wave_packet psi[4]) {
    using namespace arma;

    // get abs(psi)²
    auto get_abs = [] (const cx_mat & m) {
        mat ret(m.n_rows / 2, m.n_cols);
        auto ptr0 = m.memptr();
        auto ptr1 = ret.memptr();
        for (unsigned i = 0; i < m.n_elem; i += 2) {
            (*ptr1++) = std::norm(ptr0[i]) + std::norm(ptr0[i + 1]);
        }
        return ret;
    };

    vec n[4];
    for (int i = 0; i < 4; ++i) {
        n[i] = get_abs(psi[i].data) * psi[i].W;
    }

    // scaling factor
    static constexpr double scale = - c::e * 4 / M_PI / M_PI / d::dx / d::d_g / d::d_g;

    // scaling and doping
    data = (- n[LV] - n[RV] + n[LC] + n[RC]) * scale + d::n0;
}

arma::vec charge_density_impl::get_bound_states(const potential & phi) {
    double phi0, phi1, phi2, limit;

    // check for bound states in valence band
    phi0 = arma::min(phi.data(d::s)) - 0.5 * d::E_g;
    phi1 = arma::max(phi.data(d::g)) - 0.5 * d::E_g;
    phi2 = arma::min(phi.data(d::d)) - 0.5 * d::E_g;
    limit = phi0 > phi2 ? phi0 : phi2;
    if (limit < phi1) {
        return get_bound_states(phi, limit, phi1);
    }

    // check for bound states in conduction band
    phi0 = arma::max(phi.data(d::s)) + 0.5 * d::E_g;
    phi1 = arma::min(phi.data(d::g)) + 0.5 * d::E_g;
    phi2 = arma::max(phi.data(d::d)) + 0.5 * d::E_g;
    limit = phi0 < phi2 ? phi0 : phi2;
    if (limit > phi1) {
        return get_bound_states(phi, phi1, limit);
    }

    return arma::vec(arma::uword(0));
}

arma::vec charge_density_impl::get_bound_states(const potential & phi, double E0, double E1) {
    using namespace arma;

    static constexpr double tol = 1e-10;

    span range{d::s2.a, d::d2.b};
    vec a = d::t_vec(range);
    vec a2 = a % a;
    vec b = phi.twice(range);

    double E2;
    int i0, i1;
    int s0, s1, s2;

    s0 = eval(a, a2, b, E0);
    s1 = eval(a, a2, b, E1);

    // check if no bound states in this interval
    if (s1 - s0 == 0) {
        return vec(uword(0));
    }

    unsigned n = 2;
    vec E = vec(1025);
    ivec s = ivec(1025);
    E(0) = E0;
    E(1) = E1;
    s(0) = s0;
    s(1) = s1;

    unsigned n_bound = 0;
    vec E_bound(100);

    // stack for recursion
    std::stack<std::pair<int, int>> stack;

    // push first interval to stack
    stack.push(std::make_pair(0, 1));

    // repeat until all intervals inspected
    while (!stack.empty()) {
        const auto & i = stack.top();
        i0 = i.first;
        i1 = i.second;

        stack.pop();

        // load data
        E0 = E(i0);
        E1 = E(i1);
        s0 = s(i0);
        s1 = s(i1);

        // mid energy
        E2 = 0.5 * (E0 + E1);

        // if interval size sufficiently small enough, add new bound state
        if (E1 - E0 <= tol) {
            if (E_bound.size() <= n_bound) {
                E_bound.resize(n_bound * 2);
            }
            E_bound(n_bound++) = E2;
        } else {
            // evaluate s at mid energy
            s2 = eval(a, a2, b, E2);

            // add intervals to stack if they contain bound states
            if (s1 - s2 > 0) {
                stack.push(std::make_pair(n, i1));
            }
            if (s2 - s0 > 0) {
                stack.push(std::make_pair(i0, n));
            }

            // save E2 and s2
            if (E.size() <= n) {
                E.resize(2 * n - 1);
                s.resize(2 * n - 1);
            }
            E(n) = E2;
            s(n) = s2;
            ++n;
        }
    }

    E_bound.resize(n_bound);
    return E_bound;
}

template<bool zero_check = true>
int charge_density_impl::eval(const arma::vec & a, const arma::vec & a2, const arma::vec & b, double E) {
    int n = b.size();

    static const double eps = c::epsilon();

    // first iteration (i = 0)
    double q;
    double q0 = b[0] - E;
    int s = q0 < 0 ? 1 : 0;

    // start with i = 1
    for (int i = 1; i < n; ++i) {
        if (zero_check && (q0 == 0)) {
            q = b[i] - E - a[i - 1] / eps;
        } else {
            q = b[i] - E - a2[i - 1] / q0;
        }

        q0 = q;
        if (q < 0) {
            ++s;
        }
    }

    return s;
}

template<bool source>
arma::vec charge_density_impl::get_A(const potential & phi, const double E) {
    using namespace arma;

    // calculate 1 column of green's function
    cx_double Sigma_s, Sigma_d;
    cx_vec G = green_col<source>(phi, E, Sigma_s, Sigma_d);

    // get spectral function for each orbital (2 values per unit cell)
    vec A_twice;
    if (source) {
        A_twice = std::abs(2 * Sigma_s.imag()) * real(G % conj(G)); // G .* conj(G) = abs(G).^2
    } else {
        A_twice = std::abs(2 * Sigma_d.imag()) * real(G % conj(G));
    }

    // reduce spectral function to 1 value per unit cell (simple addition of both values)
    vec A = vec(d::N_x);
    for (unsigned i = 0; i < A.size(); ++i) {
        A(i) = A_twice(2 * i) + A_twice(2 * i + 1);
    }

    return A;
}

#endif

