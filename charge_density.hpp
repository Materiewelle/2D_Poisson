#ifndef CHARGE_DENSITY_HPP_HEADER
#define CHARGE_DENSITY_HPP_HEADER

#include <armadillo>
#include <stack>
#include <unordered_map>

// forward declarations
class potential;
class wave_packet;

class charge_density {
public:

    // The total current can be understood to consist of 4 parts:
    arma::vec lv; // coming from left, states in the valence band
    arma::vec rv; // coming from right, states in the valence band
    arma::vec lc; // coming from left, states in the conduction band
    arma::vec rc; // coming from right, states in the conduction band
    arma::vec total;

    inline charge_density();
    inline charge_density(const device & d, const potential & phi, arma::vec E[4], arma::vec W[4]);
    inline charge_density(const device & d, const wave_packet psi[4], const potential & phi);
};

namespace charge_density_impl {

    static constexpr int initial_waypoints = 40; // minimum number of integration intervalls

    // Energy-integration boundaries above and below the source/drain potential
    static constexpr double E_min = -1.5;
    static constexpr double E_max = +1.5;

    static constexpr double rel_tol = 8e-3; // tolerated relative error in adaptive simpson integration

    static inline arma::vec get_bound_states(const device & d, const potential & phi);
    static inline arma::vec get_bound_states(const device & d, const potential & phi, double E0, double E1);
    template<bool zero_check = true>
    static inline int eval(const arma::vec & a, const arma::vec & a2, const arma::vec & b, double E);

    template<bool source>
    static inline arma::vec get_A(const device & d, const potential & phi, double E);

    static inline arma::vec & get_n0(const device & d);
    static std::unordered_map<std::string, arma::vec> n0_d;

}

#endif

//----------------------------------------------------------------------------------------------------------------------

#ifndef CHARGE_DENSITY_HPP_BODY
#define CHARGE_DENSITY_HPP_BODY

charge_density::charge_density() {
}

charge_density::charge_density(const device & d, const potential & phi, arma::vec E[4], arma::vec W[4]) {
    /* This is the main method for computing the charge density in steady state calculations. The Energy
     * values at which the spectral function was evaluated by the adative simpson integration algorithm
     * and the curresponding weights at theese points are stored in E and W, respectively. */

    using namespace arma;
    using namespace charge_density_impl;

    // get a prediction of bound-state levels in the channel region
    auto E_bound = get_bound_states(d, phi);

    // get integration intervals
    auto get_intervals = [&] (double E_min, double E_max) {
        /* This lambda creates an initial set of integration intervals. If bound states
         * have been predicted, narrow intervals around them are also created in order to
         * prevent the adaptive integration scheme from missing them. */

        vec lin = linspace(E_min, E_max, initial_waypoints);

        // merge the bound-state intervals and the initial intervals together in case that boundstates have been predicted
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
    vec i_sv = get_intervals(phi.s() + E_min, phi.s() - 0.5 * d.E_gc);
    vec i_sc = get_intervals(phi.s() + 0.5 * d.E_gc, phi.s() + E_max);
    vec i_dv = get_intervals(phi.d() + E_min, phi.d() - 0.5 * d.E_gc);
    vec i_dc = get_intervals(phi.d() + 0.5 * d.E_gc, phi.d() + E_max);

    // calculate charge density
    auto I_s = [&] (double E) -> vec {
        // spectral funtion in source
        vec A = get_A<true>(d, phi, E);
        double f = fermi(E - phi.s(), d.F_sc);
        for (int i = 0; i < d.N_x; ++i) {
            /* check for each point wether the particle is counted
             * as an electron or a hole (E below branching point) */
            A(i) *= fermi<true>(f, E - phi(i));
        }
        return A;
    };
    auto I_d = [&] (double E) -> vec {
        vec A = get_A<false>(d, phi, E);
        // spectral funtion in drain
        double f = fermi(E - phi.d(), d.F_dc);
        for (int i = 0; i < d.N_x; ++i) {
            A(i) *= fermi<true>(f, E - phi(i));
        }
        return A;
    };

    /* integrate the lambdas defined above over the given energy-ranges.
     * keep the energy-values at which the spectral function has been evaluated
       and the associated weights in the sum. */
    lv = integral(I_s, d.N_x, i_sv, rel_tol, c::epsilon(), E[LV], W[LV]);
    rv = integral(I_d, d.N_x, i_dv, rel_tol, c::epsilon(), E[RV], W[RV]);
    lc = integral(I_s, d.N_x, i_sc, rel_tol, c::epsilon(), E[LC], W[LC]);
    rc = integral(I_d, d.N_x, i_dc, rel_tol, c::epsilon(), E[RC], W[RC]);

    // scaling
    double scale = - 0.5 * c::e / M_PI / M_PI / d.r_cnt / d.dr / d.dx;
    lv *= scale;
    rv *= scale;
    lc *= scale;
    rc *= scale;

    // calculate total charge density and add the background charge introduced by dopands
    total = lv + rv + lc + rc + get_n0(d);
}

charge_density::charge_density(const device & d, const wave_packet psi[4], const potential & phi)
    : lv(d.N_x), rv(d.N_x), lc(d.N_x), rc(d.N_x) {
    /* The overloaded version of the charge-density calculation method which is based on wave-functions.
     * It is used in time-dependent simulations. */

    using namespace arma;
    using namespace charge_density_impl;

    auto get_n = [&d, &phi] (const wave_packet & psi, vec & n) {
        /* this lambda performs the actual computation of the charge-density
         * from a set of wave-functions. */
        n.fill(0.0);

        #pragma omp parallel
        {
            vec n_thread(n.size());
            n_thread.fill(0.0);

            /* loop over all energies. Computation for each energy
             * value is independent, so the loop is inheritely parallel */
            #pragma omp for schedule(static) nowait
            for (unsigned i = 0; i < psi.E0.size(); ++i) {
                // get fermi factor and weight
                double f = psi.F0(i);
                double W = psi.W(i);
                for (int j = 0; j < d.N_x; ++j) { // loop over all unit cells
                    double a = std::norm((*psi.data)(2 * j    , i)); // first orbital
                    double b = std::norm((*psi.data)(2 * j + 1, i)); // second orbital
//                    n_thread(j) += (a + b) * W * ((psi.E(j, i) >= phi(j)) ? f : (f - 1));
                    // multiply with electron/hole occupation
                    n_thread(j) += (a + b) * W * fermi<true>(f, psi.E(j, i) - phi(j));
                }
            }
            // no implied barrier (nowait clause)

            #pragma omp critical
            {
                n += n_thread;
            }
        }
    };

    // call the lambda defined above for every contribution
    get_n(psi[LV], lv);
    get_n(psi[RV], rv);
    get_n(psi[LC], lc);
    get_n(psi[RC], rc);

    // the charge is distributed on the tube's surface. Scale accordingly...
    double scale = - 0.5 * c::e / M_PI / M_PI / d.r_cnt / d.dr / d.dx;
    lv *= scale;
    rv *= scale;
    lc *= scale;
    rc *= scale;

    total = lv + rv + lc + rc + get_n0(d);
}

arma::vec charge_density_impl::get_bound_states(const device & d, const potential & phi) {
    /* This method tries to make a prediction of possible bound-state levels in the gate-region
     * by solving for the closed system's hamiltonian's eigenvalues. */

    double phi0, phi1, phi2, limit;

    // check for bound states in valence band
    // ----------------------------------------------
    /* first get the energy-range in which bound
     * states are even possible. */
    phi0 = arma::min(phi.data(d.sg)) - 0.5 * d.E_g;
    phi1 = arma::max(phi.data(d.g )) - 0.5 * d.E_g;
    phi2 = arma::min(phi.data(d.dg)) - 0.5 * d.E_g;
    limit = phi0 > phi2 ? phi0 : phi2;
    if (limit < phi1) {
        return get_bound_states(d, phi, limit, phi1);
    }
    // ----------------------------------------------

    // check for bound states in conduction band
    // ----------------------------------------------
    phi0 = arma::max(phi.data(d.sg)) + 0.5 * d.E_g;
    phi1 = arma::min(phi.data(d.g )) + 0.5 * d.E_g;
    phi2 = arma::max(phi.data(d.dg)) + 0.5 * d.E_g;
    limit = phi0 < phi2 ? phi0 : phi2;
    if (limit > phi1) {
        return get_bound_states(d, phi, phi1, limit);
    }
    // ----------------------------------------------

    return arma::vec(arma::uword(0));
}

arma::vec charge_density_impl::get_bound_states(const device & d, const potential & phi, double E0, double E1) {
    /* This overloaded method implements the actual bound-state prediction algorithm */

    using namespace arma;

    static constexpr double tol = 1e-10;

    span range{d.sg2.a, d.dg2.b}; // only look at the gated region

    // for building the hamiltonian only in the gated region
    vec a = d.t_vec(range);
    vec a2 = a % a;
    vec b = phi.twice(range);

    double E2;
    int i0, i1;
    int s0, s1, s2;

    s0 = eval(a, a2, b, E0);
    s1 = eval(a, a2, b, E1);

    /* Check if no bound states in this interval.
     * This case needs to be handeled separately. */
    if (s1 - s0 == 0) {
        return vec(uword(0));
    }

    /* ----------------------------------------------
     * In the following, Sturm's theorem is used to
     * find the smallest magnitude eigenvalues of H
     * in the given range up to a certain accurancy.
     * ---------------------------------------------- */

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
arma::vec charge_density_impl::get_A(const device & d, const potential & phi, const double E) {
    using namespace arma;

    // calculate 1 column of green's function
    cx_double Sigma_s, Sigma_d;
    cx_vec G = green_col<source>(d, phi, E, Sigma_s, Sigma_d);

    // get spectral function for each orbital (2 values per unit cell)
    vec A_twice;
    if (source) {
        A_twice = std::abs(2 * Sigma_s.imag()) * real(G % conj(G)); // G .* conj(G) = abs(G).^2
    } else {
        A_twice = std::abs(2 * Sigma_d.imag()) * real(G % conj(G));
    }

    // reduce spectral function to 1 value per unit cell (simple addition of both values)
    vec A = vec(d.N_x);
    for (unsigned i = 0; i < A.size(); ++i) {
        A(i) = A_twice(2 * i) + A_twice(2 * i + 1);
    }

    return A;
}

arma::vec & charge_density_impl::get_n0(const device & d) {
    using namespace arma;

    auto it = n0_d.find(d.name);
    if (it != std::end(n0_d)) {
        return it->second;
    }

    vec x0, x1, x2, x3, w0, w1, w2, w3;

    // valence band in contact region
    vec nvc = integral([&] (double E) {
        double dos = E / sqrt(4*d.tc1*d.tc1*d.tc2*d.tc2 - (E*E - d.tc1*d.tc1 - d.tc2*d.tc2) * (E*E - d.tc1*d.tc1 - d.tc2*d.tc2));
        vec ret = arma::vec(2);
        ret(0) = (1 - fermi(E, d.F_sc)) * dos;
        ret(1) = (1 - fermi(E, d.F_dc)) * dos;
        return ret;
    }, 2, linspace(E_min, -0.5 * d.E_gc, 100), rel_tol, c::epsilon(), x0, w0);

    // conduction band in contact region
    vec ncc = integral([&] (double E) {
        double dos = E / sqrt(4*d.tc1*d.tc1*d.tc2*d.tc2 - (E*E - d.tc1*d.tc1 - d.tc2*d.tc2) * (E*E - d.tc1*d.tc1 - d.tc2*d.tc2));
        vec ret = arma::vec(2);
        ret(0) = fermi(E, d.F_sc) * dos;
        ret(1) = fermi(E, d.F_dc) * dos;
        return ret;
    }, 2, linspace(0.5 * d.E_gc, E_max, 100), rel_tol, c::epsilon(), x1, w1);

    // valence band in central region
    vec nvsgd = integral([&] (double E) {
        double dos = E / sqrt(4*d.t1*d.t1*d.t2*d.t2 - (E*E - d.t1*d.t1 - d.t2*d.t2) * (E*E - d.t1*d.t1 - d.t2*d.t2));
        vec ret = arma::vec(3);
        ret(0) = (1 - fermi(E, d.F_s)) * dos;
        ret(1) = (1 - fermi(E, d.F_g)) * dos;
        ret(2) = (1 - fermi(E, d.F_d)) * dos;
        return ret;
    }, 3, linspace(E_min, - 0.5 * d.E_g, 100), rel_tol, c::epsilon(), x2, w2);

    // conduction band in central region
    vec ncsgd = integral([&] (double E) {
        double dos = E / sqrt(4*d.t1*d.t1*d.t2*d.t2 - (E*E - d.t1*d.t1 - d.t2*d.t2) * (E*E - d.t1*d.t1 - d.t2*d.t2));
        vec ret = arma::vec(3);
        ret(0) = fermi(E, d.F_s) * dos;
        ret(1) = fermi(E, d.F_g) * dos;
        ret(2) = fermi(E, d.F_d) * dos;
        return ret;
    }, 3, linspace(0.5 * d.E_g, E_max, 100), rel_tol, c::epsilon(), x3, w3);

    // total charge density in contact regions
    vec nc = nvc + ncc;
    // total charge density in central region
    vec nsgd = nvsgd + ncsgd;

    vec ret(d.N_x);
    ret(d.sc).fill(nc(0));
    if (d.sox. a < d.sox.b) {
        ret(d.sox).fill(nsgd(0));
    }
    ret(d.sg).fill(0);
    ret(d.g).fill(nsgd(1));
    ret(d.dg).fill(0);
    if (d.dox.a < d.dox.b) {
        ret(d.dox).fill(nsgd(2));
    }
    ret(d.dc).fill(nc(1));

    ret *= 2 * c::e / M_PI / M_PI / d.r_cnt / d.dr / d.dx; // spintel inside (?)

    n0_d[d.name] = ret;
    return n0_d[d.name];
}

#endif

