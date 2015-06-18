#ifndef STEADY_STATE_HPP
#define STEADY_STATE_HPP

#include "device.hpp"
#include "voltage.hpp"
#include "charge_density.hpp"
#include "current.hpp"

class steady_state {
public:
    static constexpr auto dphi_threshold = 1e-9;
    static constexpr auto max_iterations = 50;

    device d;
    voltage V;
    charge_density n;
    potential phi;
    current I;
    arma::vec E[4];
    arma::vec W[4];

    inline steady_state();
    inline steady_state(const device & dd, const voltage & V);
    inline steady_state(const device & dd, const voltage & V, const charge_density & n0);

    template<bool smooth = true>
    inline bool solve();

    template<bool reuse=true>
    static inline void output(const device & d, const voltage & V0, double V_d1, int N, arma::vec & V_d, arma::vec & I);
    template<bool reuse=true>
    static inline void transfer(const device & d, const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & I);
};

//----------------------------------------------------------------------------------------------------------------------

steady_state::steady_state() {
}

steady_state::steady_state(const device & dd, const voltage & VV)
    : d(dd), V(VV), n() {
}

steady_state::steady_state(const device & dd, const voltage & VV, const charge_density & n0)
    : d(dd), V(VV), n(n0) {
}

template<bool smooth>
bool steady_state::solve() {
    /* computes a self-consistent solution for the given voltages */
    using namespace std;

    // get the right-hand side vector in poisson's equation
    arma::vec R0 = potential_impl::get_R0(d, V);

    // solve poisson's equation with flat charge density
    phi = potential(d, R0);

    // this variable will hold the maximum deviation of phi
    double dphi;

    // iteration counter
    int it;

    anderson mr_neo(phi.data);

    // repeat until potential does not differ from previous iteration or the maximum number of iterations has been reached
    for (it = 1; it <= max_iterations; ++it) {
        // update charge density
        n = { d, phi, E, W };

        // update potential
        dphi = phi.update(d, R0, n, mr_neo);

        //cout << V.s << ", " << V.g << ", " << V.d;
        //cout << ": iteration " << it << ": rel deviation is " << dphi/dphi_threshold << endl;

        // check wether we call this situation self-consitent
        if (dphi < dphi_threshold) {
            break;
        }

        /* It is often a good idea to straighten out the potential inside the contact regions
         * in order to keep the self-consitency algorithm from diverging during the first few steps.
         * This could be understood as limiting the first few updates to the central region. */
        if (smooth) {
            if (it < 3) {
                phi.smooth(d);
            }
        }
    }

    // get current
    I = current(d, phi);

    bool converged = !(dphi > dphi_threshold);
//    cout << V.s << ", " << V.g << ", " << V.d << ", ";
    string conv_text = converged ? "converged!" : "DIVERGED!!!";
    cout << it << " iterations, reldev=" << dphi/dphi_threshold << ", " << conv_text << ", n_E = " << E[0].size() + E[1].size() + E[3].size() + E[4].size() << endl;
    return converged;

    // check if actually converged
    //if (!converged) {
        //cout << "Warning: steady_state::solve did not converge after " << it << " iterations!" << endl;
    //    return false;
    //} else {
    //    return true;
    //}
}

template<bool reuse>
void steady_state::output(const device & d, const voltage & V0, double V_d1, int N, arma::vec & V_d, arma::vec & I) {
    /* This swipes the drain voltage from a given initial state to V_d1 in n steps. The previous self-constent
     * solution can be reused as a first guess for the next voltage point. This can speed up the calculation of
     * successive voltage points in some scenarios (if the points are close together). The voltage points and corresponding
     * results for the drain-current are stored in the vectors V_d and I. */

    // initialize result-vectors
    V_d = arma::linspace(V0.d, V_d1, N);
    I = arma::vec(N);

    // solve for the first voltage-point
    steady_state s(d, V0);
    bool conv = s.solve();
    I(0) = s.I.total(0);

    for (int i = 1; i < N; ++i) {
        std::cout << "Step " << i+1 << "/" << N << ": V_d=" << V_d(i) << ": ";
        std::flush(std::cout);
        voltage V = { V0.s, V0.g, V_d(i) };
        if (reuse && conv) {
            s = steady_state(d, V, s.n);
            conv = s.solve<false>();
            /* don't reuse the previous result
             * but try again with n=0 */
            if (!conv) {
                s = steady_state(d, V);
                conv = s.solve();
            }
        } else {
            s = steady_state(d, V);
            conv = s.solve();
        }
        I(i) = s.I.total(0);
    }
}

template<bool reuse>
void steady_state::transfer(const device & d, const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & I) {
    /* Same as steady_state::output, but here the gate voltage is swiped instead. */

    V_g = arma::linspace(V0.g, V_g1, N);
    I = arma::vec(N);

    steady_state s(d, V0);
    bool conv = false;


    for (int i = 0; i < N; ++i) {
        std::cout << "Step " << i+1 << "/" << N << ": V_g=" << V_g(i) << ": ";
        std::flush(std::cout);
        voltage V = { V0.s, V_g(i), V0.d };
        if (reuse && conv) {
            s = steady_state(d, V, s.n);
            conv = s.solve<false>();
            /* don't reuse the previous result
             * but try again with n=0 */
            if (!conv) {
                s = steady_state(d, V);
                conv = s.solve();
            }
        } else {
            s = steady_state(d, V);
            conv = s.solve();
        }
        I(i) = s.I.total(0);
    }
}

#endif
