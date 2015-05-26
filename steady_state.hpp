#ifndef STEADY_STATE_HPP
#define STEADY_STATE_HPP

#include <armadillo>

#include "anderson.hpp"
#include "charge_density.hpp"
#include "current.hpp"
#include "potential.hpp"
#include "voltage.hpp"

class steady_state {
public:
<<<<<<< HEAD
    static constexpr auto dphi_threshold = 1e-8;
    static constexpr auto max_iterations = 100;
=======
    static constexpr auto dphi_threshold = 1e-5;
    static constexpr auto max_iterations = 1;
>>>>>>> 422f66cfb66211b0a7febec58fb6eacfca72a79c

    voltage V;
    charge_density n;
    potential phi;
    current I;
    arma::vec E[4];
    arma::vec W[4];

    inline steady_state(const voltage & V);
    inline steady_state(const voltage & V, const charge_density & n0);

    template<bool smooth = true>
    inline bool solve();

    static inline void output(const voltage & V0, double V_d1, int N, arma::vec & V_d, arma::vec & I);
    template<bool reuse=true>
    static inline void transfer(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & I);
};

//----------------------------------------------------------------------------------------------------------------------

steady_state::steady_state(const voltage & VV)
    : V(VV), n() {
}

steady_state::steady_state(const voltage & VV, const charge_density & n0)
    : V(VV), n(n0) {
}

template<bool smooth>
bool steady_state::solve() {
    using namespace std;

    // init potential
    arma::vec R0 = potential_impl::get_R0(V);
    phi = potential(R0, n);

    // dphi = norm(delta_phi)
    double dphi;

    // iteration counter
    int it;

    anderson mr_neo(phi.data);

    // repeat until potential does not change anymore or iteration limit has been reached
    for (it = 1; it <= max_iterations; ++it) {
        // update charge density
        n.update(phi, E, W);

        // update potential
        dphi = phi.update(R0, n, mr_neo);

//        cout << V.s << ", " << V.g << ", " << V.d;
//        cout << ": iteration " << it << ": rel deviation is " << dphi/dphi_threshold << endl;

        // check if dphi is small enough
        if (dphi < dphi_threshold) {
            break;
        }

        if (smooth) {
            // smooth potential in the beginning
            if (it < 3) {
                phi.smooth();
            }
        }
    }

    // get current
    I = current(phi);

    bool converged = !(dphi > dphi_threshold);
//    cout << V.s << ", " << V.g << ", " << V.d;
    string conv_text = converged ? "converged!" : "DIVERGED!!!";
    cout << it << " iterations, reldev=" << dphi/dphi_threshold << ", " << conv_text << endl;
    return converged;

    // check if actually converged
    //if (!converged) {
        //cout << "Warning: steady_state::solve did not converge after " << it << " iterations!" << endl;
    //    return false;
    //} else {
    //    return true;
    //}
}

void steady_state::output(const voltage & V0, double V_d1, int N, arma::vec & V_d, arma::vec & I) {
    V_d = arma::linspace(V0.d, V_d1, N);
    I = arma::vec(N);

    steady_state s(V0);
    bool conv = s.solve();
    I(0) = s.I.total(0);

    for (int i = 1; i < N; ++i) {
        voltage V = { V0.s, V0.g, V_d(i) };
        if (conv) {
            s = steady_state(V, s.n);
            conv = s.solve<false>();
        } else {
            s = steady_state(V);
            conv = s.solve();
        }
        I(i) = s.I.total(0);
    }
}

template<bool reuse>
void steady_state::transfer(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & I) {
    V_g = arma::linspace(V0.g, V_g1, N);
    I = arma::vec(N);

    steady_state s(V0);
    bool conv = false;

//    int diverged = 0;
//    int max_div = 3;

    for (int i = 0; i < N; ++i) {
        std::cout << "Step " << i+1 << "/" << N << ": V_g=" << V_g(i) << ": ";
        std::flush(std::cout);
        voltage V = { V0.s, V_g(i), V0.d };
        if (reuse && conv) {
            s = steady_state(V, s.n);
            conv = s.solve<false>();
            if (!conv) {
                s = steady_state(V);
                conv = s.solve();
            }
        } else {
//            if(++diverged >= max_div) {
//                break;
//            }
            s = steady_state(V);
            conv = s.solve();
        }
        I(i) = s.I.total(0);
    }
}

#endif

