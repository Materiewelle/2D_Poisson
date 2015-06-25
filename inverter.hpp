#ifndef INVERTER_HPP
#define INVERTER_HPP

#include <armadillo>
#include <fstream>
#include <vector>

#include "device.hpp"
#include "movie.hpp"
#include "time_evolution.hpp"
#include "gnuplot.hpp"

class inverter {
public:
    // always needed
    device n_fet;
    device p_fet;
    steady_state s_n;
    steady_state s_p;

    // only for transient simulations
    double capacitance;
    signal sg;
    time_evolution te_n;
    time_evolution te_p;
    arma::vec V_out;

    inline inverter(const device & n, const device & p, double c = 1e-12);

    inline bool solve(const voltage & V, double & V_o); // solve steady state
    template<bool make_movie = false>
    inline void solve(const signal & sig);               // solve time-evolution

    // steady-state gate-voltage sweep (output curve)
    inline void output(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & V_out);

    inline void save();
};

//----------------------------------------------------------------------------------------------------------------------

inverter::inverter(const device & n, const device & p, double c)
    : n_fet(n), p_fet(p), capacitance(c) {
}

bool inverter::solve(const voltage & V, double & V_o) {
    /* In steady-state simulations, a root finding algorithm
     * (namely the Brent-algorithm) is used to find the voltage
     * at the output terminal for which the currents through the
     * devices cancel each other.
     * The physical solution has been found in this case, as
     * it satisfies the law of local conservation of charge. */

    auto delta_I = [&] (double V_o) {
        /* this lamda takes a certain output voltage,
         * computes a self-consitent steady-state solution
         * for each device and returns the difference in current. */


        s_n = steady_state(n_fet, {V.s, V.g, V_o});
        s_p = steady_state(p_fet, {V.d, V.g, V_o});

        std::cout << "(" << n_fet.name << ") " << V.s << ", " << V.g << ", " << V_o << ": ";
        std::flush(std::cout);
        s_n.solve();
        std::cout << "(" << p_fet.name << ") " << V.d << ", " << V.g << ", " <<  V_o << ": ";
        std::flush(std::cout);
        s_p.solve();

        return s_n.I.total(0) + s_p.I.total(0);
    };

    // find the output-voltage at which delta_I has a root
    return brent(delta_I, 0.0, 0.5, 0.0005, V_o);
}

template<bool make_movie>
void inverter::solve(const signal & sig) {
    sg = sig; // copy to member (for saving)

    // initialize result vector
    V_out = arma::vec(sg.N_t);
    V_out.fill(0);

    // get the steady state solution for this inverter
    if (!solve(sg[0], V_out(0))) {
        std::cout << "inverter: steady_state did not converge" << std::endl;
        return;
    }

    // setup time-evolution objects
    te_n = std::move(time_evolution(s_n, sg));
    te_p = std::move(time_evolution(s_p, sg));

    if (make_movie) {
        std::vector<std::pair<int, int>> E_i(16);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                E_i[i * 4 + j] = std::make_pair(i, (int)(j * s_n.E[i].size() * 0.25));
            }
        }
        movie mov_n(te_n, E_i);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                E_i[i * 4 + j] = std::make_pair(i, (int)(j * s_p.E[i].size() * 0.25));
            }
        }
        movie mov_p(te_p, E_i);
    }

    /* the time-evolution objects keep track
     * of the time. Observe te_n's watch.
     * (It should show the same time as te_p's...) */
    const unsigned & m = te_n.m;

    while (m < sg.N_t) {
        /* The capacitance is charged due to the difference in output-currents.
         * The following is basically the differential equation for charging a capacitor. */
        V_out(m) = V_out(m-1) - (te_n.I[m - 1].d() + te_p.I[m - 1].d()) * time_evolution::dt / capacitance;
        std::cout << "V_out = " << V_out(m) << std::endl;

        /* pin the devices' internal
         * potentials to the one caused
         * by the charge stored on the capacitor */
        te_n.sg[m].d = V_out(m);
        te_p.sg[m].s = V_out(m);

        // tick-tock on the clock
        te_n.step();
        te_p.step();
    }
}

void inverter::output(const voltage & V0, double V_g1, int N, arma::vec & V_g, arma::vec & V_out) {
    V_g = arma::linspace(V0.g, V_g1, N);
    V_out = arma::vec(N);

    for (int i = 0; i < N; ++i) {
        if (solve({V0.s, V_g(i), V0.d}, V_out(i))) {
            std::cout << "\nstep " << i << ": in " << V_g(i) << "V -> out " << V_out(i) << "V" << std::endl;
        } else {
            std::cout << V_g(i) << ": ERROR!" << std::endl;
        }
    }
}

void inverter::save() {
    te_n.save();
    te_p.save();
    V_out.save(save_folder() + "/V_out.arma");
    std::ofstream just_C(save_folder() + "/C.txt");
    just_C << capacitance;
    just_C.close();

    // make a plot of V_out and save it as a png
    gnuplot gp;
    gp << "set terminal png rounded color enhanced font 'arial,12'\n";
    gp << "set title 'Inverter output voltage'\n";
    gp << "set xlabel 't / ps'\n";
    gp << "set ylabel 'V_{out} / V'\n";
    gp << "set format x '%1.2f'\n";
    gp << "set format y '%1.2f'\n";
    gp << "set output '" << save_folder() << "/V_out.pdf'\n";
    gp.add(std::make_pair(sg.t * 1e12, V_out));
    gp.plot();
}

#endif
