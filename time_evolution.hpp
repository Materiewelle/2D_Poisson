#ifndef TIME_EVOLUTION_HPP
#define TIME_EVOLUTION_HPP

#include <armadillo>
#include <functional>
#include <fstream>
#include <iomanip>

#include "device.hpp"
#include "current.hpp"
#include "potential.hpp"
#include "charge_density.hpp"
#include "voltage.hpp"
#include "wave_packet.hpp"
#include "sd_quantity.hpp"
#include "signal.hpp"
#include "steady_state.hpp"

class time_evolution {
public:
    static constexpr double dphi_threshold = 1e-8;
    static constexpr int max_iterations = 25;
    static constexpr int memory_cutoff = 2000;
    static constexpr double dt = 5e-16;                     // timestep
    static const double g; // debug doesn't work otherwise

    unsigned m;
    signal sg;
    device d;
    std::vector<current> I;
    std::vector<potential> phi;
    std::vector<charge_density> n;
    wave_packet psi[4];

    inline time_evolution();
    inline time_evolution(const device & dd, const signal & sgg);
    inline time_evolution(const steady_state & s, const signal & sgg);

    inline void solve();
    inline void step();
    inline void save();

    inline void add_callback(const std::function<void(void)> & f);

private:
    sd_vec u;
    sd_vec L;
    sd_vec q;
public:
    sd_vec qsum;
private:

    arma::cx_mat H_eff;
    sd_vec old_L;
    arma::cx_mat cx_eye;

    std::vector<std::function<void()>> callbacks;

    inline void init(const steady_state & s);
    inline void calculate_q();
    inline void callback();
};
const double time_evolution::g = 0.5 * time_evolution::dt * c::e / c::h_bar;

//----------------------------------------------------------------------------------------------------------------------

time_evolution::time_evolution() {
}

time_evolution::time_evolution(const device & dd, const signal & sgg)
//    : m(1), sg(sgg), d(dd), I(sg.N_t), phi(sg.N_t), n(sg.N_t), u(sg.N_t), L(sg.N_t), q(sg.N_t), qsum(sg.N_t - 1) {
    : m(1), sg(sgg), d(dd), I(sg.N_t), phi(sg.N_t), n(sg.N_t), u(sg.N_t), L(sg.N_t), q(memory_cutoff+1), qsum(memory_cutoff) {

    // solve steady state
    steady_state s(d, sg.V[0]);
    s.solve<true>();

    // initialize
    init(s);
}

time_evolution::time_evolution(const steady_state & s, const signal & sgg)
    : m(1), sg(sgg), d(s.d), I(sg.N_t), phi(sg.N_t), n(sg.N_t), u(sg.N_t), L(sg.N_t), q(sg.N_t), qsum(sg.N_t - 1) {

    // initialize
    init(s);
}

void time_evolution::solve() {
    while (m < sg.N_t) {
        step();
    }
}

void time_evolution::step() {
    using namespace arma;
    using namespace std::complex_literals;

    // estimate charge density from previous values
    n[m].total = (m == 1) ? n[0].total : (2 * n[m-1].total - n[m-2].total);

    // prepare right side of poisson equation
    vec R0 = potential_impl::get_R0(d, sg.V[m]);

    // first guess for the potential
    phi[m] = potential(d, R0, n[m]);

    // prepare anderson
    anderson mr_neo(phi[m].data);

    // current data becomes old data
    for (int i = 0; i < 4; ++i) {
        psi[i].remember();
    }
    old_L = L;

    // self-consistency loop
    for (int it = 0; it < max_iterations; ++it) {
        // diagonal of H with self-energy
        H_eff.diag() = conv_to<cx_vec>::from(0.5 * (phi[m].twice + phi[m-1].twice));
        H_eff(        0,        0) -= 1i * g * q.s(0);
        H_eff(2*d.N_x-1,2*d.N_x-1) -= 1i * g * q.d(0);

        // crank-nicolson propagator
        arma::cx_mat U_eff = arma::solve(cx_eye + 1i * g * H_eff, cx_eye - 1i * g * H_eff);

        // inv
        sd_vec inv;
        inv.s = inverse_col< true>(cx_vec(1i * g * d.t_vec), cx_vec(1.0 + 1i * g * H_eff.diag()));
        inv.d = inverse_col<false>(cx_vec(1i * g * d.t_vec), cx_vec(1.0 + 1i * g * H_eff.diag()));

        // u
//        u.s(m) = 0.5 * (phi[m].s() + phi[m - 1].s()) - phi[0].s();
//        u.d(m) = 0.5 * (phi[m].d() + phi[m - 1].d()) - phi[0].d();
//        u.s(m) = (1.0 - 0.5i * g * u.s(m)) / (1.0 + 0.5i * g * u.s(m));
//        u.d(m) = (1.0 - 0.5i * g * u.d(m)) / (1.0 + 0.5i * g * u.d(m));
        u.s(m - 1) = 0.5 * (phi[m].s() + phi[m - 1].s()) - phi[0].s();
        u.d(m - 1) = 0.5 * (phi[m].d() + phi[m - 1].d()) - phi[0].d();
        u.s(m - 1) = (1.0 - 0.5i * g * u.s(m - 1)) / (1.0 + 0.5i * g * u.s(m - 1));
        u.d(m - 1) = (1.0 - 0.5i * g * u.d(m - 1)) / (1.0 + 0.5i * g * u.d(m - 1));

        // Lambda
//        L.s({1, m}) = old_L.s({1, m}) * u.s(m) * u.s(m);
//        L.d({1, m}) = old_L.d({1, m}) * u.d(m) * u.d(m);
        L.s({0, m - 1}) = old_L.s({0, m - 1}) * u.s(m - 1) * u.s(m - 1);
        L.d({0, m - 1}) = old_L.d({0, m - 1}) * u.d(m - 1) * u.d(m - 1);

        if (m == 1) {
            for (int i = 0; i < 4; ++i) {
                psi[i].memory_init();
                psi[i].source_init(d, u, q);
                psi[i].propagate(U_eff, inv);
                psi[i].update_E(d, phi[m], phi[0]);
            }
        } else {
            sd_vec affe;
//            affe.s = - g * g * L.s({1, m - 1}) % qsum.s({sg.N_t-m, sg.N_t-2}) / u.s({1, m - 1}) / u.s(m);
//            affe.d = - g * g * L.d({1, m - 1}) % qsum.d({sg.N_t-m, sg.N_t-2}) / u.d({1, m - 1}) / u.d(m);
            arma::uword cu = memory_cutoff;
            if (m <= cu + 1) {
                affe.s = - g * g * L.s({0, m - 2}) % qsum.s({cu + 1 - m, cu - 1}) / u.s({0, m - 2}) / u.s(m - 1);
                affe.d = - g * g * L.d({0, m - 2}) % qsum.d({cu + 1 - m, cu - 1}) / u.d({0, m - 2}) / u.d(m - 1);
            } else {
                affe.s = - g * g * L.s({m - cu - 1, m - 2}) % qsum.s({0, cu + 1 - 2}) / u.s({m - cu - 1, m - 2}) / u.s(m - 1);
                affe.d = - g * g * L.d({m - cu - 1, m - 2}) % qsum.d({0, cu + 1 - 2}) / u.d({m - cu - 1, m - 2}) / u.s(m - 1);
            }

            // propagate wave functions of modes inside bands
            for (int i = 0; i < 4; ++i) {
                psi[i].memory_update(affe, m);
//                psi[i].source_update(u, L, qsum, m, sg.N_t);
                psi[i].source_update(u, L, qsum, m);
                psi[i].propagate(U_eff, inv);
                psi[i].update_E(d, phi[m], phi[0]);
            }
        }

        // update n
        n[m] = {d, psi, phi[m] };

        // update potential
        auto dphi = phi[m].update(d, R0, n[m], mr_neo);

//        cout << m << ": iteration " << it << ": rel deviation is " << dphi / dphi_threshold << endl;

        // check if dphi is small enough
        if (dphi < dphi_threshold) {
            std::cout << "(" << d.name << ") timestep " << m << ": t=" << m * dt * 1e12 << std::setprecision(5) << std::fixed << "ps, " << it + 1 << " iterations, reldev=" << dphi / dphi_threshold << std::endl;
            break;
        }
    }

    // update sum
    for (int i = 0; i < 4; ++i) {
        psi[i].update_sum(m);
    }

    // calculate current
    I[m] = current(d, psi, phi[m]);

    // increase m for next time step
    ++m;

    callback();
}

void time_evolution::save() {
    std::cout << "(" << d.name << ") saving time-dependent observables... ";
    std::flush(std::cout);

    arma::mat phi_mat(d.N_x, sg.N_t);
    arma::mat n_mat(d.N_x, sg.N_t);
    arma::mat I_mat(d.N_x, sg.N_t);
    arma::mat V_mat(sg.N_t, 3);

    for (unsigned i = 0; i < sg.N_t; ++i) {
        phi_mat.col(i) = phi[i].data;
        n_mat.col(i) = n[i].total;
        I_mat.col(i) = I[i].total;
        V_mat(i, 0) = sg.V[i].s;
        V_mat(i, 1) = sg.V[i].g;
        V_mat(i, 2) = sg.V[i].d;
    }

    std::string subfolder(save_folder() + "/" + d.name);
    system("mkdir -p " + subfolder);

    phi_mat.save(subfolder + "/phi.arma");
    n_mat.save(subfolder + "/n.arma");
    I_mat.save(subfolder + "/I.arma");
    d.x.save(subfolder + "/xtics.arma");
    sg.t.save(subfolder + "/ttics.arma");
    V_mat.save(subfolder + "/V.arma");

    std::ofstream device_params(subfolder + "/device.ini");
    device_params << d.to_string();
    device_params.close();

    /* produce plots of purely time-dependent
     * observables and save them as PDFs */
    gnuplot gp;
    gp << "set terminal pdf rounded color enhanced font 'arial,12'\n";
    gp << "set xlabel 't / ps'\n";

    // source and drain current
    gp << "set format x '%1.2f'\n";
    gp << "set format y '%1.0g'\n";
    arma::vec I_s(sg.N_t);
    arma::vec I_d(sg.N_t);
    for (unsigned i = 0; i < sg.N_t; ++i) {
        I_s(i) = I[i].s();
        I_d(i) = I[i].d();
    }
    gp << "set title '" << d.name << " time-dependent source current'\n";
    gp << "set ylabel 'I_{s} / A'\n";
    gp << "set output '" << subfolder << "/I_s.pdf'\n";
    gp.add(std::make_pair(sg.t * 1e12, I_s));
    gp.plot();
    gp.reset();
    gp << "set title '" << d.name << " time-dependent drain voltage'\n";
    gp << "set ylabel 'I_{d} / A'\n";
    gp << "set output '" << subfolder << "/I_d.pdf'\n";
    gp.add(std::make_pair(sg.t * 1e12, I_d));
    gp.plot();

    std::cout << " done!\n";
}

void time_evolution::add_callback(const std::function<void(void)> & f) {
    callbacks.push_back(f);
}

void time_evolution::init(const steady_state & s) {
    using namespace arma;

    // save results from steady state
    I[0]   = s.I;
    phi[0] = s.phi;
    n[0]   = s.n;

    // get initial wavefunctions
//    psi[LV].init< true>(d, s.E[LV], s.W[LV], phi[0], sg.N_t);
//    psi[RV].init<false>(d, s.E[RV], s.W[RV], phi[0], sg.N_t);
//    psi[LC].init< true>(d, s.E[LC], s.W[LC], phi[0], sg.N_t);
//    psi[RC].init<false>(d, s.E[RC], s.W[RC], phi[0], sg.N_t);
    psi[LV].init< true>(d, s.E[LV], s.W[LV], phi[0]);
    psi[RV].init<false>(d, s.E[RV], s.W[RV], phi[0]);
    psi[LC].init< true>(d, s.E[LC], s.W[LC], phi[0]);
    psi[RC].init<false>(d, s.E[RC], s.W[RC], phi[0]);

    // precalculate q-values
    calculate_q();

    // build constant part of Hamiltonian
    H_eff = cx_mat(2 * d.N_x, 2 * d.N_x);
    H_eff.fill(0);
    H_eff.diag(+1) = conv_to<cx_vec>::from(d.t_vec);
    H_eff.diag(-1) = conv_to<cx_vec>::from(d.t_vec);

    // setup lambda
    L.s.fill(1.0);
    L.d.fill(1.0);

    // complex unity matrix
    cx_eye = eye<cx_mat>(2 * d.N_x, 2 * d.N_x);
}

void time_evolution::calculate_q() {
    using namespace arma;
    using namespace std;
    using mat22 = cx_mat::fixed<2, 2>;

    cout << "(" << d.name << ") precalculating the q-values..."; flush(cout);

    uword cu = memory_cutoff + 1; // before: size = d.N-t - 1, now: size = memory_cutoff

    // shortcuts
    const double t1 = d.tc1;
    const double t12 = t1 * t1;
    const double t2 = d.tc2;
    const double t22 = t2 * t2;
    static const double g2 = g * g;
    static const mat22 eye2 = { 1, 0, 0, 1 };

    // get q values dependent on potential in lead
    auto get_q = [&] (double phi0) {
        // storage
//        cx_vec qq(sg.N_t + 3);
        cx_vec qq(cu + 3);
//        vector<mat22> p(sg.N_t + 3);
        vector<mat22> p(cu + 3);

        // hamiltonian in lead
        mat22 h = { phi0, t1, t1, phi0};

        // coupling hamiltonian
        mat22 Vau = { 0, t2, 0, 0 };

        // set first 3 values of q and p to 0
        for (int i = 0; i < 3; ++i) {
            qq(i) = 0;
            p[i] = { 0, 0, 0, 0 };
        }

        // first actual q value (wih pq-formula)
        auto a = (1.0 + 2i * g * phi0 + g2 * (t12 - t22 - phi0*phi0)) / g2 / (1.0 + 1i * g * phi0);
        qq(3) = - 0.5 * a + sqrt(0.25 * a * a + t22 / g2);

        // first actual p value
        p[3] = inv(eye2 + 1i * g * h + mat22({ g2 * qq(3), 0, 0, 0 }));

        // calculate A & C parameters
        mat22 A = eye2 + 1i * g * h + g2 * Vau.t() * p[3] * Vau;
        auto C = A(0,0) * A(1,1) - A(0,1) * A(1,0);

        // loop over all time steps
//        for (unsigned i = 4; i < sg.N_t + 3; ++i) {
        for (uword i = 4; i < cu + 3; ++i) {
            // perform sum
            mat22 R = { 0, 0, 0, 0 };
            for (uword k = 4; k < i; ++k) {
                R += (p[k] + 2 * p[k - 1] + p[k - 2]) * Vau * p[i - k + 3];
            }

            // calculate B parameter
            mat22 B = (eye2 - 1i * g * h) * p[i - 1] - g2 * Vau.t() * ((2 * p[i - 1] + p[i - 2]) * Vau * p[3] + R);

            // calculate next p values
            p[i](1,1) = (A(1,0) * B(0,1) - A(0,0) * B(1,1)) / (g2 * t22 * p[3](0,1) * A(1,0) - C);
            p[i](0,1) = (B(1,1) - A(1,1) * p[i](1,1)) / A(1,0);
            p[i](0,0) = (A(1,1) * B(0,0) - A(0,1) * B(1,0) - g2 * t22 * p[3](0,0) * p[i](1,1) * A(1,1)) / C;
            p[i](1,0) = (B(1,0) - A(1,0) * p[i](0,0)) / A(1,1);

            // calculate next q value
            qq(i) = t22 * p[i](1,1);
        }

        return qq;
    };

    // calculate and save q values
//    q.s = get_q(phi[0].s())({3, sg.N_t + 2});
    q.s = get_q(phi[0].s())({3, cu + 2});
//    q.d = get_q(phi[0].d())({3, sg.N_t + 2});
    q.d = get_q(phi[0].d())({3, cu + 2});

    // sum of two following q-values reversed
//    for (unsigned i = 0; i < sg.N_t - 1; ++i) {
//        qsum.s(i) = q.s(sg.N_t - 1 - i) + q.s(sg.N_t - 2 - i);
//        qsum.d(i) = q.d(sg.N_t - 1 - i) + q.d(sg.N_t - 2 - i);
//    }
    for (unsigned i = 0; i < cu - 1; ++i) {
        qsum.s(i) = q.s(cu - 1 - i) + q.s(cu - 2 - i);
        qsum.d(i) = q.d(cu - 1 - i) + q.d(cu - 2 - i);
    }
    cout << " done!" << endl;
}

void time_evolution::callback() {
    using namespace std;

    for (auto i = begin(callbacks); i != end(callbacks); ++i) {
        (*i)();
    }
}

#endif
