#ifndef WAVE_PACKET_HPP
#define WAVE_PACKET_HPP

#include <armadillo>

#include "constant.hpp"
#include "device.hpp"
#include "potential.hpp"
#include "time_params.hpp"
#include "sd_quantity.hpp"

enum {
    LV = 0,
    RV = 1,
    LC = 2,
    RC = 3,
    LT = 4,
    RT = 5
};

class wave_packet {
public:
    arma::vec E0;
    arma::vec W;
    arma::cx_mat data;
    arma::mat E;

    template<bool left>
    inline void init(const device & d, const arma::vec & E, const arma::vec & W, const potential & phi);

    inline void memory_init();
    inline void memory_update(const sd_vec & affe, unsigned m);

    inline void source_init(const device & d, const sd_vec & u, const sd_vec & q);
    inline void source_update(const sd_vec & u, const sd_vec & L, const sd_vec & qsum, int m);

    template<class T>
    inline void propagate(const T & U_eff, const sd_vec & inv);

    inline void remember();

    inline void update_sum(int m);

    inline void update_E(const device & d, const potential & phi, const potential & phi0);

private:
    sd_vec in;
    sd_vec out;

    sd_mat sum;

    sd_vec source;
    sd_vec memory;

    bool l;

    // from previous time step
    arma::cx_mat old_data;
    sd_vec old_source;
};

//----------------------------------------------------------------------------------------------------------------------

template<bool left>
void wave_packet::init(const device & d, const arma::vec & EE, const arma::vec & WW, const potential & phi) {
    using namespace arma;

    E0 = EE;
    W = WW;
    data = cx_mat(d.N_x * 2, E0.size());
    E = mat(d.N_x, E0.size());
    in = sd_vec(E0.size());
    out = sd_vec(E0.size());
    sum = sd_mat(t::N_t, E0.size());
    source = sd_vec(E0.size());
    memory = sd_vec(E0.size());
    l = left;

    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < E0.size(); ++i) {
        // calculate 1 column of green's function
        cx_double Sigma_s, Sigma_d;
        cx_vec G = green_col<left>(d, phi, E0(i), Sigma_s, Sigma_d);

        // calculate wave function
        if (left) {
            G *= std::sqrt(cx_double(- 2 * Sigma_s.imag()));
        } else {
            G *= std::sqrt(cx_double(- 2 * Sigma_d.imag()));
        }

        // extract data
        data.col(i) = G;
        in.s(i)  = G(0);
        in.d(i)  = G(G.size() - 1);

        // calculate first layer in the leads analytically
        out.s(i) = ((E0(i) - phi.s()) * in.s(i) - d.tc1 * G(           1)) / d.tc2;
        out.d(i) = ((E0(i) - phi.d()) * in.d(i) - d.tc1 * G(G.size() - 2)) / d.tc2;
    }

    update_E(d, phi, phi);
}

void wave_packet::memory_init() {
    memory.s.fill(0.0);
    memory.d.fill(0.0);
}

void wave_packet::memory_update(const sd_vec & affe, unsigned m) {
    memory.s = (affe.s.st() * sum.s.rows({1, m - 1})).st();
    memory.d = (affe.d.st() * sum.d.rows({1, m - 1})).st();
}

void wave_packet::source_init(const device & d, const sd_vec & u, const sd_vec & q) {
    using namespace std::complex_literals;
    source.s = - 2i * t::g * u.s(1) * (d.tc2 * out.s + 1i * t::g * q.s(0) * in.s) / (1.0 + 1i * t::g * E0);
    source.d = - 2i * t::g * u.d(1) * (d.tc2 * out.d + 1i * t::g * q.d(0) * in.d) / (1.0 + 1i * t::g * E0);
}

void wave_packet::source_update(const sd_vec & u, const sd_vec & L, const sd_vec & qsum, int m) {
    using namespace std::complex_literals;
    static constexpr auto g2 = t::g * t::g;

    source.s = (old_source.s % (1 - 1i * t::g * E0) * u.s(m) * u.s(m-1) + 2 * g2 * L.s(1) / u.s(m) * qsum.s(t::N_t-m) * in.s) / (1 + 1i * t::g * E0);
    source.d = (old_source.d % (1 - 1i * t::g * E0) * u.d(m) * u.d(m-1) + 2 * g2 * L.d(1) / u.d(m) * qsum.d(t::N_t-m) * in.d) / (1 + 1i * t::g * E0);
}

template<class T>
void wave_packet::propagate(const T & U_eff, const sd_vec & inv) {
    data = U_eff * old_data + arma::kron(source.s.st() + memory.s.st(), inv.s) + arma::kron(source.d.st() + memory.d.st(), inv.d);
}

void wave_packet::remember() {
    old_data = data;
    old_source = source;
}

void wave_packet::update_sum(int m) {
    sum.s.row(m) = old_data.row(0) + data.row(0);
    sum.d.row(m) = old_data.row(old_data.n_rows - 1) + data.row(data.n_rows - 1);
}

void wave_packet::update_E(const device & d, const potential & phi, const potential & phi0) {
    using namespace std;

    for (unsigned i = 0; i < E.n_cols; ++i) {
        for (unsigned j = 1; j < E.n_rows - 1; ++j) {
            double n = 1.0 / (norm(data(2 * j, i)) + norm(data(2 * j + 1, i)));
            double m1 = 2 * (real(data(2 * j, i)) * real(data(2 * j + 1, i)) + imag(data(2 * j, i)) * imag(data(2 * j + 1, i)));
            arma::cx_double m2 = conj(data(2 * j, i)) * data(2 * j - 1, i);
            arma::cx_double m3 = conj(data(2 * j + 1, i)) * data(2 * j + 2, i);
            E(j, i) = phi(j) + n * (d.t_vec(2 * j) * m1 + real(d.t_vec(2 * j - 1) * m2 + d.t_vec(2 * j + 1) * m3));
        }
    }

    if (l) {
        for (unsigned i = 0; i < E.n_cols; ++i) {
            E(0, i) = E0(i) + phi.s() - phi0.s();
            E(E.n_rows - 1, i) = E(E.n_rows - 2, i);
        }
    } else {
        for (unsigned i = 0; i < E.n_cols; ++i) {
            E(E.n_rows - 1, i) = E0(i) + phi.d() - phi0.d();
            E(0, i) = E(1, i);
        }
    }
}

#endif
