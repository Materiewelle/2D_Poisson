#ifndef WAVE_PACKET_HPP_HEADER
#define WAVE_PACKET_HPP_HEADER

#include <armadillo>

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
    arma::vec F0;
    arma::vec W;
    arma::cx_mat * data;
    arma::mat E;

    inline wave_packet();
    inline wave_packet(const wave_packet & psi);
    inline wave_packet(wave_packet && psi);

    inline wave_packet & operator=(const wave_packet & psi);
    inline wave_packet & operator=(wave_packet && psi);

    template<bool left>
//    inline void init(const device & d, const arma::vec & E, const arma::vec & W, const potential & phi, unsigned N_t);
    inline void init(const device & d, const arma::vec & E, const arma::vec & W, const potential & phi);

    inline void memory_init();
    inline void memory_update(const sd_vec & affe, unsigned m);

    inline void source_init(const device & d, const sd_vec & u, const sd_vec & q);
//    inline void source_update(const sd_vec & u, const sd_vec & L, const sd_vec & qsum, int m, int N_t);
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
    sd_vec old_source;
    arma::cx_mat * old_data;

    arma::cx_mat data1;
    arma::cx_mat data2;
};

#endif

//----------------------------------------------------------------------------------------------------------------------

#ifndef WAVE_PACKET_HPP_BODY
#define WAVE_PACKET_HPP_BODY

wave_packet::wave_packet() {
}

wave_packet::wave_packet(const wave_packet & psi) :
    E0(psi.E0),
    F0(psi.F0),
    W(psi.W),
    E(psi.E),
    in(psi.in),
    out(psi.out),
    sum(psi.sum),
    source(psi.source),
    memory(psi.memory),
    l(psi.l),
    old_source(psi.old_source),
    data1(psi.data1),
    data2(psi.data2) {
    data = (psi.data == &psi.data1) ? &data1 : ((psi.data == &psi.data2) ? &data2 : nullptr);
    old_data = (psi.old_data == &psi.data1) ? &data1 : ((psi.old_data == &psi.data2) ? &data2 : nullptr);
}
wave_packet::wave_packet(wave_packet && psi)
    : l(psi.l) {
    E0 = std::move(psi.E0);
    F0 = std::move(psi.F0);
    W = std::move(psi.W);
    E = std::move(psi.E);
    in = std::move(psi.in);
    out = std::move(psi.out);
    sum = std::move(psi.sum);
    source = std::move(psi.source);
    memory = std::move(psi.memory);
    old_source = std::move(psi.old_source);
    data1 = std::move(psi.data1);
    data2 = std::move(psi.data2);
    data = psi.data;
    old_data = psi.data;
}

wave_packet & wave_packet::operator=(const wave_packet & psi) {
    // "Don't do it in practice. The whole thing is ugly beyond description."
    new(this) wave_packet(psi);
    return *this;
}
wave_packet & wave_packet::operator=(wave_packet && psi) {
    new(this) wave_packet(psi);
    return *this;
}

template<bool left>
//void wave_packet::init(const device & d, const arma::vec & EE, const arma::vec & WW, const potential & phi, unsigned N_t) {
void wave_packet::init(const device & d, const arma::vec & EE, const arma::vec & WW, const potential & phi) {
    using namespace arma;

    E0 = EE;
    if (left) {
        F0 = fermi(E0 - phi.s(), d.F_sc);
    } else {
        F0 = fermi(E0 - phi.d(), d.F_dc);
    }
    W = WW;
    data1 = cx_mat(d.N_x * 2, E0.size());
    data = &data1;
    E = mat(d.N_x, E0.size());
    in = sd_vec(E0.size());
    out = sd_vec(E0.size());
//    sum = sd_mat(N_t, E0.size());
    sum = sd_mat(time_evolution::memory_cutoff, E0.size());
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
        data->col(i) = G;
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
//    memory.s = (affe.s.st() * sum.s.rows({1, m - 1})).st();
//    memory.d = (affe.d.st() * sum.d.rows({1, m - 1})).st();
    unsigned cu = time_evolution::memory_cutoff;
    if (m <= cu) {
        memory.s = (affe.s.st() * sum.s.rows({0, m - 2})).st();
        memory.d = (affe.d.st() * sum.d.rows({0, m - 2})).st();
    } else {
        unsigned n = (m - 1) % cu;
        memory.s = (affe.s({0, cu - n - 1}).st() * sum.s.rows({n, cu - 1})).st();
        memory.d = (affe.d({0, cu - n - 1}).st() * sum.d.rows({n, cu - 1})).st();
        if (n > 0) {
            memory.s += (affe.s({cu - n, cu - 1}).st() * sum.s.rows({0, n - 1})).st();
            memory.d += (affe.d({cu - n, cu - 1}).st() * sum.d.rows({0, n - 1})).st();
        }
    }
}

void wave_packet::source_init(const device & d, const sd_vec & u, const sd_vec & q) {
    using namespace std::complex_literals;

    static const double g = time_evolution::g;

//    source.s = - 2i * g * u.s(1) * (d.tc2 * out.s + 1i * g * q.s(0) * in.s) / (1.0 + 1i * g * E0);
//    source.d = - 2i * g * u.d(1) * (d.tc2 * out.d + 1i * g * q.d(0) * in.d) / (1.0 + 1i * g * E0);
    source.s = - 2i * g * u.s(0) * (d.tc2 * out.s + 1i * g * q.s(0) * in.s) / (1.0 + 1i * g * E0);
    source.d = - 2i * g * u.d(0) * (d.tc2 * out.d + 1i * g * q.d(0) * in.d) / (1.0 + 1i * g * E0);
}

//void wave_packet::source_update(const sd_vec & u, const sd_vec & L, const sd_vec & qsum, int m, int N_t) {
void wave_packet::source_update(const sd_vec & u, const sd_vec & L, const sd_vec & qsum, int m) {
    using namespace std::complex_literals;
    static const double g = time_evolution::g;
    static const double g2 = g * g;

//    source.s = (old_source.s % (1 - 1i * g * E0) * u.s(m) * u.s(m-1) + 2 * g2 * L.s(1) / u.s(m) * qsum.s(N_t-m) * in.s) / (1 + 1i * g * E0);
//    source.d = (old_source.d % (1 - 1i * g * E0) * u.d(m) * u.d(m-1) + 2 * g2 * L.d(1) / u.d(m) * qsum.d(N_t-m) * in.d) / (1 + 1i * g * E0);

    int n = time_evolution::memory_cutoff + 1 - m;
    auto qss = (n >= 0) ? qsum.s(n) : 0.0;
    auto qsd = (n >= 0) ? qsum.d(n) : 0.0;
    source.s = (old_source.s % (1 - 1i * g * E0) * u.s(m - 1) * u.s(m - 2) + 2 * g2 * L.s(0) / u.s(m - 1) * qss * in.s) / (1 + 1i * g * E0);
    source.d = (old_source.d % (1 - 1i * g * E0) * u.d(m - 1) * u.d(m - 2) + 2 * g2 * L.d(0) / u.d(m - 1) * qsd * in.d) / (1 + 1i * g * E0);
}

template<class T>
void wave_packet::propagate(const T & U_eff, const sd_vec & inv) {
    *data = U_eff * (*old_data) + arma::kron(source.s.st() + memory.s.st(), inv.s) + arma::kron(source.d.st() + memory.d.st(), inv.d);
}

void wave_packet::remember() {
    if (data == &data1) {
        data     = &data2;
        old_data = &data1;
    } else {
        data     = &data1;
        old_data = &data2;
    }
    old_source = source;
}

void wave_packet::update_sum(int m) {
//    sum.s.row(m) = old_data->row(0) + data->row(0);
//    sum.d.row(m) = old_data->row(old_data->n_rows - 1) + data->row(data->n_rows - 1);
    int n = (m - 1) % time_evolution::memory_cutoff;
    int end = old_data->n_rows - 1;
    sum.s.row(n) = old_data->row(  0) + data->row(  0);
    sum.d.row(n) = old_data->row(end) + data->row(end);
}

void wave_packet::update_E(const device & d, const potential & phi, const potential & phi0) {
    using namespace std;

    for (unsigned i = 0; i < E.n_cols; ++i) {
        for (unsigned j = 1; j < E.n_rows - 1; ++j) {
            double n = 1.0 / (norm((*data)(2 * j, i)) + norm((*data)(2 * j + 1, i)));
            double m1 = 2 * (real((*data)(2 * j, i)) * real((*data)(2 * j + 1, i)) + imag((*data)(2 * j, i)) * imag((*data)(2 * j + 1, i)));
            arma::cx_double m2 = conj((*data)(2 * j, i)) * (*data)(2 * j - 1, i);
            arma::cx_double m3 = conj((*data)(2 * j + 1, i)) * (*data)(2 * j + 2, i);
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
