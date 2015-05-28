#ifndef MOVIE_HPP
#define MOVIE_HPP

#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <armadillo>
#include <stdio.h>
#include <utility>

#include "wave_packet.hpp"
#include "gnuplot.hpp"
#include "device.hpp"
#include "time_params.hpp"

static auto now() {
    using namespace std;
    time_t rawtime;
    struct tm* timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,80,"%Y-%m-%d-%H-%M-%S",timeinfo);
    return string(buffer);
}

class movie {
public:
    const std::string parent_folder = "/tmp/movie_tmpdir";
    const std::string folder = parent_folder + "/" + now();

    inline void frame(const int m, const potential & phi);
    inline void mp4();

    movie(device & dev, wave_packet psi[6], arma::uvec E_ind[6]);

private:
    int calls = 0;
    int frames = 0;

    double phimin = -1.5;
    double phimax = +0.5;

    wave_packet * psi_;
    arma::uvec * E_ind_;
    const int frame_skip = 5;

    gnuplot gp;
    device d;
};

movie::movie(device & dev, wave_packet psi[6], arma::uvec E_ind[6]) : gp(), d(dev) {

    psi_ = psi;
    E_ind_ = E_ind;

    phimin = ((arma::min(psi_[0].E) < arma::min(psi_[1].E) ? arma::min(psi_[0].E) : arma::min(psi_[1].E))) - 0.2;
    phimax = ((arma::max(psi_[2].E) < arma::max(psi_[3].E) ? arma::max(psi_[2].E) : arma::max(psi_[3].E))) + 0.2;

    std::stringstream ss("");

    // produce folder tree
    ss << "mkdir -p " << folder;
    system(ss.str().c_str());
    for (int i = 0; i < 6; ++i) {
        if (psi_[i].E.n_elem < 1) continue;
        for (unsigned j = 0; j < E_ind_[i].n_rows; ++j) {
            ss.str("");
            ss << "mkdir -p " << folder << "/E_lattice=" << i << "/energy=" << psi_[i].E(E_ind_[i](j));
            system(ss.str().c_str());
        }
    }

    // gnuplot setup
    gp << "set terminal pngcairo rounded enhanced colour size 960,540 font 'arial,16'\n";
    gp << "set style line 66 lc rgb RWTH_Schwarz_50 lt 1 lw 2\n";
    gp << "set border 3 ls 66\n";
    gp << "set tics nomirror\n";
    gp << "set style line 1 lc rgb RWTH_Gruen\n";
    gp << "set style line 2 lc rgb RWTH_Rot\n";
    gp << "set style line 3 lc rgb RWTH_Blau\n";
    gp << "set style line 4 lc rgb RWTH_Blau\n";
}

void movie::frame(const int m, const potential & phi) {
    using namespace arma;
    vec E_line(d.N_x);
    std::cout << "Producing movie-frames... ";
    std::flush(std::cout);

    vec vband = phi.data;
    vband(d.sc)  += -0.5 * d.E_gc;
    vband(d.s)   += -0.5 * d.E_g;
    vband(d.sox) += -0.5 * d.E_g;
    vband(d.g)   += -0.5 * d.E_g;
    vband(d.dox) += -0.5 * d.E_g;
    vband(d.d)   += -0.5 * d.E_g;
    vband(d.dc)  += -0.5 * d.E_gc;

    vec cband = phi.data;
    cband(d.sc)  += +0.5 * d.E_gc;
    cband(d.s)   += +0.5 * d.E_g;
    cband(d.sox) += +0.5 * d.E_g;
    cband(d.g)   += +0.5 * d.E_g;
    cband(d.dox) += +0.5 * d.E_g;
    cband(d.d)   += +0.5 * d.E_g;
    cband(d.dc)  += +0.5 * d.E_gc;

    if ((calls++ % frame_skip) == 0) {
        for (int i = 0; i < 6; ++i) {
            if (psi_[i].E.n_elem < 1) continue;
            for (unsigned j = 0; j < E_ind_[i].n_rows; ++j) {

                // this is a line that indicates the wave's injection energy
                E_line.fill(psi_[i].E(E_ind_[i](j)));
                E_line -= (i % 2 == 0) ? phi.s() : phi.d();

                // set correct output file
                char filename[20];
                snprintf(filename, 7, "%04d.png", calls);
                gp << "set output \"" << folder
                   << "/E_lattice=" << i
                   << "/energy=" << psi_[i].E(E_ind_[i](j))
                   << "/" << filename << "\"\n";

                char timestring[20];
                snprintf(timestring, 5, "%1.4f", m * t::dt * 1e12); // time in picoseconds
                gp << "set multiplot layout 1,2 title 't = " << timestring << " ps'\n";

                // psi-plot:
                gp << "set xlabel 'x / nm'\n";
                gp << "set key top right\n";
                gp << "set ylabel '{/Symbol Y}'\n";
                gp << "set yrange [-3:3]\n";
                gp << "p "
                      "'-' w l ls 1 lw 2 t 'real', "
                      "'-' w l ls 2 lw 2 t 'imag', "
                      "'-' w l ls 3 lw 2 t '+abs', "
                      "'-' w l ls 3 lw 2 t '-abs'\n";
                arma::vec data[4];
                data[0] = arma::real(psi_[i].data.col(j));
                data[1] = arma::imag(psi_[i].data.col(j));
                data[2] = arma::abs(psi_[i].data.col(j));
                data[3] = -data[2];
                for (unsigned p = 0; p < 4; ++p) {
                    for(unsigned k = 0; k < d.N_x; ++k) {
                        gp << d.x(k) << " " << data[p](k) << std::endl;
                    }
                    gp << "e" << std::endl;
                }

                // phi-plot:
                gp << "set ylabel 'E / eV'\n";
                gp << "set yrange [" << phimin << ":" << phimax << "]\n";
                gp << "p "
                      "'-' w l ls 3 lw 2 notitle, "
                      "'-' w l ls 3 lw 2 notitle, "
                      "'-' w l ls 2 lw 2 t 'injection energy'\n";
                data[0] = cband;
                data[1] = vband;
                data[2] = arma::abs(psi_[i].data.col(j));
                for (unsigned p = 0; p < 3; ++p) {
                    for(unsigned k = 0; k < d.N_x; ++k) {
                        gp << d.x(k) << " " << data[p](k) << std::endl;
                    }
                    gp << "e" << std::endl;
                }

                gp << "unset multiplot\n";
            }
        }
    }
    std::cout << "done!" << std::endl;
}

void movie::mp4() {
    std::stringstream ss;
    for (int i = 0; i < 6; ++i) {
        if (psi_[i].E.n_elem < 1) continue;
        for (unsigned j = 0; j < E_ind_[i].n_rows; ++j) {
            ss.str("");
            ss << "cd " << folder << "/E_lattice=" << i << "/energy=" << psi_[i].E(E_ind_[i](j));
            system(ss.str().c_str());

            // run the command to make an mp4 out of all the stuff
            system("ffmpeg -framerate 40 -pattern_type glob -i 'tmpfs_dir/*.png' -c:v libx264 -r 30 -pix_fmt yuv420p movie.mp4");
        }
    }
}



#endif
