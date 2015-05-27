#ifndef MOVIE_HPP
#define MOVIE_HPP

#include <string>
#include <ctime>
#include <armadillo>
#include <stdio.h>
#include <utility>

#include "wave_packet.hpp"
#include "gnuplot.hpp"
#include "device.hpp"

class movie {
public:
    const string parent_folder = "/tmp/movie_tmpdir";
    const string folder = parent_folder + "/" + ctime(& time(0));

    inline void frame(const double t, const potential & phi);
    inline void mp4();

    movie(const wave_packet psi[6], const arma::uvec E_ind[6]);

private:
    int calls = 0;
    int frames = 0;

    const double phimin;
    const double phimax;

    const wave_packet psi_[6];
    const arma::uvec E_ind_[6];
    const int frame_skip = 5;

    gnuplot gp;
    device d;
};

movie::movie(device & dev, const wave_packet psi[6], const arma::uvec E_ind[6]) : d(dev), psi_(psi), E_ind_(E_ind), gp() {

    phimin = ((arma::min(psi_[0].E) < arma::min(psi_[1].E) ? arma::min(psi_[0].E) : arma::min(psi_[1].E))) - 0.2;
    phimax = ((arma::max(psi_[2].E) < arma::max(psi_[3].E) ? arma::max(psi_[2].E) : arma::max(psi_[3].E))) + 0.2;

    // produce folder tree
    system("mkdir -p " + folder);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < E_indices_[i].n_rows; ++j) {
            system("mkdir -p "
                   + folder
                   + "E_lattice=" + std::to_string(i)
                   + "/energy=" + std::to_string(psi_[i].E(j)));
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

void movie::frame(const double t, const potential & phi) {
    arma::vec E_line(d.N_x);
    std::cout << "Producing movie-frames... ";
    std::flush(std::cout);
    using namespace arma;

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
            for (int j = 0; j < E_indices_[i].n_rows; ++j) {

                // this is a line that indicates the wave's injection energy
                E_line.fill(psi[i].E(E_indices_[i](j)));
                E_line -= (i % 2 == 0) ? phi.s() : phi.d();

                // set correct output file
                char filename[7];
                snprintf(filename, 7, "%04d.png", calls);
                gp << "set output " << folder
                   << "E_lattice=" << std::to_string(i)
                   << "/energy=" << std::to_string(psi_[i].E(j))
                   << "/" << filename << "\n";

                char timestring[5];
                snprintf(time, 5, "%1.4f", t * 1e12); // time in picoseconds
                gp << "multiplot layout 1,2 title 't = " << timestring << " ps'\n";

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
                    for(unsigned k = 0; k < psi_[i].n_rows; ++k) {
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
                data[2] = arma::abs(psi[i].data.col(j));
                for (unsigned p = 0; p < 3; ++p) {
                    for(unsigned k = 0; k < psi_[i].n_rows; ++k) {
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
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < E_indices_[i].n_rows; ++j) {
            // change to the correct folder
            system("cd " + folder
                   + "E_lattice=" + std::to_string(i)
                   + "/energy=" + std::to_string(psi_[i].E(j)));

            // run the command to make an mp4 out of all the stuff
            system("ffmpeg -framerate 40 -pattern_type glob -i 'tmpfs_dir/*.png' -c:v libx264 -r 30 -pix_fmt yuv420p movie.mp4");
        }
    }
}



#endif
