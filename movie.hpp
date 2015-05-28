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

static inline void shell(const std::stringstream & command) {
    system(command.str().c_str());
}
static inline void system(const std::string & s) {
    system(s.c_str());
}

static inline auto now() {
    using namespace std;
    time_t rawtime;
    struct tm* timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,80,"%Y-%m-%d-%H-%M-%S",timeinfo);
    return string(buffer);
}

static inline std::string lattice_name(int i) {
    std::string names[6] = {
        "left_valence",
        "right_valence",
        "left_conduction",
        "right_conduction",
        "left_tunnel",
        "right_tunnel" };
    return names[i];
}

class movie {
public:
    const std::string parent_folder = "/tmp/movie_tmpdir";
    const std::string folder = parent_folder + "/" + now();

    inline void frame(const int m, const potential & phi, const wave_packet psi[6]);
    inline void mp4(const wave_packet psi[6]);

    inline movie(const device & dev, const wave_packet psi[6], const arma::uvec E_i[6]);

private:
    int calls;
    int frames;
    const int frame_skip = 5;

    const double phimin = -1.5;
    const double phimax = +0.5;

    device d;
    arma::uvec E_ind[6];

    gnuplot gp;
    arma::vec band_offset;

    inline std::string output_folder(int lattice, double E);
    inline std::string output_file(int lattice, double E, int frame_number);
};

movie::movie(const device & dev, const wave_packet psi[6], const arma::uvec E_i[6])
    : calls(0), frames(0), d(dev), band_offset(d.N_x) {

    std::copy(E_i, E_i + 6, E_ind);

    // produce folder tree
    for (int i = 0; i < 6; ++i) {
        if (psi[i].E.size() < 1) {
            continue;
        }
        for (unsigned j = 0; j < E_ind[i].size(); ++j) {
            system("mkdir -p " + output_folder(i, psi[i].E(E_ind[i](j))));
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

    // band offsets for band drawing
    band_offset(d.sc).fill(0.5 * d.E_gc);
    band_offset(d.s).fill(0.5 * d.E_g);
    band_offset(d.sox).fill(0.5 * d.E_g);
    band_offset(d.g).fill(0.5 * d.E_g);
    band_offset(d.dox).fill(0.5 * d.E_g);
    band_offset(d.d).fill(0.5 * d.E_g);
    band_offset(d.dc).fill(0.5 * d.E_gc);
}

void movie::frame(const int m, const potential & phi, const wave_packet psi[6]) {
    using namespace arma;

    if ((calls++ % frame_skip) == 0) {
        std::cout << "Producing movie-frames... ";
        std::flush(std::cout);
        for (int i = 0; i < 6; ++i) {

            if (psi[i].E.size() < 1) {
                continue;
            }

            for (unsigned j = 0; j < E_ind[i].size(); ++j) {

                double E = psi[i].E(E_ind[i](j)) - ((i % 2 == 0) ? phi.s() : phi.d());

                // this is a line that indicates the wave's injection energy
                vec E_line(d.N_x);
                E_line.fill(E);

                // set correct output file
                gp << "set output \"" << output_file(i, psi[i].E(E_ind[i](j)), frames++) << "\"\n";
                gp << "set multiplot layout 1,2 title 't = " << std::setprecision(4) << m * t::dt << " ps'\n";

                // psi-plot:
                gp << "set xlabel 'x / nm'\n";
                gp << "set key top right\n";
                gp << "set ylabel '{/Symbol Y}'\n";
                gp << "set yrange [-3:3]\n";
                gp << "p "
                      "'-' w l ls 1 lw 2 t 'real', "
                      "'-' w l ls 2 lw 2 t 'imag', "
                      "'-' w l ls 3 lw 2 t 'abs', "
                      "'-' w l ls 3 lw 2 notitle\n";
                arma::vec data[4];
                data[0] = arma::real(psi[i].data.col(j));
                data[1] = arma::imag(psi[i].data.col(j));
                data[2] = arma::abs(psi[i].data.col(j));
                data[3] = -data[2];
                for (unsigned p = 0; p < 4; ++p) {
                    for(int k = 0; k < d.N_x; ++k) {
                        gp << d.x(k) << " " << data[p](2 * k) << std::endl;
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
                data[0] = phi.data - band_offset;
                data[1] = phi.data + band_offset;
                data[2] = E_line;


                for (unsigned p = 0; p < 3; ++p) {
                    for(int k = 0; k < d.N_x; ++k) {
                        gp << d.x(k) << " " << data[p](k) << std::endl;
                    }
                    gp << "e" << std::endl;
                }

                gp << "unset multiplot\n";
            }
        }
        std::cout << "done!" << std::endl;
    }

}

void movie::mp4(const wave_packet psi[6]) {
    static const std::string ffmpeg_call = "ffmpeg "
                                           "-framerate 40 "
                                           "-pattern_type glob -i '*.png' "
                                           "-c:v libx264 -r 30 -pix_fmt yuv420p"
                                           " movie.mp4";
    for (int i = 0; i < 6; ++i) {

        if (psi[i].E.size() < 1) {
            continue;
        }

        for (unsigned j = 0; j < E_ind[i].size(); ++j) {
            // change to output folder
            system("cd " + output_folder(i, j));

            // run the command to make an mp4 out of all the stuff
            system(ffmpeg_call);
        }
    }
}

std::string movie::output_folder(int lattice, double E) {
    std::stringstream ss;
//    psi_[lattice].E(E_ind_[lattice](E_number))
    ss << folder << "/" << lattice_name(lattice) << "/energy=" << E << "eV";
    return ss.str();
}

std::string movie::output_file(int lattice, double E, int frame_number) {
    using namespace std;

    std::stringstream ss;
    ss << output_folder(lattice, E) << "/" << setfill('0') << setw(4) << frame_number << ".png";
    return ss.str();
}



#endif
