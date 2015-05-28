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

static void shell(const std::stringstream & command) {
    system(command.str().c_str());
    return;
}

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

static std::string lattice_name(int i) {
    switch (i) {
    case 0:
        return("left_valence");
    case 1:
        return("right_valence");
    case 2:
        return("left_conduction");
    case 3:
        return("right_conduction");
    case 4:
        return("left_tunnel");
    case 5:
        return("right_tunnel");
    default:
        return("unknown");
    }
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

    const double phimin = -1.5;
    const double phimax = +0.5;

    wave_packet * psi_;
    arma::uvec * E_ind_;
    const int frame_skip = 5;

    gnuplot gp;
    device d;
    arma::vec band_offset;

    inline std::string output_folder(const int lattice, const int E_number);
    inline std::string output_file(const int lattice, const int E_number, const int frame_number);
};

movie::movie(device & dev, wave_packet psi[6], arma::uvec E_ind[6]) : gp(), d(dev), band_offset(d.N_x) {

    psi_ = psi;
    E_ind_ = E_ind;

    std::stringstream ss("");
    // produce folder tree
    for (int i = 0; i < 6; ++i) {
        if (psi_[i].E.size() < 1) continue;
        for (unsigned j = 0; j < E_ind_[i].size(); ++j) {
            ss.str("");
            ss << "mkdir -p " << output_folder(i, j);
            shell(ss);
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


    // band offsets for band drowing
    band_offset(d.sc)  += -0.5 * d.E_gc;
    band_offset(d.s)   += -0.5 * d.E_g;
    band_offset(d.sox) += -0.5 * d.E_g;
    band_offset(d.g)   += -0.5 * d.E_g;
    band_offset(d.dox) += -0.5 * d.E_g;
    band_offset(d.d)   += -0.5 * d.E_g;
    band_offset(d.dc)  += -0.5 * d.E_gc;
}

void movie::frame(const int m, const potential & phi) {
    using namespace arma;
    vec E_line(d.N_x);

    if ((calls++ % frame_skip) == 0) {
        std::cout << "Producing movie-frames... ";
        std::flush(std::cout);
        for (int i = 0; i < 6; ++i) {
            if (psi_[i].E.size() < 1) continue;
            for (unsigned j = 0; j < E_ind_[i].size(); ++j) {

                double E = psi_[i].E(E_ind_[i](j)) - ((i % 2 == 0) ? phi.s() : phi.d());

                // this is a line that indicates the wave's injection energy
                E_line.fill(E);

                // set correct output file
                gp << "set output \"" << output_file(i, j, frames++) << "\"\n";

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
                data[0] = arma::real(psi_[i].data.col(j));
                data[1] = arma::imag(psi_[i].data.col(j));
                data[2] = arma::abs(psi_[i].data.col(j));
                data[3] = -data[2];
                for (unsigned p = 0; p < 4; ++p) {
                    for(int k = 0; k < d.N_x; ++k) {
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
                data[0] = phi.data - band_offset;
                data[1] = phi.data + band_offset;
                data[2] = arma::abs(psi_[i].data.col(j));


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

void movie::mp4() {
    static const std::stringstream ffmpeg_call("ffmpeg "
                                               "-framerate 40 "
                                               "-pattern_type glob -i '*.png' "
                                               "-c:v libx264 -r 30 -pix_fmt yuv420p"
                                               " movie.mp4");
    for (int i = 0; i < 6; ++i) {
        if (psi_[i].E.n_elem < 1) continue;
        for (unsigned j = 0; j < E_ind_[i].n_rows; ++j) {
            // change to output folder
            std::stringstream ss("cd ");
            ss << output_folder(i, j);
            shell(ss);

            // run the command to make an mp4 out of all the stuff
            shell(ffmpeg_call);
        }
    }
}

std::string movie::output_folder(const int lattice, const int E_number) {
    std::stringstream ss("");
    ss << folder << "/" << lattice_name(lattice) << "/energy=" << psi_[lattice].E(E_ind_[lattice](E_number)) << "eV";
    return ss.str();
}

std::string movie::output_file(const int lattice, const int E_number, const int frame_number) {
    std::stringstream ss("");
    ss << output_folder(lattice, E_number) << "/" << std::setfill('0') << std::setw(4) << frame_number << ".png";
    return ss.str();
}



#endif
