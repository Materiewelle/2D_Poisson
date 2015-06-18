#ifndef MOVIE_HPP
#define MOVIE_HPP

#include <iomanip>
#include <vector>

#include "time_evolution.hpp"

class movie {
public:
    // provide an initialized time_evolution object with solved steady-state
    inline movie(time_evolution & t_ev, const std::vector<std::pair<int, int>> & E_i);

private:
    int frames; // the current number of frames that have been produced
    static constexpr int frame_skip = 1;

    static constexpr double phimin = -1.5;
    static constexpr double phimax = +1.0;

    time_evolution & te;
    std::vector<std::pair<int, int>> E_ind;

    gnuplot gp;
    arma::vec band_offset;

    inline void frame();
    inline void mp4();

    inline std::string output_folder(int lattice, double E);
    inline std::string output_file(int lattice, double E, int frame_number);
};

static inline const std::string & lattice_name(int i) {
    static const std::string names[6] = {
        "left_valence",
        "right_valence",
        "left_conduction",
        "right_conduction",
        "left_tunnel",
        "right_tunnel"
    };

    return names[i];
}

movie::movie(time_evolution & t_ev, const std::vector<std::pair<int, int>> & E_i)
    : frames(0), te(t_ev), E_ind(E_i), band_offset(te.d.N_x) {
    using namespace std::string_literals;

    // produce folder tree
    for (unsigned i = 0; i < E_ind.size(); ++i) {
        int lattice = E_ind[i].first;
        double E = te.psi[lattice].E0(E_ind[i].second);
        system("mkdir -p " + output_folder(lattice, E));
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
    band_offset(te.d.sc).fill(0.5 * te.d.E_gc);
    band_offset({te.d.sox.a, te.d.dox.b}).fill(0.5 * te.d.E_g);
    band_offset(te.d.dc).fill(0.5 * te.d.E_gc);

    t_ev.add_callback([this] () {
        this->frame();
    });
}

void movie::frame() {
    using namespace arma;

    if (((te.m - 1) % frame_skip) == 0) {
        for (unsigned i = 0; i < E_ind.size(); ++i) {
            int lattice = E_ind[i].first;
            double E = te.psi[lattice].E0(E_ind[i].second);

            // this is a line that indicates the wave's injection energy
            vec E_line(te.d.N_x);
            E_line = te.psi[lattice].E.col(E_ind[i].second);

            // set correct output file
            gp << "set output \"" << output_file(lattice, E, frames) << "\"\n";
            gp << "set multiplot layout 1,2 title 't = " << std::setprecision(3) << std::fixed << (te.m - 1) * time_evolution::dt * 1e12 << " ps'\n";


            /* Having all the stuff we want to plot
             * organized in an array allows us to use
             * a loop and kill some redundancy */
            arma::vec data[7];

            arma::cx_vec wavefunction = te.psi[lattice].data->col(E_ind[i].second);
            data[0] = arma::real(wavefunction);
            data[1] = arma::imag(wavefunction);
            data[2] = +arma::abs(wavefunction);
            data[3] = -arma::abs(wavefunction);

            data[4] = te.phi[te.m - 1].data - band_offset;
            data[5] = te.phi[te.m - 1].data + band_offset;
            data[6] = E_line;

            // setup psi-plot:
            gp << "set xlabel 'x / nm'\n";
            gp << "set key top right\n";
            gp << "set ylabel '{/Symbol Y}'\n";
            gp << "set yrange [-3:3]\n";
            gp << "p "
                  "'-' w l ls 1 lw 2 t 'real', "
                  "'-' w l ls 2 lw 2 t 'imag', "
                  "'-' w l ls 3 lw 2 t 'abs', "
                  "'-' w l ls 3 lw 2 notitle\n";

            // pipe data to gnuplot
            for (unsigned p = 0; p < 7; ++p) {
                for(int k = 0; k < te.d.N_x; ++k) {
                    if (p == 4) { // setup bands-plot
                        gp << "set ylabel 'E / eV'\n";
                        gp << "set yrange [" << phimin << ":" << phimax << "]\n";
                        gp << "p "
                              "'-' w l ls 3 lw 2 notitle, "
                              "'-' w l ls 3 lw 2 t 'band edges', "
                              "'-' w l ls 2 lw 2 t '<E_{{/Symbol Y}}>(x)'\n";
                    }
                    gp << te.d.x(k) << " " << ((p < 4) ? data[p](2 * k) : data[p](k)) << std::endl;
                }
                gp << "e" << std::endl;
            }

            gp << "unset multiplot\n";
        }
        std::flush(gp);
        std::cout << "produced a movie-frame in this step!" << std::endl;

        // update frame number
        ++frames;
    }
}

void movie::mp4() {
    std::cout << "producung mp4 video files from frames...";
    std::flush(std::cout);
    static const std::string ffmpeg1 = "ffmpeg "
                                       "-framerate 30 "
                                       "-pattern_type glob -i '";
    static const std::string ffmpeg2 = "/*.png' "
                                       "-c:v libx264 -r 30 -pix_fmt yuv420p ";

    for (unsigned i = 0; i < E_ind.size(); ++i) {
        int lattice = E_ind[i].first;
        double E = te.psi[lattice].E0(E_ind[i].second);
        // run the command to make an mp4 out of all the stuff
        system(ffmpeg1 + output_folder(lattice, E) + ffmpeg2 + output_folder(lattice, E) + "/movie.mp4");
    }
    std::cout << " done!" << std::endl;
}

std::string movie::output_file(int lattice, double E, int frame_number) {
    std::stringstream ss;
    ss << output_folder(lattice, E) << "/" << std::setfill('0') << std::setw(4) << frame_number << ".png";
    return ss.str();
}

std::string movie::output_folder(int lattice, double E) {
    std::stringstream ss;
    ss << save_folder() << "/" << lattice_name(lattice) << "/energy=" << std::setprecision(2) << E << "eV";
    return ss.str();
}

#endif
