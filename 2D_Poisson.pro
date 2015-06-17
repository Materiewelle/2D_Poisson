TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += .
DEPENDPATH += .

SOURCES += main.cpp

HEADERS += \
    anderson.hpp \
    charge_density.hpp \
    constant.hpp \
    current.hpp \
    fermi.hpp \
    green.hpp \
    integral.hpp \
    inverse.hpp \
    steady_state.hpp \
    wave_packet.hpp \
    gnuplot.hpp \
    rwth.hpp \
    time_evolution.hpp \
    voltage.hpp \
    sd_quantity.hpp \
    movie.hpp \
    inverter.hpp \
    brent.hpp \
    device.hpp \
    potential.hpp \
    system.hpp \
    signal.hpp

LIBS += -lblas -lgomp -lsuperlu

QMAKE_CXXFLAGS = -std=c++14 -march=native -fopenmp

# optimize as hard as possible in release mode
QMAKE_CXXFLAGS_RELEASE = -Ofast -fno-finite-math-only
