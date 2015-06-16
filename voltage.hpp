#ifndef VOLTAGE_HPP
#define VOLTAGE_HPP

class voltage {
public:
    double s;
    double g;
    double d;
};

inline voltage operator+(const voltage & V0, const voltage & V1) {
    return voltage { V0.s + V1.s, V0.g + V1.g, V0.d + V1.d };
}
inline voltage operator-(const voltage & V0, const voltage & V1) {
    return voltage { V0.s - V1.s, V0.g - V1.g, V0.d - V1.d };
}
inline voltage operator-(const voltage & V0) {
    return voltage { - V0.s, - V0.g, - V0.d };
}
inline voltage operator*(const voltage & V0, double m) {
    return voltage { V0.s * m, V0.g * m, V0.d * m };
}
inline voltage operator*(double m, const voltage & V0) {
    return V0 * m;
}
inline voltage operator/(const voltage & V0, double m) {
    return voltage { V0.s / m, V0.g / m, V0.d / m };
}

#endif
