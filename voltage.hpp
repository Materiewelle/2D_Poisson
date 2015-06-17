#ifndef VOLTAGE_HPP
#define VOLTAGE_HPP

template<class T>
class triple {
public:
    T s;
    T g;
    T d;

    inline triple(T ss, T gg, T dd);
    inline triple(T v);
    inline triple();
};

template<class T>
triple<T>::triple(T ss, T gg, T dd)
    : s(ss), g(gg), d(dd) {
}

template<class T>
triple<T>::triple(T v)
    : triple(v, v, v) {
}

template<class T>
triple<T>::triple()
    : triple(T()) {
}

template<class T>
inline triple<T> operator+(const triple<T> & t0, double x) {
    return triple<T> { t0.s + x, t0.g + x, t0.d + x };
}
template<class T>
inline triple<T> operator+(double x, const triple<T> & t0) {
    return t0 + x;
}
template<class T>
inline triple<T> operator+(const triple<T> & t0, const triple<T> & t1) {
    return triple<T> { t0.s + t1.s, t0.g + t1.g, t0.d + t1.d };
}
template<class T>
inline triple<T> operator-(const triple<T> & t0) {
    return triple<T> { - t0.s, - t0.g, - t0.d };
}
template<class T>
inline triple<T> operator-(const triple<T> & t0, double x) {
    return t0 + (-x);
}
template<class T>
inline triple<T> operator-(double x, const triple<T> & t0) {
    return -(t0 - x);
}
template<class T>
inline triple<T> operator-(const triple<T> & t0, const triple<T> & t1) {
    return t0 + (-t1);
}
template<class T>
inline triple<T> operator*(const triple<T> & t0, double m) {
    return triple<T> { t0.s * m, t0.g * m, t0.d * m };
}
template<class T>
inline triple<T> operator*(double m, const triple<T> & t0) {
    return t0 * m;
}
template<class T>
inline triple<T> operator*(const triple<T> & t0, const triple<T> & t1) {
    return triple<T> { t0.s * t1.s, t0.g * t1.g, t0.d * t1.d };
}
template<class T>
inline triple<T> operator/(const triple<T> & t0, double m) {
    return triple<T> { t0.s / m, t0.g / m, t0.d / m };
}
template<class T, class F>
inline triple<T> func(F && f, const triple<T> & t) {
    triple<T> ret;

    ret.s = f(t.s);
    ret.g = f(t.g);
    ret.d = f(t.d);

    return ret;
}

using voltage = triple<double>;
using tripled = triple<double>;
using triplei = triple<int>;

#endif
