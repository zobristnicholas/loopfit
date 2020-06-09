from libcpp.string cimport string
from libcpp cimport bool as bool_t

cdef extern from "ceres_fit.hpp":
    double detuning[T](const T, const bool_t, const double*, const double*)

    double complex resonance[T](const T, bool_t, const double*, const double*)

    double complex baseline[T](const T, const double, const double*)

    double complex mixer(const double complex, const double*, const double*)

    double complex model[T](const T, const double, const bool_t, const double*, const double*, const double*,
                            const double*, const double*)

    void calibrate[T, U](const T, U&, U&, const bool_t, const double, const double*, const double*, const double*,
                         const double*)

    string fit[T, U](const T*, const U*, const U*, const double complex*, const unsigned int, const double,
                     const bool_t, const bool_t, const bool_t, const bool_t, const bool_t, const bool_t, const int,
                     int&, int&, bool_t&, double*, double*, double*, double*, double*)
