from libcpp.string cimport string
from libcpp cimport bool as bool_t

cdef extern from "ceres_fit.hpp":

    double detuning(const double, const bool_t, const double*, const double*)

    double complex resonance(const double, bool_t, const double*, const double*)

    double complex baseline(const double, const double, const double*)

    double complex model(const double, const double, const bool_t, const double*, const double*, const double*,
                         const double*, const double*)

    double complex calibrate(const double, double&, double&, const double, const double*, const double*, const double*)

    string fit(const double*, const double*, const double*, const unsigned int, const double, const bool_t,
               const bool_t, const bool_t, const bool_t, const bool_t, const bool_t, double*, double*, double*, double*,
               double*)

