from libcpp.string cimport string
from libcpp cimport bool as bool_t

cdef extern from "ceres_fit.hpp":
    double complex resonance(double, bool_t, const double*, const double*)
    double complex baseline(double, double, const double*)
    double complex model(double, double, bool_t, const double*, const double*, const double*, const double*,
                         const double*)
    double complex calibrate(double, double&, double&, double, double*, double*, double*);
    string fit(double*, double*, double*, unsigned int, double, bool_t, bool_t, bool_t, bool_t, bool_t, double*,
               double*, double*, double*, double*)

