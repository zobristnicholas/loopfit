cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow
from libcpp cimport bool as bool_t


# fit parameter defaults
cdef double DEFAULT_QI
cdef double DEFAULT_QC
cdef double DEFAULT_F0
cdef double DEFAULT_XA
cdef double DEFAULT_A
cdef double DEFAULT_GAIN0
cdef double DEFAULT_GAIN1
cdef double DEFAULT_GAIN2
cdef double DEFAULT_PHASE0
cdef double DEFAULT_PHASE1
cdef double DEFAULT_ALPHA
cdef double DEFAULT_BETA
cdef double DEFAULT_GAMMA
cdef double DEFAULT_DELTA

# independent variable defaults
cdef double DEFAULT_FM
cdef bool_t DEFAULT_DECREASING

# numpy types
ctypedef np.float32_t float32_t
ctypedef np.float64_t float64_t
ctypedef fused  float_t:
    np.float32_t
    np.float64_t
ctypedef fused  float2_t:  # for when two float types are needed
    np.float32_t
    np.float64_t
ctypedef np.complex128_t complex64_t
ctypedef np.complex128_t complex128_t
ctypedef fused  complex_t:
    np.complex64_t
    np.complex128_t


cdef inline create_resonance_block(double pr[4], double qi, double qc, double f0, double xa):
    pr[:] = [qi, qc, f0, xa]


cdef inline create_detuning_block(double pd[1], double a):
    pd[:] = [sqrt(a)]


cdef inline create_baseline_block(double pb[5], double gain0, double gain1, double gain2, double phase0, double phase1):
    pb[:] = [gain0, gain1, gain2, phase0, phase1]


cdef inline create_imbalance_block(double pi[2], double alpha, double beta):
    pi[:] = [alpha, beta]


cdef inline create_offset_block(double po[2], double gamma, double delta):
    po[:] = [gamma, delta]


cdef inline create_parameter_blocks(double pr[4], double pd[1], double pb[5], double pi[2], double po[2], double qi,
                                    double qc, double f0, double xa, double a, double gain0, double gain1, double gain2,
                                    double phase0, double phase1, double alpha, double beta, double gamma,
                                    double delta):
    create_resonance_block(pr, qi, qc, f0, xa)
    create_detuning_block(pd, a)
    create_baseline_block(pb, gain0, gain1, gain2, phase0, phase1)
    create_imbalance_block(pi, alpha, beta)
    create_offset_block(po, gamma, delta)
