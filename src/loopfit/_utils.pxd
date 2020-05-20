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
cdef double DEFAULT_GAMMA
cdef double DEFAULT_I_OFFSET
cdef double DEFAULT_Q_OFFSET

# independent variable defaults
cdef double DEFAULT_FM
cdef bool_t DEFAULT_DECREASING

# numpy types
ctypedef np.float64_t DTYPE_float64_t
ctypedef np.complex128_t DTYPE_complex128_t


cdef inline create_resonance_block(double pr[4], double qi, double qc, double f0, double xa):
    pr[:] = [qi, qc, f0, xa]


cdef inline create_detuning_block(double pd[1], double a):
    pd[:] = [sqrt(a)]


cdef inline create_baseline_block(double pb[5], double gain0, double gain1, double gain2, double phase0, double phase1):
    pb[:] = [gain0, gain1, gain2, phase0, phase1]


cdef inline create_imbalance_block(double pi[2], double alpha, double gamma):
    pi[:] = [alpha, gamma]


cdef inline create_offset_block(double po[2], double i_offset, double q_offset):
    po[:] = [i_offset, q_offset]


cdef inline create_parameter_blocks(double pr[4], double pd[1], double pb[5], double pi[2], double po[2], double qi,
                                    double qc, double f0, double xa, double a, double gain0, double gain1, double gain2,
                                    double phase0, double phase1, double alpha, double gamma, double i_offset,
                                    double q_offset):
    create_resonance_block(pr, qi, qc, f0, xa)
    create_detuning_block(pd, a)
    create_baseline_block(pb, gain0, gain1, gain2, phase0, phase1)
    create_imbalance_block(pi, alpha, gamma)
    create_offset_block(po, i_offset, q_offset)
