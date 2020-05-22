cimport cython
import logging
import numpy as np
cimport numpy as np
from libc.math cimport pow
from libcpp.string cimport string
from libcpp cimport bool as bool_t

from ._utils import *
from ._utils cimport *
from .ceres_fit cimport (resonance as resonance_c, baseline as baseline_c, model as model_c, fit as fit_c,
                         calibrate as calibrate_c, detuning as detuning_c)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


@cython.boundscheck(False)
@cython.wraparound(False)
cdef calibrate_vectorized(np.ndarray[DTYPE_float64_t, ndim=1] f, np.ndarray[DTYPE_float64_t, ndim=1] i,
                          np.ndarray[DTYPE_float64_t, ndim=1] q, double fm, double pb[], double pi[], double po[]):
    if f.shape[0] != i.shape[0] or f.shape[0] != q.shape[0]:
        raise ValueError("All input arrays must have the same size.")
    for ii in range(f.shape[0]):
        calibrate_c(f[ii], i[ii], q[ii], fm, pb, pi, po)


def calibrate(f, i=None, q=None, *,
              z=None,
              double fm=DEFAULT_FM,
              double gain0=DEFAULT_GAIN0,
              double gain1=DEFAULT_GAIN1,
              double gain2=DEFAULT_GAIN2,
              double phase0=DEFAULT_PHASE0,
              double phase1=DEFAULT_PHASE1,
              double alpha=DEFAULT_ALPHA,
              double gamma=DEFAULT_GAMMA,
              double i_offset=DEFAULT_I_OFFSET,
              double q_offset=DEFAULT_Q_OFFSET,
              **kwargs):
    # check inputs
    if fm < 0: fm = np.median(f)  # default depends on f
    f = np.asarray(f, dtype=np.float64)  # no copy if already an array and dtype matches
    # create the parameter blocks
    cdef double pb[5], pi[2], po[2]
    create_baseline_block(pb, gain0, gain1, gain2, phase0, phase1)
    create_imbalance_block(pi, alpha, gamma)
    create_offset_block(po, i_offset, q_offset)
    # calibrate the data
    if (i is not None) and (q is not None):
        i = np.array(i, dtype=np.float64, copy=True)  # copy since calibrate_c is in-place
        q = np.array(q, dtype=np.float64, copy=True)
        calibrate_vectorized(f.ravel(), i.ravel(), q.ravel(), fm, pb, pi, po)
        return i, q
    elif z is not None:
        z = np.array(z, dtype=np.complex128, copy=True)
        calibrate_vectorized(f.ravel(), z.ravel().real, z.ravel().imag, fm, pb, pi, po)
        return z
    else:
        raise ValueError("Neither i and q or z were supplied as keyword arguments.")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef baseline_vectorized(np.ndarray[DTYPE_float64_t, ndim=1] f, double fm, double pb[]):
    cdef np.ndarray[DTYPE_complex128_t, ndim=1] result = np.empty(f.shape[0], dtype=np.complex128)
    for ii in range(f.shape[0]):
        result[ii] = baseline_c(f[ii], fm, pb)
    return result

def baseline(f, *,
             double fm=DEFAULT_FM,
             double gain0=DEFAULT_GAIN0,
             double gain1=DEFAULT_GAIN1,
             double gain2=DEFAULT_GAIN2,
             double phase0=DEFAULT_PHASE0,
             double phase1=DEFAULT_PHASE1,
             **kwargs):
    if fm < 0: fm = np.median(f)  # default depends on f
    f = np.asarray(f, dtype=np.float64)  # no copy if already an array and dtype matches
    # create the parameter blocks
    cdef double pb[5]
    create_baseline_block(&pb[0], gain0, gain1, gain2, phase0, phase1)
    # call function
    result = baseline_vectorized(f.ravel(), fm, &pb[0])
    return result.reshape(f.shape)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef resonance_vectorized(np.ndarray[DTYPE_float64_t, ndim=1] f, bool_t decreasing, double pr[], double pd[]):
    cdef np.ndarray[DTYPE_complex128_t, ndim=1] result = np.empty(f.shape[0], dtype=np.complex128)
    for ii in range(f.shape[0]):
        result[ii] = resonance_c(f[ii], decreasing, pr, pd)
    return result


def resonance(f, *,
              bool_t decreasing=DEFAULT_DECREASING,
              double qi=DEFAULT_QI,
              double qc=DEFAULT_QC,
              double f0=DEFAULT_F0,
              double xa=DEFAULT_XA,
              double a=DEFAULT_A,
              **kwargs):
    f = np.asarray(f, dtype=np.float64)  # no copy if already an array and dtype matches
    # create the parameter blocks
    cdef double pr[4], pd[1]
    create_resonance_block(&pr[0], qi, qc, f0, xa)
    create_detuning_block(&pd[0], a)
    # call function
    result = resonance_vectorized(f.ravel(), decreasing, &pr[0], &pd[0])
    return result.reshape(f.shape)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef detuning_vectorized(np.ndarray[DTYPE_float64_t, ndim=1] f, bool_t decreasing, double pr[], double pd[]):
    cdef np.ndarray[DTYPE_float64_t, ndim=1] result = np.empty(f.shape[0], dtype=np.float64)
    for ii in range(f.shape[0]):
        result[ii] = detuning_c(f[ii], decreasing, pr, pd)
    return result


def detuning(f, *,
             bool_t decreasing=DEFAULT_DECREASING,
             double qi=DEFAULT_QI,
             double qc=DEFAULT_QC,
             double f0=DEFAULT_F0,
             double xa=DEFAULT_XA,
             double a=DEFAULT_A,
             **kwargs):
    if f0 < 0: f0 = kwargs.get(fm, np.median(f))
    f = np.asarray(f, dtype=np.float64)  # no copy if already an array and dtype matches
    # create the parameter blocks
    cdef double pr[4], pd[1]
    create_resonance_block(&pr[0], qi, qc, f0, xa)
    create_detuning_block(&pd[0], a)
    # call function
    result = detuning_vectorized(f.ravel(), decreasing, &pr[0], &pd[0])
    return result.reshape(f.shape)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef model_vectorized(np.ndarray[DTYPE_float64_t, ndim=1] f, double fm, bool_t decreasing, double pr[], double pd[],
                      double pb[], double pi[], double po[]):
    cdef np.ndarray[DTYPE_complex128_t, ndim=1] result = np.empty(f.shape[0], dtype=np.complex128)
    for ii in range(f.shape[0]):
        result[ii] = model_c(f[ii], fm, decreasing, pr, pd, pb, pi, po)
    return result


def model(f, *,
          double fm=DEFAULT_FM,
          bool_t decreasing=DEFAULT_DECREASING,
          double qi=DEFAULT_QI,
          double qc=DEFAULT_QC,
          double f0=DEFAULT_F0,
          double xa=DEFAULT_XA,
          double a=DEFAULT_A,
          double gain0=DEFAULT_GAIN0,
          double gain1=DEFAULT_GAIN1,
          double gain2=DEFAULT_GAIN2,
          double phase0=DEFAULT_PHASE0,
          double phase1=DEFAULT_PHASE1,
          double alpha=DEFAULT_ALPHA,
          double gamma=DEFAULT_GAMMA,
          double i_offset=DEFAULT_I_OFFSET,
          double q_offset=DEFAULT_Q_OFFSET,
          **kwargs):
    # check inputs
    if fm < 0: fm = np.median(f)  # default depends on f
    if f0 < 0: f0 = fm
    f = np.asarray(f, dtype=np.float64)  # no copy if already an array and dtype matches
    # create the parameter blocks
    cdef double pr[4], pd[1], pb[5], pi[2], po[2]
    create_parameter_blocks(&pr[0], &pd[0], &pb[0], &pi[0], &po[0], qi, qc, f0, xa, a, gain0, gain1, gain2, phase0,
                            phase1, alpha, gamma, i_offset, q_offset)
    # call function
    result = model_vectorized(f.ravel(), fm, decreasing, &pr[0], &pd[0], &pb[0], &pi[0], &po[0])
    return result.reshape(f.shape)


def guess(f, i, q, *, nonlinear=False, imbalance=None, offset=None, **kwargs):
        # estimate mixer correction from calibration data
        alpha, gamma, i_offset, q_offset = compute_mixer_calibration(offset, imbalance, **kwargs)
        # remove the IQ mixer offset and imbalance
        i, q = calibrate(f, i, q, alpha=alpha, gamma=gamma, i_offset=i_offset, q_offset=q_offset, gain0=1.0,
                         gain1=0.0, gain2=0.0, phase0=0.0, phase1=0.0, fm=1.0)
        # compute the magnitude and phase of the scattering parameter
        magnitude = np.sqrt(i**2 + q**2)
        phase = np.unwrap(np.arctan2(q, i))
        # calculate useful indices
        f_index_end = len(f) - 1  # last frequency index
        f_index_5pc = max(int(len(f) * 0.05), 2)  # end of first 5% of data
        # set up a unit-less, reduced, midpoint frequency for baselines
        fm = kwargs.get('fm', np.median(f))  # frequency at the center of the data
        def xm(fx): return (fx - fm) / fm
        # get the magnitude and phase data to fit
        mag_ends = np.concatenate((magnitude[:f_index_5pc], magnitude[-f_index_5pc + 1:]))
        phase_ends = np.concatenate((phase[:f_index_5pc], phase[-f_index_5pc + 1:]))
        freq_ends = xm(np.concatenate((f[:f_index_5pc], f[-f_index_5pc + 1:])))
        # calculate the gain polynomials
        gain_poly = np.polyfit(freq_ends, mag_ends, 2)
        phase_poly = np.polyfit(freq_ends, phase_ends, 1)
        # guess f0
        f_index_min = np.argmin(magnitude - np.polyval(gain_poly, xm(f)))
        f0_guess = f[f_index_min]
        # set some bounds (resonant frequency should not be within 5% of file end)
        f_min = min(f[f_index_5pc],  f[f_index_end - f_index_5pc])
        f_max = max(f[f_index_5pc],  f[f_index_end - f_index_5pc])
        if not f_min < f0_guess < f_max: f0_guess = fm
        # guess Q values
        mag_max = np.polyval(gain_poly, xm(f[f_index_min]))
        mag_min = magnitude[f_index_min]
        fwhm = np.sqrt((mag_max**2 + mag_min**2) / 2.)  # fwhm is for power not amplitude
        fwhm_mask = magnitude < fwhm
        bandwidth = np.abs(f[fwhm_mask][-1] - f[fwhm_mask][0])
        # Q0 = f0 / fwhm bandwidth
        q0_guess = f0_guess / bandwidth if bandwidth != 0 else 1e4
        # Q0 / Qi = min(mag) / max(mag)
        qi_guess = q0_guess * mag_max / mag_min if mag_min != 0 else 1e5
        if qi_guess == 0: qi_guess = 1e5
        if q0_guess == 0: q0_guess = 1e4
        # 1 / Q0 = 1 / Qc + 1 / Qi
        qc_guess = 1. / (1. / q0_guess - 1. / qi_guess)
        if qc_guess == 0: qc_guess = 1e4

        params = {'qi': qi_guess, 'qc': qc_guess, 'f0': f0_guess, 'xa': 0.0, 'a': 0 if not nonlinear else 0.0025,
                  'gain0': gain_poly[2], 'gain1': gain_poly[1], 'gain2': gain_poly[0], 'phase0': phase_poly[1],
                  'phase1': phase_poly[0], 'i_offset': i_offset, 'q_offset': q_offset, 'alpha': alpha, 'gamma': gamma,
                  'fm': fm}
        # input kwargs take priority if they are parameters
        params.update({key: value for key, value in kwargs.items() if key in params.keys()})
        # coerce to float type
        params = {key: float(value) for key, value in params.items()}
        return params


@cython.boundscheck(False)
@cython.wraparound(False)
def fit(np.ndarray[DTYPE_float64_t, ndim=1] f,
        np.ndarray[DTYPE_float64_t, ndim=1] i,
        np.ndarray[DTYPE_float64_t, ndim=1] q, *,
        double fm=DEFAULT_FM,
        bool_t decreasing=DEFAULT_DECREASING,
        bool_t baseline=True,
        bool_t nonlinear=False,
        bool_t imbalance=False,
        bool_t offset=False,
        bool_t numerical=False,
        double qi=DEFAULT_QI,
        double qc=DEFAULT_QC,
        double f0=DEFAULT_F0,
        double xa=DEFAULT_XA,
        double a=DEFAULT_A,
        double gain0=DEFAULT_GAIN0,
        double gain1=DEFAULT_GAIN1,
        double gain2=DEFAULT_GAIN2,
        double phase0=DEFAULT_PHASE0,
        double phase1=DEFAULT_PHASE1,
        double alpha=DEFAULT_ALPHA,
        double gamma=DEFAULT_GAMMA,
        double i_offset=DEFAULT_I_OFFSET,
        double q_offset=DEFAULT_Q_OFFSET,
        **kwargs):
    # check that all of the arrays are the same size
    if f.shape[0] != i.shape[0] or f.shape[0] != q.shape[0]:
        raise ValueError("All input arrays must have the same size.")

    # fm is not a fit parameter. It sets the frequency normalization for the gain and phase background.
    if fm < 0: fm = np.median(f)
    if f0 < 0: f0 = fm

    # create memoryviews of the numpy arrays to send to the C++ fitting code
    if not f.flags['C_CONTIGUOUS']:
        f = np.ascontiguousarray(f)
    if not i.flags['C_CONTIGUOUS']:
        i = np.ascontiguousarray(i)
    if not q.flags['C_CONTIGUOUS']:
        q = np.ascontiguousarray(q)
    cdef double[::1] f_view = f
    cdef double[::1] i_view = i
    cdef double[::1] q_view = q

    # create the parameter blocks
    cdef double pr[4], pd[1], pb[5], pi[2], po[2]
    create_parameter_blocks(&pr[0], &pd[0], &pb[0], &pi[0], &po[0], qi, qc, f0, xa, a, gain0, gain1, gain2, phase0,
                            phase1, alpha, gamma, i_offset, q_offset)

    # run the fitting code
    cdef string out
    summary = fit_c(&f_view[0], &i_view[0], &q_view[0], f_view.shape[0], fm, decreasing, baseline, nonlinear, imbalance,
                    offset, numerical, &pr[0], &pd[0], &pb[0], &pi[0], &po[0]).decode("utf-8").strip()
    log.info(summary)

    # return the fitted parameter values
    params = {'fm': fm, 'decreasing': decreasing, 'baseline': baseline, 'nonlinear': nonlinear,
              'imbalance': imbalance, 'offset': offset}  # independent
    params.update({"summary": summary})  # metrics
    params.update({'qi': pr[0], 'qc': pr[1], 'f0': pr[2], 'xa': pr[3]})  # resonance
    params.update({'a': pow(pd[0], 2)})  # detuning
    params.update({'gain0': pb[0], 'gain1': pb[1], 'gain2': pb[2], 'phase0': pb[3], 'phase1': pb[4]})  # baseline
    params.update({'alpha': pi[0], 'gamma': pi[1]})  # imbalance
    params.update({'i_offset': po[0], 'q_offset': po[1]})  # offset
    return params
