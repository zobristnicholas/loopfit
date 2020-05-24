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
                         calibrate as calibrate_c, detuning as detuning_c, mixer as mixer_c)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


@cython.boundscheck(False)
@cython.wraparound(False)
cdef calibrate_vectorized(np.ndarray[float_t, ndim=1] f, np.ndarray[float64_t, ndim=1] i,
                          np.ndarray[float64_t, ndim=1] q, double fm, double pb[], double pi[], double po[]):
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
              double beta=DEFAULT_BETA,
              double gamma=DEFAULT_GAMMA,
              double delta=DEFAULT_DELTA,
              **kwargs):
    """
    The transmission baseline and the offset, phase imbalance, and amplitude
    imbalance of an IQ mixer can be removed from data using this function.

    This function inverts the following equation:
        z = mixer(baseline(f) * z)
    where the mixer and baseline functions have their parameters suppressed.
    See their docstrings for more details.

    Args:
        f: numpy.ndarray, numpy.float64 or numpy.float32
            The frequency or frequencies corresponding to the mixer data. Data
            not in a numpy array will be coerced into that format.
        i: numpy.ndarray, float (optional)
            The in-phase component of the mixer's output to be calibrated. q
            must be supplied if i is supplied. i cannot be used in combination
            with the z keyword argument.
        q: numpy.ndarray, float (optional)
            The quadrature component of the mixer's output to be calibrated. i
            must be supplied if q is supplied. q cannot be used in combination
            with the z keyword argument.
        z: numpy.ndarray, complex (optional)
            If neither i or q are supplied, then this keyword argument must be.
            z represents the complex mixer output (i + 1j * q).
        fm:  float (optional)
            The reference frequency for the gain and phase parameters of the
            baseline. See the baseline docstring for more details.
        gain0: float (optional)
            The zeroth order gain coefficient. See the baseline docstring for
            more details.
        gain1: float (optional)
            The first order gain coefficient. See the baseline docstring for
            more details.
        gain2: float (optional)
            The second order gain coefficient. See the baseline docstring for
            more details.
        phase0: float (optional)
            The zeroth order phase coefficient. See the baseline docstring for
            more details.
        phase1: float (optional)
            The first order phase coefficient. See the baseline docstring for
            more details.
        alpha: float (optional)
            The mixer amplitude imbalance. See the mixer docstring for more
            details.
        beta: float (optional)
            The mixer phase imbalance. See the mixer docstring for more
            details.
        gamma: float (optional)
            The mixer in-phase component offset. See the mixer docstring for
            more details.
        delta: float (optional)
            The mixer quadrature component offset. See the mixer docstring for
            more details.
        optional keyword arguments:
            All other keyword arguments are ignored allowing the output of fit
            to be supplied as a double starred argument.

    Returns:
        i: numpy.ndarray, numpy.float64
            The calibrated in-phase mixer component. Only returned if i and q
            are specified.
        q: numpy.ndarray, numpy.float64
            The calibrated quadrature mixer component. Only returned if i and q
            are specified.
        z: numpy.ndarray, numpy.complex128
            The calibrated complex mixer data. Only returned if z is specified.

    """
    # check inputs
    if fm < 0: fm = np.median(f)  # default depends on f
    f = np.asarray(f)  # no copy if already an array
    # create the parameter blocks
    cdef double pb[5], pi[2], po[2]
    create_baseline_block(pb, gain0, gain1, gain2, phase0, phase1)
    create_imbalance_block(pi, alpha, beta)
    create_offset_block(po, gamma, delta)
    # initialize output i & q arrays (copy since calibrate_c is in-place)
    if (i is not None) and (q is not None):
        i = np.array(i, dtype=np.float64, copy=True)  # copy since calibrate_c is in-place
        q = np.array(q, dtype=np.float64, copy=True)
        z_output = False
    elif z is not None:
        i = np.array(z.real, dtype=np.float64, copy=True)  # copy since calibrate_c is in-place
        q = np.array(z.imag, dtype=np.float64, copy=True)
        z_output = True
    else:
        raise ValueError("Neither i and q or z were supplied as keyword arguments.")
    # define types
    cdef np.ndarray[float64_t, ndim=1] i_ravel = i.ravel()
    cdef np.ndarray[float64_t, ndim=1] q_ravel = q.ravel()
    cdef np.ndarray[float32_t, ndim=1] f_ravel32
    cdef np.ndarray[float64_t, ndim=1] f_ravel64
    if f.dtype == np.float64:
        f_ravel64 = f.ravel()
    elif f.dtype == np.float32:
        f_ravel32 = f.ravel()
    else:
        raise ValueError(f"Invalid data type for f: {f.dtype}. Only float32 and float64 are supported.")
    # calibrate the data
    if f.dtype == np.float64:
        calibrate_vectorized(f_ravel64, i_ravel, q_ravel, fm, pb, pi, po)
    elif f.dtype == np.float32:
        calibrate_vectorized(f_ravel32, i_ravel, q_ravel, fm, pb, pi, po)
    if z_output:
        return i + 1j * q
    else:
        return i, q


@cython.boundscheck(False)
@cython.wraparound(False)
cdef baseline_vectorized(np.ndarray[float_t, ndim=1] f, double fm, double pb[]):
    cdef np.ndarray[complex128_t, ndim=1] result = np.empty(f.shape[0], dtype=np.complex128)
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
    """
    This function models the baseline transmission approximated by the
    quadratic gain and linear phase frequency polynomials:
        (gain0 + gain1 * xm + gain2 * xm**2) * exp(1j * (phase0 + phase1 * xm))
    where xm = (f - fm) / fm.

    Args:
        f: numpy.ndarray, numpy.float64 or numpy.float32
            The frequency or frequencies at which to evaluate the function.
            Data not in a numpy array will be coerced into that format.
        fm:  float (optional)
            The reference frequency for the gain and phase parameters of the
            baseline.
        gain0: float (optional)
            The zeroth order gain coefficient.
        gain1: float (optional)
            The first order gain coefficient.
        gain2: float (optional)
            The second order gain coefficient.
        phase0: float (optional)
            The zeroth order phase coefficient.
        phase1: float (optional)
            The first order phase coefficient.
        optional keyword arguments:
            All other keyword arguments are ignored allowing the output of fit
            to be supplied as a double starred argument.

    Returns:
        z: numpy.ndarray, numpy.complex128
            The complex baseline transmission.

    """
    if fm < 0: fm = np.median(f)  # default depends on f
    f = np.asarray(f)  # no copy if already an array
    # create the parameter blocks
    cdef double pb[5]
    create_baseline_block(&pb[0], gain0, gain1, gain2, phase0, phase1)
    # call function
    cdef np.ndarray[float32_t, ndim=1] f_ravel32
    cdef np.ndarray[float64_t, ndim=1] f_ravel64
    if f.dtype == np.float64:
        f_ravel64 = f.ravel()
        z = baseline_vectorized(f_ravel64, fm, &pb[0])
    elif f.dtype == np.float32:
        f_ravel32 = f.ravel()
        z = baseline_vectorized(f_ravel32, fm, &pb[0])
    else:
        raise ValueError(f"Invalid data type for f: {f.dtype}. Only float32 and float64 are supported.")
    return z.reshape(f.shape)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef resonance_vectorized(np.ndarray[float_t, ndim=1] f, bool_t decreasing, double pr[], double pd[]):
    cdef np.ndarray[complex128_t, ndim=1] result = np.empty(f.shape[0], dtype=np.complex128)
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
    """
    This function models the resonator transmission given by the asymmetric
    hanger equation:
        (qc + 2j * qi * qc * (x + xa)) / (qi + qc + 2j * qi * qc * x)
    where x is given by the detuning function. See its docstring for more
    details.

    Args:
        f: numpy.ndarray, numpy.float64 or numpy.float32
            The frequency or frequencies at which to evaluate the function.
            Data not in a numpy array will be coerced into that format.
        decreasing: bool
        qi: float
        qc: float
        f0: float
        xa: float
        a: float
        optional keyword arguments:
            All other keyword arguments are ignored allowing the output of fit
            to be supplied as a double starred argument.

    Returns:

    """
    f = np.asarray(f)  # no copy if already an array
    # create the parameter blocks
    cdef double pr[4], pd[1]
    create_resonance_block(&pr[0], qi, qc, f0, xa)
    create_detuning_block(&pd[0], a)
    # call function
    cdef np.ndarray[float32_t, ndim=1] f_ravel32
    cdef np.ndarray[float64_t, ndim=1] f_ravel64
    if f.dtype == np.float64:
        f_ravel64 = f.ravel()
        result = resonance_vectorized(f_ravel64, decreasing, &pr[0], &pd[0])
    elif f.dtype == np.float32:
        f_ravel32 = f.ravel()
        result = resonance_vectorized(f_ravel32, decreasing, &pr[0], &pd[0])
    else:
        raise ValueError(f"Invalid data type for f: {f.dtype}. Only float32 and float64 are supported.")
    return result.reshape(f.shape)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef detuning_vectorized(np.ndarray[float_t, ndim=1] f, bool_t decreasing, double pr[], double pd[]):
    cdef np.ndarray[float64_t, ndim=1] result = np.empty(f.shape[0], dtype=np.float64)
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
    if f0 < 0: f0 = kwargs.get('fm', np.median(f))
    f = np.asarray(f)  # no copy if already an array
    # create the parameter blocks
    cdef double pr[4], pd[1]
    create_resonance_block(&pr[0], qi, qc, f0, xa)
    create_detuning_block(&pd[0], a)
    # call function
    cdef np.ndarray[float32_t, ndim=1] f_ravel32
    cdef np.ndarray[float64_t, ndim=1] f_ravel64
    if f.dtype == np.float64:
        f_ravel64 = f.ravel()
        result = detuning_vectorized(f_ravel64, decreasing, &pr[0], &pd[0])
    elif f.dtype == np.float32:
        f_ravel32 = f.ravel()
        result = detuning_vectorized(f_ravel32, decreasing, &pr[0], &pd[0])
    else:
        raise ValueError(f"Invalid data type for f: {f.dtype}. Only float32 and float64 are supported.")
    return result.reshape(f.shape)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef mixer_vectorized(np.ndarray[complex128_t, ndim=1] z, double pi[], double po[]):
    cdef np.ndarray[complex128_t, ndim=1] result = np.empty(z.shape[0], dtype=np.complex128)
    for ii in range(z.shape[0]):
        result[ii] = mixer_c(z[ii], pi, po)
    return result


def mixer(i=None, q=None, *,
          z=None,
          double alpha=DEFAULT_ALPHA,
          double beta=DEFAULT_BETA,
          double gamma=DEFAULT_GAMMA,
          double delta=DEFAULT_DELTA,
          **kwargs):
    # create the parameter blocks
    cdef double pi[2], po[2]
    create_imbalance_block(pi, alpha, beta)
    create_offset_block(po, gamma, delta)
    # initialize output i & q arrays
    if (i is not None) and (q is not None):
        i = np.asarray(i)
        q = np.asarray(q)
        if i.shape != q.shape or np.iscomplex(i).any() or np.iscomplex(q).any():
            raise ValueError("i and q must have the same shape and be real.")
        z = i + 1j * q  # automatically creates np.complex128 type
        z_output = False
    elif z is not None:
        z = np.asarray(z, dtype=np.complex128)
        z_output = True
    else:
        raise ValueError("Neither i and q or z were supplied as keyword arguments.")
    # call function
    cdef np.ndarray[complex128_t, ndim=1] z_ravel = z.ravel()
    result = mixer_vectorized(z_ravel, pi, po)
    result = result.reshape(z.shape)
    if z_output:
        return result
    else:
        return result.real, result.imag


@cython.boundscheck(False)
@cython.wraparound(False)
cdef model_vectorized(np.ndarray[float_t, ndim=1] f, double fm, bool_t decreasing, double pr[], double pd[],
                      double pb[], double pi[], double po[]):
    cdef np.ndarray[complex128_t, ndim=1] result = np.empty(f.shape[0], dtype=np.complex128)
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
          double beta=DEFAULT_BETA,
          double gamma=DEFAULT_GAMMA,
          double delta=DEFAULT_DELTA,
          **kwargs):
    # check inputs
    if fm < 0: fm = np.median(f)  # default depends on f
    if f0 < 0: f0 = fm
    f = np.asarray(f, dtype=np.float64)  # no copy if already an array and dtype matches
    # create the parameter blocks
    cdef double pr[4], pd[1], pb[5], pi[2], po[2]
    create_parameter_blocks(&pr[0], &pd[0], &pb[0], &pi[0], &po[0], qi, qc, f0, xa, a, gain0, gain1, gain2, phase0,
                            phase1, alpha, beta, gamma, delta)
    # call function
    cdef np.ndarray[float32_t, ndim=1] f_ravel32
    cdef np.ndarray[float64_t, ndim=1] f_ravel64
    if f.dtype == np.float64:
        f_ravel64 = f.ravel()
        result = model_vectorized(f_ravel64, fm, decreasing, &pr[0], &pd[0], &pb[0], &pi[0], &po[0])
    elif f.dtype == np.float32:
        f_ravel32 = f.ravel()
        result = model_vectorized(f_ravel32, fm, decreasing, &pr[0], &pd[0], &pb[0], &pi[0], &po[0])
    else:
        raise ValueError(f"Invalid data type for f: {f.dtype}. Only float32 and float64 are supported.")
    return result.reshape(f.shape)


def guess(f, i, q, *, nonlinear=False, imbalance=None, offset=None, **kwargs):
        # estimate mixer correction from calibration data
        alpha, beta, gamma, delta = compute_mixer_calibration(offset, imbalance, **kwargs)
        # remove the IQ mixer offset and imbalance
        i, q = calibrate(f, i, q, alpha=alpha, beta=beta, gamma=gamma, delta=delta, gain0=1.0,
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
                  'phase1': phase_poly[0], 'gamma': gamma, 'delta': delta, 'alpha': alpha, 'beta': beta,
                  'fm': fm}
        # input kwargs take priority if they are parameters
        params.update({key: value for key, value in kwargs.items() if key in params.keys()})
        # coerce to float type
        params = {key: float(value) for key, value in params.items()}
        return params


@cython.boundscheck(False)
@cython.wraparound(False)
def fit(np.ndarray[float_t, ndim=1] f,
        np.ndarray[float2_t, ndim=1] i,
        np.ndarray[float2_t, ndim=1] q, *,
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
        double beta=DEFAULT_BETA,
        double gamma=DEFAULT_GAMMA,
        double delta=DEFAULT_DELTA,
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
    cdef float_t[::1] f_view = f
    cdef float2_t[::1] i_view = i
    cdef float2_t[::1] q_view = q

    # create the parameter blocks
    cdef double pr[4], pd[1], pb[5], pi[2], po[2]
    create_parameter_blocks(&pr[0], &pd[0], &pb[0], &pi[0], &po[0], qi, qc, f0, xa, a, gain0, gain1, gain2, phase0,
                            phase1, alpha, beta, gamma, delta)

    # run the fitting code
    cdef string out
    summary = fit_c(&f_view[0], &i_view[0], &q_view[0], f_view.shape[0], fm, decreasing, baseline, nonlinear, imbalance,
                    offset, numerical, &pr[0], &pd[0], &pb[0], &pi[0], &po[0]).decode("utf-8").strip()
    log.debug(summary)

    # return the fitted parameter values
    params = {'fm': fm, 'decreasing': decreasing, 'baseline': baseline, 'nonlinear': nonlinear,
              'imbalance': imbalance, 'offset': offset}  # independent
    params.update({"summary": summary})  # metrics
    params.update({'qi': pr[0], 'qc': pr[1], 'f0': pr[2], 'xa': pr[3]})  # resonance
    params.update({'a': pow(pd[0], 2)})  # detuning
    params.update({'gain0': pb[0], 'gain1': pb[1], 'gain2': pb[2], 'phase0': pb[3], 'phase1': pb[4]})  # baseline
    params.update({'alpha': pi[0], 'beta': pi[1]})  # imbalance
    params.update({'gamma': po[0], 'delta': po[1]})  # offset
    return params
