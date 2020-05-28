cimport cython
import logging
import numpy as np
cimport numpy as np
from libc.math cimport pow
from libcpp.string cimport string
from libcpp cimport bool as bool_t
from scipy.ndimage import label, find_objects

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
    The transmission baseline and the IQ mixer offset, phase imbalance, and
    amplitude imbalance can be removed from data using this function.

    This function solves the equation,
        z = mixer(baseline(f) * resonance(f)),
    for resonance(f), where mixer(), baseline(), and resonance() have their
    parameters suppressed. See their docstrings for more details.

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
            baseline. See the baseline() docstring for more details.
        gain0: float (optional)
            The zeroth order gain coefficient. See the baseline() docstring for
            more details.
        gain1: float (optional)
            The first order gain coefficient. See the baseline() docstring for
            more details.
        gain2: float (optional)
            The second order gain coefficient. See the baseline() docstring for
            more details.
        phase0: float (optional)
            The zeroth order phase coefficient. See the baseline() docstring
            for more details.
        phase1: float (optional)
            The first order phase coefficient. See the baseline() docstring for
            more details.
        alpha: float (optional)
            The mixer amplitude imbalance. See the mixer() docstring for more
            details.
        beta: float (optional)
            The mixer phase imbalance. See the mixer() docstring for more
            details.
        gamma: float (optional)
            The mixer in-phase component offset. See the mixer() docstring for
            more details.
        delta: float (optional)
            The mixer quadrature component offset. See the mixer() docstring
            for more details.
        optional keyword arguments:
            All other keyword arguments are ignored allowing the output of
            fit() to be supplied as a double starred argument.

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
    This function models the baseline transmission approximated by a quadratic
    gain polynomial and a linear phase polynomial:
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
    where x is given by the detuning() function. See its docstring for more
    details.

    The asymmetric hanger equation has been derived by many people with
    slightly different parametrizations for the resonance asymmetry. For
    example, see
        M. S. Khalil et al. (2012) (doi.org/10.1063/1.3692073)
        A. Megrant et al. (2012) (doi.org/10.1063/1.3693409).

    The specific functional form used here ensures that the resonance asymmetry
    parameter, xa, is independent from the coupling, qc, and internal, qi,
    quality factors. It is taken from
        K. Geerlings et al. (2012) (doi.org/10.1063/1.4710520).

    Args:
        f: numpy.ndarray, numpy.float64 or numpy.float32
            The frequency or frequencies at which to evaluate the function.
            Data not in a numpy array will be coerced into that format.
        decreasing: boolean
            A parameter determining the direction of the frequency sweep. See
            the detuning() docstring for more details.
        qi: float
            The internal quality factor of the resonator.
        qc: float
            The coupling quality factor of the resonator.
        f0: float
            The low power resonance frequency. See the detuning() docstring for
            more details.
        xa: float
            The fractional resonance asymmetry.
        a: float
            The resonance inductive nonlinearity. See the detuning() docstring
            for more details.
        optional keyword arguments:
            All other keyword arguments are ignored allowing the output of
            fit() to be supplied as a double starred argument.

    Returns:
        z: numpy.ndarray, numpy.complex128
            The complex resonance transmission.

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
        z = resonance_vectorized(f_ravel64, decreasing, &pr[0], &pd[0])
    elif f.dtype == np.float32:
        f_ravel32 = f.ravel()
        z = resonance_vectorized(f_ravel32, decreasing, &pr[0], &pd[0])
    else:
        raise ValueError(f"Invalid data type for f: {f.dtype}. Only float32 and float64 are supported.")
    return z.reshape(f.shape)


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
    """
    For low probe powers the detuning is defined as follows:
        x = (f - f0) / f0

    However, nonlinear effects can cause feedback between the resonance
    frequency and the probe tone at high powers. A particular case of this
    nonlinearity, where the kinetic inductance depends quadratically on the
    current, is treated here. For the derivation of this effect see
        L. J. Swenson et al. (2013) (doi.org/10.1063/1.4794808)

    As defined in the paper, the parameter, a, quantifies the amount of
    nonlinearity present in the data with 0 being none and 4 * sqrt(3) / 9
    being the value at which the resonance bifurcates.

    In the bifurcation regime, the detuning is multivalued at some frequencies.
    The value chosen is determined by the parameter, decreasing, which
    corresponds to the frequency sweep direction.

    Args:
        f: numpy.ndarray, numpy.float64 or numpy.float32
            The frequency or frequencies at which to evaluate the function.
            Data not in a numpy array will be coerced into that format.
        decreasing: boolean
            A parameter determining the direction of the frequency sweep.
        qi: float
            The internal quality factor of the resonator. See the resonance()
            docstring for more details.
        qc: float
            The coupling quality factor of the resonator. See the resonance()
            docstring for more details.
        f0: float
            The low power resonance frequency. See the resonance() docstring
            for more details.
        xa: float
            The fractional resonance asymmetry. See the resonance() docstring
            for more details.
        a: float
            The resonance inductive nonlinearity.
        optional keyword arguments:
            All other keyword arguments are ignored allowing the output of
            fit() to be supplied as a double starred argument.

    Returns:
        x: numpy.ndarray, numpy.float64
            The resonance detuning.

    """
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
        x = detuning_vectorized(f_ravel64, decreasing, &pr[0], &pd[0])
    elif f.dtype == np.float32:
        f_ravel32 = f.ravel()
        x = detuning_vectorized(f_ravel32, decreasing, &pr[0], &pd[0])
    else:
        raise ValueError(f"Invalid data type for f: {f.dtype}. Only float32 and float64 are supported.")
    return x.reshape(f.shape)


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
    """
    This function represents the effect of an uncalibrated IQ mixer on
    transmission data. The functional form for this distortion is given by the
    following:
        i -> i + gamma
        q -> alpha * cos(beta) * q + alpha * sin(beta) * i + delta
    i and q are the in-phase and quadrature components of the mixer. The
    amplitude imbalance, alpha, and the phase imbalance, gamma, are referenced
    to the  in-phase signal. gamma and delta represent DC offsets in the
    in-phase and quadrature signals respectively.

    For measurements taken by a calibrated system, like a vector network
    analyzer, this function can be ignored by setting the parameters to the
    following values:
        alpha = 1
        beta = 0
        gamma = 0
        delta = 0

    Some IQ mixers use a convention where I and Q are swapped, resulting in the
    resonance being traced backwards with frequency in the IQ plane. In order
    to fit this case without modifying the definitions of i and q, beta should
    be set equal to pi.

    Args:
        i: numpy.ndarray, float (optional)
            The true, undistorted, in-phase component of the signal. q must be
            supplied if i is supplied. i cannot be used in combination with the
            z keyword argument.
        q: numpy.ndarray, float (optional)
            The true, undistorted, quadrature component of the signal. i must
            be supplied if q is supplied. q cannot be used in combination with
            the z keyword argument.
        z: numpy.ndarray, complex (optional)
            If neither i or q are supplied, then this keyword argument must be.
            z represents the complex signal (i + 1j * q).
        alpha: float (optional)
            The mixer amplitude imbalance.
        beta: float (optional)
            The mixer phase imbalance.
        gamma: float (optional)
            The mixer in-phase component offset.
        delta: float (optional)
            The mixer quadrature component offset.
        optional keyword arguments:
            All other keyword arguments are ignored allowing the output of
            fit() to be supplied as a double starred argument.

    Returns:
        i: numpy.ndarray, numpy.float64
            The distorted in-phase mixer component. Only returned if i and q
            are specified.
        q: numpy.ndarray, numpy.float64
            The distorted quadrature mixer component. Only returned if i and q
            are specified.
        z: numpy.ndarray, numpy.complex128
            The distorted complex mixer data. Only returned if z is specified.

    """
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
    """
    This is the model used in fit(). It has the following functional form:
        z = mixer(baseline(f) * resonance(f)),
    where the parameters in mixer(), baseline(), and resonance() have been
    suppressed.

    Args:
        f: numpy.ndarray, numpy.float64 or numpy.float32
            The frequency or frequencies at which to evaluate the function.
            Data not in a numpy array will be coerced into that format.
        fm:  float (optional)
            The reference frequency for the gain and phase parameters of the
            baseline. See the baseline() docstring for more details.
        decreasing: boolean
            A parameter determining the direction of the frequency sweep. See
            the detuning() docstring for more details.
        qi: float
            The internal quality factor of the resonator. See the resonance()
            docstring for more details.
        qc: float
            The coupling quality factor of the resonator. See the resonance()
            docstring for more details.
        f0: float
            The low power resonance frequency. See the resonance() docstring
            for more details.
        xa: float
            The fractional resonance asymmetry. See the resonance() docstring
            for more details.
        a: float
            The resonance inductive nonlinearity. See the detuning() docstring
            for more details.
        gain0: float (optional)
            The zeroth order gain coefficient. See the baseline() docstring for
            more details.
        gain1: float (optional)
            The first order gain coefficient. See the baseline() docstring for
            more details.
        gain2: float (optional)
            The second order gain coefficient. See the baseline() docstring for
            more details.
        phase0: float (optional)
            The zeroth order phase coefficient. See the baseline() docstring
            for more details.
        phase1: float (optional)
            The first order phase coefficient. See the baseline() docstring for
            more details.
        alpha: float (optional)
            The mixer amplitude imbalance. See the mixer() docstring for more
            details.
        beta: float (optional)
            The mixer phase imbalance. See the mixer() docstring for more
            details.
        gamma: float (optional)
            The mixer in-phase component offset. See the mixer() docstring for
            more details.
        delta: float (optional)
            The mixer quadrature component offset. See the mixer() docstring
            for more details.
        optional keyword arguments:
            All other keyword arguments are ignored allowing the output of
            fit() to be supplied as a double starred argument.

    Returns:
        z: numpy.ndarray, numpy.complex128
            The complex model of the data.

    """
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
        z = model_vectorized(f_ravel64, fm, decreasing, &pr[0], &pd[0], &pb[0], &pi[0], &po[0])
    elif f.dtype == np.float32:
        f_ravel32 = f.ravel()
        z = model_vectorized(f_ravel32, fm, decreasing, &pr[0], &pd[0], &pb[0], &pi[0], &po[0])
    else:
        raise ValueError(f"Invalid data type for f: {f.dtype}. Only float32 and float64 are supported.")
    return z.reshape(f.shape)


def guess(f, i=None, q=None, *,
          z=None,
          nonlinear=False,
          imbalance=None,
          offset=None,
          **kwargs):
    """
    Determine a suitable guess for the starting point of the nonlinear
    optimization from the data.

    Args:
        f: numpy.ndarray, float
            The frequency or frequencies corresponding to the mixer data. Data
            not in a numpy array will be coerced into that format.
        i: numpy.ndarray, float (optional)
            The in-phase component of the data to be fit. q must be supplied
            if i is supplied. i cannot be used in combination with the z
            keyword argument.
        q: numpy.ndarray, float (optional)
            The quadrature component of the data to be fit. i must be supplied
            if q is supplied. q cannot be used in combination with the z
            keyword argument.
        z: numpy.ndarray, complex (optional)
            If neither i or q are supplied, then this keyword argument must be.
            z represents the complex data (i + 1j * q).
        nonlinear: boolean
            Sets the nonlinear detuning parameter, a, to a value slightly above
            zero for the fit. This is important because the Jacobian of the
            model with respect to a is zero when a=0, so it won't be fit
            properly if initialized to this value. If False, a is set to zero.
        imbalance: numpy.ndarray, complex, shape=(L, M) or (M,)
            A calibration data set for the mixer imbalance:
                A frequency f is put into the mixer's LO port, and a frequency
                f + ∆f is put into the mixer's RF port. The output is then low
                pass filtered with a cutoff below f. f is near the frequency of
                the data set and ∆f is within the measurement bandwidth.

                The complex mixer output (i + 1j * q) then oscillates at a
                frequency ∆f. M samples of the output covering many periods
                can be used to estimate alpha and beta. L independent
                measurements can be averaged together.

            If not provided, no mixer distortion is assumed.
        offset: numpy.ndarray, complex, shape=(N,)
            A calibration data set for the mixer offset:
                A measurement of the complex mixer output (i + 1j * q) when the
                signal going to the resonator is removed and the signal at the
                LO port remains the same. N independent measurements can be
                added together.
        optional keyword arguments:
            Any of the fit parameters can be specified and its value will be
            used in the guess. 'fm' may also be supplied to fix the reference
            frequency for the baseline parameters.

    Returns:
        params: dictionary
            A dictionary of parameter values to use in the fit. 'fm' is also
            included so that baseline parameters are properly referenced in the
            fit.

    """
    # estimate mixer correction from calibration data
    alpha, beta, gamma, delta = compute_mixer_calibration(offset, imbalance, **kwargs)
    # parse input data
    if (i is not None) and (q is not None):
        i = np.asarray(i)
        q = np.asarray(q)
        if i.shape != q.shape or np.iscomplex(i).any() or np.iscomplex(q).any():
            raise ValueError("i and q must have the same shape and be real.")
    elif z is not None:
        z = np.asarray(z)
        i = z.real
        q = z.imag
    else:
        raise ValueError("Neither i and q or z were supplied as keyword arguments.")
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
    regions, _ = label(fwhm_mask)  # find the regions where magnitude < fwhm
    region = regions[f_index_min]  # pick the one that includes the minimum
    f_masked = f[find_objects(regions, max_label=region)[-1]]  # mask f to only include that region
    bandwidth = f_masked.max() - f_masked.min()  # find the bandwidth
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
        int max_iterations=2000,
        int threads=1,
        sigma=None,
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
    """
    Fit resonance data to the model() function.

    Args:
        f: numpy.ndarray, numpy.float64 or numpy.float32
            The frequency or frequencies corresponding to the mixer data. Data
            not in a numpy array will be coerced into that format.
        i: numpy.ndarray, float
            The in-phase component of the mixer's output to be calibrated. q
            must be supplied if i is supplied.
        q: numpy.ndarray, float
            The quadrature component of the mixer's output to be calibrated. i
            must be supplied if q is supplied.
        fm:  float (optional)
            The reference frequency for the gain and phase parameters of the
            baseline. See the baseline() docstring for more details.
        decreasing: boolean
            A parameter determining the direction of the frequency sweep. See
            the detuning() docstring for more details.
        baseline: boolean
            If False, the baseline parameters gain0, gain1, gain2, phase0, and
            phase1 are held constant during the fit. If True, they are allowed
            to vary.
        nonlinear: boolean
            If True, the nonlinear parameter, a, is allowed to vary during the
            fit. If False, it is held constant.
        imbalance: boolean
            If True, the imbalance parameters, alpha and beta, are allowed to
            vary during the fit. If False, they are held constant.
        offset: boolean
            If True, the offset parameters, gamma and delta, are allowed to
            vary during the fit. If False, they are held constant.
        numerical: boolean
            If True, the numerical Jacobian is used during the fit. If False
            the analytic Jacobian is used. Note: the numerical Jacobian will
            almost always lead to worse results. It is included here only for
            comparison to the analytic case and to other methods that use the
            numerical version.
        max_iterations: integer
            The maximum number of iterations to allow in the fit before
            stopping.
        threads: integer
            The number of threads to use in the fit. This parameter may be
            ignored if ceres-solver was not compiled with threading enabled.
        sigma: numpy.ndarray, complex
            This argument represents the standard deviation of the data and can
            be a single value or an array of the same length as the data. The
            real part corresponds to i and the imaginary part to q. If
            supplied, the chi_squared, aic, and bic statistics will be computed
            and added to the result at the end of the fit.
        qi: float
            The starting value for the internal quality factor of the
            resonator. See the resonance() docstring for more details.
        qc: float
            The starting value for the coupling quality factor of the
            resonator. See the resonance() docstring for more details.
        f0: float
            The starting value for the low power resonance frequency. See the
            resonance() docstring for more details.
        xa: float
            The starting value for the fractional resonance asymmetry. See the
            resonance() docstring for more details.
        a: float
            The starting value for the resonance inductive nonlinearity. See
            the detuning() docstring for more details.
        gain0: float (optional)
            The starting value for the zeroth order gain coefficient. See the
            baseline() docstring for more details.
        gain1: float (optional)
            The starting value for the first order gain coefficient. See the
            baseline() docstring for more details.
        gain2: float (optional)
            The starting value for the second order gain coefficient. See the
            baseline() docstring for more details.
        phase0: float (optional)
            The starting value for the zeroth order phase coefficient. See the
            baseline() docstring for more details.
        phase1: float (optional)
            The starting value for the first order phase coefficient. See the
            baseline() docstring for more details.
        alpha: float (optional)
            The starting value for the mixer amplitude imbalance. See the
            mixer() docstring for more details.
        beta: float (optional)
            The starting value for the mixer phase imbalance. See the mixer()
            docstring for more details.
        gamma: float (optional)
            The starting value for the mixer in-phase component offset. See the
            mixer() docstring for more details.
        delta: float (optional)
            The starting value for the mixer quadrature component offset. See
            the mixer() docstring for more details.
        optional keyword arguments:
            All other keyword arguments are ignored allowing the output of
            this function to be supplied as a guess to another fit as a double
            starred argument.

    Returns:
        result: dictionary
            A dictionary of the fit parameter values. The fit options fm,
            decreasing, baseline, nonlinear, imbalance, offset, and
            max_iterations are included in the dictionary along with
                threads: integer
                    The actual number of threads used in the fit. This value
                    may not be the same as used in the function call depending
                    on if ceres-solver was installed with threading enabled.
                summary: string
                    A fit report generated by ceres-solver.
                varied: integer
                    The number of parameters varied in the fit.
                success: boolean
                    If the fit converged without any errors or reaching the
                    max_iterations, True is returned. Otherwise, False is
                    returned.
                chi_squared: float
                    The chi squared metric for the fit. It is only included in
                    the dictionary if sigma was supplied in the function call.
                aic: float
                    The Akaike information criterion for the fit.  It is only
                    included in the dictionary if sigma was supplied in the
                    function call.
                bic: float
                    The Bayesian information criterion  for the fit.  It is
                    only included in the dictionary if sigma was supplied in
                    the function call.

    """
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
    cdef int varied
    cdef bool_t success
    summary = fit_c(&f_view[0], &i_view[0], &q_view[0], f_view.shape[0], fm, decreasing, baseline, nonlinear, imbalance,
                    offset, numerical, max_iterations, threads, varied, success, &pr[0], &pd[0], &pb[0], &pi[0],
                    &po[0]).decode("utf-8").strip()
    log.debug(summary)

    # return the fitted parameter values
    result = {'fm': fm, 'decreasing': decreasing, 'baseline': baseline, 'nonlinear': nonlinear, 'imbalance': imbalance,
              'offset': offset, 'max_iterations': max_iterations, 'threads': threads}  # independent
    result.update({"summary": summary, "varied": varied, "success": success})  # metrics
    result.update({'qi': pr[0], 'qc': pr[1], 'f0': pr[2], 'xa': pr[3]})  # resonance
    result.update({'a': pow(pd[0], 2)})  # detuning
    result.update({'gain0': pb[0], 'gain1': pb[1], 'gain2': pb[2], 'phase0': pb[3], 'phase1': pb[4]})  # baseline
    result.update({'alpha': pi[0], 'beta': pi[1]})  # imbalance
    result.update({'gamma': po[0], 'delta': po[1]})  # offset
    # compute statistics
    cdef np.ndarray[complex128_t, ndim=1] m
    if sigma is not None:
        m = model(f, **result)
        sigma = np.broadcast_to(sigma, f.shape[0])
        chi_squared = (((m.real - i) / sigma.real)**2).sum() + (((m.imag - q) / sigma.imag)**2).sum()
        # -2 * Log(likelihood)
        scaled_likelihood = chi_squared + np.log(2 * np.pi * sigma.real).sum() + np.log(2 * np.pi * sigma.imag).sum()
        aic = 2 * result['varied'] + scaled_likelihood
        bic = result['varied'] * np.log(2 * f.shape[0]) + scaled_likelihood
        result.update({'chi_squared': chi_squared, 'aic': aic, 'bic': bic})
    return result
