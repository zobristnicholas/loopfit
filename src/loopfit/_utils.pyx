import numpy as np
from ._utils cimport *

# fit parameter defaults
cdef double DEFAULT_QI = 1e5
cdef double DEFAULT_QC = 1e4
cdef double DEFAULT_F0 = -1.0  # negative means derive from f
cdef double DEFAULT_XA = 0.0
cdef double DEFAULT_A = 0.0
cdef double DEFAULT_GAIN0 = 1.0
cdef double DEFAULT_GAIN1 = 0.0
cdef double DEFAULT_GAIN2 = 0.0
cdef double DEFAULT_PHASE0 = 0.0
cdef double DEFAULT_PHASE1 = 0.0
cdef double DEFAULT_ALPHA = 1.0
cdef double DEFAULT_GAMMA = 0.0
cdef double DEFAULT_I_OFFSET = 0.0
cdef double DEFAULT_Q_OFFSET = 0.0

# independent variable defaults
cdef double DEFAULT_FM = -1.0  # negative means derive from f
cdef bool_t DEFAULT_DECREASING = False


def bandpass(data):
    fft_data = np.fft.rfft(data)
    fft_data[:, 0] = 0
    indices = np.array([np.arange(fft_data[0, :].size)] * fft_data[:, 0].size)
    f_data_ind = np.argmax(np.abs(fft_data), axis=-1)[:, np.newaxis]
    fft_data[np.logical_or(indices < f_data_ind - 1, indices > f_data_ind + 1)] = 0
    data_new = np.fft.irfft(fft_data, data[0, :].size)
    return data_new, f_data_ind


def compute_mixer_calibration(offset, imbalance, **kwargs):
    if offset is not None:
        z_offset = np.mean(offset)
        i_offset = z_offset.real if 'i_offset' not in kwargs.keys() else kwargs['i_offset']
        q_offset = z_offset.imag if 'q_offset' not in kwargs.keys() else kwargs['q_offset']
    else:
        i_offset = kwargs.get('i_offset', DEFAULT_I_OFFSET)
        q_offset = kwargs.get('q_offset', DEFAULT_Q_OFFSET)
    if imbalance is not None:
        imbalance = np.atleast_2d(imbalance)
        # bandpass filter the I and Q signals
        n = imbalance.shape[0]
        ip, f_i_ind = bandpass(imbalance.real)
        qp, f_q_ind = bandpass(imbalance.imag)
        # compute alpha and gamma
        amp = np.sqrt(2 * np.mean(qp**2, axis=-1))
        if 'alpha' not in kwargs.keys():
            alpha = np.sqrt(2 * np.mean(ip**2, axis=-1)) / amp
            alpha = np.mean(alpha)
        else:
            alpha = kwargs['alpha']
        if 'gamma' not in kwargs.keys():
            ratio = np.angle(np.fft.rfft(ip)[np.arange(n), f_i_ind[:, 0]] /
                             np.fft.rfft(qp)[np.arange(n), f_q_ind[:, 0]])  # for arcsine branch
            gamma = np.arcsin(np.sign(ratio) * 2 * np.mean(qp * ip, axis=-1) / (alpha * amp**2)) + np.pi * (ratio < 0)
            gamma = np.mean(gamma)
        else:
            gamma = kwargs['gamma']
    else:
        alpha = kwargs.get('alpha', DEFAULT_ALPHA)
        gamma = kwargs.get('gamma', DEFAULT_GAMMA)
    return alpha, gamma, i_offset, q_offset
