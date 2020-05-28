import pytest
import loopfit as lf
import numpy as np
from numpy.testing import assert_allclose

KEYS = ['qi', 'qc', 'xa', 'a', 'f0']


def test_fit(f, model, parameters, random):
    a = parameters.get('a', 0.0)
    alpha = parameters.get('alpha', 1.0)
    beta = parameters.get('beta', 0.0)
    gamma = parameters.get('gamma', 0.0)
    delta = parameters.get('delta', 0.0)
    z = model + random.normal(0, 0.02, model.shape) + 1j * random.normal(0, 0.02, model.shape)
    # guess by providing the offset if supplied and picking the branch of beta.
    init_params = lf.guess(f, z=z, beta=0 if np.abs(beta) < np.pi / 2 else np.pi, gamma=gamma, delta=delta,
                           nonlinear=True if a else False)
    result = lf.fit(f, z.real, z.imag, nonlinear=True if a else False,
                    imbalance=True if np.any([alpha != 1.0, beta]) else False, **init_params)
    assert result['success'], "The fit did not succeed."
    # check that fit is correct to ~10% for the parameters we care about
    assert_allclose([parameters[key] for key in KEYS], [result[key] for key in KEYS], rtol=1e-1, atol=1e-6)
