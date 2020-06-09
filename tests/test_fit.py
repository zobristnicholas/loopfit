import loopfit as lf
import numpy as np
from numpy.testing import assert_allclose

KEYS = ['qi', 'qc', 'xa', 'a', 'f0']


def check_guess(f, result, guess):
    for parameter, value in guess.items():
        message = f"The guess for {parameter} is incorrect. Used: {value}, Reported: {result['guess'][parameter]}."
        assert value == result['guess'][parameter], message + "\n" + result['summary']
    assert_allclose(lf.model(f, **result['guess']), lf.model(f, **guess))


def check_fit(f, result, guess, parameters):
    assert result['success'], result['summary']
    # check that fit is correct to ~10% for the parameters we care about
    assert_allclose([parameters[key] for key in KEYS], [result[key] for key in KEYS], rtol=1e-1, atol=1e-6,
                    err_msg=result['summary'])
    check_guess(f, result, guess)


def test_fit(f, guess, parameters, data, sigma):
    """Test that the fit works for different parameters."""
    a = parameters.get('a', 0.0)
    alpha = parameters.get('alpha', 1.0)
    beta = parameters.get('beta', 0.0)
    result = lf.fit(f, data.real, data.imag, nonlinear=True if a else False, sigma=sigma,
                    imbalance=True if np.any([alpha != 1.0, beta]) else False, **guess)
    check_fit(f, result, guess, parameters)


def test_fit_z(f, guess, parameters, data, sigma):
    """Test that the fit works for different parameters with the z interface."""
    a = parameters.get('a', 0.0)
    alpha = parameters.get('alpha', 1.0)
    beta = parameters.get('beta', 0.0)
    result = lf.fit(f, z=data, nonlinear=True if a else False, sigma=sigma,
                    imbalance=True if np.any([alpha != 1.0, beta]) else False, **guess)
    check_fit(f, result, guess, parameters)


def test_fit_vector_sigma(f, guess, parameters, data, sigma):
    """Test that the fit works for different parameters using a vector sigma."""
    a = parameters.get('a', 0.0)
    alpha = parameters.get('alpha', 1.0)
    beta = parameters.get('beta', 0.0)
    sigma = np.full_like(data, sigma)
    result = lf.fit(f, data.real, data.imag, nonlinear=True if a else False, sigma=sigma,
                    imbalance=True if np.any([alpha != 1.0, beta]) else False, **guess)
    check_fit(f, result, guess, parameters)


def test_fit_numerical(f, random):
    """Test that the fit works using the numerical Jacobian."""
    # Don't test with the full batch of parameters because this method is less robust and slower
    p = {'f0': 4.0004, 'qi': 400000, 'qc': 90000, 'xa': 0.0, 'a': 1.0, 'alpha': 1.1, 'beta': 0.3,
         'gamma': 0.1, 'delta': 2.0, 'gain0': 2.0, 'gain1': -5000.0, 'gain2': 1250000.0, 'phase0': 4.1,
         'phase1': -4300.0, 'decreasing': True, 'fm': 4.0004}
    m = lf.model(f, **p)
    s = 0.02 + 0.02j
    n = (random.normal(0, s.real, m.shape) + 1j * random.normal(0, s.imag, m.shape)).astype(m.dtype)
    d = m + n
    g = lf.guess(f, z=d, beta=0 if np.abs(p['beta']) < np.pi / 2 else np.pi, gamma=p['gamma'], delta=p['delta'],
                 nonlinear=True if p['a'] else False, decreasing=p['decreasing'])
    r = lf.fit(f, d.real, d.imag, nonlinear=True if p['a'] else False, sigma=s, numerical=True,
               imbalance=True if np.any([p['alpha'] != 1.0, p['beta']]) else False, **g)
    check_fit(f, r, g, p)
