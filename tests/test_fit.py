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
