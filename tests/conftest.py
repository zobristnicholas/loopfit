import pytest
import numpy as np
import loopfit as lf


@pytest.fixture
def random():
    return np.random.RandomState(0)


@pytest.fixture(params=[np.float32, np.float64])
def f(request):
    return np.linspace(4, 4.002, 1000, dtype=request.param)


@pytest.fixture(params=[{'f0': 4.0012, 'qi': 100000, 'qc': 30000, 'xa': 5e-6, 'a': 0.8, 'alpha': 1.2, 'beta': 0.2,
                         'gain0': 3.0, 'gain1': 1000.0, 'gain2': 500000.0, 'phase0': 2.0, 'phase1': -1000.0},
                        {'f0': 4.0002, 'qi': 200000, 'qc': 10000, 'xa': -5e-6, 'a': 0.0, 'alpha': 0.9, 'beta': 3.1,
                         'gamma': 1.9, 'delta': -3.0, 'gain0': 10.0, 'gain1': -1000.0, 'gain2': 800000.0, 'phase0': 1.0,
                         'phase1': -2000.0},
                        {'f0': 4.0004, 'qi': 500000, 'qc': 80000, 'xa': 0.0, 'a': 0.1, 'alpha': 1.0, 'beta': 0.0,
                         'gamma': 0.0, 'delta': 0.0, 'gain0': 2.0, 'gain1': -3000.0, 'gain2': 950000.0, 'phase0': 4.1,
                         'phase1': -4300.0},
                        {'f0': 4.0004, 'qi': 400000, 'qc': 90000, 'xa': 0.0, 'a': 1.0, 'alpha': 1.1, 'beta': 0.3,
                         'gamma': 0.1, 'delta': 2.0, 'gain0': 2.0, 'gain1': -5000.0, 'gain2': 1250000.0, 'phase0': 4.1,
                         'phase1': -4300.0, 'decreasing': True, 'fm': 4.0004},
                        {'f0': 4.0004, 'qi': 400000, 'qc': 90000, 'xa': 0.0, 'a': 1.0, 'alpha': 1.1, 'beta': 0.3,
                         'gamma': 0.1, 'delta': 2.0, 'gain0': 2.0, 'gain1': -5000.0, 'gain2': 1250000.0, 'phase0': 4.1,
                         'phase1': -4300.0, 'decreasing': False, 'fm': 4.0004}])
def parameters(request):
    return request.param


@pytest.fixture(params=[np.complex64, np.complex128])
def model(request, f, parameters):
    return lf.model(f, **parameters).astype(request.param)


@pytest.fixture
def baseline(f, parameters):
    return lf.baseline(f, **parameters)


@pytest.fixture
def resonance(f, parameters):
    return lf.resonance(f, **parameters)


@pytest.fixture(params=[0.02 + 0.02j, 0.01 + 0.03j])
def sigma(request):
    return request.param


@pytest.fixture
def data(model, random, sigma):
    noise = (random.normal(0, sigma.real, model.shape) +
             1j * random.normal(0, sigma.imag, model.shape)).astype(model.dtype)
    return model + noise


@pytest.fixture
def guess(f, data, parameters):
    a = parameters.get('a', 0.0)
    beta = parameters.get('beta', 0.0)
    gamma = parameters.get('gamma', 0.0)
    delta = parameters.get('delta', 0.0)
    decreasing = parameters.get('decreasing', False)
    # guess by providing the offset if supplied and picking the branch of beta.
    return lf.guess(f, z=data, beta=0 if np.abs(beta) < np.pi / 2 else np.pi, gamma=gamma, delta=delta,
                    nonlinear=True if a else False, decreasing=decreasing)
