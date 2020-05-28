import loopfit as lf
from numpy.testing import assert_allclose


def test_components(f, model, baseline, resonance, parameters):
    assert_allclose(model, lf.mixer(f, z=baseline * resonance, **parameters))


def test_calibrate(f, model, resonance, parameters):
    # rtol raised to accommodate differences between 32 and 64 bit floats
    assert_allclose(resonance, lf.calibrate(f, z=model, **parameters), rtol=1e-5)
    i, q = lf.calibrate(f, model.real, model.imag, **parameters)
    assert_allclose(resonance, i + 1j * q, rtol=1e-5)
