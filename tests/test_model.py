import loopfit as lf
from numpy.testing import assert_allclose


def test_components(f, model, baseline, resonance, parameters):
    """Test that  mixer(baseline * resonance) = model."""
    assert_allclose(model, lf.mixer(z=baseline * resonance, **parameters))
    z = baseline * resonance  # check alternative signature
    i, q = lf.mixer(z.real, z.imag, **parameters)
    assert_allclose(model, i + 1j * q)


def test_calibrate(f, model, resonance, parameters):
    """Test that calibrate(model) = resonance."""
    # rtol raised to accommodate differences between 32 and 64 bit floats
    assert_allclose(resonance, lf.calibrate(f, z=model, **parameters), rtol=1e-5)
    i, q = lf.calibrate(f, model.real, model.imag, **parameters)  # check alternative signature
    assert_allclose(resonance, i + 1j * q, rtol=1e-5)
