import loopfit as lf
import numpy as np
from numpy.testing import assert_allclose


def test_components_iq(model, baseline, resonance, parameters):
    """Test that  mixer(baseline * resonance) = model (i/q interface)."""
    z = baseline * resonance
    i, q = lf.mixer(z.real, z.imag, **parameters)
    assert_allclose(model, i + 1j * q)


def test_components_z(model, baseline, resonance, parameters):
    """Test that  mixer(baseline * resonance) = model (z interface)."""
    assert_allclose(model, lf.mixer(z=baseline * resonance, **parameters))


def test_calibrate_iq(f, model, resonance, parameters):
    """Test that calibrate(model) = resonance (iq interface)."""
    # rtol raised to accommodate differences between 32 and 64 bit floats
    i, q = lf.calibrate(f, model.real, model.imag, **parameters)  # check alternative signature
    assert_allclose(resonance, i + 1j * q, rtol=1e-5)


def test_calibrate_z(f, model, resonance, parameters):
    """Test that calibrate(model) = resonance (z interface)."""
    # rtol raised to accommodate differences between 32 and 64 bit floats
    assert_allclose(resonance, lf.calibrate(f, z=model, **parameters), rtol=1e-5)


def test_calibrate_center(f, model, parameters):
    """Test if the calibration properly centers the data."""
    radius = np.abs(lf.calibrate(f, z=model, center=True, **parameters))
    # rtol raised to accommodate differences between 32 and 64 bit floats
    assert_allclose(radius[0], radius, rtol=1e-5)  # may break if Qi nonlinearity ever is implemented
