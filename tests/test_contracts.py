import numpy as np
import pytest

from contracts import PredictionRequest, SpectrumBatch


def test_spectrum_batch_enforces_pipeline_shape_and_monotonic_wavelengths():
    wavelength = np.linspace(1.0, 2.0, 1000)
    flux = np.zeros((2, 1000))
    batch = SpectrumBatch(wavelength, flux)
    assert batch.wavelength_um.shape == (1000,)
    assert batch.flux.shape == (2, 1000)


def test_spectrum_batch_rejects_non_monotonic_grid():
    wavelength = np.linspace(1.0, 2.0, 1000)
    wavelength[500] = wavelength[499]
    with pytest.raises(ValueError, match="strictly monotonic"):
        SpectrumBatch(wavelength, np.zeros((1, 1000)))


def test_prediction_request_requires_normalized_1000_point_input():
    request = PredictionRequest(np.full(1000, 0.5))
    assert request.noisy_flux.shape == (1000,)
    with pytest.raises(ValueError):
        PredictionRequest(np.ones(999))
    with pytest.raises(ValueError):
        PredictionRequest(np.full(1000, 1.1))
