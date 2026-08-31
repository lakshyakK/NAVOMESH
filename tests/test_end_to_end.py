"""Role 6 integration smoke test.

This test is intentionally dependency-light and validates the shared contract
from normalized observation through a stand-in recovery, then scientific
validation. Real Role 1/2/3/4/5 modules can replace the stand-ins while keeping
the same contract boundary.
"""
import numpy as np

from contracts import PredictionRequest, PredictionResult, SpectrumBatch, ValidationMetrics


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_full_contract_pipeline():
    wavelength = np.linspace(1.0, 2.0, 1000)
    clean = 0.5 + 0.2 * np.sin(8 * wavelength)
    noisy = np.clip(clean + 0.03 * np.sin(60 * wavelength), 0, 1)

    physics = SpectrumBatch(wavelength, clean[None, :])
    request = PredictionRequest(noisy)
    recovered = PredictionResult(noisy * 0.2 + clean * 0.8)

    rmse_raw = rmse(request.noisy_flux, physics.flux[0])
    rmse_recovered = rmse(recovered.recovered_flux, physics.flux[0])
    metrics = ValidationMetrics(
        rmse_raw=rmse_raw,
        rmse_recovered=rmse_recovered,
        mae_recovered=float(np.mean(np.abs(recovered.recovered_flux - clean))),
        pearson_r=float(np.corrcoef(recovered.recovered_flux, clean)[0, 1]),
        metadata={"noise_regime": "smoke"},
    )

    assert metrics.rmse_recovered < metrics.rmse_raw
    assert metrics.pearson_r > 0.9
