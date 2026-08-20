"""Shared integration contracts for the SIH037 exoplanet spectrum pipeline.

Role 6 owns these immutable, typed handoff structures so that physics,
data/noise, ML inference, validation, and UI modules exchange predictable data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SPECTRUM_LENGTH = 1000


def _readonly(array: np.ndarray, *, ndim: int | None = None) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"Expected {ndim}D array, got shape {value.shape}")
    value = value.copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class SpectrumBatch:
    """A wavelength grid plus a batch of spectra."""

    wavelength_um: np.ndarray
    flux: np.ndarray

    def __post_init__(self) -> None:
        wavelength = _readonly(self.wavelength_um, ndim=1)
        flux = _readonly(self.flux, ndim=2)
        if wavelength.shape != (SPECTRUM_LENGTH,):
            raise ValueError(f"Wavelength grid must have shape ({SPECTRUM_LENGTH},)")
        if flux.shape[1] != SPECTRUM_LENGTH:
            raise ValueError(f"Flux must have {SPECTRUM_LENGTH} wavelength points")
        if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(flux)):
            raise ValueError("Spectrum data must contain only finite values")
        if np.any(np.diff(wavelength) <= 0):
            raise ValueError("Wavelength grid must be strictly monotonic")
        object.__setattr__(self, "wavelength_um", wavelength)
        object.__setattr__(self, "flux", flux)


@dataclass(frozen=True)
class DatasetSplit:
    """Non-leaking train/validation/test handoff."""

    train: SpectrumBatch
    validation: SpectrumBatch
    test: SpectrumBatch


@dataclass(frozen=True)
class PredictionRequest:
    """Single normalized spectrum sent to the inference module."""

    noisy_flux: np.ndarray

    def __post_init__(self) -> None:
        flux = _readonly(self.noisy_flux, ndim=1)
        if flux.shape != (SPECTRUM_LENGTH,):
            raise ValueError(f"noisy_flux must have shape ({SPECTRUM_LENGTH},)")
        if not np.all(np.isfinite(flux)) or np.any((flux < 0) | (flux > 1)):
            raise ValueError("noisy_flux must be finite and normalized to [0, 1]")
        object.__setattr__(self, "noisy_flux", flux)


@dataclass(frozen=True)
class PredictionResult:
    """Recovered spectrum returned by the ML layer."""

    recovered_flux: np.ndarray

    def __post_init__(self) -> None:
        flux = _readonly(self.recovered_flux, ndim=1)
        if flux.shape != (SPECTRUM_LENGTH,):
            raise ValueError(f"recovered_flux must have shape ({SPECTRUM_LENGTH},)")
        if not np.all(np.isfinite(flux)):
            raise ValueError("recovered_flux must contain only finite values")
        object.__setattr__(self, "recovered_flux", flux)


@dataclass(frozen=True)
class ValidationMetrics:
    """Authoritative metrics consumed by reporting and the UI."""

    rmse_raw: float
    rmse_recovered: float
    mae_recovered: float
    pearson_r: float
    metadata: Mapping[str, float | str]

    @property
    def rmse_reduction(self) -> float:
        if self.rmse_raw <= 0:
            return 0.0
        return 1.0 - self.rmse_recovered / self.rmse_raw

    @property
    def target_passed(self) -> bool:
        return self.rmse_recovered < 0.3 * self.rmse_raw


@dataclass(frozen=True)
class DemoSample:
    """Complete sample for the Streamlit demo mode."""

    wavelength_um: Sequence[float]
    noisy_flux: Sequence[float]
    recovered_flux: Sequence[float]
    ground_truth_flux: Sequence[float] | None = None
