# SIH037 — Exoplanet Atmospheric Spectrum Recovery

> Integrated systems architecture for recovering exoplanet atmospheric spectra from noise-contaminated telescope observations.

## Pipeline

**Physics simulation → Noise/artifact engine → PyTorch 1D CNN recovery → Scientific validation → Streamlit UI → Integrated platform**

The project follows the SIH037 Role Handbook and uses 1,000-point wavelength spectra as the shared interface between stages.

## Role 6 — Systems Integrator & Project Lead

Role 6 owns repository architecture, integration contracts, end-to-end verification, documentation, and release discipline.

### Repository contract

- `contracts.py` — immutable, typed handoff structures shared across roles.
- `tests/test_contracts.py` — contract-level shape, normalization, and wavelength checks.
- `tests/test_end_to_end.py` — dependency-light end-to-end integration smoke test.
- `.gitignore` — excludes local environments and generated artifacts.
- `environment.yml` — unified environment definition.

### Expected project layout

```text
.
├── app.py
├── contracts.py
├── inference.py
├── environment.yml
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.npz
│       ├── val.npz
│       └── test.npz
├── models/
│   └── spectrum_unet.pt
├── reports/
│   ├── metrics.json
│   └── validation_report.md
└── tests/
    ├── test_contracts.py
    └── test_end_to_end.py
```

## Handoff interfaces

| Sender → Recipient | Required package |
| --- | --- |
| Role 2 → Role 3 | Clean spectra + parameters in HDF5/CSV, 1,000-point monotonic wavelength grid (µm) |
| Role 3 → Role 1 | Train/validation/test `.npz`, arrays shaped `[Batch, 1, 1000]` |
| Role 1 → Role 4 | `spectrum_unet.pt` and prediction arrays on the frozen test set |
| Role 1 → Role 5 | Standalone `inference.py` exporting `predict_spectrum()` |
| Role 4 → Role 6 | JSON metrics, residual heatmaps, and validation markdown report |

## Integration gates

1. Physics output contains finite flux values and a strictly monotonic wavelength grid.
2. Data splits are isolated by parameter regimes; normalization is not fitted across all splits.
3. ML inference accepts a normalized 1,000-point 1D spectrum and returns a 1,000-point recovered spectrum.
4. Validation reports RMSE, MAE, Pearson correlation, residual diagnostics, and stress-test outcomes.
5. UI handles invalid inputs with friendly errors and provides a safe demo mode.
6. `pytest` passes in the clean project environment before release.

## Release discipline

Freeze dependencies and the integrated codebase 24 hours before presentation. Changes after the freeze require an explicit integration decision and a clean verification run.

## Current status

Role 6 foundation has been added on `feature/role-6-integration`. The repository currently starts from a minimal README, so role-specific modules from Roles 1–5 are not yet present in this repository and are not fabricated here.
