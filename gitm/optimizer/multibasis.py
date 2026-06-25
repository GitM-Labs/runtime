"""Multi-basis anomaly confirmation.

A single large residual is a weak signal — measurement noise, a one-off
scheduler hiccup, or a genuine deviation all look the same in raw position
space. The multi-basis filter requires an anomaly to be confirmed in **two or
more independent bases** before the deviation monitor emits a violation, which
cuts the false-positive rate that would otherwise flood causal attribution.

Two bases here:

* **sequence-position (identity)** — a robust z-score on the residual at its
  position. Catches a value that stands out against the per-op distribution.
* **frequency-domain** — subtract a low-pass (smooth-trend) reconstruction via
  rFFT and z-score the high-frequency remainder. Catches a *localized spike*
  while ignoring slow drift that the position basis would also flag.

A residual is confirmed only if both bases flag it. Series too short for a
meaningful spectrum fall back to the position basis alone (documented, not
silent) so short traces still get detection.
"""

from __future__ import annotations

import numpy as np

#: Below this length the frequency basis is unreliable; use position basis only.
MIN_LEN_FOR_FREQ = 8


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score — robust to the very outliers we're detecting.

    When MAD is 0 (>half the values identical — common for residual series that
    are mostly on-prediction with a few spikes) the std would be inflated by the
    spike itself, so fall back to the mean absolute deviation, which the spike
    barely moves.
    """
    med = np.median(x)
    dev = np.abs(x - med)
    mad = np.median(dev)
    if mad > 0:
        scale = 1.4826 * mad
    else:
        mean_ad = float(dev.mean())
        scale = 1.2533 * mean_ad if mean_ad > 0 else 1e-12  # MeanAD->sigma constant
    return dev / scale


def multibasis_anomalies(series: list[float] | np.ndarray, *, z: float = 3.0) -> np.ndarray:
    """Return a boolean mask of indices confirmed anomalous in 2+ bases.

    ``z`` is the robust-z threshold each basis must exceed. For series shorter
    than :data:`MIN_LEN_FOR_FREQ` only the position basis is applied.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Basis 1 — sequence position.
    b1 = _robust_z(x) > z
    if n < MIN_LEN_FOR_FREQ:
        return b1

    # Basis 2 — frequency domain: high-pass residual after removing the
    # smooth low-frequency trend.
    spectrum = np.fft.rfft(x)
    keep = max(1, n // 8)  # lowest ~1/8 of frequencies = the trend
    lp = spectrum.copy()
    lp[keep:] = 0
    smooth = np.fft.irfft(lp, n=n)
    highpass = x - smooth
    b2 = _robust_z(highpass) > z

    return b1 & b2  # confirmed in BOTH bases


def confirmed_positions(series_by_op: dict[str, list[float]], *, z: float = 3.0) -> set[tuple[str, int]]:
    """Multi-basis-confirmed ``(op, index)`` pairs across all op residual series."""
    out: set[tuple[str, int]] = set()
    for op, vals in series_by_op.items():
        mask = multibasis_anomalies(vals, z=z)
        for i, flagged in enumerate(mask):
            if flagged:
                out.add((op, i))
    return out
