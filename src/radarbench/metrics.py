"""Scoring a detection mask against known ground truth.

Two numbers, which are the familiar classification metrics under radar names:

  Pd  - probability of detection. Of the targets really present, what fraction
        was found. This is recall.
  Pfa - probability of false alarm. Of the cells containing nothing, what
        fraction was flagged. This is the false-positive rate, counted per
        *cell* rather than per cluster, because that is what the CFAR
        threshold maths is defined against.

Any detector can trade one for the other by moving its threshold, so neither
number means anything alone. The meaningful statement always fixes Pfa and
then asks how Pd behaves - which is what `sweep` does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .scene import Target
from .waveform import RadarConfig


@dataclass(frozen=True)
class FrameScore:
    n_targets: int
    n_detected: int
    n_false_cells: int
    n_eligible_cells: int


def truth_bins(config: RadarConfig, target: Target) -> tuple[int, int]:
    """The (Doppler, range) cell a target should land in."""
    r_bin = int(round(target.range_m / config.range_bin_spacing_m))
    d_bin = int(np.argmin(np.abs(config.velocity_axis() - target.velocity_mps)))
    return d_bin, r_bin


def score_frame(
    mask: np.ndarray,
    config: RadarConfig,
    targets: Sequence[Target],
    detect_tolerance: int = 2,
    exclusion: int = 6,
) -> FrameScore:
    """Score one detection mask.

    Args:
        mask: boolean detections over the range-Doppler grid.
        targets: what was actually placed in the scene.
        detect_tolerance: a target counts as found if any flagged cell lies
            within this many cells of its true position. Covers straddle - a
            target between two bins splits its energy across both.
        exclusion: cells within this radius of any target are excluded from
            the false-alarm count. Wider than `detect_tolerance` so that
            target energy spilling into neighbouring cells is never scored as
            a false alarm; that would flatter the detector.
    """
    n_detected = 0
    excluded = np.zeros_like(mask, dtype=bool)

    for target in targets:
        d_bin, r_bin = truth_bins(config, target)

        d_lo, d_hi = d_bin - detect_tolerance, d_bin + detect_tolerance + 1
        r_lo, r_hi = r_bin - detect_tolerance, r_bin + detect_tolerance + 1
        if _window(mask, d_lo, d_hi, r_lo, r_hi).any():
            n_detected += 1

        _mark(excluded, d_bin - exclusion, d_bin + exclusion + 1,
              r_bin - exclusion, r_bin + exclusion + 1)

    eligible = ~excluded
    return FrameScore(
        n_targets=len(targets),
        n_detected=n_detected,
        n_false_cells=int((mask & eligible).sum()),
        n_eligible_cells=int(eligible.sum()),
    )


def _window(a: np.ndarray, d_lo: int, d_hi: int, r_lo: int, r_hi: int) -> np.ndarray:
    """Slice with wraparound in Doppler and clipping in range."""
    rows = np.arange(d_lo, d_hi) % a.shape[0]
    cols = np.arange(max(0, r_lo), min(a.shape[1], r_hi))
    return a[np.ix_(rows, cols)]


def _mark(a: np.ndarray, d_lo: int, d_hi: int, r_lo: int, r_hi: int) -> None:
    rows = np.arange(d_lo, d_hi) % a.shape[0]
    cols = np.arange(max(0, r_lo), min(a.shape[1], r_hi))
    a[np.ix_(rows, cols)] = True
