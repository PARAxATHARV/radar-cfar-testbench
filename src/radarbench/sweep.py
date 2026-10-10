"""The measurement harness - the actual product of this bench.

A detector cannot be judged by looking at one pretty range-Doppler picture. It
has to be characterised: swept across signal strengths and background types,
many independent trials each, with the achieved false-alarm rate measured
rather than assumed.

Everything is seeded. Re-running a sweep with the same seed reproduces the
numbers exactly, which is the difference between evidence and anecdote.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .dsp import Detector, range_doppler
from .metrics import score_frame
from .scene import Background, Target
from .simulate import simulate_frame
from .waveform import RadarConfig


@dataclass
class SweepPoint:
    """Pooled result for one (background, detector, SNR) combination."""

    background: str
    detector: str
    snr_db: float
    n_targets: int = 0
    n_detected: int = 0
    n_false_cells: int = 0
    n_eligible_cells: int = 0

    @property
    def pd(self) -> float:
        return self.n_detected / self.n_targets if self.n_targets else float("nan")

    @property
    def pfa(self) -> float:
        if not self.n_eligible_cells:
            return float("nan")
        return self.n_false_cells / self.n_eligible_cells

    @property
    def pd_stderr(self) -> float:
        """Binomial standard error - how much of a curve wiggle is just noise."""
        if not self.n_targets:
            return float("nan")
        p = self.pd
        return float(np.sqrt(max(p * (1 - p), 0.0) / self.n_targets))


@dataclass
class SweepResult:
    config: RadarConfig
    pfa_design: float
    n_trials: int
    seed: int
    points: list[SweepPoint] = field(default_factory=list)

    def for_detector(self, detector: str, background: str) -> list[SweepPoint]:
        pts = [
            p
            for p in self.points
            if p.detector == detector and p.background == background
        ]
        return sorted(pts, key=lambda p: p.snr_db)

    def detectors(self) -> list[str]:
        return list(dict.fromkeys(p.detector for p in self.points))

    def backgrounds(self) -> list[str]:
        return list(dict.fromkeys(p.background for p in self.points))

    def pooled_pfa(self, detector: str, background: str) -> tuple[float, int, int]:
        """False-alarm rate pooled over every SNR, for tight statistics.

        Pfa is a property of the detector and the background, not of the
        targets, so pooling across SNR is legitimate and buys an order of
        magnitude more cells to measure against.
        """
        pts = self.for_detector(detector, background)
        false_cells = sum(p.n_false_cells for p in pts)
        eligible = sum(p.n_eligible_cells for p in pts)
        return (false_cells / eligible if eligible else float("nan"), false_cells,
                eligible)


def _seed_for(seed: int, background_name: str, snr_db: float) -> list[int]:
    """Derive a per-cell seed that is stable across processes and machines.

    Deliberately not Python's built-in `hash`: it is salted per interpreter
    run, so a sweep seeded through it would produce different numbers every
    time while *looking* reproducible. CRC32 is boring and deterministic,
    which is exactly what is wanted.
    """
    name_hash = zlib.crc32(background_name.encode("utf-8"))
    snr_key = int(round((snr_db + 1000.0) * 100))  # shift to stay non-negative
    return [seed, name_hash, snr_key]


def random_targets(
    config: RadarConfig,
    snr_db: float,
    n_targets: int,
    rng: np.random.Generator,
    min_separation_bins: int = 20,
) -> list[Target]:
    """Scatter targets across the grid at a fixed SNR.

    Positions are re-drawn every trial so the result characterises the
    detector, not one lucky placement. Targets are kept apart so they do not
    sit in each other's CFAR training rings - mutual masking is a real and
    interesting effect, but it is a different experiment.
    """
    margin = 40
    usable = config.n_range_bins - 2 * margin
    v_max = config.max_velocity_mps

    chosen: list[int] = []
    targets: list[Target] = []
    for _ in range(n_targets):
        for _attempt in range(100):
            r_bin = int(rng.integers(margin, margin + usable))
            if all(abs(r_bin - c) >= min_separation_bins for c in chosen):
                chosen.append(r_bin)
                break
        else:
            continue

        # Stay clear of zero Doppler: a stationary target at zero velocity sits
        # on top of where real ground clutter would be, which is a separate
        # problem this bench does not yet model.
        velocity = rng.uniform(0.15, 0.85) * v_max * rng.choice([-1.0, 1.0])
        targets.append(
            Target(
                range_m=r_bin * config.range_bin_spacing_m,
                velocity_mps=float(velocity),
                snr_db=snr_db,
            )
        )
    return targets


def run_sweep(
    config: RadarConfig,
    backgrounds: Sequence[Background],
    detectors: Sequence[Detector],
    snr_grid: Sequence[float],
    n_trials: int = 40,
    pfa: float = 1e-3,
    targets_per_frame: int = 3,
    seed: int = 20261010,
    progress: bool = True,
) -> SweepResult:
    """Monte Carlo over background x SNR, scoring every detector per frame.

    All detectors see the *same* frame, which makes the comparison paired: any
    difference between them is the estimator, not a different noise draw.
    """
    result = SweepResult(config=config, pfa_design=pfa, n_trials=n_trials, seed=seed)
    index = {
        (b.name, d.name, float(s)): SweepPoint(b.name, d.name, float(s))
        for b in backgrounds
        for d in detectors
        for s in snr_grid
    }
    result.points = list(index.values())

    total = len(backgrounds) * len(snr_grid)
    done = 0

    for background in backgrounds:
        for snr_db in snr_grid:
            rng = np.random.default_rng(_seed_for(seed, background.name, snr_db))
            for _ in range(n_trials):
                targets = random_targets(config, float(snr_db), targets_per_frame, rng)
                frame = simulate_frame(config, targets, background, rng)
                power_map = range_doppler(frame, config)

                for detector in detectors:
                    mask, _ = detector.detect(power_map, pfa)
                    score = score_frame(mask, config, targets)
                    point = index[(background.name, detector.name, float(snr_db))]
                    point.n_targets += score.n_targets
                    point.n_detected += score.n_detected
                    point.n_false_cells += score.n_false_cells
                    point.n_eligible_cells += score.n_eligible_cells

            done += 1
            if progress:
                print(
                    f"  [{done:2d}/{total}] {background.name:26s} "
                    f"SNR {snr_db:+6.1f} dB",
                    flush=True,
                )

    return result
