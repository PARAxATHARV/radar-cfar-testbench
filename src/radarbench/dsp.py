"""Signal processing: raw frame -> range-Doppler map -> detections.

Two stages, then two competing detectors.

Stage 1, pulse compression. Correlate each received pulse against a copy of the
transmitted chirp. The correlation spikes at the echo's delay, which converts a
long low-peak pulse into a sharp one without having transmitted a short pulse.

Stage 2, Doppler processing. Within one range bin, look across pulses: a moving
target's echo phase advances a fixed amount each pulse, so the sequence is a
complex sinusoid whose frequency is the target's velocity. An FFT across pulses
reads that frequency off. Stationary things land at zero Doppler.

The result is a grid indexed by (velocity, range) holding received power.

Stage 3 is detection, and it is where this bench earns its keep. See the two
detector classes below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, optimize, signal, special

from .waveform import RadarConfig


def pulse_compress(frame: np.ndarray, config: RadarConfig) -> np.ndarray:
    """Correlate every pulse with the transmitted chirp.

    Convolving with the time-reversed conjugate chirp *is* correlation.

    The output is trimmed to the **fully supported** region: offsets where the
    filter overlaps real samples along its whole length. Partially overlapped
    edge outputs carry artificially low energy, which would read as a genuine
    drop in the noise floor at far range and quietly change how the detector
    behaves there. The raw frame is captured `n_pulse_samples - 1` samples
    longer than the processed swath precisely so this trim is lossless.
    """
    mf = config.matched_filter()
    full = signal.fftconvolve(frame, mf[None, :], mode="full", axes=1)
    start = config.n_pulse_samples - 1
    return full[:, start : start + config.n_range_bins]


def range_doppler(frame: np.ndarray, config: RadarConfig) -> np.ndarray:
    """Full chain: raw frame -> range-Doppler power map.

    Returns a real array (n_pulses, n_range_bins). The Doppler axis is
    fftshifted, so row 0 is the most negative velocity and the centre row is
    zero Doppler.

    A Hann window is applied across pulses before the Doppler FFT. Without it a
    strong target's spectral sidelobes smear across every velocity bin at that
    range and manufacture false targets. Range sidelobe weighting on the
    matched filter is deliberately *not* applied - it would cost SNR and this
    bench is not studying sidelobe performance.
    """
    compressed = pulse_compress(frame, config)
    windowed = compressed * np.hanning(config.n_pulses)[:, None]
    spectrum = np.fft.fftshift(np.fft.fft(windowed, axis=0), axes=0)
    # Square-law detector. CFAR threshold maths assumes power, not amplitude.
    return np.abs(spectrum) ** 2


def to_db(power_map: np.ndarray, floor_db: float = -200.0) -> np.ndarray:
    """Power in dB with a floor, so empty cells do not become -inf."""
    with np.errstate(divide="ignore"):
        return np.maximum(10.0 * np.log10(power_map), floor_db)


# --- detectors ---------------------------------------------------------------
#
# A fixed brightness threshold cannot work: the background level varies across
# the map (clutter here, clear sky there), so one cutoff is simultaneously too
# strict in one region and too permissive in another.
#
# CFAR - Constant False Alarm Rate - instead estimates the background locally,
# from a ring of training cells around each cell under test, and sets that
# cell's threshold as a multiple of the local estimate. Guard cells sit between
# the cell under test and the training ring so a target's own energy cannot
# inflate the estimate of its own background and hide it.
#
# The two detectors differ only in how they turn the training ring into one
# number. That single choice is what this whole bench measures.


@dataclass(frozen=True)
class CfarGeometry:
    """Training/guard ring dimensions, in cells, per side."""

    guard_doppler: int = 2
    guard_range: int = 2
    train_doppler: int = 4
    train_range: int = 8

    @property
    def window_shape(self) -> tuple[int, int]:
        return (
            2 * (self.guard_doppler + self.train_doppler) + 1,
            2 * (self.guard_range + self.train_range) + 1,
        )

    @property
    def guard_shape(self) -> tuple[int, int]:
        return (2 * self.guard_doppler + 1, 2 * self.guard_range + 1)

    @property
    def n_training_cells(self) -> int:
        w, g = self.window_shape, self.guard_shape
        n = w[0] * w[1] - g[0] * g[1]
        if n <= 0:
            raise ValueError("empty training ring: widen train or shrink guard")
        return n

    @property
    def is_range_only(self) -> bool:
        """True when training cells all share the cell under test's Doppler bin."""
        return self.train_doppler == 0 and self.guard_doppler == 0

    def describe(self) -> str:
        """Short tag naming the training geometry, used in figures and tables."""
        if self.is_range_only:
            return "range-only"
        return "2D ring"

    def footprint(self) -> np.ndarray:
        """Boolean mask of the training ring (guards and centre excluded)."""
        fp = np.ones(self.window_shape, dtype=bool)
        d0 = self.train_doppler
        r0 = self.train_range
        fp[d0 : d0 + self.guard_shape[0], r0 : r0 + self.guard_shape[1]] = False
        return fp


def _pad_for_window(power_map: np.ndarray, geometry: CfarGeometry) -> np.ndarray:
    """Extend the map so every real cell has a full training ring.

    Doppler wraps around - velocity is periodic in the FFT, so the cell past
    the last Doppler bin genuinely is the first one. Range does not wrap, so
    those edges repeat their outermost cell instead.

    Done explicitly rather than through each filter's `mode`, because scipy's
    rank filter accepts only one mode for all axes. Padding here keeps both
    detectors on *identical* edge handling, so any difference between them is
    the estimator and not an artefact of the boundaries.
    """
    pd = geometry.guard_doppler + geometry.train_doppler
    pr = geometry.guard_range + geometry.train_range
    padded = np.pad(power_map, ((pd, pd), (0, 0)), mode="wrap")
    return np.pad(padded, ((0, 0), (pr, pr)), mode="edge")


def _crop(padded: np.ndarray, geometry: CfarGeometry) -> np.ndarray:
    pd = geometry.guard_doppler + geometry.train_doppler
    pr = geometry.guard_range + geometry.train_range
    return padded[pd : padded.shape[0] - pd, pr : padded.shape[1] - pr]


class Detector(ABC):
    """Produces a per-cell threshold for a target false-alarm rate."""

    name: str

    @abstractmethod
    def threshold(self, power_map: np.ndarray, pfa: float) -> np.ndarray:
        """Per-cell detection threshold in linear power."""

    def detect(self, power_map: np.ndarray, pfa: float) -> tuple[np.ndarray, np.ndarray]:
        thr = self.threshold(power_map, pfa)
        return power_map > thr, thr


class CaCfar(Detector):
    """Cell-averaging CFAR: background estimated as the *mean* of the ring.

    Optimal when the background really is Gaussian, and it is the standard
    textbook detector. Its weakness is structural: a mean is dragged upward by
    extreme values, so a spiky background inflates every local estimate in a
    correlated way and the achieved false-alarm rate stops matching the design.
    """

    def __init__(self, geometry: CfarGeometry | None = None, label: str | None = None):
        self.geometry = geometry or CfarGeometry()
        self.name = label or f"CA-CFAR {self.geometry.describe()}"

    def threshold(self, power_map: np.ndarray, pfa: float) -> np.ndarray:
        g = self.geometry
        n_window = g.window_shape[0] * g.window_shape[1]
        n_guard = g.guard_shape[0] * g.guard_shape[1]
        n_train = g.n_training_cells

        padded = _pad_for_window(power_map, g)

        # uniform_filter returns the window mean; scale back to a sum so the
        # guard region can be subtracted out, leaving only the training ring.
        sum_window = (
            ndimage.uniform_filter(padded, size=g.window_shape, mode="nearest")
            * n_window
        )
        sum_guard = (
            ndimage.uniform_filter(padded, size=g.guard_shape, mode="nearest") * n_guard
        )
        background = _crop((sum_window - sum_guard) / n_train, g)

        return ca_cfar_alpha(n_train, pfa) * background


class OsCfar(Detector):
    """Order-statistic CFAR: background estimated from the k-th *sorted* cell.

    Instead of averaging the ring, sort it and take one value - a median-like
    statistic. A handful of enormous returns no longer drag the estimate up, so
    the detector keeps its false-alarm rate under a heavy-tailed background.

    The cost, visible in the Pd curves, is a small loss of sensitivity under
    clean Gaussian noise, where the mean is the better estimator. That tradeoff
    is the engineering result this bench exists to quantify.
    """

    def __init__(
        self,
        geometry: CfarGeometry | None = None,
        rank_fraction: float = 0.75,
        label: str | None = None,
    ):
        self.geometry = geometry or CfarGeometry()
        self.rank_fraction = rank_fraction
        n = self.geometry.n_training_cells
        # 1-indexed rank into the ascending-sorted training cells.
        self.rank = max(1, min(n, int(round(rank_fraction * n))))
        self.name = label or f"OS-CFAR {self.geometry.describe()}"

    def threshold(self, power_map: np.ndarray, pfa: float) -> np.ndarray:
        g = self.geometry
        background = _crop(
            ndimage.rank_filter(
                _pad_for_window(power_map, g),
                rank=self.rank - 1,  # ndimage ranks are 0-indexed
                footprint=g.footprint(),
                mode="nearest",
            ),
            g,
        )
        alpha = os_cfar_alpha(g.n_training_cells, self.rank, pfa)
        return alpha * background


# --- threshold multipliers ---------------------------------------------------


def ca_cfar_alpha(n_train: int, pfa: float) -> float:
    """Cell-averaging multiplier: alpha = N * (Pfa^(-1/N) - 1).

    Derived assuming exponentially distributed cell power, i.e. a Gaussian
    background. When that assumption fails, so does the resulting false-alarm
    rate - which is the point of the experiment.
    """
    if not 0 < pfa < 1:
        raise ValueError("pfa must be in (0, 1)")
    return n_train * (pfa ** (-1.0 / n_train) - 1.0)


def os_cfar_alpha(n_train: int, rank: int, pfa: float) -> float:
    """Order-statistic multiplier, solved numerically (Rohling, 1983).

    For the k-th of N ascending-sorted exponential training cells,

        Pfa = k * C(N, k) * B(alpha + N - k + 1, k)

    which has no closed form in alpha, so it is inverted by bisection in log
    space (the factorials overflow otherwise).
    """
    if not 0 < pfa < 1:
        raise ValueError("pfa must be in (0, 1)")
    if not 1 <= rank <= n_train:
        raise ValueError("rank must lie in [1, n_train]")

    log_prefix = (
        np.log(rank)
        + special.gammaln(n_train + 1)
        - special.gammaln(rank + 1)
        - special.gammaln(n_train - rank + 1)
    )

    def log_pfa_error(alpha: float) -> float:
        return (
            log_prefix
            + special.betaln(alpha + n_train - rank + 1, rank)
            - np.log(pfa)
        )

    return float(optimize.brentq(log_pfa_error, 1e-9, 1e9, xtol=1e-10, rtol=1e-12))


# --- turning a detection mask into target reports ----------------------------


@dataclass(frozen=True)
class Detection:
    range_m: float
    velocity_mps: float
    range_bin: int
    doppler_bin: int
    snr_over_threshold_db: float


def extract_detections(
    power_map: np.ndarray,
    mask: np.ndarray,
    threshold: np.ndarray,
    config: RadarConfig,
) -> list[Detection]:
    """Collapse connected clusters of flagged cells into one report each.

    One target lights up several adjacent cells, so raw mask cells are grouped
    by connectivity and only each group's peak is reported.
    """
    labels, n_groups = ndimage.label(mask)
    if n_groups == 0:
        return []

    range_axis = config.range_axis()
    velocity_axis = config.velocity_axis()

    peaks = ndimage.maximum_position(power_map, labels, range(1, n_groups + 1))

    out = []
    for d_bin, r_bin in peaks:
        d_bin, r_bin = int(d_bin), int(r_bin)
        out.append(
            Detection(
                range_m=float(range_axis[r_bin]),
                velocity_mps=float(velocity_axis[d_bin]),
                range_bin=r_bin,
                doppler_bin=d_bin,
                snr_over_threshold_db=float(
                    10.0 * np.log10(power_map[d_bin, r_bin] / threshold[d_bin, r_bin])
                ),
            )
        )
    out.sort(key=lambda d: d.range_m)
    return out
