"""Targets and background models.

The background is the part that matters for this bench.

Detection theory textbooks assume the background is thermal noise: complex
Gaussian, so echo *power* is exponentially distributed. That assumption is what
makes a cell-averaging detector's false-alarm maths work.

Real backgrounds are not that. Echoes from terrain, buildings, sea and rain are
**spiky**: occasional returns far larger than the mean, much more often than a
Gaussian model predicts. Radar engineers model this with heavy-tailed
distributions - Weibull and K being the two standard choices.

The first three models are normalised to the same mean power, so the only
thing that changes between them is the *shape* of the tail. That isolation is
deliberate: any difference in detector behaviour across those three is
attributable to tail heaviness and nothing else.

`PatchyClutter` is the exception, and necessarily so - it adds correlated
clutter *on top of* a thermal floor rather than replacing it, so its mean
power is higher. Target SNR stays referenced to the thermal floor throughout,
which is the standard radar convention and keeps a given SNR meaning the same
thing in every background.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import special


@dataclass(frozen=True)
class Target:
    """A point reflector.

    Attributes:
        range_m: distance from the radar.
        velocity_mps: radial velocity, positive means closing.
        snr_db: per-sample SNR *before* pulse compression and Doppler
            integration. The range-Doppler cell sees this plus the config's
            processing gain (about 35 dB at defaults), so values well below
            zero are still comfortably detectable.
    """

    range_m: float
    velocity_mps: float
    snr_db: float = 0.0


class Background(ABC):
    """A statistical model for everything that is not a target."""

    name: str

    @abstractmethod
    def sample(self, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        """Draw complex samples with mean power 1 (callers scale)."""

    def spikiness(self, rng: np.random.Generator, n: int = 400_000) -> float:
        """Ratio of the 99.9th-percentile power to the mean power, in dB.

        A single honest number for "how heavy is the tail". Thermal noise sits
        near 8.4 dB; anything well above that will stress a cell-averaging
        detector.
        """
        power = np.abs(self.sample((n,), rng)) ** 2
        return float(10.0 * np.log10(np.percentile(power, 99.9) / power.mean()))


class Thermal(Background):
    """Complex Gaussian - the textbook assumption. Power is exponential."""

    name = "thermal (Gaussian)"

    def sample(self, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        scale = np.sqrt(0.5)  # split power evenly between I and Q
        return rng.normal(0, scale, shape) + 1j * rng.normal(0, scale, shape)


class Weibull(Background):
    """Weibull-amplitude clutter with uniform phase.

    `shape` = 2 reproduces Rayleigh amplitude, i.e. ordinary Gaussian clutter.
    Below 2 the tail gets heavier; 1.2 is representative of spiky land clutter.
    """

    def __init__(self, shape: float = 1.2):
        if shape <= 0:
            raise ValueError("Weibull shape must be positive")
        self.shape = shape
        self.name = f"Weibull (shape {shape:g})"
        # E[a^2] for unit-scale Weibull is gamma(1 + 2/k); divide it out so
        # mean power is 1 regardless of shape.
        self._norm = np.sqrt(special.gamma(1.0 + 2.0 / shape))

    def sample(self, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        amplitude = rng.weibull(self.shape, shape) / self._norm
        phase = rng.uniform(0, 2 * np.pi, shape)
        return amplitude * np.exp(1j * phase)


class KDistributed(Background):
    """K-distributed clutter: Gamma-modulated Gaussian speckle.

    The standard sea- and land-clutter model. Physically it says the local mean
    power itself fluctuates (large-scale structure - swell, terrain) while fast
    speckle rides on top. Small `shape` means spikier; above about 10 it is
    indistinguishable from Gaussian.
    """

    def __init__(self, shape: float = 0.5):
        if shape <= 0:
            raise ValueError("K shape must be positive")
        self.shape = shape
        self.name = f"K-distributed (shape {shape:g})"

    def sample(self, s: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        # Gamma texture with unit mean, so total mean power stays 1.
        texture = rng.gamma(self.shape, 1.0 / self.shape, s)
        speckle = rng.normal(0, np.sqrt(0.5), s) + 1j * rng.normal(0, np.sqrt(0.5), s)
        return np.sqrt(texture) * speckle


class PatchyClutter(Background):
    """Spatially correlated clutter: patches that persist through a dwell.

    Why the models above are not enough
    -----------------------------------
    Thermal, Weibull and KDistributed draw every cell independently. But the
    processing chain sums ~100 samples in pulse compression and then 64 pulses
    in the Doppler FFT. Adding up hundreds of independent random values drives
    the result toward a Gaussian whatever its input distribution - the central
    limit theorem - so an independently drawn heavy tail is *flattened back
    into ordinary noise* before any detector ever sees it. Measured on this
    bench, the independent models leave CA-CFAR and OS-CFAR indistinguishable.

    Real clutter survives coherent processing because it is not independent. A
    tree line or a wave crest is the *same* physical scatterer across every
    pulse of a dwell, and it spans several neighbouring range cells. This model
    reproduces both correlations:

      - **range**: one Gamma texture value per patch of `patch_cells` range
        cells, so the heavy tail is preserved exactly (drawn, not smoothed)
        while neighbouring cells share a local mean power;
      - **slow time**: speckle low-pass filtered across pulses, so the clutter
        decorrelates over `dwell_corr_pulses` rather than instantly, which
        concentrates it near zero Doppler the way stationary clutter is.

    The result is what a CFAR training ring actually faces: cells a few bins
    apart with wildly different power. A mean is dragged upward by the bright
    ones; an order statistic is not.

    Unlike the models above, this one sits **on top of a thermal noise floor**
    rather than replacing it, which is the physical situation: a receiver has
    thermal noise everywhere and clutter only where something is reflecting.
    Without that floor, cells away from the clutter ridge would be empty
    rather than merely quiet, and a target placed there would be detectable at
    an SNR no real radar could reach. Mean power is therefore `1 + CNR` rather
    than 1, and `Target.snr_db` stays referenced to the thermal floor - the
    standard radar convention, and the only one under which an SNR means the
    same thing across all four backgrounds.
    """

    def __init__(
        self,
        shape: float = 0.5,
        patch_cells: int = 8,
        dwell_corr_pulses: float = 6.0,
        cnr_db: float = 20.0,
    ):
        if shape <= 0:
            raise ValueError("shape must be positive")
        if patch_cells < 1:
            raise ValueError("patch_cells must be at least 1")
        self.shape = shape
        self.patch_cells = patch_cells
        self.dwell_corr_pulses = dwell_corr_pulses
        self.cnr_db = cnr_db
        self.name = f"patchy clutter (CNR {cnr_db:g} dB)"

    def sample(self, s: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        n_pulses, n_samples = s

        # Texture: one Gamma draw per patch, repeated across the patch's cells.
        # Drawing coarse rather than smoothing a fine field keeps the marginal
        # distribution exactly Gamma, so the tail is not quietly averaged away.
        n_patches = int(np.ceil(n_samples / self.patch_cells))
        patches = rng.gamma(self.shape, 1.0 / self.shape, n_patches)
        texture = np.repeat(patches, self.patch_cells)[:n_samples]

        # Speckle, correlated across pulses. Unit-energy kernel so filtering
        # shapes the spectrum without changing total power.
        speckle = rng.normal(0, np.sqrt(0.5), s) + 1j * rng.normal(0, np.sqrt(0.5), s)
        if self.dwell_corr_pulses > 0:
            kernel = _unit_energy_gaussian(self.dwell_corr_pulses, n_pulses)
            speckle = np.apply_along_axis(
                lambda col: np.convolve(col, kernel, mode="same"), 0, speckle
            )

        clutter = np.sqrt(texture)[None, :] * speckle
        thermal = Thermal().sample(s, rng)
        return thermal + np.sqrt(10.0 ** (self.cnr_db / 10.0)) * clutter


def _unit_energy_gaussian(sigma: float, max_len: int) -> np.ndarray:
    """Gaussian kernel scaled so that filtering white noise preserves power.

    Capped at the signal length: numpy's 'same' convolution returns the longer
    of the two inputs, so a kernel wider than the pulse count would silently
    change the array shape.
    """
    half = max(min(int(np.ceil(3 * sigma)), (max_len - 1) // 2), 1)
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / np.sqrt(np.sum(k**2))


def standard_backgrounds() -> list[Background]:
    """The backgrounds the characterisation sweeps over.

    Ordered by how hard they are on a cell-averaging detector: the first three
    are independent cell to cell and get Gaussianised by the processing chain;
    the last is correlated and does not.
    """
    return [
        Thermal(),
        Weibull(shape=1.2),
        KDistributed(shape=0.5),
        PatchyClutter(shape=0.5),
    ]
