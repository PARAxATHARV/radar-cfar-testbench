"""Generate raw radar frames with known ground truth.

Output is the data cube a receiver would hand you: complex samples shaped
(n_pulses, n_range_bins). Axis 1 is fast time - samples within one pulse
repetition interval, which carries range. Axis 0 is slow time - pulse index,
which carries velocity.

Why synthetic
-------------
This bench measures how well a detector works. That is impossible without
knowing what was actually there, at a precisely controlled signal strength,
against a precisely controlled background. No recorded dataset gives you that;
you cannot ask the sky for a target at exactly -28 dB SNR in K-distributed
clutter. Controlled ground truth is not a shortcut here, it is the instrument.

Signal model
------------
A target at range R, radial velocity v contributes, at pulse m:

    echo[m, :] = A * exp(j * (2*pi*f_d*m*PRI + phi)) * chirp_delayed(2R/c)

    f_d = 2v / lambda    the Doppler shift

The delay puts the echo at the right place along fast time; the per-pulse phase
term is the velocity. Between pulses the target moves only centimetres - far
too little to change range bin - but at a 10 cm wavelength that is a large,
measurable fraction of a cycle, which is exactly why a second FFT across pulses
recovers velocity.

Known simplifications, stated rather than hidden:
  - no range migration within a dwell (targets stay in their range bin)
  - no antenna pattern, scan modulation or eclipsing
  - background is statistically independent cell to cell, so there is no
    coherent zero-Doppler ground-clutter ridge
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .scene import Background, Target
from .waveform import SPEED_OF_LIGHT, RadarConfig


def simulate_frame(
    config: RadarConfig,
    targets: Sequence[Target],
    background: Background,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one raw IQ frame.

    The background is generated with mean power 1, so a target's `snr_db` is
    read directly against it. Comparing backgrounds therefore isolates tail
    shape: mean power is identical across models, only the distribution differs.
    """
    frame = background.sample((config.n_pulses, config.n_raw_samples), rng)

    for target in targets:
        frame += _target_echo(config, target, rng)

    return frame


def _target_echo(
    config: RadarConfig, target: Target, rng: np.random.Generator
) -> np.ndarray:
    """One target's contribution across the whole frame."""
    delay_s = 2.0 * target.range_m / SPEED_OF_LIGHT
    pulse = _delayed_chirp(config, delay_s)

    doppler_hz = 2.0 * target.velocity_mps / config.wavelength_m
    m = np.arange(config.n_pulses)
    slow_time_phase = 2 * np.pi * doppler_hz * m * config.pri_s

    # Absolute range contributes a phase the receiver cannot know; randomising
    # it stops a detector from accidentally depending on a fixed value.
    phi = rng.uniform(0, 2 * np.pi)
    amplitude = 10.0 ** (target.snr_db / 20.0)

    return amplitude * np.exp(1j * (slow_time_phase[:, None] + phi)) * pulse[None, :]


def _delayed_chirp(config: RadarConfig, delay_s: float) -> np.ndarray:
    """The transmitted chirp, delayed by `delay_s`, in the receive window.

    Applied as a linear phase ramp in the frequency domain rather than by
    shifting whole samples, so a target can sit between two range bins - which
    real targets do, and which costs a little peak amplitude (straddle loss).
    Rounding to the nearest sample would hide that effect and flatter the
    detector.
    """
    n = config.n_raw_samples
    tx = np.zeros(n, dtype=np.complex128)
    tx[: config.n_pulse_samples] = config.chirp()

    freqs = np.fft.fftfreq(n, d=1.0 / config.sample_rate_hz)
    return np.fft.ifft(np.fft.fft(tx) * np.exp(-2j * np.pi * freqs * delay_s))
