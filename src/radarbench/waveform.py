"""Radar waveform parameters and the transmit pulse.

Everything downstream derives from `RadarConfig`, so the physics constants live
in exactly one place.

How a pulse radar measures range
--------------------------------
Send a pulse, time the echo, distance = c * delay / 2. The difficulty is that
light is fast: a target 15 m away echoes back in 100 ns. Timing that precisely
needs a very short pulse, but a short pulse carries little energy and cannot
reach far.

The way out is a **chirp** (linear frequency modulated pulse): a long pulse
whose frequency ramps steadily from low to high. It is long, so it carries
energy; but every instant inside it has a distinguishable frequency, so the
receiver can still pin down the delay precisely by correlating the echo against
a stored copy of the transmitted pulse. That correlation step is called pulse
compression, and the sharpening it buys is the time-bandwidth product B * T_p.

Defaults model an S-band ground-based surveillance radar, which is the class of
sensor this bench is aimed at - not automotive radar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0  # m/s


@dataclass(frozen=True)
class RadarConfig:
    """Waveform and processing parameters for one coherent processing interval."""

    carrier_hz: float = 3.0e9  # S-band
    bandwidth_hz: float = 10.0e6  # chirp sweep, sets range resolution
    pulse_width_s: float = 5.0e-6  # chirp duration
    sample_rate_hz: float = 20.0e6  # receiver ADC rate
    pri_s: float = 200.0e-6  # pulse repetition interval
    n_pulses: int = 64  # pulses per coherent processing interval
    n_range_bins: int = 512  # samples in the processed receive window

    # --- basic derived quantities ----------------------------------------

    @property
    def wavelength_m(self) -> float:
        """lambda = c / f_c. About 10 cm at S-band."""
        return SPEED_OF_LIGHT / self.carrier_hz

    @property
    def n_pulse_samples(self) -> int:
        """Length of the transmitted chirp in samples."""
        return int(round(self.pulse_width_s * self.sample_rate_hz))

    @property
    def n_raw_samples(self) -> int:
        """Samples the receiver must capture to fill `n_range_bins` cleanly.

        Pulse compression is a convolution with an `n_pulse_samples`-long
        filter, so the first and last `n_pulse_samples - 1` outputs are only
        partially overlapped and carry artificially low energy. Capturing a
        longer raw window and trimming to the valid region removes that edge
        taper, which would otherwise look like a real drop in noise at far
        range - and make the detector appear to behave differently there.
        """
        return self.n_range_bins + self.n_pulse_samples - 1

    @property
    def time_bandwidth_product(self) -> float:
        """B * T_p - how much pulse compression sharpens the echo."""
        return self.bandwidth_hz * self.pulse_width_s

    @property
    def compression_gain_db(self) -> float:
        """SNR improvement from pulse compression, 10*log10(B * T_p)."""
        return 10.0 * np.log10(self.time_bandwidth_product)

    @property
    def integration_gain_db(self) -> float:
        """SNR improvement from coherently summing n_pulses in the Doppler FFT."""
        return 10.0 * np.log10(self.n_pulses)

    @property
    def processing_gain_db(self) -> float:
        """Total SNR gain from raw sample to range-Doppler cell."""
        return self.compression_gain_db + self.integration_gain_db

    # --- range ------------------------------------------------------------

    @property
    def range_resolution_m(self) -> float:
        """dR = c / (2B). Two targets closer than this merge into one."""
        return SPEED_OF_LIGHT / (2.0 * self.bandwidth_hz)

    @property
    def range_bin_spacing_m(self) -> float:
        """Distance between adjacent samples, c / (2 * F_s).

        Deliberately finer than the resolution (the receiver oversamples), so a
        target is never lost between two bins.
        """
        return SPEED_OF_LIGHT / (2.0 * self.sample_rate_hz)

    @property
    def swath_m(self) -> float:
        """Depth of the processed range window."""
        return self.n_range_bins * self.range_bin_spacing_m

    def range_axis(self) -> np.ndarray:
        """Range in metres for each range bin."""
        return np.arange(self.n_range_bins) * self.range_bin_spacing_m

    # --- velocity ----------------------------------------------------------

    @property
    def max_velocity_mps(self) -> float:
        """Unambiguous velocity, +/- lambda / (4 * PRI).

        Past this the pulse-to-pulse phase step exceeds +/- pi and the measured
        velocity folds. Real radars resolve the fold by alternating PRFs; this
        bench stays inside one PRF and simply declares the limit.
        """
        return self.wavelength_m / (4.0 * self.pri_s)

    @property
    def velocity_resolution_mps(self) -> float:
        """dv = lambda / (2 * n_pulses * PRI). Longer dwell, finer velocity."""
        return self.wavelength_m / (2.0 * self.n_pulses * self.pri_s)

    def velocity_axis(self) -> np.ndarray:
        """Radial velocity in m/s per Doppler bin, fftshifted so it is signed.

        Positive means closing on the radar.
        """
        doppler_hz = np.fft.fftshift(np.fft.fftfreq(self.n_pulses, d=self.pri_s))
        return doppler_hz * self.wavelength_m / 2.0

    # --- the transmitted pulse ---------------------------------------------

    def chirp(self) -> np.ndarray:
        """Baseband linear-FM chirp, unit amplitude.

        Instantaneous frequency sweeps linearly across `bandwidth_hz` over the
        pulse, which in phase terms is a quadratic: exp(j * pi * k * t^2) with
        ramp rate k = B / T_p.
        """
        n = self.n_pulse_samples
        t = (np.arange(n) - (n - 1) / 2.0) / self.sample_rate_hz
        ramp_rate = self.bandwidth_hz / self.pulse_width_s
        return np.exp(1j * np.pi * ramp_rate * t**2)

    def matched_filter(self) -> np.ndarray:
        """Pulse-compression filter: time-reversed conjugate of the chirp.

        Convolving the echo with this is the same as correlating it against the
        transmitted pulse, which peaks at the echo's delay.
        """
        return np.conj(self.chirp()[::-1])

    def describe(self) -> str:
        return (
            f"carrier             {self.carrier_hz / 1e9:.2f} GHz "
            f"(lambda {self.wavelength_m * 100:.1f} cm)\n"
            f"bandwidth           {self.bandwidth_hz / 1e6:.1f} MHz\n"
            f"pulse width         {self.pulse_width_s * 1e6:.1f} us "
            f"(B*T = {self.time_bandwidth_product:.0f})\n"
            f"PRI                 {self.pri_s * 1e6:.0f} us "
            f"({self.n_pulses} pulses = {self.n_pulses * self.pri_s * 1e3:.1f} ms dwell)\n"
            f"grid                {self.n_pulses} Doppler x {self.n_range_bins} range "
            f"= {self.n_pulses * self.n_range_bins} cells\n"
            f"range resolution    {self.range_resolution_m:.1f} m "
            f"(bins {self.range_bin_spacing_m:.1f} m, swath {self.swath_m / 1000:.2f} km)\n"
            f"velocity resolution {self.velocity_resolution_mps:.2f} m/s "
            f"(unambiguous +/-{self.max_velocity_mps:.1f} m/s)\n"
            f"processing gain     {self.processing_gain_db:.1f} dB "
            f"({self.compression_gain_db:.1f} compression "
            f"+ {self.integration_gain_db:.1f} integration)"
        )
