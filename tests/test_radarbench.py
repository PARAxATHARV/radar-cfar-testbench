"""Correctness gates.

These are not smoke tests. Each one pins a property that, if it broke, would
make the bench's numbers wrong while everything still appeared to run - which
is the failure mode that matters for a measurement instrument.
"""

from __future__ import annotations

import numpy as np
import pytest

from radarbench import (
    Background,
    CaCfar,
    CfarGeometry,
    KDistributed,
    OsCfar,
    PatchyClutter,
    RadarConfig,
    Target,
    Thermal,
    Weibull,
    ca_cfar_alpha,
    extract_detections,
    os_cfar_alpha,
    pulse_compress,
    range_doppler,
    run_sweep,
    simulate_frame,
)
from radarbench.metrics import score_frame, truth_bins

# --- waveform ----------------------------------------------------------------


def test_derived_quantities_match_textbook_formulas():
    c = RadarConfig()
    assert c.range_resolution_m == pytest.approx(
        299_792_458.0 / (2 * c.bandwidth_hz)
    )
    assert c.max_velocity_mps == pytest.approx(c.wavelength_m / (4 * c.pri_s))
    assert c.velocity_resolution_mps == pytest.approx(
        c.wavelength_m / (2 * c.n_pulses * c.pri_s)
    )


def test_matched_filter_compresses_the_chirp():
    """Peak-to-sidelobe must stay high, or weak targets drown in strong ones."""
    c = RadarConfig()
    chirp = c.chirp()
    compressed = np.abs(np.convolve(chirp, c.matched_filter()))
    peak = compressed.argmax()
    assert compressed[peak] == pytest.approx(c.n_pulse_samples, rel=1e-9)
    sidelobes = np.concatenate([compressed[: peak - 4], compressed[peak + 5 :]])
    assert 20 * np.log10(compressed[peak] / sidelobes.max()) > 12.0


class _NoBackground(Background):
    """Signal with no background, so processing gain can be measured directly."""

    name = "none"

    def sample(self, shape, rng):  # noqa: ARG002
        return np.zeros(shape, dtype=complex)


def test_processing_gain_matches_the_claim():
    """`processing_gain_db` is quoted in the README and in every figure
    subtitle. Measure the real chain and fail if the claim drifts from it.

    An advertised sensitivity nobody checks is how a bench ends up lying
    confidently: the number stays plausible while the code moves underneath.
    """
    c = RadarConfig()
    rng = np.random.default_rng(0)
    signal = simulate_frame(
        c, [Target(1500.0, 40.0, snr_db=0.0)], _NoBackground(), rng
    )
    noise = simulate_frame(c, [], Thermal(), rng)

    def snr_db(sig_power: np.ndarray, noise_power: np.ndarray) -> float:
        return 10 * np.log10(sig_power.max() / noise_power.mean())

    raw = snr_db(np.abs(signal) ** 2, np.abs(noise) ** 2)
    processed = snr_db(range_doppler(signal, c), range_doppler(noise, c))

    assert processed - raw == pytest.approx(c.processing_gain_db, abs=1.0)


def test_compression_gain_is_not_the_critically_sampled_figure():
    """Guards a specific, easy mistake: quoting 10*log10(B*T) while sampling
    at twice the Nyquist rate understates the real gain by 3 dB."""
    c = RadarConfig()
    assert c.sample_rate_hz == pytest.approx(2 * c.bandwidth_hz)
    assert c.compression_gain_db == pytest.approx(
        10 * np.log10(c.n_pulse_samples)
    )
    assert c.compression_gain_db > 10 * np.log10(c.time_bandwidth_product) + 2.5


# --- simulation and DSP ------------------------------------------------------


def test_target_lands_in_the_right_cell():
    """The whole bench is worthless if truth and measurement disagree."""
    c = RadarConfig()
    target = Target(range_m=1200.0, velocity_mps=55.0, snr_db=30.0)
    power = range_doppler(
        simulate_frame(c, [target], Thermal(), np.random.default_rng(0)), c
    )
    d_bin, r_bin = np.unravel_index(power.argmax(), power.shape)
    truth_d, truth_r = truth_bins(c, target)
    assert abs(int(r_bin) - truth_r) <= 1
    assert abs(int(d_bin) - truth_d) <= 1


def test_noise_floor_is_flat_across_range():
    """Guards the convolution-edge bug: partial overlap at the far edge made
    the noise floor sag, which would read as better detection at long range."""
    c = RadarConfig()
    power = range_doppler(
        simulate_frame(c, [], Thermal(), np.random.default_rng(1)), c
    )
    per_bin = power.mean(axis=0)
    near, far = per_bin[: c.n_range_bins // 4].mean(), per_bin[-c.n_range_bins // 4 :].mean()
    assert 10 * abs(np.log10(far / near)) < 1.0  # within 1 dB


def test_raw_frame_is_longer_than_the_processed_swath():
    c = RadarConfig()
    frame = simulate_frame(c, [], Thermal(), np.random.default_rng(2))
    assert frame.shape == (c.n_pulses, c.n_raw_samples)
    assert pulse_compress(frame, c).shape == (c.n_pulses, c.n_range_bins)


# --- backgrounds -------------------------------------------------------------


@pytest.mark.parametrize(
    "background", [Thermal(), Weibull(1.2), Weibull(2.0), KDistributed(0.5)]
)
def test_independent_backgrounds_have_unit_mean_power(background):
    """Equal mean power is what makes the comparison across models fair."""
    rng = np.random.default_rng(3)
    power = np.abs(background.sample((400_000,), rng)) ** 2
    assert power.mean() == pytest.approx(1.0, rel=0.02)


def test_heavier_shape_means_heavier_tail():
    rng = np.random.default_rng(4)
    assert Weibull(1.0).spikiness(rng) > Weibull(2.0).spikiness(rng)
    assert KDistributed(0.3).spikiness(rng) > KDistributed(5.0).spikiness(rng)


def test_patchy_clutter_concentrates_near_zero_doppler():
    """The defining property: correlated clutter survives coherent processing
    as a ridge. If it spread evenly, the central finding would be an artefact."""
    c = RadarConfig(n_pulses=32, n_range_bins=256)
    power = range_doppler(
        simulate_frame(c, [], PatchyClutter(), np.random.default_rng(5)), c
    )
    per_doppler = power.mean(axis=1)
    centre = c.n_pulses // 2
    ridge = per_doppler[centre - 1 : centre + 2].mean()
    off_ridge = np.concatenate(
        [per_doppler[: centre - 4], per_doppler[centre + 5 :]]
    ).mean()
    assert 10 * np.log10(ridge / off_ridge) > 20.0


def test_patchy_clutter_keeps_a_thermal_floor():
    """Off-ridge cells must still contain noise, or targets placed there are
    detectable at an SNR no real receiver could reach."""
    c = RadarConfig(n_pulses=32, n_range_bins=256)
    power = range_doppler(
        simulate_frame(c, [], PatchyClutter(cnr_db=20.0), np.random.default_rng(6)), c
    )
    thermal = range_doppler(
        simulate_frame(c, [], Thermal(), np.random.default_rng(6)), c
    )
    centre = c.n_pulses // 2
    off_ridge = np.concatenate(
        [power[: centre - 6], power[centre + 7 :]]
    ).mean()
    assert 10 * abs(np.log10(off_ridge / thermal.mean())) < 3.0


# --- CFAR --------------------------------------------------------------------


def test_threshold_multipliers_are_monotonic_in_pfa():
    n = CfarGeometry().n_training_cells
    assert ca_cfar_alpha(n, 1e-6) > ca_cfar_alpha(n, 1e-3)
    assert os_cfar_alpha(n, 186, 1e-6) > os_cfar_alpha(n, 186, 1e-3)


def test_os_cfar_alpha_inverts_its_own_equation():
    """Solved numerically, so verify the solution actually satisfies Pfa."""
    from scipy import special

    n, k, pfa = 248, 186, 1e-4
    alpha = os_cfar_alpha(n, k, pfa)
    log_pfa = (
        np.log(k)
        + special.gammaln(n + 1)
        - special.gammaln(k + 1)
        - special.gammaln(n - k + 1)
        + special.betaln(alpha + n - k + 1, k)
    )
    assert np.exp(log_pfa) == pytest.approx(pfa, rel=1e-6)


@pytest.mark.parametrize("detector", [CaCfar(), OsCfar()])
def test_achieved_pfa_matches_design_in_gaussian_noise(detector):
    """The core calibration claim. If this drifts, every number in the report
    is measured against a threshold that does not mean what it says."""
    c = RadarConfig(n_pulses=32, n_range_bins=256)
    pfa = 1e-3
    flagged = total = 0
    for i in range(12):
        power = range_doppler(
            simulate_frame(c, [], Thermal(), np.random.default_rng(100 + i)), c
        )
        mask, _ = detector.detect(power, pfa)
        flagged += int(mask.sum())
        total += mask.size
    assert 0.5 * pfa < flagged / total < 2.0 * pfa


def test_cfar_finds_a_strong_target_without_false_alarms():
    c = RadarConfig()
    target = Target(range_m=1500.0, velocity_mps=-40.0, snr_db=-5.0)
    power = range_doppler(
        simulate_frame(c, [target], Thermal(), np.random.default_rng(7)), c
    )
    mask, thr = CaCfar().detect(power, 1e-6)
    detections = extract_detections(power, mask, thr, c)
    assert len(detections) == 1
    assert detections[0].range_m == pytest.approx(target.range_m, abs=2 * c.range_bin_spacing_m)


def test_range_only_geometry_is_recognised():
    assert CfarGeometry(guard_doppler=0, train_doppler=0).is_range_only
    assert not CfarGeometry().is_range_only


def test_empty_training_ring_is_rejected():
    with pytest.raises(ValueError):
        _ = CfarGeometry(train_doppler=0, train_range=0).n_training_cells


# --- scoring -----------------------------------------------------------------


def test_scoring_separates_hits_from_false_alarms():
    c = RadarConfig(n_pulses=32, n_range_bins=256)
    target = Target(range_m=900.0, velocity_mps=40.0, snr_db=0.0)
    d_bin, r_bin = truth_bins(c, target)

    mask = np.zeros((c.n_pulses, c.n_range_bins), dtype=bool)
    mask[d_bin, r_bin] = True  # a hit
    mask[(d_bin + 12) % c.n_pulses, r_bin + 60] = True  # a false alarm

    score = score_frame(mask, c, [target])
    assert score.n_detected == 1
    assert score.n_false_cells == 1


def test_cells_near_a_target_are_excluded_from_false_alarms():
    """Target energy spilling into neighbours must not be scored as a false
    alarm - that would flatter every detector equally and hide real failures."""
    c = RadarConfig(n_pulses=32, n_range_bins=256)
    target = Target(range_m=900.0, velocity_mps=40.0, snr_db=0.0)
    d_bin, r_bin = truth_bins(c, target)

    mask = np.zeros((c.n_pulses, c.n_range_bins), dtype=bool)
    mask[d_bin, r_bin + 3] = True  # just outside detect tolerance, inside exclusion

    score = score_frame(mask, c, [target])
    assert score.n_false_cells == 0


# --- reproducibility ---------------------------------------------------------


def test_sweep_is_reproducible():
    """Seeded end to end. A sweep that cannot be re-run is an anecdote."""
    c = RadarConfig(n_pulses=16, n_range_bins=128)
    kwargs = dict(
        backgrounds=[Thermal()],
        detectors=[CaCfar()],
        snr_grid=[-24.0],
        n_trials=3,
        pfa=1e-3,
        seed=99,
        progress=False,
    )
    a = run_sweep(c, **kwargs)
    b = run_sweep(c, **kwargs)
    assert a.points[0].n_detected == b.points[0].n_detected
    assert a.points[0].n_false_cells == b.points[0].n_false_cells
