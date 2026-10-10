"""Six experiments that demonstrate the ideas this bench rests on.

    python scripts/explain.py          # list them
    python scripts/explain.py 3        # run one
    python scripts/explain.py all      # run all six

Each prints its own evidence. Nothing here is asserted that the numbers on
screen do not show, so changing a parameter and re-running is the fastest way
to check whether you have actually understood a claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radarbench import (  # noqa: E402
    Background,
    CaCfar,
    CfarGeometry,
    KDistributed,
    PatchyClutter,
    RadarConfig,
    Target,
    Thermal,
    pulse_compress,
    range_doppler,
    simulate_frame,
)

GAUSSIAN_SPIKINESS_DB = 8.4  # 99.9th percentile over mean, exponential power


class _Silent(Background):
    """No background at all, so a target's own strength can be measured."""

    name = "none"

    def sample(self, shape, rng):  # noqa: ARG002
        return np.zeros(shape, dtype=complex)


def banner(n: int, title: str, question: str) -> None:
    print()
    print("=" * 74)
    print(f"  {n}. {title}")
    print(f"     {question}")
    print("=" * 74)


def verdict(text: str) -> None:
    print()
    for line in text.strip().splitlines():
        print(f"  >> {line.strip()}")
    print()


# -----------------------------------------------------------------------------


def exp1_why_a_chirp() -> None:
    banner(1, "Why a chirp instead of a short pulse",
           "Can one pulse be both strong enough and short enough?")

    c = RadarConfig()
    # A short pulse with the SAME bandwidth lasts about 1/B.
    short_len = max(int(round(c.sample_rate_hz / c.bandwidth_hz)), 1)
    chirp_len = c.n_pulse_samples

    print(f"\n  bandwidth {c.bandwidth_hz/1e6:.0f} MHz fixes range resolution at "
          f"{c.range_resolution_m:.1f} m for BOTH options.\n")
    print(f"  short pulse : {short_len:3d} samples  "
          f"({short_len/c.sample_rate_hz*1e6:.2f} us)")
    print(f"  chirp       : {chirp_len:3d} samples  "
          f"({c.pulse_width_s*1e6:.2f} us)")

    # Energy is what sets detection range; amplitude is capped by the transmitter.
    energy_ratio_db = 10 * np.log10(chirp_len / short_len)
    print("\n  both transmit at the same peak amplitude, so energy scales with")
    print(f"  duration: the chirp carries {energy_ratio_db:.1f} dB more.")
    print(f"  time-bandwidth product B*T = {c.time_bandwidth_product:.0f}")
    print(f"  compression gain 10*log10({c.n_pulse_samples} samples) = "
          f"{c.compression_gain_db:.1f} dB   (see experiment 2)")

    verdict(f"""
        Same resolution, {energy_ratio_db:.0f} dB more energy. That is the whole trick:
        resolution comes from BANDWIDTH, detection range comes from ENERGY,
        and a chirp lets you buy them separately instead of trading one
        against the other.
    """)


def exp2_pulse_compression() -> None:
    banner(2, "Pulse compression",
           "Where does the gain actually show up?")

    c = RadarConfig()
    snr_db = -25.0
    target = Target(range_m=1500.0, velocity_mps=40.0, snr_db=snr_db)
    rng = np.random.default_rng(1)

    # Measure signal and noise separately. Taking the peak of signal+noise
    # would just report the largest noise spike when the target is weak,
    # which is exactly the regime of interest.
    signal = simulate_frame(c, [target], _Silent(), rng)
    noise = simulate_frame(c, [], Thermal(), rng)

    def stage_snr(sig: np.ndarray, noi: np.ndarray, is_power: bool = False) -> float:
        s = sig if is_power else np.abs(sig) ** 2
        n = noi if is_power else np.abs(noi) ** 2
        return 10 * np.log10(s.max() / n.mean())

    raw = stage_snr(signal, noise)
    comp = stage_snr(pulse_compress(signal, c), pulse_compress(noise, c))
    full = stage_snr(range_doppler(signal, c), range_doppler(noise, c), is_power=True)

    print(f"\n  target planted at {target.range_m:.0f} m, "
          f"{snr_db:+.0f} dB per raw sample")
    print("  (signal and noise generated separately, so these are the target's")
    print("   own strength and not the biggest noise spike nearby)\n")
    print(f"  peak SNR in the raw frame        {raw:+6.1f} dB   "
          f"{'<- buried' if raw < 6 else ''}")
    print(f"  after pulse compression          {comp:+6.1f} dB   (+{comp-raw:.1f})")
    print(f"  after the Doppler FFT too        {full:+6.1f} dB   (+{full-comp:.1f})")
    print(f"\n  predicted: {c.compression_gain_db:.1f} dB compression "
          f"+ {c.integration_gain_db:.1f} dB integration "
          f"= {c.processing_gain_db:.1f} dB total")
    print(f"  measured total: {full - raw:.1f} dB")
    print(f"\n  note the compression figure is 10*log10({c.n_pulse_samples} samples)"
          f" = {c.compression_gain_db:.1f} dB,")
    print(f"  not the textbook 10*log10(B*T) = "
          f"{10*np.log10(c.time_bandwidth_product):.1f} dB. B*T assumes the receiver")
    print("  samples critically at F_s = B; this one samples at 2B, so the matched")
    print("  filter has twice as many samples to add up.")

    verdict(f"""
        A target {raw:+.0f} dB below the noise ends up {full:+.0f} dB above it, with no
        extra transmit power. Compression gathers the chirp's spread-out
        energy into one spike; the Doppler FFT then adds the pulses
        coherently while their noise adds incoherently.
    """)


def exp3_doppler_is_phase() -> None:
    banner(3, "Velocity is a phase rotation",
           "How can centimetres of movement be measurable?")

    c = RadarConfig()
    v = 40.0
    target = Target(range_m=1500.0, velocity_mps=v, snr_db=40.0)  # loud, to see phase
    frame = simulate_frame(c, [target], Thermal(), np.random.default_rng(3))
    compressed = pulse_compress(frame, c)
    r_bin = int(round(target.range_m / c.range_bin_spacing_m))

    moved_per_pulse = v * c.pri_s
    print(f"\n  target closing at {v:.0f} m/s, PRI {c.pri_s*1e6:.0f} us")
    print(f"  distance moved between pulses: {moved_per_pulse*100:.3f} cm")
    print(f"  range bin size:                {c.range_bin_spacing_m*100:.0f} cm "
          f"-> it never leaves its bin\n")
    print(f"  but the wavelength is only {c.wavelength_m*100:.1f} cm, and the path")
    print("  changes by TWICE the movement (out and back), so the echo phase")
    print(f"  turns by 2 * {moved_per_pulse*100:.3f} / {c.wavelength_m*100:.1f} "
          f"= {2*moved_per_pulse/c.wavelength_m:.3f} of a cycle each pulse.\n")

    phases = np.angle(compressed[:8, r_bin])
    steps = np.degrees(np.diff(np.unwrap(phases)))
    print("  measured phase of the echo, pulse by pulse:")
    for i, p in enumerate(phases):
        step = f"   step {steps[i-1]:+7.1f} deg" if i else ""
        print(f"     pulse {i}  {np.degrees(p):+8.1f} deg{step}")

    predicted = 360 * 2 * v * c.pri_s / c.wavelength_m
    measured_v = c.velocity_axis()[
        int(np.argmax(np.abs(np.fft.fftshift(np.fft.fft(compressed[:, r_bin])))))
    ]
    print(f"\n  predicted step {predicted:+.1f} deg/pulse, "
          f"measured {steps.mean():+.1f} deg/pulse")
    print(f"  FFT across pulses recovers {measured_v:+.1f} m/s "
          f"(truth {v:+.1f}, bin size {c.velocity_resolution_mps:.1f})")

    verdict("""
        The target moves far too little to change range bin, but the radar
        measures distance in wavelengths, and a wavelength is 10 cm. A steady
        phase step across pulses is a sinusoid in slow time, so an FFT reads
        the velocity straight off.
    """)


def exp4_why_not_a_fixed_threshold() -> None:
    banner(4, "Why a fixed threshold cannot work",
           "What breaks if you just pick a brightness cutoff?")

    c = RadarConfig(n_pulses=32, n_range_bins=256)
    power = range_doppler(
        simulate_frame(c, [], PatchyClutter(), np.random.default_rng(4)), c
    )
    centre = c.n_pulses // 2
    busy = np.s_[centre - 1 : centre + 2, :]   # the clutter ridge
    quiet = np.s_[: centre - 6, :]             # clear sky

    print("\n  one map, two regions:")
    print(f"    clutter ridge   mean power {10*np.log10(power[busy].mean()):+6.1f} dB")
    print(f"    clear sky       mean power {10*np.log10(power[quiet].mean()):+6.1f} dB")
    print(f"    difference      {10*np.log10(power[busy].mean()/power[quiet].mean()):.0f} dB\n")

    # Tune a fixed threshold so the quiet region behaves, then see the ridge.
    thr_quiet = np.percentile(power[quiet], 99.9)
    print("  pick a fixed threshold that gives 0.1% false alarms in clear sky:")
    print(f"    false-alarm rate in clear sky   {(power[quiet] > thr_quiet).mean():.4%}")
    print(f"    same threshold, on the ridge    {(power[busy] > thr_quiet).mean():.4%}"
          f"   <- unusable")

    thr_busy = np.percentile(power[busy], 99.9)
    print("\n  now tune it for the ridge instead:")
    print(f"    false-alarm rate on the ridge   {(power[busy] > thr_busy).mean():.4%}")
    print(f"    same threshold, in clear sky    {(power[quiet] > thr_busy).mean():.4%}"
          f"   <- blind")
    print(f"    (that threshold is "
          f"{10*np.log10(thr_busy/thr_quiet):.0f} dB above the clear-sky one, so a "
          f"real target\n     out there has to be {10*np.log10(thr_busy/thr_quiet):.0f} "
          f"dB stronger than it needs to be)")

    print("\n  CFAR instead, one rule and no region tuning (design Pfa 0.1000%):")
    geometries = [
        ("2D ring", CfarGeometry()),
        ("range-only", CfarGeometry(guard_doppler=0, train_doppler=0,
                                    guard_range=2, train_range=16)),
    ]
    for tag, geom in geometries:
        mask, _ = CaCfar(geom).detect(power, 1e-3)
        note = "  <- still broken" if mask[busy].mean() > 0.01 else ""
        print(f"    {tag:<12} ridge {mask[busy].mean():8.4%}   "
              f"clear sky {mask[quiet].mean():8.4%}{note}")

    verdict("""
        No single number works in both places at once. CFAR does not pick a
        better number; it picks a different number for every cell, from that
        cell's own neighbourhood.

        But note the second line: WHICH neighbourhood matters as much as the
        idea itself. A ring that reaches across the ridge averages clutter
        cells into the background of cells that are nowhere near it. Keeping
        the training cells inside the cell's own Doppler bin fixes it. That
        is the main finding of this whole project, visible in one experiment.
    """)


def exp5_guard_cells() -> None:
    banner(5, "What guard cells are for",
           "Can a target hide itself?")

    c = RadarConfig()
    guarded = CfarGeometry()
    unguarded = CfarGeometry(guard_doppler=0, guard_range=0)

    print("\n  A cell's threshold comes from the ring around it. With no guard")
    print("  cells that ring touches the target itself, so the target helps set")
    print("  the bar it then has to clear.\n")
    print(f"  {'target':>8} | {'echo':>9} | {'thr, 2 guards':>14} | "
          f"{'thr, 0 guards':>14} | inflation")
    print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*14}-+-{'-'*14}-+----------")

    for snr in (-20.0, -10.0, 0.0, 10.0):
        target = Target(range_m=1500.0, velocity_mps=-40.0, snr_db=snr)
        power = range_doppler(
            simulate_frame(c, [target], Thermal(), np.random.default_rng(5)), c
        )
        d_bin = int(np.argmin(np.abs(c.velocity_axis() - target.velocity_mps)))
        r_bin = int(round(target.range_m / c.range_bin_spacing_m))

        _, thr_g = CaCfar(guarded).detect(power, 1e-5)
        _, thr_u = CaCfar(unguarded).detect(power, 1e-5)
        g_db = 10 * np.log10(thr_g[d_bin, r_bin])
        u_db = 10 * np.log10(thr_u[d_bin, r_bin])
        print(f"  {snr:>+6.0f} dB | {10*np.log10(power[d_bin, r_bin]):>6.1f} dB | "
              f"{g_db:>11.1f} dB | {u_db:>11.1f} dB | {u_db - g_db:>+6.1f} dB")

    verdict("""
        The guarded threshold barely moves - it is measuring the background,
        which is the same in every row. The unguarded one climbs with the
        target, because the target is in its own training ring.

        A loud target still clears its own inflated bar, so this rarely
        changes its own verdict. The damage is done to its NEIGHBOURS: that
        inflated background estimate is shared by every nearby cell, so a
        weaker target beside a strong one is the one that disappears. Guard
        cells keep the estimate describing the background and not the target.
    """)


def exp6_why_correlation_matters() -> None:
    banner(6, "Why correlated clutter is the only one that matters",
           "Heavy tails go in - do they come out?")

    c = RadarConfig(n_pulses=32, n_range_bins=256)
    rng = np.random.default_rng(6)

    def spikiness_db(power: np.ndarray) -> float:
        return float(10 * np.log10(np.percentile(power, 99.9) / power.mean()))

    print("\n  'spikiness' = 99.9th percentile power over mean power.")
    print(f"  pure Gaussian noise sits at about {GAUSSIAN_SPIKINESS_DB:.1f} dB.\n")
    print(f"  {'background':<28} {'raw samples':>12} {'after processing':>18}")
    print(f"  {'-'*28} {'-'*12} {'-'*18}")

    for bg in (Thermal(), KDistributed(0.5), PatchyClutter()):
        raw = spikiness_db(np.abs(bg.sample((64, 4096), rng)) ** 2)
        processed = spikiness_db(range_doppler(simulate_frame(c, [], bg, rng), c))
        note = ""
        if processed < GAUSSIAN_SPIKINESS_DB + 3:
            note = "  <- flattened to Gaussian"
        elif processed > GAUSSIAN_SPIKINESS_DB + 10:
            note = "  <- survives"
        print(f"  {bg.name:<28} {raw:>9.1f} dB {processed:>15.1f} dB{note}")

    print(f"\n  pulse compression sums {c.n_pulse_samples} samples, the Doppler FFT "
          f"sums {c.n_pulses}")
    print("  pulses. Adding that many INDEPENDENT values drives the result toward")
    print("  a Gaussian - the central limit theorem - whatever the input looked like.")

    verdict("""
        A heavy tail only reaches the detector if it is CORRELATED, because
        correlated values do not average out. That is why terrain and sea
        clutter are a real problem and why simply drawing spiky independent
        numbers is not a model of them. It is also the mistake this project
        made first: the original clutter model was independent, measured as
        harmless, and had to be rebuilt.
    """)


EXPERIMENTS = [
    ("Why a chirp instead of a short pulse", exp1_why_a_chirp),
    ("Pulse compression: where the gain shows up", exp2_pulse_compression),
    ("Velocity is a phase rotation", exp3_doppler_is_phase),
    ("Why a fixed threshold cannot work", exp4_why_not_a_fixed_threshold),
    ("What guard cells are for", exp5_guard_cells),
    ("Why correlated clutter is the only one that matters", exp6_why_correlation_matters),
]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        print("  experiments:")
        for i, (title, _) in enumerate(EXPERIMENTS, start=1):
            print(f"    {i}. {title}")
        return 0

    if argv[0] == "all":
        for _, fn in EXPERIMENTS:
            fn()
        return 0

    try:
        index = int(argv[0])
        _, fn = EXPERIMENTS[index - 1]
    except (ValueError, IndexError):
        print(f"no experiment {argv[0]!r}; run without arguments to list them")
        return 1
    fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
