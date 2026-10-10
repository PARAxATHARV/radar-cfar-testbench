"""One frame, end to end, with figures.

    python scripts/demo.py

Writes reports/figures/range_doppler.png and reports/figures/threshold_cut.png
and prints truth against detections.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from radarbench import (  # noqa: E402
    CaCfar,
    OsCfar,
    RadarConfig,
    Target,
    Thermal,
    extract_detections,
    range_doppler,
    simulate_frame,
)
from radarbench.plots import apply_style, plot_range_doppler, plot_threshold_cut  # noqa: E402

FIGURES = ROOT / "reports" / "figures"

SCENE = [
    Target(range_m=640.0, velocity_mps=62.0, snr_db=-15.0),
    Target(range_m=1480.0, velocity_mps=-34.0, snr_db=-18.0),
    Target(range_m=1520.0, velocity_mps=48.0, snr_db=-16.0),
    Target(range_m=2810.0, velocity_mps=-88.0, snr_db=-13.0),
    Target(range_m=3250.0, velocity_mps=21.0, snr_db=-20.0),
]
PFA = 1e-5


def main() -> int:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    config = RadarConfig()
    print(config.describe(), "\n")

    rng = np.random.default_rng(20261010)
    frame = simulate_frame(config, SCENE, Thermal(), rng)
    power_map = range_doppler(frame, config)

    detectors = [CaCfar(), OsCfar()]
    results, thresholds = {}, {}
    for det in detectors:
        mask, thr = det.detect(power_map, PFA)
        results[det.name] = extract_detections(power_map, mask, thr, config)
        thresholds[det.name] = thr

    for name, dets in results.items():
        print(f"{name}: {len(dets)} detections at Pfa {PFA:.0e}")
        for d in dets:
            print(
                f"   R = {d.range_m:7.1f} m   v = {d.velocity_mps:+7.1f} m/s"
                f"   {d.snr_over_threshold_db:+5.1f} dB over threshold"
            )
        print()

    primary = detectors[0]
    plot_range_doppler(
        power_map,
        config,
        targets=SCENE,
        detections=results[primary.name],
        title="Five targets in thermal noise",
        subtitle=(
            f"{config.n_pulses} pulses x {config.n_range_bins} range bins   |   "
            f"{primary.name} at Pfa {PFA:.0e}   |   "
            f"processing gain {config.processing_gain_db:.1f} dB"
        ),
    )
    out = FIGURES / "range_doppler.png"
    plt.savefig(out)
    plt.close("all")
    print(f"wrote {out.relative_to(ROOT)}")

    strongest = max(SCENE, key=lambda t: t.snr_db)
    d_bin = int(np.argmin(np.abs(config.velocity_axis() - strongest.velocity_mps)))
    fig = plot_threshold_cut(power_map, thresholds, config, d_bin, targets=SCENE)
    out = FIGURES / "threshold_cut.png"
    fig.savefig(out)
    plt.close("all")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
