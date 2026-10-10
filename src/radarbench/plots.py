"""Figures.

One visual identity across every figure, defined once here. The palette is
dark with a solid background rather than transparent, so figures read the same
whether GitHub is in light or dark mode.

Colour carries meaning and is used for nothing else:
    TRUTH    cyan    - where a target actually was
    DETECT   amber   - where a detector said one was
    CA       coral   - the cell-averaging detector
    OS       cyan    - the order-statistic detector
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .dsp import Detection, to_db
from .scene import Background, Target
from .sweep import SweepResult
from .waveform import RadarConfig

BG = "#0e1116"
PANEL = "#141a22"
TEXT = "#d8dee9"
MUTED = "#8b98a9"
GRID = "#242c38"
TRUTH = "#4dd2ff"
DETECT = "#ffb000"
SERIES = ["#ff6b6b", "#4dd2ff", "#9d7bff", "#5ddba0"]
HEATMAP = "magma"


def apply_style() -> None:
    """Install the house style. Call once before plotting."""
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.facecolor": PANEL,
            "legend.edgecolor": GRID,
            "legend.fontsize": 8.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, -0.012, text, ha="center", va="top", color=MUTED, fontsize=8.5)


# --- the hero figure ---------------------------------------------------------


def plot_range_doppler(
    power_map: np.ndarray,
    config: RadarConfig,
    targets: Sequence[Target] = (),
    detections: Sequence[Detection] = (),
    title: str = "Range-Doppler map",
    subtitle: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """The map, with ground truth and detections overlaid.

    Truth is drawn as an open cyan circle, a detection as an amber cross. A
    circle with a cross through it is a hit; a bare circle is a miss; a bare
    cross is a false alarm. The figure is readable without a key.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(9.5, 4.4))

    db = to_db(power_map)
    extent = (
        0.0,
        config.swath_m / 1000.0,
        config.velocity_axis()[0],
        config.velocity_axis()[-1],
    )
    # Clip the floor to the 60th percentile: the bottom of the noise carries no
    # information and stretching the colour range over it hides the targets.
    im = ax.imshow(
        db,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=HEATMAP,
        vmin=float(np.percentile(db, 60)),
        vmax=float(db.max()),
        interpolation="nearest",
    )

    # Both markers are hollow so the cell itself stays visible: a target is only
    # a pixel or two wide, and a filled marker would hide the very thing the
    # figure is evidence of. A square around a circle is a hit, a bare circle a
    # miss, a bare square a false alarm - readable without the legend.
    for t in targets:
        ax.plot(
            t.range_m / 1000.0, t.velocity_mps, "o",
            mfc="none", mec=TRUTH, mew=1.5, ms=13, zorder=3,
        )
    for d in detections:
        ax.plot(
            d.range_m / 1000.0, d.velocity_mps, "s",
            mfc="none", mec=DETECT, mew=1.5, ms=22, zorder=4,
        )

    ax.set_xlabel("range (km)")
    ax.set_ylabel("radial velocity (m/s)")
    ax.set_title(title if not subtitle else f"{title}\n", loc="left")
    if subtitle:
        ax.text(
            0.0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=8.5,
            va="bottom",
        )
    ax.grid(False)

    if created:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.015)
        cb.set_label("power (dB)", color=TEXT, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(GRID)
        ax.legend(
            handles=[
                Line2D([], [], marker="o", mfc="none", mec=TRUTH, mew=1.5, ms=9,
                       ls="none", label="ground truth"),
                Line2D([], [], marker="s", mfc="none", mec=DETECT, mew=1.5, ms=12,
                       ls="none", label="detection"),
            ],
            loc="upper right",
            framealpha=0.9,
        )
    return ax


# --- characterisation figures ------------------------------------------------


def plot_pd_curves(result: SweepResult) -> plt.Figure:
    """Probability of detection against SNR, one panel per background.

    The curve that matters: at a fixed false-alarm rate, how strong does a
    target have to be before the detector reliably sees it. Shaded bands are
    binomial standard error, so a wiggle inside the band is not a result.
    """
    backgrounds = result.backgrounds()
    fig, axes = plt.subplots(
        1, len(backgrounds), figsize=(4.3 * len(backgrounds), 3.6), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, bg in zip(axes, backgrounds, strict=True):
        for i, det in enumerate(result.detectors()):
            pts = result.for_detector(det, bg)
            x = np.array([p.snr_db for p in pts])
            y = np.array([p.pd for p in pts])
            e = np.array([p.pd_stderr for p in pts])
            colour = SERIES[i % len(SERIES)]
            ax.plot(x, y, "-o", color=colour, label=det)
            ax.fill_between(x, y - e, y + e, color=colour, alpha=0.18, lw=0)
        ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        ax.set_title(bg, loc="left")
        ax.set_xlabel("per-sample SNR (dB)")
        ax.set_ylim(-0.03, 1.03)

    axes[0].set_ylabel("probability of detection")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        f"Detection performance at design Pfa = {result.pfa_design:.0e}",
        color=TEXT,
        fontsize=12,
        fontweight="bold",
        x=0.5,
    )
    _caption(
        fig,
        f"{result.n_trials} trials per point, seed {result.seed}. "
        f"Bands are binomial standard error. Dotted line marks Pd = 0.5.",
    )
    fig.tight_layout()
    return fig


def plot_pfa_fidelity(result: SweepResult) -> plt.Figure:
    """Achieved false-alarm rate against the designed one.

    The honest question about any CFAR detector: did the constant-false-alarm
    promise actually hold. Bars at 1.0 kept their contract.
    """
    backgrounds = result.backgrounds()
    detectors = result.detectors()
    width = 0.8 / len(detectors)
    x = np.arange(len(backgrounds))

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for i, det in enumerate(detectors):
        ratios, labels = [], []
        for bg in backgrounds:
            measured, false_cells, eligible = result.pooled_pfa(det, bg)
            ratios.append(measured / result.pfa_design)
            labels.append(f"{false_cells:,}/{eligible:,}")
        pos = x + (i - (len(detectors) - 1) / 2) * width
        bars = ax.bar(pos, ratios, width * 0.92, label=det,
                      color=SERIES[i % len(SERIES)], edgecolor="none")
        for bar, ratio in zip(bars, ratios, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, ratio * 1.08,
                    f"{ratio:.2g}x", ha="center", va="bottom",
                    color=TEXT, fontsize=7.6, fontweight="bold")

    ax.axhline(1.0, color=TRUTH, ls="--", lw=1.4)
    ax.text(-0.46, 1.06, "contract kept", color=TRUTH, fontsize=8.5,
            ha="left", va="bottom")
    # Log scale: the interesting range spans from 0.8x to ~90x, and a linear
    # axis would flatten everything that held into an indistinguishable strip.
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(backgrounds, fontsize=8.5)
    ax.set_ylabel("achieved Pfa / design Pfa")
    ax.set_title("Did the false-alarm contract hold?", loc="left")
    ax.legend(loc="upper left")
    _caption(
        fig,
        f"Design Pfa = {result.pfa_design:.0e}, pooled over every SNR point. "
        "1.0 means the achieved rate matched the design exactly.",
    )
    fig.tight_layout()
    return fig


def plot_backgrounds(
    backgrounds: Sequence[Background], rng: np.random.Generator, n: int = 200_000
) -> plt.Figure:
    """Why the backgrounds differ: the tail, on a log axis.

    All three carry identical mean power. Only the shape of the tail changes,
    which is what isolates the experiment.
    """
    fig, (ax_hist, ax_tail) = plt.subplots(1, 2, figsize=(9.6, 3.5))

    for i, bg in enumerate(backgrounds):
        power = np.abs(bg.sample((n,), rng)) ** 2
        colour = SERIES[i % len(SERIES)]
        ax_hist.hist(
            10 * np.log10(np.maximum(power, 1e-12)),
            bins=180, density=True, histtype="step", lw=1.6,
            color=colour, label=f"{bg.name}  (+{bg.spikiness(rng):.1f} dB tail)",
        )
        srt = np.sort(power)[::-1]
        ax_tail.loglog(
            np.arange(1, n + 1) / n, srt / power.mean(), color=colour, lw=1.6
        )

    ax_hist.set_xlabel("cell power (dB relative to mean)")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("Distribution of cell power", loc="left")
    ax_hist.legend(loc="upper left", fontsize=7.8)
    ax_hist.set_xlim(-40, 25)

    ax_tail.set_xlabel("fraction of cells exceeding")
    ax_tail.set_ylabel("power / mean power")
    ax_tail.set_title("Tail: how often extremes happen", loc="left")

    _caption(
        fig,
        "All three have identical mean power - only tail shape differs. "
        "Tail figure in the legend is the 99.9th percentile over the mean.",
    )
    fig.tight_layout()
    return fig


def plot_background_maps(
    maps: dict[str, np.ndarray], config: RadarConfig
) -> plt.Figure:
    """Range-Doppler maps of different backgrounds, on one shared colour scale.

    This is the figure that explains the whole result. Independent backgrounds
    look like flat noise after processing, however heavy their tail was going
    in. Correlated clutter does not: it survives as a bright ridge at zero
    Doppler, a few bins tall and spanning range in patches. That ridge is what
    a training ring has to cope with.
    """
    names = list(maps)
    fig, axes = plt.subplots(1, len(names), figsize=(5.0 * len(names), 3.5),
                             sharey=True)
    axes = np.atleast_1d(axes)

    stacked = np.concatenate([to_db(m).ravel() for m in maps.values()])
    vmin, vmax = np.percentile(stacked, 50), stacked.max()

    extent = (0.0, config.swath_m / 1000.0,
              config.velocity_axis()[0], config.velocity_axis()[-1])
    for ax, name in zip(axes, names, strict=True):
        im = ax.imshow(to_db(maps[name]), aspect="auto", origin="lower",
                       extent=extent, cmap=HEATMAP, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(name, loc="left")
        ax.set_xlabel("range (km)")
        ax.grid(False)
    axes[0].set_ylabel("radial velocity (m/s)")

    cb = fig.colorbar(im, ax=axes.tolist(), pad=0.012)
    cb.set_label("power (dB)", color=TEXT, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(GRID)
    _caption(
        fig,
        "Shared colour scale, no targets present. Independent backgrounds are "
        "Gaussianised by coherent processing; correlated clutter is not.",
    )
    return fig


def plot_threshold_cut(
    power_map: np.ndarray,
    thresholds: dict[str, np.ndarray],
    config: RadarConfig,
    doppler_bin: int,
    targets: Sequence[Target] = (),
) -> plt.Figure:
    """One velocity slice: signal against each detector's adaptive threshold.

    This is the clearest picture of what CFAR does. The thresholds are not
    flat lines - they rise and fall with the local background, which is the
    entire idea.
    """
    rng_axis = config.range_axis() / 1000.0
    fig, ax = plt.subplots(figsize=(9.6, 3.5))

    ax.plot(rng_axis, to_db(power_map[doppler_bin]), color=MUTED, lw=1.0,
            label="received power")
    for i, (name, thr) in enumerate(thresholds.items()):
        ax.plot(rng_axis, to_db(thr[doppler_bin]), lw=1.6,
                color=SERIES[i % len(SERIES)], label=f"{name} threshold")

    velocity = config.velocity_axis()[doppler_bin]
    for t in targets:
        if abs(t.velocity_mps - velocity) <= config.velocity_resolution_mps:
            ax.axvline(t.range_m / 1000.0, color=TRUTH, ls="--", lw=1.1, alpha=0.8)

    ax.set_xlabel("range (km)")
    ax.set_ylabel("power (dB)")
    ax.set_title(
        f"Adaptive thresholds along the {velocity:+.1f} m/s velocity cut", loc="left"
    )
    ax.legend(loc="upper right", ncol=3)
    _caption(
        fig,
        "The thresholds track the local background instead of sitting flat. "
        "Dashed cyan lines mark true targets in this cut.",
    )
    fig.tight_layout()
    return fig
