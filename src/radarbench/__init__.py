"""radarbench - a test and evaluation bench for radar detectors.

Generates radar scenes with known ground truth, runs detectors over them, and
measures where they break.
"""

from .dsp import (
    CaCfar,
    CfarGeometry,
    Detection,
    Detector,
    OsCfar,
    ca_cfar_alpha,
    extract_detections,
    os_cfar_alpha,
    pulse_compress,
    range_doppler,
    to_db,
)
from .metrics import FrameScore, score_frame, truth_bins
from .scene import (
    Background,
    KDistributed,
    PatchyClutter,
    Target,
    Thermal,
    Weibull,
    standard_backgrounds,
)
from .simulate import simulate_frame
from .sweep import SweepPoint, SweepResult, random_targets, run_sweep
from .waveform import RadarConfig

__all__ = [
    "RadarConfig",
    "Target",
    "Background",
    "Thermal",
    "Weibull",
    "KDistributed",
    "PatchyClutter",
    "standard_backgrounds",
    "simulate_frame",
    "pulse_compress",
    "range_doppler",
    "to_db",
    "Detector",
    "CaCfar",
    "OsCfar",
    "CfarGeometry",
    "Detection",
    "extract_detections",
    "ca_cfar_alpha",
    "os_cfar_alpha",
    "FrameScore",
    "score_frame",
    "truth_bins",
    "run_sweep",
    "random_targets",
    "SweepPoint",
    "SweepResult",
]
