"""
Type definitions for the DIP Object Detection project.

Defines all shared data structures used across the detection pipeline,
evaluation framework, and GUI components.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# Bounding box: (x, y, width, height) — top-left corner origin
BBox = Tuple[int, int, int, int]

# Typed image array alias for readability
ImageArray = NDArray[np.uint8]


# ---------------------------------------------------------------------------
# Detection Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result produced by a single detection method."""

    method_name: str
    bounding_box: BBox
    confidence: float  # 0.0 – 1.0
    execution_time_ms: float
    similarity_map: Optional[NDArray[np.float64]] = None


@dataclass
class EvaluationResult:
    """Evaluation of a detection result against ground truth."""

    method_name: str
    iou: float
    localization_error: float  # Euclidean distance between centers (px)
    is_detected: bool  # True if IoU >= threshold
    detection_result: DetectionResult


@dataclass
class RobustnessTestResult:
    """Result of testing a method under a specific transformation."""

    method_name: str
    transformation: str  # e.g. "rotation_30", "scale_0.5"
    evaluation: EvaluationResult


# ---------------------------------------------------------------------------
# Configuration Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""

    apply_histogram_eq: bool = False
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    apply_gaussian_blur: bool = False
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    apply_median_filter: bool = False
    median_kernel_size: int = 5
    apply_contrast_stretch: bool = False
    contrast_low_pct: float = 2.0
    contrast_high_pct: float = 98.0
    apply_background_removal: bool = False
    apply_morphology: bool = False
    morph_operation: str = "opening"  # opening | closing | erosion | dilation
    morph_kernel_size: int = 5


@dataclass
class BenchmarkConfig:
    """Configuration for running benchmarks and robustness tests."""

    iou_threshold: float = 0.3
    scales: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5]
    )
    rotation_angles: List[float] = field(
        default_factory=lambda: [15.0, 30.0, 45.0, 90.0]
    )
    brightness_offsets: List[int] = field(
        default_factory=lambda: [-60, -30, 30, 60]
    )
    contrast_factors: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.5, 2.0]
    )
    noise_sigmas: List[float] = field(
        default_factory=lambda: [10.0, 25.0, 50.0]
    )
