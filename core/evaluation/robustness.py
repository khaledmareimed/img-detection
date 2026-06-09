"""
Robustness testing utilities.

Applies geometric and photometric transformations to query objects
for evaluating detector resilience.
"""

from typing import Callable, Dict, List

import cv2
import numpy as np

from utils.types import (
    BBox,
    DetectionResult,
    EvaluationResult,
    ImageArray,
    RobustnessTestResult,
)
from core.evaluation.metrics import evaluate_detection


def apply_rotation(image: ImageArray, angle: float) -> ImageArray:
    """Rotate the image by *angle* degrees around its centre."""
    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)

    cos_a = np.abs(matrix[0, 0])
    sin_a = np.abs(matrix[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    matrix[0, 2] += (new_w - w) / 2.0
    matrix[1, 2] += (new_h - h) / 2.0

    return cv2.warpAffine(image, matrix, (new_w, new_h))


def apply_scale(image: ImageArray, factor: float) -> ImageArray:
    """Scale the image by the given factor."""
    h, w = image.shape[:2]
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def apply_brightness(image: ImageArray, offset: int) -> ImageArray:
    """Shift pixel intensities by *offset* (clamped to [0, 255])."""
    result = image.astype(np.int16) + offset
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_contrast(image: ImageArray, factor: float) -> ImageArray:
    """Scale pixel intensities around the mean by *factor*."""
    mean = image.mean()
    result = (image.astype(np.float64) - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_noise(image: ImageArray, sigma: float) -> ImageArray:
    """Add Gaussian noise with standard deviation *sigma*."""
    noise = np.random.normal(0, sigma, image.shape)
    result = image.astype(np.float64) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Robustness test runner
# ---------------------------------------------------------------------------

def get_default_transformations() -> Dict[str, List]:
    """Return the default set of transformations and their parameters."""
    return {
        "rotation": [15, 30, 45, 90],
        "scale": [0.5, 0.75, 1.25, 1.5],
        "brightness": [-60, -30, 30, 60],
        "contrast": [0.5, 0.75, 1.5, 2.0],
        "noise": [10, 25, 50],
    }


_TRANSFORM_FNS = {
    "rotation": apply_rotation,
    "scale": apply_scale,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "noise": apply_noise,
}


def run_robustness_suite(
    scene: ImageArray,
    template: ImageArray,
    ground_truth: BBox,
    detect_fn: Callable[[ImageArray, ImageArray], DetectionResult],
    transformations: Dict[str, List] = None,
    iou_threshold: float = 0.3,
) -> List[RobustnessTestResult]:
    """Run all transformations on the *template* and evaluate detection.

    Transformations are applied to the query object only (per spec).

    Args:
        scene: The static scene image.
        template: The original query object image.
        ground_truth: Ground-truth bounding box in the scene.
        detect_fn: A detection function ``(scene, template) -> DetectionResult``.
        transformations: Dict mapping transform name to list of parameter values.
        iou_threshold: IoU threshold for detection success.

    Returns:
        List of RobustnessTestResult, one per transformation variant.
    """
    if transformations is None:
        transformations = get_default_transformations()

    results: List[RobustnessTestResult] = []

    for transform_name, params in transformations.items():
        fn = _TRANSFORM_FNS.get(transform_name)
        if fn is None:
            continue

        for param in params:
            transformed = fn(template, param)
            detection = detect_fn(scene, transformed)
            evaluation = evaluate_detection(
                detection, ground_truth, iou_threshold
            )
            tag = f"{transform_name}_{param}"
            results.append(
                RobustnessTestResult(
                    method_name=detection.method_name,
                    transformation=tag,
                    evaluation=evaluation,
                )
            )

    return results
