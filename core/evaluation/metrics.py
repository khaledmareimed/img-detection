"""
Detection evaluation metrics.

Implements IoU, localization error, and detection accuracy
for comparing predicted bounding boxes against ground truth.
"""

from typing import Optional

import numpy as np

from utils.types import BBox, DetectionResult, EvaluationResult


def compute_iou(bbox1: BBox, bbox2: BBox) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Each box is (x, y, width, height) with top-left origin.
    Returns a value in [0, 1].
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Intersection rectangle
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def compute_localization_error(bbox1: BBox, bbox2: BBox) -> float:
    """Euclidean distance between the centres of two bounding boxes."""
    cx1 = bbox1[0] + bbox1[2] / 2.0
    cy1 = bbox1[1] + bbox1[3] / 2.0
    cx2 = bbox2[0] + bbox2[2] / 2.0
    cy2 = bbox2[1] + bbox2[3] / 2.0
    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def evaluate_detection(
    detection: DetectionResult,
    ground_truth: BBox,
    iou_threshold: float = 0.3,
) -> EvaluationResult:
    """Evaluate a single detection result against ground truth.

    Args:
        detection: The detection result to evaluate.
        ground_truth: Ground-truth bounding box (x, y, w, h).
        iou_threshold: Minimum IoU to consider the detection successful.

    Returns:
        EvaluationResult with IoU, localization error, and pass/fail flag.
    """
    iou = compute_iou(detection.bounding_box, ground_truth)
    loc_error = compute_localization_error(detection.bounding_box, ground_truth)

    return EvaluationResult(
        method_name=detection.method_name,
        iou=iou,
        localization_error=loc_error,
        is_detected=(iou >= iou_threshold),
        detection_result=detection,
    )
