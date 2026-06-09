"""
Benchmark runner.

Runs all detection methods on a given scene/object pair,
collects results, and produces comparison data.
"""

import json
import os
import csv
from typing import List, Dict, Callable, Tuple, Optional

from utils.types import (
    BBox,
    ImageArray,
    DetectionResult,
    EvaluationResult,
    BenchmarkConfig,
)
from core.evaluation.metrics import evaluate_detection
from core.detection.template_matching import detect_template
from core.detection.edge_detection import detect_by_edges
from core.detection.histogram_matching import detect_by_histogram
from core.detection.hybrid_detection import detect_hybrid


def get_all_detection_methods() -> List[
    Tuple[str, Callable[[ImageArray, ImageArray], DetectionResult]]
]:
    """Return a list of (name, function) pairs for all detection methods."""
    return [
        ("Template Matching (NCC)", lambda s, t: detect_template(s, t, method="ncc")),
        ("Template Matching (SSD)", lambda s, t: detect_template(s, t, method="ssd")),
        ("Edge Detection (Canny)", lambda s, t: detect_by_edges(s, t, edge_method="canny")),
        ("Edge Detection (Sobel)", lambda s, t: detect_by_edges(s, t, edge_method="sobel")),
        ("Histogram (Bhattacharyya)", lambda s, t: detect_by_histogram(s, t, metric="bhattacharyya")),
        ("Histogram (Chi-Square)", lambda s, t: detect_by_histogram(s, t, metric="chi_square")),
        ("Hybrid (Hist + NCC)", lambda s, t: detect_hybrid(s, t)),
    ]


def run_benchmark(
    scene: ImageArray,
    template: ImageArray,
    ground_truth: BBox,
    methods: Optional[List[Tuple[str, Callable]]] = None,
    iou_threshold: float = 0.3,
) -> List[EvaluationResult]:
    """Run all detection methods and evaluate each against ground truth.

    Args:
        scene: Full scene image.
        template: Query object image.
        ground_truth: Ground-truth bounding box (x, y, w, h).
        methods: List of (name, detect_fn) pairs. Defaults to all methods.
        iou_threshold: IoU threshold for detection success.

    Returns:
        List of EvaluationResult sorted by IoU (descending).
    """
    if methods is None:
        methods = get_all_detection_methods()

    results: List[EvaluationResult] = []

    for name, detect_fn in methods:
        detection = detect_fn(scene, template)
        evaluation = evaluate_detection(detection, ground_truth, iou_threshold)
        results.append(evaluation)

    results.sort(key=lambda r: r.iou, reverse=True)
    return results


def format_results_table(results: List[EvaluationResult]) -> str:
    """Format evaluation results as an aligned text table."""
    header = f"{'Method':<35} {'IoU':>8} {'Loc Err':>10} {'Conf':>8} {'Time(ms)':>10} {'Pass':>6}"
    lines = [header, "-" * len(header)]

    for r in results:
        d = r.detection_result
        lines.append(
            f"{r.method_name:<35} {r.iou:>8.4f} {r.localization_error:>10.1f} "
            f"{d.confidence:>8.4f} {d.execution_time_ms:>10.1f} "
            f"{'Yes' if r.is_detected else 'No':>6}"
        )

    return "\n".join(lines)


def save_results_csv(
    results: List[EvaluationResult],
    filepath: str,
) -> None:
    """Save evaluation results to a CSV file."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "IoU", "Localization Error",
            "Confidence", "Time (ms)", "Detected",
            "BBox X", "BBox Y", "BBox W", "BBox H",
        ])
        for r in results:
            d = r.detection_result
            bx, by, bw, bh = d.bounding_box
            writer.writerow([
                r.method_name, f"{r.iou:.4f}",
                f"{r.localization_error:.1f}",
                f"{d.confidence:.4f}", f"{d.execution_time_ms:.1f}",
                r.is_detected, bx, by, bw, bh,
            ])


def load_ground_truth(filepath: str) -> Dict[str, Dict[str, BBox]]:
    """Load ground-truth annotations from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    result: Dict[str, Dict[str, BBox]] = {}
    for scene_name, objects in data.items():
        result[scene_name] = {}
        for obj_name, coords in objects.items():
            result[scene_name][obj_name] = tuple(coords)  # type: ignore
    return result
