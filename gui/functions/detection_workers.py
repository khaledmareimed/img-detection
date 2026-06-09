"""
Background detection workers.

Provides factory functions that create threaded workers for
single detection, benchmark comparison, and robustness testing.
"""

import threading
import tkinter as tk
from tkinter import messagebox
from typing import Callable, Optional, List

from utils.types import ImageArray, EvaluationResult, BBox
from core.evaluation.metrics import evaluate_detection
from core.evaluation.benchmark import (
    get_all_detection_methods,
    run_benchmark,
)
from core.evaluation.robustness import run_robustness_suite


def resolve_detection_fn(
    method_name: str,
) -> Callable:
    """Find a detection function by its display name."""
    methods = get_all_detection_methods()
    for name, fn in methods:
        if name == method_name:
            return fn
    return methods[0][1]


def run_single_detection(
    root: tk.Tk,
    scene: ImageArray,
    template: ImageArray,
    method_name: str,
    ground_truth: Optional[BBox],
    on_result: Callable[[EvaluationResult], None],
) -> None:
    """Run a single detection method in a background thread."""

    def _worker() -> None:
        try:
            detect_fn = resolve_detection_fn(method_name)
            detection = detect_fn(scene, template)

            if ground_truth:
                evaluation = evaluate_detection(detection, ground_truth)
            else:
                evaluation = EvaluationResult(
                    method_name=detection.method_name,
                    iou=0.0, localization_error=0.0,
                    is_detected=False, detection_result=detection,
                )
            root.after(0, lambda: on_result(evaluation))
        except Exception as e:
            msg = str(e)
            root.after(0, lambda m=msg: messagebox.showerror("Error", m))

    threading.Thread(target=_worker, daemon=True).start()


def run_all_detection(
    root: tk.Tk,
    scene: ImageArray,
    template: ImageArray,
    ground_truth: Optional[BBox],
    on_results: Callable[[List[EvaluationResult]], None],
) -> None:
    """Run all detection methods in a background thread."""

    def _worker() -> None:
        try:
            gt = ground_truth or (0, 0, 0, 0)
            results = run_benchmark(scene, template, gt)
            root.after(0, lambda: on_results(results))
        except Exception as e:
            msg = str(e)
            root.after(0, lambda m=msg: messagebox.showerror("Error", m))

    threading.Thread(target=_worker, daemon=True).start()


def run_robustness(
    root: tk.Tk,
    scene: ImageArray,
    template: ImageArray,
    ground_truth: BBox,
    method_name: str,
    on_results: Callable,
    on_status: Callable[[str], None],
) -> None:
    """Run robustness tests in a background thread."""

    def _worker() -> None:
        try:
            detect_fn = resolve_detection_fn(method_name)
            results = run_robustness_suite(
                scene, template, ground_truth, detect_fn,
            )
            root.after(0, lambda: on_results(results))
            passed = sum(1 for r in results if r.evaluation.is_detected)
            root.after(
                0, lambda: on_status(
                    f"Robustness: {passed}/{len(results)} passed"
                )
            )
        except Exception as e:
            msg = str(e)
            root.after(0, lambda m=msg: messagebox.showerror("Error", m))

    threading.Thread(target=_worker, daemon=True).start()