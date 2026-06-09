"""
Main GUI application — orchestrates all panels with a responsive
grid-based layout using CustomTkinter.
"""

import os
import sys
from tkinter import messagebox, filedialog
from typing import Optional, List



import customtkinter as ctk

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.types import ImageArray, EvaluationResult, BBox
from utils.image_io import load_image
from gui.functions.gui_helpers import COLORS, FONT_BODY, FONT_SMALL
from gui.functions.preprocessing_pipeline import apply_preprocessing_pipeline
from gui.functions.detection_workers import (
    run_single_detection, run_all_detection, run_robustness,
)
from gui.components.image_panel import ImagePanel
from gui.components.control_panel import ControlPanel
from gui.components.results_panel import ResultsPanel
from gui.components.metrics_panel import MetricsPanel
from gui.components.preprocessing_panel import PreprocessingPanel
from core.evaluation.benchmark import save_results_csv, load_ground_truth


class DIPApp:
    """Main application for DIP Object Detection."""

    def __init__(self) -> None:
        self.root = ctk.CTk()
        self.root.title("DIP Object Detector — An-Najah National University")
        self.root.geometry("1360x820")
        self.root.minsize(1000, 650)

        self._project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self._scene: Optional[ImageArray] = None
        self._template: Optional[ImageArray] = None
        self._ground_truth: Optional[BBox] = None
        self._gt_data: dict = {}
        self._scene_name: str = ""
        self._last_results: List[EvaluationResult] = []

        self._load_annotations()
        self._build_layout()

    def _build_layout(self) -> None:
        # Main grid: 3 columns, 2 rows + status bar
        self.root.grid_rowconfigure(0, weight=3)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_columnconfigure(0, weight=0, minsize=280)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=0, minsize=310)

        # Left: Control panel
        self._control = ControlPanel(
            self.root,
            on_load_scene=self._load_scene,
            on_load_object=self._load_object,
            on_run_detection=self._run_detection,
            on_run_all=self._run_all,
            on_run_robustness=self._run_robustness,
            on_export=self._export_results,
            on_set_gt=self._start_gt_drawing,
        )
        self._control.grid(row=0, column=0, rowspan=2, sticky="nsew",
                           padx=(4, 2), pady=4)

        # Centre: Image panel
        self._image_panel = ImagePanel(self.root)
        self._image_panel.grid(row=0, column=1, sticky="nsew",
                               padx=2, pady=(4, 2))

        # Right: Results panel
        self._results = ResultsPanel(self.root)
        self._results.grid(row=0, column=2, rowspan=2, sticky="nsew",
                           padx=(2, 4), pady=4)

        # Bottom centre: Preprocessing + Metrics
        bottom = ctk.CTkFrame(self.root, fg_color="transparent")
        bottom.grid(row=1, column=1, sticky="nsew", padx=2, pady=(2, 4))
        bottom.grid_columnconfigure(0, weight=0)
        bottom.grid_columnconfigure(1, weight=1)
        bottom.grid_rowconfigure(0, weight=1)

        self._preproc_panel = PreprocessingPanel(bottom)
        self._preproc_panel.grid(row=0, column=0, sticky="nsew",
                                  padx=(0, 2))

        self._metrics = MetricsPanel(bottom)
        self._metrics.grid(row=0, column=1, sticky="nsew", padx=(2, 0))

        # Status bar
        self._status_var = ctk.StringVar(
            value="Ready — Load images to begin"
        )
        ctk.CTkLabel(
            self.root, textvariable=self._status_var,
            font=FONT_SMALL, text_color=COLORS["text_dim"],
            fg_color=COLORS["bg_card"], corner_radius=0,
            anchor="w", padx=12, height=28,
        ).grid(row=2, column=0, columnspan=3, sticky="ew")

    def _load_annotations(self) -> None:
        gt_path = os.path.join(
            self._project_root, "data", "ground_truth", "annotations.json"
        )
        if os.path.exists(gt_path):
            try:
                self._gt_data = load_ground_truth(gt_path)
            except Exception:
                self._gt_data = {}

    def _load_scene(self, path: str) -> None:
        try:
            self._scene = load_image(path)
            self._image_panel.set_scene(self._scene)
            self._scene_name = os.path.splitext(os.path.basename(path))[0]
            self._ground_truth = None
            self._update_gt_label()
            s = f"Scene loaded: {self._scene_name}"
            if self._scene_name in self._gt_data:
                s += " (GT available)"
            self._status_var.set(s)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load scene:\n{e}")

    def _load_object(self, path: str) -> None:
        try:
            self._template = load_image(path)
            self._image_panel.set_object(self._template)
            obj_name = os.path.splitext(os.path.basename(path))[0]
            self._ground_truth = None
            if self._scene_name in self._gt_data:
                gt_set = self._gt_data[self._scene_name]
                if obj_name in gt_set:
                    gt = gt_set[obj_name]
                    if any(v > 0 for v in gt):
                        self._ground_truth = gt
            self._update_gt_label()
            tag = " + GT" if self._ground_truth else ""
            self._status_var.set(f"Object loaded: {obj_name}{tag}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load object:\n{e}")

    def _start_gt_drawing(self) -> None:
        if self._scene is None:
            messagebox.showwarning("No Scene", "Load a scene image first.")
            return
        self._status_var.set("Click and drag on the scene to draw GT box")
        self._image_panel.start_gt_drawing(self._on_gt_drawn)

    def _on_gt_drawn(self, bbox: BBox) -> None:
        self._ground_truth = bbox
        self._update_gt_label()
        self._image_panel.show_ground_truth(bbox)
        self._status_var.set(
            f"GT set: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")

    def _update_gt_label(self) -> None:
        if self._ground_truth:
            x, y, w, h = self._ground_truth
            self._control.gt_label_var.set(f"GT: ({x}, {y}, {w}, {h})")
        else:
            self._control.gt_label_var.set("No GT set")

    def _run_detection(self, method_name: str) -> None:
        if not self._scene_and_template_loaded():
            return
        self._status_var.set(f"Running {method_name}...")
        self.root.update()
        tmpl = self._apply_preprocessing(self._template)
        run_single_detection(
            self.root, self._scene, tmpl, method_name,
            self._ground_truth, self._show_single,
        )

    def _run_all(self) -> None:
        if not self._scene_and_template_loaded():
            return
        self._status_var.set("Running all methods...")
        self.root.update()
        tmpl = self._apply_preprocessing(self._template)
        run_all_detection(
            self.root, self._scene, tmpl,
            self._ground_truth, self._show_all,
        )

    def _run_robustness(self) -> None:
        if not self._scene_and_template_loaded():
            return
        if not self._ground_truth:
            messagebox.showwarning(
                "No GT", "Draw ground truth first using the GT button.")
            return
        name = self._control.method_var.get()
        self._status_var.set(f"Robustness test: {name}...")
        self.root.update()
        run_robustness(
            self.root, self._scene, self._template,
            self._ground_truth, name,
            self._metrics.plot_robustness,
            lambda s: self._status_var.set(s),
        )

    def _export_results(self) -> None:
        if not self._last_results:
            messagebox.showinfo("No Results", "Run detection first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Results", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if path:
            save_results_csv(self._last_results, path)
            self._status_var.set(f"Saved: {os.path.basename(path)}")

    def _scene_and_template_loaded(self) -> bool:
        if self._scene is None or self._template is None:
            messagebox.showwarning("Missing", "Load scene + object first.")
            return False
        return True

    def _show_single(self, r: EvaluationResult) -> None:
        d = r.detection_result
        self._results.show_single_result(r)
        self._image_panel.show_detection(
            d.bounding_box, d.method_name, d.confidence)
        self._status_var.set(
            f"{d.method_name} — Conf: {d.confidence:.3f} | "
            f"IoU: {r.iou:.3f} | {d.execution_time_ms:.0f}ms")

    def _show_all(self, results: List[EvaluationResult]) -> None:
        self._last_results = results
        self._results.show_comparison(results)
        self._metrics.plot_comparison(results)
        if results:
            best = results[0]
            self._results.show_single_result(best)
            self._image_panel.show_detection(
                best.detection_result.bounding_box,
                "Best: " + best.method_name,
                best.detection_result.confidence)
        p = sum(1 for r in results if r.is_detected)
        self._status_var.set(
            f"All methods done — {p}/{len(results)} passed")

    def _apply_preprocessing(self, image: ImageArray) -> ImageArray:
        flags = self._control.get_preprocessing_flags()
        config = self._preproc_panel.get_config()
        result = apply_preprocessing_pipeline(image, flags, config)
        if any(flags.values()):
            self.root.after(
                0, lambda img=result: self._image_panel.set_preprocessed(img))
        else:
            self.root.after(0, self._image_panel.clear_preprocessed)
        return result

    def run(self) -> None:
        self.root.mainloop()
