"""
Left-side control panel — image loading, GT, method selection,
preprocessing toggles, and action buttons.
"""

import os
import customtkinter as ctk
from tkinter import filedialog
from typing import Callable

from gui.functions.gui_helpers import (
    COLORS, FONT_HEADING, FONT_BODY, FONT_SMALL, styled_button,
)


class ControlPanel(ctk.CTkScrollableFrame):
    """Scrollable left sidebar with all user controls."""

    def __init__(
        self, parent, *,
        on_load_scene: Callable[[str], None],
        on_load_object: Callable[[str], None],
        on_run_detection: Callable[[str], None],
        on_run_all: Callable[[], None],
        on_run_robustness: Callable[[], None],
        on_export: Callable[[], None],
        on_set_gt: Callable[[], None],
    ) -> None:
        super().__init__(parent, width=270, fg_color=COLORS["bg_card"])

        self._cbs = {
            "scene": on_load_scene, "object": on_load_object,
            "detect": on_run_detection, "all": on_run_all,
            "robust": on_run_robustness, "export": on_export,
            "gt": on_set_gt,
        }
        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": (2, 2)}

        # ── Image Loading ──
        self._section("📂  Image Loading")
        styled_button(self, "Load Scene Image", self._load_scene,
                       color="#6366f1").pack(fill="x", **pad)
        self._scene_lbl = ctk.CTkLabel(
            self, text="No scene loaded", font=FONT_SMALL,
            text_color=COLORS["text_dim"],
        )
        self._scene_lbl.pack(anchor="w", padx=12)

        styled_button(self, "Load Query Object", self._load_object,
                       color="#6366f1").pack(fill="x", **pad)
        self._obj_lbl = ctk.CTkLabel(
            self, text="No object loaded", font=FONT_SMALL,
            text_color=COLORS["text_dim"],
        )
        self._obj_lbl.pack(anchor="w", padx=12)

        self._sep()

        # ── Ground Truth ──
        self._section("📍  Ground Truth")
        styled_button(self, "Draw Ground Truth Box", self._cbs["gt"],
                       color=COLORS["cyan"], hover="#22d3ee"
                       ).pack(fill="x", **pad)
        self.gt_label_var = ctk.StringVar(value="No GT set")
        ctk.CTkLabel(
            self, textvariable=self.gt_label_var, font=FONT_SMALL,
            text_color=COLORS["text_dim"],
        ).pack(anchor="w", padx=12)

        self._sep()

        # ── Detection Method ──
        self._section("🔍  Detection Method")
        self.method_var = ctk.StringVar(value="Template Matching (NCC)")
        ctk.CTkComboBox(
            self, variable=self.method_var,
            values=[
                "Template Matching (NCC)", "Template Matching (SSD)",
                "Edge Detection (Canny)", "Edge Detection (Sobel)",
                "Histogram (Bhattacharyya)", "Histogram (Chi-Square)",
                "Hybrid (Hist + NCC)",
            ],
            font=FONT_SMALL, dropdown_font=FONT_SMALL,
            state="readonly", corner_radius=6,
        ).pack(fill="x", **pad)

        self._sep()

        # ── Preprocessing ──
        self._section("🔧  Preprocessing")
        self.preproc_vars = {}
        for label, key in [
            ("Histogram Equalization", "hist_eq"), ("CLAHE", "clahe"),
            ("Gaussian Blur", "gaussian"), ("Median Filter", "median"),
            ("Contrast Stretching", "contrast"),
            ("Background Removal", "bg_removal"),
        ]:
            var = ctk.BooleanVar(value=False)
            self.preproc_vars[key] = var
            ctk.CTkCheckBox(
                self, text=label, variable=var,
                font=FONT_SMALL, corner_radius=4,
                fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            ).pack(anchor="w", padx=14, pady=1)

        self._sep()

        # ── Actions ──
        self._section("▶  Actions")
        styled_button(self, "Run Detection", self._run_detection,
                       color="#22c55e", hover="#16a34a").pack(fill="x", **pad)
        styled_button(self, "Run All Methods", self._run_all,
                       color="#3b82f6", hover="#2563eb").pack(fill="x", **pad)
        styled_button(self, "Robustness Test", self._run_robustness,
                       color="#f59e0b", hover="#d97706").pack(fill="x", **pad)
        styled_button(self, "Export Results", self._cbs["export"],
                       color=COLORS["bg_input"], hover="#475569"
                       ).pack(fill="x", **pad)

    # ── Helpers ──

    def _section(self, text: str) -> None:
        ctk.CTkLabel(self, text=text, font=FONT_HEADING,
                     text_color=COLORS["text"]).pack(
            anchor="w", padx=10, pady=(10, 2),
        )

    def _sep(self) -> None:
        ctk.CTkFrame(self, height=1, fg_color=COLORS["border"]).pack(
            fill="x", padx=10, pady=8,
        )

    def _images_dir(self) -> str:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(this_dir))
        d = os.path.join(root, "images")
        return d if os.path.isdir(d) else root

    def _load_scene(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Scene Image", initialdir=self._images_dir(),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")],
        )
        if p:
            self._scene_lbl.configure(text=os.path.basename(p))
            self._cbs["scene"](p)

    def _load_object(self) -> None:
        p = filedialog.askopenfilename(
            title="Select Query Object", initialdir=self._images_dir(),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")],
        )
        if p:
            self._obj_lbl.configure(text=os.path.basename(p))
            self._cbs["object"](p)

    def _run_detection(self) -> None:
        self._cbs["detect"](self.method_var.get())

    def _run_all(self) -> None:
        self._cbs["all"]()

    def _run_robustness(self) -> None:
        self._cbs["robust"]()

    def get_preprocessing_flags(self) -> dict:
        return {k: v.get() for k, v in self.preproc_vars.items()}
