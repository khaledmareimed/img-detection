"""
Preprocessing settings panel — sliders and spinboxes for
fine-tuning preprocessing parameters.
"""

import customtkinter as ctk
from typing import Dict

from utils.types import PreprocessingConfig
from gui.functions.gui_helpers import COLORS, FONT_HEADING, FONT_SMALL


class PreprocessingPanel(ctk.CTkFrame):
    """Compact panel with parameter sliders for preprocessing."""

    def __init__(self, parent) -> None:
        super().__init__(parent, width=280, fg_color=COLORS["bg_card"])
        self._vars: Dict[str, ctk.Variable] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        ctk.CTkLabel(self, text="🔧  Preprocessing Settings",
                     font=FONT_HEADING, text_color=COLORS["text"]).pack(
            anchor="w", padx=10, pady=(8, 4),
        )

        params = [
            ("CLAHE Clip",    "clahe_clip",    2.0,  0.5, 10.0, 0.5),
            ("CLAHE Tile",    "clahe_tile",    8,    2,   32,   1),
            ("Gauss Kernel",  "gauss_k",       5,    3,   15,   2),
            ("Gauss Sigma",   "gauss_sigma",   1.0,  0.1, 5.0,  0.1),
            ("Median Kernel", "median_k",      5,    3,   15,   2),
            ("Contrast Low%", "contrast_low",  2.0,  0.0, 10.0, 0.5),
            ("Contrast Hi%",  "contrast_high", 98.0, 90., 100., 0.5),
            ("Morph Kernel",  "morph_k",       5,    3,   15,   2),
        ]

        for label, key, default, lo, hi, step in params:
            row = ctk.CTkFrame(self, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=1)

            ctk.CTkLabel(row, text=label, font=FONT_SMALL, width=100,
                         text_color=COLORS["text_dim"], anchor="w"
                         ).pack(side="left")

            var = ctk.DoubleVar(value=default)
            self._vars[key] = var

            slider = ctk.CTkSlider(
                row, from_=lo, to=hi, variable=var,
                width=110, height=14,
                fg_color=COLORS["bg_input"],
                progress_color=COLORS["accent"],
                button_color=COLORS["accent"],
                button_hover_color=COLORS["accent_hover"],
            )
            slider.pack(side="left", padx=(4, 4))

            val_lbl = ctk.CTkLabel(row, text=f"{default}", font=FONT_SMALL,
                                    text_color=COLORS["text"], width=40)
            val_lbl.pack(side="left")
            var.trace_add("write", lambda *_, v=var, l=val_lbl:
                          l.configure(text=f"{v.get():.1f}"))

        # Morph operation
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(4, 8))
        ctk.CTkLabel(row, text="Morph Op", font=FONT_SMALL, width=100,
                     text_color=COLORS["text_dim"], anchor="w").pack(side="left")
        self._morph_var = ctk.StringVar(value="opening")
        ctk.CTkComboBox(
            row, variable=self._morph_var, width=130,
            values=["erosion", "dilation", "opening", "closing"],
            state="readonly", font=FONT_SMALL,
        ).pack(side="left", padx=4)

    def get_config(self) -> PreprocessingConfig:
        return PreprocessingConfig(
            clahe_clip_limit=self._vars["clahe_clip"].get(),
            clahe_tile_size=int(self._vars["clahe_tile"].get()),
            gaussian_kernel_size=int(self._vars["gauss_k"].get()),
            gaussian_sigma=self._vars["gauss_sigma"].get(),
            median_kernel_size=int(self._vars["median_k"].get()),
            contrast_low_pct=self._vars["contrast_low"].get(),
            contrast_high_pct=self._vars["contrast_high"].get(),
            morph_kernel_size=int(self._vars["morph_k"].get()),
            morph_operation=self._morph_var.get(),
        )
