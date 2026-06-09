"""
Metrics visualisation panel — comparison and robustness charts.
"""

import customtkinter as ctk
from typing import List

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.types import EvaluationResult, RobustnessTestResult
from gui.functions.gui_helpers import COLORS, FONT_HEADING, FONT_BODY


class MetricsPanel(ctk.CTkFrame):
    """Bottom panel with Comparison / Robustness chart tabs."""

    def __init__(self, parent) -> None:
        super().__init__(parent, fg_color=COLORS["bg_card"])
        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Tabview
        self._tabs = ctk.CTkTabview(
            self, fg_color=COLORS["bg"],
            segmented_button_fg_color=COLORS["bg_card"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_unselected_color=COLORS["bg_input"],
            corner_radius=8,
        )
        self._tabs.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._tabs.add("Comparison")
        self._tabs.add("Robustness")

    def plot_comparison(self, results: List[EvaluationResult]) -> None:
        tab = self._tabs.tab("Comparison")
        for w in tab.winfo_children():
            w.destroy()
        if not results:
            return

        fig = Figure(figsize=(8, 2.2), dpi=100, facecolor=COLORS["bg"])
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        names = [r.method_name[:22] for r in results]
        ious = [r.iou for r in results]
        times = [r.detection_result.execution_time_ms for r in results]
        colors = [COLORS["success"] if r.is_detected else COLORS["error"]
                  for r in results]

        ax1.barh(range(len(names)), ious, color=colors, height=0.6)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=7, color="#cbd5e1")
        ax1.set_xlabel("IoU", fontsize=8, color="#cbd5e1")
        ax1.set_title("IoU by Method", fontsize=9, color="#e2e8f0",
                       fontweight="bold")
        ax1.axvline(x=0.3, color=COLORS["warning"], linestyle="--",
                    linewidth=0.8)
        self._style_ax(ax1)

        ax2.barh(range(len(names)), times, color="#7c3aed", height=0.6)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=7, color="#cbd5e1")
        ax2.set_xlabel("Time (ms)", fontsize=8, color="#cbd5e1")
        ax2.set_title("Execution Time", fontsize=9, color="#e2e8f0",
                       fontweight="bold")
        self._style_ax(ax2)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_robustness(self, results: List[RobustnessTestResult]) -> None:
        tab = self._tabs.tab("Robustness")
        for w in tab.winfo_children():
            w.destroy()
        if not results:
            return

        fig = Figure(figsize=(8, 2.2), dpi=100, facecolor=COLORS["bg"])
        ax = fig.add_subplot(111)

        labels = [r.transformation for r in results]
        ious = [r.evaluation.iou for r in results]
        colors = [COLORS["success"] if r.evaluation.is_detected
                  else COLORS["error"] for r in results]

        ax.bar(range(len(labels)), ious, color=colors, width=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6,
                            color="#cbd5e1")
        ax.set_ylabel("IoU", fontsize=8, color="#cbd5e1")
        ax.set_title("Robustness Test Results", fontsize=9, color="#e2e8f0",
                      fontweight="bold")
        ax.axhline(y=0.3, color=COLORS["warning"], linestyle="--",
                   linewidth=0.8)
        self._style_ax(ax)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._tabs.set("Robustness")

    @staticmethod
    def _style_ax(ax) -> None:
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
