"""
Results panel — detection stats and comparison table.
"""

import customtkinter as ctk
from tkinter import ttk
from typing import List

from utils.types import EvaluationResult
from gui.functions.gui_helpers import COLORS, FONT_HEADING, FONT_BODY, FONT_SMALL, FONT_MONO


class ResultsPanel(ctk.CTkFrame):
    """Right-side panel showing detection results and comparison."""

    def __init__(self, parent) -> None:
        super().__init__(parent, width=300, fg_color=COLORS["bg_card"])
        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        ctk.CTkLabel(self, text="📊  Results", font=FONT_HEADING,
                     text_color=COLORS["text"]).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 4),
        )

        # ── Single result stats ──
        stats_frame = ctk.CTkFrame(self, fg_color=COLORS["bg"],
                                    corner_radius=8)
        stats_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        stats_frame.grid_columnconfigure(1, weight=1)

        self._stat_labels = {}
        stats = [
            ("Method:", "method"), ("Bounding Box:", "bbox"),
            ("Confidence:", "confidence"), ("IoU:", "iou"),
            ("Loc. Error:", "loc_error"), ("Detected:", "detected"),
            ("Time:", "time"),
        ]
        for i, (label, key) in enumerate(stats):
            ctk.CTkLabel(
                stats_frame, text=label, font=FONT_SMALL,
                text_color=COLORS["text_dim"], anchor="w",
            ).grid(row=i, column=0, sticky="w", padx=(10, 4), pady=1)
            val = ctk.CTkLabel(
                stats_frame, text="—", font=FONT_MONO,
                text_color=COLORS["text"], anchor="w",
            )
            val.grid(row=i, column=1, sticky="ew", padx=(0, 10), pady=1)
            self._stat_labels[key] = val

        # ── Comparison table ──
        ctk.CTkLabel(self, text="📋  Method Comparison", font=FONT_BODY,
                     text_color=COLORS["text"]).grid(
            row=2, column=0, sticky="nw", padx=12, pady=(8, 2),
        )

        tree_frame = ctk.CTkFrame(self, fg_color=COLORS["bg"],
                                   corner_radius=8)
        tree_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Configure Treeview style for dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                         background=COLORS["bg"],
                         foreground=COLORS["text"],
                         fieldbackground=COLORS["bg"],
                         borderwidth=0, font=("Segoe UI", 9))
        style.configure("Dark.Treeview.Heading",
                         background=COLORS["bg_card"],
                         foreground=COLORS["text"],
                         borderwidth=0, font=("Segoe UI", 9, "bold"))
        style.map("Dark.Treeview",
                   background=[("selected", COLORS["accent"])])

        cols = ("method", "iou", "confidence", "time", "status")
        self._tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings",
            height=7, style="Dark.Treeview",
        )
        for col, hdr, w in [
            ("method", "Method", 130), ("iou", "IoU", 50),
            ("confidence", "Conf.", 50), ("time", "ms", 55),
            ("status", "✓", 30),
        ]:
            self._tree.heading(col, text=hdr)
            anchor = "w" if col == "method" else "center"
            self._tree.column(col, width=w, minwidth=w, anchor=anchor)

        sb = ttk.Scrollbar(tree_frame, orient="vertical",
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

    # ── Public API ──

    def show_single_result(self, r: EvaluationResult) -> None:
        d = r.detection_result
        bx, by, bw, bh = d.bounding_box
        self._stat_labels["method"].configure(text=d.method_name)
        self._stat_labels["bbox"].configure(text=f"({bx},{by}) {bw}×{bh}")
        self._stat_labels["confidence"].configure(text=f"{d.confidence:.4f}")
        self._stat_labels["iou"].configure(text=f"{r.iou:.4f}")
        self._stat_labels["loc_error"].configure(
            text=f"{r.localization_error:.1f} px")
        det = "✅ Yes" if r.is_detected else "❌ No"
        clr = COLORS["success"] if r.is_detected else COLORS["error"]
        self._stat_labels["detected"].configure(text=det, text_color=clr)
        self._stat_labels["time"].configure(
            text=f"{d.execution_time_ms:.1f} ms")

    def show_comparison(self, results: List[EvaluationResult]) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)
        for r in results:
            d = r.detection_result
            self._tree.insert("", "end", values=(
                d.method_name, f"{r.iou:.3f}", f"{d.confidence:.3f}",
                f"{d.execution_time_ms:.0f}",
                "✅" if r.is_detected else "❌",
            ))

    def clear(self) -> None:
        for lbl in self._stat_labels.values():
            lbl.configure(text="—", text_color=COLORS["text"])
        for item in self._tree.get_children():
            self._tree.delete(item)
