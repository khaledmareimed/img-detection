"""
Image display panel — scene canvas, query object preview,
preprocessed preview, and interactive GT drawing.
"""

import tkinter as tk
from typing import Callable, Optional, Tuple

import customtkinter as ctk
from PIL import Image, ImageTk

from utils.types import ImageArray, BBox
from utils.image_io import bgr_to_rgb
from utils.drawing import draw_bounding_box, draw_detection_heatmap
from gui.functions.gui_helpers import COLORS, FONT_HEADING, FONT_BODY, FONT_SMALL
from gui.functions.gt_drawing import GTDrawingHandler


class ImagePanel(ctk.CTkFrame):
    """Centre panel for scene display and object previews."""

    def __init__(self, parent) -> None:
        super().__init__(parent, fg_color=COLORS["bg"])
        self._scene_image: Optional[ImageArray] = None
        self._tk_scene: Optional[ImageTk.PhotoImage] = None
        self._tk_obj: Optional[ImageTk.PhotoImage] = None
        self._tk_pre: Optional[ImageTk.PhotoImage] = None
        self._scene_scale: float = 1.0
        self._scene_offset: Tuple[int, int] = (0, 0)
        self._build_ui()
        self._gt = GTDrawingHandler(self._scene_canvas)

    def _build_ui(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        ctk.CTkLabel(self, text="🖼  Image Display", font=FONT_HEADING,
                     text_color=COLORS["text"]).grid(
            row=0, column=0, sticky="w", padx=12, pady=(8, 2),
        )

        # Scene canvas (uses raw tk.Canvas for image drawing)
        scene_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_card"],
                                    corner_radius=10, border_width=1,
                                    border_color=COLORS["border"])
        scene_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        scene_frame.grid_rowconfigure(0, weight=1)
        scene_frame.grid_columnconfigure(0, weight=1)

        self._scene_canvas = tk.Canvas(
            scene_frame, bg=COLORS["bg"], highlightthickness=0, bd=0,
        )
        self._scene_canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._scene_hint = ctk.CTkLabel(
            scene_frame, text="Load a scene image to begin",
            font=FONT_BODY, text_color=COLORS["text_dim"],
        )
        self._scene_hint.grid(row=0, column=0)

        # Object row (Original + Preprocessed)
        obj_row = ctk.CTkFrame(self, fg_color="transparent")
        obj_row.grid(row=2, column=0, sticky="ew", padx=8, pady=(2, 8))
        obj_row.grid_columnconfigure(0, weight=1)
        obj_row.grid_columnconfigure(1, weight=1)

        # Original object
        self._obj_frame = ctk.CTkFrame(obj_row, fg_color=COLORS["bg_card"],
                                        corner_radius=8, border_width=1,
                                        border_color=COLORS["border"])
        self._obj_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        ctk.CTkLabel(self._obj_frame, text="Query Object (Original)",
                     font=FONT_SMALL, text_color=COLORS["text_dim"]
                     ).pack(anchor="w", padx=8, pady=(4, 0))
        self._obj_canvas = tk.Canvas(
            self._obj_frame, bg=COLORS["bg"], highlightthickness=0, height=130,
        )
        self._obj_canvas.pack(fill="both", expand=True, padx=4, pady=4)

        # Preprocessed
        self._pre_frame = ctk.CTkFrame(obj_row, fg_color=COLORS["bg_card"],
                                        corner_radius=8, border_width=1,
                                        border_color=COLORS["border"])
        ctk.CTkLabel(self._pre_frame, text="After Preprocessing",
                     font=FONT_SMALL, text_color=COLORS["text_dim"]
                     ).pack(anchor="w", padx=8, pady=(4, 0))
        self._pre_canvas = tk.Canvas(
            self._pre_frame, bg=COLORS["bg"], highlightthickness=0, height=130,
        )
        self._pre_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self._pre_visible = False

    # ── Public API ──

    def set_scene(self, img: ImageArray) -> None:
        self._scene_image = img
        self._scene_hint.grid_forget()
        self._render_scene(img)

    def set_object(self, img: ImageArray) -> None:
        self._render_to_canvas(img, self._obj_canvas, "_tk_obj")
        self.clear_preprocessed()

    def set_preprocessed(self, img: ImageArray) -> None:
        if not self._pre_visible:
            self._pre_frame.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
            self._pre_visible = True
        self._render_to_canvas(img, self._pre_canvas, "_tk_pre")

    def clear_preprocessed(self) -> None:
        if self._pre_visible:
            self._pre_frame.grid_forget()
            self._pre_visible = False
        self._pre_canvas.delete("all")
        self._tk_pre = None

    def show_detection(self, bbox: BBox, label: str = "Detected",
                       confidence: float = 0.0) -> None:
        if self._scene_image is None:
            return
        tag = f"{label} ({confidence:.2f})"
        ann = draw_bounding_box(self._scene_image, bbox,
                                 color=(0, 255, 0), label=tag)
        self._render_scene(ann)

    def show_ground_truth(self, bbox: BBox) -> None:
        if self._scene_image is None:
            return
        ann = draw_bounding_box(self._scene_image, bbox,
                                 color=(0, 255, 255), label="Ground Truth")
        self._render_scene(ann)

    def show_heatmap(self, smap) -> None:
        if self._scene_image is None or smap is None:
            return
        self._render_scene(draw_detection_heatmap(self._scene_image, smap))

    def start_gt_drawing(self, cb: Callable[[BBox], None]) -> None:
        if self._scene_image is None:
            return
        h, w = self._scene_image.shape[:2]
        self._gt.start(cb, self._scene_scale, self._scene_offset, (h, w))

    def clear(self) -> None:
        self._scene_canvas.delete("all")
        self._obj_canvas.delete("all")
        self._scene_image = None
        self.clear_preprocessed()

    # ── Rendering ──

    def _render_scene(self, img: ImageArray) -> None:
        self._scene_canvas.update_idletasks()
        cw = max(self._scene_canvas.winfo_width(), 300)
        ch = max(self._scene_canvas.winfo_height(), 200)
        rgb = bgr_to_rgb(img)
        pil = Image.fromarray(rgb)
        iw, ih = pil.size
        self._scene_scale = min(cw / iw, ch / ih, 1.0)
        dw = max(1, int(iw * self._scene_scale))
        dh = max(1, int(ih * self._scene_scale))
        self._scene_offset = ((cw - dw) // 2, (ch - dh) // 2)
        if self._scene_scale < 1.0:
            pil = pil.resize((dw, dh), Image.Resampling.LANCZOS)
        self._tk_scene = ImageTk.PhotoImage(pil)
        self._scene_canvas.delete("all")
        self._scene_canvas.create_image(
            cw // 2, ch // 2, image=self._tk_scene, anchor="center")

    def _render_to_canvas(self, img: ImageArray, canvas: tk.Canvas,
                          attr: str) -> None:
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)
        rgb = bgr_to_rgb(img)
        pil = Image.fromarray(rgb)
        w, h = pil.size
        s = min(cw / w, ch / h, 1.0)
        if s < 1.0:
            pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))),
                              Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        setattr(self, attr, photo)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
