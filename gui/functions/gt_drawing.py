"""
Interactive ground truth drawing handler.

Manages click-and-drag rectangle drawing on a Tkinter Canvas,
converting canvas coordinates back to image coordinates using
the stored scale and offset.
"""

import tkinter as tk
from typing import Callable, Optional, Tuple

from utils.types import BBox
from gui.functions.gui_helpers import COLORS, FONT_BODY


class GTDrawingHandler:
    """Manages interactive ground truth bounding box drawing."""

    def __init__(self, canvas: tk.Canvas) -> None:
        self._canvas = canvas
        self._drawing: bool = False
        self._start: Optional[Tuple[int, int]] = None
        self._rect_id: Optional[int] = None
        self._callback: Optional[Callable[[BBox], None]] = None

        # Hint label shown during drawing mode
        self._hint = tk.Label(
            self._canvas,
            text="🎯  Click and drag to draw ground truth box",
            bg="#7c3aed", fg="#ffffff", font=FONT_BODY,
            padx=8, pady=4,
        )

    @property
    def is_drawing(self) -> bool:
        return self._drawing

    def start(
        self,
        callback: Callable[[BBox], None],
        scene_scale: float,
        scene_offset: Tuple[int, int],
        image_shape: Tuple[int, int],
    ) -> None:
        """Enable drawing mode on the canvas.

        Args:
            callback: Called with ``(x, y, w, h)`` in image coords.
            scene_scale: Display scale factor (image → canvas).
            scene_offset: Canvas offset ``(ox, oy)`` for centering.
            image_shape: Original image ``(height, width)``.
        """
        self._callback = callback
        self._scale = scene_scale
        self._offset = scene_offset
        self._img_h, self._img_w = image_shape
        self._drawing = True

        self._canvas.config(cursor="crosshair")
        self._hint.place(relx=0.5, y=8, anchor="n")

        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

    def cancel(self) -> None:
        """Cancel drawing mode without invoking the callback."""
        self._cleanup()

    # ------------------------------------------------------------------
    # Mouse event handlers
    # ------------------------------------------------------------------

    def _on_press(self, event: tk.Event) -> None:
        """Record the starting corner of the rectangle."""
        self._start = (event.x, event.y)
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)
        self._rect_id = self._canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#00ffff", width=2, dash=(4, 4),
        )

    def _on_drag(self, event: tk.Event) -> None:
        """Resize the rectangle as the user drags."""
        if self._start and self._rect_id:
            self._canvas.coords(
                self._rect_id,
                self._start[0], self._start[1],
                event.x, event.y,
            )

    def _on_release(self, event: tk.Event) -> None:
        """Finalise the rectangle and convert to image coords."""
        if not self._start:
            return

        cx1, cy1 = self._start
        cx2, cy2 = event.x, event.y

        # Ensure top-left → bottom-right
        cx1, cx2 = min(cx1, cx2), max(cx1, cx2)
        cy1, cy2 = min(cy1, cy2), max(cy1, cy2)

        # Reject tiny drags (accidental clicks)
        if (cx2 - cx1) < 5 or (cy2 - cy1) < 5:
            if self._rect_id:
                self._canvas.delete(self._rect_id)
            return

        # Convert canvas coords → image coords
        ox, oy = self._offset
        scale = self._scale
        ix = int((cx1 - ox) / scale)
        iy = int((cy1 - oy) / scale)
        iw = int((cx2 - cx1) / scale)
        ih = int((cy2 - cy1) / scale)

        # Clamp to image bounds
        ix = max(0, min(ix, self._img_w - 1))
        iy = max(0, min(iy, self._img_h - 1))
        iw = max(1, min(iw, self._img_w - ix))
        ih = max(1, min(ih, self._img_h - iy))

        bbox: BBox = (ix, iy, iw, ih)

        self._cleanup()

        if self._callback:
            self._callback(bbox)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Remove drawing mode bindings and visuals."""
        self._drawing = False
        self._start = None
        self._canvas.config(cursor="")
        self._hint.place_forget()
        if self._rect_id:
            self._canvas.delete(self._rect_id)
            self._rect_id = None
        self._canvas.unbind("<ButtonPress-1>")
        self._canvas.unbind("<B1-Motion>")
        self._canvas.unbind("<ButtonRelease-1>")
