"""
Drawing utilities for bounding boxes, labels, and heatmap overlays.

Uses OpenCV drawing primitives for annotation on result images.
"""

from typing import Tuple, Optional, List

import cv2
import numpy as np

from utils.types import ImageArray, BBox


def draw_bounding_box(
    image: ImageArray,
    bbox: BBox,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    label: Optional[str] = None,
    font_scale: float = 0.7,
) -> ImageArray:
    """Draw a bounding box on the image with an optional label tag.

    Args:
        image: Source image (BGR). A copy is returned; original is unchanged.
        bbox: (x, y, width, height) of the bounding box.
        color: BGR color tuple for the box and label background.
        thickness: Line thickness in pixels.
        label: Optional text to render above the box.
        font_scale: Font scale for the label text.

    Returns:
        Annotated copy of the input image.
    """
    result = image.copy()
    x, y, w, h = bbox

    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

    if label:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )[0]
        text_w, text_h = text_size
        img_h, img_w = result.shape[:2]

        # Keep the label fully inside the image.  This prevents long method
        # names such as "Hybrid (Neutral + Edge)" from being cut off when the
        # detected object is close to the right image border.
        label_x = max(0, min(x, img_w - text_w - 8))

        if y - text_h - 12 >= 0:
            label_y = y - 4
            bg_top_left = (label_x, label_y - text_h - 8)
            bg_bottom_right = (label_x + text_w + 8, label_y + 2)
            text_origin = (label_x + 4, label_y - 4)
        else:
            label_y = min(img_h - 2, y + h + text_h + 12)
            bg_top_left = (label_x, label_y - text_h - 8)
            bg_bottom_right = (label_x + text_w + 8, label_y + 2)
            text_origin = (label_x + 4, label_y - 4)

        cv2.rectangle(result, bg_top_left, bg_bottom_right, color, -1)
        cv2.putText(
            result,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
        )

    return result


def draw_multiple_boxes(
    image: ImageArray,
    boxes: List[Tuple[BBox, str]],
    colors: Optional[List[Tuple[int, int, int]]] = None,
) -> ImageArray:
    """Draw several labelled bounding boxes with distinct colors."""
    default_colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    result = image.copy()
    for i, (bbox, label) in enumerate(boxes):
        color = (
            colors[i] if colors else default_colors[i % len(default_colors)]
        )
        result = draw_bounding_box(result, bbox, color=color, label=label)
    return result


def draw_detection_heatmap(
    image: ImageArray,
    similarity_map: np.ndarray,
    alpha: float = 0.4,
) -> ImageArray:
    """Overlay a colour-coded similarity heatmap on the image.

    The similarity map is normalised to [0, 255], colour-mapped with JET,
    and alpha-blended onto the original image.
    """
    norm_map = similarity_map.astype(np.float64)
    min_val, max_val = norm_map.min(), norm_map.max()

    if max_val - min_val > 1e-8:
        norm_map = (norm_map - min_val) / (max_val - min_val) * 255.0
    else:
        norm_map = np.zeros_like(norm_map)

    norm_map = norm_map.astype(np.uint8)

    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(norm_map, (w, h))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    return cv2.addWeighted(image, 1.0 - alpha, heatmap_colored, alpha, 0)
