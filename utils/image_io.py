"""
Image I/O utilities for loading, saving, and converting images.

Uses OpenCV for basic I/O operations (allowed by project spec).
All images are stored in BGR format internally (OpenCV convention).
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.types import ImageArray


def load_image(filepath: str) -> ImageArray:
    """Load an image from disk in BGR format.

    Args:
        filepath: Absolute or relative path to the image file.

    Returns:
        The loaded image as a NumPy array in BGR color order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV fails to decode the image.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {filepath}")
    return image


def save_image(image: ImageArray, filepath: str) -> None:
    """Save an image to disk, creating parent directories if needed."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    success = cv2.imwrite(filepath, image)
    if not success:
        raise IOError(f"Failed to save image to: {filepath}")


def bgr_to_rgb(image: ImageArray) -> ImageArray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: ImageArray) -> ImageArray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_gray(image: ImageArray) -> ImageArray:
    """Convert BGR image to grayscale using luminosity weights."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_hsv(image: ImageArray) -> ImageArray:
    """Convert BGR image to HSV color space."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def gray_to_bgr(image: ImageArray) -> ImageArray:
    """Convert single-channel grayscale to 3-channel BGR."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def resize_image(
    image: ImageArray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
) -> ImageArray:
    """Resize an image by explicit dimensions or a scale factor.

    Provide *either* ``scale`` *or* one/both of ``width``/``height``.
    When only one dimension is given the aspect ratio is preserved.
    """
    h, w = image.shape[:2]

    if scale is not None:
        new_w = int(w * scale)
        new_h = int(h * scale)
    elif width is not None and height is not None:
        new_w, new_h = width, height
    elif width is not None:
        ratio = width / w
        new_w = width
        new_h = int(h * ratio)
    elif height is not None:
        ratio = height / h
        new_h = height
        new_w = int(w * ratio)
    else:
        return image

    # Guard against zero-size results
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    interpolation = (
        cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
    )
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def get_image_dimensions(image: ImageArray) -> Tuple[int, int, int]:
    """Return ``(height, width, channels)`` of an image."""
    if len(image.shape) == 2:
        h, w = image.shape
        return h, w, 1
    return image.shape[0], image.shape[1], image.shape[2]
