"""
Morphological image processing operations.

Structuring elements are created from scratch.
Erosion/dilation use cv2 for performance (filtering is an allowed
basic operation).  Opening/closing compose these primitives.
"""

import cv2
import numpy as np

from utils.types import ImageArray


def create_structuring_element(
    size: int, shape: str = "rect"
) -> np.ndarray:
    """Create a binary structuring element from scratch.

    Args:
        size: Side length (must be odd).
        shape: ``"rect"``, ``"cross"``, or ``"ellipse"``.

    Returns:
        Binary 2-D array of shape ``(size, size)``.
    """
    if size % 2 == 0:
        size += 1

    if shape == "rect":
        return np.ones((size, size), dtype=np.uint8)

    if shape == "cross":
        se = np.zeros((size, size), dtype=np.uint8)
        mid = size // 2
        se[mid, :] = 1
        se[:, mid] = 1
        return se

    if shape == "ellipse":
        se = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if ((i - center) ** 2 + (j - center) ** 2) <= center ** 2:
                    se[i, j] = 1
        return se

    raise ValueError(f"Unknown structuring element shape: {shape}")


def erode(image: ImageArray, kernel_size: int = 3) -> ImageArray:
    """Morphological erosion (minimum within structuring element)."""
    se = create_structuring_element(kernel_size)
    return cv2.erode(image, se, iterations=1)


def dilate(image: ImageArray, kernel_size: int = 3) -> ImageArray:
    """Morphological dilation (maximum within structuring element)."""
    se = create_structuring_element(kernel_size)
    return cv2.dilate(image, se, iterations=1)


def morphological_opening(
    image: ImageArray, kernel_size: int = 3
) -> ImageArray:
    """Opening = erosion followed by dilation."""
    return dilate(erode(image, kernel_size), kernel_size)


def morphological_closing(
    image: ImageArray, kernel_size: int = 3
) -> ImageArray:
    """Closing = dilation followed by erosion."""
    return erode(dilate(image, kernel_size), kernel_size)


def apply_morphology(
    image: ImageArray, operation: str, kernel_size: int = 5
) -> ImageArray:
    """Dispatch to the requested morphological operation."""
    ops = {
        "erosion": erode,
        "dilation": dilate,
        "opening": morphological_opening,
        "closing": morphological_closing,
    }
    if operation not in ops:
        raise ValueError(
            f"Unknown operation '{operation}'. "
            f"Choose from {list(ops.keys())}"
        )
    return ops[operation](image, kernel_size)
