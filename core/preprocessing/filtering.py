"""
Spatial filtering utilities.

Provides Gaussian blur, median filtering, and 2-D convolution.
Uses cv2.filter2D for fast convolution (filtering is an allowed
basic operation per the project spec).  Custom kernel creation
is still implemented from scratch.
"""

import cv2
import numpy as np

from utils.types import ImageArray


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a 2-D Gaussian kernel from scratch.

    Args:
        size: Kernel side length (must be odd).
        sigma: Standard deviation of the Gaussian.

    Returns:
        Normalised 2-D kernel of shape ``(size, size)``.
    """
    if size % 2 == 0:
        size += 1

    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Fast 2-D convolution of *image* with *kernel*.

    Uses cv2.filter2D (a basic filtering operation allowed by the
    project spec) for performance.  Works on single-channel arrays.
    """
    return cv2.filter2D(
        image.astype(np.float64), -1, kernel,
        borderType=cv2.BORDER_REFLECT,
    )


def gaussian_blur(
    image: ImageArray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> ImageArray:
    """Apply Gaussian blur using a custom-built kernel.

    The kernel is created from scratch; only the convolution uses
    the optimised cv2.filter2D.
    """
    kernel = create_gaussian_kernel(kernel_size, sigma)
    result = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(result, 0, 255).astype(np.uint8)


def median_filter(image: ImageArray, kernel_size: int = 5) -> ImageArray:
    """Apply a median filter to an image.

    Uses cv2.medianBlur for performance (filtering is an allowed
    basic operation).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)
