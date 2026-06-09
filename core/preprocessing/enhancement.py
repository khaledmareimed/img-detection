"""
Image enhancement utilities implemented from scratch.

Provides histogram equalization, CLAHE, and contrast stretching
using only NumPy operations (no OpenCV histogram functions).
"""

import numpy as np

from utils.types import ImageArray


def compute_histogram(image: ImageArray, bins: int = 256) -> np.ndarray:
    """Compute the intensity histogram of a single-channel image.

    Args:
        image: 2-D grayscale image (uint8).
        bins: Number of histogram bins (default 256).

    Returns:
        1-D array of length ``bins`` with pixel counts.
    """
    hist = np.bincount(image.ravel(), minlength=bins).astype(np.int64)
    return hist[:bins]


def compute_cdf(hist: np.ndarray) -> np.ndarray:
    """Compute the cumulative distribution function from a histogram."""
    cdf = np.cumsum(hist).astype(np.float64)
    # Normalise to [0, 255]
    cdf_min = cdf[cdf > 0].min()
    total = cdf[-1]
    if total - cdf_min == 0:
        return np.zeros_like(cdf)
    cdf_normalised = (cdf - cdf_min) / (total - cdf_min) * 255.0
    return np.clip(cdf_normalised, 0, 255).astype(np.uint8)


def histogram_equalization(image: ImageArray) -> ImageArray:
    """Apply global histogram equalization to a grayscale image.

    Computes the CDF of pixel intensities and remaps every pixel
    to spread the histogram across the full [0, 255] range.
    """
    if len(image.shape) == 3:
        # Convert to YCrCb, equalise only the luminance channel
        ycrcb = _bgr_to_ycrcb(image)
        ycrcb[:, :, 0] = _equalize_channel(ycrcb[:, :, 0])
        return _ycrcb_to_bgr(ycrcb)
    return _equalize_channel(image)


def _equalize_channel(channel: np.ndarray) -> np.ndarray:
    """Equalize a single 2-D channel."""
    hist = compute_histogram(channel)
    lut = compute_cdf(hist)
    return lut[channel]


def clahe(
    image: ImageArray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> ImageArray:
    """Contrast-Limited Adaptive Histogram Equalization (CLAHE).

    Divides the image into tiles, clips histograms at ``clip_limit``,
    and uses bilinear interpolation between neighbouring tiles to
    avoid block artefacts.
    """
    if len(image.shape) == 3:
        ycrcb = _bgr_to_ycrcb(image)
        ycrcb[:, :, 0] = _clahe_channel(
            ycrcb[:, :, 0], clip_limit, tile_size
        )
        return _ycrcb_to_bgr(ycrcb)
    return _clahe_channel(image, clip_limit, tile_size)


def _clahe_channel(
    channel: np.ndarray, clip_limit: float, tile_size: int
) -> np.ndarray:
    """Apply CLAHE to a single 2-D channel."""
    h, w = channel.shape
    tile_h = max(1, h // tile_size)
    tile_w = max(1, w // tile_size)

    result = np.zeros_like(channel, dtype=np.float64)

    for ty in range(tile_size):
        for tx in range(tile_size):
            y0 = ty * tile_h
            x0 = tx * tile_w
            y1 = min(y0 + tile_h, h)
            x1 = min(x0 + tile_w, w)

            tile = channel[y0:y1, x0:x1]
            hist = compute_histogram(tile)

            # Clip histogram
            n_pixels = tile.size
            limit = int(clip_limit * n_pixels / 256)
            excess = np.sum(np.maximum(hist - limit, 0))
            hist = np.minimum(hist, limit)
            hist += excess // 256

            lut = compute_cdf(hist)
            result[y0:y1, x0:x1] = lut[tile]

    return result.astype(np.uint8)


def contrast_stretching(
    image: ImageArray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> ImageArray:
    """Linear contrast stretching between percentile bounds.

    Pixels below ``low_percentile`` are mapped to 0,
    pixels above ``high_percentile`` are mapped to 255.
    """
    result = image.astype(np.float64)
    low = np.percentile(result, low_percentile)
    high = np.percentile(result, high_percentile)

    if high - low < 1e-8:
        return image

    result = (result - low) / (high - low) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal colour-space helpers (avoid OpenCV dependency in this module)
# ---------------------------------------------------------------------------

def _bgr_to_ycrcb(image: ImageArray) -> ImageArray:
    """BGR → YCrCb using standard BT.601 coefficients."""
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


def _ycrcb_to_bgr(image: ImageArray) -> ImageArray:
    """YCrCb → BGR."""
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
