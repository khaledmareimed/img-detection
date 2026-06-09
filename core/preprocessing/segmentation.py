"""
Image segmentation and background removal utilities.

Provides thresholding (Otsu), HSV-based background removal,
and object cropping for query image preprocessing.
"""

import numpy as np

from utils.types import ImageArray, BBox
from utils.image_io import bgr_to_hsv, bgr_to_gray


def otsu_threshold(image: ImageArray) -> int:
    """Compute the optimal binarisation threshold using Otsu's method.

    Exhaustively tests every threshold in [0, 255] and picks the one
    that maximises the between-class variance.

    Args:
        image: Single-channel (grayscale) image.

    Returns:
        The optimal threshold value.
    """
    if len(image.shape) == 3:
        image = bgr_to_gray(image)

    hist = np.zeros(256, dtype=np.float64)
    for v in image.ravel():
        hist[v] += 1
    hist /= hist.sum()

    best_thresh = 0
    best_variance = 0.0
    cumulative_sum = 0.0
    cumulative_mean = 0.0
    global_mean = np.sum(np.arange(256) * hist)

    for t in range(256):
        cumulative_sum += hist[t]
        if cumulative_sum == 0 or cumulative_sum == 1:
            continue
        cumulative_mean += t * hist[t]

        mean_bg = cumulative_mean / cumulative_sum
        mean_fg = (global_mean - cumulative_mean) / (1 - cumulative_sum)

        between_var = (
            cumulative_sum * (1 - cumulative_sum) * (mean_bg - mean_fg) ** 2
        )
        if between_var > best_variance:
            best_variance = between_var
            best_thresh = t

    return best_thresh


def threshold_binary(
    image: ImageArray, thresh: int = -1
) -> ImageArray:
    """Binarise a grayscale image.

    If *thresh* is ``-1`` the Otsu threshold is computed automatically.
    """
    if len(image.shape) == 3:
        image = bgr_to_gray(image)

    if thresh < 0:
        thresh = otsu_threshold(image)

    result = np.zeros_like(image)
    result[image > thresh] = 255
    return result


def remove_background_hsv(
    image: ImageArray,
    low_h: int = 0,
    high_h: int = 180,
    low_s: int = 30,
    high_s: int = 255,
    low_v: int = 30,
    high_v: int = 255,
) -> ImageArray:
    """Create a foreground mask using HSV thresholding.

    Pixels within the HSV range are considered foreground (255),
    everything else is background (0).
    """
    hsv = bgr_to_hsv(image)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mask = np.zeros(h.shape, dtype=np.uint8)
    condition = (
        (h >= low_h) & (h <= high_h)
        & (s >= low_s) & (s <= high_s)
        & (v >= low_v) & (v <= high_v)
    )
    mask[condition] = 255
    return mask


def create_object_mask(image: ImageArray) -> ImageArray:
    """Create a binary mask isolating the foreground object.

    Uses a combination of Otsu thresholding on saturation +
    value channels to separate the object from a uniform background.
    """
    hsv = bgr_to_hsv(image)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Threshold saturation
    s_thresh = otsu_threshold(s_channel)
    s_mask = np.zeros_like(s_channel)
    s_mask[s_channel > max(s_thresh, 20)] = 255

    # Threshold value — use inverted for dark objects on light bg
    v_thresh = otsu_threshold(v_channel)
    v_mask_bright = np.zeros_like(v_channel)
    v_mask_bright[v_channel > v_thresh] = 255
    v_mask_dark = np.zeros_like(v_channel)
    v_mask_dark[v_channel <= v_thresh] = 255

    # Choose whichever value mask has better overlap with saturation mask
    overlap_bright = np.sum((s_mask > 0) & (v_mask_bright > 0))
    overlap_dark = np.sum((s_mask > 0) & (v_mask_dark > 0))
    v_mask = v_mask_bright if overlap_bright >= overlap_dark else v_mask_dark

    # Combine: pixel is foreground if either channel says so
    combined = np.zeros_like(s_channel)
    combined[(s_mask > 0) | (v_mask > 0)] = 255
    return combined


def crop_to_object(image: ImageArray) -> ImageArray:
    """Crop an image to the tight bounding box of the foreground object.

    Returns the cropped image with background removed (set to zero).
    """
    mask = create_object_mask(image)
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return image  # No foreground found, return original

    x, y, w, h = bbox
    return image[y : y + h, x : x + w].copy()


def mask_to_bbox(mask: ImageArray) -> BBox:
    """Compute the tight bounding box around non-zero pixels."""
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None  # type: ignore[return-value]

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return (int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1))
