"""
Color Histogram Matching detection.

Computes HS color histograms and uses sliding window comparison
with Bhattacharyya or Chi-Square distance.  HSV conversion is done
once and histogram bins are precomputed for speed.
"""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import fftconvolve

from utils.types import ImageArray, DetectionResult
from utils.image_io import bgr_to_hsv, resize_image


def compute_color_histogram(
    image: ImageArray,
    bins: Tuple[int, int] = (16, 16),
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a 2-D Hue-Saturation histogram (vectorized).

    Args:
        image: BGR colour image.
        bins: (h_bins, s_bins) number of bins per channel.
        mask: Optional binary mask.

    Returns:
        Normalised 2-D histogram of shape (h_bins, s_bins).
    """
    hsv = bgr_to_hsv(image)
    h_ch = hsv[:, :, 0].ravel()  # 0–179
    s_ch = hsv[:, :, 1].ravel()  # 0–255

    h_bins, s_bins = bins
    h_idx = np.clip((h_ch * h_bins / 180.0).astype(int), 0, h_bins - 1)
    s_idx = np.clip((s_ch * s_bins / 256.0).astype(int), 0, s_bins - 1)

    if mask is not None:
        valid = mask.ravel() > 0
        h_idx = h_idx[valid]
        s_idx = s_idx[valid]

    flat_idx = h_idx * s_bins + s_idx
    hist = np.bincount(flat_idx, minlength=h_bins * s_bins).astype(np.float64)
    hist = hist.reshape(h_bins, s_bins)

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Bhattacharyya distance between two histograms. 0 = identical."""
    bc = np.sum(np.sqrt(hist1 * hist2))
    return float(np.sqrt(max(0.0, 1.0 - np.clip(bc, 0.0, 1.0))))


def chi_square_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Chi-Square distance between two histograms. 0 = identical."""
    denom = hist1 + hist2
    valid = denom > 1e-10
    return float(np.sum((hist1[valid] - hist2[valid]) ** 2 / denom[valid]))


def histogram_similarity(
    hist1: np.ndarray, hist2: np.ndarray, metric: str = "bhattacharyya"
) -> float:
    """Similarity score in [0, 1] (higher = more similar)."""
    if metric == "bhattacharyya":
        return 1.0 - bhattacharyya_distance(hist1, hist2)
    elif metric == "chi_square":
        return 1.0 / (1.0 + chi_square_distance(hist1, hist2))
    raise ValueError(f"Unknown metric: {metric}")


def _fast_sliding_histogram(
    scene_hsv: np.ndarray,
    tmpl_hist: np.ndarray,
    window_h: int,
    window_w: int,
    bins: Tuple[int, int],
    metric: str,
    stride: int,
) -> np.ndarray:
    """Fast sliding-window histogram comparison.

    Pre-bins the HSV channels as one-hot maps, then uses FFT
    convolution to compute per-bin counts in every window position.
    """
    h_bins, s_bins = bins
    total_bins = h_bins * s_bins

    # Pre-compute bin indices for the entire scene
    h_idx = np.clip(
        (scene_hsv[:, :, 0].astype(np.float64) * h_bins / 180.0).astype(int),
        0, h_bins - 1,
    )
    s_idx = np.clip(
        (scene_hsv[:, :, 1].astype(np.float64) * s_bins / 256.0).astype(int),
        0, s_bins - 1,
    )
    flat_idx = h_idx * s_bins + s_idx

    sh, sw = scene_hsv.shape[:2]
    rh = sh - window_h + 1
    rw = sw - window_w + 1
    if rh <= 0 or rw <= 0:
        return np.array([[0.0]])

    # Build per-bin count maps using FFT convolution
    kernel = np.ones((window_h, window_w), dtype=np.float64)
    tmpl_flat = tmpl_hist.ravel()
    n_pixels = window_h * window_w

    # For Bhattacharyya: score = sum(sqrt(p*q)) where p, q are normalised
    # We compute bin counts via convolution, then normalise and compare
    sim_map = np.zeros((rh, rw), dtype=np.float64)

    for b in range(total_bins):
        if tmpl_flat[b] < 1e-10:
            continue
        bin_mask = (flat_idx == b).astype(np.float64)
        count_map = fftconvolve(bin_mask, kernel[::-1, ::-1], mode="valid")
        patch_prob = np.maximum(count_map / n_pixels, 0.0)

        if metric == "bhattacharyya":
            sim_map += np.sqrt(patch_prob * tmpl_flat[b])
        else:
            # Chi-square accumulation
            denom = patch_prob + tmpl_flat[b]
            with np.errstate(divide="ignore", invalid="ignore"):
                chi = np.where(
                    denom > 1e-10,
                    (patch_prob - tmpl_flat[b]) ** 2 / denom,
                    0.0,
                )
            sim_map += chi

    if metric == "bhattacharyya":
        # Bhattacharyya coefficient → similarity
        sim_map = np.clip(sim_map, 0.0, 1.0)
    else:
        # Chi-square distance → similarity
        sim_map = 1.0 / (1.0 + sim_map)

    # Subsample to stride
    if stride > 1:
        sim_map = sim_map[::stride, ::stride]

    return sim_map


def detect_by_histogram(
    scene: ImageArray,
    template: ImageArray,
    metric: str = "bhattacharyya",
    stride: int = 8,
    bins: Tuple[int, int] = (16, 16),
    scales: Optional[List[float]] = None,
) -> DetectionResult:
    """Full colour histogram detection with FFT-accelerated sliding window.

    Pre-bins the scene HSV channels and uses FFT convolution to compute
    per-bin counts, avoiding per-window histogram recomputation.
    """
    start = time.perf_counter()
    th, tw = template.shape[:2]

    if scales is None:
        sh, sw = scene.shape[:2]
        if th > sh * 0.5 or tw > sw * 0.5:
            scales = [s for s in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                       if int(th * s) < sh and int(tw * s) < sw
                       and min(int(th * s), int(tw * s)) >= 20]
            if not scales:
                scales = [1.0]
        else:
            # Always try multiple scales even for smaller templates
            scales = [s for s in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                       if int(th * s) <= sh and int(tw * s) <= sw
                       and min(int(th * s), int(tw * s)) >= 20]
            if not scales:
                scales = [1.0]

    scene_hsv = bgr_to_hsv(scene)

    best_score = -1.0
    best_bbox = (0, 0, tw, th)
    best_map: Optional[np.ndarray] = None

    for scale in scales:
        if abs(scale - 1.0) > 1e-6:
            scaled_tmpl = resize_image(template, scale=scale)
        else:
            scaled_tmpl = template

        sth, stw = scaled_tmpl.shape[:2]
        if sth > scene.shape[0] or stw > scene.shape[1]:
            continue
        if sth < 20 or stw < 20:
            continue

        tmpl_hist = compute_color_histogram(scaled_tmpl, bins=bins)

        sim_map = _fast_sliding_histogram(
            scene_hsv, tmpl_hist, sth, stw, bins, metric, stride
        )

        max_val = sim_map.max()
        if max_val > best_score:
            best_score = max_val
            best_map = sim_map
            loc = np.unravel_index(np.argmax(sim_map), sim_map.shape)
            # Map from subsampled sim_map coords back to image coords.
            # Then refine at stride=1 in a local neighbourhood for
            # precise localization.
            coarse_y = int(loc[0] * stride)
            coarse_x = int(loc[1] * stride)

            # Refinement: re-run at stride=1 in a ±stride window
            if stride > 1:
                sh_img, sw_img = scene.shape[:2]
                y_lo = max(0, coarse_y - stride)
                y_hi = min(sh_img - sth, coarse_y + stride)
                x_lo = max(0, coarse_x - stride)
                x_hi = min(sw_img - stw, coarse_x + stride)

                best_refine_score = -1.0
                best_rx, best_ry = coarse_x, coarse_y
                for ry in range(y_lo, y_hi + 1):
                    for rx in range(x_lo, x_hi + 1):
                        patch = scene[ry:ry + sth, rx:rx + stw]
                        ph = compute_color_histogram(patch, bins=bins)
                        sc = histogram_similarity(tmpl_hist, ph, metric)
                        if sc > best_refine_score:
                            best_refine_score = sc
                            best_rx, best_ry = rx, ry
                best_bbox = (best_rx, best_ry, stw, sth)
                if best_refine_score > best_score:
                    best_score = best_refine_score
            else:
                best_bbox = (coarse_x, coarse_y, stw, sth)

    elapsed = (time.perf_counter() - start) * 1000.0
    return DetectionResult(
        method_name=f"Histogram Matching ({metric.capitalize()})",
        bounding_box=best_bbox,
        confidence=float(np.clip(best_score, 0, 1)),
        execution_time_ms=elapsed,
        similarity_map=best_map,
    )


# ---------------------------------------------------------------------------
# Rotation-aware wrapper
# ---------------------------------------------------------------------------

def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by angle_deg with full canvas (no cropping)."""
    import cv2
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    return cv2.warpAffine(img, M, (new_w, new_h))


def detect_by_histogram_rotation(
    scene: ImageArray,
    template: ImageArray,
    metric: str = "bhattacharyya",
    stride: int = 8,
    bins: Tuple[int, int] = (16, 16),
    angles: Optional[List[float]] = None,
) -> "DetectionResult":
    """Histogram matching with rotation search."""
    if angles is None:
        angles = [0, 45, 90, 135, 180, 225, 270, 315,
                  -15, -10, -5, 5, 10, 15,
                  75, 80, 85, 95, 100, 105]

    best_score = -1.0
    best_result = None
    best_angle = 0.0

    for angle in angles:
        rotated = template if abs(angle) < 1e-3 else _rotate_image(template, angle)
        result = detect_by_histogram(scene, rotated, metric=metric, stride=stride, bins=bins)
        if result.confidence > best_score:
            best_score = result.confidence
            best_result = result
            best_angle = angle

    assert best_result is not None
    return DetectionResult(
        method_name=f"{best_result.method_name} [rot={best_angle:.0f}°]",
        bounding_box=best_result.bounding_box,
        confidence=best_result.confidence,
        execution_time_ms=best_result.execution_time_ms,
        similarity_map=best_result.similarity_map,
    )
