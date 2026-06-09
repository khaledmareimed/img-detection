"""
Template Matching detection using NCC and SSD.

Implements Normalized Cross-Correlation and Sum of Squared Differences
from scratch using FFT-accelerated cross-correlation for speed.
Multi-scale support including anisotropic (width/height independent)
scaling for handling aspect ratio mismatches.
"""

import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve

from utils.types import ImageArray, DetectionResult
from utils.image_io import bgr_to_gray, resize_image


def _local_sums(img: np.ndarray, th: int, tw: int):
    """Compute local sums and squared sums using FFT convolution.

    Returns (local_sum, local_sq_sum) of shape (rh, rw).
    """
    img_f = img.astype(np.float64)
    kernel = np.ones((th, tw), dtype=np.float64)
    local_sum = fftconvolve(img_f, kernel[::-1, ::-1], mode="valid")
    local_sq_sum = fftconvolve(img_f ** 2, kernel[::-1, ::-1], mode="valid")
    return local_sum, local_sq_sum


def normalized_cross_correlation(
    scene: np.ndarray, template: np.ndarray
) -> np.ndarray:
    """Compute NCC similarity map using FFT cross-correlation.

    Both inputs must be 2-D grayscale arrays.
    Returns a 2-D array where higher values = better match.
    """
    scene_f = scene.astype(np.float64)
    tmpl_f = template.astype(np.float64)
    th, tw = tmpl_f.shape
    n = th * tw

    tmpl_mean = tmpl_f.mean()
    tmpl_centered = tmpl_f - tmpl_mean
    tmpl_energy = np.sqrt(np.sum(tmpl_centered ** 2))

    if tmpl_energy < 1e-8:
        rh = scene_f.shape[0] - th + 1
        rw = scene_f.shape[1] - tw + 1
        return np.zeros((max(1, rh), max(1, rw)))

    cross = fftconvolve(scene_f, tmpl_centered[::-1, ::-1], mode="valid")
    local_sum, local_sq_sum = _local_sums(scene_f, th, tw)
    local_mean = local_sum / n
    local_var = np.maximum(local_sq_sum / n - local_mean ** 2, 0.0)
    local_energy = np.sqrt(local_var * n)

    with np.errstate(divide="ignore", invalid="ignore"):
        ncc = np.where(
            local_energy > 1e-8,
            cross / (local_energy * tmpl_energy),
            0.0,
        )
    return np.clip(ncc, -1.0, 1.0)


def sum_squared_differences(
    scene: np.ndarray, template: np.ndarray
) -> np.ndarray:
    """Compute SSD similarity map using FFT (inverted: max = best).

    Mean-centers both the template and each scene patch before
    computing SSD so the metric matches structure/texture rather
    than absolute brightness.
    """
    scene_f = scene.astype(np.float64)
    tmpl_f = template.astype(np.float64)
    th, tw = tmpl_f.shape
    n = th * tw

    # Mean-center the template
    tmpl_centered = tmpl_f - tmpl_f.mean()
    tmpl_sq_sum = np.sum(tmpl_centered ** 2)

    # Per-window mean and centered squared sum via FFT
    local_sum, local_sq_sum = _local_sums(scene_f, th, tw)
    local_mean = local_sum / n
    # sum((patch - patch_mean)^2) = sum(patch^2) - n * mean^2
    local_sq_sum_centered = np.maximum(local_sq_sum - n * local_mean ** 2, 0.0)

    # Cross-correlation of scene with mean-centered template
    cross = fftconvolve(scene_f, tmpl_centered[::-1, ::-1], mode="valid")
    # Subtract mean contribution: sum((patch-mu) * tmpl_c) = cross - mu * sum(tmpl_c)
    # sum(tmpl_centered) == 0 by construction, so no correction needed.

    ssd = np.maximum(local_sq_sum_centered - 2.0 * cross + tmpl_sq_sum, 0.0)
    max_ssd = ssd.max()
    if max_ssd > 0:
        return 1.0 - ssd / max_ssd
    return np.ones_like(ssd)


def _generate_scale_pairs(
    th: int, tw: int, sh: int, sw: int,
) -> List[Tuple[float, float]]:
    """Generate (scale_h, scale_w) pairs for anisotropic search.

    Produces uniform scales AND cross-combinations of
    different height/width scales to handle aspect ratio changes.
    Anisotropic pairs are limited to aspect ratios within 1.5× of
    each other to avoid distorted template matches.
    """
    # Base scale values — start at 0.3 to avoid tiny templates
    base = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Filter to valid sizes (minimum 20px dimension)
    valid_h = [s for s in base if 20 <= int(th * s) < sh]
    valid_w = [s for s in base if 20 <= int(tw * s) < sw]

    if not valid_h:
        valid_h = [1.0]
    if not valid_w:
        valid_w = [1.0]

    pairs = set()
    # Uniform scales
    for s in base:
        if s in valid_h and s in valid_w:
            pairs.add((s, s))

    # Anisotropic: only combos with aspect ratio within 1.5×
    for sh_s in valid_h:
        for sw_s in valid_w:
            ratio = max(sh_s, sw_s) / max(min(sh_s, sw_s), 1e-6)
            if ratio <= 1.5:
                pairs.add((sh_s, sw_s))

    return sorted(pairs)


def _resize_anisotropic(
    img: np.ndarray, scale_h: float, scale_w: float
) -> np.ndarray:
    """Resize with independent height/width scale factors."""
    import cv2
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale_w))
    new_h = max(1, int(h * scale_h))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def detect_template(
    scene: ImageArray,
    template: ImageArray,
    method: str = "ncc",
    scales: Optional[List[float]] = None,
) -> DetectionResult:
    """Full template matching pipeline with multi-scale support.

    When the template is larger than half the scene, it automatically
    tries anisotropic scaling (independent W/H) to handle aspect
    ratio mismatches between the template and the scene object.
    """
    start = time.perf_counter()

    scene_gray = bgr_to_gray(scene) if len(scene.shape) == 3 else scene
    tmpl_gray = bgr_to_gray(template) if len(template.shape) == 3 else template
    sh, sw = scene_gray.shape
    th, tw = tmpl_gray.shape

    match_fn = (
        normalized_cross_correlation if method == "ncc"
        else sum_squared_differences
    )

    # Build scale pairs
    if scales is not None:
        # User-provided uniform scales
        scale_pairs = [(s, s) for s in scales]
    else:
        # Always do multi-scale anisotropic search
        scale_pairs = _generate_scale_pairs(th, tw, sh, sw)

    best_score = -1.0
    best_bbox = (0, 0, tw, th)
    best_map: Optional[np.ndarray] = None

    for s_h, s_w in scale_pairs:
        if abs(s_h - 1.0) < 1e-6 and abs(s_w - 1.0) < 1e-6:
            scaled = tmpl_gray
        else:
            scaled = _resize_anisotropic(tmpl_gray, s_h, s_w)

        sth, stw = scaled.shape
        if sth > sh or stw > sw or sth < 20 or stw < 20:
            continue

        sim_map = match_fn(scene_gray, scaled)
        max_val = sim_map.max()

        if max_val > best_score:
            best_score = max_val
            best_map = sim_map
            loc = np.unravel_index(np.argmax(sim_map), sim_map.shape)
            best_bbox = (int(loc[1]), int(loc[0]), stw, sth)

    elapsed = (time.perf_counter() - start) * 1000.0

    return DetectionResult(
        method_name=f"Template Matching ({method.upper()})",
        bounding_box=best_bbox,
        confidence=float(np.clip(best_score, 0, 1)),
        execution_time_ms=elapsed,
        similarity_map=best_map,
    )


# ---------------------------------------------------------------------------
# Rotation-aware wrapper
# ---------------------------------------------------------------------------

def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by angle_deg (counterclockwise) with full canvas."""
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


def detect_template_rotation(
    scene: ImageArray,
    template: ImageArray,
    method: str = "ncc",
    angles: Optional[List[float]] = None,
    scales: Optional[List[float]] = None,
) -> DetectionResult:
    """Template matching with rotation search.

    Tries the template at multiple rotation angles and picks the
    angle + position that gives the highest similarity score.
    Default angles cover 0/90/180/270 (handles 90-deg mismatches)
    plus fine steps near 0 and 90 for slight tilts.
    """
    if angles is None:
        angles = [0, 45, 90, 135, 180, 225, 270, 315,
                  -15, -10, -5, 5, 10, 15,
                  75, 80, 85, 95, 100, 105]

    best_score = -1.0
    best_result: Optional[DetectionResult] = None
    best_angle = 0.0

    for angle in angles:
        rotated = template if abs(angle) < 1e-3 else _rotate_image(template, angle)
        result = detect_template(scene, rotated, method=method, scales=scales)
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
