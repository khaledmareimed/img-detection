"""
Edge-Based Object Detection.

Implements Sobel and Canny edge detectors with custom kernels but
fast cv2-based convolution.  Edge overlap matching uses vectorized
NumPy for performance.
"""

import time
from typing import List, Optional

import cv2
import numpy as np
from scipy.signal import fftconvolve

from utils.types import ImageArray, DetectionResult
from utils.image_io import bgr_to_gray, resize_image
from core.preprocessing.filtering import convolve2d, create_gaussian_kernel


# 3×3 Sobel kernels (defined from scratch)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def sobel_gradients(image: np.ndarray):
    """Compute gradient magnitude and direction using Sobel operators."""
    img_f = image.astype(np.float64)
    gx = convolve2d(img_f, SOBEL_X)
    gy = convolve2d(img_f, SOBEL_Y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    return magnitude, direction


def sobel_edge_detection(
    image: ImageArray, threshold: int = 50
) -> ImageArray:
    """Compute binary edge map using custom Sobel operators."""
    if len(image.shape) == 3:
        image = bgr_to_gray(image)
    magnitude, _ = sobel_gradients(image)
    edges = np.zeros_like(image)
    edges[magnitude > threshold] = 255
    return edges


def canny_edge_detection(
    image: ImageArray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    sigma: float = 1.4,
) -> ImageArray:
    """Full Canny edge detection pipeline (vectorized).

    Steps: Gaussian smoothing → Sobel gradients → Non-max suppression
           → Double threshold → Hysteresis.
    """
    if len(image.shape) == 3:
        image = bgr_to_gray(image)

    # Step 1: Gaussian smoothing (fast via cv2.filter2D)
    kernel = create_gaussian_kernel(5, sigma)
    smoothed = convolve2d(image.astype(np.float64), kernel)

    # Step 2: Sobel gradients
    magnitude, direction = sobel_gradients(smoothed.astype(np.uint8))

    # Step 3: Non-maximum suppression (vectorized)
    suppressed = _non_max_suppression_fast(magnitude, direction)

    # Step 4–5: Double threshold + hysteresis
    edges = _hysteresis_fast(suppressed, low_threshold, high_threshold)
    return edges.astype(np.uint8)


def _non_max_suppression_fast(
    magnitude: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Vectorized non-maximum suppression."""
    h, w = magnitude.shape
    result = np.zeros_like(magnitude)

    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    # Quantise angles into 4 directions
    # 0: horizontal (0, 180)
    # 1: diagonal /  (45)
    # 2: vertical    (90)
    # 3: diagonal \  (135)

    inner_mag = magnitude[1:-1, 1:-1]
    inner_angle = angle[1:-1, 1:-1]

    # Neighbours along each direction
    # Direction 0 (horizontal): compare left and right
    mask0 = ((inner_angle < 22.5) | (inner_angle >= 157.5))
    n1_0 = magnitude[1:-1, :-2]  # left
    n2_0 = magnitude[1:-1, 2:]   # right

    # Direction 1 (diagonal /): compare top-right and bottom-left
    mask1 = ((inner_angle >= 22.5) & (inner_angle < 67.5))
    n1_1 = magnitude[:-2, 2:]    # top-right
    n2_1 = magnitude[2:, :-2]    # bottom-left

    # Direction 2 (vertical): compare top and bottom
    mask2 = ((inner_angle >= 67.5) & (inner_angle < 112.5))
    n1_2 = magnitude[:-2, 1:-1]  # top
    n2_2 = magnitude[2:, 1:-1]   # bottom

    # Direction 3 (diagonal \): compare top-left and bottom-right
    mask3 = ((inner_angle >= 112.5) & (inner_angle < 157.5))
    n1_3 = magnitude[:-2, :-2]   # top-left
    n2_3 = magnitude[2:, 2:]     # bottom-right

    # Build the max-neighbour arrays
    n1 = np.where(mask0, n1_0, np.where(mask1, n1_1, np.where(mask2, n1_2, n1_3)))
    n2 = np.where(mask0, n2_0, np.where(mask1, n2_1, np.where(mask2, n2_2, n2_3)))

    # Keep pixel only if it's >= both neighbours
    keep = (inner_mag >= n1) & (inner_mag >= n2)
    result[1:-1, 1:-1] = np.where(keep, inner_mag, 0)

    return result


def _hysteresis_fast(
    suppressed: np.ndarray, low: int, high: int
) -> np.ndarray:
    """Fast hysteresis thresholding using cv2.connectedComponents."""
    strong = (suppressed >= high).astype(np.uint8) * 255
    weak = (suppressed >= low).astype(np.uint8) * 255

    # Dilate strong edges to connect to nearby weak edges
    kernel = np.ones((3, 3), dtype=np.uint8)
    # Iteratively grow strong edges into weak edge regions
    result = strong.copy()
    for _ in range(5):
        dilated = cv2.dilate(result, kernel, iterations=1)
        result = np.where((dilated > 0) & (weak > 0), 255, result)
        result = result.astype(np.uint8)

    return result


# -------------------------------------------------------------------
# Edge-Based Detection
# -------------------------------------------------------------------

def edge_overlap_score(
    scene_edges: np.ndarray, tmpl_edges: np.ndarray
) -> np.ndarray:
    """Compute edge overlap score map — vectorized.

    Uses FFT convolution in 'valid' mode to correctly compute
    the number of overlapping edge pixels at every position.
    """
    th, tw = tmpl_edges.shape
    sh, sw = scene_edges.shape
    rh = sh - th + 1
    rw = sw - tw + 1

    if rh <= 0 or rw <= 0:
        return np.array([[0.0]])

    scene_f = (scene_edges > 0).astype(np.float64)
    tmpl_f = (tmpl_edges > 0).astype(np.float64)
    tmpl_count = tmpl_f.sum()

    if tmpl_count < 1:
        return np.zeros((rh, rw))

    # FFT convolution in 'valid' mode gives the correct
    # cross-correlation output without manual cropping.
    overlap_map = fftconvolve(
        scene_f, tmpl_f[::-1, ::-1], mode="valid"
    )

    return np.maximum(overlap_map, 0.0) / tmpl_count


def detect_by_edges(
    scene: ImageArray,
    template: ImageArray,
    edge_method: str = "canny",
    scales: Optional[List[float]] = None,
) -> DetectionResult:
    """Full edge-based detection pipeline."""
    start = time.perf_counter()

    scene_gray = bgr_to_gray(scene) if len(scene.shape) == 3 else scene
    tmpl_gray = bgr_to_gray(template) if len(template.shape) == 3 else template

    edge_fn = canny_edge_detection if edge_method == "canny" else sobel_edge_detection

    # Auto-generate scales if template is too large
    if scales is None:
        sh, sw = scene_gray.shape
        th, tw = tmpl_gray.shape
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

    best_score = -1.0
    best_bbox = (0, 0, tmpl_gray.shape[1], tmpl_gray.shape[0])
    best_map: Optional[np.ndarray] = None

    # Compute scene edges once outside the scale loop
    scene_edges = edge_fn(scene_gray)

    for scale in scales:
        if abs(scale - 1.0) > 1e-6:
            scaled_tmpl = resize_image(tmpl_gray, scale=scale)
        else:
            scaled_tmpl = tmpl_gray

        sth, stw = scaled_tmpl.shape
        if sth > scene_gray.shape[0] or stw > scene_gray.shape[1]:
            continue
        if sth < 20 or stw < 20:
            continue

        tmpl_edges = edge_fn(scaled_tmpl)
        tmpl_edge_density = (tmpl_edges > 0).mean()
        if tmpl_edge_density < 0.01:
            continue
        sim_map = edge_overlap_score(scene_edges, tmpl_edges)
        max_val = sim_map.max()

        if max_val > best_score:
            best_score = max_val
            best_map = sim_map
            loc = np.unravel_index(np.argmax(sim_map), sim_map.shape)
            best_bbox = (int(loc[1]), int(loc[0]), stw, sth)

    elapsed = (time.perf_counter() - start) * 1000.0
    return DetectionResult(
        method_name=f"Edge Detection ({edge_method.capitalize()})",
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


def detect_by_edges_rotation(
    scene: ImageArray,
    template: ImageArray,
    edge_method: str = "canny",
    angles: Optional[List[float]] = None,
) -> DetectionResult:
    """Edge-based detection with rotation search."""
    if angles is None:
        angles = [0, 45, 90, 135, 180, 225, 270, 315,
                  -15, -10, -5, 5, 10, 15,
                  75, 80, 85, 95, 100, 105]

    best_score = -1.0
    best_result: Optional[DetectionResult] = None
    best_angle = 0.0

    for angle in angles:
        rotated = template if abs(angle) < 1e-3 else _rotate_image(template, angle)
        result = detect_by_edges(scene, rotated, edge_method=edge_method)
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
