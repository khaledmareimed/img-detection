"""
Preprocessing pipeline — applies a chain of preprocessing steps
based on user configuration.
"""

from utils.types import ImageArray, PreprocessingConfig


def apply_preprocessing_pipeline(
    image: ImageArray,
    flags: dict,
    config: PreprocessingConfig,
) -> ImageArray:
    """Apply enabled preprocessing steps to an image.

    Steps are applied in a fixed order to ensure consistent results:
    1. Background removal (crop to object)
    2. Histogram equalization
    3. CLAHE
    4. Gaussian blur
    5. Median filter
    6. Contrast stretching

    Args:
        image: Input image (BGR).
        flags: Dict of ``{step_key: bool}`` toggles from the control panel.
        config: Detailed parameter configuration from the preprocessing panel.

    Returns:
        Preprocessed copy of the image.
    """
    result = image.copy()

    if flags.get("bg_removal"):
        from core.preprocessing.segmentation import crop_to_object
        result = crop_to_object(result)

    if flags.get("hist_eq"):
        from core.preprocessing.enhancement import histogram_equalization
        result = histogram_equalization(result)

    if flags.get("clahe"):
        from core.preprocessing.enhancement import clahe
        result = clahe(
            result, config.clahe_clip_limit, config.clahe_tile_size
        )

    if flags.get("gaussian"):
        from core.preprocessing.filtering import gaussian_blur
        result = gaussian_blur(
            result, config.gaussian_kernel_size, config.gaussian_sigma
        )

    if flags.get("median"):
        from core.preprocessing.filtering import median_filter
        result = median_filter(result, config.median_kernel_size)

    if flags.get("contrast"):
        from core.preprocessing.enhancement import contrast_stretching
        result = contrast_stretching(
            result, config.contrast_low_pct, config.contrast_high_pct
        )

    return result
