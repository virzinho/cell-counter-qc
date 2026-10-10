"""Transparent classical baselines; no learned weights or diagnostic claims."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, filters, measure as skmeasure, morphology, segmentation
from skimage.feature import peak_local_max


@dataclass(frozen=True)
class Config:
    sigma: float = 1.0
    min_area: int = 80
    min_distance: int = 8
    threshold_factor: float = 0.8

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.sigma, self.min_area, self.min_distance, self.threshold_factor)):
            raise ValueError("Segmentation parameters must be finite")
        if self.sigma < 0 or self.min_area < 1 or self.min_distance < 1 or self.threshold_factor <= 0:
            raise ValueError("Invalid segmentation parameters")


def normalize(image: np.ndarray) -> np.ndarray:
    """Convert a single 2D image to grayscale and robustly scale to [0, 1]."""
    x = np.asarray(image)
    if x.ndim == 3 and x.shape[-1] in (3, 4):
        x = color.rgb2gray(x[..., :3])
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError("Use a single 2D grayscale or RGB image, not a volume or stack")
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Image contains NaN or infinite values")
    low, high = np.percentile(x, [1, 99.8])
    if high <= low:
        low, high = float(x.min()), float(x.max())
    if high <= low:
        return np.zeros_like(x)
    return np.clip((x - low) / (high - low), 0, 1)


def segment(image: np.ndarray, method: str = "watershed", config: Config = Config()) -> np.ndarray:
    if method not in ("connected", "watershed"):
        raise ValueError("Method must be connected or watershed")
    x = normalize(image)
    if not x.any():
        return np.zeros(x.shape, dtype=np.int32)
    smooth = filters.gaussian(x, sigma=config.sigma)
    foreground = smooth > filters.threshold_otsu(smooth) * config.threshold_factor
    foreground = morphology.remove_small_objects(foreground, max_size=config.min_area - 1)
    foreground = ndi.binary_fill_holes(foreground)
    if method == "connected":
        labels = skmeasure.label(foreground)
    else:
        distance = ndi.distance_transform_edt(foreground)
        peaks = peak_local_max(distance, min_distance=config.min_distance,
                               labels=foreground.astype(np.uint8), exclude_border=False)
        markers = np.zeros(x.shape, dtype=np.int32)
        if len(peaks):
            markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        # Guarantee a marker in small or flat components lacking a local peak.
        components = skmeasure.label(foreground)
        for region in skmeasure.regionprops(components):
            coords = region.coords
            if not markers[tuple(coords.T)].any():
                p = coords[np.argmax(distance[tuple(coords.T)])]
                markers[tuple(p)] = int(markers.max()) + 1
        labels = segmentation.watershed(-distance, markers, mask=foreground)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= config.min_area
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)
    return segmentation.relabel_sequential(labels)[0].astype(np.int32)


def measure(labels: np.ndarray) -> pd.DataFrame:
    columns = ["label", "area_px2", "centroid_y_px", "centroid_x_px", "equivalent_diameter_px", "eccentricity", "solidity"]
    rows = [[r.label, r.area, *r.centroid, r.equivalent_diameter_area, r.eccentricity, r.solidity]
            for r in skmeasure.regionprops(labels)]
    return pd.DataFrame(rows, columns=columns)


def overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return segmentation.mark_boundaries(normalize(image), labels, color=(0.2, 1.0, 0.7), mode="outer")


def quality(image: np.ndarray) -> dict[str, float]:
    """Descriptive indicators, not probabilities or calibrated pass/fail rules."""
    x = normalize(image)
    return {"focus_laplacian_variance": float(ndi.laplace(x).var()),
            "contrast_std": float(x.std()),
            "foreground_fraction": float((x > 0.25).mean())}
