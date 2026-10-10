"""Interchangeable segmentation strategies with explicit preprocessing contracts."""
from abc import ABC, abstractmethod
import numpy as np
from .segmentation import Config, normalize, segment
from .metrics import evaluate


class BaseSegmenter(ABC):
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return normalize(image)

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Accept raw intensities and return integer instance labels."""

    def evaluate(self, truth: np.ndarray, prediction: np.ndarray) -> dict:
        return evaluate(truth, prediction)


class WatershedSegmenter(BaseSegmenter):
    def __init__(self, config: Config = Config(), method: str = "watershed") -> None:
        if method not in {"watershed", "connected"}:
            raise ValueError("Unknown classical method")
        self.config, self.method = config, method

    def segment(self, image: np.ndarray) -> np.ndarray:
        # The existing function owns preprocessing, preserving notebook results.
        return segment(image, self.method, self.config)


class DeepLearningSegmenter(BaseSegmenter):
    """Optional pretrained StarDist; imports and weights load only on selection."""
    def __init__(self, model_name: str = "2D_versatile_fluo") -> None:
        try:
            from stardist.models import StarDist2D
            from csbdeep.utils import normalize as neural_normalize
        except ImportError as exc:
            raise RuntimeError("Install neural dependencies: pip install -e '.[neural]'") from exc
        self._normalize = neural_normalize
        self.model = StarDist2D.from_pretrained(model_name)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Validate shape/finiteness, but use the model's documented normalization.
        normalize(image)
        if image.ndim != 2:
            raise ValueError("StarDist fluorescence model requires a 2D grayscale image")
        return self._normalize(image, 1, 99.8).astype(np.float32)

    def segment(self, image: np.ndarray) -> np.ndarray:
        x = self.preprocess(image)
        if not np.any(x):
            return np.zeros(x.shape, dtype=np.int32)
        labels, _ = self.model.predict_instances(x)
        return labels.astype(np.int32)
