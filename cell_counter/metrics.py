"""Instance matching at IoU >= 0.5, plus count and foreground metrics."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import relabel_sequential


def evaluate(truth: np.ndarray, prediction: np.ndarray, threshold: float = 0.5) -> dict:
    a = relabel_sequential(np.asarray(truth, dtype=np.int64))[0]
    b = relabel_sequential(np.asarray(prediction, dtype=np.int64))[0]
    if a.shape != b.shape:
        raise ValueError("Ground truth and prediction must have the same shape")
    na, nb = int(a.max()), int(b.max())
    table = np.bincount((a * (nb + 1) + b).ravel(), minlength=(na + 1) * (nb + 1)).reshape(na + 1, nb + 1)
    intersection = table[1:, 1:]
    union = table.sum(1)[1:, None] + table.sum(0)[None, 1:] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
    tp = 0
    if na and nb:
        # Maximize valid match count first, IoU only breaks ties.
        score = (iou >= threshold).astype(float) + iou / (min(na, nb) + 1)
        rows, cols = linear_sum_assignment(-score)
        tp = int((iou[rows, cols] >= threshold).sum())
    fp, fn = nb - tp, na - tp
    af, bf = a > 0, b > 0
    denominator = int(af.sum() + bf.sum())
    return {"true_count": na, "predicted_count": nb, "count_error": nb - na,
            "absolute_count_error": abs(nb - na), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / nb if nb else (1.0 if na == 0 else 0.0),
            "recall": tp / na if na else 1.0,
            "instance_f1": 2 * tp / (2 * tp + fp + fn) if na + nb else 1.0,
            "foreground_dice": 2 * (af & bf).sum() / denominator if denominator else 1.0}
