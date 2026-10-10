"""CPU pipeline with per-model process isolation and sampled resident memory."""
import logging
import multiprocessing as mp
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from PIL import Image
from scipy.ndimage import gaussian_filter
from .dataset import read_image, decode_mask
from .segmentation import Config, normalize, measure, overlay

LOG = logging.getLogger(__name__)


def degrade(image: np.ndarray, mode: str, sigma: float, seed: int) -> np.ndarray:
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Degradation sigma must be finite and nonnegative")
    if mode == "none":
        return image
    x = normalize(image)
    if mode == "blur":
        return gaussian_filter(x, sigma)
    if mode == "noise":
        return np.clip(x + np.random.default_rng(seed).normal(0, sigma, x.shape), 0, 1)
    raise ValueError("Unknown degradation")


def _worker(connection, settings: dict, model: str) -> None:
    try:
        # These settings deliberately define a CPU comparison, including StarDist.
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
        from .segmenters import WatershedSegmenter, DeepLearningSegmenter
        image = read_image(settings["input"])
        x = degrade(image, settings["degradation"], settings["sigma"], settings["seed"])
        start = time.perf_counter()
        engine = (WatershedSegmenter(Config(**settings["segmentation"]), settings["classical_method"]) if model == "classical"
                  else DeepLearningSegmenter(settings["neural_model"]))
        initialization_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        labels = engine.segment(x)
        cold_ms = (time.perf_counter() - start) * 1000
        elapsed = []
        for _ in range(settings["repeats"]):
            start = time.perf_counter()
            labels = engine.segment(x)
            elapsed.append((time.perf_counter() - start) * 1000)
        output = Path(settings["output"]) / model
        output.mkdir(parents=True, exist_ok=True)
        measure(labels).to_csv(output / "measurements.csv", index=False)
        Image.fromarray(labels.astype(np.int32)).save(output / "labels.tif")
        Image.fromarray((overlay(x, labels) * 255).astype(np.uint8)).save(output / "overlay.png")
        import sys
        if sys.platform == "win32":
            info = psutil.Process().memory_info()
            peak_rss_mb = info.peak_wset / 1024 ** 2
        else:
            import resource
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_rss_mb = peak_rss / (1024 ** 2 if sys.platform == "darwin" else 1024)
        row = {"peak_process_rss_mb": peak_rss_mb, "model": model, "count": int(labels.max()), "initialization_ms": initialization_ms,
               "cold_inference_ms": cold_ms, "warm_median_ms": float(np.median(elapsed)),
               "repeats": settings["repeats"], "device": "cpu", "degradation": settings["degradation"],
               "sigma": settings["sigma"]}
        if settings.get("truth"):
            row.update(engine.evaluate(decode_mask(settings["truth"]), labels))
        connection.send((True, row))
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def run_model(settings: dict, model: str) -> dict:
    import os
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ[key] = "1"
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, settings, model))
    process.start()
    sender.close()
    peak = 0
    started = time.monotonic()
    try:
        while True:
            try:
                peak = max(peak, psutil.Process(process.pid).memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if receiver.poll(.01):
                try:
                    success, result = receiver.recv()
                except EOFError as exc:
                    raise RuntimeError(f"{model} worker exited without results") from exc
                break
            if not process.is_alive():
                raise RuntimeError(f"{model} worker exited with code {process.exitcode}")
            if time.monotonic() - started > settings["timeout_seconds"]:
                raise TimeoutError(f"{model} exceeded execution timeout")
        if not success:
            raise RuntimeError(result)
        result["sampled_peak_process_rss_mb"] = peak / 1024 ** 2 if peak else None
        LOG.info("%s: %s nuclei, warm inference %.2f ms, peak process RSS %.1f MiB",
                 model, result["count"], result["warm_median_ms"], result["peak_process_rss_mb"])
        return result
    finally:
        if process.is_alive():
            process.join(timeout=2)
        if process.is_alive():
            process.terminate()
        process.join()
        receiver.close()


def report(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "comparison.csv", index=False)
    metrics = [("warm_median_ms", "Warm inference (ms)"),
               ("peak_process_rss_mb", "Process peak RSS (MiB)")]
    if "instance_f1" in frame:
        metrics += [("instance_f1", "Instance F1 at IoU ≥ 0.5")]
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for ax, (metric, title) in zip(axes[0], metrics):
        ax.bar(frame.model, frame[metric], color=["#246b8e", "#de8531"][:len(frame)])
        ax.set_title(title)
        if metric == "instance_f1":
            ax.set_ylim(0, 1)
    figure.suptitle("Single-image CPU profile · not a dataset-wide benchmark")
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=300)
    figure.savefig(output / "comparison.pdf")
    plt.close(figure)
