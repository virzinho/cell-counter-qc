"""Select parameters on validation only; evaluate once on the published test split."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cell_counter.dataset import load_sample, split_names
from cell_counter.metrics import evaluate
from cell_counter.segmentation import Config, segment, normalize, overlay, quality


def conditions(image, name):
    x = normalize(image)
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    # Conditions operate in normalized intensity space, then the pipeline
    # normalizes them again. Thus pure brightness scaling is a control.
    return {"original": x, "blur_sigma_2": gaussian_filter(x, 2),
            "blur_sigma_4": gaussian_filter(x, 4), "noise_std_0.08": np.clip(x + rng.normal(0, .08, x.shape), 0, 1),
            "brightness_x0.4": x * .4,
            "blur2_noise0.08": np.clip(gaussian_filter(x, 2) + rng.normal(0, .08, x.shape), 0, 1)}


def bootstrap_mean(values, seed=42):
    v = np.asarray(values, dtype=float)
    samples = np.random.default_rng(seed).choice(v, size=(2000, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/bbbc039")
    args = parser.parse_args()
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    names = {s: split_names(args.data, s) for s in ["training", "validation", "test"]}
    for s, t in [("training", "validation"), ("training", "test"), ("validation", "test")]:
        assert not set(names[s]) & set(names[t]), "Overlapping splits"
    validation = [(n, *load_sample(args.data, n)) for n in names["validation"]]
    candidates = [Config(min_area=a, min_distance=d, threshold_factor=f)
                  for a in [40, 80] for d in [6, 10] for f in [.7, 1.0]]
    tuning = []
    selected = {}
    for method in ["connected", "watershed"]:
        best = None
        # Peak distance does not affect connected components; avoid duplicate runs.
        configs = candidates if method == "watershed" else [c for c in candidates if c.min_distance == 6]
        for config in configs:
            metrics = [evaluate(gt, segment(x, method, config)) for _, x, gt in validation]
            f1 = np.mean([m["instance_f1"] for m in metrics])
            mae = np.mean([m["absolute_count_error"] for m in metrics])
            tuning.append({"method": method, **asdict(config), "validation_f1": f1, "validation_count_mae": mae})
            if best is None or (f1, -mae) > best[0]:
                best = ((f1, -mae), config)
            print(f"Validation {method} {asdict(config)} F1={f1:.3f}", flush=True)
        selected[method] = best[1]
    pd.DataFrame(tuning).to_csv(out / "validation_search.csv", index=False)
    (out / "selected_config.json").write_text(json.dumps({m: asdict(c) for m, c in selected.items()}, indent=2) + "\n")
    rows = []
    for i, name in enumerate(names["test"]):
        image, truth = load_sample(args.data, name)
        for condition, x in conditions(image, name).items():
            for method, config in selected.items():
                start = time.perf_counter()
                prediction = segment(x, method, config)
                elapsed = time.perf_counter() - start
                rows.append({"image": name, "condition": condition, "method": method,
                             **evaluate(truth, prediction), **quality(x), "seconds": elapsed})
        print(f"Test {i + 1}/{len(names['test'])}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image_results.csv", index=False)
    summary = []
    for (method, condition), group in df.groupby(["method", "condition"], sort=True):
        row = {"method": method, "condition": condition, "n_images": len(group)}
        for metric in ["absolute_count_error", "instance_f1", "foreground_dice"]:
            mean, low, high = bootstrap_mean(group[metric])
            row.update({metric: mean, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
        row["median_seconds"] = group.seconds.median()
        row["micro_instance_f1"] = 2 * group.tp.sum() / (2 * group.tp.sum() + group.fp.sum() + group.fn.sum())
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    meta = {"dataset": "BBBC039v1", "split_sizes": {s: len(n) for s, n in names.items()},
            "parameter_selection": "Maximum mean validation instance F1, tie broken by count MAE",
            "seed": 42, "bootstrap_resamples": 2000, "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-image", "pandas", "matplotlib"]}}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    order = list(conditions(np.zeros((5, 5)), "plot"))
    labels = ["Original", "Blur σ=2", "Blur σ=4", "Noise .08", "Brightness ×.4", "Blur + noise"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for method, color in [("connected", "#8B8FA8"), ("watershed", "#007E87")]:
        g = summary[summary.method == method].set_index("condition").loc[order]
        for ax, metric in zip(axs, ["instance_f1", "absolute_count_error"]):
            y = g[metric].to_numpy()
            ax.errorbar(range(len(order)), y, yerr=np.vstack([y - g[f"{metric}_ci_low"], g[f"{metric}_ci_high"] - y]),
                        label=method.title(), color=color, marker="o", capsize=3)
            ax.set_xticks(range(len(order)), labels, rotation=25, ha="right")
            ax.grid(axis="y", alpha=.15)
    axs[0].set(title="Instance matching (IoU ≥ 0.5)", ylabel="Mean instance F1", ylim=(0, 1))
    axs[1].set(title="Counting error", ylabel="Mean absolute count error / image")
    axs[0].legend(frameon=False)
    fig.suptitle(f"BBBC039 · {len(names['test'])} held-out images · 95% image-bootstrap intervals", fontsize=14)
    fig.savefig(out / "robustness.png", dpi=160)
    plt.close(fig)
    # Use the first official validation image for the demo, avoiding test cherry-picking.
    name, image, truth = validation[0]
    prediction = segment(image, "watershed", selected["watershed"])
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, display, title in zip(axs, [normalize(image), overlay(image, truth), overlay(image, prediction)],
                                  ["Fluorescence image", f"Reference: {truth.max()} nuclei", f"Watershed: {prediction.max()} nuclei"]):
        ax.imshow(display, cmap="gray"); ax.set_title(title); ax.axis("off")
    fig.suptitle("Validation example · nuclei boundaries", fontsize=14)
    fig.savefig(out / "segmentation_example.png", dpi=160)
    plt.close(fig)
    originals = df[(df.condition == "original") & (df.method == "watershed")]
    # Show the worst original test failure explicitly, alongside best and median.
    order_failure = originals.sort_values("instance_f1")
    picked = [order_failure.iloc[0], order_failure.iloc[len(order_failure)//2], order_failure.iloc[-1]]
    fig, axs = plt.subplots(3, 2, figsize=(10, 10), layout="constrained")
    for i, row in enumerate(picked):
        image, truth = load_sample(args.data, row.image)
        prediction = segment(image, "watershed", selected["watershed"])
        axs[i, 0].imshow(overlay(image, truth)); axs[i, 1].imshow(overlay(image, prediction))
        axs[i, 0].set_title(f"Reference: {truth.max()} nuclei")
        axs[i, 1].set_title(f"Prediction: {prediction.max()} · F1 {row.instance_f1:.2f}")
        for ax in axs[i]: ax.axis("off")
    fig.suptitle("Failure analysis · worst, median and best test F1", fontsize=14)
    fig.savefig(out / "failure_analysis.png", dpi=140)
    plt.close(fig)
    print(summary[["method", "condition", "absolute_count_error", "instance_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
