"""Configurable CLI; existing positional invocation remains supported."""
import argparse
import json
import logging
from pathlib import Path
import sys
import math
import importlib.metadata
import platform
import yaml
from .segmentation import Config
from .pipeline import run_model, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile classical / pretrained neural nuclei segmentation on CPU")
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--model", choices=["classical", "neural", "both"])
    parser.add_argument("--method", choices=["watershed", "connected"])
    parser.add_argument("--degradation", choices=["none", "blur", "noise"])
    parser.add_argument("--sigma", type=float, help="Degradation sigma: pixels for blur, normalized intensity for noise")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--min-area", type=int)
    parser.add_argument("--min-distance", type=int)
    parser.add_argument("--threshold-factor", type=float)
    args = parser.parse_args()
    settings = dict(input=None, output="outputs", model="classical", truth=None, degradation="none",
                    sigma=2., seed=42, repeats=5, timeout_seconds=600, neural_model="2D_versatile_fluo",
                    classical_method="watershed", segmentation=dict(sigma=1., min_area=40, min_distance=10, threshold_factor=.7))
    if args.config:
        try:
            config = yaml.safe_load(args.config.read_text()) or {}
            if not isinstance(config, dict) or set(config) - set(settings):
                raise ValueError("Config must be a mapping with supported keys only")
            settings.update({key: value for key, value in config.items() if key != "segmentation"})
            settings["segmentation"].update(config.get("segmentation", {}))
            for key in ("input", "output", "truth"):
                if settings[key]:
                    settings[key] = str((args.config.resolve().parent / settings[key]).resolve())
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            parser.error(str(exc))
    for key in ("output", "truth", "model", "degradation", "sigma", "repeats"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = str(value) if isinstance(value, Path) else value
    if args.image:
        settings["input"] = str(args.image.resolve())
    for key in ("min_area", "min_distance", "threshold_factor"):
        if getattr(args, key) is not None:
            settings["segmentation"][key] = getattr(args, key)
    if args.method:
        settings["classical_method"] = args.method
    try:
        Config(**settings["segmentation"])
        if not settings["input"] or not Path(settings["input"]).is_file():
            raise ValueError("Provide an existing image or --config config.yaml")
        if settings["model"] not in {"classical", "neural", "both"}:
            raise ValueError("Invalid model")
        if settings["degradation"] not in {"none", "blur", "noise"}:
            raise ValueError("Invalid degradation")
        if not isinstance(settings["repeats"], int) or settings["repeats"] < 1:
            raise ValueError("repeats must be a positive integer")
        if not math.isfinite(settings["sigma"]) or settings["sigma"] < 0:
            raise ValueError("sigma must be finite and nonnegative")
        if settings["truth"] and not Path(settings["truth"]).is_file():
            raise ValueError("Ground truth file does not exist")
        if settings["classical_method"] not in {"watershed", "connected"}:
            raise ValueError("Invalid classical method")
        if not math.isfinite(settings["timeout_seconds"]) or settings["timeout_seconds"] <= 0:
            raise ValueError("timeout_seconds must be positive")
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(output / "pipeline.log"), logging.StreamHandler()])
    logging.info("Input %s; degradation %s sigma=%s; CPU workers with one thread requested", settings["input"], settings["degradation"], settings["sigma"])
    if not settings["truth"]:
        logging.warning("No annotation supplied: accuracy metrics will not be computed")
    models = ["classical", "neural"] if settings["model"] == "both" else [settings["model"]]
    try:
        rows = [run_model(settings, model) for model in models]
        report(rows, output)
        (output / "effective_config.json").write_text(json.dumps(settings, indent=2))
        packages = {}
        for name in ("numpy", "scipy", "scikit-image", "stardist", "tensorflow-cpu", "psutil", "PyYAML"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        (output / "environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "packages": packages, "threads": 1, "device": "cpu"}, indent=2))
        logging.info("Reports written to %s", output)
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
