import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset import read_image
from .segmentation import Config, segment, measure, overlay


def main():
    parser = argparse.ArgumentParser(description="Segment bright fluorescent nuclei in a single 2D image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--method", choices=["connected", "watershed"], default="watershed")
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--min-distance", type=int, default=8)
    parser.add_argument("--threshold-factor", type=float, default=0.8)
    args = parser.parse_args()
    x = read_image(args.image)
    labels = segment(x, args.method, Config(min_area=args.min_area, min_distance=args.min_distance, threshold_factor=args.threshold_factor))
    args.output.mkdir(parents=True, exist_ok=True)
    measure(labels).to_csv(args.output / "measurements.csv", index=False)
    Image.fromarray(labels.astype(np.uint16)).save(args.output / "labels.tif")
    Image.fromarray((overlay(x, labels) * 255).astype(np.uint8)).save(args.output / "overlay.png")
    print(f"Detected {int(labels.max())} nuclei. Results: {args.output}")


if __name__ == "__main__":
    main()
