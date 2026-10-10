"""BBBC039: published image splits and separately decoded nucleus instances."""
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.measure import label


def read_image(path):
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are unsupported; export one 2D frame")
        return np.array(image)


def decode_mask(path):
    mask = read_image(path)
    # Broad's reference decoder uses the first channel and labels equal-valued
    # connected regions. Binary labeling would merge touching instances.
    if mask.ndim == 3:
        mask = mask[..., 0]
    return label(mask, background=0, connectivity=2).astype(np.int32)


def split_names(root, split):
    if split not in ("training", "validation", "test"):
        raise ValueError("Unknown split")
    return [line.strip() for line in (Path(root) / "metadata" / f"{split}.txt").read_text().splitlines() if line.strip()]


def load_sample(root, name):
    root = Path(root)
    return read_image(root / "images" / Path(name).with_suffix(".tif")), decode_mask(root / "masks" / name)
