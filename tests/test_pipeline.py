import numpy as np
import pytest
from cell_counter import Config, segment, measure, normalize
from cell_counter.metrics import evaluate
from cell_counter.dataset import decode_mask
from PIL import Image


def disks():
    yy, xx = np.mgrid[:96, :96]
    image = np.zeros((96, 96), dtype=float)
    for y, x in [(25, 25), (65, 65)]:
        image[(yy-y)**2 + (xx-x)**2 < 10**2] = 1
    return image


@pytest.mark.parametrize("method", ["connected", "watershed"])
def test_count_isolated_nuclei(method):
    labels = segment(disks(), method, Config(min_area=30))
    assert labels.max() == 2
    assert len(measure(labels)) == 2


def test_touching_nuclei_are_separated():
    yy, xx = np.mgrid[:80, :80]
    image = (((yy-40)**2 + (xx-31)**2 < 12**2) | ((yy-40)**2 + (xx-49)**2 < 12**2)).astype(float)
    assert segment(image, "connected", Config(min_area=30)).max() == 1
    assert segment(image, "watershed", Config(min_area=30, min_distance=8)).max() == 2


def test_blank_and_nonfinite():
    assert segment(np.ones((40, 40))).max() == 0
    assert measure(segment(np.zeros((40, 40)))).empty
    with pytest.raises(ValueError): normalize(np.full((10, 10), np.nan))
    with pytest.raises(ValueError): normalize(np.zeros((10, 10, 10)))


def test_uint16_is_not_truncated():
    x = (disks() * 50000).astype(np.uint16)
    assert segment(x).max() == 2


def test_instance_metrics_catch_merged_prediction():
    truth = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    merged = np.ones_like(truth)
    score = evaluate(truth, merged)
    assert score["foreground_dice"] == 1
    assert score["fn"] == 1
    assert score["instance_f1"] == pytest.approx(2/3)


def test_label_ids_do_not_affect_matching():
    truth = np.array([[10, 10, 0], [0, 99, 99]])
    pred = np.array([[7, 7, 0], [0, 3, 3]])
    assert evaluate(truth, pred)["instance_f1"] == 1
    assert evaluate(np.zeros((3, 3), int), np.zeros((3, 3), int))["instance_f1"] == 1


def test_colored_touching_ground_truth(tmp_path):
    image = np.zeros((6, 6, 3), np.uint8)
    image[1:5, 1:3] = [50, 1, 1]
    image[1:5, 3:5] = [100, 1, 1]
    path = tmp_path / "mask.png"
    Image.fromarray(image).save(path)
    assert decode_mask(path).max() == 2


def test_brightness_normalization_control():
    assert np.allclose(normalize(disks()), normalize(disks() * .4))
