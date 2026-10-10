"""Test integration contracts and observable failure behavior."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from cell_counter.metrics import evaluate
from cell_counter.pipeline import degrade
from cell_counter.segmenters import BaseSegmenter, WatershedSegmenter
from cell_counter.segmentation import Config

ROOT = Path(__file__).resolve().parents[1]


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseSegmenter()


def test_precision_recall_distinguish_misses_and_false_positives():
    truth = np.array([[1, 1, 0, 2, 2, 0]])
    missed = np.array([[1, 1, 0, 0, 0, 0]])
    score = evaluate(truth, missed)
    assert score['precision'] == 1 and score['recall'] == .5
    score = evaluate(missed, truth)
    assert score['precision'] == .5 and score['recall'] == 1


def test_noise_is_repeatable_and_changes_with_seed():
    image = np.arange(100).reshape(10, 10)
    assert np.array_equal(degrade(image, 'noise', .08, 42), degrade(image, 'noise', .08, 42))
    assert not np.array_equal(degrade(image, 'noise', .08, 42), degrade(image, 'noise', .08, 43))
    with pytest.raises(ValueError):
        degrade(image, 'blur', float('nan'), 42)


def test_cli_config_relative_paths_and_outputs(tmp_path):
    config = tmp_path / 'config.yaml'
    output = tmp_path / 'results'
    config.write_text(f'input: {ROOT / "examples/validation_01.tif"}\noutput: results\nrepeats: 1\n')
    run = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--config', str(config)],
                         cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert (output / 'classical/labels.tif').exists()
    assert (output / 'comparison.pdf').exists()
    assert (output / 'pipeline.log').exists()
    settings = json.loads((output / 'effective_config.json').read_text())
    assert settings['output'] == str(output)


def test_cli_rejects_unknown_config_without_writing_results(tmp_path):
    config = tmp_path / 'bad.yaml'
    config.write_text('typo_parameter: 2\n')
    run = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--config', str(config)],
                         cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 2
    assert 'supported keys' in run.stderr
    assert not (tmp_path / 'outputs').exists()


def test_nonfinite_config_rejected():
    with pytest.raises(ValueError):
        Config(sigma=float('nan'))
