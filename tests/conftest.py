"""Shared pytest fixtures."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg  # noqa: E402


@pytest.fixture(scope="session")
def splits():
    """Train / test frames, rebuilt from the raw CSV if needed."""
    from src.data import make_dataset
    if not cfg.TRAIN_FILE.exists():
        make_dataset.main()
    return {n: make_dataset.load_split(n) for n in ["train", "test"]}


@pytest.fixture(scope="session")
def cascades():
    if not all(cfg.cascade_file(m).exists() for m in cfg.MODEL_NAMES):
        pytest.skip("models not trained yet - run `make train`")
    from src.models.train_model import load_cascades
    return load_cascades()


@pytest.fixture(scope="session")
def breakpoints():
    from src.data.make_dataset import load_breakpoints
    return load_breakpoints()
