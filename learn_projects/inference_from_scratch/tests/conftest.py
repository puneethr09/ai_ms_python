from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b"
REFERENCE = ROOT / "fixtures" / "reference.npz"


@pytest.fixture(scope="session")
def model_dir() -> Path:
    if not (MODEL_DIR / "model.safetensors").exists():
        pytest.skip("model missing: run python scripts/download_model.py")
    return MODEL_DIR


@pytest.fixture(scope="session")
def ref() -> dict[str, np.ndarray]:
    if not REFERENCE.exists():
        pytest.skip("fixtures missing: run python scripts/make_reference.py")
    return dict(np.load(REFERENCE))


@pytest.fixture(scope="session")
def config(model_dir):
    from nanoinfer.model import Config
    return Config.from_json(model_dir / "config.json")


@pytest.fixture(scope="session")
def weights(model_dir):
    from nanoinfer import safetensors
    return safetensors.load(model_dir / "model.safetensors")


def indexed(ref: dict, prefix: str) -> list[int]:
    return sorted(int(k.rsplit("_", 1)[1]) for k in ref if k.startswith(prefix))
