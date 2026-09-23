import numpy as np
from safetensors.torch import load_file


def test_matches_reference_loader(model_dir, weights):
    expected = {k: v.float().numpy() for k, v in load_file(model_dir / "model.safetensors").items()}
    assert set(weights) == set(expected)
    for name, arr in expected.items():
        assert weights[name].dtype == np.float32, name
        assert weights[name].shape == arr.shape, name
        np.testing.assert_array_equal(weights[name], arr, err_msg=name)
