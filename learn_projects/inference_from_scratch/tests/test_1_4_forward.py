"""The full model against Hugging Face fp32. The layer test names the first layer that diverges."""
import numpy as np
import pytest

from conftest import indexed
from nanoinfer.model import Qwen2


@pytest.fixture(scope="module")
def outputs(config, weights, ref):
    model = Qwen2(config, weights)
    return {i: model.forward(ref[f"prompt_ids_{i}"].tolist()) for i in indexed(ref, "prompt_ids_")}


def test_hidden_states_layer_by_layer(outputs, ref):
    for i, (_, hidden) in outputs.items():
        expected = ref[f"hidden_{i}"]
        assert len(hidden) == len(expected), "expected [embeddings, layer 0 .. L-2 outputs, final_norm(layer L-1)]"
        for layer, (got, exp) in enumerate(zip(hidden, expected)):
            scale = np.abs(exp).max()
            err = np.abs(got - exp).max()
            assert err <= 1e-4 * max(scale, 1.0), f"prompt {i}: hidden[{layer}] max err {err:.3g} (scale {scale:.3g})"


def test_logits(outputs, ref):
    for i, (logits, _) in outputs.items():
        expected = ref[f"logits_{i}"]
        np.testing.assert_allclose(logits, expected, atol=1e-3, rtol=1e-3, err_msg=f"prompt {i}")
        assert (logits.argmax(-1) == expected.argmax(-1)).all()
