"""Each building block checked in isolation against PyTorch / Hugging Face."""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RotaryEmbedding, apply_rotary_pos_emb

from nanoinfer.model import apply_rope, attention, rms_norm, rope_tables, swiglu_mlp

rng = np.random.default_rng(0)
ATOL = 1e-5


def rand(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def test_rms_norm():
    x, w = rand(7, 896), rand(896)
    expected = F.rms_norm(torch.tensor(x), (896,), torch.tensor(w), eps=1e-6).numpy()
    np.testing.assert_allclose(rms_norm(x, w, 1e-6), expected, atol=ATOL, rtol=1e-5)


def test_rope(model_dir, config):
    hf_config = AutoConfig.from_pretrained(model_dir)
    seq, n_heads, n_kv, d = 9, 14, 2, 64
    q, k = rand(1, n_heads, seq, d), rand(1, n_kv, seq, d)
    positions = torch.arange(seq)[None]
    cos_t, sin_t = Qwen2RotaryEmbedding(hf_config)(torch.tensor(q), positions)
    q_exp, k_exp = apply_rotary_pos_emb(torch.tensor(q), torch.tensor(k), cos_t, sin_t)

    cos, sin = rope_tables(d, 64, config.rope_theta)
    np.testing.assert_allclose(cos[:seq], cos_t[0].numpy(), atol=ATOL)
    np.testing.assert_allclose(sin[:seq], sin_t[0].numpy(), atol=ATOL)
    np.testing.assert_allclose(apply_rope(q[0], cos[:seq], sin[:seq]), q_exp[0].numpy(), atol=ATOL)
    np.testing.assert_allclose(apply_rope(k[0], cos[:seq], sin[:seq]), k_exp[0].numpy(), atol=ATOL)


def test_causal_gqa_attention():
    q, k, v = rand(14, 11, 64), rand(2, 11, 64), rand(2, 11, 64)
    expected = F.scaled_dot_product_attention(
        torch.tensor(q), torch.tensor(k), torch.tensor(v), is_causal=True, enable_gqa=True).numpy()
    np.testing.assert_allclose(attention(q, k, v), expected, atol=ATOL, rtol=1e-4)


def test_swiglu_mlp(model_dir):
    hf_config = AutoConfig.from_pretrained(model_dir)
    mlp = Qwen2MLP(hf_config).eval()
    x = rand(5, hf_config.hidden_size)
    with torch.no_grad():
        expected = mlp(torch.tensor(x)).numpy()
    got = swiglu_mlp(x, mlp.gate_proj.weight.detach().numpy(), mlp.up_proj.weight.detach().numpy(),
                     mlp.down_proj.weight.detach().numpy())
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)
