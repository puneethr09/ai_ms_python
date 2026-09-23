import pytest

from conftest import indexed
from nanoinfer.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tok(model_dir):
    return Tokenizer.from_file(model_dir / "tokenizer.json")


def cases(ref):
    return [(str(ref[f"{p}_text_{i}"]), ref[f"{p}_ids_{i}"].tolist())
            for p in ("tok", "prompt") for i in indexed(ref, f"{p}_ids_")]


def test_encode_matches_hf(tok, ref):
    for text, ids in cases(ref):
        assert tok.encode(text) == ids, repr(text)


def test_decode_roundtrip(tok, ref):
    for text, ids in cases(ref):
        assert tok.decode(ids) == text, repr(text)
