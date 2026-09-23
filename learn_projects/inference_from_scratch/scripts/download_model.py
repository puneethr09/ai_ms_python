from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-0.5b"


def main() -> None:
    path = snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
    )
    print(f"downloaded {MODEL_ID} -> {path}")


if __name__ == "__main__":
    main()
