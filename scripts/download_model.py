# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.25"]
# ///
"""Qwen2.5-0.5B-Instruct ağırlıklarını HF Hub'dan models/ altına indirir.

Kullanım (repo kökünden):
    uv run scripts/download_model.py
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=TARGET_DIR)
        size_mb = Path(path).stat().st_size / 1e6
        print(f"{name:<24} {size_mb:>10.1f} MB")
    print(f"\nModel hazır: {TARGET_DIR}")


if __name__ == "__main__":
    main()
