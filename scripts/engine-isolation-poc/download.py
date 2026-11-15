"""Download only pinned files used by the experiment, with a disk-space guard."""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "kokoro": (
        "hexgrad/Kokoro-82M",
        "f3ff3571791e39611d31c381e3a41a3af07b4987",
        ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],
        328_000_000,
    ),
    "chatterbox": (
        "ResembleAI/chatterbox-turbo",
        "749d1c1a46eb10492095d68fbcf55691ccf137cd",
        [
            "conds.pt",
            "merges.txt",
            "s3gen_meanflow.safetensors",
            "t3_turbo_v1.safetensors",
            "tokenizer_config.json",
            "ve.safetensors",
            "vocab.json",
        ],
        2_988_000_000,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=MODELS)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    repo, revision, files, expected_bytes = MODELS[args.engine]
    args.destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.destination).free
    print(f"{args.engine}: expected={expected_bytes} bytes, free={free} bytes", flush=True)
    if free < expected_bytes + 2_000_000_000:
        raise RuntimeError("Insufficient disk space for model and 2 GB safety margin")
    for filename in files:
        hf_hub_download(
            repo_id=repo, revision=revision, filename=filename, local_dir=args.destination
        )
    actual = sum((args.destination / filename).stat().st_size for filename in files)
    if actual > expected_bytes:
        raise RuntimeError(f"Model download exceeds planned size: {actual} bytes")
    print(f"{args.engine}: downloaded model files={actual} bytes, revision={revision}", flush=True)


if __name__ == "__main__":
    main()
