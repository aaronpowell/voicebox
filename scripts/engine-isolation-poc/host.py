"""Opt-in experiment: Kokoro in-process and Chatterbox in a subprocess."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def run_worker(
    worker_python: str, runner: str | None, model_dir: Path, text: str, timeout: int
) -> tuple[dict, bytes]:
    worker = Path(__file__).with_name("worker.py")
    with tempfile.TemporaryDirectory(prefix="voicebox-worker-poc-") as directory:
        root = Path(directory)
        audio_path = root / "audio.wav"
        result_path = root / "result.json"
        request = {
            "text": text,
            "model_dir": str(model_dir.resolve()),
            "audio_path": str(audio_path),
            "result_path": str(result_path),
        }
        command = ([runner, "-cpu", "IvyBridge"] if runner else []) + [
            worker_python,
            str(worker),
        ]
        completed = subprocess.run(
            command, input=json.dumps(request), text=True, timeout=timeout, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Chatterbox worker exited with status {completed.returncode}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        audio = audio_path.read_bytes()
        if result.get("sample_rate", 0) <= 0 or result.get("samples", 0) <= 0 or not audio:
            raise ValueError("Chatterbox worker returned invalid audio metadata")
        return result, audio


def generate_kokoro(model_dir: Path, text: str) -> tuple[dict, bytes]:
    import io

    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KModel, KPipeline

    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(model_dir / "config.json"),
        model=str(model_dir / "kokoro-v1_0.pth"),
    ).to("cpu").eval()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
    with torch.inference_mode():
        chunks = [
            result.audio.detach().cpu().numpy().reshape(-1)
            for result in pipeline(text, voice=str(model_dir / "voices" / "af_heart.pt"))
            if result.audio is not None
        ]
    if not chunks:
        raise ValueError("Kokoro returned no audio")
    audio = np.concatenate(chunks)
    if not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError("Kokoro returned invalid or silent audio")
    output = io.BytesIO()
    sf.write(output, audio, 24000, format="WAV")
    return {"sample_rate": 24000, "samples": int(audio.size)}, output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=("kokoro", "chatterbox"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--worker-python")
    parser.add_argument("--runner")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    text = "Hello there."
    if args.engine == "kokoro":
        result, audio = generate_kokoro(args.model_dir, text)
    else:
        if not args.worker_python:
            parser.error("--worker-python is required for chatterbox")
        result, audio = run_worker(
            args.worker_python, args.runner, args.model_dir, text, args.timeout
        )
    import io

    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(io.BytesIO(audio), dtype="float32")
    if (
        samples.ndim != 1
        or samples.size != result["samples"]
        or sample_rate != result["sample_rate"]
        or not np.isfinite(samples).all()
        or not np.any(samples)
    ):
        raise ValueError(f"{args.engine} returned invalid or silent WAV data")
    print(json.dumps({"engine": args.engine, **result, "wav_bytes": len(audio)}), flush=True)


if __name__ == "__main__":
    main()
